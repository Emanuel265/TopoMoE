"""
model_wrapper.py

Drop-in MoEGPT compatible with the training script:

    from model_wrapper import MoEGPT

    model = MoEGPT(
        use_topomoe=True,
        vocab_size=50257,
        seq_len=128,
        hidden=256,
        num_heads=4,
        num_layers=6,
        num_experts=8,
        ep_size=1,
        k=2,
        aux_weight=0.01,
    )

    outputs = model(input_ids=..., attention_mask=..., labels=...)
    loss    = outputs.loss     # scalar
    logits  = outputs.logits   # (B, T, vocab_size)

NOTE on label shifting
──────────────────────
The training script pre-shifts labels before calling forward():
    labels[:, :-1] = input_ids[:, 1:]
    labels[:, -1]  = pad_token_id
This model therefore computes CE loss directly on the supplied labels
with NO internal shift. Shifting twice would train on t+2 → t (silent bug).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
import time

import logging

from deepspeed.moe.layer import MoE

from model.topology import Topology
from src.model.expert_placement import build_migration_plan, get_expert_permutation_single_layer
# from src.model.greedy_placement import GreedyPlacementOptimizer

import hashlib

@dataclass
class MoEModelOutput:
    loss:     Optional[torch.Tensor]
    logits:   torch.Tensor
    aux_loss: torch.Tensor

@dataclass
class ExpertPlacementConfig:
    rebalance_steps: List[int]   = field(default_factory=lambda: [])
    max_swap_iterations:     int   = 60
    min_improvement_frac:    float = 0.005
    affinity_decay:          float = 0.9
    migrate_timeout_sec:     float = 60.0

class TransformerBlock(nn.Module):

    moe: MoE

    def __init__(
        self,
        use_topomoe: bool,
        layer_idx:   int,
        hidden:      int,
        num_heads:   int,
        num_experts: int,
        ep_size:     int,
        k:           int,
    ):
        super().__init__()

        # ── Self-attention ──────────────────────────────
        self.ln_attn = nn.LayerNorm(hidden)
        self.attn    = nn.MultiheadAttention(
            embed_dim=hidden,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
        )

        # ── MoE FFN ─────────────────────────────────────
        self.ln_ffn = nn.LayerNorm(hidden)

        expert = nn.Sequential(
            nn.Linear(hidden, hidden * 4),
            nn.GELU(),
            nn.Linear(hidden * 4, hidden),
        )

        moe_kwargs = dict(
            hidden_size=hidden,
            expert=expert,
            num_experts=num_experts,
            ep_size=ep_size,
            k=k,
            capacity_factor=1.25,
            drop_tokens=False,
            use_residual=False,
        )

        self.moe = MoE(**moe_kwargs)

    def forward(
        self,
        x:         torch.Tensor,
        attn_mask: Optional[torch.Tensor],
    ):
        # Pre-norm attention
        h           = self.ln_attn(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x           = x + attn_out

        # Pre-norm MoE FFN
        h                 = self.ln_ffn(x)
        moe_out, l_aux, _ = self.moe(h)
        x                 = x + moe_out

        return x, l_aux

class MoEGPT(nn.Module):

    layers: nn.ModuleList[TransformerBlock]

    def __init__(
        self,
        use_topomoe: bool,
        topology:    Topology,
        token_d:     float,
        alpha:       float,
        beta:        float,
        gamma:       float,
        vocab_size:  int,
        seq_len:     int,
        hidden:      int,
        num_heads:   int,
        num_layers:  int,
        num_experts: int,
        ep_size:     int,
        k:           int,
        rank:        int,
        aux_weight:  float = 0.01,
    ):
        super().__init__()
        self.seq_len    = seq_len
        self.aux_weight = aux_weight
        self.hidden = hidden

        self.rank = rank
        self.num_layers = num_layers
        self.num_experts = num_experts

        # Embeddings
        self.token_emb = nn.Embedding(vocab_size, hidden)
        self.pos_emb   = nn.Embedding(seq_len, hidden)

        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(
                use_topomoe=use_topomoe,
                layer_idx=i,
                hidden=hidden,
                num_heads=num_heads,
                num_experts=num_experts,
                ep_size=ep_size,
                k=k,
            )
            for i in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, vocab_size, bias=False)
        self.head.weight = self.token_emb.weight

        self.register_buffer(
            "_causal_mask",
            torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool(),
            persistent=False,
        )

        self._init_weights()

        self.topology = topology
        self.token_d = token_d
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.register_buffer(
            "_causal_mask",
            torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool(),
            persistent=False,
        )

        self.coaccess = None  # Will initialize in forward pass


    # ── Weight initialisation ────────────────────────────
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ── Forward ──────────────────────────────────────────

    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels:         Optional[torch.Tensor] = None,
    ) -> MoEModelOutput:

        B, T = input_ids.shape
        device = input_ids.device

        # Initialize coaccess matrices on first forward pass
        if self.coaccess is None:
            self.coaccess = [
                torch.zeros((self.num_experts, self.num_experts), device=device)
                for _ in range(self.num_layers - 1)
            ]
            # Track number of batches for proper averaging
            self.coaccess_batch_count = 0

        # ── Embeddings ──────────────────────────────────
        positions = torch.arange(T, device=device).unsqueeze(0)
        x = self.token_emb(input_ids) + self.pos_emb(positions)

        # ── Causal attention mask (additive, -inf for future positions) ──
        causal    = self._causal_mask[:T, :T]
        attn_mask = torch.zeros(T, T, device=device, dtype=x.dtype)
        attn_mask.masked_fill_(causal, float("-inf"))

        # ── Transformer layers ──────────────────────────
        total_aux = torch.tensor(0.0, device=device)

        for layer in self.layers:
            x, l_aux  = layer(x, attn_mask)
            total_aux += l_aux
        
        # Compute co-access matrices (separate loop)
        for l in range(len(self.layers) - 1):
            mask_l  = self.layers[l].moe.deepspeed_moe.last_dispatch_mask
            mask_l1 = self.layers[l+1].moe.deepspeed_moe.last_dispatch_mask

            # mask_l shape: [S, E, C] - S tokens, E experts, C capacity
            # Sum over capacity dimension to get [S, E] - which expert each token uses
            expert_assignments_l = mask_l.sum(dim=-1).float()   # [S, E]
            expert_assignments_l1 = mask_l1.sum(dim=-1).float() # [S, E]
            
            # Compute co-access matrix: [E, E] 
            # Entry (i,j) = number of tokens that used expert i in layer l AND expert j in layer l+1
            coaccess_update = expert_assignments_l.T @ expert_assignments_l1
            
            dist.all_reduce(coaccess_update, group=self.layers[l].moe.deepspeed_moe.ep_group)
            
            self.coaccess[l] = 0.5 * self.coaccess[l] + coaccess_update
        
        # Increment batch counter
        self.coaccess_batch_count += 1

        # ── Head ────────────────────────────────────────
        x      = self.ln_f(x)
        logits = self.head(x)

        # ── Loss ────────────────────────────────────────
        loss = None
        if labels is not None:
            ce_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
            loss = ce_loss + self.aux_weight * total_aux

        return MoEModelOutput(
            loss=loss,
            logits=logits,
            aux_loss=total_aux.detach(),
        )
    
    def execute_migration_p2p(self, layer_id, expert_a, expert_b):
        self.layers[layer_id].moe.deepspeed_moe.execute_migrations_p2p(expert_a, expert_b)
    
    def execute_migration_alltoallv(self, layer_id, perm_old, perm_new, num_local_experts):
        self.layers[layer_id].moe.deepspeed_moe.migrate_experts_alltoallv(perm_old, perm_new, num_local_experts)

    def rebalance_experts(self):
        """
        Rebalance experts across all MoE layers using coaccess matrices.
        """

        for l in range(self.num_layers - 1):
            perm_old = self.layers[l + 1].moe.deepspeed_moe.expert_permutation

            perm_new = get_expert_permutation_single_layer(
                topology=self.topology,
                perm_init=perm_old,
                coaccess_matrix=self.coaccess[l],
                token_d=self.token_d,
                alpha=self.alpha,
                beta=self.beta,
                gamma=self.gamma,
                replicas=None,
                max_iters=100,
            )

            if self.rank == 0:
                print("Layer l: ", l, ", new perm: ", perm_new, "with coaccess: ", self.coaccess[l])

            if (False):
                migration_plan = build_migration_plan(perm_old, perm_new, self.num_experts // self.num_layers)
                self.execute_migration_p2p(l + 1, migration_plan)
            else:
                self.execute_migration_alltoallv(l + 1, perm_old, perm_new, self.num_experts // self.num_layers)

            self.layers[l + 1].moe.deepspeed_moe.expert_permutation = perm_new