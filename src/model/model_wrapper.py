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

from src.model.expert_placement import get_expert_placement_single_layer
from src.model.topology import get_topology
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
    def __init__(
        self,
        use_topomoe: bool,
        placement_manager_config: ExpertPlacementConfig,
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

        if use_topomoe:
            self.moe = MoE(layer_idx, placement_manager_config, **moe_kwargs)
        else:
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
    def __init__(
        self,
        use_topomoe: bool,
        placement_manager_config: ExpertPlacementConfig,
        vocab_size:  int,
        seq_len:     int,
        hidden:      int,
        num_heads:   int,
        num_layers:  int,
        num_experts: int,
        ep_size:     int,
        k:           int,
        aux_weight:  float = 0.01,
    ):
        super().__init__()
        self.seq_len    = seq_len
        self.aux_weight = aux_weight
        self.hidden = hidden

        # Embeddings
        self.token_emb = nn.Embedding(vocab_size, hidden)
        self.pos_emb   = nn.Embedding(seq_len, hidden)

        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(
                use_topomoe=use_topomoe,
                placement_manager_config=placement_manager_config,
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
            total_aux = total_aux + l_aux

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

logger_ep = logging.getLogger("expert_placement")

class AffinityAccumulator:
    """
    Builds expert-to-expert affinity from the comm_matrix that
    MOELayer already collects every forward pass.

    comm_matrix[r, e] = number of tokens rank r sent to expert e.

    Two experts are "co-active" on a rank if the same rank sends
    many tokens to both.  We use the outer product of each row as
    the per-rank affinity contribution.

    Result: affinity[i][j] = how tightly coupled expert i and j are
            (higher → prefer same GPU)
    """

    def __init__(self, num_global_experts: int, num_layers: int, decay: float = 0.9):
        self.num_experts = num_global_experts
        self.decay       = decay
        self.affinity: torch.Tensor = torch.zeros(
            num_layers - 1, num_global_experts, num_global_experts, dtype=torch.float32
        )

    @torch.no_grad()
    def update(self, comm_matrix: torch.Tensor):
        """
        comm_matrix: [num_layers, num_global_experts, num_global_experts]  (already gathered, on any device)
        """
        new_aff = comm_matrix.float().cpu()  

        # total = new_aff.sum()
        # if total > 0:
        #     new_aff = new_aff / total

        # EMA update
        self.affinity += new_aff
        # if self.affinity is None:
        # else:
        #     self.affinity = self.decay * self.affinity + (1.0 - self.decay) * new_aff

    def get(self) -> torch.Tensor:
        """Returns [E, E] affinity matrix (CPU, float32)."""
        return self.affinity.clone()

    def reset(self):
        self.affinity.zero_()

# ──────────────────────────────────────────────────────────────
# Expert migrator
# ──────────────────────────────────────────────────────────────

class ExpertMigrator:
    """
    Physically moves expert weights (and optimizer state) between GPUs
    using P2P communication.

    Relies on the read/write helpers already present in Experts:
        get_expert_state_dict(local_idx)  → dict of CPU tensors
        load_expert_state_dict(local_idx, sd)
    """

    def __init__(
        self,
        experts_module,
        ep_group,
        num_local_experts: int,
        ep_size:           int,
        device:            torch.device = None,
        optimizer          = None,
        transfer_optim_after_step = 100,
    ):
        self.module           = experts_module
        self.ep_group          = ep_group
        self.num_local_experts = num_local_experts
        self.ep_size           = ep_size
        self.my_rank           = dist.get_rank(ep_group)
        global_rank            = dist.get_rank()
        self.device            = device or torch.device(f"cuda:{global_rank % torch.cuda.device_count()}")
        self.optimizer         = optimizer
        self.transfer_optim_after_step = transfer_optim_after_step

    # ── helpers ───────────────────────────────────────────────

    def _global_to_local(self, global_expert_id: int) -> Tuple[int, int]:
        """Returns (rank, local_idx) for a global expert id."""
        rank      = global_expert_id // self.num_local_experts
        local_idx = global_expert_id %  self.num_local_experts
        return rank, local_idx

    def _state_dict_to_tensor(self, sd: dict) -> torch.Tensor:
        """Flatten a state dict into a single 1-D float32 tensor for transmission."""
        parts = [v.detach().float().cpu().reshape(-1) for v in sd.values()]
        return torch.cat(parts)

    def _tensor_to_state_dict(self, flat: torch.Tensor, sd_template: dict) -> dict:
        """Reconstruct a state dict from a flat tensor using a template for shapes."""
        new_sd, offset = {}, 0
        for k, v in sd_template.items():
            numel    = v.numel()
            new_sd[k] = flat[offset:offset + numel].reshape(v.shape).to(v.dtype)
            offset  += numel
        return new_sd

    def _get_expert_params(self, local_idx: int, layer_i: int) -> List[torch.nn.Parameter]:
        """Return the parameter list for one local expert."""
        return list(self.module.layers[layer_i].moe.deepspeed_moe.experts.get_expert(local_idx).parameters())

    def _migrate_optimizer_state(
        self,
        step:        int,
        layer_i:     int,
        local_idx:   int,
        from_rank:   int,
        to_rank:     int,
        from_global: int,
        to_global:   int,
    ):
        """
        Transfer AdamW state (step, exp_avg, exp_avg_sq) for every parameter
        of the expert at local_idx.  Must be called BEFORE load_expert_state_dict
        so the parameter objects are still the ones the optimizer knows about.

        Sender serialises all state tensors into one flat buffer and sends it.
        Receiver reconstructs and updates optimizer.state in-place.
        """
        if self.optimizer is None:
            return

        params = self._get_expert_params(local_idx, layer_i)

        if step < self.transfer_optim_after_step:
            if self.my_rank == to_rank:
                for p in params:
                    if p in self.optimizer.state:
                        self.optimizer.state[p]['exp_avg'].zero_()
                        self.optimizer.state[p]['exp_avg_sq'].zero_()
            return

        # ── Build flat buffer of optimizer state on sender ────
        size_tensor = torch.zeros(1, dtype=torch.long, device=self.device)

        if self.my_rank == from_rank:
            state_parts = []
            for p in params:
                s = self.optimizer.state.get(p, {})
                for key in ("exp_avg", "exp_avg_sq"):
                    if key in s:
                        state_parts.append(s[key].detach().float().cpu().reshape(-1))
                    else:
                        state_parts.append(torch.zeros(p.numel()))
            flat_optim = torch.cat(state_parts).to(self.device)
            size_tensor[0] = flat_optim.numel()

        dist.broadcast(size_tensor, src=from_global, group=self.ep_group)
        numel = int(size_tensor.item())

        if numel == 0:
            return

        # ── P2P transfer ──────────────────────────────────────
        if self.my_rank == from_rank:
            dist.send(flat_optim, dst=to_global)

        elif self.my_rank == to_rank:
            flat_optim = torch.zeros(numel, dtype=torch.float32, device=self.device)
            dist.recv(flat_optim, src=from_global)

            offset = 0
            for p in params:
                if p not in self.optimizer.state:
                    self.optimizer.state[p] = {}
                for key in ("exp_avg", "exp_avg_sq"):
                    chunk = flat_optim[offset:offset + p.numel()]
                    self.optimizer.state[p][key] = (
                        chunk.reshape(p.shape).to(p.dtype).to(p.device)
                    )
                    offset += p.numel()

    # ── main ──────────────────────────────────────────────────

    @torch.no_grad()
    def execute_migrations(self, migrations: List[Tuple[int, int, int]], layer_i, layer_placement, step):
        """
        migrations: list of (global_expert_id, from_rank, to_rank)

        For each migration:
          1. Migrate optimizer state (before weights, while param objects are stable)
          2. Migrate expert weights
          3. Barrier

        from_rank / to_rank are group-local ranks (0..ep_size-1).
        dist.send / dist.recv use global ranks — no group= argument.
        """
        if not migrations:
            return

        print(f"Rank {self.my_rank}: executing {len(migrations)} expert migrations for layer {layer_i}")

        for global_eid, from_rank, to_rank in migrations:
            local_idx = global_eid % self.num_local_experts # TODO: check if this is fine

            from_global = dist.get_global_rank(self.ep_group, from_rank)
            to_global   = dist.get_global_rank(self.ep_group, to_rank)

            # ── 1. Optimizer state — must happen before weight transfer ──
            self._migrate_optimizer_state(
                step, layer_i, local_idx, from_rank, to_rank, from_global, to_global
            )

            # ── 2. Broadcast flat weight tensor size from sender ──────────
            size_tensor = torch.zeros(1, dtype=torch.long, device=self.device)
            if self.my_rank == from_rank:
                sd   = self.module.layers[layer_i].moe.deepspeed_moe.experts.get_expert_state_dict(local_idx)
                flat = self._state_dict_to_tensor(sd).to(self.device)
                size_tensor[0] = flat.numel()

            dist.broadcast(size_tensor, src=from_global, group=self.ep_group)
            numel = int(size_tensor.item())

            # ── 3. Transfer weights ───────────────────────────────────────
            if self.my_rank == from_rank:
                dist.send(flat, dst=to_global)
                logger_ep.debug(
                    f"  Sent   expert g{global_eid} → rank {to_rank}  ({numel} params)"
                )

            elif self.my_rank == to_rank:
                recv_flat   = torch.zeros(numel, dtype=torch.float32, device=self.device)
                dist.recv(recv_flat, src=from_global)
                sd_template = self.module.layers[layer_i].moe.deepspeed_moe.experts.get_expert_state_dict(local_idx)
                new_sd      = self._tensor_to_state_dict(recv_flat.cpu(), sd_template)
                self.module.layers[layer_i].moe.deepspeed_moe.experts.load_expert_state_dict(local_idx, new_sd)
                logger_ep.debug(
                    f"  Recv'd expert g{global_eid} from rank {from_rank}  ({numel} params)"
                )

        dist.barrier(group=self.ep_group)
        logger_ep.info(f"Rank {self.my_rank}: migrations complete")

class ExpertPlacementManager:
    """
    Coordinates affinity tracking, placement optimization, and migration.

    The callback is installed on every MOELayer and receives exp_counts
    directly — the simplest possible accumulation, no matrix math.

    All heavy work (comm matrix, optimization, migration) runs in step()
    which is called from the training loop after engine.step() where all
    ranks are synchronised.

    MOELayer wiring (add after self.gate() in both Tutel and non-Tutel path):

        if self._routing_callback is not None:
            self._routing_callback(self.exp_counts.detach())
    """

    def __init__(
        self,
        model,
        ep_group,
        num_local_experts: int,
        ep_size:           int,
        config:            ExpertPlacementConfig = None,
        device:            torch.device = None,
        optimizer=None,
    ):
        self.model             = model
        self.ep_group          = ep_group
        self.num_local_experts = num_local_experts
        self.ep_size           = ep_size
        self.num_experts       = ep_size * num_local_experts
        self.config            = config or ExpertPlacementConfig()
        self.device            = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_layers        = len(model.layers)
        self.optimizer         = optimizer
        self._rebalance_pending = False
        self.topology = get_topology()

        self.placements: List[List[int]] = [[
            e // num_local_experts for e in range(self.num_experts)
        ] for _ in range(self.num_layers)]

        self.accumulator = AffinityAccumulator(
            self.num_experts, self.num_layers, decay=self.config.affinity_decay
        )
        self.migrator: Optional[ExpertMigrator] = None

        self.rebalance_steps   = config.rebalance_steps
        self.global_step       = 0
        self.rebalance_history = []

        self._indices_buffer: Dict[int, List[torch.Tensor]] = {
            i: [] for i in range(self.num_layers)
        }

        self._install_callbacks(model)

    # ── Callback wiring ───────────────────────────────────────
    def _install_callbacks(self, model) -> None:
        for i, block in enumerate(model.layers):
            moe_layer = block.moe.deepspeed_moe

            assert hasattr(moe_layer, "_routing_callback"), (
                f"MOELayer at block {i} has no _routing_callback attribute. "
                "Add `self._routing_callback = None` to MOELayer.__init__ and "
                "call it after self.gate() in the Tutel forward path."
            )

            def callback(indices: torch.Tensor, layer_i: int = i) -> None:
                ind = indices.detach().cpu()
                valid = ind.flatten()
                valid = valid[valid >= 0]
                # print(f"Layer: {layer_i}, indices: {valid}, len: {len(valid)}, per Layer exp-count: {np.bincount(valid)}")
                self._indices_buffer[layer_i].append(valid)

            moe_layer._routing_callback = callback
    
    def _uninstall_callbacks(self, model) -> None:
        for i, block in enumerate(model.layers):
            moe_layer = block.moe.deepspeed_moe

            assert hasattr(moe_layer, "_routing_callback"), (
                f"MOELayer at block {i} has no _routing_callback attribute. "
                "Add `self._routing_callback = None` to MOELayer.__init__ and "
                "call it after self.gate() in the Tutel forward path."
            )

            self._indices_buffer[i] = []
            moe_layer._routing_callback = None

    # ── called from _set_ep_group ─────────────────────────────
    def set_ep_group(self, ep_group):
        self.ep_group = ep_group
        global_rank  = dist.get_rank()
        self.device  = torch.device(f"cuda:{global_rank % torch.cuda.device_count()}")
        self.migrator = ExpertMigrator(
            experts_module   = self.model,
            ep_group         = ep_group,
            num_local_experts= self.num_local_experts,
            ep_size          = self.ep_size,
            device           = self.device,
            optimizer        = self.optimizer,
        )

    # ── called from training loop after engine.step() ─────────
    def hash_expert(self, module: torch.nn.Module) -> str:
        hasher = hashlib.sha256()

        for p in module.parameters():
            tensor = p.detach().cpu().contiguous()
            hasher.update(tensor.numpy().tobytes())

        return hasher.hexdigest()

    def step(self, step):
        """
        Call once per training step after engine.step().
        All ranks are synchronised here.

        1. Build comm matrix from buffered exp_counts
        2. Feed accumulator
        3. Clear buffer
        4. If rebalance is due: optimise → broadcast → migrate
        """
        self.global_step = step
        if len(self.rebalance_steps) == 0:
            return

        # ── 1. Build comm matrix from buffered indices ────────
        comm_matrix = self._build_comm_matrix_from_buffer()
        if comm_matrix is not None:
            self.accumulator.update(comm_matrix)

        # ── 2. Clear buffer for next step ─────────────────────
        self._clear_buffer()

        # ── 3. Check if rebalance is due ──────────────────────
        if (self.rebalance_steps
                and self.global_step == self.rebalance_steps[0]
                and self.ep_group is not None
                and self.migrator is not None):
            self.rebalance_steps = self.rebalance_steps[1:]
            print(f"[Expert Placement Manager] Step {self.global_step} — rebalancing. Next Rebalance(s) at: {self.rebalance_steps}")
            def expert_hash(expert):
                import hashlib
                h = hashlib.sha256()
                for p in expert.parameters():
                    h.update(p.detach().cpu().numpy().tobytes())
                return h.hexdigest()[:8]
            if dist.get_rank() == 0:
                for i, layer_i in enumerate(self.model.layers): 
                    for j, expert_i in enumerate(layer_i.moe.deepspeed_moe.experts.deepspeed_experts):
                        print(f"Rank {dist.get_rank()}: Before rebalance - Layer {i} Expert {j} hash: {expert_hash(expert_i)}")
            before = time.time()
            self._rebalance()
            if dist.get_rank() == 0:
                for i, layer_i in enumerate(self.model.layers): 
                    for j, expert_i in enumerate(layer_i.moe.deepspeed_moe.experts.deepspeed_experts):
                        print(f"Rank {dist.get_rank()}: After rebalance - Layer {i} Expert {j} hash: {expert_hash(expert_i)}")
            print(f"[Expert Placement Manager] Rebalance took: {time.time() - before:.3f}s")
        elif len(self.rebalance_steps) == 0:
            self._uninstall_callbacks(self.model)

    # ── comm matrix construction ──────────────────────────────
    def _build_comm_matrix_from_buffer(self) -> Optional[torch.Tensor]:
        """
        Build exact [num_experts, num_experts] expert-to-expert comm matrix
        from Tutel's per-token gate indices accumulated in _indices_buffer.

        For each consecutive layer pair (i, i+1):
          - Concatenate all buffered index tensors for both layers
          - Compute all k1*k2 expert-pair combinations per token via broadcasting
          - Scatter-add into [E, E] matrix (single operation for all steps)

        All-reduce across ranks so every rank holds global counts.
        Each rank only sees tokens routed through its local expert shard.
        """
        if not all(len(self._indices_buffer[i]) > 0 for i in range(self.num_layers)):
            return None

        num_pairs = self.num_layers - 1
        matrix = torch.zeros(num_pairs, self.num_experts, self.num_experts, dtype=torch.float32)

        for i in range(self.num_layers - 1):
            # Concatenate all steps: list of [S] or [S*k] → [total_S]
            idx_i  = torch.cat(self._indices_buffer[i],     dim=0)
            idx_i1 = torch.cat(self._indices_buffer[i + 1], dim=0)

            # Normalise to 2-D [total_S, k]
            if idx_i.dim()  == 1: idx_i  = idx_i.unsqueeze(1)
            if idx_i1.dim() == 1: idx_i1 = idx_i1.unsqueeze(1)

            T, k1 = idx_i.shape
            k2     = idx_i1.shape[1]

            # All k1*k2 expert-pair combinations per token via broadcasting
            a = idx_i.unsqueeze(2).expand(T, k1, k2).reshape(-1).long()
            b = idx_i1.unsqueeze(1).expand(T, k1, k2).reshape(-1).long()

            matrix[i].index_put_((a, b), torch.ones(len(a)), accumulate=True)

        # All-reduce: each rank only saw its local shard of tokens
        if dist.is_available() and dist.is_initialized():
            matrix_cuda = matrix.to(self.device)
            dist.all_reduce(matrix_cuda, op=dist.ReduceOp.SUM)
            matrix = matrix_cuda.cpu()

        return matrix   # [E, E]

    def _clear_buffer(self) -> None:
        for lst in self._indices_buffer.values():
            lst.clear()

    # ── rebalance pipeline ────────────────────────────────────
    def _rebalance(self):
        my_rank     = dist.get_rank(self.ep_group)
        global_rank = dist.get_rank()
        device      = torch.device(f"cuda:{global_rank % torch.cuda.device_count()}")

        affinity = self.accumulator.get()

        for layer_i, layer_placement in enumerate(self.placements):
            if layer_i == 0:
                continue

            if my_rank == 0:
                new_placement, new_replica = get_expert_placement_single_layer(
                    topology        = self.topology,
                    num_experts     = self.num_experts,
                    prev_placement  = self.placements[layer_i - 1],
                    token_d         = self.model.hidden,
                    alpha           = self.config.alpha,
                    beta            = self.config.beta,
                    gamma           = self.config.gamma,
                    tau_balance     = self.config.tau_balance,
                    replica_placement = None,
                    coaccess_matrix = affinity[layer_i - 1]
                )
                self.rebalance_history.append({"step": self.global_step})
                print(f"[Expert Placement Manager] affinity:\n{affinity[layer_i-1]}")
            else:
                new_placement = [0] * self.num_experts
                new_replica   = [0] * self.num_experts

            # Broadcast new placement from rank 0 to all
            pt = torch.tensor(new_placement, dtype=torch.int32, device=device)
            group_rank0_global = dist.get_global_rank(self.ep_group, 0)
            dist.broadcast(pt, src=group_rank0_global, group=self.ep_group)
            new_placement = pt.tolist()

            # Only migrate experts that actually changed device
            migrations = [
                (eid, layer_placement[eid], new_placement[eid])
                for eid in range(self.num_experts)
                if layer_placement[eid] != new_placement[eid]
            ]

            if migrations:
                print(f"migrations: {migrations}, new placement: {new_placement}")
                dist.barrier(group=self.ep_group)
                self.migrator.execute_migrations(migrations, layer_i, layer_placement, self.global_step)

            self.placements[layer_i] = new_placement

    # ── introspection ─────────────────────────────────────────
    def get_placement(self) -> Dict[int, int]:
        return {e: r for e, r in enumerate(self.placement)}

    def pending_migrations(self, new_placement: List[int]) -> List[Tuple[int, int, int]]:
        return [(e, self.placement[e], new_placement[e])
                for e in range(self.num_experts) if self.placement[e] != new_placement[e]]
