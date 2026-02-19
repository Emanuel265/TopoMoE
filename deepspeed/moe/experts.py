# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import copy
from typing import List, Optional

import torch
from torch import nn
from ..utils import logger


class Experts(nn.Module):

    def __init__(self, expert: nn.Module, num_local_experts: int = 1, expert_group_name: Optional[str] = None) -> None:
        print("[CUSTOM DEBUG] test3")
        super(Experts, self).__init__()

        print(f"[CUSTOM DEBUG] init {num_local_experts} Experts {expert_group_name if expert_group_name is not None else 'None'}")

        self.deepspeed_experts = nn.ModuleList([copy.deepcopy(expert) for _ in range(num_local_experts)])
        self.num_local_experts = num_local_experts

        # TODO: revisit allreduce for moe.gate...
        for expert in self.deepspeed_experts:
            # TODO: Create param groups to handle expert + data case (e.g. param.group = moe_group)
            for param in expert.parameters():
                param.allreduce = False
                param.group_name = expert_group_name

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        chunks = inputs.chunk(self.num_local_experts, dim=1)
        expert_outputs: List[torch.Tensor] = []

        print("[CUSTOM DEBUG] test4")
        # print(f"[CUSTOM DEBUG] {inputs}")
        logger.info(f"[CUSTOM DEBUG] Experts forward, input shape: {inputs.shape}, num_local_experts: {self.num_local_experts}")

        for chunk, expert in zip(chunks, self.deepspeed_experts):
            out = expert(chunk)
            if isinstance(out, tuple):
                out = out[0]  # Ignore the bias term for now
            expert_outputs += [out]

        return torch.cat(expert_outputs, dim=1)

    # ------------------------------------------------------------
    # Read helpers
    # ------------------------------------------------------------

    def get_expert_state_dict(self, local_expert_idx: int) -> dict:
        """
        Return a detached, cloned state_dict of a local expert.

        Safe to call during training (read-only).
        """
        expert = self.get_expert(local_expert_idx)
        return {k: v.detach().clone() for k, v in expert.state_dict().items()}

    # ------------------------------------------------------------
    # Write helpers
    # ------------------------------------------------------------

    def load_expert_state_dict(self, local_expert_idx: int, state_dict: dict, strict: bool = True) -> None:
        """
        Overwrite the weights of a local expert.

        IMPORTANT:
        - Must be called under torch.no_grad()
        - Should be called only at synchronization-safe points
        """
        expert = self.get_expert(local_expert_idx)
        with torch.no_grad():
            expert.load_state_dict(state_dict, strict=strict)