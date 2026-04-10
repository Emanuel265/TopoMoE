import time

import torch
import torch.distributed as dist
from torch import nn
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import logging

from src.model.model_wrapper import ExpertPlacementConfig

class GreedyPlacementOptimizer:
    """
    Given an affinity matrix and the current expert→GPU mapping,
    find a better mapping via greedy pairwise swaps.

    Expert→GPU mapping is represented as a flat list:
        placement[global_expert_id] = gpu_rank
    """

    def __init__(self, num_experts: int, num_gpus: int, experts_per_gpu: int, config: ExpertPlacementConfig):
        self.num_experts    = num_experts
        self.num_gpus       = num_gpus
        self.epg            = experts_per_gpu
        self.config         = config

    # ── cost helpers ──────────────────────────────────────────
    def _cross_gpu_cost(self, aff: torch.Tensor, placement: List[int]) -> float:
        cost = 0.0
        A = aff.numpy()
        for i in range(self.num_experts):
            for j in range(i + 1, self.num_experts):
                if placement[i] != placement[j]:
                    cost += A[i, j]
        return cost

    def _swap_delta(self, A, placement: List[int], ea: int, eb: int) -> float:
        """O(E) delta computation – no need to recompute full cost."""
        ga, gb = placement[ea], placement[eb]
        if ga == gb:
            return 0.0
        delta = 0.0
        for k in range(self.num_experts):
            if k == ea or k == eb:
                continue
            gk = placement[k]
            old = (A[ea, k] if ga != gk else 0.0) + (A[eb, k] if gb != gk else 0.0)
            new = (A[ea, k] if gb != gk else 0.0) + (A[eb, k] if ga != gk else 0.0)
            delta += new - old
        return delta

    # ── main ──────────────────────────────────────────────────

    def optimize(self, affinity: torch.Tensor, placement: List[int]) -> Tuple[List[int], Dict]:
            A = affinity.numpy()
            p = placement.copy()

            init_cost = self._cross_gpu_cost(affinity, p)
            cur_cost  = init_cost

            if init_cost == 0.0:
                print("[GreedyPlacementOptimizer] init_cost is 0, skipping")
                return p, dict(init_cost=0.0, final_cost=0.0,
                            improve_pct=0.0, swaps=0, iters=0)

            swaps = 0

            experts_on: Dict[int, List[int]] = {g: [] for g in range(self.num_gpus)}
            for e, g in enumerate(p):
                experts_on[g].append(e)

            for iteration in range(self.config.max_swap_iterations):
                best_delta, best_pair = 0.0, None

                for gpu_a in range(self.num_gpus):
                    for gpu_b in range(gpu_a + 1, self.num_gpus):
                        for ea in experts_on[gpu_a]:
                            for eb in experts_on[gpu_b]:
                                d = self._swap_delta(A, p, ea, eb)
                                if d < best_delta:
                                    best_delta, best_pair = d, (ea, eb)

                if best_pair is None:
                    print(f"[GreedyPlacementOptimizer] Converged after {iteration} iterations "
                        f"with {swaps} swaps")
                    break

                ea, eb   = best_pair
                ga, gb   = p[ea], p[eb]

                p[ea], p[eb] = gb, ga
                experts_on[ga].remove(ea); experts_on[ga].append(eb)
                experts_on[gb].remove(eb); experts_on[gb].append(ea)

                cur_cost += best_delta
                swaps    += 1

                # print(f"[GreedyPlacementOptimizer] iter {iteration:3d} | "
                #     f"swap e{ea}↔e{eb} (gpu{ga}↔gpu{gb}) | "
                #     f"delta {best_delta:+.6f} | cost {cur_cost:.6f}")

                if abs(best_delta) / init_cost < self.config.min_improvement_frac:
                    print(f"[GreedyPlacementOptimizer] Early stop: marginal gain "
                        f"{best_delta/init_cost:.4%} < threshold "
                        f"{self.config.min_improvement_frac:.4%}")
                    break

            improve_pct = (init_cost - cur_cost) / init_cost * 100
            stats = dict(init_cost=init_cost, final_cost=cur_cost,
                        improve_pct=improve_pct, swaps=swaps, iters=iteration + 1)
            print(f"[GreedyPlacementOptimizer] {improve_pct:.1f}% cross-GPU traffic reduction "
                f"({swaps} swaps, {iteration + 1} iters)")
            return p, stats

