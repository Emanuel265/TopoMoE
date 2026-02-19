"""
Expert Placement Optimizer – DeepSpeed Integration
===================================================
Drop-in integration for the existing MOELayer / Experts / MoE classes.

HOW TO USE
----------
1.  In sharded_moe.py  → import and instantiate ExpertPlacementManager in MOELayer.__init__
2.  In sharded_moe.py  → call manager.on_forward() at the end of MOELayer.forward()
3.  In experts.py      → already has get/load_expert_state_dict ✓

Nothing else needs to change.

ALGORITHM OVERVIEW
------------------
The comm_matrix you already collect has shape [ep_world_size, num_global_experts].
  row i  = tokens sent FROM rank i
  col j  = tokens destined TO expert j

From this we build an expert-to-expert affinity:
  affinity[i][j]  ∝  how often tokens that visited expert i also visited expert j
  (approximated via co-assignment in the same batch using exp_counts per rank)

We then run a greedy swap: find the pair of experts on different GPUs whose
swap reduces cross-GPU token traffic the most, repeat until convergence.

After finding the optimal assignment we physically migrate expert weights
between GPUs via point-to-point sends (torch.distributed.send/recv).
"""

import torch
import torch.distributed as dist
from torch import nn
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging

logger_ep = logging.getLogger("expert_placement")


# ──────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────

@dataclass
class ExpertPlacementConfig:
    rebalance_every_n_steps: int   = 2   # trigger every N forward passes
    warmup_steps:            int   = 300   # don't rebalance during warmup
    max_swap_iterations:     int   = 60    # greedy iterations per rebalance
    min_improvement_frac:    float = 0.005 # stop if gain < 0.5 % per swap
    affinity_decay:          float = 0.9   # EMA decay for affinity matrix
    migrate_timeout_sec:     float = 60.0  # safeguard for dist ops


# ──────────────────────────────────────────────────────────────
# Affinity accumulator
# ──────────────────────────────────────────────────────────────

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

    def __init__(self, num_global_experts: int, device: torch.device, decay: float = 0.9):
        self.num_experts = num_global_experts
        self.device      = device
        self.decay       = decay
        # EMA affinity matrix – stays on CPU to save GPU memory
        self.affinity: torch.Tensor = torch.zeros(
            num_global_experts, num_global_experts, dtype=torch.float32
        )
        self.step = 0

    @torch.no_grad()
    def update(self, comm_matrix: torch.Tensor):
        """
        comm_matrix: [ep_world_size, num_global_experts]  (already gathered, on any device)
        """
        mat = comm_matrix.float().cpu()  # [R, E]

        # For each rank, the co-activation signal is the outer product of its token counts.
        # Sum over ranks → [E, E] affinity contribution this step.
        new_aff = torch.einsum('re,rf->ef', mat, mat)   # [E, E]

        # Remove diagonal (self-affinity is not useful)
        new_aff.fill_diagonal_(0.0)

        # Normalize so scale doesn't drift
        total = new_aff.sum()
        if total > 0:
            new_aff = new_aff / total

        # EMA update
        if self.step == 0:
            self.affinity = new_aff
        else:
            self.affinity = self.decay * self.affinity + (1.0 - self.decay) * new_aff

        self.step += 1

    def get(self) -> torch.Tensor:
        """Returns [E, E] affinity matrix (CPU, float32)."""
        return self.affinity.clone()

    def reset(self):
        self.affinity.zero_()
        self.step = 0


# ──────────────────────────────────────────────────────────────
# Greedy placement optimizer
# ──────────────────────────────────────────────────────────────

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
        self.epg            = experts_per_gpu   # experts per GPU (assumed uniform)
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
        swaps     = 0

        for iteration in range(self.config.max_swap_iterations):
            best_delta, best_pair = 0.0, None

            # Iterate over cross-GPU pairs only
            for gpu_a in range(self.num_gpus):
                experts_a = [e for e in range(self.num_experts) if p[e] == gpu_a]
                for gpu_b in range(gpu_a + 1, self.num_gpus):
                    experts_b = [e for e in range(self.num_experts) if p[e] == gpu_b]
                    for ea in experts_a:
                        for eb in experts_b:
                            d = self._swap_delta(A, p, ea, eb)
                            if d < best_delta:
                                best_delta, best_pair = d, (ea, eb)

            if best_pair is None:
                logger_ep.info(f"Greedy converged after {iteration} iterations")
                break

            ea, eb = best_pair
            p[ea], p[eb] = p[eb], p[ea]
            cur_cost += best_delta
            swaps    += 1

            if init_cost > 0 and abs(best_delta) / init_cost < self.config.min_improvement_frac:
                logger_ep.info(f"Early stop: marginal gain {best_delta/init_cost:.4f}")
                break

        improve_pct = (init_cost - cur_cost) / init_cost * 100 if init_cost > 0 else 0.0
        stats = dict(init_cost=init_cost, final_cost=cur_cost,
                     improve_pct=improve_pct, swaps=swaps, iters=iteration + 1)
        logger_ep.info(
            f"Placement opt: {improve_pct:.1f}% cross-GPU traffic reduction "
            f"({swaps} swaps, {iteration+1} iters)"
        )
        return p, stats


# ──────────────────────────────────────────────────────────────
# Expert migrator
# ──────────────────────────────────────────────────────────────

class ExpertMigrator:
    """
    Physically moves expert weights between GPUs using P2P communication.

    Relies on the read/write helpers already present in Experts:
        get_expert_state_dict(local_idx)  → dict of CPU tensors
        load_expert_state_dict(local_idx, sd)
    """

    def __init__(self, experts_module, ep_group, num_local_experts: int, ep_size: int):
        self.experts           = experts_module   # deepspeed_moe.experts  (Experts instance)
        self.ep_group          = ep_group
        self.num_local_experts = num_local_experts
        self.ep_size           = ep_size
        self.my_rank           = dist.get_rank(ep_group)

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
            numel = v.numel()
            new_sd[k] = flat[offset:offset + numel].reshape(v.shape).to(v.dtype)
            offset += numel
        return new_sd

    # ── main ──────────────────────────────────────────────────

    @torch.no_grad()
    def execute_migrations(self, migrations: List[Tuple[int, int, int]]):
        """
        migrations: list of (global_expert_id, from_rank, to_rank)

        Strategy: process one migration at a time.
        For each migration the sender sends, the receiver receives.
        All other ranks are idle for that migration (simple & correct;
        can be parallelized later with async ops if needed).
        """
        if not migrations:
            return

        logger_ep.info(f"Rank {self.my_rank}: executing {len(migrations)} expert migrations")

        for global_eid, from_rank, to_rank in migrations:
            from_local = global_eid % self.num_local_experts
            to_local   = global_eid % self.num_local_experts  # same slot on destination

            # ── Determine flat tensor size (all ranks need to know) ──
            # Use a dummy broadcast: rank `from_rank` computes the size.
            size_tensor = torch.zeros(1, dtype=torch.long)
            if self.my_rank == from_rank:
                sd   = self.experts.get_expert_state_dict(from_local)
                flat = self._state_dict_to_tensor(sd)
                size_tensor[0] = flat.numel()

            dist.broadcast(size_tensor, src=from_rank, group=self.ep_group)
            numel = size_tensor.item()

            # ── Sender → Receiver ────────────────────────────────────
            if self.my_rank == from_rank:
                dist.send(flat, dst=to_rank, group=self.ep_group)
                logger_ep.debug(f"  Sent   expert g{global_eid} → rank {to_rank}  ({numel} params)")

            elif self.my_rank == to_rank:
                recv_flat = torch.zeros(numel, dtype=torch.float32)
                dist.recv(recv_flat, src=from_rank, group=self.ep_group)
                # Reconstruct state dict using the current expert as shape template
                sd_template = self.experts.get_expert_state_dict(to_local)
                new_sd      = self._tensor_to_state_dict(recv_flat, sd_template)
                self.experts.load_expert_state_dict(to_local, new_sd)
                logger_ep.debug(f"  Recv'd expert g{global_eid} from rank {from_rank}  ({numel} params)")

            # All other ranks: nothing to do for this migration.

        # Barrier to make sure all migrations finished before training resumes.
        dist.barrier(group=self.ep_group)
        logger_ep.info(f"Rank {self.my_rank}: migrations complete")


# ──────────────────────────────────────────────────────────────
# Main manager – wire everything together
# ──────────────────────────────────────────────────────────────

class ExpertPlacementManager:
    """
    Coordinates affinity tracking, placement optimization, and migration.

    Instantiate once per MOELayer, then call on_forward() after every
    forward pass (pass the comm_matrix that MOELayer already computes).

    Example usage inside MOELayer.forward() – add at the very end:

        # existing line:
        self.comm_matrix_history.append(comm_matrix)

        # NEW – add this:
        self.placement_manager.on_forward(comm_matrix)

    And in MOELayer.__init__ – add after self.ep_size = ep_size:

        self.placement_manager = ExpertPlacementManager(
            experts_module   = experts,          # Experts instance
            ep_group         = None,             # set later via _set_ep_group
            num_local_experts= num_local_experts,
            ep_size          = ep_size,
            config           = ExpertPlacementConfig(),
        )

    Then in _set_ep_group():
        self.placement_manager.ep_group = ep_group
        self.placement_manager.migrator = ExpertMigrator(
            experts_module   = self.experts,
            ep_group         = ep_group,
            num_local_experts= self.num_local_experts,
            ep_size          = self.ep_size,
        )
    """

    def __init__(
        self,
        experts_module,
        ep_group,                    # may be None at init time (set in _set_ep_group)
        num_local_experts: int,
        ep_size:           int,
        config:            ExpertPlacementConfig = None,
        device:            torch.device = None,
    ):
        self.experts           = experts_module
        self.ep_group          = ep_group
        self.num_local_experts = num_local_experts
        self.ep_size           = ep_size
        self.num_experts       = ep_size * num_local_experts
        self.config            = config or ExpertPlacementConfig()
        self.device            = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initial naive placement: expert e lives on rank e // num_local_experts
        self.placement: List[int] = [
            e // num_local_experts for e in range(self.num_experts)
        ]

        self.accumulator = AffinityAccumulator(
            self.num_experts, self.device, decay=self.config.affinity_decay
        )
        self.optimizer = GreedyPlacementOptimizer(
            self.num_experts, self.ep_size, num_local_experts, self.config
        )
        self.migrator: Optional[ExpertMigrator] = None  # created after ep_group is set

        self.global_step        = 0
        self.last_rebalance     = 0
        self.rebalance_history  = []

    # ── called from _set_ep_group ─────────────────────────────

    def set_ep_group(self, ep_group):
        self.ep_group = ep_group
        self.migrator = ExpertMigrator(
            experts_module   = self.experts,
            ep_group         = ep_group,
            num_local_experts= self.num_local_experts,
            ep_size          = self.ep_size,
        )

    # ── main entry point ──────────────────────────────────────

    def on_forward(self, comm_matrix: torch.Tensor):
        """
        Call at the end of every MOELayer.forward().

        comm_matrix: [ep_world_size, num_global_experts]  (already all-gathered)
        """
        self.global_step += 1

        # Always update affinity statistics
        self.accumulator.update(comm_matrix)

        # Skip rebalancing during warmup or if not ready
        if self.global_step < self.config.warmup_steps:
            return
        if self.ep_group is None or self.migrator is None:
            return

        steps_since = self.global_step - self.last_rebalance
        if steps_since < self.config.rebalance_every_n_steps:
            return

        self._rebalance()

    # ── rebalance pipeline ────────────────────────────────────

    def _rebalance(self):
        """Run optimization on rank 0, broadcast result, migrate weights."""
        my_rank = dist.get_rank(self.ep_group)

        # ── 1. Compute new placement (rank 0 only) ────────────
        if my_rank == 0:
            affinity    = self.accumulator.get()   # [E, E] CPU
            new_placement, stats = self.optimizer.optimize(affinity, self.placement)
            self.rebalance_history.append({"step": self.global_step, **stats})
            logger_ep.info(
                f"Step {self.global_step} | rebalance: "
                f"{stats['improve_pct']:.1f}% improvement | {stats['swaps']} swaps"
            )
        else:
            new_placement = [0] * self.num_experts

        # ── 2. Broadcast new placement to all ranks ───────────
        pt = torch.tensor(new_placement, dtype=torch.int32)
        dist.broadcast(pt, src=0, group=self.ep_group)
        new_placement = pt.tolist()

        # ── 3. Compute migrations ─────────────────────────────
        migrations = []   # (global_expert_id, from_rank, to_rank)
        for eid, new_rank in enumerate(new_placement):
            old_rank = self.placement[eid]
            if old_rank != new_rank:
                migrations.append((eid, old_rank, new_rank))

        # ── 4. Execute migrations ─────────────────────────────
        if migrations:
            self.migrator.execute_migrations(migrations)

        # ── 5. Update local state ─────────────────────────────
        self.placement      = new_placement
        self.last_rebalance = self.global_step

        # Reset accumulator so next window starts fresh
        self.accumulator.reset()

    # ── introspection ─────────────────────────────────────────

    def get_placement(self) -> Dict[int, int]:
        """Returns {global_expert_id: gpu_rank}."""
        return {e: r for e, r in enumerate(self.placement)}

    def pending_migrations(self, new_placement: List[int]) -> List[Tuple[int, int, int]]:
        return [(e, self.placement[e], new_placement[e])
                for e in range(self.num_experts) if self.placement[e] != new_placement[e]]