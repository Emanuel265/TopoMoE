import random
from collections import defaultdict
from sortedcontainers import SortedDict
from collections import Counter
import numpy as np
from sklearn.cluster import SpectralClustering
from scipy.optimize import linear_sum_assignment

import torch
import deepspeed

from src.model.cost_function import single_layer_cost
from src.model.topology import Topology


def get_expert_permutation_single_layer(
    topology,
    perm_init: list[int],
    coaccess_matrix: torch.Tensor,
    token_d: float,
    alpha: float,
    beta: float,
    gamma: float,
    replicas: list[int] | None = None,
    max_iters: int = 100,
):
    """
    Optimize expert permutation for a single layer using your cost function.
    """

    import random
    import numpy as np

    if isinstance(coaccess_matrix, torch.Tensor):
        coaccess = coaccess_matrix.detach().cpu().numpy()
    else:
        coaccess = coaccess_matrix

    perm = perm_init[:]
    best_perm = perm[:]

    best_cost = single_layer_cost(
        best_perm, coaccess, topology,
        token_d, alpha, beta, gamma, replicas
    )

    print("init cost: ", best_cost, best_perm)

    E = len(perm)

    # ------------------------------------------------------------
    # Local swap optimization
    # ------------------------------------------------------------
    for _ in range(max_iters):
        a, b = random.sample(range(E), 2)

        new_perm = best_perm[:]
        new_perm[a], new_perm[b] = new_perm[b], new_perm[a]

        new_cost = single_layer_cost(
            new_perm, coaccess, topology,
            token_d, alpha, beta, gamma, replicas
        )

        if new_cost < best_cost:
            best_cost = new_cost
            best_perm = new_perm
    
    print("cost after: ", best_cost, best_perm)

    return best_perm

def build_migration_plan(
    perm_old,
    perm_new,
    num_local_experts
):
    """
    Build direct expert migration plan (no swaps).

    Returns:
        List of (src_rank, dst_rank, src_local_idx, dst_local_idx, expert_id)
    """

    migrations = []

    def _get_rank_and_local(perm, e):
        pos = perm[e]
        rank = pos // num_local_experts
        local_idx = pos % num_local_experts
        return rank, local_idx

    for e in range(len(perm_old)):

        src_rank, src_idx = _get_rank_and_local(perm_old, e)
        dst_rank, dst_idx = _get_rank_and_local(perm_new, e)

        # skip if already in correct place
        if src_rank == dst_rank and src_idx == dst_idx:
            continue

        migrations.append((
            src_rank,
            dst_rank,
            src_idx,
            dst_idx,
            e
        ))

    return migrations