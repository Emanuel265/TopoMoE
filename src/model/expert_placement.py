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


def get_expert_placement_single_layer(
    topology,
    placement_init: list[int],
    coaccess_matrix: torch.Tensor,
    token_d: float,
    alpha: float,
    beta: float,
    gamma: float,
    replicas: list[int] | None = None,
    max_iters: int = 100,
):
    """
    Optimize expert placement for a single layer using your cost function.
    
    Args:
        topology: Topology object
        placement_init: [num_experts] initial placement where placement[expert_id] = device_id
        coaccess_matrix: [num_experts x num_experts] co-access matrix
        token_d: token dimension scaling factor
        alpha, beta, gamma: cost weighting factors
        replicas: optional replica placement
        max_iters: number of optimization iterations
        
    Returns:
        list[int]: optimized placement where placement[expert_id] = device_id
    """

    if isinstance(coaccess_matrix, torch.Tensor):
        coaccess = coaccess_matrix.detach().cpu().numpy()
    else:
        coaccess = coaccess_matrix

    placement = placement_init[:]
    best_placement = placement[:]

    best_cost = single_layer_cost(
        best_placement, coaccess, topology,
        token_d, alpha, beta, gamma, replicas
    )

    print("init cost: ", best_cost, best_placement)

    num_experts = len(placement)
    num_devices = len(topology.devices)

    # ------------------------------------------------------------
    # Local swap optimization - swap device assignments of two experts
    # ------------------------------------------------------------
    for _ in range(max_iters):
        # Pick two random experts
        expert_a, expert_b = random.sample(range(num_experts), 2)

        # Swap their device assignments
        new_placement = best_placement[:]
        new_placement[expert_a], new_placement[expert_b] = \
            new_placement[expert_b], new_placement[expert_a]

        new_cost = single_layer_cost(
            new_placement, coaccess, topology,
            token_d, alpha, beta, gamma, replicas
        )

        if new_cost < best_cost:
            best_cost = new_cost
            best_placement = new_placement
    
    print("cost after: ", best_cost, best_placement)

    return best_placement


def build_migration_plan(
    placement_old: list[int],
    placement_new: list[int]
):
    """
    Build direct expert migration plan from old to new placement.
    
    Args:
        placement_old: [num_experts] where placement_old[expert_id] = old_device_id
        placement_new: [num_experts] where placement_new[expert_id] = new_device_id
        num_local_experts: number of experts per device (for computing local indices)
    
    Returns:
        List of (src_rank, dst_rank, src_local_idx, dst_local_idx, expert_id)
        
    Note: This assumes experts are stored contiguously on each device.
          If device D has experts [e1, e2, e3], they occupy local indices [0, 1, 2].
    """

    migrations = []
    
    # Track number of experts on each device to compute local indices
    old_device_counts = defaultdict(int)
    new_device_counts = defaultdict(int)
    
    old_local_indices = {}
    for expert_id in range(len(placement_old)):
        device_id = placement_old[expert_id]
        old_local_indices[expert_id] = old_device_counts[device_id]
        old_device_counts[device_id] += 1
    
    new_local_indices = {}
    for expert_id in range(len(placement_new)):
        device_id = placement_new[expert_id]
        new_local_indices[expert_id] = new_device_counts[device_id]
        new_device_counts[device_id] += 1

    # Build migration list
    for expert_id in range(len(placement_old)):
        src_rank = placement_old[expert_id]
        dst_rank = placement_new[expert_id]
        src_idx = old_local_indices[expert_id]
        dst_idx = new_local_indices[expert_id]

        # Skip if already in correct place
        if src_rank == dst_rank and src_idx == dst_idx:
            continue

        migrations.append((
            src_rank,
            dst_rank,
            src_idx,
            dst_idx,
            expert_id
        ))

    return migrations