import numpy as np
import torch
from src.model.topology import *

def comm_cost_single_layer(
        placement: list[int],
        coaccess: torch.Tensor | np.ndarray,
        topology: Topology,
        token_d: float,
    ) -> float:    
    """
    Communication cost for a single layer.
    
    Args:
        placement: [num_experts] list where placement[expert_id] = device_id
        coaccess: [num_experts x num_experts] token co-access matrix
        topology: Topology object with bandwidth matrices
        token_d: token dimension/scaling factor
    """
    if isinstance(coaccess, torch.Tensor):
        coaccess = coaccess.detach().cpu().numpy()

    num_experts = len(placement)
    total_tokens = np.sum(coaccess)
    
    # Ensure matrices are built
    if topology.bw_matrix is None:
        topology.build_matrices()

    # Build bandwidth matrix between experts based on their device placement
    # This is now O(E^2) instead of O(E^2 * lookups)
    bw_matrix = np.zeros((num_experts, num_experts), dtype=float)
    for i in range(num_experts):
        for j in range(num_experts):
            dev_i = placement[i]
            dev_j = placement[j]
            bw_matrix[i, j] = topology.bw_matrix[dev_i, dev_j]

    # Avoid division by infinity (same device traffic)
    bw_matrix = np.where(np.isinf(bw_matrix), 1e12, bw_matrix)
    
    # Communication cost = tokens * data_per_token / bandwidth
    cost_matrix = coaccess * (token_d / (bw_matrix + 1e-9))
    np.fill_diagonal(cost_matrix, 0.0)
    total_cost = np.sum(cost_matrix)

    # Normalize by worst case (all traffic on slowest link)
    min_bw = topology.bw_matrix[topology.bw_matrix > 0].min() if np.any(topology.bw_matrix > 0) else 1e-9
    worst_case = total_tokens * token_d / min_bw
    norm_cost = total_cost / (worst_case + 1e-9)
    
    return float(np.clip(norm_cost, 0, 1))


def load_cost_single_layer(
    placement: list[int],
    coaccess: np.ndarray,
    num_devices: int
) -> tuple[float, np.ndarray]:    
    """
    Load imbalance cost for a single layer.

    Args:
        placement: [num_experts] list where placement[expert_id] = device_id
        coaccess: [num_experts x num_experts] token co-access
        num_devices: total number of devices

    Returns:
        float: normalized load imbalance cost [0,1]
        device_token_counts: tokens per device
    """
    num_experts = len(placement)
    device_token_counts = np.zeros(num_devices, dtype=float)
    
    # Sum tokens across rows to get tokens per expert
    tokens_per_expert = np.sum(coaccess, axis=0)

    for expert_id in range(num_experts):
        dev_id = placement[expert_id]
        device_token_counts[dev_id] += tokens_per_expert[expert_id]

    if np.any(device_token_counts == 0):
        return 1.0, device_token_counts

    mean_load = device_token_counts.mean()
    load = np.sum((device_token_counts - mean_load) ** 2)
    worst_case = (np.sum(device_token_counts) - mean_load) ** 2 + (num_devices - 1) * (mean_load ** 2)
    norm_load = load / (worst_case + 1e-9)
    return float(np.clip(norm_load, 0.0, 1.0)), device_token_counts


def fail_cost_single_layer(placement: list[int], replicas: list[int] | None = None) -> float:
    """
    Failure/replica cost for a single layer.

    Args:
        placement: [num_experts] list where placement[expert_id] = device_id
        replicas: optional [num_experts] list where replicas[expert_id] = replica_device_id

    Returns:
        float: normalized failure cost [0,1]
              (fraction of experts whose primary and replica are on same device)
    """
    if replicas is None:
        return 0.0
    placement = np.array(placement)
    replicas = np.array(replicas)
    # Cost is high when primary and replica are on same device (bad for fault tolerance)
    norm_fail = np.sum(placement == replicas) / len(placement)
    return float(np.clip(norm_fail, 0.0, 1.0))


def single_layer_cost(
    placement: list[int],
    coaccess: np.ndarray,
    topology,
    token_d: float,
    alpha: float,
    beta: float,
    gamma: float,
    replicas: list[int] | None = None
) -> float:
    """
    Total cost for a single layer combining communication, load, and failure.

    Args:
        placement: [num_experts] list where placement[expert_id] = device_id
        coaccess: [num_experts x num_experts] co-access matrix
        topology: Topology object
        token_d: token scaling factor
        alpha, beta, gamma: weighting factors
        replicas: optional replica device IDs

    Returns:
        float: total weighted cost
    """
    if isinstance(coaccess, torch.Tensor):
        coaccess = coaccess.detach().cpu().numpy()

    if coaccess.ndim == 3:
        coaccess = coaccess[0]
    
    comm = comm_cost_single_layer(
        placement,
        coaccess,
        topology.link_map,
        token_d,
        len(topology.devices)
    )

    load, _ = load_cost_single_layer(
        placement,
        coaccess,
        len(topology.devices)
    )

    fail = fail_cost_single_layer(placement, replicas)
    return alpha * comm + beta * load + gamma * fail