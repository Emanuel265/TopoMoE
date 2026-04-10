import numpy as np
import torch
from src.model.topology import *

def comm_cost_single_layer(placement: list[int], coaccess: torch.Tensor | np.ndarray, link_map: dict, token_d: float) -> float:
    """
    Communication cost for a single layer.
    """
    if isinstance(coaccess, torch.Tensor):
        coaccess = coaccess.detach().cpu().numpy()

    num_experts = len(placement)
    total_tokens = np.sum(coaccess)

    bw_matrix = np.zeros((num_experts, num_experts), dtype=float)
    for i in range(num_experts):
        for j in range(num_experts):
            if i == j:
                bw_matrix[i, j] = np.inf
            else:
                a, b = placement[i], placement[j]
                key = (min(a,b), max(a,b))
                link = link_map.get(key)
                bw_matrix[i,j] = float(link.bandwidth) if link and getattr(link,"bandwidth",None) else 1e-9

    cost_matrix = coaccess * (token_d / bw_matrix)
    np.fill_diagonal(cost_matrix, 0.0)
    total_cost = np.sum(cost_matrix)

    min_bw = min((float(l.bandwidth) for l in link_map.values() if getattr(l,"bandwidth",0) > 0), default=1e-9)
    norm_cost = total_cost / (total_tokens * token_d / min_bw + 1e-9)
    return float(np.clip(norm_cost,0,1))


def load_cost_single_layer(placement: list[int], coaccess: np.ndarray, num_devices: int) -> tuple[float, np.ndarray]:
    """
    Load imbalance cost for a single layer.

    Args:
        placement: [num_experts] list of device IDs
        coaccess: [num_experts x num_experts] token co-access
        num_devices: total number of devices

    Returns:
        float: normalized load imbalance cost [0,1]
        device_token_counts: tokens per device
    """
    device_token_counts = np.zeros(num_devices, dtype=float)
    tokens_per_expert = np.sum(coaccess, axis=0)
    np.add.at(device_token_counts, placement, tokens_per_expert)

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
        placement: [num_experts] list of device IDs
        replicas: optional [num_experts] list of replica device IDs

    Returns:
        float: normalized failure cost [0,1]
    """
    if replicas is None:
        return 0.0
    placement = np.array(placement)
    replicas = np.array(replicas)
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
        placement: [num_experts] device IDs
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
    # print(f"[CUSTOM DEBUG] single_layer_cost called with placement: {placement}, coaccess: {coaccess}, token_d: {token_d}, alpha: {alpha}, beta: {beta}, gamma: {gamma}, replicas: {replicas}")
    comm = comm_cost_single_layer(placement, coaccess, topology.link_map, token_d)
    load, _ = load_cost_single_layer(placement, coaccess, len(topology.devices))
    fail = fail_cost_single_layer(placement, replicas)
    return alpha * comm + beta * load + gamma * fail

def total_cost(
    placements: list[list[int]],
    coaccess_matrix: torch.Tensor | np.ndarray,
    topology,
    token_d: float,
    alpha: float,
    beta: float,
    gamma: float,
    replicas_matrix: list[list[int]] | None = None
) -> float:
    """
    Computes total cost across multiple layers.

    Args:
        placements: list of [num_experts] placements per layer
        coaccess_matrix: tensor or numpy array [num_layers x num_experts x num_experts]
        topology: Topology object
        token_d: token scaling factor
        alpha, beta, gamma: weights
        replicas_matrix: optional list of replicas per layer

    Returns:
        float: total weighted cost
    """
    print(f"[CUSTOM DEBUG] total_cost called with placements: {placements}, coaccess_matrix: {coaccess_matrix}, coaccess_matrix.shape: {coaccess_matrix.shape}, token_d: {token_d}, alpha: {alpha}, beta: {beta}, gamma: {gamma}, replicas_matrix: {replicas_matrix}")
    coaccess_np = (
        coaccess_matrix.detach().cpu().numpy()
        if isinstance(coaccess_matrix, torch.Tensor)
        else np.array(coaccess_matrix, dtype=float)
    )

    num_layers = len(placements)
    total = 0.0
    for l in range(num_layers):
        replicas = replicas_matrix[l] if replicas_matrix is not None else None
        total += single_layer_cost(
            placements[l],
            coaccess_np[l],
            topology,
            token_d,
            alpha,
            beta,
            gamma,
            replicas
        )
    return total