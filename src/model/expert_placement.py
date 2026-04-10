import random
from collections import defaultdict
from sortedcontainers import SortedDict
from collections import Counter
import numpy as np
from sklearn.cluster import SpectralClustering
from scipy.optimize import linear_sum_assignment

import torch
import deepspeed
import tutel

from src.model.cost_function import single_layer_cost
from src.model.topology import Topology


def get_cliques_at_level(topology: Topology, level: int):
    """Build cliques of devices at a given hierarchy level."""
    cliques = [{d.id} for d in topology.devices.values()]

    for link in topology.links[level]:
        dev_a, dev_b = link.devices
        clique_a = next((c for c in cliques if dev_a.id in c), None)
        clique_b = next((c for c in cliques if dev_b.id in c), None)
        if clique_a is not None and clique_b is not None and clique_a is not clique_b:
            clique_a.update(clique_b)
            cliques.remove(clique_b)

    return [list(c) for c in cliques]


def spectral_grouping(coaccess_matrix, num_clusters):
    """Return expert clusters via spectral clustering."""
    clustering = SpectralClustering(
        n_clusters=num_clusters, affinity='precomputed', assign_labels='kmeans', random_state=0
    )
    labels = clustering.fit_predict(coaccess_matrix)
    clusters = defaultdict(list)
    for idx, label in enumerate(labels):
        clusters[label].append(idx)
    return list(clusters.values())


def select_experts_from_clusters(clusters, clique_size):
    """Select experts from spectral clusters for a clique."""
    clusters.sort(key=len, reverse=True)
    selected = []

    # Take from largest cluster first
    for cluster in clusters:
        if len(cluster) >= clique_size:
            selected = cluster[:clique_size]
            del cluster[:clique_size]
            return selected

    # Combine from multiple clusters if needed
    while len(selected) < clique_size and clusters:
        cluster = clusters[0]
        take = min(clique_size - len(selected), len(cluster))
        selected.extend(cluster[:take])
        del cluster[:take]
        if not cluster:
            clusters.pop(0)

    return selected

def rebalance_load_no_cliques(mapping, tau_balance):
    """Rebalance experts across devices to respect max load (no cliques)."""
    device_to_experts = defaultdict(list)
    for expert, device in enumerate(mapping):
        device_to_experts[device].append(expert)

    device_loads = {d: len(experts) for d, experts in device_to_experts.items()}

    for _ in range(len(device_loads) * 2):
        overloaded = [d for d, load in device_loads.items() if load > tau_balance]
        underloaded = [d for d, load in device_loads.items() if load < tau_balance]
        if not overloaded or not underloaded:
            break

        # pick the most loaded device and least loaded device
        o_device = max(overloaded, key=lambda d: device_loads[d])
        u_device = min(underloaded, key=lambda d: device_loads[d])

        if not device_to_experts[o_device]:
            continue

        # move one expert from overloaded to underloaded
        expert = device_to_experts[o_device].pop()
        mapping[expert] = u_device
        device_to_experts[u_device].append(expert)

        device_loads[o_device] -= 1
        device_loads[u_device] += 1

    return mapping

def get_expert_placement(
    topology,
    num_experts: int,
    current_placement: list[list[int]],
    token_d: int,
    alpha: float,
    beta: float,
    gamma: float,
    tau_balance: float,
    replica_placement: list[list[int]] = None,
    coaccess_matrix: torch.Tensor = None,  # shape (n_layers, num_experts, num_experts)
    max_iters: int = 50,
) -> tuple[list[list[int]], list[list[int]]]:
    """
    Compute expert placement for a multi-layer MoE.

    Uses an external cost(...) and rebalance_load_no_cliques(...) for optimization.
    Logic unchanged; refactored for readability and maintainability.
    """
    # print(f"\nStarting expert redistribution with current placement {current_placement}. Decision is based on coaccess_matrix: {coaccess_matrix}")

    # ============================================================
    # Validation
    # ============================================================
    if not topology or not hasattr(topology, "devices") or len(topology.devices) == 0:
        print("GetPlacement: ERROR: Invalid topology (missing devices).")
        return None

    if coaccess_matrix is None or not isinstance(coaccess_matrix, torch.Tensor):
        print("GetPlacement: ERROR: coaccess_matrix must be a torch.Tensor.")
        return None

    if coaccess_matrix.dim() != 3:
        raise ValueError("coaccess_matrix must be 3D: (n_layers, num_experts, num_experts)")

    n_layers = int(coaccess_matrix.size(0))
    devices = sorted(topology.devices.values(), key=lambda d: d.id)
    device_ids = [d.id for d in devices]
    num_devices = len(devices)

    # ============================================================
    # Adjacency Map
    # ============================================================
    def build_adjacency(devices, link_map):
        adjacency = {d.id: set() for d in devices}
        if isinstance(link_map, dict):
            for key in link_map.keys():
                a, b = tuple(key)
                adjacency[a].add(b)
                adjacency[b].add(a)
        else:
            print("GetPlacement: topology.link_map not found; using full connectivity.")
            for d in devices:
                adjacency[d.id] = set(dd.id for dd in devices if dd.id != d.id)
        return {k: sorted(v) for k, v in adjacency.items()}

    link_map = getattr(topology, "link_map", None)
    adjacency = build_adjacency(devices, link_map)
    # print(f"GetPlacement: adjacency map: {adjacency}")

    # ============================================================
    # Bandwidth Helper
    # ============================================================
    def get_bandwidth_between(a_id: int, b_id: int) -> float:
        """Return link bandwidth between devices or small fallback value."""
        if a_id == b_id:
            return float("inf")
        if link_map:
            key = tuple(sorted((a_id, b_id)))
            link = link_map.get(key)
            if link and getattr(link, "bandwidth", None) is not None:
                bw = max(float(link.bandwidth), 1e-9)
                return bw
        return 1e-9  # default small bandwidth

    # ============================================================
    # Initialize placements
    # ============================================================
    placements = [[-1] * num_experts for _ in range(n_layers)]

    if current_placement and len(current_placement[0]) == num_experts:
        placements[0] = current_placement[0][:]
        # print("GetPlacement: using provided current_placement for layer 0.")
    else:
        placements[0] = [device_ids[e % num_devices] for e in range(num_experts)]
        # print(f"GetPlacement: initial placement layer 0: {placements[0]}")

    # ============================================================
    # Per-Layer Placement Optimization
    # ============================================================
    for l in range(1, n_layers):
        # print(f"\nGetPlacement: processing layer {l}/{n_layers - 1}")
        layer_comm = coaccess_matrix[l]
        prev_mapping = (
            current_placement[l - 1][:]
            if current_placement and len(current_placement) > l - 1
            else placements[l - 1][:]
        )

        total_comm = float(torch.sum(layer_comm).item())
        comm_prob = layer_comm.float() / total_comm if total_comm > 0 else layer_comm.clone().float()

        # --------------------------------------------------------
        # Build communication cost matrix
        # --------------------------------------------------------
        comm_cost_matrix = np.zeros((num_experts, num_experts), dtype=float)
        total_bandwidth = 0.0

        for ei in range(num_experts):
            src_dev = prev_mapping[ei]
            for ej in range(num_experts):
                candidate_dev = prev_mapping[ej]
                if src_dev == candidate_dev:
                    comm_cost_matrix[ei, ej] = 0.0
                    continue

                bw = get_bandwidth_between(src_dev, candidate_dev)
                total_bandwidth += bw if np.isfinite(bw) else 0.0
                prob = float(comm_prob[ei, ej].item())
                comm_cost_matrix[ei, ej] = 0.0 if prob <= 0 else prob * token_d / max(bw, 1e-9)

        # print(f"GetPlacement: comm_cost_matrix {comm_cost_matrix}; total_bandwidth={total_bandwidth:.3e}")

        # --------------------------------------------------------
        # Hungarian Assignment
        # --------------------------------------------------------
        try:
            row_ind, col_ind = linear_sum_assignment(comm_cost_matrix)
        except Exception as e:
            print(f"GetPlacement: linear_sum_assignment failed: {e}, using identity mapping.")
            row_ind, col_ind = np.arange(num_experts), np.arange(num_experts)

        placement_mapping = [-1] * num_experts
        for src_idx, tgt_idx in zip(row_ind, col_ind):
            placement_mapping[tgt_idx] = prev_mapping[src_idx]

        # Fill unassigned experts round-robin
        for e in range(num_experts):
            if placement_mapping[e] == -1:
                placement_mapping[e] = device_ids[e % num_devices]

        # print(f"GetPlacement: mapping after assignment for layer {l}: {placement_mapping}")

        # --------------------------------------------------------
        # Local Search Improvement
        # --------------------------------------------------------
        placements[l] = placement_mapping[:]
        best_mapping = placements[l][:]

        coaccess_for_cost = coaccess_matrix.detach().cpu()
        best_cost = single_layer_cost(placements[l], coaccess_for_cost, topology, token_d, alpha, beta, gamma, replica_placement)

        for it in range(max_iters):
            e = random.randrange(num_experts)
            current_dev = best_mapping[e]
            candidates = adjacency.get(current_dev, device_ids) or device_ids
            new_dev = random.choice(candidates)
            if new_dev == current_dev:
                continue

            tmp_mapping = best_mapping[:]
            tmp_mapping[e] = new_dev

            # Make a full placements copy with this layer modified
            tmp_placements = [p[:] if i != l else tmp_mapping[:] for i, p in enumerate(placements)]
            tmp_cost = single_layer_cost(tmp_mapping, coaccess_for_cost, topology, token_d, alpha, beta, gamma, replica_placement)

            if tmp_cost < best_cost:
                best_cost, best_mapping = tmp_cost, tmp_mapping
                print(f"GetPlacement: iter {it}: improved cost -> {best_cost:.6f} (expert {e} → {new_dev})")

        placements[l] = best_mapping[:]  # update final layer mapping
        # print(f"GetPlacement: best cost after local search (layer {l}): {best_cost} with mapping {best_mapping}")

        # --------------------------------------------------------
        # Rebalance Placement
        # --------------------------------------------------------
        try:
            rebalanced = rebalance_load_no_cliques(best_mapping, tau_balance) or best_mapping[:]
        except Exception as ex:
            print(f"GetPlacement: rebalance_load_no_cliques failed: {ex}, using best_mapping.")
            rebalanced = best_mapping[:]

        placements[l] = rebalanced
        # print(f"GetPlacement: layer {l} device loads: {dict(Counter(rebalanced))}")

    print(f"GetPlacement: Calculated following new placement: {placements}")

    # ============================================================
    # Replica Placement
    # ============================================================
    print(f"\nGetPlacement: Starting replica placement from current replica placement {replica_placement}")
    replica_placements = [[-1] * num_experts for _ in range(n_layers)]

    for l in range(n_layers):
        primary_map = placements[l]
        used_devices = Counter(primary_map)
        device_loads = {d: used_devices.get(d, 0) for d in device_ids}
        replica_map = [-1] * num_experts

        for e, primary_dev in enumerate(primary_map):
            candidates = [d for d in device_ids if d != primary_dev]
            best_candidate = None

            # Prefer devices outside same clique
            if hasattr(topology, "links") and isinstance(topology.links, dict):
                try:
                    cliques = get_cliques_at_level(topology, level=0)
                    clique_of_primary = next((c for c in cliques if primary_dev in c), [])
                    far_candidates = [d for d in candidates if d not in clique_of_primary]
                    if far_candidates:
                        candidates = far_candidates
                except Exception:
                    pass

            # Choose least-loaded device
            best_candidate = min(candidates, key=lambda d: device_loads[d])
            replica_map[e] = best_candidate
            device_loads[best_candidate] += 1

        replica_placements[l] = replica_map
        # print(f"Replica placement layer {l}: {replica_map}")
    
    print(f"Replica placement {replica_placements}")

    # ============================================================
    # Validation
    # ============================================================
    # for l_idx, layer_map in enumerate(placements):
    #     device_counts = Counter(layer_map)
    #     for d in device_ids:
    #         if device_counts.get(d, 0) == 0:
    #             raise ValueError(f"GETPlacement: Invalid placement: device {d} has 0 experts in layer {l_idx}.")

    # print("\nFinal placements across all layers:")
    # for l, layer_map in enumerate(placements):
    #     print(f"Layer {l}: {layer_map} (loads: {dict(Counter(layer_map))})")

    return placements, replica_placements


def get_expert_placement_single_layer(
    topology,
    num_experts: int,
    prev_placement: list[int],
    token_d: int,
    alpha: float,
    beta: float,
    gamma: float,
    tau_balance: float,
    replica_placement: list[int] = None,
    coaccess_matrix: torch.Tensor = None,
    max_iters: int = 50,
) -> tuple[list[int], list[int]]:
    """
    Compute expert placement for a single MoE layer given the coaccess
    matrix between the previous layer and this layer.
    """

    # ============================================================
    # Validation
    # ============================================================
    if not topology or not hasattr(topology, "devices") or len(topology.devices) == 0:
        print("GetPlacement: ERROR: Invalid topology.")
        return None, None

    if coaccess_matrix is None or not isinstance(coaccess_matrix, torch.Tensor):
        print("GetPlacement: ERROR: coaccess_matrix must be a torch.Tensor.")
        return None, None

    if coaccess_matrix.dim() != 2:
        raise ValueError("coaccess_matrix must be 2D: (num_experts, num_experts)")

    devices     = sorted(topology.devices.values(), key=lambda d: d.id)
    device_ids  = [d.id for d in devices]
    num_devices = len(devices)

    # ============================================================
    # Adjacency Map
    # ============================================================
    def build_adjacency(devices, link_map):
        adjacency = {d.id: set() for d in devices}
        if isinstance(link_map, dict):
            for key in link_map.keys():
                a, b = tuple(key)
                adjacency[a].add(b)
                adjacency[b].add(a)
        else:
            for d in devices:
                adjacency[d.id] = set(dd.id for dd in devices if dd.id != d.id)
        return {k: sorted(v) for k, v in adjacency.items()}

    link_map  = getattr(topology, "link_map", None)
    adjacency = build_adjacency(devices, link_map)

    # ============================================================
    # Bandwidth Helper
    # ============================================================
    def get_bandwidth_between(a_id: int, b_id: int) -> float:
        if a_id == b_id:
            return float("inf")
        if link_map:
            key  = tuple(sorted((a_id, b_id)))
            link = link_map.get(key)
            if link and getattr(link, "bandwidth", None) is not None:
                return max(float(link.bandwidth), 1e-9)
        return 1e-9

    # ============================================================
    # Communication Cost Matrix
    # ============================================================
    total_comm = float(torch.sum(coaccess_matrix).item())
    comm_prob  = (coaccess_matrix.float() / total_comm
                  if total_comm > 0
                  else coaccess_matrix.clone().float())

    comm_cost_matrix = np.zeros((num_experts, num_experts), dtype=float)
    for ei in range(num_experts):
        src_dev = prev_placement[ei]
        for ej in range(num_experts):
            candidate_dev = prev_placement[ej]
            if src_dev == candidate_dev:
                continue
            bw   = get_bandwidth_between(src_dev, candidate_dev)
            prob = float(comm_prob[ei, ej].item())
            comm_cost_matrix[ei, ej] = 0.0 if prob <= 0 else prob * token_d / max(bw, 1e-9)

    # ============================================================
    # Hungarian Assignment
    # ============================================================
    try:
        row_ind, col_ind = linear_sum_assignment(comm_cost_matrix)
    except Exception as e:
        print(f"GetPlacement: linear_sum_assignment failed: {e}, using identity.")
        row_ind, col_ind = np.arange(num_experts), np.arange(num_experts)

    placement = [-1] * num_experts
    for src_idx, tgt_idx in zip(row_ind, col_ind):
        placement[tgt_idx] = prev_placement[src_idx]

    # Fill unassigned round-robin
    for e in range(num_experts):
        if placement[e] == -1:
            placement[e] = device_ids[e % num_devices]

    # ============================================================
    # Local Search Improvement
    # ============================================================
    best_mapping = placement[:]
    coaccess_cpu = coaccess_matrix.detach().cpu()

    # Wrap into 2-layer structure cost() expects
    placements_wrap = [prev_placement[:], best_mapping[:]]
    best_cost = single_layer_cost(placements_wrap[1], coaccess_cpu.unsqueeze(0), topology, token_d, alpha, beta, gamma, replica_placement)

    for it in range(max_iters):
        e           = random.randrange(num_experts)
        current_dev = best_mapping[e]
        candidates  = adjacency.get(current_dev, device_ids) or device_ids
        new_dev     = random.choice(candidates)
        if new_dev == current_dev:
            continue

        tmp_mapping    = best_mapping[:]
        tmp_mapping[e] = new_dev
        tmp_placements = [prev_placement[:], tmp_mapping[:]]
        tmp_cost       = single_layer_cost(tmp_mapping, coaccess_cpu.unsqueeze(0), topology, token_d, alpha, beta, gamma, replica_placement)

        if tmp_cost < best_cost:
            best_cost, best_mapping = tmp_cost, tmp_mapping
            print(f"GetPlacement: iter {it}: improved cost -> {best_cost:.6f} (expert {e} → {new_dev})")

    # ============================================================
    # Rebalance
    # ============================================================
    try:
        best_mapping = rebalance_load_no_cliques(best_mapping, tau_balance) or best_mapping
    except Exception as ex:
        print(f"GetPlacement: rebalance failed: {ex}, using best_mapping.")

    print(f"GetPlacement: final placement: {best_mapping}")

    # ============================================================
    # Replica Placement
    # ============================================================
    used_devices  = Counter(best_mapping)
    device_loads  = {d: used_devices.get(d, 0) for d in device_ids}
    replica_map   = [-1] * num_experts

    for e, primary_dev in enumerate(best_mapping):
        candidates = [d for d in device_ids if d != primary_dev]

        if hasattr(topology, "links") and isinstance(topology.links, dict):
            try:
                cliques           = get_cliques_at_level(topology, level=0)
                clique_of_primary = next((c for c in cliques if primary_dev in c), [])
                far_candidates    = [d for d in candidates if d not in clique_of_primary]
                if far_candidates:
                    candidates = far_candidates
            except Exception:
                pass

        best_candidate    = min(candidates, key=lambda d: device_loads[d])
        replica_map[e]    = best_candidate
        device_loads[best_candidate] += 1

    print(f"GetPlacement: replica placement: {replica_map}")
    return best_mapping, replica_map