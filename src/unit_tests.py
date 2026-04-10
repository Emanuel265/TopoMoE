from escho_copy.expert_placement import get_expert_placement_single_layer

def test_simple_dominant_mapping():
    import torch

    num_experts = 3

    prev_placement = [0, 1, 2]

    coaccess = torch.tensor([
        [10, 1, 1],
        [1, 10, 1],
        [1, 1, 10],
    ], dtype=torch.float32)

    placement, _ = get_expert_placement_single_layer(
        topology=None,
        num_experts=num_experts,
        prev_placement=prev_placement,
        token_d=1,
        alpha=1,
        beta=1,
        gamma=1,
        tau_balance=1,
        coaccess_matrix=coaccess,
    )

    assert placement == prev_placement


def test_many_to_one_clustering():
    import torch

    prev_placement = [0, 1, 2]

    coaccess = torch.tensor([
        [1, 1, 10],
        [1, 1, 9],
        [1, 1, 1],
    ], dtype=torch.float32)

    placement, _ = get_expert_placement_single_layer(
        topology=None,
        num_experts=3,
        prev_placement=prev_placement,
        token_d=1,
        alpha=1,
        beta=1,
        gamma=1,
        tau_balance=1,
        coaccess_matrix=coaccess,
    )

    assert placement[2] == 0

def test_cross_device_preference():
    import torch

    prev_placement = [0, 0, 1, 1]

    coaccess = torch.tensor([
        [10, 10, 10, 10],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
    ], dtype=torch.float32)

    placement, _ = get_expert_placement_single_layer(
        topology=None,
        num_experts=4,
        prev_placement=prev_placement,
        token_d=1,
        alpha=1,
        beta=1,
        gamma=1,
        tau_balance=1,
        coaccess_matrix=coaccess,
    )

    assert all(p == 0 for p in placement)

def test_noisy_dominance():
    import torch

    prev_placement = [0, 1, 2]

    coaccess = torch.tensor([
        [10, 2, 2],
        [3, 9, 3],
        [2, 2, 8],
    ], dtype=torch.float32)

    placement, _ = get_expert_placement_single_layer(
        topology=None,
        num_experts=3,
        prev_placement=prev_placement,
        token_d=1,
        alpha=1,
        beta=1,
        gamma=1,
        tau_balance=1,
        coaccess_matrix=coaccess,
    )

    assert placement == [0, 1, 2]

def test_extreme_hotspot():
    import torch

    prev_placement = [0, 1, 2, 3]

    coaccess = torch.zeros(4, 4)
    coaccess[1, :] = 100

    placement, _ = get_expert_placement_single_layer(
        topology=None,
        num_experts=4,
        prev_placement=prev_placement,
        token_d=1,
        alpha=1,
        beta=1,
        gamma=1,
        tau_balance=1,
        coaccess_matrix=coaccess,
    )

    assert all(p == prev_placement[1] for p in placement)

if __name__ == "__main__":
    # test_simple_dominant_mapping()
    # test_many_to_one_clustering()
    # test_cross_device_preference()
    test_noisy_dominance()
    # test_extreme_hotspot()

    print("All tests passed!")