# deepspeed/moe/registry.py

import torch.distributed as dist

class ExpertRegistry:
    def __init__(self, num_experts):
        world_size = dist.get_world_size()
        self.expert_to_rank = {
            i: i % world_size for i in range(num_experts)
        }

    def owner(self, expert_id):
        return self.expert_to_rank[expert_id]

    def move_expert(self, expert_id, new_rank):
        self.expert_to_rank[expert_id] = new_rank

    def state_dict(self):
        return self.expert_to_rank

    def load_state_dict(self, state):
        self.expert_to_rank = state
