import torch.distributed as dist

def is_dist_ready() -> bool:
    return dist.is_available() and dist.is_initialized()

def get_tp_rank() -> int:
    return dist.get_rank() if is_dist_ready() else 0

def get_tp_world_size() -> int:
    return dist.get_world_size() if is_dist_ready() else 1