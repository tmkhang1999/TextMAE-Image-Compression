"""Small helpers that make single-process runs and DDP runs look the same."""
import torch
import torch.distributed as dist


def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    return dist.get_rank() if is_dist_avail_and_initialized() else 0


def get_world_size():
    return dist.get_world_size() if is_dist_avail_and_initialized() else 1


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    """torch.save, but only from rank 0 so workers do not overwrite each other."""
    if is_main_process():
        torch.save(*args, **kwargs)


def all_reduce_mean(x):
    """Average a python scalar over all processes (no-op for one process)."""
    world_size = get_world_size()
    if world_size <= 1:
        return x
    x_reduce = torch.tensor(x).cuda()
    dist.all_reduce(x_reduce)
    x_reduce /= world_size
    return x_reduce.item()
