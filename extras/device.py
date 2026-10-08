import torch


def best_device():
    """CUDA, then Apple Silicon (MPS), then CPU. Half precision on GPUs, float32 on CPU."""
    if torch.cuda.is_available():
        return "cuda", torch.float16
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32
