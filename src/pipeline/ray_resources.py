from torch.cuda import is_available

from src.utils import get_logger

logger = get_logger(__name__)


def assign_ray_resources(
    requested_device: str,
    cpu_resources: float,
    gpu_resources: float,
) -> tuple[str, dict[str, float]]:
    """Dynamically resolve Ray resource allocations based on hardware availability.

    Args:
        requested_device: User-configured device preference ('auto', 'cuda', or 'cpu').
        cpu_resources: Number of CPU cores requested per trial.
        gpu_resources: Fractional or integer GPUs requested per trial.

    Returns:
        tuple[str, dict[str, float]]: Effective compute device ('cuda' or 'cpu') and
            sanitized resource mapping for Ray Tune.
    """
    has_cuda = is_available()
    requested_device = requested_device.lower()

    if requested_device == "auto":
        effective_device = "cuda" if has_cuda else "cpu"
    elif requested_device == "cuda" and not has_cuda:
        logger.warning(
            "CUDA requested if config but not available. Falling back to CPU."
        )
        effective_device = "cpu"
    else:
        effective_device = requested_device

    use_gpu = effective_device == "cuda"
    gpu_res = gpu_resources if use_gpu else 0.0

    return effective_device, {"cpu": cpu_resources, "gpu": gpu_res}
