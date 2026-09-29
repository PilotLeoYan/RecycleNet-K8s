"""Reproducibility utilities for deterministic random state configuration."""

import os
import random

import numpy as np
import torch

from src.components.data_transform import seed_worker
from src.config.schema import ReproducibilityConfig
from src.utils import get_logger

logger = get_logger(__name__)

__all__ = ["make_reproducibility", "seed_worker"]


def make_reproducibility(config: ReproducibilityConfig) -> None:
    """Set random seeds across Python, NumPy, PyTorch, and CUDA backends.

    Args:
        config: Configuration containing random seed values.
    """
    os.environ["PYTHONHASHSEED"] = str(config.random_seed)
    random.seed(config.random_seed)
    np.random.seed(config.numpy_seed)
    torch.manual_seed(config.torch_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.torch_seed)
        torch.cuda.manual_seed_all(config.torch_seed)

        if config.deterministic:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True

    if config.deterministic:
        torch.use_deterministic_algorithms(True, warn_only=config.warn_only)
        logger.info("Strict deterministic mode activated (bitwise reproducibility).")
    else:
        torch.use_deterministic_algorithms(False)
        logger.info(
            "High-throughput training mode activated (statistical reproducibility)."
        )
