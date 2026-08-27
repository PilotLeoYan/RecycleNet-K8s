"""Reproducibility utilities for deterministic random state configuration."""

import os
import random

import numpy as np
import torch

from src.config.schema import ReproducibilityConfig
from src.utils import get_logger

logger = get_logger(__name__)


def make_reproducibility(config: ReproducibilityConfig) -> None:
    """Sets random seeds across Python, NumPy, PyTorch, and CUDA backends.

    Enforces deterministic cuDNN execution algorithms and disables cuDNN benchmarking
    to guarantee experiment reproducibility across multiple runs.

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
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    logger.info("Seeds set successfully.")
