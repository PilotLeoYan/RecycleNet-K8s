"""Reproducibility utilities for deterministic random state configuration."""

import os
import random
from dataclasses import dataclass

import numpy as np
import torch

from src.utils import get_logger

logger = get_logger(__name__)


@dataclass
class ReproducibilityConfig:
    """Random seed configuration settings for environment reproducibility.

    Attributes:
        random_seed: Seed for standard Python random and hashseed generation.
        numpy_seed: Seed for NumPy pseudorandom number generators.
        torch_seed: Seed for PyTorch CPU and CUDA random number generators.
    """

    random_seed: int = 42
    numpy_seed: int = 42
    torch_seed: int = 42


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
