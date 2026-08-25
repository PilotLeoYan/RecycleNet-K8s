"""Optimizer factory module for model parameter optimization."""

from collections.abc import Iterator

import torch
from torch.nn.parameter import Parameter
from torch.optim import Optimizer


def get_optimizer(params: Iterator[Parameter]) -> Optimizer:
    """Instantiates and returns the default optimizer for model parameters.

    Args:
        params: Iterable of trainable model parameters.

    Returns:
        Optimizer: AdamW optimizer configured for gradient updates.
    """
    return torch.optim.AdamW(params=params)
