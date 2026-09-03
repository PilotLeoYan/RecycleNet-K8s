"""Optimizer factory module for model parameter optimization."""

from collections.abc import Iterable

import torch
from torch.nn.parameter import Parameter
from torch.optim import Optimizer


def get_optimizer(
    params: Iterable[Parameter],
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
) -> Optimizer:
    """Instantiates and returns the default optimizer for model parameters.

    Args:
        params: Iterable of trainable model parameters.
        learning_rate: Learning rate for gradient descent updates.
        weight_decay: L2 penalty factor for regularization.

    Returns:
        Optimizer: AdamW optimizer configured with the specified hyperparameters.
    """
    return torch.optim.AdamW(
        params=params,
        lr=learning_rate,
        weight_decay=weight_decay,
    )
