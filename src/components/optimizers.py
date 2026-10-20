"""Optimizer and learning rate scheduler factory module."""

from collections.abc import Iterable

import torch
from torch.nn.parameter import Parameter
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    LRScheduler,
    ReduceLROnPlateau,
    SequentialLR,
)


def get_optimizer(
    params: Iterable[Parameter],
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
) -> Optimizer:
    """Instantiate and return the default optimizer for model parameters.

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


def get_scheduler(
    optimizer: Optimizer,
    scheduler_type: str = "cosine",
    epochs: int = 10,
    min_lr: float = 1e-5,
    warmup_epochs: int = 0,
    patience: int = 2,
    factor: float = 0.5,
) -> LRScheduler | ReduceLROnPlateau | None:
    """Instantiate and return the configured learning rate scheduler.

    Args:
        optimizer: Optimizer instance whose learning rate will be scheduled.
        scheduler_type: Type of scheduler ('cosine', 'plateau', or 'none').
        epochs: Total number of training epochs (used for CosineAnnealingLR).
        min_lr: Minimum learning rate threshold.
        warmup_epochs: Number of linear warmup epochs (applicable to cosine).
        patience: Epoch patience for ReduceLROnPlateau before decaying LR.
        factor: Decay multiplicative factor for ReduceLROnPlateau.

    Returns:
        Configured LRScheduler, ReduceLROnPlateau, or None if disabled.

    Raises:
        ValueError: If an unsupported scheduler_type is provided.
    """
    scheduler_type_norm = (scheduler_type or "").strip().lower()

    if scheduler_type_norm in ("none", ""):
        return None

    if scheduler_type_norm == "cosine":
        if warmup_epochs > 0 and epochs > warmup_epochs:
            warmup = LinearLR(
                optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=warmup_epochs,
            )
            cosine = CosineAnnealingLR(
                optimizer,
                T_max=epochs - warmup_epochs,
                eta_min=min_lr,
            )
            return SequentialLR(
                optimizer,
                schedulers=[warmup, cosine],
                milestones=[warmup_epochs],
            )
        return CosineAnnealingLR(
            optimizer,
            T_max=max(epochs, 1),
            eta_min=min_lr,
        )

    if scheduler_type_norm == "plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=factor,
            patience=patience,
            min_lr=min_lr,
        )

    raise ValueError(
        f"Unsupported scheduler type '{scheduler_type}'. "
        "Allowed options are: 'cosine', 'plateau', 'none'."
    )
