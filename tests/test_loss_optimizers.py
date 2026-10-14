import torch
from torch import nn

from src.components.loss_functions import get_criterion
from src.components.optimizers import get_optimizer, get_scheduler


def test_get_criterion() -> None:
    criterion = get_criterion()
    assert isinstance(criterion, nn.CrossEntropyLoss)

    logits = torch.tensor([[2.0, 1.0, 0.1]])
    target = torch.tensor([0])
    loss = criterion(logits, target)
    assert loss.item() > 0


def test_get_optimizer() -> None:
    dummy_layer = nn.Linear(10, 2)
    optimizer = get_optimizer(
        dummy_layer.parameters(), learning_rate=0.005, weight_decay=0.01
    )

    assert isinstance(optimizer, torch.optim.AdamW)
    assert len(optimizer.param_groups[0]["params"]) == 2
    assert optimizer.param_groups[0]["lr"] == 0.005
    assert optimizer.param_groups[0]["weight_decay"] == 0.01


def test_get_scheduler_cosine() -> None:
    dummy_layer = nn.Linear(10, 2)
    optimizer = get_optimizer(dummy_layer.parameters(), learning_rate=0.01)
    scheduler = get_scheduler(optimizer, scheduler_type="cosine", epochs=5, min_lr=1e-5)

    assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
    optimizer.step()
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] < 0.01


def test_get_scheduler_cosine_with_warmup() -> None:
    dummy_layer = nn.Linear(10, 2)
    optimizer = get_optimizer(dummy_layer.parameters(), learning_rate=0.01)
    scheduler = get_scheduler(
        optimizer, scheduler_type="cosine", epochs=10, warmup_epochs=2, min_lr=1e-5
    )

    assert isinstance(scheduler, torch.optim.lr_scheduler.SequentialLR)


def test_get_scheduler_plateau() -> None:
    dummy_layer = nn.Linear(10, 2)
    optimizer = get_optimizer(dummy_layer.parameters(), learning_rate=0.01)
    scheduler = get_scheduler(
        optimizer, scheduler_type="plateau", patience=1, factor=0.5, min_lr=1e-4
    )

    assert isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
    scheduler.step(1.0)
    scheduler.step(1.0)
    scheduler.step(1.0)
    assert optimizer.param_groups[0]["lr"] < 0.01


def test_get_scheduler_none() -> None:
    dummy_layer = nn.Linear(10, 2)
    optimizer = get_optimizer(dummy_layer.parameters())
    scheduler = get_scheduler(optimizer, scheduler_type="none")
    assert scheduler is None


def test_get_scheduler_invalid() -> None:
    import pytest

    dummy_layer = nn.Linear(10, 2)
    optimizer = get_optimizer(dummy_layer.parameters())
    with pytest.raises(ValueError, match="Unsupported scheduler type"):
        get_scheduler(optimizer, scheduler_type="invalid_type")
