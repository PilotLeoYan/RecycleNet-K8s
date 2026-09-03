import torch
from torch import nn

from src.components.loss_functions import get_criterion
from src.components.optimizers import get_optimizer


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
