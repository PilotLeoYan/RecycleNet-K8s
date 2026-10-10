"""Unit tests for TrainStep component."""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.components.train_step import TrainStep


def test_train_step_execution() -> None:
    """Test TrainStep._train_step executes forward and backward passes correctly."""
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * 8 * 8, 2),
    )
    x = torch.randn(8, 3, 8, 8)
    y = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    dataset = TensorDataset(x, y)
    train_loader = DataLoader(dataset, batch_size=4)
    val_loader = DataLoader(dataset, batch_size=4)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    step_runner = TrainStep(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device="cpu",
    )

    loss = step_runner._train_step()
    assert isinstance(loss, float)
    assert loss > 0.0


def test_valid_step_execution() -> None:
    """Test TrainStep._valid_step calculates validation loss and metrics."""
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * 8 * 8, 2),
    )
    x = torch.randn(8, 3, 8, 8)
    y = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    dataset = TensorDataset(x, y)
    train_loader = DataLoader(dataset, batch_size=4)
    val_loader = DataLoader(dataset, batch_size=4)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    step_runner = TrainStep(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device="cpu",
    )

    vloss, vmetrics = step_runner._valid_step()
    assert isinstance(vloss, float)
    assert "accuracy" in vmetrics
    assert "precision" in vmetrics
    assert "recall" in vmetrics
    assert "f1_score" in vmetrics


def test_train_valid_step_combined() -> None:
    """Test TrainStep.train_valid_step returns combined training and val metrics."""
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * 8 * 8, 2),
    )
    x = torch.randn(8, 3, 8, 8)
    y = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    dataset = TensorDataset(x, y)
    train_loader = DataLoader(dataset, batch_size=4)
    val_loader = DataLoader(dataset, batch_size=4)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    step_runner = TrainStep(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device="cpu",
    )

    metrics = step_runner.train_valid_step()
    assert "train_loss" in metrics
    assert "valid_loss" in metrics
    assert "valid_accuracy" in metrics
    assert "valid_f1_score" in metrics
    assert isinstance(metrics["train_loss"], float)
    assert isinstance(metrics["valid_loss"], float)


def test_train_step_empty_loader() -> None:
    """Test TrainStep handles empty DataLoaders gracefully."""
    model = nn.Linear(4, 2)
    empty_dataset = TensorDataset(torch.empty(0, 4), torch.empty(0, dtype=torch.int64))
    empty_loader = DataLoader(empty_dataset, batch_size=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    step_runner = TrainStep(
        model=model,
        train_loader=empty_loader,
        val_loader=empty_loader,
        criterion=criterion,
        optimizer=optimizer,
        device="cpu",
    )

    loss = step_runner._train_step()
    assert loss == 0.0

    vloss, vmetrics = step_runner._valid_step()
    assert vloss == 0.0
    assert isinstance(vmetrics, dict)
