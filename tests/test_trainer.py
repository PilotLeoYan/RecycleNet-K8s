from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.components.trainer import ModelTrainer


def test_trainer_train_step() -> None:
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * 16 * 16, 2),
    )

    x = torch.randn(8, 3, 16, 16)
    y = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    dataset = TensorDataset(x, y)
    train_loader = DataLoader(dataset, batch_size=4)
    val_loader = DataLoader(dataset, batch_size=4)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device="cpu",
    )

    loss = trainer._train_step()
    assert isinstance(loss, float)
    assert loss > 0.0


def test_trainer_valid_step() -> None:
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * 16 * 16, 2),
    )

    x = torch.randn(8, 3, 16, 16)
    y = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    dataset = TensorDataset(x, y)
    train_loader = DataLoader(dataset, batch_size=4)
    val_loader = DataLoader(dataset, batch_size=4)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device="cpu",
    )

    vloss, metrics = trainer._valid_step()
    assert isinstance(vloss, float)
    assert "accuracy" in metrics
    assert "f1_score" in metrics


def test_trainer_fit(tmp_path: Path) -> None:
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * 16 * 16, 2),
    )

    x = torch.randn(8, 3, 16, 16)
    y = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    dataset = TensorDataset(x, y)
    train_loader = DataLoader(dataset, batch_size=4)
    val_loader = DataLoader(dataset, batch_size=4)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device="cpu",
    )

    weights_dir = tmp_path / "weights"
    saved_path = trainer.fit(0, epochs=2, weights_path=str(weights_dir), patience=2)

    assert saved_path.exists()


def test_trainer_with_logmodel(tmp_path: Path) -> None:
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * 16 * 16, 2),
    )

    x = torch.randn(8, 3, 16, 16)
    y = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    dataset = TensorDataset(x, y)
    train_loader = DataLoader(dataset, batch_size=4)
    val_loader = DataLoader(dataset, batch_size=4)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    logged_epochs: list[dict[str, object]] = []

    class MockLogModel:
        def log_epoch(
            self,
            train_loss: float,
            valid_loss: float,
            valid_metrics: dict[str, float],
            step: int,
        ) -> None:
            logged_epochs.append(
                {
                    "train_loss": train_loss,
                    "valid_loss": valid_loss,
                    "valid_metrics": valid_metrics,
                    "step": step,
                }
            )

    mock_logmodel = MockLogModel()

    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device="cpu",
        logmodel=mock_logmodel,  # type: ignore
    )

    assert trainer.logmodel is not None

    weights_dir = tmp_path / "weights"
    saved_path = trainer.fit(0, epochs=2, weights_path=str(weights_dir), patience=2)

    assert saved_path.exists()
    assert len(logged_epochs) == 2
    assert logged_epochs[0]["step"] == 0
    assert logged_epochs[1]["step"] == 1
    valid_metrics = logged_epochs[0]["valid_metrics"]
    assert isinstance(valid_metrics, dict)
    assert "accuracy" in valid_metrics
