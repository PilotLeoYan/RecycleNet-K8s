"""Unit tests for distributed test evaluation component TestDistributed."""

import numpy as np
import torch
from matplotlib.figure import Figure
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.components.evals import TestDistributed


def test_test_distributed_step() -> None:
    """Test TestDistributed._test_step collects probabilities, targets, and metrics."""
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * 8 * 8, 2),
    )
    x = torch.randn(6, 3, 8, 8)
    y = torch.tensor([0, 1, 0, 1, 0, 1])
    dataset = TensorDataset(x, y)
    test_loader = DataLoader(dataset, batch_size=2)
    criterion = nn.CrossEntropyLoss()

    evaluator = TestDistributed(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        num_classes=2,
        classes_name=["cardboard", "glass"],
        device="cpu",
    )

    probas, targets, metrics = evaluator._test_step()

    assert isinstance(probas, np.ndarray)
    assert isinstance(targets, np.ndarray)
    assert len(probas) == 6
    assert probas.shape == (6, 2)
    assert len(targets) == 6
    assert "test_loss" in metrics
    assert "test_accuracy" in metrics
    assert "confusion_matrix" in metrics
    assert isinstance(metrics["confusion_matrix"], Figure)


def test_test_distributed_compute_metrics() -> None:
    """Test compute_distributed_test_metrics returns full dictionary including AUC."""
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * 8 * 8, 2),
    )
    x = torch.randn(6, 3, 8, 8)
    y = torch.tensor([0, 1, 0, 1, 0, 1])
    dataset = TensorDataset(x, y)
    test_loader = DataLoader(dataset, batch_size=2)
    criterion = nn.CrossEntropyLoss()

    evaluator = TestDistributed(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        num_classes=2,
        classes_name=["cardboard", "glass"],
        device="cpu",
    )

    metrics = evaluator.compute_distributed_test_metrics()

    assert "auc" in metrics
    assert "test_loss" in metrics
    assert "test_accuracy" in metrics
    assert "test_precision" in metrics
    assert "test_recall" in metrics
    assert "test_f1_score" in metrics
    assert "confusion_matrix" in metrics
    assert 0.0 <= metrics["test_accuracy"] <= 1.0
    assert 0.0 <= metrics["auc"] <= 1.0


def test_test_distributed_local_metrics_edge_cases() -> None:
    """Test _local_metrics with zero samples."""
    empty_ds = TensorDataset(
        torch.empty(0, 4),
        torch.empty(0, dtype=torch.int64),
    )
    evaluator = TestDistributed(
        model=nn.Linear(4, 2),
        test_loader=DataLoader(empty_ds),
        criterion=nn.CrossEntropyLoss(),
        num_classes=2,
        classes_name=["a", "b"],
        device="cpu",
    )

    total_samples = torch.zeros(1, dtype=torch.int64)
    loss_sum = torch.zeros(1, dtype=torch.float64)
    conf_matrix = torch.zeros((2, 2), dtype=torch.int64)

    metrics = evaluator._local_metrics(total_samples, loss_sum, conf_matrix)
    assert metrics["test_loss"] == 0.0
    assert metrics["test_accuracy"] == 0.0
    assert isinstance(metrics["confusion_matrix"], Figure)
