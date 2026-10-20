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
    assert metrics["test_f1_score"] == 0.0
    assert "per_class_metrics" in metrics
    assert metrics["per_class_metrics"]["a"]["support"] == 0
    assert isinstance(metrics["confusion_matrix"], Figure)


def test_test_distributed_macro_f1_and_per_class_metrics() -> None:
    """Test true Macro-F1 calculation vs harmonic mean and per-class metrics."""
    empty_ds = TensorDataset(
        torch.empty(0, 4),
        torch.empty(0, dtype=torch.int64),
    )
    evaluator = TestDistributed(
        model=nn.Linear(4, 2),
        test_loader=DataLoader(empty_ds),
        criterion=nn.CrossEntropyLoss(),
        num_classes=2,
        classes_name=["cardboard", "glass"],
        device="cpu",
    )

    # Asymmetric confusion matrix where mean(per_class_f1) != harmonic_mean(P, R)
    # Class 0: TP=1, FP=9, FN=0 -> P0=0.1, R0=1.0, F1_0 = 2*1/(10+1) = 2/11
    # Class 1: TP=1, FP=0, FN=9 -> P1=1.0, R1=0.1, F1_1 = 2*1/(1+10) = 2/11
    # True Macro-F1 = 2/11 ≈ 0.181818...
    # Harmonic mean of Macro-P (0.55) and Macro-R (0.55) would be 0.55!
    conf_matrix = torch.tensor([[1, 0], [9, 1]], dtype=torch.int64)
    total_samples = torch.tensor([11], dtype=torch.int64)
    loss_sum = torch.tensor([2.2], dtype=torch.float64)

    metrics = evaluator._local_metrics(total_samples, loss_sum, conf_matrix)

    expected_f1 = 2.0 / 11.0
    assert np.isclose(metrics["test_f1_score"], expected_f1)
    assert not np.isclose(metrics["test_f1_score"], 0.55)

    per_class = metrics["per_class_metrics"]
    assert set(per_class.keys()) == {"cardboard", "glass"}
    assert np.isclose(per_class["cardboard"]["precision"], 0.1)
    assert np.isclose(per_class["cardboard"]["recall"], 1.0)
    assert np.isclose(per_class["cardboard"]["f1_score"], expected_f1)
    assert per_class["cardboard"]["support"] == 1
    assert np.isclose(per_class["glass"]["precision"], 1.0)
    assert np.isclose(per_class["glass"]["recall"], 0.1)
    assert np.isclose(per_class["glass"]["f1_score"], expected_f1)
    assert per_class["glass"]["support"] == 10
