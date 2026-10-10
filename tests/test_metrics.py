"""Unit tests for multi-class classification metric computation utilities."""

import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

from src.components.metrics import (
    calculate_roc_auc,
    confusion,
    confusion_matrix_display,
    get_loss_metrics,
    get_metrics,
)


def test_get_metrics_perfect_score() -> None:
    """Test get_metrics returns 1.0 across all metrics when predictions match labels."""
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 1, 2])

    metrics = get_metrics(y_true, y_pred)

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1_score"] == 1.0


def test_get_metrics_imperfect_score() -> None:
    """Test get_metrics with partially correct predictions."""
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])

    metrics = get_metrics(y_true, y_pred)

    assert metrics["accuracy"] == 0.75
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["f1_score"] <= 1.0


def test_get_metrics_zero_division() -> None:
    """Test get_metrics handles zero division when a class is never predicted."""
    y_true = np.array([0, 1, 2])
    y_pred = np.array([0, 0, 0])

    metrics = get_metrics(y_true, y_pred)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["f1_score"] <= 1.0


def test_get_loss_metrics() -> None:
    """Test get_loss_metrics consolidates train loss, validation loss, and metrics."""
    vmetrics = {
        "accuracy": 0.85,
        "f1_score": 0.80,
    }
    result = get_loss_metrics(loss=0.35, vloss=0.42, vmetrics=vmetrics)

    assert result["train_loss"] == 0.35
    assert result["valid_loss"] == 0.42
    assert result["valid_accuracy"] == 0.85
    assert result["valid_f1_score"] == 0.80


def test_calculate_roc_auc_valid() -> None:
    """Test calculate_roc_auc calculates correct OvR ROC AUC score."""
    y_true = np.array([0, 1, 2])
    y_score = np.array(
        [
            [0.9, 0.05, 0.05],
            [0.1, 0.8, 0.1],
            [0.05, 0.05, 0.9],
        ]
    )

    roc = calculate_roc_auc(y_true, y_score)
    assert 0.0 <= roc <= 1.0
    assert roc == 1.0


def test_calculate_roc_auc_error_handling() -> None:
    """Test calculate_roc_auc returns 0.0 when shapes or classes are invalid."""
    y_true = np.array([0, 1])
    y_score = np.array([[0.5]])

    roc = calculate_roc_auc(y_true, y_score)
    assert roc == 0.0


def test_confusion() -> None:
    """Test confusion function generates a valid ConfusionMatrixDisplay instance."""
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])

    cm_disp = confusion(y_true, y_pred)
    assert isinstance(cm_disp, ConfusionMatrixDisplay)
    assert cm_disp.confusion_matrix.shape == (2, 2)


def test_confusion_matrix_display() -> None:
    """Test confusion_matrix_display constructs display from given matrix and labels."""
    cm = np.array([[5, 1], [2, 4]])
    display = confusion_matrix_display(cm, display_labels=["cat", "dog"])

    assert isinstance(display, ConfusionMatrixDisplay)
    assert display.display_labels == ["cat", "dog"]


def test_get_metrics_empty() -> None:
    """Test get_metrics returns 0.0 values when given empty arrays."""
    metrics = get_metrics(np.array([]), np.array([]))
    assert metrics["accuracy"] == 0.0
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1_score"] == 0.0


def test_calculate_roc_auc_empty() -> None:
    """Test calculate_roc_auc returns 0.0 when given empty arrays or 1D score."""
    roc = calculate_roc_auc(np.array([]), np.empty((0, 2)))
    assert roc == 0.0

    roc_1d = calculate_roc_auc(np.array([0, 1]), np.array([0.2, 0.8]))
    assert roc_1d == 0.0
