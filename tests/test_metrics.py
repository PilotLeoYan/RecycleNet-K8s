import numpy as np

from src.components.metrics import calculate_roc_auc, confusion, evals


def test_evals_perfect_score() -> None:
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 1, 2])

    metrics = evals(y_true, y_pred)

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1_score"] == 1.0


def test_evals_imperfect_score() -> None:
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])

    metrics = evals(y_true, y_pred)

    assert metrics["accuracy"] == 0.75
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["f1_score"] <= 1.0


def test_calculate_roc_auc() -> None:
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
    # Invalid dimension / mismatched shape should return 0.0 without crashing
    y_true = np.array([0, 1])
    y_score = np.array([[0.5]])

    roc = calculate_roc_auc(y_true, y_score)
    assert roc == 0.0


def test_confusion_matrix_display() -> None:
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])

    cm_disp = confusion(y_true, y_pred)
    assert cm_disp.confusion_matrix is not None
    assert cm_disp.confusion_matrix.shape == (2, 2)
