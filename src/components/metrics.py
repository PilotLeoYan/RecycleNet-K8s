"""Evaluation metrics computation for multi-class classification."""

import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

AVERAGE = "macro"
MULTI_CLASS = "ovr"


def evals(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Computes standard multi-class evaluation metrics.

    Calculates accuracy, macro-averaged precision, recall, and F1-score.

    Args:
        y_true: 1D array of ground truth class integer labels.
        y_pred: 1D array of predicted class integer labels.

    Returns:
        dict[str, float]: Dictionary containing accuracy, precision, recall,
            and f1_score.
    """
    metrics: dict[str, float] = dict()

    # 1. Accuracy
    metrics["accuracy"] = accuracy_score(y_true, y_pred, normalize=True)

    # 2. Precision
    metrics["precision"] = precision_score(
        y_true, y_pred, average=AVERAGE, zero_division=0
    )

    # 3. Recall
    metrics["recall"] = recall_score(y_true, y_pred, average=AVERAGE, zero_division=0)

    # 4. F1-Score
    metrics["f1_score"] = f1_score(y_true, y_pred, average=AVERAGE, zero_division=0)

    return metrics


def calculate_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Computes macro-averaged One-vs-Rest (OvR) ROC AUC score.

    Args:
        y_true: 1D array of ground truth class labels.
        y_score: 2D array of predicted probabilities with shape (n_samples, n_classes).

    Returns:
        float: Computed ROC AUC score, or 0.0 if calculation fails.
    """
    try:
        num_classes = y_score.shape[1]
        return float(
            roc_auc_score(
                y_true,
                y_score,
                average=AVERAGE,
                multi_class=MULTI_CLASS,
                labels=np.arange(num_classes),
            )
        )
    except ValueError, IndexError:
        return 0.0


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> ConfusionMatrixDisplay:
    """Generates a ConfusionMatrixDisplay object from true and predicted labels.

    Args:
        y_true: 1D array of ground truth class labels.
        y_pred: 1D array of predicted class labels.

    Returns:
        ConfusionMatrixDisplay: Scikit-learn confusion matrix display container.
    """
    cm = confusion_matrix(y_true, y_pred)

    return ConfusionMatrixDisplay(confusion_matrix=cm)
