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


def get_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute standard multi-class evaluation metrics.

    Calculates accuracy, macro-averaged precision, recall, and F1-score.

    Args:
        y_true: 1D array of ground truth class integer labels.
        y_pred: 1D array of predicted class integer labels.

    Returns:
        dict[str, float]: Dictionary containing accuracy, precision, recall,
            and f1_score.
    """
    if len(y_true) == 0:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
        }

    metrics: dict[str, float] = {}
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred, normalize=True))
    metrics["precision"] = float(
        precision_score(y_true, y_pred, average=AVERAGE, zero_division=0)
    )
    metrics["recall"] = float(
        recall_score(y_true, y_pred, average=AVERAGE, zero_division=0)
    )
    metrics["f1_score"] = float(
        f1_score(y_true, y_pred, average=AVERAGE, zero_division=0)
    )
    return metrics


def get_loss_metrics(
    loss: float,
    vloss: float,
    vmetrics: dict[str, float],
) -> dict[str, float]:
    """Combine training loss, validation loss, and validation metrics.

    Args:
        loss: Average training loss.
        vloss: Average validation loss.
        vmetrics: Dictionary of validation evaluation metrics.

    Returns:
        dict[str, float]: Consolidated dictionary containing train_loss, valid_loss,
            and prefixed validation metrics.
    """
    metrics = {f"valid_{k}": float(v) for k, v in vmetrics.items()}
    metrics["train_loss"] = float(loss)
    metrics["valid_loss"] = float(vloss)
    return metrics


def calculate_roc_auc(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> float:
    """Compute macro-averaged One-vs-Rest (OvR) ROC AUC score.

    Args:
        y_true: 1D array of ground truth class labels.
        y_score: 2D array of predicted probabilities with shape (n_samples, n_classes).

    Returns:
        float: Computed ROC AUC score, or 0.0 if calculation fails or is undefined.
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
    except (ValueError, IndexError):
        return 0.0


def confusion(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> ConfusionMatrixDisplay:
    """Generate a ConfusionMatrixDisplay object from true and predicted labels.

    Args:
        y_true: 1D array of ground truth class labels.
        y_pred: 1D array of predicted class labels.

    Returns:
        ConfusionMatrixDisplay: Scikit-learn confusion matrix display container.
    """
    cm = confusion_matrix(y_true, y_pred)
    return ConfusionMatrixDisplay(confusion_matrix=cm)


def confusion_matrix_display(
    cm: np.ndarray,
    display_labels: list[str] | None = None,
) -> ConfusionMatrixDisplay:
    """Create a ConfusionMatrixDisplay from an existing confusion matrix array.

    Args:
        cm: 2D confusion matrix array.
        display_labels: Optional list of class display names.

    Returns:
        ConfusionMatrixDisplay: Scikit-learn confusion matrix display container.
    """
    return ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
