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
    """Compute ROC AUC One-vs-Rest."""
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
    cm = confusion_matrix(y_true, y_pred)

    return ConfusionMatrixDisplay(confusion_matrix=cm)
