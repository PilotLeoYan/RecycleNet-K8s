"""Model evaluation component for assessing performance on the test dataset split."""

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from src.components.log_model import LogModel
from src.components.metrics import calculate_roc_auc, confusion, evals
from src.utils import get_logger

logger = get_logger(__name__)

CMAP = "Blues"


class Evaluator:
    """Evaluate a trained model checkpoint against the test dataset and log results.

    Attributes:
        test_loader: DataLoader providing the test data partition.
        device: Computation device used for inference.
        logmodel: Optional MLflow logging helper.
    """

    def __init__(self,
        test_loader: DataLoader,
        device: torch.device | str,
        logmodel: LogModel | None = None,
    ):
        """Initialize Evaluator with test DataLoader and target device.

        Args:
            test_loader: DataLoader for the test partition.
            device: Target torch device or device string ('cuda', 'cpu').
            logmodel: Optional MLflow logging helper.
        """
        self.test_loader = test_loader
        self.device = torch.device(device) if isinstance(device, str) else device
        self.logmodel = logmodel

    @torch.inference_mode()
    def _test_model(
        self,
        model: torch.nn.Module,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run batch inference on the test dataset to collect predictions.

        Args:
            model: PyTorch model to evaluate.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: Arrays of predicted class
                labels, predicted probabilities, and ground truth labels.
        """
        model.to(self.device)
        model.eval()

        batchs_predictions: list[np.ndarray] = []
        batchs_probas: list[np.ndarray] = []
        batchs_labels: list[np.ndarray] = []

        for batch_x, batch_y in self.test_loader:
            batch_x = batch_x.to(self.device, non_blocking=True)
            batch_y = batch_y.to(self.device, non_blocking=True)

            logits = model(batch_x)

            # convert logits to predictions
            batch_predictions = torch.argmax(logits, dim=1)

            # convert logits to probabilities
            batch_probas = torch.softmax(logits, dim=1)

            batchs_predictions.extend(batch_predictions.detach().cpu().numpy())
            batchs_probas.extend(batch_probas.detach().cpu().numpy())
            batchs_labels.extend(batch_y.cpu().numpy())

        predics = np.array(batchs_predictions)
        probas = np.array(batchs_probas)
        labels = np.array(batchs_labels)

        return predics, probas, labels

    def evaluate(
        self,
        model: torch.nn.Module,
    ) -> dict[str, float]:
        """Compute test metrics, generate confusion matrix, and log to MLflow.

        Args:
            model: PyTorch model to evaluate.

        Returns:
            dict[str, float]: Evaluation metrics dictionary including ROC-AUC score.
        """
        predictions, probas, labels = self._test_model(model)

        metrics = evals(labels, predictions)
        roc = calculate_roc_auc(labels, probas)
        all_metrics = {**metrics, "roc_auc": roc}

        logger.info(
            "Test evaluation results - Accuracy: %.4f, Precision: %.4f, Recall: %.4f, F1: %.4f, ROC-AUC: %.4f",
            metrics["accuracy"],
            metrics["precision"],
            metrics["recall"],
            metrics["f1_score"],
            roc,
        )

        cm_disp = confusion(labels, predictions)
        fig_cm = cm_disp.plot(cmap=CMAP).figure_

        try:
            if self.logmodel is not None:
                self.logmodel.log_test(roc, metrics, fig_cm)
        finally:
            plt.close(fig_cm)

        return all_metrics
