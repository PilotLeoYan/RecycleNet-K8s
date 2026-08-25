"""Model evaluation component for assessing performance on the test dataset split."""

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.components.log_model import LogModel
from src.components.metrics import calculate_roc_auc, confusion, evals

CMAP = "Blues"


class Evaluator:
    """Evaluates a trained model checkpoint against the test dataset and logs results.

    Attributes:
        model: PyTorch model architecture.
        weights_path: Path to the trained model weights checkpoint (.pth).
        test_loader: DataLoader providing the test data partition.
        device: Computation device used for inference.
        log_model: MLflow logging helper.
    """

    def __init__(
        self,
        model: nn.Module,
        weights_path: Path | str,
        test_loader: DataLoader,
        device: torch.device | str,
    ):
        """Initializes Evaluator with model, weights, DataLoader, and device.

        Args:
            model: PyTorch model to evaluate.
            weights_path: Path or string to the checkpoint .pth file.
            test_loader: DataLoader for the test partition.
            device: Target torch device or device string ('cuda', 'cpu').
        """
        self.model = model
        self.weights_path = (
            Path(weights_path) if isinstance(weights_path, str) else weights_path
        )
        self.test_loader = test_loader
        self.device = torch.device(device) if isinstance(device, str) else device

        self.log_model: LogModel = LogModel()
        # move model to the same device
        self.model.to(self.device)

    @torch.inference_mode()
    def _test_model(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Runs batch inference on the test dataset to collect predictions.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: Arrays of predicted class
                labels, predicted probabilities, and ground truth labels.
        """
        self.model.load_state_dict(
            torch.load(self.weights_path, map_location=self.device)
        )
        self.model.eval()

        batchs_predictions: list[np.ndarray] = []
        batchs_probas: list[np.ndarray] = []
        batchs_labels: list[np.ndarray] = []

        for batch_x, batch_y in self.test_loader:
            batch_x = batch_x.to(self.device, non_blocking=True)
            batch_y = batch_y.to(self.device, non_blocking=True)

            logits = self.model(batch_x)

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

    def evaluate(self) -> None:
        """Computes test metrics, generates confusion matrix, and logs to MLflow."""
        predictions, probas, labels = self._test_model()

        metrics = evals(labels, predictions)
        roc = calculate_roc_auc(labels, probas)

        cm_disp = confusion(labels, predictions)
        fig_cm = cm_disp.plot(cmap=CMAP).figure_

        self.log_model.log_test(roc, metrics, fig_cm)
