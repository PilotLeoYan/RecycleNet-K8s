"""Training and validation step executor for PyTorch models."""

import numpy as np
import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from src.components.metrics import get_loss_metrics, get_metrics


class TrainStep:
    """Execute training and validation mini-batch iterations.

    Attributes:
        model: PyTorch neural network model.
        train_loader: DataLoader providing training batches.
        val_loader: DataLoader providing validation batches.
        criterion: Loss function criterion module.
        optimizer: Optimizer instance for parameter updates.
        device: Computation device (e.g., 'cuda' or 'cpu').
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: torch.nn.Module,
        optimizer: Optimizer,
        device: torch.device | str,
    ) -> None:
        """Initialize TrainStep with execution dependencies and compute device.

        Args:
            model: PyTorch neural network model to train and evaluate.
            train_loader: DataLoader providing training samples.
            val_loader: DataLoader providing validation samples.
            criterion: Loss function module (e.g., CrossEntropyLoss).
            optimizer: Optimizer instance (e.g., AdamW).
            device: Computation device or device identifier string.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = torch.device(device) if isinstance(device, str) else device

    def _train_step(self) -> float:
        """Execute a single training epoch across all mini-batches.

        Returns:
            float: Average training loss over all training samples.
        """
        self.model.train()
        running_loss = 0.0
        total_samples = 0

        for batch_x, batch_y in self.train_loader:
            batch_x = batch_x.to(self.device, non_blocking=True)
            batch_y = batch_y.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            y_pred = self.model(batch_x)
            loss = self.criterion(y_pred, batch_y)
            loss.backward()
            self.optimizer.step()

            batch_size = batch_y.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size

        if total_samples == 0:
            return 0.0
        return running_loss / total_samples

    @torch.inference_mode()
    def _valid_step(self) -> tuple[float, dict[str, float]]:
        """Execute a validation evaluation pass across all mini-batches.

        Returns:
            tuple[float, dict[str, float]]: Average validation loss and computed
                metrics dictionary containing accuracy, precision, recall, and f1_score.
        """
        self.model.eval()
        running_vloss = 0.0
        total_samples = 0

        batch_predictions: list[np.ndarray] = []
        batch_labels: list[np.ndarray] = []

        for vbatch_x, vbatch_y in self.val_loader:
            vbatch_x = vbatch_x.to(self.device, non_blocking=True)
            vbatch_y = vbatch_y.to(self.device, non_blocking=True)

            vy_pred = self.model(vbatch_x)
            vloss = self.criterion(vy_pred, vbatch_y)

            batch_size = vbatch_y.size(0)
            running_vloss += vloss.item() * batch_size
            total_samples += batch_size

            preds = torch.argmax(vy_pred, dim=1)
            batch_predictions.extend(preds.detach().cpu().numpy())
            batch_labels.extend(vbatch_y.cpu().numpy())

        if total_samples == 0:
            return 0.0, get_metrics(np.array([]), np.array([]))

        avg_loss_v = running_vloss / total_samples
        metrics = get_metrics(np.array(batch_labels), np.array(batch_predictions))
        return avg_loss_v, metrics

    def train_valid_step(self) -> dict[str, float]:
        """Run a combined training epoch and validation evaluation step.

        Returns:
            dict[str, float]: Dictionary containing train_loss, valid_loss,
                valid_accuracy, valid_precision, valid_recall, and valid_f1_score.
        """
        loss = self._train_step()
        vloss, vmetrics = self._valid_step()
        return get_loss_metrics(loss, vloss, vmetrics)
