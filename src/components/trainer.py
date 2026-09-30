"""Training loop execution, validation monitoring, early stopping, and checkpointing."""

import copy
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from src.components import LogModel
from src.components.metrics import evals
from src.utils import get_logger

logger = get_logger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent


class ModelTrainer:
    """Orchestrate model training, validation, early stopping, and tracking.

    Attributes:
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        criterion: Loss function module.
        optimizer: Optimization algorithm instance.
        device: Computation device (e.g., 'cuda' or 'cpu').
        logmodel: Optional MLflow logging helper.
    """

    def __init__(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        device: torch.device | str,
        logmodel: LogModel | None = None,
    ):
        """Initialize ModelTrainer with training dependencies and target device.

        Args:
            train_loader: DataLoader providing training mini-batches.
            val_loader: DataLoader providing validation mini-batches.
            criterion: Loss function module (e.g. CrossEntropyLoss).
            optimizer: Optimizer instance (e.g. AdamW).
            device: Target torch device or device string ('cuda', 'cpu').
            logmodel: Optional MLflow logging helper.
        """
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = torch.device(device) if isinstance(device, str) else device
        self.logmodel = logmodel

    def _train_step(self, model: torch.nn.Module) -> float:
        """Execute a single training epoch across all mini-batches in train_loader.

        Args:
            model: PyTorch neural network being trained.

        Returns:
            float: Average training loss across all samples in the epoch.
        """
        model.train()
        running_loss = 0.0

        for batch_x, batch_y in self.train_loader:
            batch_x = batch_x.to(self.device, non_blocking=True)
            batch_y = batch_y.to(self.device, non_blocking=True)

            y_pred = model(batch_x)
            loss = self.criterion(y_pred, batch_y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * batch_y.size(0)

        avg_loss = running_loss / len(self.train_loader.dataset)  # type: ignore
        return avg_loss  # type: ignore

    @torch.inference_mode()
    def _valid_step(self, model: torch.nn.Module) -> tuple[float, dict[str, float]]:
        """Execute a validation pass over val_loader computing loss and metrics.

        Args:
            model: PyTorch model being evaluated.

        Returns:
            tuple[float, dict[str, float]]: Average validation loss and computed
                metrics dictionary.
        """
        model.eval()
        running_vloss = 0.0

        batchs_predictions: list[np.ndarray] = []
        batchs_labels: list[np.ndarray] = []

        for vbatch_x, vbatch_y in self.val_loader:
            vbatch_x = vbatch_x.to(self.device, non_blocking=True)
            vbatch_y = vbatch_y.to(self.device, non_blocking=True)

            vy_pred = model(vbatch_x)
            vloss = self.criterion(vy_pred, vbatch_y)

            running_vloss += vloss.item() * vbatch_y.size(0)

            # convert logits to predictions
            predictions = torch.argmax(vy_pred, dim=1)
            batchs_predictions.extend(predictions.detach().cpu().numpy())
            batchs_labels.extend(vbatch_y.cpu().numpy())

        avg_loss_v = running_vloss / len(self.val_loader.dataset)  # type: ignore
        metrics = evals(np.array(batchs_labels), np.array(batchs_predictions))
        return avg_loss_v, metrics  # type: ignore

    def fit(
        self,
        model: torch.nn.Module,
        epochs: int,
        patience: int = 3,
    ) -> torch.nn.Module:
        """Run the complete training and validation cycle with early stopping.

        Args:
            model: PyTorch neural network to train.
            epochs: Maximum number of training epochs to execute.
            patience: Number of epochs without improvement before early stopping.

        Returns:
            torch.nn.Module: The trained model restored to its optimal weights.
        """
        best_loss = float("inf")
        epochs_no_improve = 0

        model.to(self.device)
        best_state = copy.deepcopy(model.state_dict())

        for epoch in range(epochs):
            train_loss = self._train_step(model)
            valid_loss, valid_metrics = self._valid_step(model)

            if self.logmodel is not None:
                self.logmodel.log_epoch(
                    train_loss=train_loss,
                    valid_loss=valid_loss,
                    valid_metrics=valid_metrics,
                    step=epoch,
                )

            new_best = False
            if valid_loss < best_loss:
                new_best = True
                best_loss = valid_loss
                epochs_no_improve = 0
                # warning, if the model is too big
                # maybe can cause Out of Memory
                best_state = copy.deepcopy(model.state_dict())

            logger.info(
                "epoch: %i, loss: %.4f, v_loss: %.4f%s",
                epoch,
                train_loss,
                valid_loss,
                ", ⭐️" if new_best else "",
            )

            if new_best:
                continue

            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

        model.load_state_dict(best_state)
        return model
