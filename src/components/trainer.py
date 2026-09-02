"""Training loop execution, validation monitoring, early stopping, and checkpointing."""

from datetime import date
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from src.components.log_model import LogModel
from src.components.metrics import evals
from src.utils import get_logger

logger = get_logger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent


class ModelTrainer:
    """Orchestrates model training, validation, early stopping, and tracking.

    Attributes:
        model: PyTorch model being trained.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        criterion: Loss function module.
        optimizer: Optimization algorithm instance.
        device: Computation device (e.g., 'cuda' or 'cpu').
        log_model: MLflow logging helper.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        device: torch.device | str,
    ):
        """Initializes ModelTrainer with training dependencies and target device.

        Args:
            model: PyTorch neural network to train.
            train_loader: DataLoader providing training mini-batches.
            val_loader: DataLoader providing validation mini-batches.
            criterion: Loss function module (e.g. CrossEntropyLoss).
            optimizer: Optimizer instance (e.g. AdamW).
            device: Target torch device or device string ('cuda', 'cpu').
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = torch.device(device) if isinstance(device, str) else device

        self.log_model: LogModel = LogModel()
        self.model.to(self.device)

    def _train_step(self) -> float:
        """Executes a single training epoch across all mini-batches in train_loader.

        Returns:
            float: Average training loss across all samples in the epoch.
        """
        self.model.train()
        running_loss = 0.0

        for batch_x, batch_y in self.train_loader:
            batch_x = batch_x.to(self.device, non_blocking=True)
            batch_y = batch_y.to(self.device, non_blocking=True)

            y_pred = self.model(batch_x)
            loss = self.criterion(y_pred, batch_y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * batch_y.size(0)

        avg_loss = running_loss / len(self.train_loader.dataset)  # type: ignore
        return avg_loss  # type: ignore

    @torch.inference_mode()
    def _valid_step(self) -> tuple[float, dict[str, float]]:
        """Executes a validation pass over val_loader computing loss and metrics.

        Returns:
            tuple[float, dict[str, float]]: Average validation loss and computed
                metrics dictionary.
        """
        self.model.eval()
        running_vloss = 0.0

        batchs_predictions: list[np.ndarray] = []
        batchs_labels: list[np.ndarray] = []

        for vbatch_x, vbatch_y in self.val_loader:
            vbatch_x = vbatch_x.to(self.device, non_blocking=True)
            vbatch_y = vbatch_y.to(self.device, non_blocking=True)

            vy_pred = self.model(vbatch_x)
            vloss = self.criterion(vy_pred, vbatch_y)

            running_vloss += vloss.item() * vbatch_y.size(0)

            # convert logits to predictions
            predictions = torch.argmax(vy_pred, dim=1)
            batchs_predictions.extend(predictions.detach().cpu().numpy())
            batchs_labels.extend(vbatch_y.cpu().numpy())

        avg_loss_v = running_vloss / len(self.val_loader.dataset)  # type: ignore
        metrics = evals(np.array(batchs_labels), np.array(batchs_predictions))
        return avg_loss_v, metrics  # type: ignore

    def _save_weights(self, weights_path: Path, is_best: bool = False) -> Path:
        """Saves current model weights (state dict) to the specified path.

        Args:
            weights_path: Directory path where weights will be stored.
            is_best: Whether this checkpoint represents the best validation loss so far.

        Returns:
            Path: Full filepath of the saved .pth weights checkpoint.
        """
        if is_best:
            path = weights_path / Path(f"best-{date.today()}.pth")
        else:
            path = weights_path / Path(f"{date.today()}.pth")

        weights_path.mkdir(parents=True, exist_ok=True)

        torch.save(
            self.model.state_dict(),
            path,
        )
        return path

    def _registry_model(self, best_path: Path) -> None:
        """Loads best checkpoint and logs model artifact and signature to MLflow.

        Args:
            best_path: Filepath of the best weights checkpoint.
        """
        self.model.load_state_dict(torch.load(best_path, map_location=self.device))
        self.model.eval()

        dummy_input = torch.randn(1, *self.val_loader.dataset[0][0].shape).to(
            self.device
        )
        with torch.no_grad():
            dummy_output = self.model(dummy_input)

        self.log_model.log_model(
            dummy_input.detach().cpu().numpy(),
            dummy_output.detach().cpu().numpy(),
            self.model,
        )

    def fit(self, epochs: int, weights_path: str, patience: int = 3) -> Path:
        """Runs the complete training and validation cycle with early stopping.

        Args:
            epochs: Maximum number of training epochs to execute.
            weights_path: Destination directory string to save checkpoints.
            patience: Number of epochs without improvement before early stopping.

        Returns:
            Path: Path to the best saved model weights checkpoint.
        """
        path = Path(weights_path)

        best_loss = float("inf")
        epochs_no_improve = 0
        best_path = None

        for epoch in range(epochs):
            train_loss = self._train_step()
            valid_loss, valid_metrics = self._valid_step()

            self.log_model.log_epoch(
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
                best_path = self._save_weights(path, True)

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

        if best_path is None:
            best_path = self._save_weights(path, False)

        if best_path.exists():
            self._registry_model(best_path)

        return best_path
