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
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        device: torch.device | str,
    ):
        """"""
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = torch.device(device) if isinstance(device, str) else device

        self.log_model: LogModel = LogModel()
        self.model.to(self.device)

    def _train_step(self) -> float:
        """"""
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
        """"""
        self.model.eval()
        running_vloss = 0.0

        batchs_predictions: list[np.ndarray] = []
        batchs_labels: list[np.ndarray] = []

        for vbatch_x, vbatch_y in self.val_loader:
            vbatch_x = vbatch_x.to(self.device, non_blocking=True)
            vbatch_y = vbatch_y.to(self.device, non_blocking=True)

            vy_pred = self.model(vbatch_x)
            vloss = self.criterion(vy_pred, vbatch_y)

            running_vloss += vloss.item()

            # convert logits to predictions
            predictions = torch.argmax(vy_pred, dim=1)
            batchs_predictions.extend(predictions.detach().cpu().numpy())
            batchs_labels.extend(vbatch_y.cpu().numpy())

        avg_loss_v = running_vloss / len(self.val_loader)
        metrics = evals(np.array(batchs_labels), np.array(batchs_predictions))
        return avg_loss_v, metrics

    def _save_weights(self, weights_path: Path, is_best: bool = False) -> Path:
        """"""
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
        """"""
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
