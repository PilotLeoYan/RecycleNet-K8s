# TODO:
# add metrics as a dict at the end of:
# - _train_step()
# - _valid_step()

from datetime import date
from pathlib import Path

import mlflow
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


class ModelTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        device: torch.device,
        weights_path: Path,
    ):
        """"""
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.weights_path = weights_path

        self.model.to(self.device)

    def _train_step(self) -> tuple[float, dict[str, float]]:
        """"""
        self.model.train()
        running_loss = 0.0

        # edit this line
        metrics: dict[str, float] = dict()

        for batch_x, batch_y in self.train_loader:
            batch_x = batch_x.to(self.device, non_blocking=True)
            batch_y = batch_y.to(self.device, non_blocking=True)

            y_pred = self.model(batch_x)
            loss = self.criterion(y_pred, batch_y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
        avg_loss = running_loss / len(self.train_loader)
        return avg_loss, metrics

    @torch.inference_mode()
    def _valid_step(self) -> tuple[float, dict[str, float]]:
        """"""
        self.model.eval()
        running_vloss = 0.0

        # edit this line
        metrics: dict[str, float] = dict()

        for vbatch_x, vbatch_y in self.val_loader:
            vbatch_x = vbatch_x.to(self.device, non_blocking=True)
            vbatch_y = vbatch_y.to(self.device, non_blocking=True)

            vy_pred = self.model(vbatch_x)
            vloss = self.criterion(vy_pred, vbatch_y)

            running_vloss += vloss.item()
        avg_loss_v = running_vloss / len(self.val_loader)
        return avg_loss_v, metrics

    def _save_weights(self, is_best: bool = False) -> Path:
        """"""
        if is_best:
            path = self.weights_path / f"best-{date.today()}.pth"
        else:
            path = self.weights_path / f"{date.today()}.pth"

        torch.save(
            self.model.state_dict(),
            path,
        )
        return path

    def fit(self, epochs: int, patience: int = 3) -> None:
        """"""
        best_loss = float("inf")
        epochs_no_improve = 0

        for epoch in range(epochs):
            train_loss, train_metrics = self._train_step()
            valid_loss, valid_metrics = self._valid_step()

            metrics_to_log = {
                "train_loss": train_loss,
                "train_some_metric": -1.0,
                "valid_loss": valid_loss,
                "valid_some_metric": -2.0,
            }

            mlflow.log_metrics(
                metrics_to_log,
                step=epoch,
            )

            if valid_loss < best_loss:
                best_loss = valid_loss
                epochs_no_improve = 0
                self._save_weights(True)
                continue

            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

        self._save_weights()
