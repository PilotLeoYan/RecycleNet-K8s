"""Distributed model evaluation component for assessing test performance."""

from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from matplotlib.figure import Figure
from sklearn.metrics import ConfusionMatrixDisplay
from torch.utils.data import DataLoader

from src.components.metrics import calculate_roc_auc

CMAP = "Blues"


class TestDistributed:
    """Evaluate a trained model across distributed workers on the test dataset.

    Attributes:
        model: PyTorch neural network model to evaluate.
        test_loader: DataLoader providing the test dataset partition.
        criterion: Loss function module.
        num_classes: Total number of target classification classes.
        classes_name: List of class label strings.
        device: Computation device used for evaluation.
    """

    __test__ = False

    def __init__(
        self,
        model: torch.nn.Module,
        test_loader: DataLoader,
        criterion: torch.nn.Module,
        num_classes: int,
        classes_name: list[str],
        device: torch.device | str,
    ) -> None:
        """Initialize TestDistributed evaluation component.

        Args:
            model: PyTorch model to evaluate.
            test_loader: DataLoader for test partition.
            criterion: Loss function criterion (e.g., CrossEntropyLoss).
            num_classes: Number of distinct classes.
            classes_name: Human-readable class names.
            device: Computation device or identifier string.
        """
        self.model = model
        self.test_loader = test_loader
        self.criterion = criterion
        self.num_classes = num_classes
        self.classes_name = classes_name
        self.device = torch.device(device) if isinstance(device, str) else device

    def _local_metrics(
        self,
        total_samples: torch.Tensor,
        loss_sum: torch.Tensor,
        conf_matrix: torch.Tensor,
    ) -> dict[str, Any]:
        """Compute evaluation metrics and confusion matrix from aggregated tensors.

        Args:
            total_samples: 1D tensor containing total number of evaluated samples.
            loss_sum: 1D tensor containing accumulated total loss.
            conf_matrix: 2D tensor containing the confusion matrix.

        Returns:
            dict[str, Any]: Dictionary containing test_loss, test_accuracy,
                test_precision, test_recall, test_f1_score, and confusion_matrix figure.
        """
        total = int(total_samples.item())
        avg_loss = float((loss_sum / total).item()) if total > 0 else 0.0

        cm = conf_matrix.detach().cpu().numpy()
        tp = cm.diagonal()
        accuracy = float(tp.sum() / total) if total > 0 else 0.0

        col_sums = cm.sum(axis=0)
        row_sums = cm.sum(axis=1)

        precision = float((tp / np.maximum(col_sums, 1)).mean())
        recall = float((tp / np.maximum(row_sums, 1)).mean())
        f1_score = float(2.0 * (precision * recall) / (precision + recall + 1e-8))

        display_labels = (
            self.classes_name
            if len(self.classes_name) == self.num_classes
            else [str(i) for i in range(self.num_classes)]
        )
        cm_disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=display_labels,
        )
        fig_cm: Figure = cm_disp.plot(cmap=CMAP).figure_

        return {
            "test_loss": avg_loss,
            "test_accuracy": accuracy,
            "test_precision": precision,
            "test_recall": recall,
            "test_f1_score": f1_score,
            "confusion_matrix": fig_cm,
        }

    @torch.inference_mode()
    def _test_step(
        self,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """Execute test inference pass and gather metrics across distributed workers.

        Returns:
            tuple[np.ndarray, np.ndarray, dict[str, Any]]: Tuple containing global
                probabilities array, global targets array, and test metrics dictionary.
        """
        self.model.eval()

        local_probas_list: list[np.ndarray] = []
        local_targets_list: list[np.ndarray] = []

        local_loss_sum = torch.zeros(1, dtype=torch.float64, device=self.device)
        local_total_samples = torch.zeros(1, dtype=torch.int64, device=self.device)
        local_conf_matrix = torch.zeros(
            (self.num_classes, self.num_classes),
            dtype=torch.int64,
            device=self.device,
        )

        for images, targets in self.test_loader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            logits = self.model(images)
            loss = self.criterion(logits, targets)
            probas = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            local_probas_list.append(probas.detach().cpu().numpy())
            local_targets_list.append(targets.detach().cpu().numpy())

            batch_size = targets.size(0)
            local_loss_sum += loss.item() * batch_size
            local_total_samples += batch_size

            indices = targets * self.num_classes + preds
            local_conf_matrix += torch.bincount(
                indices, minlength=self.num_classes**2
            ).reshape(self.num_classes, self.num_classes)

        if local_probas_list:
            local_probas_arr = np.concatenate(local_probas_list, axis=0)
            local_targets_arr = np.concatenate(local_targets_list, axis=0)
        else:
            local_probas_arr = np.empty((0, self.num_classes), dtype=np.float32)
            local_targets_arr = np.empty((0,), dtype=np.int64)

        if dist.is_initialized():
            dist.all_reduce(local_conf_matrix, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_loss_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_total_samples, op=dist.ReduceOp.SUM)

            world_size = dist.get_world_size()
            gathered_probas: list[np.ndarray | None] = [None] * world_size
            gathered_targets: list[np.ndarray | None] = [None] * world_size
            dist.all_gather_object(gathered_probas, local_probas_arr)
            dist.all_gather_object(gathered_targets, local_targets_arr)

            global_probas = np.concatenate(
                [p for p in gathered_probas if p is not None], axis=0
            )
            global_targets = np.concatenate(
                [t for t in gathered_targets if t is not None], axis=0
            )
        else:
            global_probas = local_probas_arr
            global_targets = local_targets_arr

        metrics = self._local_metrics(
            total_samples=local_total_samples,
            loss_sum=local_loss_sum,
            conf_matrix=local_conf_matrix,
        )

        return global_probas, global_targets, metrics

    @torch.inference_mode()
    def compute_distributed_test_metrics(self) -> dict[str, Any]:
        """Compute end-to-end distributed evaluation metrics including ROC AUC score.

        Returns:
            dict[str, Any]: Comprehensive test metrics dictionary including auc,
                test_loss, test_accuracy, test_precision, test_recall, test_f1_score,
                and confusion_matrix figure.
        """
        global_probas, global_targets, metrics = self._test_step()
        global_auc = calculate_roc_auc(global_targets, global_probas)
        metrics["auc"] = global_auc
        return metrics
