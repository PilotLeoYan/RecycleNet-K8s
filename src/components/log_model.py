"""MLflow tracking integration for logging metrics, artifacts, and PyTorch models."""

from datetime import datetime
from typing import Any

import mlflow
import numpy as np
import torch
from matplotlib import pyplot as plt
from mlflow import ActiveRun
from mlflow.models import infer_signature

from src.config import TrackingConfig


class LogModel:
    """Handle communication with MLflow for run tracking and model registry.

    Attributes:
        config: Tracking configuration settings for MLflow.
    """

    def __init__(
        self,
        config: TrackingConfig | None = None,
        registered_model_name: str | None = None,
    ) -> None:
        """Initialize the LogModel tracking helper.

        Args:
            config: MLflow tracking configuration settings. If None, default
                TrackingConfig is used.
            registered_model_name: Optional override for the registered model name.
        """
        self.config = config.model_copy() if config is not None else TrackingConfig()
        if registered_model_name is not None:
            self.config.registered_model_name = registered_model_name

    def set_experiment_tracking(self) -> None:
        """Configure MLflow tracking URI and active experiment name."""
        mlflow.set_tracking_uri(self.config.tracking_uri)
        mlflow.set_experiment(self.config.experiment_name)

    def start_run(self) -> ActiveRun:
        """Initialize and start a new MLflow active run with timestamped name.

        Returns:
            ActiveRun: The newly started MLflow active run object.
        """
        self.set_experiment_tracking()
        run_name = f"mobilenetv3_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        return mlflow.start_run(run_name=run_name)

    def log_epoch(
        self,
        train_loss: float,
        valid_loss: float,
        valid_metrics: dict[str, float],
        step: int,
    ) -> None:
        """Log training and validation metrics for a specific epoch step.

        Args:
            train_loss: Average loss on the training dataset.
            valid_loss: Average loss on the validation dataset.
            valid_metrics: Dictionary containing accuracy, precision, recall, and
                f1_score.
            step: Epoch index (step) for MLflow metric history.
        """
        if not mlflow.active_run():
            return

        mlflow.log_metrics(
            {
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "valid_accuracy": valid_metrics["accuracy"],
                "valid_precision": valid_metrics["precision"],
                "valid_recall": valid_metrics["recall"],
                "valid_f1_score": valid_metrics["f1_score"],
            },
            step=step,
        )

    def log_test(
        self,
        roc: float,
        metrics: dict[str, float],
        fig_cm: Any,
    ) -> None:
        """Log final test evaluation metrics and confusion matrix plot to MLflow.

        Args:
            roc: Area Under ROC Curve score on test dataset.
            metrics: Dictionary of test metrics (accuracy, precision, recall, f1_score).
            fig_cm: Matplotlib Figure object displaying the confusion matrix.
        """
        if not mlflow.active_run():
            return

        metrics_to_log: dict[str, float] = {
            "test_roc": roc,
            "test_accuracy": metrics.get("accuracy", 0.0),
            "test_precision": metrics.get("precision", 0.0),
            "test_recall": metrics.get("recall", 0.0),
            "test_f1_score": metrics.get("f1_score", 0.0),
        }
        if "test_loss" in metrics:
            metrics_to_log["test_loss"] = float(metrics["test_loss"])

        mlflow.log_metrics(metrics_to_log)

        if fig_cm is not None:
            mlflow.log_figure(fig_cm, "confusion_matrix.png")
            plt.close(fig_cm)  # close figures to avoid memory accumulation

    def log_model(
        self,
        dummy_input: np.ndarray,
        dummy_output: np.ndarray,
        model: Any,
        registered_model_name: str | None = None,
    ) -> None:
        """Log and register the trained PyTorch model with schema signature.

        Args:
            dummy_input: Sample input array for schema signature inference.
            dummy_output: Corresponding model output array for schema inference.
            model: Trained PyTorch model instance to log.
            registered_model_name: Optional override for the registered model name.
        """
        if not mlflow.active_run():
            return

        signature = infer_signature(dummy_input, dummy_output)
        cpu_model = model.to("cpu") if hasattr(model, "to") else model
        target_name = registered_model_name or self.config.registered_model_name

        mlflow.pytorch.log_model(
            pytorch_model=cpu_model,
            name="model",
            signature=signature,
            input_example=dummy_input,
            serialization_format="pickle",
            registered_model_name=target_name,
            pip_requirements=[
                "torch",
                "torchvision",
                "cloudpickle",
            ],
        )

    def register_checkpoint(
        self,
        model: torch.nn.Module,
        input_shape: tuple[int, ...],
    ) -> None:
        """Register the trained PyTorch model checkpoint in MLflow Model Registry.

        Args:
            model: PyTorch model instance to log and register.
            input_shape: Input tensor shape tuple for dummy input inference.
        """
        if not mlflow.active_run():
            return

        model.to("cpu")
        model.eval()

        dummy_input = torch.randn(input_shape).to("cpu")
        with torch.no_grad():
            dummy_output = model(dummy_input)

        self.log_model(
            dummy_input=dummy_input.detach().numpy(),
            dummy_output=dummy_output.detach().numpy(),
            model=model,
        )

    def log_pipeline_metadata(
        self,
        idx_to_class: dict[str, str],
        tags: dict[str, str],
        params: dict[str, Any],
        config_dict: dict[str, Any],
    ) -> None:
        """Log pipeline metadata, tags, and configuration to MLflow.

        Args:
            idx_to_class: Mapping from class indices to class labels.
            tags: Dictionary of MLflow run tags.
            params: Dictionary of pipeline hyperparameters and settings.
            config_dict: Serialized configuration dictionary.
        """
        if not mlflow.active_run():
            return

        mlflow.log_dict(idx_to_class, "classes_mapping.json")
        mlflow.set_tags(tags)
        mlflow.log_params(params)
        mlflow.log_dict(config_dict, "run_config.json")
