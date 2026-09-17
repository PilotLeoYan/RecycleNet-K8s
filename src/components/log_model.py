"""MLflow tracking integration for logging metrics, artifacts, and PyTorch models."""

from typing import Any

import mlflow
import numpy as np
from matplotlib import pyplot as plt
from mlflow.models import infer_signature

from src.config.schema import TrackingConfig


class LogModel:
    """Handles communication with MLflow for run tracking and model registry.

    Attributes:
        config: Tracking configuration settings for MLflow.
    """

    def __init__(
        self,
        config: TrackingConfig | None = None,
        registered_model_name: str | None = None,
    ) -> None:
        """Initializes the LogModel tracking helper.

        Args:
            config: MLflow tracking configuration settings. If None, default
                TrackingConfig is used.
            registered_model_name: Optional override for the registered model name.
        """
        self.config = config.model_copy() if config is not None else TrackingConfig()
        if registered_model_name is not None:
            self.config.registered_model_name = registered_model_name

    def log_epoch(
        self,
        train_loss: float,
        valid_loss: float,
        valid_metrics: dict[str, float],
        step: int,
    ) -> None:
        """Logs training and validation metrics for a specific epoch step.

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
        """Logs final test evaluation metrics and confusion matrix plot to MLflow.

        Args:
            roc: Area Under ROC Curve score on test dataset.
            metrics: Dictionary of test metrics (accuracy, precision, recall, f1_score).
            fig_cm: Matplotlib Figure object displaying the confusion matrix.
        """
        if not mlflow.active_run():
            return

        mlflow.log_metrics(
            {
                "test_roc": roc,
                "test_accuracy": metrics["accuracy"],
                "test_precision": metrics["precision"],
                "test_recall": metrics["recall"],
                "test_f1_score": metrics["f1_score"],
            }
        )

        mlflow.log_figure(fig_cm, "confusion_matrix.png")
        plt.close(fig_cm)  # close figures to avoid memory accumulation

    def log_model(
        self,
        dummy_input: np.ndarray,
        dummy_output: np.ndarray,
        model: Any,
        registered_model_name: str | None = None,
    ) -> None:
        """Logs and registers the trained PyTorch model with schema signature.

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
        target_model_name = (
            registered_model_name
            if registered_model_name is not None
            else self.config.registered_model_name
        )

        mlflow.pytorch.log_model(
            pytorch_model=cpu_model,
            name="model",
            signature=signature,
            input_example=dummy_input,
            serialization_format="pickle",
            registered_model_name=target_model_name,
            pip_requirements=[
                "torch",
                "torchvision",
                "cloudpickle",
            ],
        )
