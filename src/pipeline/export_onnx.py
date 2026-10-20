"""ONNX export pipeline for RecycleNet PyTorch model serialization and validation."""

import argparse
from pathlib import Path
from typing import Any, cast

import mlflow
import mlflow.pytorch
import numpy as np
import onnx
import onnxruntime as ort
import torch

from src.components.model import build_mobilenet_v3
from src.config.schema import AppConfig
from src.utils import RecycleNetException, get_logger

logger = get_logger(__name__)

ONNX_OPSET = 17
DEFAULT_OUTPUT_PATH = Path("models/recyclenet.onnx")
PARITY_TOLERANCE = 1e-5


def export_to_onnx(
    model: torch.nn.Module,
    input_shape: tuple[int, ...],
    output_path: Path,
) -> Path:
    """Export a PyTorch model to ONNX format.

    Args:
        model: Trained PyTorch model in eval mode.
        input_shape: Input tensor shape (batch, channels, height, width).
        output_path: Destination path for the exported ONNX file.

    Returns:
        Path: Resolved path to the exported ONNX file.

    Raises:
        RecycleNetException: If export fails.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    dummy_input = torch.randn(input_shape)

    try:
        torch.onnx.export(
            model,
            (dummy_input,),
            str(output_path),
            opset_version=ONNX_OPSET,
            input_names=["image"],
            output_names=["logits"],
            dynamic_axes={
                "image": {0: "batch_size"},
                "logits": {0: "batch_size"},
            },
            dynamo=False,
        )
        logger.info("Model exported to ONNX at %s (opset=%d)", output_path, ONNX_OPSET)
        return output_path.resolve()
    except Exception as e:
        raise RecycleNetException("Failed to export model to ONNX", e) from e


def validate_onnx_graph(onnx_path: Path) -> None:
    """Validate the exported ONNX graph structure and integrity.

    Args:
        onnx_path: Path to the ONNX model file.

    Raises:
        RecycleNetException: If validation fails.
    """
    try:
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        logger.info("ONNX graph validation passed for %s", onnx_path)
    except Exception as e:
        raise RecycleNetException("ONNX graph validation failed", e) from e


def verify_numerical_parity(
    model: torch.nn.Module,
    onnx_path: Path,
    input_shape: tuple[int, ...],
    tolerance: float = PARITY_TOLERANCE,
) -> float:
    """Verify numerical parity between PyTorch and ONNX Runtime outputs.

    Args:
        model: Trained PyTorch model in eval mode.
        onnx_path: Path to the exported ONNX model.
        input_shape: Input tensor shape for the test sample.
        tolerance: Maximum allowed absolute difference between outputs.

    Returns:
        float: Maximum absolute difference observed between outputs.

    Raises:
        RecycleNetException: If max difference exceeds tolerance.
    """
    dummy_input = torch.randn(input_shape)

    model.eval()
    with torch.no_grad():
        torch_out: np.ndarray[Any, np.dtype[np.float32]] = model(dummy_input).numpy()

    session = ort.InferenceSession(str(onnx_path))
    ort_inputs = {session.get_inputs()[0].name: dummy_input.numpy()}
    ort_out: np.ndarray[Any, np.dtype[np.float32]] = session.run(None, ort_inputs)[0]

    max_diff = float(np.max(np.abs(torch_out - ort_out)))
    logger.info("Numerical parity check — max |pytorch - onnx| = %.2e", max_diff)

    if max_diff > tolerance:
        raise RecycleNetException(
            f"Numerical parity check failed: "
            f"max_diff={max_diff:.2e} > tolerance={tolerance:.2e}"
        )

    logger.info("Parity check PASSED (tolerance=%.2e)", tolerance)
    return max_diff


class ONNXExportPipeline:
    """Orchestrate ONNX export, validation, parity check, and MLflow artifact logging.

    Attributes:
        config: Application configuration settings.
        output_path: Destination path for exported ONNX model.
    """

    def __init__(
        self,
        config: AppConfig,
        output_path: Path = DEFAULT_OUTPUT_PATH,
    ) -> None:
        """Initialize the ONNX export pipeline.

        Args:
            config: Root application configuration object.
            output_path: Output path for ONNX file. Defaults to models/recyclenet.onnx.
        """
        self.config = config
        self.output_path = output_path

    def _load_model_from_mlflow(self, num_classes: int) -> torch.nn.Module:
        """Load the best registered model from MLflow Model Registry.

        Args:
            num_classes: Number of classification output classes.

        Returns:
            torch.nn.Module: Loaded PyTorch model in eval mode on CPU.

        Raises:
            RecycleNetException: If model loading fails.
        """
        try:
            mlflow.set_tracking_uri(self.config.tracking.tracking_uri)
            model_uri = f"models:/{self.config.tracking.registered_model_name}/latest"
            logger.info("Loading model from MLflow registry: %s", model_uri)
            model = mlflow.pytorch.load_model(model_uri, map_location="cpu")
            model = cast(torch.nn.Module, model)
            model.eval()
            logger.info("Model loaded successfully from MLflow registry.")
            return model  # type: ignore[return-value]
        except Exception as e:
            raise RecycleNetException(
                f"Failed to load model from MLflow registry "
                f"'{self.config.tracking.registered_model_name}'",
                e,
            ) from e

    def _load_model_from_checkpoint(
        self, checkpoint_path: Path, num_classes: int
    ) -> torch.nn.Module:
        """Load a PyTorch model from a local checkpoint file.

        Args:
            checkpoint_path: Path to the .pt checkpoint file.
            num_classes: Number of classification output classes.

        Returns:
            torch.nn.Module: Loaded PyTorch model in eval mode on CPU.

        Raises:
            RecycleNetException: If checkpoint loading fails.
        """
        try:
            model: torch.nn.Module = cast(
                torch.nn.Module, build_mobilenet_v3(num_classes)
            )
            checkpoint: dict[str, Any] = torch.load(
                checkpoint_path, map_location="cpu", weights_only=True
            )
            state_dict = checkpoint.get("model_state", checkpoint)
            model.load_state_dict(state_dict)
            model.eval()
            logger.info("Model loaded from checkpoint: %s", checkpoint_path)
            return model
        except Exception as e:
            raise RecycleNetException(
                f"Failed to load checkpoint from {checkpoint_path}", e
            ) from e

    def run(
        self,
        num_classes: int = 6,
        checkpoint_path: Path | None = None,
    ) -> Path:
        """Execute the full ONNX export pipeline.

        Args:
            num_classes: Number of target classes. Defaults to 6 (TrashNet).
            checkpoint_path: Optional path to a local .pt checkpoint. If None,
                loads from MLflow registry.

        Returns:
            Path: Resolved path to the exported ONNX file.

        Raises:
            RecycleNetException: If any stage of the pipeline fails.
        """
        logger.info("Starting ONNX export pipeline...")

        if checkpoint_path is not None:
            model = self._load_model_from_checkpoint(checkpoint_path, num_classes)
        else:
            model = self._load_model_from_mlflow(num_classes)

        input_shape = (1, 3, *self.config.transformation.image_size)

        onnx_path = export_to_onnx(model, input_shape, self.output_path)
        validate_onnx_graph(onnx_path)
        max_diff = verify_numerical_parity(model, onnx_path, input_shape)

        if mlflow.active_run():
            mlflow.log_artifact(str(onnx_path), artifact_path="onnx")
            mlflow.log_metric("onnx_parity_max_diff", max_diff)
            logger.info("ONNX artifact logged to MLflow.")

        logger.info("ONNX export pipeline completed. Output: %s", onnx_path)
        return onnx_path


def main() -> None:
    """CLI entrypoint for ONNX export pipeline."""
    parser = argparse.ArgumentParser(description="Export RecycleNet model to ONNX.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/config.yaml"),
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "Path to a local .pt checkpoint file. If not provided, loads from MLflow."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output path for the ONNX file.",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=6,
        help="Number of output classes.",
    )
    args = parser.parse_args()

    config = AppConfig.from_yaml(args.config)
    pipeline = ONNXExportPipeline(config, output_path=args.output)
    pipeline.run(num_classes=args.num_classes, checkpoint_path=args.checkpoint)


if __name__ == "__main__":
    main()
