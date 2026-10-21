"""Unit tests for the ONNX export pipeline and CLI parser."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

from src.components.model import build_mobilenet_v3
from src.config.schema import (
    AppConfig,
    ExportConfig,
    IngestionConfig,
    TrainingConfig,
    TransformationConfig,
)
from src.pipeline.export_onnx import (
    ONNXExportPipeline,
    export_to_onnx,
    validate_onnx_graph,
    verify_numerical_parity,
)
from src.utils import RecycleNetException
from src.utils.parser import build_parser


class DummyClassifier(nn.Module):
    """Minimal neural network for fast ONNX export unit testing."""

    def __init__(self, num_classes: int = 6) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(3, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)  # type: ignore[no-any-return]


@pytest.fixture
def sample_config(tmp_path: Path) -> AppConfig:
    """Create a minimal AppConfig for ONNX export testing."""
    return AppConfig(
        ingestion=IngestionConfig(
            zip_source=tmp_path / "dummy.zip",
            raw_output_dir=tmp_path / "raw",
        ),
        transformation=TransformationConfig(
            image_size=(32, 32),
            batch_size=2,
            num_workers=0,
        ),
        training=TrainingConfig(epochs=1, patience=1, device="cpu"),
    )


def test_export_to_onnx_creates_valid_file(tmp_path: Path) -> None:
    """Verify export_to_onnx serializes a dummy module to disk."""
    model = DummyClassifier(num_classes=6)
    output_path = tmp_path / "models" / "test_model.onnx"

    resolved_path = export_to_onnx(model, (1, 3, 32, 32), output_path)

    assert resolved_path == output_path.resolve()
    assert resolved_path.exists()
    assert resolved_path.stat().st_size > 0


def test_validate_onnx_graph_passes_and_fails(tmp_path: Path) -> None:
    """Verify validate_onnx_graph accepts valid ONNX models and rejects bad files."""
    model = DummyClassifier(num_classes=6)
    valid_onnx = export_to_onnx(model, (1, 3, 32, 32), tmp_path / "valid.onnx")

    validate_onnx_graph(valid_onnx)

    invalid_onnx = tmp_path / "invalid.onnx"
    invalid_onnx.write_text("not a valid protobuf onnx file", encoding="utf-8")

    with pytest.raises(RecycleNetException, match="ONNX graph validation failed"):
        validate_onnx_graph(invalid_onnx)


def test_verify_numerical_parity_passes_and_diverges(tmp_path: Path) -> None:
    """Verify numerical parity succeeds on matching weights and fails on mutation."""
    model = DummyClassifier(num_classes=6)
    onnx_path = export_to_onnx(model, (1, 3, 32, 32), tmp_path / "parity.onnx")

    max_diff = verify_numerical_parity(model, onnx_path, (1, 3, 32, 32), tolerance=1e-4)
    assert max_diff <= 1e-4

    # Perturb PyTorch model weights so outputs diverge from exported ONNX
    with torch.no_grad():
        model.fc.weight.fill_(999.0)
        model.fc.bias.fill_(-999.0)

    with pytest.raises(RecycleNetException, match="Numerical parity check failed"):
        verify_numerical_parity(model, onnx_path, (1, 3, 32, 32), tolerance=1e-5)


def test_onnx_export_pipeline_with_local_checkpoint(
    sample_config: AppConfig, tmp_path: Path
) -> None:
    """Verify ONNXExportPipeline.run exports a local MobileNetV3 checkpoint."""
    num_classes = 6
    model = build_mobilenet_v3(num_classes)
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save({"model_state": model.state_dict()}, checkpoint_path)

    output_onnx = tmp_path / "exported" / "recyclenet.onnx"
    pipeline = ONNXExportPipeline(sample_config, output_path=output_onnx)

    with patch("src.pipeline.export_onnx.mlflow.active_run", return_value=None):
        result_path = pipeline.run(
            num_classes=num_classes, checkpoint_path=checkpoint_path
        )

    assert result_path == output_onnx.resolve()
    assert result_path.exists()


def test_onnx_export_pipeline_with_mlflow_registry(
    sample_config: AppConfig, tmp_path: Path
) -> None:
    """Verify ONNXExportPipeline.run loads from MLflow when checkpoint_path is None."""
    dummy_model = DummyClassifier(num_classes=6)
    output_onnx = tmp_path / "mlflow_exported" / "recyclenet.onnx"
    pipeline = ONNXExportPipeline(sample_config, output_path=output_onnx)

    with (
        patch("src.pipeline.export_onnx.mlflow.set_tracking_uri") as mock_set_uri,
        patch(
            "src.pipeline.export_onnx.mlflow.pytorch.load_model",
            return_value=dummy_model,
        ) as mock_load_model,
        patch("src.pipeline.export_onnx.mlflow.active_run", return_value=MagicMock()),
        patch("src.pipeline.export_onnx.mlflow.log_artifact") as mock_log_artifact,
        patch("src.pipeline.export_onnx.mlflow.log_metric") as mock_log_metric,
    ):
        result_path = pipeline.run(num_classes=6, checkpoint_path=None)

    mock_set_uri.assert_called_once_with(sample_config.tracking.tracking_uri)
    mock_load_model.assert_called_once()
    mock_log_artifact.assert_called_once()
    mock_log_metric.assert_called_once()
    assert result_path == output_onnx.resolve()
    assert result_path.exists()


def test_cli_parser_subcommands() -> None:
    """Verify build_parser() supports train --use-best-hpo, hpo, and export."""
    parser = build_parser()

    train_args = parser.parse_args(["train", "--use-best-hpo"])
    assert train_args.command == "train"
    assert train_args.use_best_hpo is True

    train_default_args = parser.parse_args(["train"])
    assert train_default_args.command == "train"
    assert train_default_args.use_best_hpo is False

    hpo_args = parser.parse_args(["-c", "configs/smoke.yaml", "hpo"])
    assert hpo_args.command == "hpo"
    assert hpo_args.config == Path("configs/smoke.yaml")

    export_args = parser.parse_args(
        [
            "export",
            "--checkpoint",
            "artifacts/best.pt",
            "--output",
            "models/custom.onnx",
            "--num-classes",
            "10",
        ]
    )
    assert export_args.command == "export"
    assert export_args.checkpoint == Path("artifacts/best.pt")
    assert export_args.output == Path("models/custom.onnx")
    assert export_args.num_classes == 10


def test_cli_parser_export_default_output() -> None:
    """Verify build_parser() defaults export --output to None to defer to config."""
    parser = build_parser()
    args = parser.parse_args(["export"])
    assert args.command == "export"
    assert args.output is None


def test_smoke_config_isolation() -> None:
    """Verify configs/smoke.yaml isolates artifacts and does not pollute runs."""
    config = AppConfig.from_yaml("configs/smoke.yaml")

    # Export isolation: smoke export should never touch production ONNX model
    assert config.export.output_path == Path("models/smoke_recyclenet.onnx")
    assert config.export.output_path != Path("models/recyclenet.onnx")

    # Tracking isolation: smoke training should never touch production RecycleNet
    assert config.tracking.registered_model_name == "RecycleNet_Dev"
    assert config.tracking.registered_model_name != "RecycleNet"
    assert config.tracking.experiment_name == "RecycleNet_Smoke_Test"
    assert config.tracking.experiment_name != "RecycleNet_Training"

    # HPO isolation: smoke HPO should never touch production best_hpo_config.yaml
    assert config.hpo.best_config_path == Path("configs/best_hpo_smoke_config.yaml")
    assert config.hpo.best_config_path != Path("configs/best_hpo_config.yaml")


def test_onnx_export_pipeline_smoke_resolves_config_output(tmp_path: Path) -> None:
    """Verify ONNXExportPipeline uses config.export.output_path when output is None."""
    smoke_output = tmp_path / "models" / "smoke_recyclenet.onnx"
    config = AppConfig(
        ingestion=IngestionConfig(
            zip_source=tmp_path / "dummy.zip",
            raw_output_dir=tmp_path / "raw",
        ),
        transformation=TransformationConfig(
            image_size=(32, 32),
            batch_size=2,
            num_workers=0,
        ),
        training=TrainingConfig(epochs=1, patience=1, device="cpu"),
        export=ExportConfig(output_path=smoke_output),
    )

    dummy_model = DummyClassifier(num_classes=6)
    pipeline = ONNXExportPipeline(config, output_path=None)
    assert pipeline.output_path == smoke_output

    with (
        patch("src.pipeline.export_onnx.mlflow.set_tracking_uri"),
        patch(
            "src.pipeline.export_onnx.mlflow.pytorch.load_model",
            return_value=dummy_model,
        ),
        patch("src.pipeline.export_onnx.mlflow.active_run", return_value=None),
    ):
        result_path = pipeline.run(num_classes=6, checkpoint_path=None)

    assert result_path == smoke_output.resolve()
    assert result_path.exists()
