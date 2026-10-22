"""Unit tests for the three-stage SuperTrainPipeline orchestrator."""

from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from src.config.schema import (
    AppConfig,
    ExportConfig,
    HPOConfig,
    IngestionConfig,
    TrainingConfig,
    TransformationConfig,
)
from src.pipeline.super_train_pipeline import SuperTrainPipeline
from src.utils.exception import RecycleNetException


@pytest.fixture
def mock_app_config(tmp_path: Path) -> AppConfig:
    """Create a minimal mock AppConfig for super-train pipeline testing."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_file = tmp_path / "dataset.zip"
    zip_file.touch()
    hpo_best_path = tmp_path / "artifacts" / "best_hpo_config.yaml"

    return AppConfig(
        ingestion=IngestionConfig(
            zip_source=zip_file,
            raw_output_dir=raw_dir,
        ),
        transformation=TransformationConfig(
            image_size=(16, 16),
            batch_size=2,
            num_workers=0,
            pin_memory=False,
        ),
        training=TrainingConfig(
            epochs=1,
            num_workers=1,
            device="cpu",
        ),
        hpo=HPOConfig(
            num_samples=1,
            max_epochs=1,
            grace_period=1,
            best_config_path=hpo_best_path,
        ),
        export=ExportConfig(
            output_path=tmp_path / "models" / "recyclenet.onnx",
        ),
    )


def test_super_train_pipeline_init_defaults(mock_app_config: AppConfig) -> None:
    """Verify default initialization values when optional arguments are omitted."""
    pipeline = SuperTrainPipeline(mock_app_config)

    assert pipeline.config is mock_app_config
    assert pipeline.output_path == mock_app_config.export.output_path
    assert pipeline.skip_hpo is False


def test_super_train_pipeline_init_custom_params(mock_app_config: AppConfig) -> None:
    """Verify custom output_path and skip_hpo configuration."""
    custom_path = Path("/custom/export/path/model.onnx")
    pipeline = SuperTrainPipeline(
        mock_app_config,
        output_path=custom_path,
        skip_hpo=True,
    )

    assert pipeline.config is mock_app_config
    assert pipeline.output_path == custom_path
    assert pipeline.skip_hpo is True


def test_super_train_pipeline_init_fallback_when_export_config_absent() -> None:
    """Verify fallback to models/recyclenet.onnx when export config is absent."""
    mock_cfg = MagicMock(spec=AppConfig)
    mock_cfg.export = None

    pipeline = SuperTrainPipeline(mock_cfg)
    assert pipeline.output_path == Path("models/recyclenet.onnx")


@patch("src.pipeline.super_train_pipeline.ONNXExportPipeline")
@patch("src.pipeline.super_train_pipeline.TrainPipeline")
@patch("src.pipeline.super_train_pipeline.HPOPipeline")
def test_super_train_pipeline_run_sequential_execution(
    mock_hpo_cls: MagicMock,
    mock_train_cls: MagicMock,
    mock_export_cls: MagicMock,
    mock_app_config: AppConfig,
    tmp_path: Path,
) -> None:
    """Verify all 3 stages execute sequentially when skip_hpo is False."""
    expected_onnx_path = tmp_path / "final_model.onnx"
    mock_hpo_instance = mock_hpo_cls.return_value
    mock_train_instance = mock_train_cls.return_value
    mock_export_instance = mock_export_cls.return_value
    mock_export_instance.run.return_value = expected_onnx_path

    manager = MagicMock()
    manager.attach_mock(mock_hpo_instance.run, "hpo_run")
    manager.attach_mock(mock_train_instance.run, "train_run")
    manager.attach_mock(mock_export_instance.run, "export_run")

    pipeline = SuperTrainPipeline(mock_app_config, skip_hpo=False)
    result_path = pipeline.run(num_classes=6)

    assert result_path == expected_onnx_path
    mock_hpo_cls.assert_called_once_with(mock_app_config)
    mock_hpo_instance.run.assert_called_once()

    mock_train_cls.assert_called_once_with(mock_app_config)
    mock_train_instance.run.assert_called_once()

    mock_export_cls.assert_called_once_with(
        mock_app_config, output_path=pipeline.output_path
    )
    mock_export_instance.run.assert_called_once_with(num_classes=6)

    expected_calls = [
        call.hpo_run(),
        call.train_run(),
        call.export_run(num_classes=6),
    ]
    assert manager.mock_calls == expected_calls


@patch("src.pipeline.super_train_pipeline.AppConfig.from_yaml")
@patch("src.pipeline.super_train_pipeline.ONNXExportPipeline")
@patch("src.pipeline.super_train_pipeline.TrainPipeline")
@patch("src.pipeline.super_train_pipeline.HPOPipeline")
def test_super_train_pipeline_run_skip_hpo_with_existing_hpo_config(
    mock_hpo_cls: MagicMock,
    mock_train_cls: MagicMock,
    mock_export_cls: MagicMock,
    mock_from_yaml: MagicMock,
    mock_app_config: AppConfig,
    tmp_path: Path,
) -> None:
    """Verify HPO is skipped and best HPO config is loaded when file exists."""
    best_config_file = Path(mock_app_config.hpo.best_config_path)
    best_config_file.parent.mkdir(parents=True, exist_ok=True)
    best_config_file.write_text("dummy: content", encoding="utf-8")

    mock_override_config = MagicMock(spec=AppConfig)
    mock_from_yaml.return_value = mock_override_config

    expected_onnx_path = tmp_path / "exported.onnx"
    mock_export_cls.return_value.run.return_value = expected_onnx_path

    pipeline = SuperTrainPipeline(mock_app_config, skip_hpo=True)
    result_path = pipeline.run()

    assert result_path == expected_onnx_path
    mock_hpo_cls.assert_not_called()
    mock_from_yaml.assert_called_once_with(best_config_file)
    mock_train_cls.assert_called_once_with(mock_override_config)
    mock_train_cls.return_value.run.assert_called_once()
    mock_export_cls.assert_called_once_with(
        mock_override_config, output_path=pipeline.output_path
    )


@patch("src.pipeline.super_train_pipeline.AppConfig.from_yaml")
@patch("src.pipeline.super_train_pipeline.ONNXExportPipeline")
@patch("src.pipeline.super_train_pipeline.TrainPipeline")
@patch("src.pipeline.super_train_pipeline.HPOPipeline")
def test_super_train_pipeline_run_skip_hpo_without_existing_hpo_config(
    mock_hpo_cls: MagicMock,
    mock_train_cls: MagicMock,
    mock_export_cls: MagicMock,
    mock_from_yaml: MagicMock,
    mock_app_config: AppConfig,
    tmp_path: Path,
) -> None:
    """Verify fallback to base config when skip_hpo is True but file is missing."""
    best_config_file = Path(mock_app_config.hpo.best_config_path)
    if best_config_file.exists():
        best_config_file.unlink()

    expected_onnx_path = tmp_path / "exported.onnx"
    mock_export_cls.return_value.run.return_value = expected_onnx_path

    pipeline = SuperTrainPipeline(mock_app_config, skip_hpo=True)
    result_path = pipeline.run()

    assert result_path == expected_onnx_path
    mock_hpo_cls.assert_not_called()
    mock_from_yaml.assert_not_called()
    mock_train_cls.assert_called_once_with(mock_app_config)
    mock_train_cls.return_value.run.assert_called_once()
    mock_export_cls.assert_called_once_with(
        mock_app_config, output_path=pipeline.output_path
    )


@patch("src.pipeline.super_train_pipeline.HPOPipeline")
def test_super_train_pipeline_reraises_recyclenet_exception(
    mock_hpo_cls: MagicMock,
    mock_app_config: AppConfig,
) -> None:
    """Verify RecycleNetException is re-raised directly without wrapping."""
    original_error = RecycleNetException("Direct domain failure in HPO")
    mock_hpo_cls.return_value.run.side_effect = original_error

    pipeline = SuperTrainPipeline(mock_app_config, skip_hpo=False)

    with pytest.raises(RecycleNetException) as exc_info:
        pipeline.run()

    assert exc_info.value is original_error


@patch("src.pipeline.super_train_pipeline.HPOPipeline")
def test_super_train_pipeline_wraps_generic_exception(
    mock_hpo_cls: MagicMock,
    mock_app_config: AppConfig,
) -> None:
    """Verify generic Exception is wrapped in RecycleNetException with chaining."""
    generic_error = RuntimeError("Low-level OS socket error")
    mock_hpo_cls.return_value.run.side_effect = generic_error

    pipeline = SuperTrainPipeline(mock_app_config, skip_hpo=False)

    with pytest.raises(RecycleNetException) as exc_info:
        pipeline.run()

    assert "Critical failure in super-train pipeline" in str(exc_info.value)
    assert exc_info.value.__cause__ is generic_error
