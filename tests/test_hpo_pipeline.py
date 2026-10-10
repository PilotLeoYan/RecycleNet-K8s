"""Unit and integration tests for HPOPipeline."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.config.schema import (
    AppConfig,
    HPOConfig,
    IngestionConfig,
    TrackingConfig,
    TrainingConfig,
    TransformationConfig,
)
from src.pipeline.hpo_pipeline import HPOPipeline, train_eval_trial
from src.utils.exception import RecycleNetException


@pytest.fixture
def mock_app_config(tmp_path: Path) -> AppConfig:
    """Create a minimal mock AppConfig for HPO testing."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "cardboard").mkdir()
    (raw_dir / "glass").mkdir()

    zip_file = tmp_path / "dataset.zip"
    zip_file.touch()

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
            dataloader_n_workers=1,
            cpu_resources_per_trial=1,
            gpu_resources_per_trial=0.0,
            device="cpu",
        ),
        tracking=TrackingConfig(
            experiment_name="test_hpo_exp",
            tracking_uri=f"sqlite:///{tmp_path}/mlflow.db",
        ),
    )


def test_hpo_pipeline_initialization(mock_app_config: AppConfig) -> None:
    """Test HPOPipeline initializes components properly."""
    pipeline = HPOPipeline(mock_app_config)
    assert pipeline.config == mock_app_config
    assert pipeline.ingestion is not None


def test_hpo_pipeline_run_success(
    mock_app_config: AppConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test HPOPipeline.run executes end-to-end with mocked Tuner."""
    pipeline = HPOPipeline(mock_app_config)

    dataset_dir = tmp_path / "extracted_data"
    (dataset_dir / "cardboard").mkdir(parents=True)
    (dataset_dir / "glass").mkdir(parents=True)
    monkeypatch.setattr(pipeline.ingestion, "extract_dataset", lambda: dataset_dir)

    mock_result_grid = MagicMock()
    mock_result_grid.errors = []
    mock_best_result = MagicMock()
    mock_best_result.config = {"learning_rate": 0.001, "batch_size": 8}
    mock_best_result.metrics = {"valid_loss": 0.35, "valid_accuracy": 0.88}
    mock_result_grid.get_best_result.return_value = mock_best_result

    mock_tuner = MagicMock()
    mock_tuner.fit.return_value = mock_result_grid

    with patch("src.pipeline.hpo_pipeline.tune.Tuner", return_value=mock_tuner):
        best_config, best_metrics = pipeline.run()

    assert best_config == {"learning_rate": 0.001, "batch_size": 8}
    assert best_metrics == {"valid_loss": 0.35, "valid_accuracy": 0.88}


def test_hpo_pipeline_run_with_errors_raises(
    mock_app_config: AppConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test HPOPipeline.run raises RecycleNetException when trial execution fails."""
    pipeline = HPOPipeline(mock_app_config)

    dataset_dir = tmp_path / "extracted_data"
    (dataset_dir / "cardboard").mkdir(parents=True)
    monkeypatch.setattr(pipeline.ingestion, "extract_dataset", lambda: dataset_dir)

    mock_result_grid = MagicMock()
    mock_result_grid.errors = [RuntimeError("Trial failed due to OOM")]
    mock_result_grid.num_errors = 1

    mock_tuner = MagicMock()
    mock_tuner.fit.return_value = mock_result_grid

    with patch("src.pipeline.hpo_pipeline.tune.Tuner", return_value=mock_tuner):
        with pytest.raises(RecycleNetException):
            pipeline.run()


def test_train_eval_trial_execution(mock_app_config: AppConfig, tmp_path: Path) -> None:
    """Test train_eval_trial execution with mocked components."""
    trial_config = {
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "batch_size": 2,
        "dataloader_n_workers": 0,
        "device": "cpu",
        "max_epochs": 1,
    }

    dummy_loader = MagicMock()
    dummy_loader.dataset = [1, 2]

    with (
        patch("src.pipeline.hpo_pipeline.DataTransformation") as mock_dt,
        patch("src.pipeline.hpo_pipeline.TrainStep") as mock_ts,
        patch("src.pipeline.hpo_pipeline.tune.report") as mock_report,
    ):
        dt_instance = MagicMock()
        dt_instance.classes = ["c1", "c2"]
        dt_instance.get_dataloaders.return_value = (
            dummy_loader,
            dummy_loader,
            dummy_loader,
        )
        mock_dt.return_value = dt_instance

        ts_instance = MagicMock()
        ts_instance.train_valid_step.return_value = {
            "train_loss": 0.5,
            "valid_loss": 0.4,
            "valid_accuracy": 0.8,
            "valid_f1_score": 0.8,
        }
        mock_ts.return_value = ts_instance

        train_eval_trial(trial_config, data_dir=tmp_path, app_config=mock_app_config)

        assert mock_report.call_count == 1


def test_hpo_pipeline_run_no_best_result_raises(
    mock_app_config: AppConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test HPOPipeline.run raises RecycleNetException when get_best_result is None."""
    pipeline = HPOPipeline(mock_app_config)

    dataset_dir = tmp_path / "extracted_data"
    (dataset_dir / "cardboard").mkdir(parents=True)
    monkeypatch.setattr(pipeline.ingestion, "extract_dataset", lambda: dataset_dir)

    mock_result_grid = MagicMock()
    mock_result_grid.errors = []
    mock_result_grid.get_best_result.return_value = None

    mock_tuner = MagicMock()
    mock_tuner.fit.return_value = mock_result_grid

    with patch("src.pipeline.hpo_pipeline.tune.Tuner", return_value=mock_tuner):
        with pytest.raises(RecycleNetException):
            pipeline.run()
