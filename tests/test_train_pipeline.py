"""Unit and integration tests for TrainPipeline."""

from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import torch
from matplotlib.figure import Figure
from PIL import Image

from src.components.model import build_mobilenet_v3
from src.config.schema import (
    AppConfig,
    IngestionConfig,
    TrackingConfig,
    TrainingConfig,
    TransformationConfig,
)
from src.pipeline.train_pipeline import (
    TrainPipeline,
    _log_test_metrics_to_mlflow,
    train_loop_per_worker,
)
from src.utils import resolve_tracking_uri
from src.utils.exception import RecycleNetException


@pytest.fixture
def mock_app_config(tmp_path: Path) -> AppConfig:
    """Create a minimal mock AppConfig for pipeline testing."""
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
            epochs=2,
            num_workers=1,
            learning_rate=0.001,
            weight_decay=0.0001,
            device="cpu",
        ),
        tracking=TrackingConfig(
            experiment_name="test_train_exp",
            tracking_uri=f"sqlite:///{tmp_path}/mlflow.db",
        ),
    )


def test_train_pipeline_initialization(mock_app_config: AppConfig) -> None:
    """Test TrainPipeline initializes components properly."""
    pipeline = TrainPipeline(mock_app_config)
    assert pipeline.config == mock_app_config
    assert pipeline.logmodel is not None
    assert pipeline.ingestion is not None
    assert pipeline.transformation is not None


def test_train_pipeline_run_success(
    mock_app_config: AppConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test TrainPipeline.run completes end-to-end with mocked TorchTrainer."""
    pipeline = TrainPipeline(mock_app_config)

    dataset_dir = tmp_path / "extracted_data"
    cardboard_dir = dataset_dir / "cardboard"
    glass_dir = dataset_dir / "glass"
    cardboard_dir.mkdir(parents=True)
    glass_dir.mkdir(parents=True)

    dummy_img = Image.new("RGB", (16, 16), color="red")
    dummy_img.save(cardboard_dir / "img1.png")
    dummy_img.save(glass_dir / "img1.png")

    monkeypatch.setattr(pipeline.ingestion, "extract_dataset", lambda: dataset_dir)

    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir(parents=True)
    dummy_model = build_mobilenet_v3(2)
    torch.save({"model_state": dummy_model.state_dict()}, ckpt_dir / "model.pt")

    class MockCheckpoint:
        @contextmanager
        def as_directory(self) -> Generator[str]:
            yield str(ckpt_dir)

    metrics_df = pd.DataFrame(
        [
            {
                "train_loss": 0.5,
                "valid_loss": 0.4,
                "valid_accuracy": 0.8,
                "valid_precision": 0.8,
                "valid_recall": 0.8,
                "valid_f1_score": 0.8,
            },
            {
                "train_loss": 0.3,
                "valid_loss": 0.25,
                "valid_accuracy": 0.9,
                "valid_precision": 0.9,
                "valid_recall": 0.9,
                "valid_f1_score": 0.9,
            },
            {
                "test_loss": 0.22,
                "test_accuracy": 0.92,
                "test_precision": 0.92,
                "test_recall": 0.92,
                "test_f1_score": 0.92,
                "auc": 0.95,
                "confusion_matrix": None,
            },
        ]
    )

    mock_result = MagicMock()
    mock_result.checkpoint = MockCheckpoint()
    mock_result.metrics_dataframe = metrics_df

    mock_logmodel = MagicMock()
    mock_run = MagicMock()
    mock_run.info.run_id = "test-run-123"

    @contextmanager
    def run_context() -> Generator[MagicMock]:
        yield mock_run

    mock_logmodel.start_run.return_value = run_context()
    pipeline.logmodel = mock_logmodel

    with patch(
        "src.pipeline.train_pipeline.TorchTrainer.fit",
        return_value=mock_result,
    ):
        pipeline.run()

    # Verify MLflow interactions
    assert mock_logmodel.log_pipeline_metadata.called
    _, meta_kwargs = mock_logmodel.log_pipeline_metadata.call_args
    assert "scheduler" in meta_kwargs["params"]
    assert "min_lr" in meta_kwargs["params"]
    assert "warmup_epochs" in meta_kwargs["params"]
    assert "split_distribution" in meta_kwargs
    # Epoch/test metrics are logged by the rank-0 worker, not the driver.
    assert not mock_logmodel.log_epoch.called
    assert mock_logmodel.register_checkpoint.called


def test_train_pipeline_run_no_checkpoint_raises(
    mock_app_config: AppConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test TrainPipeline.run raises exception when no checkpoint is generated."""
    pipeline = TrainPipeline(mock_app_config)

    dataset_dir = tmp_path / "extracted_data"
    cardboard_dir = dataset_dir / "cardboard"
    cardboard_dir.mkdir(parents=True)
    dummy_img = Image.new("RGB", (16, 16), color="red")
    dummy_img.save(cardboard_dir / "img1.png")
    monkeypatch.setattr(pipeline.ingestion, "extract_dataset", lambda: dataset_dir)

    mock_result = MagicMock()
    mock_result.checkpoint = None

    with patch(
        "src.pipeline.train_pipeline.TorchTrainer.fit",
        return_value=mock_result,
    ):
        with pytest.raises(RecycleNetException, match="valid checkpoint"):
            pipeline.run()


def test_train_loop_per_worker_execution(tmp_path: Path) -> None:
    """Test train_loop_per_worker executes with mocked Ray and data loader."""
    config: dict[str, Any] = {
        "reproducibility": MagicMock(
            torch_seed=42, random_seed=42, numpy_seed=42, deterministic=False
        ),
        "transformation": MagicMock(
            image_size=(16, 16), batch_size=2, num_workers=0, pin_memory=False
        ),
        "raw_path": tmp_path,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "epochs": 1,
    }

    dummy_loader = MagicMock()
    dummy_loader.dataset = [1, 2]

    with (
        patch("ray.train.torch.get_device", return_value="cpu"),
        patch("src.pipeline.train_pipeline.DataTransformation") as mock_dt,
        patch("ray.train.torch.prepare_data_loader", side_effect=lambda x: x),
        patch("ray.train.torch.prepare_model", side_effect=lambda x: x),
        patch("src.pipeline.train_pipeline.TrainStep") as mock_ts,
        patch("src.pipeline.train_pipeline.TestDistributed") as mock_td,
        patch("ray.train.report") as mock_report,
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

        td_instance = MagicMock()
        td_instance.compute_distributed_test_metrics.return_value = {
            "test_loss": 0.3,
            "test_accuracy": 0.9,
            "auc": 0.95,
        }
        mock_td.return_value = td_instance

        train_loop_per_worker(config)

        assert mock_report.call_count == 2
        final_report = mock_report.call_args_list[-1].args[0]
        assert "history" in final_report
        assert len(final_report["history"]) == 1
        assert "learning_rate" in final_report["history"][0]
        assert final_report["history"][0]["learning_rate"] == 0.001


def test_train_pipeline_run_trainer_fit_exception_raises(
    mock_app_config: AppConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test TrainPipeline.run raises RecycleNetException when TorchTrainer.fit fails."""
    pipeline = TrainPipeline(mock_app_config)

    dataset_dir = tmp_path / "extracted_data"
    cardboard_dir = dataset_dir / "cardboard"
    cardboard_dir.mkdir(parents=True)
    dummy_img = Image.new("RGB", (16, 16), color="red")
    dummy_img.save(cardboard_dir / "img1.png")
    monkeypatch.setattr(pipeline.ingestion, "extract_dataset", lambda: dataset_dir)

    with patch(
        "src.pipeline.train_pipeline.TorchTrainer.fit",
        side_effect=RuntimeError("Ray cluster connection failed"),
    ):
        with pytest.raises(RecycleNetException, match="Failure during the distributed"):
            pipeline.run()


def test_train_loop_per_worker_early_stopping(tmp_path: Path) -> None:
    """Test train_loop_per_worker stops early when valid_loss does not improve."""
    config: dict[str, Any] = {
        "reproducibility": MagicMock(
            torch_seed=42, random_seed=42, numpy_seed=42, deterministic=False
        ),
        "transformation": MagicMock(
            image_size=(16, 16), batch_size=2, num_workers=0, pin_memory=False
        ),
        "raw_path": tmp_path,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "epochs": 10,
        "patience": 2,
    }

    dummy_loader = MagicMock()
    dummy_loader.dataset = [1, 2]

    with (
        patch("ray.train.torch.get_device", return_value="cpu"),
        patch("src.pipeline.train_pipeline.DataTransformation") as mock_dt,
        patch("ray.train.torch.prepare_data_loader", side_effect=lambda x: x),
        patch("ray.train.torch.prepare_model", side_effect=lambda x: x),
        patch("src.pipeline.train_pipeline.TrainStep") as mock_ts,
        patch("src.pipeline.train_pipeline.TestDistributed") as mock_td,
        patch("ray.train.report") as mock_report,
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
        # Epoch 0: valid_loss=0.5 (best)
        # Epoch 1: valid_loss=0.6 (patience_counter=1)
        # Epoch 2: valid_loss=0.7 (patience_counter=2 >= patience -> stops!)
        ts_instance.train_valid_step.side_effect = [
            {
                "train_loss": 0.5,
                "valid_loss": 0.5,
                "valid_accuracy": 0.8,
                "valid_f1_score": 0.8,
            },
            {
                "train_loss": 0.45,
                "valid_loss": 0.6,
                "valid_accuracy": 0.78,
                "valid_f1_score": 0.78,
            },
            {
                "train_loss": 0.4,
                "valid_loss": 0.7,
                "valid_accuracy": 0.75,
                "valid_f1_score": 0.75,
            },
        ]
        mock_ts.return_value = ts_instance

        td_instance = MagicMock()
        td_instance.compute_distributed_test_metrics.return_value = {
            "test_loss": 0.48,
            "test_accuracy": 0.82,
            "auc": 0.90,
        }
        mock_td.return_value = td_instance

        train_loop_per_worker(config)

        # 3 epochs executed (not 10), plus 1 final evaluation report = 4 reports
        assert ts_instance.train_valid_step.call_count == 3
        assert mock_report.call_count == 4
        final_report_arg = mock_report.call_args_list[-1].args[0]
        assert len(final_report_arg["history"]) == 3
        assert all("learning_rate" in r for r in final_report_arg["history"])


def test_train_loop_per_worker_plateau_scheduler(tmp_path: Path) -> None:
    """Test train_loop_per_worker with ReduceLROnPlateau scheduler."""
    config: dict[str, Any] = {
        "reproducibility": MagicMock(
            torch_seed=42, random_seed=42, numpy_seed=42, deterministic=False
        ),
        "transformation": MagicMock(
            image_size=(16, 16), batch_size=2, num_workers=0, pin_memory=False
        ),
        "raw_path": tmp_path,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "epochs": 1,
        "scheduler": "plateau",
        "min_lr": 1e-5,
        "patience": 2,
    }

    dummy_loader = MagicMock()
    dummy_loader.dataset = [1, 2]

    with (
        patch("ray.train.torch.get_device", return_value="cpu"),
        patch("src.pipeline.train_pipeline.DataTransformation") as mock_dt,
        patch("ray.train.torch.prepare_data_loader", side_effect=lambda x: x),
        patch("ray.train.torch.prepare_model", side_effect=lambda x: x),
        patch("src.pipeline.train_pipeline.TrainStep") as mock_ts,
        patch("src.pipeline.train_pipeline.TestDistributed") as mock_td,
        patch("ray.train.report") as mock_report,
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

        td_instance = MagicMock()
        td_instance.compute_distributed_test_metrics.return_value = {
            "test_loss": 0.3,
            "test_accuracy": 0.9,
            "auc": 0.95,
        }
        mock_td.return_value = td_instance

        train_loop_per_worker(config)

        assert mock_report.call_count == 2
        final_report = mock_report.call_args_list[-1].args[0]
        assert "history" in final_report
        assert final_report["history"][0]["learning_rate"] == 0.001


def test_log_test_metrics_to_mlflow_logs_all_metrics() -> None:
    """Worker helper logs test metrics, per-class JSON and confusion matrix."""
    fig = Figure()
    client = MagicMock()
    per_class = {"glass": {"precision": 0.9, "recall": 0.8, "f1_score": 0.85}}
    _log_test_metrics_to_mlflow(
        client,
        "run-1",
        {
            "test_loss": 0.2,
            "test_accuracy": 0.9,
            "auc": 0.95,
            "per_class_metrics": per_class,
            "confusion_matrix": fig,
        },
    )

    logged = {
        c.kwargs["key"]: c.kwargs["value"] for c in client.log_metric.call_args_list
    }
    assert logged["test_loss"] == 0.2
    assert logged["test_roc"] == 0.95
    assert logged["test_f1_glass"] == 0.85
    client.log_dict.assert_called_once_with(
        "run-1", per_class, "per_class_metrics.json"
    )
    client.log_figure.assert_called_once()


def test_resolve_tracking_uri_makes_sqlite_path_absolute() -> None:
    """Relative SQLite URIs become absolute so Ray workers hit the same DB."""
    uri = resolve_tracking_uri("sqlite:///mlflow.db")
    assert uri == f"sqlite:///{Path('mlflow.db').resolve()}"
    assert resolve_tracking_uri("http://host:5000") == "http://host:5000"
    assert resolve_tracking_uri("sqlite:////abs/m.db") == "sqlite:////abs/m.db"
