from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import numpy as np
import pytest
from torch import nn

from src.components.log_model import LogModel
from src.config.schema import TrackingConfig


def test_log_model_no_active_run() -> None:
    logger_model = LogModel()

    # When no active run exists, methods should return safely without raising errors
    logger_model.log_epoch(
        train_loss=0.5,
        valid_loss=0.4,
        valid_metrics={
            "accuracy": 0.8,
            "precision": 0.8,
            "recall": 0.8,
            "f1_score": 0.8,
        },
        step=0,
    )

    fig, _ = plt.subplots()
    logger_model.log_test(
        roc=0.9,
        metrics={
            "accuracy": 0.8,
            "precision": 0.8,
            "recall": 0.8,
            "f1_score": 0.8,
        },
        fig_cm=fig,
    )

    dummy_input = np.random.randn(1, 3, 32, 32).astype(np.float32)
    dummy_output = np.random.randn(1, 2).astype(np.float32)
    model = nn.Linear(32, 2)
    logger_model.log_model(dummy_input, dummy_output, model)


def test_log_epoch_with_active_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    db_path = tmp_path / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    logger_model = LogModel()

    with mlflow.start_run():
        logger_model.log_epoch(
            train_loss=0.5,
            valid_loss=0.4,
            valid_metrics={
                "accuracy": 0.8,
                "precision": 0.8,
                "recall": 0.8,
                "f1_score": 0.8,
            },
            step=1,
        )

        fig, _ = plt.subplots()
        logger_model.log_test(
            roc=0.85,
            metrics={
                "accuracy": 0.8,
                "precision": 0.8,
                "recall": 0.8,
                "f1_score": 0.8,
            },
            fig_cm=fig,
        )


def test_log_model_registered_model_name_from_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_kwargs: dict[str, object] = {}

    def mock_log_model(**kwargs: object) -> None:
        captured_kwargs.update(kwargs)

    info_obj = type("MockInfo", (), {"run_id": "123"})()
    mock_run = type("MockRun", (), {"info": info_obj})()
    monkeypatch.setattr(mlflow, "active_run", lambda: mock_run)
    monkeypatch.setattr(mlflow.pytorch, "log_model", mock_log_model)

    config = TrackingConfig(registered_model_name="CustomRecycleNet")
    logger = LogModel(config=config)

    assert logger.config.registered_model_name == "CustomRecycleNet"

    dummy_input = np.random.randn(1, 3, 32, 32).astype(np.float32)
    dummy_output = np.random.randn(1, 2).astype(np.float32)
    model = nn.Linear(32, 2)

    logger.log_model(dummy_input, dummy_output, model)
    assert captured_kwargs.get("registered_model_name") == "CustomRecycleNet"


def test_log_model_registered_model_name_dynamic_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_kwargs: dict[str, object] = {}

    def mock_log_model(**kwargs: object) -> None:
        captured_kwargs.update(kwargs)

    info_obj = type("MockInfo", (), {"run_id": "123"})()
    mock_run = type("MockRun", (), {"info": info_obj})()
    monkeypatch.setattr(mlflow, "active_run", lambda: mock_run)
    monkeypatch.setattr(mlflow.pytorch, "log_model", mock_log_model)

    config = TrackingConfig(registered_model_name="BaseModelName")
    logger = LogModel(config=config)

    dummy_input = np.random.randn(1, 3, 32, 32).astype(np.float32)
    dummy_output = np.random.randn(1, 2).astype(np.float32)
    model = nn.Linear(32, 2)

    logger.log_model(
        dummy_input, dummy_output, model, registered_model_name="DynamicOverrideModel"
    )
    assert captured_kwargs.get("registered_model_name") == "DynamicOverrideModel"


def test_log_model_init_override() -> None:
    logger = LogModel(registered_model_name="DirectOverride")
    assert logger.config.registered_model_name == "DirectOverride"
