from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pytest
from torch import nn

from src.components.log_model import LogModel


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
