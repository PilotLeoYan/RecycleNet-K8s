"""FastAPI dependencies for RecycleNet API.

Loads the RecycleNet (MobileNetV3-Small) PyTorch model into memory once during
the application lifespan so that all inference requests share the same in-memory
model without paying the cost of loading weights on every request.

The model and the class-index→label mapping are attached to ``app.state`` so that
any route can access them via ``request.app.state.model`` and
``request.app.state.idx_to_class``.

Model resolution order
-----------------------
1. Path in the ``RECYCLENET_MODEL_PATH`` environment variable.
2. MLflow Model Registry (latest Production version of "RecycleNet") when
   ``RECYCLENET_MLFLOW_URI`` is set.
3. Falls back to rebuilding the MobileNetV3-Small architecture with random
   weights (useful for local development without a trained checkpoint).
"""

import json
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI

from src.components.model import (
    build_mobilenet_v3,  # import directo, evita el __init__ de components
)

logger = logging.getLogger("recyclenet.api")

# Default TrashNet 6-class label mapping
# (cardboard, glass, metal, paper, plastic, trash)
_DEFAULT_CLASSES: dict[int, str] = {
    0: "cardboard",
    1: "glass",
    2: "metal",
    3: "paper",
    4: "plastic",
    5: "trash",
}


def _load_from_path(
    model_path: Path, num_classes: int
) -> tuple[torch.nn.Module, dict[int, str]]:
    """Load a saved ``state_dict`` from disk into the MobileNetV3-Small architecture."""
    state = torch.load(model_path, map_location="cpu", weights_only=False)
    idx_to_class = _DEFAULT_CLASSES.copy()

    # If the checkpoint includes class metadata, use it
    if isinstance(state, dict):
        if "classes" in state and isinstance(state["classes"], list):
            idx_to_class = {i: c for i, c in enumerate(state["classes"])}
            num_classes = len(idx_to_class)
        elif "class_to_idx" in state and isinstance(state["class_to_idx"], dict):
            idx_to_class = {v: k for k, v in state["class_to_idx"].items()}
            num_classes = len(idx_to_class)

    model = build_mobilenet_v3(num_classes=num_classes, freeze_base=False)

    # Support both bare state_dict and {"model_state_dict": ...} checkpoints.
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    model.load_state_dict(state)
    logger.info("Model loaded from %s with %d classes", model_path, num_classes)
    return model, idx_to_class


def _load_from_mlflow(
    mlflow_uri: str, num_classes: int
) -> tuple[torch.nn.Module, dict[int, str]]:
    """Load the latest Production model from the MLflow Model Registry."""
    import mlflow.pytorch  # lazy import – mlflow is optional

    mlflow.set_tracking_uri(mlflow_uri)
    model_uri = "models:/RecycleNet/Production"
    logger.info("Loading model from MLflow: %s", model_uri)
    model: torch.nn.Module = mlflow.pytorch.load_model(model_uri, map_location="cpu")

    # Try to fetch the classes_mapping.json artifact from the same run.
    idx_to_class = _DEFAULT_CLASSES.copy()
    try:
        client = mlflow.tracking.MlflowClient(tracking_uri=mlflow_uri)
        versions = client.get_latest_versions("RecycleNet", stages=["Production"])
        if versions:
            run_id = versions[0].run_id
            local_path = mlflow.artifacts.download_artifacts(
                run_id=run_id,
                artifact_path="classes_mapping.json",
                tracking_uri=mlflow_uri,
            )
            raw: dict[str, str] = json.loads(Path(local_path).read_text())
            idx_to_class = {int(k): v for k, v in raw.items()}
            logger.info("Class mapping loaded from MLflow artifact: %s", idx_to_class)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Could not load class mapping from MLflow: %s – using defaults", exc
        )

    return model, idx_to_class


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:  # noqa: ANN001
    """Load the RecycleNet model at startup and release it on shutdown.

    Resolution order:
    1. ``RECYCLENET_MODEL_PATH`` env-var  → local ``.pt`` / ``.pth`` checkpoint.
    2. ``RECYCLENET_MLFLOW_URI`` env-var  → MLflow Model Registry (Production).
    3. Fallback: fresh MobileNetV3-Small with random weights (dev mode).
    """
    idx_to_class: dict[int, str] = _DEFAULT_CLASSES.copy()
    num_classes = len(idx_to_class)

    model_path_env = os.getenv("RECYCLENET_MODEL_PATH")
    default_local_path = Path("models/recyclenet_mobilenetv3.pt")
    mlflow_uri_env = os.getenv("RECYCLENET_MLFLOW_URI")

    try:
        if model_path_env and Path(model_path_env).exists():
            model_path = Path(model_path_env)
            model, idx_to_class = _load_from_path(model_path, num_classes)

        elif default_local_path.exists():
            logger.info("Found local model at default path: %s", default_local_path)
            model, idx_to_class = _load_from_path(default_local_path, num_classes)

        elif mlflow_uri_env:
            model, idx_to_class = _load_from_mlflow(mlflow_uri_env, num_classes)
            num_classes = len(idx_to_class)

        else:
            logger.warning(
                "No RECYCLENET_MODEL_PATH or local checkpoint found. "
                "Running with untrained MobileNetV3-Small weights (dev mode)."
            )
            model = build_mobilenet_v3(num_classes=num_classes, freeze_base=False)

        model.eval()
        app.state.model = model
        app.state.idx_to_class = idx_to_class
        app.state.model_ready = True
        logger.info("RecycleNet model ready. Classes: %s", idx_to_class)

    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load model: %s", exc)
        app.state.model = None
        app.state.idx_to_class = idx_to_class
        app.state.model_ready = False

    yield

    # ── Cleanup ──────────────────────────────────────────────────────────────
    app.state.model = None
    app.state.model_ready = False
    logger.info("RecycleNet model released.")
