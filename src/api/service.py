"""Service layer and inference orchestration for RecycleNet API.

Follows SOLID principles:
- Single Responsibility: discrete classes for preprocessing, telemetry,
  and inference.
- Open/Closed & Dependency Inversion: abstract InferenceModelProtocol
  allows interchangeable model backends (ONNX Runtime, PyTorch, etc.).
"""

from __future__ import annotations

import io
import json
import logging
import os
import subprocess  # nosec B404
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Protocol

import anyio
import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from PIL import Image
from torchvision.transforms import v2

from src.api.schemas import InfoResponse, PredictionResponse, Telemetry

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

DEFAULT_LOCAL_ONNX = Path("models/recyclenet.onnx")


class InferenceModelProtocol(Protocol):
    """Protocol defining callable interface for model inference."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Run forward inference given a batch tensor and return logits."""
        ...


class OnnxModelWrapper:
    """Callable wrapper around an ONNX Runtime InferenceSession.

    Accepts a PyTorch Tensor or NumPy array of shape ``(B, 3, 224, 224)``,
    runs inference via ONNX Runtime, and returns a PyTorch Tensor of logits
    ``(B, num_classes)``.
    """

    def __init__(self, session: ort.InferenceSession) -> None:
        self.session = session
        self.input_name: str = session.get_inputs()[0].name
        self.output_name: str = session.get_outputs()[0].name

    def __call__(self, x: torch.Tensor | np.ndarray) -> torch.Tensor:
        """Execute forward inference."""
        if isinstance(x, torch.Tensor):
            x_np = x.detach().cpu().numpy().astype(np.float32)
        else:
            x_np = np.asarray(x, dtype=np.float32)

        raw_outputs: list[np.ndarray] = self.session.run(
            [self.output_name], {self.input_name: x_np}
        )
        return torch.from_numpy(raw_outputs[0])

    def eval(self) -> None:
        """No-op for compatibility with PyTorch model interface."""


class ImagePreprocessor:
    """Encapsulates image decoding and normalization transformations."""

    def __init__(self) -> None:
        self.transform = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Resize(size=(224, 224)),
                v2.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

    def preprocess(self, image_bytes: bytes) -> torch.Tensor:
        """Convert raw image bytes into a normalized 4D tensor ready for inference.

        Raises HTTPException(status_code=400) if image decoding fails or format
        is unsupported.
        """
        try:
            raw_img = Image.open(io.BytesIO(image_bytes))
            img_format = (raw_img.format or "").upper()
            if img_format not in {"JPEG", "PNG", "JPG", "MPO"}:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Unsupported image format: {raw_img.format}. "
                        "Only JPEG and PNG are supported."
                    ),
                )
            pil_image = raw_img.convert("RGB")
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=400, detail=f"Cannot decode image: {exc}"
            ) from exc

        tensor: torch.Tensor = self.transform(pil_image)
        return tensor.unsqueeze(0)

    def __call__(self, image_bytes: bytes) -> torch.Tensor:
        """Callable shorthand for preprocess."""
        return self.preprocess(image_bytes)


class TelemetryParser:
    """Helper for parsing and sanitizing telemetry data from JSON or parameters."""

    @staticmethod
    def parse(
        telemetry: str | None = None,
        latitude: float | None = None,
        longitude: float | None = None,
    ) -> Telemetry | None:
        """Parse telemetry coordinates from JSON string or explicit coordinates.

        Sanitizes JSON inputs against CRLF log injection.
        """
        telemetry_obj: Telemetry | None = None
        if telemetry and telemetry.strip():
            try:
                raw_tel = json.loads(telemetry)
                if isinstance(raw_tel, dict):
                    lat_raw = raw_tel.get("latitude")
                    lon_raw = raw_tel.get("longitude")
                    lat = float(lat_raw) if lat_raw is not None else None
                    lon = float(lon_raw) if lon_raw is not None else None
                    telemetry_obj = Telemetry(latitude=lat, longitude=lon)
            except Exception as exc:
                safe_telemetry = (
                    telemetry.replace("\r", "").replace("\n", "")
                    if telemetry is not None
                    else None
                )
                logger.warning(
                    "Could not parse telemetry JSON: %s (%s)", safe_telemetry, exc
                )

        if telemetry_obj is None and (latitude is not None or longitude is not None):
            telemetry_obj = Telemetry(latitude=latitude, longitude=longitude)

        return telemetry_obj


def get_git_commit() -> str:
    """Retrieve the current Git commit hash from environment or git command."""
    env_commit = os.getenv("GIT_COMMIT")
    if env_commit:
        return env_commit.strip()

    try:
        result = subprocess.run(  # nosec B603 B607
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=2,
        )
        commit = result.stdout.strip()
        if commit:
            return commit
    except Exception as exc:
        logger.debug("Could not resolve git commit via subprocess: %s", exc)

    return "unknown"


class InferenceService:
    """Orchestrates image preprocessing, inference execution, and responses."""

    def __init__(
        self,
        preprocessor: ImagePreprocessor | None = None,
        telemetry_parser: TelemetryParser | None = None,
    ) -> None:
        self.preprocessor = preprocessor or ImagePreprocessor()
        self.telemetry_parser = telemetry_parser or TelemetryParser()

    @staticmethod
    def _run_inference(
        model: InferenceModelProtocol, input_tensor: torch.Tensor
    ) -> torch.Tensor:
        """Execute model inference in worker thread under torch.inference_mode."""
        with torch.inference_mode():
            logits: torch.Tensor = model(input_tensor)
            return F.softmax(logits, dim=1)[0]

    async def predict(
        self,
        model: InferenceModelProtocol | None,
        idx_to_class: dict[int, str] | None,
        image_bytes: bytes,
        latitude: float | None = None,
        longitude: float | None = None,
        telemetry_str: str | None = None,
        telemetry: str | None = None,
    ) -> PredictionResponse:
        """Run asynchronous inference with offloaded execution and telemetry parsing."""
        if model is None:
            raise HTTPException(
                status_code=503, detail="Model not loaded – service unavailable"
            )

        mapping = idx_to_class if idx_to_class is not None else _DEFAULT_CLASSES

        input_tensor = self.preprocessor.preprocess(image_bytes)
        probs: torch.Tensor = await anyio.to_thread.run_sync(
            self._run_inference, model, input_tensor
        )

        top1_idx: int = int(probs.argmax().item())
        top1_conf: float = float(probs[top1_idx].item())
        predicted_class: str = mapping.get(top1_idx, f"class_{top1_idx}")

        class_probabilities: dict[str, float] = {
            mapping.get(i, f"class_{i}"): round(float(p), 6)
            for i, p in enumerate(probs.tolist())
        }

        tel_input = telemetry_str if telemetry_str is not None else telemetry
        telemetry_obj = self.telemetry_parser.parse(
            telemetry=tel_input,
            latitude=latitude,
            longitude=longitude,
        )

        logger.info(
            "Prediction: %s (%.2f%%)  telemetry=%s",
            predicted_class,
            top1_conf * 100,
            telemetry_obj,
        )

        return PredictionResponse(
            predicted_class=predicted_class,
            confidence=round(top1_conf, 6),
            class_probabilities=class_probabilities,
            telemetry=telemetry_obj,
        )

    def get_info(
        self,
        app_state: Any = None,
        *,
        model_ready: bool | None = None,
        classes: list[str] | None = None,
    ) -> InfoResponse:
        """Retrieve service metadata, git commit, and model status."""
        if model_ready is None:
            model_ready = (
                getattr(app_state, "model_ready", False)
                if app_state is not None
                else False
            )

        if classes is None:
            idx_to_class = (
                getattr(app_state, "idx_to_class", None)
                if app_state is not None
                else None
            )
            if idx_to_class:
                classes = [name for _, name in sorted(idx_to_class.items())]
            else:
                classes = list(_DEFAULT_CLASSES.values())

        return InfoResponse(
            git_commit=get_git_commit(),
            classes=classes,
            model_ready=model_ready,
        )


def _resolve_model_path() -> Path:
    """Resolve the ONNX model file path."""
    model_path_env = os.getenv("RECYCLENET_MODEL_PATH")
    if model_path_env and Path(model_path_env).exists():
        logger.info("Using model path from RECYCLENET_MODEL_PATH: %s", model_path_env)
        return Path(model_path_env)

    if DEFAULT_LOCAL_ONNX.exists():
        logger.info("Found local ONNX model at %s", DEFAULT_LOCAL_ONNX)
        return DEFAULT_LOCAL_ONNX

    raise FileNotFoundError(
        f"ONNX model not found at {DEFAULT_LOCAL_ONNX} or RECYCLENET_MODEL_PATH."
    )


def _load_class_mapping(model_dir: Path) -> dict[int, str]:
    """Attempt to load class mapping JSON from directory, falling back to defaults."""
    local_mapping = model_dir / "classes_mapping.json"
    if local_mapping.exists():
        try:
            with open(local_mapping, encoding="utf-8") as f:
                data: dict[str, str] = json.load(f)
                return {int(k): v for k, v in data.items()}
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to parse local classes_mapping.json: %s", exc)

    return _DEFAULT_CLASSES.copy()


def _load_onnx_session(model_path: Path) -> tuple[OnnxModelWrapper, dict[int, str]]:
    """Initialize ONNX Runtime inference session with CPU provider."""
    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 2
    opts.intra_op_num_threads = 2
    session = ort.InferenceSession(
        str(model_path),
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )
    wrapper = OnnxModelWrapper(session)
    idx_to_class = _load_class_mapping(model_path.parent)
    logger.info("ONNX model loaded successfully from %s", model_path)
    return wrapper, idx_to_class


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the RecycleNet ONNX model at startup and release it on shutdown."""
    if getattr(app.state, "model", None) is not None:
        logger.info(
            "RecycleNet model already initialized in app state; skipping ONNX load."
        )
        if getattr(app.state, "service", None) is None:
            app.state.service = InferenceService()
        yield
        return

    try:
        model_path = _resolve_model_path()
        model, idx_to_class = _load_onnx_session(model_path)

        app.state.model = model
        app.state.idx_to_class = idx_to_class
        app.state.model_ready = True
        app.state.service = InferenceService()
        logger.info("RecycleNet ONNX model ready. Classes: %s", idx_to_class)

    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load ONNX model: %s", exc)
        app.state.model = None
        app.state.idx_to_class = _DEFAULT_CLASSES.copy()
        app.state.model_ready = False
        app.state.service = InferenceService()

    yield

    # ── Cleanup ──────────────────────────────────────────────────────────────
    app.state.model = None
    app.state.model_ready = False
    app.state.service = None
    logger.info("RecycleNet model released.")


__all__ = [
    "DEFAULT_LOCAL_ONNX",
    "ImagePreprocessor",
    "InferenceModelProtocol",
    "InferenceService",
    "OnnxModelWrapper",
    "TelemetryParser",
    "_DEFAULT_CLASSES",
    "_load_class_mapping",
    "_load_onnx_session",
    "_resolve_model_path",
    "get_git_commit",
    "lifespan",
]
