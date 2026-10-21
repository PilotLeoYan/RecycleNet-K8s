"""FastAPI dependencies for RecycleNet API.

Loads the RecycleNet (MobileNetV3-Small) ONNX model into memory once during
the application lifespan so that all inference requests share the same in-memory
session without paying the cost of loading model files on every request.

The model is loaded using ONNX Runtime. An ``OnnxModelWrapper`` provides a
callable interface ``model(tensor) -> logits_tensor`` ensuring backwards
compatibility with the predict route while offering fast, lightweight ONNX
inference.

Model resolution order:
1. Path in the ``RECYCLENET_MODEL_PATH`` environment variable (if exists).
2. Local file ``models/recyclenet.onnx``.
"""

import json
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from fastapi import FastAPI

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
async def lifespan(app: FastAPI) -> AsyncIterator[None]:  # noqa: ANN001
    """Load the RecycleNet ONNX model at startup and release it on shutdown."""
    try:
        model_path = _resolve_model_path()
        model, idx_to_class = _load_onnx_session(model_path)

        app.state.model = model
        app.state.idx_to_class = idx_to_class
        app.state.model_ready = True
        logger.info("RecycleNet ONNX model ready. Classes: %s", idx_to_class)

    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load ONNX model: %s", exc)
        app.state.model = None
        app.state.idx_to_class = _DEFAULT_CLASSES.copy()
        app.state.model_ready = False

    yield

    # ── Cleanup ──────────────────────────────────────────────────────────────
    app.state.model = None
    app.state.model_ready = False
    logger.info("RecycleNet model released.")
