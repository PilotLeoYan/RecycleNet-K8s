"""FastAPI dependencies and backward-compatible facade for RecycleNet API.

Re-exports model loader components and lifespan context manager from the service layer.
"""

from __future__ import annotations

from src.api.service import (
    _DEFAULT_CLASSES,
    DEFAULT_LOCAL_ONNX,
    InferenceService,
    OnnxModelWrapper,
    _load_class_mapping,
    _load_onnx_session,
    _resolve_model_path,
    lifespan,
)

__all__ = [
    "DEFAULT_LOCAL_ONNX",
    "InferenceService",
    "OnnxModelWrapper",
    "_DEFAULT_CLASSES",
    "_load_class_mapping",
    "_load_onnx_session",
    "_resolve_model_path",
    "lifespan",
]
