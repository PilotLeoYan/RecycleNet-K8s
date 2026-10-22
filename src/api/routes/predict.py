"""Prediction endpoint for RecycleNet API.

Accepts an image either as a multipart ``UploadFile`` (``file``) **or** as a
base64-encoded string (``image_base64``) in a form field. Optional telemetry
metadata (GPS coordinates) can be included.
"""

from __future__ import annotations

import base64
import logging

import torch
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from src.api.schemas import PredictionResponse, PredictionResult, Telemetry
from src.api.service import (
    _DEFAULT_CLASSES,
    ImagePreprocessor,
    InferenceModelProtocol,
    InferenceService,
    TelemetryParser,
)

logger = logging.getLogger("recyclenet.api.predict")

router = APIRouter()

_default_service = InferenceService()


def _preprocess(image_bytes: bytes) -> torch.Tensor:
    """Backward-compatible helper to preprocess image bytes."""
    return _default_service.preprocessor.preprocess(image_bytes)


def _run_inference(
    model: InferenceModelProtocol, input_tensor: torch.Tensor
) -> torch.Tensor:
    """Backward-compatible helper to execute inference."""
    return InferenceService._run_inference(model, input_tensor)


@router.post(
    "",
    response_model=PredictionResponse,
    summary="Run RecycleNet inference on a waste image",
)
@router.post(
    "/",
    response_model=PredictionResponse,
    include_in_schema=False,
)
async def predict(
    request: Request,
    file: UploadFile | None = File(None, description="Image file (multipart upload)"),
    image_base64: str | None = Form(
        None, description="Base64-encoded image (alternative to file upload)"
    ),
    latitude: float | None = Form(None, description="GPS latitude"),
    longitude: float | None = Form(None, description="GPS longitude"),
    telemetry: str | None = Form(
        None,
        description='Optional telemetry JSON: {"latitude": 0, "longitude": 0}',
    ),
) -> PredictionResponse:
    """Classify a waste image and return the predicted category with confidence.

    Send either a multipart ``file`` **or** a ``image_base64`` form field.
    GPS coordinates are optional. Supports both ``/predict`` and ``/predict/``.
    """
    model: InferenceModelProtocol | None = getattr(request.app.state, "model", None)
    model_ready: bool = getattr(request.app.state, "model_ready", False)
    idx_to_class: dict[int, str] = getattr(
        request.app.state,
        "idx_to_class",
        _DEFAULT_CLASSES.copy(),
    )

    if not model_ready or model is None:
        raise HTTPException(
            status_code=503, detail="Model not loaded – service unavailable"
        )

    if file is not None:
        image_bytes = await file.read()
    elif image_base64:
        try:
            image_bytes = base64.b64decode(image_base64)
        except Exception as exc:
            raise HTTPException(
                status_code=400, detail="Invalid base64 image data"
            ) from exc
    else:
        raise HTTPException(
            status_code=400,
            detail="No image provided – send a 'file' or 'image_base64'",
        )

    service: InferenceService = (
        getattr(request.app.state, "service", None) or _default_service
    )

    return await service.predict(
        model=model,
        idx_to_class=idx_to_class,
        image_bytes=image_bytes,
        latitude=latitude,
        longitude=longitude,
        telemetry_str=telemetry,
    )


__all__ = [
    "ImagePreprocessor",
    "InferenceModelProtocol",
    "InferenceService",
    "PredictionResponse",
    "PredictionResult",
    "Telemetry",
    "TelemetryParser",
    "_preprocess",
    "_run_inference",
    "predict",
    "router",
]
