"""Prediction endpoint for RecycleNet API.

Accepts an image either as a multipart ``UploadFile`` (``file``) **or** as a
base64-encoded string (``image_base64``) in a form field. Optional telemetry
metadata (GPS coordinates) can be included.

The endpoint:
1. Decodes the incoming image bytes using Pillow.
2. Applies the same ImageNet preprocessing used during training
   (resize 224×224, normalize with ImageNet stats).
3. Runs a forward pass through the in-memory MobileNetV3-Small model.
4. Returns the predicted class label, confidence score, and all class
   probabilities alongside any provided telemetry.
"""

import base64
import io
import logging

import torch
import torch.nn.functional as F
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from PIL import Image
from pydantic import BaseModel, Field
from torchvision.transforms import v2

logger = logging.getLogger("recyclenet.api.predict")

router = APIRouter()

# ── Preprocessing pipeline (must match training transforms) ─────────────────
_INFERENCE_TRANSFORM = v2.Compose(
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


class Telemetry(BaseModel):
    latitude: float | None = Field(None, description="Latitude in decimal degrees")
    longitude: float | None = Field(None, description="Longitude in decimal degrees")


class PredictionResult(BaseModel):
    predicted_class: str = Field(..., description="Top-1 predicted waste category")
    confidence: float = Field(
        ..., description="Softmax confidence of the top-1 class (0–1)"
    )
    class_probabilities: dict[str, float] = Field(
        ..., description="Softmax probability for every class"
    )
    telemetry: Telemetry | None = Field(None, description="GPS metadata if provided")


def _preprocess(image_bytes: bytes) -> torch.Tensor:
    """Convert raw image bytes → normalised 4-D tensor ready for the model."""
    try:
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"Cannot decode image: {exc}"
        ) from exc

    tensor: torch.Tensor = _INFERENCE_TRANSFORM(pil_image)  # (3, 224, 224)
    return tensor.unsqueeze(0)  # (1, 3, 224, 224)


@router.post(
    "/",
    response_model=PredictionResult,
    summary="Run RecycleNet inference on a waste image",
)
async def predict(
    request: Request,
    file: UploadFile | None = File(None, description="Image file (multipart upload)"),
    image_base64: str | None = Form(
        None, description="Base64-encoded image (alternative to file upload)"
    ),
    latitude: float | None = Form(None, description="GPS latitude"),
    longitude: float | None = Form(None, description="GPS longitude"),
) -> PredictionResult:
    """Classify a waste image and return the predicted category with confidence.

    Send either a multipart ``file`` **or** a ``image_base64`` form field.
    GPS coordinates are optional.
    """
    # ── Guard: model must be loaded ──────────────────────────────────────────
    model = getattr(request.app.state, "model", None)
    model_ready = getattr(request.app.state, "model_ready", False)
    idx_to_class: dict[int, str] = getattr(
        request.app.state,
        "idx_to_class",
        {0: "cardboard", 1: "glass", 2: "metal", 3: "paper", 4: "plastic", 5: "trash"},
    )

    if not model_ready or model is None:
        raise HTTPException(
            status_code=503, detail="Model not loaded – service unavailable"
        )

    # ── Extract raw image bytes ──────────────────────────────────────────────
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

    # ── Preprocessing ────────────────────────────────────────────────────────
    input_tensor = _preprocess(image_bytes)  # (1, 3, 224, 224)

    # ── Inference ────────────────────────────────────────────────────────────
    with torch.no_grad():
        logits: torch.Tensor = model(input_tensor)  # (1, num_classes)
        probs: torch.Tensor = F.softmax(logits, dim=1)[0]  # (num_classes,)

    # ── Build response ───────────────────────────────────────────────────────
    top1_idx: int = int(probs.argmax().item())
    top1_conf: float = float(probs[top1_idx].item())
    predicted_class: str = idx_to_class.get(top1_idx, f"class_{top1_idx}")

    class_probabilities: dict[str, float] = {
        idx_to_class.get(i, f"class_{i}"): round(float(p), 6)
        for i, p in enumerate(probs.tolist())
    }

    telemetry = (
        Telemetry(latitude=latitude, longitude=longitude)
        if latitude is not None or longitude is not None
        else None
    )

    logger.info(
        "Prediction: %s (%.2f%%)  telemetry=%s",
        predicted_class,
        top1_conf * 100,
        telemetry,
    )

    return PredictionResult(
        predicted_class=predicted_class,
        confidence=round(top1_conf, 6),
        class_probabilities=class_probabilities,
        telemetry=telemetry,
    )
