"""Pydantic schemas and contract definitions for the RecycleNet API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Telemetry(BaseModel):
    """Geographic coordinates captured during waste image submission."""

    latitude: float | None = Field(
        default=None, description="Latitude in decimal degrees"
    )
    longitude: float | None = Field(
        default=None, description="Longitude in decimal degrees"
    )


class PredictionResponse(BaseModel):
    """Inference response returning top-1 class, confidence, and telemetry."""

    predicted_class: str = Field(..., description="Top-1 predicted waste category")
    confidence: float = Field(
        ..., description="Softmax confidence of the top-1 class (0–1)"
    )
    class_probabilities: dict[str, float] = Field(
        ..., description="Softmax probability for every class"
    )
    telemetry: Telemetry | None = Field(
        default=None, description="GPS metadata if provided"
    )


# Backward-compatible alias for PredictionResponse
PredictionResult = PredictionResponse


class InfoResponse(BaseModel):
    """Metadata response describing service, model architecture, and state."""

    service_name: str = "RecycleNet-API"
    version: str = "0.1.0"
    git_commit: str = Field(..., description="Git commit hash")
    model_architecture: str = "MobileNetV3"
    model_format: str = "ONNX"
    opset_version: int = 18
    input_resolution: list[int] = Field(default_factory=lambda: [224, 224])
    classes: list[str] = Field(..., description="List of recognized class labels")
    model_ready: bool = True


class HealthResponse(BaseModel):
    """Health check status response."""

    status: str = Field(..., description="Service health status")
    model_ready: bool = Field(..., description="Model readiness indicator")


__all__ = [
    "HealthResponse",
    "InfoResponse",
    "PredictionResponse",
    "PredictionResult",
    "Telemetry",
]
