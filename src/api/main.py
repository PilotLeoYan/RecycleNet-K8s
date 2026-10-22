"""FastAPI entry point for RecycleNet API."""

from __future__ import annotations

from fastapi import FastAPI

from src.api.routes import health, predict
from src.api.service import lifespan

app = FastAPI(
    title="RecycleNet API",
    description="Waste classification service using MobileNetV3 and ONNX Runtime.",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(predict.router, prefix="/predict", tags=["predict"])
app.include_router(health.router, tags=["health"])

__all__ = ["app"]
