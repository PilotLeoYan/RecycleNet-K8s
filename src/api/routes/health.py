"""Health and metadata endpoints for RecycleNet API.

Provides liveness (``/healthz``), readiness (``/ready``), general health (``/health``),
and service/model metadata (``/info``) endpoints.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from src.api.schemas import HealthResponse, InfoResponse
from src.api.service import InferenceService

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    responses={503: {"model": HealthResponse}},
    tags=["health"],
    summary="General health check",
)
async def health(request: Request) -> JSONResponse:
    """General health check endpoint reporting service and model readiness."""
    model_ready: bool = getattr(request.app.state, "model_ready", False)
    status_code = 200 if model_ready else 503
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "healthy" if model_ready else "degraded",
            "model_ready": model_ready,
        },
    )


@router.get("/healthz", tags=["health"], summary="Liveness probe")
async def healthz() -> dict[str, str]:
    """Kubernetes liveness probe.

    Returns 200 as long as the process is running.
    """
    return {"status": "ok"}


@router.get("/ready", tags=["health"], summary="Readiness probe")
async def ready(request: Request) -> JSONResponse:
    """Kubernetes readiness probe.

    Returns 200 only when the RecycleNet model is loaded and ready to serve
    inference requests. Returns 503 otherwise so Kubernetes stops routing
    traffic to the pod until it is ready.
    """
    model_ready: bool = getattr(request.app.state, "model_ready", False)

    if model_ready:
        return JSONResponse(
            status_code=200,
            content={"status": "ready", "model": "loaded"},
        )
    return JSONResponse(
        status_code=503,
        content={"status": "not_ready", "model": "loading"},
    )


@router.get(
    "/info",
    response_model=InfoResponse,
    tags=["info"],
    summary="Service and model metadata",
)
async def info(request: Request) -> InfoResponse:
    """Retrieve service and model metadata including architecture and classes."""
    service: InferenceService = (
        getattr(request.app.state, "service", None) or InferenceService()
    )
    return service.get_info(request.app.state)


__all__ = ["health", "healthz", "info", "ready", "router"]
