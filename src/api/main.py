"""FastAPI entry point for RecycleNet API."""

from fastapi import FastAPI

from .dependencies import lifespan
from .routes import health, predict

app = FastAPI(lifespan=lifespan)

app.include_router(predict.router, prefix="/predict", tags=["predict"])
app.include_router(health.router, tags=["health"])
