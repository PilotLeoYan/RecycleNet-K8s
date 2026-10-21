"""Unit and integration tests for RecycleNet FastAPI service.

Verifies health probes (/healthz, /ready, /health) and inference endpoint (/predict)
covering normal operations, error handling, telemetry, and payload validation.
"""

from __future__ import annotations

import base64
import io
from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.api.main import app
from src.components.model import build_mobilenet_v3


def _generate_test_image_bytes(
    color: tuple[int, int, int] = (120, 180, 240),
    size: tuple[int, int] = (224, 224),
    fmt: str = "JPEG",
) -> bytes:
    """Generate in-memory dummy image bytes for testing."""
    image = Image.new("RGB", size, color=color)
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return buffer.getvalue()


@pytest.fixture
def client_with_model() -> Generator[TestClient]:
    """TestClient fixture with an initialized MobileNetV3 test model in app state."""
    model = build_mobilenet_v3(num_classes=6, freeze_base=False)
    model.eval()

    app.state.model = model
    app.state.model_ready = True
    app.state.idx_to_class = {
        0: "cardboard",
        1: "glass",
        2: "metal",
        3: "paper",
        4: "plastic",
        5: "trash",
    }

    with TestClient(app) as test_client:
        app.state.model = model
        app.state.model_ready = True
        app.state.idx_to_class = {
            0: "cardboard",
            1: "glass",
            2: "metal",
            3: "paper",
            4: "plastic",
            5: "trash",
        }
        yield test_client

    app.state.model = None
    app.state.model_ready = False


@pytest.fixture
def client_without_model() -> Generator[TestClient]:
    """TestClient fixture simulating a degraded pod where model is not yet loaded."""
    with TestClient(app) as test_client:
        app.state.model = None
        app.state.model_ready = False
        yield test_client


# ─────────────────────────────────────────────────────────────────────────────
# Health Probes Tests
# ─────────────────────────────────────────────────────────────────────────────


def test_liveness_probe(client_with_model: TestClient) -> None:
    """Liveness probe /healthz should always return 200 OK."""
    response = client_with_model.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readiness_probe_ready(client_with_model: TestClient) -> None:
    """Readiness probe /ready should return 200 when model is loaded."""
    response = client_with_model.get("/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert data["model"] == "loaded"


def test_readiness_probe_not_ready(client_without_model: TestClient) -> None:
    """Readiness probe /ready should return 503 when model is not loaded."""
    response = client_without_model.get("/ready")
    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "not_ready"
    assert data["model"] == "loading"


def test_general_health_endpoint_healthy(client_with_model: TestClient) -> None:
    """General health endpoint /health should report healthy when model is ready."""
    response = client_with_model.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_ready"] is True


def test_general_health_endpoint_degraded(client_without_model: TestClient) -> None:
    """Endpoint /health should report 503 degraded when model is missing."""
    response = client_without_model.get("/health")
    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "degraded"
    assert data["model_ready"] is False


# ─────────────────────────────────────────────────────────────────────────────
# Predict Endpoint Tests
# ─────────────────────────────────────────────────────────────────────────────


def test_predict_service_unavailable_when_model_missing(
    client_without_model: TestClient,
) -> None:
    """Prediction requests should return 503 if model is not loaded."""
    image_bytes = _generate_test_image_bytes()
    response = client_without_model.post(
        "/predict/",
        files={"file": ("test.jpg", image_bytes, "image/jpeg")},
    )
    assert response.status_code == 503
    assert "Model not loaded" in response.json()["detail"]


def test_predict_no_image_provided(client_with_model: TestClient) -> None:
    """Prediction should return 400 when neither file nor base64 is sent."""
    response = client_with_model.post("/predict/", data={})
    assert response.status_code == 400
    assert "No image provided" in response.json()["detail"]


def test_predict_invalid_image_bytes(client_with_model: TestClient) -> None:
    """Prediction should return 400 when uploaded file cannot be decoded as an image."""
    corrupt_bytes = b"not_an_actual_image_content"
    response = client_with_model.post(
        "/predict/",
        files={"file": ("corrupt.jpg", corrupt_bytes, "image/jpeg")},
    )
    assert response.status_code == 400
    assert "Cannot decode image" in response.json()["detail"]


def test_predict_invalid_base64_string(client_with_model: TestClient) -> None:
    """Prediction should return 400 when base64 string is malformed."""
    response = client_with_model.post(
        "/predict/",
        data={"image_base64": "invalid_base64_content!!!"},
    )
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "Invalid base64 image data" in detail or "Cannot decode image" in detail


def test_predict_success_multipart_file(client_with_model: TestClient) -> None:
    """Prediction via multipart file should return valid classification response."""
    image_bytes = _generate_test_image_bytes()
    response = client_with_model.post(
        "/predict/",
        files={"file": ("waste.jpg", image_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()

    assert data["predicted_class"] in [
        "cardboard",
        "glass",
        "metal",
        "paper",
        "plastic",
        "trash",
    ]
    assert 0.0 <= data["confidence"] <= 1.0
    assert isinstance(data["class_probabilities"], dict)
    assert len(data["class_probabilities"]) == 6
    assert data["telemetry"] is None

    # Check softmax sum property
    total_prob = sum(data["class_probabilities"].values())
    assert pytest.approx(total_prob, rel=1e-2) == 1.0


def test_predict_success_base64_and_telemetry(client_with_model: TestClient) -> None:
    """Prediction via base64 encoded image and GPS coordinates."""
    image_bytes = _generate_test_image_bytes(color=(200, 50, 50))
    b64_str = base64.b64encode(image_bytes).decode("utf-8")

    response = client_with_model.post(
        "/predict/",
        data={
            "image_base64": b64_str,
            "latitude": "20.659698",
            "longitude": "-103.325997",
        },
    )
    assert response.status_code == 200
    data = response.json()

    assert data["predicted_class"] in [
        "cardboard",
        "glass",
        "metal",
        "paper",
        "plastic",
        "trash",
    ]
    assert 0.0 <= data["confidence"] <= 1.0
    assert data["telemetry"] is not None
    assert pytest.approx(data["telemetry"]["latitude"], rel=1e-4) == 20.6597
    assert pytest.approx(data["telemetry"]["longitude"], rel=1e-4) == -103.3260


def test_predict_multipart_with_json_telemetry_string(
    client_with_model: TestClient,
) -> None:
    """Prediction via multipart file and JSON telemetry string (as sent by Swagger)."""
    image_bytes = _generate_test_image_bytes(color=(100, 200, 100))
    response = client_with_model.post(
        "/predict/",
        files={"file": ("cardboard16.jpg", image_bytes, "image/jpeg")},
        data={"telemetry": '{"latitude": 20.659698, "longitude": -103.325997}'},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_class"] in [
        "cardboard",
        "glass",
        "metal",
        "paper",
        "plastic",
        "trash",
    ]
    assert data["telemetry"] is not None
    assert pytest.approx(data["telemetry"]["latitude"], rel=1e-4) == 20.6597
    assert pytest.approx(data["telemetry"]["longitude"], rel=1e-4) == -103.3260
