"""Unit and integration tests for RecycleNet FastAPI service.

Verifies health probes (/healthz, /ready, /health), metadata (/info), and
inference endpoint (/predict) covering normal operations, error handling,
telemetry, schema contracts, and service components.
"""

from __future__ import annotations

import base64
import io
from collections.abc import Generator
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from fastapi import HTTPException
from fastapi.testclient import TestClient
from PIL import Image

from src.api.main import app
from src.api.schemas import (
    HealthResponse,
    InfoResponse,
    PredictionResponse,
    PredictionResult,
    Telemetry,
)
from src.api.service import (
    ImagePreprocessor,
    InferenceService,
    OnnxModelWrapper,
    TelemetryParser,
    get_git_commit,
)
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
    app.state.service = InferenceService()
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
        app.state.service = InferenceService()
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
    app.state.service = None


@pytest.fixture
def client_without_model() -> Generator[TestClient]:
    """TestClient fixture simulating a degraded pod where model is not yet loaded."""
    with TestClient(app) as test_client:
        app.state.model = None
        app.state.model_ready = False
        app.state.service = None
        yield test_client


# ─────────────────────────────────────────────────────────────────────────────
# Health & Info Probes Tests
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


def test_info_endpoint_model_ready(client_with_model: TestClient) -> None:
    """Metadata endpoint /info should report service details when model is ready."""
    response = client_with_model.get("/info")
    assert response.status_code == 200
    data = response.json()

    assert data["service_name"] == "RecycleNet-API"
    assert data["version"] == "0.1.0"
    assert isinstance(data["git_commit"], str)
    assert len(data["git_commit"]) > 0
    assert data["model_architecture"] == "MobileNetV3"
    assert data["model_format"] == "ONNX"
    assert data["opset_version"] == 18
    assert data["input_resolution"] == [224, 224]
    assert data["classes"] == [
        "cardboard",
        "glass",
        "metal",
        "paper",
        "plastic",
        "trash",
    ]
    assert data["model_ready"] is True


def test_info_endpoint_model_not_loaded(client_without_model: TestClient) -> None:
    """Metadata endpoint /info should report model_ready as False when degraded."""
    response = client_without_model.get("/info")
    assert response.status_code == 200
    data = response.json()

    assert data["service_name"] == "RecycleNet-API"
    assert data["version"] == "0.1.0"
    assert data["model_ready"] is False
    assert len(data["classes"]) == 6


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


def test_predict_without_trailing_slash_no_redirect(
    client_with_model: TestClient,
) -> None:
    """POST /predict (no trailing slash) returns 200 directly without 307 redirect."""
    image_bytes = _generate_test_image_bytes()
    response = client_with_model.post(
        "/predict",
        files={"file": ("waste.jpg", image_bytes, "image/jpeg")},
        follow_redirects=False,
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


# ─────────────────────────────────────────────────────────────────────────────
# Schema Contracts Tests
# ─────────────────────────────────────────────────────────────────────────────


def test_schema_telemetry() -> None:
    """Verify Telemetry model defaults and explicit values."""
    tel_default = Telemetry()
    assert tel_default.latitude is None
    assert tel_default.longitude is None

    tel_custom = Telemetry(latitude=20.65, longitude=-103.32)
    assert tel_custom.latitude == 20.65
    assert tel_custom.longitude == -103.32


def test_schema_prediction_response_and_alias() -> None:
    """Verify PredictionResponse fields and PredictionResult backward compatibility."""
    assert PredictionResult is PredictionResponse

    resp = PredictionResponse(
        predicted_class="plastic",
        confidence=0.987,
        class_probabilities={"plastic": 0.987, "metal": 0.013},
        telemetry=Telemetry(latitude=1.0, longitude=2.0),
    )
    assert resp.predicted_class == "plastic"
    assert resp.confidence == 0.987
    assert resp.class_probabilities["plastic"] == 0.987
    assert resp.telemetry is not None
    assert resp.telemetry.latitude == 1.0


def test_schema_info_response() -> None:
    """Verify InfoResponse defaults and explicit constructor values."""
    info = InfoResponse(
        git_commit="abcdef1",
        classes=["cardboard", "glass", "metal", "paper", "plastic", "trash"],
    )
    assert info.service_name == "RecycleNet-API"
    assert info.version == "0.1.0"
    assert info.git_commit == "abcdef1"
    assert info.model_architecture == "MobileNetV3"
    assert info.model_format == "ONNX"
    assert info.opset_version == 18
    assert info.input_resolution == [224, 224]
    assert info.model_ready is True


def test_schema_health_response() -> None:
    """Verify HealthResponse structure and types."""
    health = HealthResponse(status="healthy", model_ready=True)
    assert health.status == "healthy"
    assert health.model_ready is True


# ─────────────────────────────────────────────────────────────────────────────
# Service Layer Component Tests
# ─────────────────────────────────────────────────────────────────────────────


def test_telemetry_parser() -> None:
    """Verify TelemetryParser handling of json, raw lat/lon, invalid inputs."""
    # Valid JSON string
    t1 = TelemetryParser.parse(telemetry='{"latitude": 15.5, "longitude": -90.5}')
    assert t1 is not None
    assert t1.latitude == 15.5
    assert t1.longitude == -90.5

    # Direct coordinates
    t2 = TelemetryParser.parse(latitude=10.0, longitude=20.0)
    assert t2 is not None
    assert t2.latitude == 10.0
    assert t2.longitude == 20.0

    # Malformed JSON with fallback to explicit coordinates
    t3 = TelemetryParser.parse(
        telemetry="invalid_json_str\r\n", latitude=12.0, longitude=34.0
    )
    assert t3 is not None
    assert t3.latitude == 12.0
    assert t3.longitude == 34.0

    # None inputs
    t4 = TelemetryParser.parse()
    assert t4 is None


def test_image_preprocessor_success_and_error() -> None:
    """Verify ImagePreprocessor output tensor shape/dtype and error handling."""
    preprocessor = ImagePreprocessor()
    valid_bytes = _generate_test_image_bytes()

    tensor = preprocessor.preprocess(valid_bytes)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 3, 224, 224)
    assert tensor.dtype == torch.float32

    # Callable shorthand
    tensor_call = preprocessor(valid_bytes)
    assert tensor_call.shape == (1, 3, 224, 224)

    # Corrupt image bytes should raise HTTPException 400
    with pytest.raises(HTTPException) as exc_info:
        preprocessor.preprocess(b"corrupted_image_data")
    assert exc_info.value.status_code == 400
    assert "Cannot decode image" in exc_info.value.detail


def test_onnx_model_wrapper() -> None:
    """Verify OnnxModelWrapper inference session interaction and eval() no-op."""
    mock_session = MagicMock()
    mock_input = MagicMock()
    mock_input.name = "input_tensor"
    mock_output = MagicMock()
    mock_output.name = "output_tensor"

    mock_session.get_inputs.return_value = [mock_input]
    mock_session.get_outputs.return_value = [mock_output]
    mock_session.run.return_value = [np.array([[1.0, 2.0, 3.0]], dtype=np.float32)]

    wrapper = OnnxModelWrapper(mock_session)
    wrapper.eval()  # No-op check

    # Input as torch.Tensor
    x_tensor = torch.zeros(1, 3, 224, 224)
    out_tensor = wrapper(x_tensor)
    assert isinstance(out_tensor, torch.Tensor)
    assert out_tensor.shape == (1, 3)

    # Input as np.ndarray
    x_np = np.zeros((1, 3, 224, 224), dtype=np.float32)
    out_np = wrapper(x_np)
    assert isinstance(out_np, torch.Tensor)
    assert out_np.shape == (1, 3)


@pytest.mark.anyio
async def test_inference_service_predict_and_error() -> None:
    """Verify InferenceService predict execution and missing model guard."""
    service = InferenceService()

    # Model missing should raise 503
    with pytest.raises(HTTPException) as exc_info:
        await service.predict(
            model=None,
            idx_to_class={0: "cardboard"},
            image_bytes=_generate_test_image_bytes(),
        )
    assert exc_info.value.status_code == 503

    # With dummy PyTorch model
    model = build_mobilenet_v3(num_classes=6, freeze_base=False)
    model.eval()
    idx_to_class = {
        0: "cardboard",
        1: "glass",
        2: "metal",
        3: "paper",
        4: "plastic",
        5: "trash",
    }
    img_bytes = _generate_test_image_bytes()

    result = await service.predict(
        model=model,
        idx_to_class=idx_to_class,
        image_bytes=img_bytes,
        latitude=19.43,
        longitude=-99.13,
    )
    assert isinstance(result, PredictionResponse)
    assert result.predicted_class in idx_to_class.values()
    assert result.telemetry is not None
    assert result.telemetry.latitude == 19.43


def test_get_git_commit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify get_git_commit reads env var or falls back gracefully."""
    monkeypatch.setenv("GIT_COMMIT", "test_commit_hash")
    assert get_git_commit() == "test_commit_hash"

    monkeypatch.delenv("GIT_COMMIT", raising=False)
    commit = get_git_commit()
    assert isinstance(commit, str)
    assert len(commit) > 0
