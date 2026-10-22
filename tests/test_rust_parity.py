"""Automated Train-Serve Skew & Numerical Parity Test Suite.

Validates:
1. Rust microservice structure (`rust_inference/`) and `Dockerfile.rust`.
2. Preprocessing numerical parity between PyTorch and Rust (< 1e-5).
3. End-to-end inference numerical parity (< 1e-4 tolerance).
"""

from __future__ import annotations

import base64
import io
import json
import os
import shutil
import socket
import subprocess
import time
from collections.abc import Generator
from pathlib import Path

import httpx
import numpy as np
import onnxruntime as ort
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.api.main import app
from src.api.service import ImagePreprocessor
from src.components.model import build_mobilenet_v3
from src.pipeline.export_onnx import export_to_onnx

RUST_DIR = Path("rust_inference")
CARGO_TOML = RUST_DIR / "Cargo.toml"
DOCKERFILE_RUST = Path("Dockerfile.rust")
MODEL_PATH = Path("models/recyclenet.onnx")
CLASSES_MAPPING_PATH = Path("models/classes_mapping.json")

WINDOWS_CARGO_PATH = r"C:\Users\laban\.cargo\bin;C:\Program Files\CodeBlocks\MinGW\bin"


def _get_free_port() -> int:
    """Acquire an ephemeral unused TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _is_cargo_available() -> bool:
    """Check if cargo is available in system PATH or standard Windows location."""
    if shutil.which("cargo") is not None:
        return True
    if os.name == "nt":
        for dir_str in WINDOWS_CARGO_PATH.split(";"):
            if (Path(dir_str) / "cargo.exe").exists():
                return True
    return False


def _resolve_ort_dylib() -> str | None:
    """Find ONNX Runtime C dynamic library from Python package or system paths."""
    if "ORT_DYLIB_PATH" in os.environ:
        p = Path(os.environ["ORT_DYLIB_PATH"])
        if p.exists():
            return str(p.resolve())

    # Check onnxruntime python package capi directory
    try:
        capi_dir = Path(ort.__file__).parent / "capi"
        if capi_dir.exists():
            for pattern in [
                "onnxruntime.dll",
                "libonnxruntime.so*",
                "libonnxruntime*.dylib",
            ]:
                candidates = list(capi_dir.glob(pattern))
                if candidates:
                    return str(candidates[0].resolve())
    except Exception:
        pass

    # Fallback Linux system paths
    for path_str in [
        "/usr/lib/libonnxruntime.so",
        "/usr/local/lib/libonnxruntime.so",
        "/usr/lib/x86_64-linux-gnu/libonnxruntime.so",
    ]:
        p = Path(path_str)
        if p.exists():
            return str(p.resolve())

    return None


def _resolve_rust_binary() -> Path:
    """Find or build the compiled Rust microservice binary."""
    bin_name = "rust_inference.exe" if os.name == "nt" else "rust_inference"
    release_bin = RUST_DIR / "target" / "release" / bin_name
    debug_bin = RUST_DIR / "target" / "debug" / bin_name

    if release_bin.exists():
        return release_bin.resolve()
    if debug_bin.exists():
        return debug_bin.resolve()

    if not _is_cargo_available():
        pytest.skip("Cargo toolchain not available")

    env = os.environ.copy()
    if os.name == "nt":
        env["PATH"] = f"{WINDOWS_CARGO_PATH};{env.get('PATH', '')}"

    dylib = _resolve_ort_dylib()
    if dylib:
        env["ORT_DYLIB_PATH"] = dylib

    subprocess.run(
        ["cargo", "build", "--release", "--manifest-path", str(CARGO_TOML)],
        env=env,
        check=True,
    )
    return release_bin.resolve()


@pytest.fixture(scope="module")
def onnx_model_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Resolve ONNX model path or dynamically export a small model for CI."""
    if MODEL_PATH.exists():
        model_p = MODEL_PATH.resolve()
    else:
        tmp_dir = tmp_path_factory.mktemp("ci_models")
        model_p = tmp_dir / "recyclenet.onnx"
        mapping_p = tmp_dir / "classes_mapping.json"

        default_mapping = {
            str(i): c
            for i, c in enumerate(
                ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
            )
        }
        mapping_p.write_text(json.dumps(default_mapping), encoding="utf-8")

        model = build_mobilenet_v3(num_classes=6, size="small", freeze_base=True)
        export_to_onnx(model, (1, 3, 224, 224), model_p)

    os.environ["RECYCLENET_MODEL_PATH"] = str(model_p)
    return model_p


@pytest.fixture(scope="module")
def rust_server(onnx_model_path: Path) -> Generator[str]:
    """Launch the Rust Axum inference server on an ephemeral port for testing."""
    bin_name = "rust_inference.exe" if os.name == "nt" else "rust_inference"
    has_bin = (RUST_DIR / "target" / "release" / bin_name).exists() or (
        RUST_DIR / "target" / "debug" / bin_name
    ).exists()

    if not _is_cargo_available() and not has_bin:
        pytest.skip("Cargo toolchain not available")

    port = _get_free_port()
    rust_binary = _resolve_rust_binary()

    env = os.environ.copy()
    env["PORT"] = str(port)
    env["RECYCLENET_MODEL_PATH"] = str(onnx_model_path.resolve())
    if os.name == "nt":
        env["PATH"] = f"{WINDOWS_CARGO_PATH};{env.get('PATH', '')}"

    dylib = _resolve_ort_dylib()
    if dylib:
        env["ORT_DYLIB_PATH"] = dylib

    proc = subprocess.Popen(
        [str(rust_binary)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    base_url = f"http://127.0.0.1:{port}"
    deadline = time.time() + 15.0
    ready = False

    while time.time() < deadline:
        try:
            resp = httpx.get(f"{base_url}/healthz", timeout=1.0)
            if resp.status_code == 200:
                ready = True
                break
        except Exception:
            time.sleep(0.3)

    if not ready:
        proc.kill()
        out, err = proc.communicate(timeout=3)
        msg = (
            f"Rust server failed to start within 15s. "
            f"stdout: {out.decode()}, stderr: {err.decode()}"
        )
        raise RuntimeError(msg)

    try:
        yield base_url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


@pytest.fixture(scope="module")
def python_client(onnx_model_path: Path) -> Generator[TestClient]:
    """FastAPI TestClient with lifespan context managing ONNX model in state."""
    # Ensure app state is reset so lifespan initializes fresh model from environment
    app.state.model = None
    app.state.model_ready = False
    app.state.service = None

    with TestClient(app) as client:
        yield client


def _create_test_image_bytes(
    width: int = 224, height: int = 224, pattern: str = "noise"
) -> bytes:
    """Generate in-memory test image bytes in PNG format."""
    rng = np.random.default_rng(seed=42)
    if pattern == "noise":
        arr = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    elif pattern == "black":
        arr = np.zeros((height, width, 3), dtype=np.uint8)
    elif pattern == "white":
        arr = np.full((height, width, 3), 255, dtype=np.uint8)
    elif pattern == "gradient":
        arr = np.zeros((height, width, 3), dtype=np.uint8)
        for y in range(height):
            for x in range(width):
                arr[y, x, 0] = int(y / height * 255)
                arr[y, x, 1] = int(x / width * 255)
                arr[y, x, 2] = int((x + y) / (width + height) * 255)
    else:
        arr = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)

    img = Image.fromarray(arr)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


class TestRustMicroserviceStructure:
    """Validate project structure and Dockerfile security gates."""

    def test_rust_files_exist(self) -> None:
        """Verify Rust microservice files exist."""
        assert CARGO_TOML.exists(), "Cargo.toml must exist"
        assert (RUST_DIR / "src" / "main.rs").exists(), "src/main.rs must exist"
        assert (RUST_DIR / "src" / "engine.rs").exists(), "src/engine.rs must exist"
        assert (RUST_DIR / "src" / "preprocess.rs").exists(), (
            "src/preprocess.rs must exist"
        )
        assert (RUST_DIR / "src" / "lib.rs").exists(), "src/lib.rs must exist"

    def test_cargo_toml_dependencies(self) -> None:
        """Verify Cargo.toml includes all required high-performance crates."""
        content = CARGO_TOML.read_text(encoding="utf-8")
        assert "ort" in content, "Cargo.toml must depend on ort"
        assert "axum" in content, "Cargo.toml must depend on axum"
        assert "tokio" in content, "Cargo.toml must depend on tokio"
        assert "ndarray" in content, "Cargo.toml must depend on ndarray"
        assert "image" in content, "Cargo.toml must depend on image"
        assert "serde" in content, "Cargo.toml must depend on serde"

    def test_dockerfile_rust_security_and_multistage(self) -> None:
        """Verify Dockerfile.rust implements multi-stage builds and zero-trust user."""
        assert DOCKERFILE_RUST.exists(), "Dockerfile.rust must exist"
        content = DOCKERFILE_RUST.read_text(encoding="utf-8")

        # Multi-stage verification
        assert "AS builder" in content, "Dockerfile.rust must include builder stage"
        assert "AS runtime" in content, "Dockerfile.rust must include runtime stage"

        # Non-root unprivileged security verification (UID 10001, appuser)
        assert "10001" in content, "Dockerfile.rust must configure UID 10001"
        assert "appuser" in content, "Dockerfile.rust must create appuser"
        assert "USER appuser" in content, (
            "Dockerfile.rust must drop privileges with USER appuser"
        )

        # Port configuration
        assert "EXPOSE 8080" in content, "Dockerfile.rust must expose port 8080"
        assert "ENTRYPOINT" in content, "Dockerfile.rust must define an ENTRYPOINT"


class TestPreprocessingParity:
    """Verify Train-Serve Skew check: Python vs Rust preprocessing parity."""

    @pytest.mark.parametrize("pattern", ["noise", "black", "white", "gradient"])
    def test_preprocessing_numerical_parity_224x224(
        self, pattern: str, tmp_path: Path
    ) -> None:
        """Ensure Python and Rust ImagePreprocessor produce identical tensors."""
        image_bytes = _create_test_image_bytes(224, 224, pattern=pattern)

        # 1. Python preprocessing
        py_preprocessor = ImagePreprocessor()
        py_tensor = py_preprocessor.preprocess(image_bytes).numpy()

        # 2. Rust preprocessing via compiled CLI dump
        rust_binary = _resolve_rust_binary()
        img_file = tmp_path / f"test_{pattern}.png"
        out_bin = tmp_path / f"out_{pattern}.bin"
        img_file.write_bytes(image_bytes)

        subprocess.run(
            [str(rust_binary), "--dump-preprocessed", str(img_file), str(out_bin)],
            check=True,
        )

        rust_tensor = np.fromfile(out_bin, dtype="<f4").reshape((1, 3, 224, 224))

        # 3. Assert strict numerical parity (< 1e-5)
        max_abs_diff = float(np.max(np.abs(py_tensor - rust_tensor)))
        assert max_abs_diff < 1e-5, (
            f"Train-serve skew in pattern '{pattern}'! Max diff: {max_abs_diff}"
        )


class TestEndToEndParity:
    """Verify inference parity between Python FastAPI and Rust Axum services."""

    def test_health_probes_parity(
        self, python_client: TestClient, rust_server: str
    ) -> None:
        """Verify /health, /healthz, and /ready probes match schemas and responses."""
        # /healthz (liveness)
        py_healthz = python_client.get("/healthz")
        rust_healthz = httpx.get(f"{rust_server}/healthz")
        assert py_healthz.status_code == 200
        assert rust_healthz.status_code == 200
        assert rust_healthz.json() == py_healthz.json() == {"status": "ok"}

        # /ready (readiness)
        py_ready = python_client.get("/ready")
        rust_ready = httpx.get(f"{rust_server}/ready")
        assert py_ready.status_code == 200
        assert rust_ready.status_code == 200
        assert (
            rust_ready.json()
            == py_ready.json()
            == {
                "status": "ready",
                "model": "loaded",
            }
        )

        # /health (general health)
        py_health = python_client.get("/health")
        rust_health = httpx.get(f"{rust_server}/health")
        assert py_health.status_code == 200
        assert rust_health.status_code == 200
        assert py_health.json()["status"] == rust_health.json()["status"] == "healthy"
        assert (
            py_health.json()["model_ready"] is rust_health.json()["model_ready"] is True
        )

    def test_info_metadata_parity(
        self, python_client: TestClient, rust_server: str
    ) -> None:
        """Verify /info endpoint metadata parity."""
        py_info = python_client.get("/info").json()
        rust_info = httpx.get(f"{rust_server}/info").json()

        assert py_info["service_name"] == rust_info["service_name"] == "RecycleNet-API"
        assert py_info["version"] == rust_info["version"] == "0.1.0"
        assert (
            py_info["model_architecture"]
            == rust_info["model_architecture"]
            == "MobileNetV3"
        )
        assert py_info["model_format"] == rust_info["model_format"] == "ONNX"
        assert py_info["opset_version"] == rust_info["opset_version"] == 18
        assert (
            py_info["input_resolution"] == rust_info["input_resolution"] == [224, 224]
        )
        assert py_info["classes"] == rust_info["classes"]
        assert py_info["model_ready"] is rust_info["model_ready"] is True

    def test_predict_multipart_numerical_parity(
        self, python_client: TestClient, rust_server: str
    ) -> None:
        """Verify multipart image prediction matches within 1e-4 tolerance."""
        image_bytes = _create_test_image_bytes(224, 224, pattern="noise")

        # Query Python API
        py_resp = python_client.post(
            "/predict",
            files={"file": ("test.png", image_bytes, "image/png")},
        )
        assert py_resp.status_code == 200
        py_data = py_resp.json()

        # Query Rust API
        rust_resp = httpx.post(
            f"{rust_server}/predict",
            files={"file": ("test.png", image_bytes, "image/png")},
        )
        assert rust_resp.status_code == 200
        rust_data = rust_resp.json()

        # 1. Top-1 Predicted Class Match
        assert rust_data["predicted_class"] == py_data["predicted_class"], (
            f"Class mismatch! Python={py_data['predicted_class']}, "
            f"Rust={rust_data['predicted_class']}"
        )

        # 2. Confidence Numerical Tolerance (< 1e-4)
        conf_diff = abs(py_data["confidence"] - rust_data["confidence"])
        assert conf_diff < 1e-4, f"Confidence difference too large: {conf_diff}"

        # 3. Class Probabilities Tolerance (< 1e-4) for all classes
        assert set(py_data["class_probabilities"].keys()) == set(
            rust_data["class_probabilities"].keys()
        )
        for class_name, py_prob in py_data["class_probabilities"].items():
            rust_prob = rust_data["class_probabilities"][class_name]
            prob_diff = abs(py_prob - rust_prob)
            assert prob_diff < 1e-4, (
                f"Probability diff for '{class_name}' too large: {prob_diff} "
                f"(Python={py_prob}, Rust={rust_prob})"
            )

    def test_predict_base64_parity(
        self, python_client: TestClient, rust_server: str
    ) -> None:
        """Verify prediction via image_base64 form field matches between services."""
        image_bytes = _create_test_image_bytes(224, 224, pattern="gradient")
        b64_str = base64.b64encode(image_bytes).decode("utf-8")

        py_resp = python_client.post(
            "/predict",
            data={"image_base64": b64_str},
        )
        rust_resp = httpx.post(
            f"{rust_server}/predict",
            data={"image_base64": b64_str},
        )

        assert py_resp.status_code == 200
        assert rust_resp.status_code == 200
        py_data = py_resp.json()
        rust_data = rust_resp.json()

        assert rust_data["predicted_class"] == py_data["predicted_class"]
        assert abs(py_data["confidence"] - rust_data["confidence"]) < 1e-4

    def test_predict_telemetry_parity(
        self, python_client: TestClient, rust_server: str
    ) -> None:
        """Verify telemetry GPS coordinates are parsed and returned identically."""
        image_bytes = _create_test_image_bytes(224, 224, pattern="white")

        # Explicit coordinates
        py_resp = python_client.post(
            "/predict",
            files={"file": ("white.png", image_bytes, "image/png")},
            data={"latitude": "37.7749", "longitude": "-122.4194"},
        ).json()

        rust_resp = httpx.post(
            f"{rust_server}/predict",
            files={"file": ("white.png", image_bytes, "image/png")},
            data={"latitude": "37.7749", "longitude": "-122.4194"},
        ).json()

        assert py_resp["telemetry"] == rust_resp["telemetry"]
        assert rust_resp["telemetry"]["latitude"] == pytest.approx(37.7749)
        assert rust_resp["telemetry"]["longitude"] == pytest.approx(-122.4194)

        # JSON telemetry string
        tel_json = json.dumps({"latitude": 40.7128, "longitude": -74.006})
        py_resp_json = python_client.post(
            "/predict",
            files={"file": ("white.png", image_bytes, "image/png")},
            data={"telemetry": tel_json},
        ).json()

        rust_resp_json = httpx.post(
            f"{rust_server}/predict",
            files={"file": ("white.png", image_bytes, "image/png")},
            data={"telemetry": tel_json},
        ).json()

        assert py_resp_json["telemetry"] == rust_resp_json["telemetry"]
        assert rust_resp_json["telemetry"]["latitude"] == pytest.approx(40.7128)
        assert rust_resp_json["telemetry"]["longitude"] == pytest.approx(-74.006)

    def test_predict_trailing_slash_alias_parity(
        self, python_client: TestClient, rust_server: str
    ) -> None:
        """Verify /predict/ with trailing slash works on both Python and Rust."""
        image_bytes = _create_test_image_bytes(224, 224, pattern="black")

        py_resp = python_client.post(
            "/predict/",
            files={"file": ("black.png", image_bytes, "image/png")},
        )
        rust_resp = httpx.post(
            f"{rust_server}/predict/",
            files={"file": ("black.png", image_bytes, "image/png")},
        )

        assert py_resp.status_code == 200
        assert rust_resp.status_code == 200
        assert py_resp.json()["predicted_class"] == rust_resp.json()["predicted_class"]

    def test_predict_error_handling_parity(
        self, python_client: TestClient, rust_server: str
    ) -> None:
        """Verify edge case error handling returns 400 Bad Request."""
        # No image provided
        py_no_img = python_client.post("/predict")
        rust_no_img = httpx.post(f"{rust_server}/predict")
        assert py_no_img.status_code == 400
        assert rust_no_img.status_code == 400
        assert "detail" in rust_no_img.json()

        # Invalid base64 data
        py_invalid = python_client.post(
            "/predict", data={"image_base64": "!!!not_base64!!!"}
        )
        rust_invalid = httpx.post(
            f"{rust_server}/predict", data={"image_base64": "!!!not_base64!!!"}
        )
        assert py_invalid.status_code == 400
        assert rust_invalid.status_code == 400
        assert "detail" in rust_invalid.json()
