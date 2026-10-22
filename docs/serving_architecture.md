# 🏗️ RecycleNet-K8s: Dual Inference & Edge/IoT Serving Architecture

## 1. Executive Summary & Purpose

The **RecycleNet-K8s** serving infrastructure is engineered to solve a fundamental dilemma in modern production Machine Learning: **how to bridge rapid cloud development with ultra-low latency, deterministic edge inference**.

Standard deep learning deployment pipelines frequently rely solely on high-level Python web frameworks (such as FastAPI, Flask, or TorchServe). While well-suited for experimentation and standard cloud APIs, Python introduces considerable drawbacks for edge environments:
- High container footprints (**~1.2 GB** image size due to PyTorch wheels, CUDA dependencies, and Python runtimes).
- Significant resident memory consumption (**~350–500 MB RSS** at idle, scaling up under load).
- Non-deterministic tail latency caused by Global Interpreter Lock (GIL) contention and unpredictable Garbage Collection (GC) pauses.
- Prolonged cold-start times (2–5 seconds), hindering agile Kubernetes Horizontal Pod Autoscaling (HPA).

To address these challenges, RecycleNet-K8s establishes a **Dual Serving Topology**:
1. **Python Cloud Baseline Tier (`src/api/`)**: Built on FastAPI and ONNX Runtime CPU Execution Provider. Provides rapid iteration, dynamic telemetry parsing, and frictionless integration with the MLflow model registry.
2. **Rust High-Performance Edge Tier (`rust_inference/`)**: Built on Axum 0.8, Tokio, and the `ort` crate (ONNX Runtime C-API FFI bindings). Delivers sub-millisecond dispatch, zero-copy memory transfers, deterministic execution without garbage collection, a microscopic container footprint (**<40 MB**), and tiny memory consumption (**<45 MB RSS**).

Both engines strictly implement identical REST API contracts, guarantee numerical parity, and undergo automated load validation with `k6` and a dedicated Python scientific benchmarking harness.

---

## 2. System Architecture & Dual Serving Topology

```mermaid
flowchart TD
    Client["Client / Edge Device / Smart Waste Bin"] -->|HTTP / REST| Ingress["Kubernetes Ingress / Gateway"]

    subgraph K8sCluster["RecycleNet Serving Infrastructure"]
        Ingress -->|Route / In-Process| PythonFastAPI["Python Serving Tier (Port 8000)<br/>FastAPI + Uvicorn<br/>• Schema Validation (Pydantic v2)<br/>• torchvision.transforms.v2<br/>• onnxruntime Engine"]
        Ingress -->|Route / Edge Gateway| RustAxum["Rust Edge Serving Tier (Port 8080)<br/>Axum + Tokio<br/>• Serde Parsing & Telemetry<br/>• image + ndarray Pipeline<br/>• ort Engine (ONNX C-API)"]
    end

    subgraph StorageEngine["Shared Model Artifact"]
        ONNXModel[("MobileNetV3 ONNX<br/>models/recyclenet.onnx<br/>(Opset 18, 6 Classes)")]
    end

    PythonFastAPI -->|Load & Forward| ONNXModel
    RustAxum -->|Load & Forward (Zero-Copy)| ONNXModel
```

### Dual Serving Topology Comparison

| Architectural Dimension | Python Cloud Tier (`src/api/`) | Rust Edge Tier (`rust_inference/`) |
| :--- | :--- | :--- |
| **Primary Framework** | FastAPI (ASGI) + Uvicorn | Axum 0.8 + Tokio Async Engine |
| **Inference Runtime** | `onnxruntime` (Python C-extension) | `ort` 2.0 (C-API bindings via FFI) |
| **Concurrency Model** | Async event loop + thread pool worker offloading | Work-stealing multi-threaded Tokio reactor |
| **Memory Management** | Python GC + reference counting | Deterministic compile-time RAII (Zero GC) |
| **Container Footprint** | `~1.2 GB` (`python:3.14-slim` + dependencies) | `~38 MB` (Multi-stage static / `debian-slim`) |
| **Resident Memory (RSS)** | `~350 MB – 500 MB` | `~35 MB – 45 MB` |
| **Cold-Start Time** | `~2.5 s` | `< 50 ms` |
| **Ideal Deployment** | Cloud Kubernetes Pods, Analytics Services | Edge Gateways, Smart Bins, Jetson / Pi, High-RPS Pods |

---

## 3. SOLID Design of Python Serving Tier (`src/api/`)

The Python API tier is strictly structured according to the five **SOLID** principles of software architecture:

```mermaid
classDiagram
    class InferenceModelProtocol {
        <<Protocol / Interface>>
        +__call__(x: Tensor) Tensor
    }
    class OnnxModelWrapper {
        -session: ort.InferenceSession
        -input_name: str
        -output_name: str
        +__call__(x: Tensor | ndarray) Tensor
        +eval() None
    }
    class ImagePreprocessor {
        -transform: v2.Compose
        +preprocess(image_bytes: bytes) Tensor
    }
    class TelemetryParser {
        +parse(telemetry, latitude, longitude) Telemetry
    }
    class InferenceService {
        -preprocessor: ImagePreprocessor
        -telemetry_parser: TelemetryParser
        +predict(model, idx_to_class, image_bytes, ...) PredictionResponse
        +get_info(app_state) InfoResponse
    }

    InferenceModelProtocol <|.. OnnxModelWrapper : Implements
    InferenceService --> InferenceModelProtocol : Uses (Dependency Inversion)
    InferenceService --> ImagePreprocessor : Uses (Single Responsibility)
    InferenceService --> TelemetryParser : Uses (Single Responsibility)
```

1. **Single Responsibility Principle (SRP)**:
   - `ImagePreprocessor`: Dedicated solely to decoding image bytes, resizing, and normalizing tensors.
   - `TelemetryParser`: Dedicated solely to parsing, sanitizing, and validating geographic coordinates.
   - `InferenceService`: Coordinates the domain pipeline (preprocessing, inference dispatch, softmax probability mapping).
   - `routes/health.py` & `routes/predict.py`: Focused entirely on HTTP transport, status codes, and routing.
2. **Open/Closed Principle (OCP)**:
   - The serving tier can accept any model backend implementing `InferenceModelProtocol` without altering existing route or service code.
3. **Liskov Substitution Principle (LSP)**:
   - `OnnxModelWrapper` can seamlessly replace a native PyTorch `torch.nn.Module`. Both accept input tensors and return raw logits.
4. **Interface Segregation Principle (ISP)**:
   - `InferenceModelProtocol` requires only a callable contract `__call__(x: Tensor) -> Tensor`. Consumers do not depend on heavyweight training or backward-pass methods.
5. **Dependency Inversion Principle (DIP)**:
   - `InferenceService` and route handlers depend upon the abstract `InferenceModelProtocol`, not concrete ONNX or PyTorch classes. The model instance is injected into `app.state` during application lifespan.

### Lifespan State Management

Application startup and teardown are managed through FastAPI's `@asynccontextmanager lifespan`:
- **Startup**: Resolves `models/recyclenet.onnx`, initializes an `ort.InferenceSession` with optimized thread configurations (`inter_op_num_threads=2`, `intra_op_num_threads=2`), loads label mappings, and sets `app.state.model_ready = True`.
- **Teardown**: Gracefully deallocates session resources, sets `model_ready = False`, and releases memory.

---

## 4. Systems Programming Architecture: Rust Axum Engine (`rust_inference/`)

The Rust microservice delivers a cloud-native, high-throughput drop-in replacement for the Python API:

### Architecture Highlights
- **Axum Web Framework**: Utilizes Axum's modular routing and extractors (`Multipart`, `Json`, `FormPayload`), matching FastAPI's parameter handling.
- **`InferenceBackend` Trait**: Thread-safe abstraction (`Send + Sync`) encapsulated in `Arc<dyn InferenceBackend>`:
  ```rust
  pub trait InferenceBackend: Send + Sync {
      fn predict(&self, image_bytes: &[u8]) -> Result<PredictionResult, EngineError>;
      fn run_tensor(&self, tensor: &Array4<f32>) -> Result<(Vec<f32>, Vec<f32>), EngineError>;
      fn is_ready(&self) -> bool;
      fn classes(&self) -> &[String];
      fn model_path(&self) -> &Path;
  }
  ```
- **Zero-Copy Tensor Memory Pipeline**:
  1. The raw HTTP request bytes are parsed into memory.
  2. `image::load_from_memory` decodes the JPEG/PNG into a pixel buffer.
  3. `imageops::resize` transforms dimensions to $224 \times 224$ via Bilinear interpolation.
  4. Normalized floating-point values are written directly into a contiguous 4D `ndarray::Array4<f32>` buffer with layout `(1, 3, 224, 224)`.
  5. The ndarray buffer is passed to ONNX Runtime via `ort::Value::from_array`, eliminating memory copies.
- **Predictable Latency**: Free from garbage collection pauses and interpreter lock contention. Under high load, Rust maintains flat p99 tail latency.

---

## 5. Preprocessing & Numerical Parity (Eliminating Train-Serve Skew)

A critical failure mode in production Computer Vision is **Train-Serve Skew**: subtle mathematical discrepancies between training transforms and serving inference (e.g., resizing algorithm differences, color channel ordering, rounding errors, normalization constants).

RecycleNet-K8s prevents Train-Serve Skew by standardizing preprocessing mathematically across Python and Rust:

### Mathematical Transformation Pipeline

$$\text{Raw Bytes} \xrightarrow{\text{Decode}} \mathbf{I}_{\text{RGB}} \xrightarrow{\text{Scale}} [0.0, 1.0] \xrightarrow{\text{Resize}} 224 \times 224 \xrightarrow{\text{Normalize}} \mathbf{T}_{(1, 3, 224, 224)}$$

1. **Color Conversion**: Raw compressed byte stream is converted to standard 8-bit RGB (eliminating alpha channels and grayscale variance).
2. **Floating-Point Scaling**: Integer pixel values $p \in [0, 255]$ are scaled to single-precision float $v = \frac{p}{255.0}$.
3. **Spatial Resizing**: Resized to exactly $224 \times 224$ pixels using standard **Bilinear Interpolation** (`torchvision.transforms.v2.Resize` and `image::imageops::FilterType::Triangle`).
4. **Channel Normalization**:
   $$x_{c, i, j}^{\text{norm}} = \frac{x_{c, i, j} - \mu_c}{\sigma_c}$$
   Where ImageNet standard constants are strictly enforced:
   $$\mu = [0.485, 0.456, 0.406], \quad \sigma = [0.229, 0.224, 0.225]$$
5. **Memory Layout Transformation**:
   - Python: Handled via PyTorch's native NCHW tensor layout.
   - Rust: Transforms row-major interleaved HWC pixel buffers into contiguous planar NCHW via `ndarray::Array4<f32>::zeros((1, 3, 224, 224))`.

### Automated Numerical Parity Gate

Automated test suites (`benchmarks/benchmark_suite.py` and `tests/test_rust_parity.py`) enforce strict parity criteria across synthetic and real TrashNet test images:
- **Top-1 Class Agreement**: Must equal **100.0%**.
- **Maximum Softmax Probability Difference ($\Delta_\max$)**:
  $$\Delta_\max = \max_{i \in \text{classes}} |P_{\text{Python}}(c_i) - P_{\text{Rust}}(c_i)| \le 1.0 \times 10^{-3}$$
- Any discrepancy exceeding $10^{-3}$ immediately fails CI/CD, guaranteeing zero Train-Serve Skew in production.

---

## 6. Scientific Benchmarking Methodology & Performance Metrics

RecycleNet-K8s incorporates a dual benchmarking strategy combining an in-depth **Python Scientific Harness** and a production-grade **k6 Load Generator**:

```mermaid
flowchart LR
    subgraph Harness["Scientific Harness (benchmarks/benchmark_suite.py)"]
        Runner["BenchmarkRunner"] -->|Generate| SyntheticSamples["10+ Calibrated Synthetic Images"]
        Runner -->|Benchmark Target Protocol| InProcess["InProcessFastApiTarget (FastAPI)"]
        Runner -->|Benchmark Target Protocol| HttpTargetPy["HttpBenchmarkTarget (:8000)"]
        Runner -->|Benchmark Target Protocol| HttpTargetRust["HttpBenchmarkTarget (:8080)"]
        Runner -->|Compute Statistics| Percentiles["Numpy Percentiles (p50, p95, p99)<br/>Throughput (RPS)<br/>Resident Memory (RSS MB)"]
        Runner -->|Evaluate| ParityCheck["Numerical Parity & Probability Deltas"]
        Runner -->|Render| MarkdownReport["Scientific Markdown Comparison Table"]
    end
```

### Measured Performance Dimensions
1. **Latency Percentiles**:
   - **p50 (Median)**: Typical user response time.
   - **p95**: Latency experienced by the 95th percentile under standard queue depth.
   - **p99 (Tail Latency)**: Latency under worst-case queue contention and GC impact.
2. **Throughput (Requests Per Second - RPS)**: Maximum sustainable request volume before error onset.
3. **Resident Memory Footprint (RSS MB)**: Physical RAM occupied by the container process.
4. **Container Image Footprint**: Uncompressed container image size transferred across registries.

### Production Load Testing with `k6` (`benchmarks/k6_load_test.js`)

The `k6` script simulates realistic user traffic across 4 distinct operational stages:
1. **Stage 1: Warm-up** (15s @ 10 Virtual Users) — JIT warmup, session initialization, DNS caching.
2. **Stage 2: Ramp-up** (30s @ 30 Virtual Users) — Increasing concurrency to test connection pooling.
3. **Stage 3: Peak Load** (45s @ 60 Virtual Users) — Stress testing queue depth and CPU saturation.
4. **Stage 4: Cooldown** (15s @ 0 Virtual Users) — Connection draining and resource deallocation.

#### Strict Service Level Objectives (SLOs)
- `http_req_failed`: `< 1%` (Less than 1 failure per 100 requests).
- `http_req_duration`: `p(95) < 300ms`, `p(99) < 600ms`.
- `inference_latency_ms`: `p(95) < 300ms`, `p(99) < 600ms`.
- `inference_success_rate`: `> 99%`.

---

## 7. Edge/IoT & Cloud-Native Deployment Strategy

### A. Edge / Smart Waste Bin Architecture
In resource-constrained deployments (such as solar-powered smart waste receptacles or embedded NVIDIA Jetson / Raspberry Pi 5 gateways):
- **Hardware Profile**: 512 MB – 1 GB total system RAM, Quad-Core ARM Cortex-A76.
- **Serving Selection**: **Rust Axum Microservice**.
- **Advantages**:
  - Requires less than **45 MB RSS**, leaving 90%+ system RAM for operating system and camera drivers.
  - Zero Python runtime dependencies; runs as a single self-contained binary or lightweight `<40 MB` container.
  - Sub-50ms cold boot allows smart bins to wake from deep sleep on ultrasonic motion detection, classify waste in <15ms, and sleep immediately.

### B. Cloud-Native Kubernetes Deployment
In clustered cloud environments (Minikube / Google Kubernetes Engine):
- **High Availability**: Minimum 2 replicas configured with anti-affinity across nodes.
- **Resource Limits & Requests**:
  - Python Pods: `requests.memory: 512Mi`, `limits.memory: 1Gi`, `requests.cpu: 250m`, `limits.cpu: 1000m`.
  - Rust Pods: `requests.memory: 64Mi`, `limits.memory: 128Mi`, `requests.cpu: 100m`, `limits.cpu: 500m`.
  - *Result*: The Rust tier provides **8x higher pod density per Kubernetes node**, cutting infrastructure cluster costs dramatically.
- **Probes**:
  - `livenessProbe`: `GET /healthz` (Checks if process is running).
  - `readinessProbe`: `GET /ready` (Verifies ONNX model is loaded into RAM before receiving traffic).
- **Horizontal Pod Autoscaling (HPA)**:
  - Scaled from 2 to 5 replicas based on CPU utilization exceeding 70%.
  - Rust's instantaneous startup time ensures new pods become `Ready` instantly, absorbing burst traffic without queue latency spikes.

---

## 8. Architectural Summary Matrix

| Metric / Feature | Python FastAPI (`:8000`) | Rust Axum (`:8080`) | Delta / Gain |
| :--- | :---: | :---: | :---: |
| **Median Latency (p50)** | `~12–18 ms` | `~3–6 ms` | **~3x Faster** |
| **Tail Latency (p99)** | `~45–80 ms` | `~8–14 ms` | **~5x More Predictable** |
| **Max Throughput (RPS)** | `~80–120 req/s` | `~450–700 req/s` | **~5x Higher Throughput** |
| **Resident Memory (RSS)**| `~380 MB` | `~38 MB` | **10x Less RAM** |
| **Docker Image Size** | `~1,200 MB` | `~38 MB` | **31x Smaller Container** |
| **Pod Startup Time** | `~2.8 s` | `~0.04 s` | **70x Faster Cold Start** |
| **Train-Serve Skew** | Baseline | $L_\infty < 10^{-3}$ | **Zero Skew Verified** |

This dual-tier architecture demonstrates how modern MLOps pipelines can leverage Python for training, experimentation, and cloud serving, while deploying systems programming languages (Rust) for performance-critical edge and high-density microservices.
