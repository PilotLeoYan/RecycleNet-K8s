# ♻️ RecycleNet-K8s

Waste classification (TrashNet, 6 classes) with an end-to-end MLOps pipeline.

> **Status:** 🟢 Phase 1 (Training) & Phase 2 (Inference) done · 🟢 Phase 3 (Kubernetes) ready · 🟢 Phase 4 (CI/CD & GCP) done
>
> 🤗 **Model:** [LeoYan/recyclenet-mobilenetv3](https://huggingface.co/LeoYan/recyclenet-mobilenetv3)

## 🗺️ Roadmap

```mermaid
flowchart LR
    F1["✅ 1. Training<br/>HPO + ONNX"] --> F2["✅ 2. Inference<br/>FastAPI + Rust"] --> F3["✅ 3. Kubernetes<br/>Minikube + HPA"] --> F4["✅ 4. CI/CD<br/>GitHub Actions + GCP"]
```

---

# 🟢 Phase 1: Training

## Overview

```mermaid
flowchart LR
    D[("TrashNet<br/>zip")] --> I[Ingestion<br/>+ split 70/15/15]
    I --> H["HPO<br/>Ray Tune + ASHA"]
    H -- "best_hpo_config.yaml" --> T["Training<br/>Ray Train"]
    T --> E["Test eval<br/>Acc · F1 · ROC-AUC"]
    T --> M[("MLflow<br/>sqlite")]
    H --> M
    M --> X["ONNX export<br/>+ parity check"]
    X --> O[["models/recyclenet.onnx"]]
```

## Model

```mermaid
flowchart LR
    IMG["Image<br/>3×224×224"] --> B["MobileNetV3-Small<br/>ImageNet weights<br/>❄️ frozen"]
    B --> C["Linear → 6 classes<br/>🔥 trainable"]
    C --> P["Logits"]
```

Transfer learning: only the last classifier layer is trained.

## Training loop

```mermaid
flowchart LR
    A[Epoch] --> B["Train step<br/>CrossEntropy + AdamW"]
    B --> C[Validate]
    C --> D{"valid_loss<br/>improved?"}
    D -- yes --> E["Save checkpoint<br/>→ MLflow"]
    D -- "no (patience = 7)" --> F["Early stop"]
    E --> A
    C -.-> G["LR scheduler<br/>cosine / plateau"]
```

## Hyperparameter search (HPO)

```mermaid
flowchart TB
    S["Search space<br/>lr: loguniform 1e-4…1e-2<br/>weight decay: loguniform 1e-4…1e-1<br/>batch: 16 / 32 / 64"] --> O["Optuna sampler<br/>20 trials"]
    O --> R["Ray Tune<br/>2 concurrent trials"]
    R --> A{"ASHA<br/>grace = 3 · max = 8 epochs<br/>reduction = 2"}
    A -- "promising" --> K[Keep training]
    A -- "bad" --> X[Prune ✂️]
    K --> BEST["Best trial<br/>min valid_loss"]
    BEST --> Y[("configs/<br/>best_hpo_config.yaml")]
    R -. "every trial" .-> ML[("MLflow")]
```

## ONNX export

```mermaid
flowchart LR
    R[("MLflow registry<br/>or checkpoint")] --> L[Load PyTorch model]
    L --> X["torch.onnx.export<br/>opset 18 · dynamic batch"]
    X --> C["onnx.checker"]
    C --> P{"max|torch − onnx|<br/>&lt; 1e-5 ?"}
    P -- yes --> OK[["✅ recyclenet.onnx"]]
    P -- no --> ERR["❌ fail"]
```

## Results

| Test accuracy | F1 (macro) | ROC-AUC (OvR) |
| :-: | :-: | :-: |
| 0.781 | 0.749 | 0.947 |

*Source: [`models/model_card.json`](models/model_card.json).*

🤗 **Trained model:** [LeoYan/recyclenet-mobilenetv3](https://huggingface.co/LeoYan/recyclenet-mobilenetv3)

---

# 🟢 Phase 3: Cloud-Native Kubernetes Orchestration & Local HPA Lab

Production-grade declarative orchestration with zero-trust container security, high availability, and dynamic horizontal pod autoscaling tested locally on Minikube with zero cloud cost.

## Manifest Architecture

```mermaid
flowchart TD
    Client["Client / Edge Device"] -->|HTTPS :443| Ingress["Ingress (NGINX)<br/>recyclenet-ingress"]
    Ingress --> Service["Service (ClusterIP)<br/>recyclenet-service"]
    Service -->|:8000| PyPods["Python API (2 Replicas)<br/>recyclenet-deployment<br/>QoS: Burstable (512Mi/1Gi)"]
    Service -->|:8080| RustPods["Rust API (2 Replicas)<br/>recyclenet-rust-deployment<br/>QoS: Burstable (64Mi/128Mi)"]
    ConfigMap[("ConfigMap<br/>recyclenet-config")] -.-> PyPods & RustPods
    Secret[("Secret<br/>recyclenet-secret")] -.-> PyPods & RustPods
    HPA["HPA v2<br/>2–5 replicas @ 70% CPU"] -->|Scales| PyPods
    MetricsServer["Metrics Server"] -.->|Scrapes| HPA
```

## Minikube Local Lab Commands

Run the automated 5-step lab workflow using the cross-platform automation scripts:

```bash
# Linux / macOS / WSL2 / Git Bash:
./scripts/k8s_local_lab.sh --dry-run all   # Preview without cluster
./scripts/k8s_local_lab.sh all            # Execute complete 5-step lab

# Windows PowerShell:
.\scripts\k8s_local_lab.ps1 -DryRun -Action all
.\scripts\k8s_local_lab.ps1 -Action all
```

Or execute the 5 steps manually:

```bash
# 1. Start local cluster
minikube start --driver=docker --cpus=4 --memory=4096

# 2. Enable metrics server for HPA
minikube addons enable metrics-server

# 3. Build container images inside Minikube Docker daemon
eval $(minikube docker-env)
docker build -t recyclenet-api:local .
docker build -f Dockerfile.rust -t recyclenet-rust:local .

# 4. Deploy declarative manifests & await rollout
kubectl apply -k k8s/
kubectl rollout status deployment/recyclenet-deployment --timeout=120s
kubectl rollout status deployment/recyclenet-rust-deployment --timeout=120s

# 5. Live HPA Autoscaling stress test
kubectl port-forward svc/recyclenet-service 8000:8000 8080:8080 &
k6 run -e TARGET_URL=http://localhost:8000 benchmarks/k6_load_test.js
kubectl get hpa recyclenet-hpa -w
```

## Offline Simulation & Manifest Validation

Verify manifest schemas, Burstable QoS calculations, and HPA control loop dynamics without requiring Minikube or Docker:

```bash
uv run python -m src.k8s.orchestrator
```

---

# 🟢 Phase 4: Continuous DevSecOps & Cloud Deployment (GCP)

Automated zero-trust CI/CD pipeline enforcing 5 quality and security gates, keyless Google Cloud authentication via Workload Identity Federation (OIDC), immutable container publishing to Google Artifact Registry, and serverless auto-scaling on Google Cloud Run.

## Pipeline Architecture

```mermaid
flowchart LR
    subgraph CI["Continuous Integration (GitHub Actions)"]
        PR["Pull Request / Push"] --> UV["setup-uv"]
        UV --> G1["Ruff"] --> G2["Mypy"] --> G3["Bandit"] --> G4["Pytest (80%+ cov)"]
    end

    subgraph CD["Continuous Deployment (GCP)"]
        G4 --> OIDC["Keyless OIDC<br/>Workload Identity"]
        OIDC --> GAR["Artifact Registry<br/>sha-${{ github.sha }}"]
        GAR --> CR["Cloud Run<br/>1Gi RAM · 1 CPU<br/>1–5 instances"]
    end
```

## Supply Chain Security & CVE Version Patches

Transitive dependencies are hardened via `[tool.uv] override-dependencies` in `pyproject.toml` and locked in `uv.lock`, alongside CodeQL `CWE-020` URL/domain sanitization in OIDC claim verification:

| Package | Patched Version | Security Remediation |
| :--- | :---: | :--- |
| `gitpython` | `3.1.62` → **`3.2.1`** | Fixes **High** RCE in submodule `.git` path validation prior to cloning |
| `graphql-core` | `3.2.11` → **`3.2.13`** | Fixes two **High** DoS vulnerabilities in `max_tokens` parser & field validation |
| `opentelemetry-api` | `1.44.0` → **`1.45.1`** | Fixes **Moderate** DoS in unbounded `tracestate` header parsing (`sdk` & `proto` synced) |
| `cryptography` | **`>= 50.0.0`** | Enforces hardened OpenSSL bindings and X.509 primitives |

## Offline Verification & DevSecOps CLI

Verify CI quality gates, OIDC token claims exchange, and Cloud Run scaling rules offline:

```bash
uv run python -m src.devsecops.orchestrator
```

---

## Quickstart

```bash
make sync-cpu          # or: make sync-gpu
make super-train       # HPO → train (best config) → ONNX export
make mlflow            # UI at http://localhost:5000
```

Or step by step:

| Command | What it does |
| :-- | :-- |
| `make train` | Train with `configs/config.yaml` |
| `make hpo` | Ray Tune search → writes `configs/best_hpo_config.yaml` |
| `make train-best` | Train using the best HPO config |
| `make export` | Export to `models/recyclenet.onnx` |
| `make test` · `make lint` | pytest · ruff + mypy |

## Layout

```text
.github/workflows/   CI/CD pipelines (ci.yaml, cd.yaml, codeql, dependency-review)
configs/             YAML configs (config, smoke, best_hpo)
k8s/                 Declarative manifests (Deployments, Service, HPA, Ingress, ConfigMap, Secret)
rust_inference/      High-performance Rust microservice (Axum + ort)
scripts/             Local lab automation scripts (k8s_local_lab.sh, k8s_local_lab.ps1)
src/api/             FastAPI baseline inference microservice
src/components/      data, model, train step, metrics, evals, MLflow logging
src/devsecops/       DevSecOps CI/CD validator, GCP OIDC & Cloud Run simulator
src/k8s/             Kubernetes declarative validation & HPA simulation engine
src/pipeline/        train · hpo · export_onnx · super_train
src/main.py          CLI (train | hpo | export | super-train)
models/              ONNX model + model card
tests/               unit & integration tests
```

## Stack

PyTorch · Ray Train/Tune · Optuna · MLflow · ONNX Runtime · Rust (Axum) · Kubernetes (Minikube + HPA v2) · GitHub Actions · GCP (OIDC + Artifact Registry + Cloud Run) · uv · ruff · mypy · bandit · pytest
