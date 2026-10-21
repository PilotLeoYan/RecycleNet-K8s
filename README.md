# ♻️ RecycleNet-K8s

Waste classification (TrashNet, 6 classes) with an end-to-end MLOps pipeline.

> **Status:** 🟢 Phase 1 (Training) done · 🟡 Phase 2–4 (Inference, Kubernetes, CI/CD) in progress

## 🗺️ Roadmap

```mermaid
flowchart LR
    F1["✅ 1. Training<br/>HPO + ONNX"] --> F2["🟡 2. Inference<br/>FastAPI + Rust"] --> F3["🟡 3. Kubernetes<br/>Minikube + HPA"] --> F4["🟡 4. CI/CD<br/>GitHub Actions + GCP"]
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
configs/             YAML configs (config, smoke, best_hpo)
src/components/      data, model, train step, metrics, evals, MLflow logging
src/pipeline/        train · hpo · export_onnx · super_train
src/main.py          CLI (train | hpo | export | super-train)
models/              ONNX model + model card
tests/               unit tests
```

## Stack

PyTorch · Ray Train/Tune · Optuna · MLflow · ONNX Runtime · uv · ruff · mypy · pytest
