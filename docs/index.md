# RecycleNet-K8s

**RecycleNet** is an end-to-end MLOps system for waste classification (e.g., glass, paper, cardboard, plastic, metal, and trash) using Transfer Learning in PyTorch (`MobileNetV3`), experiment tracking with MLflow, containerized deployment, and automated CI/CD pipelines.

---

## 🚀 Key Features

* **Modular Clean Architecture**: Strict separation of concerns adhering to SOLID design principles.
* **Transfer Learning**: MobileNetV3-Small fine-tuned for high accuracy and minimal latency.
* **Reproducible Pipelines**: Deterministic random seed management across Python, NumPy, PyTorch, and CUDA.
* **MLflow Tracking**: Automatic logging of hyperparameters, per-epoch metrics, evaluation artifacts, and registered models.
* **Production-Grade Logging**: Dual-mode logging (ANSI colorized console locally, structured JSON for Google Cloud Logging).
* **Strict Type Safety**: 100% type annotations verified with `mypy` and linted with `ruff`.

---

## 📦 Installation & Setup

RecycleNet uses [uv](https://docs.astral.sh/uv/) for ultra-fast dependency management:

```bash
# Clone the repository
git clone https://github.com/PilotLeoYan/RecycleNet-K8s.git
cd RecycleNet-K8s

# Install dependencies (including dev and docs groups)
uv sync --all-groups
```

---

## 🏃 Running the Training Pipeline

Execute the end-to-end training pipeline from the command line:

```bash
# Run training pipeline
uv run python -m src.main
```

Launch the local MLflow dashboard to explore metrics and artifacts:

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```
