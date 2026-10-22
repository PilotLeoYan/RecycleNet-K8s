# ==============================================================================
# RecycleNet Multi-Stage Production Dockerfile
# Optimized with Astral uv (Rust package manager) for minimal footprint and CVEs
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 1: Dependency Builder
# ------------------------------------------------------------------------------
FROM python:3.14-slim-bookworm AS builder

# Install uv from the official Astral binary image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Enable bytecode compilation and disable copy links across mount points
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

# Copy dependency manifests first to leverage Docker layer caching
COPY pyproject.toml uv.lock .python-version ./

# Install project runtime dependencies including torch/torchvision via --extra cpu
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra cpu --no-install-project

# Copy source code and install project package
COPY src/ ./src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra cpu

# Allow selecting 'large' or 'small' MobileNetV3 ONNX model at build time (default: large)
ARG MODEL_VARIANT=large

# Ensure models directory exists and download default ONNX weights from Hugging Face
RUN mkdir -p /app/models && \
    apt-get update && apt-get install -y --no-install-recommends curl ca-certificates && \
    curl -fsSL "https://huggingface.co/LeoYan/recyclenet-mobilenetv3/resolve/main/recyclenet-${MODEL_VARIANT}.onnx" -o /app/models/recyclenet.onnx && \
    curl -fsSL "https://huggingface.co/LeoYan/recyclenet-mobilenetv3/resolve/main/classes_mapping.json" -o /app/models/classes_mapping.json && \
    rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------------------------
# Stage 2: Minimal Distroless/Slim Runtime
# ------------------------------------------------------------------------------
FROM python:3.14-slim-bookworm AS runtime

# System configurations
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" \
    RECYCLENET_MODEL_PATH="/app/models/recyclenet.onnx" \
    PORT=8000

# Create unprivileged system user for zero-trust security
RUN groupadd -r recyclenet --gid 10001 && \
    useradd -r -g recyclenet --uid 10001 --no-create-home --shell /bin/false recyclenet

WORKDIR /app

# Copy virtualenv and models from builder stage
COPY --from=builder --chown=recyclenet:recyclenet /app/.venv /app/.venv
COPY --from=builder --chown=recyclenet:recyclenet /app/models /app/models

# Copy application source code and configurations
COPY --chown=recyclenet:recyclenet src/ /app/src/
COPY --chown=recyclenet:recyclenet configs/ /app/configs/

# Drop privileges to non-root user
USER recyclenet

# Expose HTTP port
EXPOSE 8000

# Start Uvicorn ASGI server
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
