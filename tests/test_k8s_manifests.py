"""Automated validation tests for Kubernetes manifests and Dockerfile.

Ensures that all declarative Kubernetes artifacts in `k8s/` and the Dockerfile
strictly adhere to the architecture and resiliency specifications of RecycleNet.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

K8S_DIR = Path("k8s")
DOCKERFILE_PATH = Path("Dockerfile")


def _load_yaml(filename: str) -> dict[str, Any]:
    file_path = K8S_DIR / filename
    assert file_path.exists(), f"Manifest {filename} does not exist in k8s/"
    with file_path.open("r", encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f)
    return data


def test_configmap_manifest_specifications() -> None:
    """Validate configmap.yaml contains all required environment variables."""
    config = _load_yaml("configmap.yaml")

    assert config["apiVersion"] == "v1"
    assert config["kind"] == "ConfigMap"
    assert config["metadata"]["name"] == "recyclenet-config"

    data = config["data"]
    assert data["PORT"] == "8000"
    assert data["RUST_PORT"] == "8080"
    assert data["RECYCLENET_MODEL_PATH"] == "/app/models/recyclenet.onnx"
    assert data["LOG_LEVEL"] == "INFO"
    assert data["RUST_LOG"] == "info"
    assert data["MODEL_VARIANT"] == "large"
    assert data["ORT_DYLIB_PATH"] == "/usr/lib/libonnxruntime.so"


def test_secret_manifest_specifications() -> None:
    """Validate secret.yaml contains base64 credentials and TLS assets."""
    secret = _load_yaml("secret.yaml")

    assert secret["apiVersion"] == "v1"
    assert secret["kind"] == "Secret"
    assert secret["metadata"]["name"] == "recyclenet-secret"
    assert secret["type"] == "Opaque"

    data = secret["data"]
    assert "RECYCLENET_API_KEY" in data
    assert "TELEMETRY_AUTH_TOKEN" in data
    assert "tls.crt" in data
    assert "tls.key" in data


def test_deployment_manifest_specifications() -> None:
    """Validate deployment.yaml adheres to RecycleNet production architecture."""
    deployment = _load_yaml("deployment.yaml")

    assert deployment["apiVersion"] == "apps/v1"
    assert deployment["kind"] == "Deployment"
    assert deployment["metadata"]["name"] == "recyclenet-deployment"

    spec = deployment["spec"]
    assert spec["replicas"] == 2, "Deployment must configure exactly 2 replicas for HA"

    # Match labels
    match_labels = spec["selector"]["matchLabels"]
    pod_labels = spec["template"]["metadata"]["labels"]
    assert match_labels["app"] == "recyclenet"
    assert pod_labels["app"] == "recyclenet"

    # Container checks
    containers = spec["template"]["spec"]["containers"]
    assert len(containers) >= 1
    api_container = containers[0]
    assert api_container["name"] == "recyclenet-api"

    # Port 8000 exposed
    ports = [p["containerPort"] for p in api_container["ports"]]
    assert 8000 in ports

    # Resource requests and limits
    resources = api_container["resources"]
    assert "requests" in resources and "limits" in resources
    assert resources["requests"]["memory"] == "512Mi"
    assert "cpu" in resources["requests"]
    assert "memory" in resources["limits"]

    # Health Probes
    liveness = api_container["livenessProbe"]
    readiness = api_container["readinessProbe"]
    assert liveness["httpGet"]["path"] in ["/healthz", "/health"]
    assert liveness["httpGet"]["port"] == 8000
    assert readiness["httpGet"]["path"] in ["/ready", "/health"]
    assert readiness["httpGet"]["port"] == 8000

    # Decoupled configuration checks
    env_from = [
        ref.get("configMapRef", {}).get("name") or ref.get("secretRef", {}).get("name")
        for ref in api_container["envFrom"]
    ]
    assert "recyclenet-config" in env_from
    assert "recyclenet-secret" in env_from

    # Security Context checks
    pod_sec = spec["template"]["spec"]["securityContext"]
    assert pod_sec["runAsNonRoot"] is True
    assert pod_sec["runAsUser"] == 10001
    assert api_container["securityContext"]["allowPrivilegeEscalation"] is False


def test_rust_deployment_manifest_specifications() -> None:
    """Validate deployment-rust.yaml adheres to high-performance edge tier specs."""
    deployment = _load_yaml("deployment-rust.yaml")

    assert deployment["apiVersion"] == "apps/v1"
    assert deployment["kind"] == "Deployment"
    assert deployment["metadata"]["name"] == "recyclenet-rust-deployment"

    spec = deployment["spec"]
    assert spec["replicas"] == 2

    # Match labels
    match_labels = spec["selector"]["matchLabels"]
    pod_labels = spec["template"]["metadata"]["labels"]
    assert match_labels["app"] == "recyclenet"
    assert match_labels["component"] == "api-rust"
    assert pod_labels["runtime"] == "rust"

    # Container checks
    containers = spec["template"]["spec"]["containers"]
    assert len(containers) >= 1
    rust_container = containers[0]
    assert rust_container["name"] == "recyclenet-rust-api"

    # Port 8080 exposed
    ports = [p["containerPort"] for p in rust_container["ports"]]
    assert 8080 in ports

    # Resources (micro footprint)
    resources = rust_container["resources"]
    assert resources["requests"]["memory"] == "64Mi"
    assert resources["limits"]["memory"] == "128Mi"

    # Health Probes on 8080
    assert rust_container["livenessProbe"]["httpGet"]["port"] == 8080
    assert rust_container["readinessProbe"]["httpGet"]["port"] == 8080

    # Non-root security
    assert spec["template"]["spec"]["securityContext"]["runAsNonRoot"] is True
    assert rust_container["securityContext"]["allowPrivilegeEscalation"] is False


def test_service_manifest_specifications() -> None:
    """Validate service.yaml exposes port 8000 and 8080 matching deployment."""
    service = _load_yaml("service.yaml")

    assert service["apiVersion"] == "v1"
    assert service["kind"] == "Service"
    assert service["metadata"]["name"] == "recyclenet-service"

    spec = service["spec"]
    assert spec["selector"]["app"] == "recyclenet"

    ports = spec["ports"]
    http_port = next((p for p in ports if p["port"] == 8000), None)
    assert http_port is not None, "Service must expose port 8000"
    assert http_port["targetPort"] == 8000

    rust_port = next((p for p in ports if p["port"] == 8080), None)
    assert rust_port is not None, "Service must expose port 8080 for Rust tier"
    assert rust_port["targetPort"] == 8080


def test_hpa_manifest_specifications() -> None:
    """Validate hpa.yaml targets deployment with CPU threshold and scaling behavior."""
    hpa = _load_yaml("hpa.yaml")

    assert hpa["apiVersion"] == "autoscaling/v2"
    assert hpa["kind"] == "HorizontalPodAutoscaler"
    assert hpa["metadata"]["name"] == "recyclenet-hpa"

    spec = hpa["spec"]
    assert spec["scaleTargetRef"]["name"] == "recyclenet-deployment"
    assert spec["minReplicas"] == 2
    assert spec["maxReplicas"] == 5

    metric = spec["metrics"][0]
    assert metric["type"] == "Resource"
    assert metric["resource"]["name"] == "cpu"
    assert metric["resource"]["target"]["averageUtilization"] == 70

    behavior = spec["behavior"]
    assert behavior["scaleUp"]["stabilizationWindowSeconds"] == 0
    assert behavior["scaleDown"]["stabilizationWindowSeconds"] == 60


def test_ingress_manifest_specifications() -> None:
    """Validate ingress.yaml configures TLS termination and routing to service."""
    ingress = _load_yaml("ingress.yaml")

    assert ingress["apiVersion"] == "networking.k8s.io/v1"
    assert ingress["kind"] == "Ingress"
    assert ingress["metadata"]["name"] == "recyclenet-ingress"

    spec = ingress["spec"]

    # TLS configuration
    assert "tls" in spec and len(spec["tls"]) > 0
    tls_entry = spec["tls"][0]
    assert tls_entry["secretName"] == "recyclenet-secret"
    assert len(tls_entry["hosts"]) > 0

    # Rules pointing to recyclenet-service:8000
    rules = spec["rules"]
    assert len(rules) > 0
    rule = rules[0]
    path_entry = rule["http"]["paths"][0]
    assert path_entry["backend"]["service"]["name"] == "recyclenet-service"
    assert path_entry["backend"]["service"]["port"]["number"] == 8000


def test_kustomization_manifest_specifications() -> None:
    """Validate kustomization.yaml resources include all manifests in logical order."""
    kustomization = _load_yaml("kustomization.yaml")

    resources = kustomization["resources"]
    expected_order = [
        "configmap.yaml",
        "secret.yaml",
        "deployment.yaml",
        "deployment-rust.yaml",
        "service.yaml",
        "hpa.yaml",
        "ingress.yaml",
    ]
    assert resources == expected_order


def test_dockerfile_multi_stage_structure() -> None:
    """Validate Dockerfile is multi-stage and optimized with uv."""
    assert DOCKERFILE_PATH.exists(), "Dockerfile must exist at repository root"
    content = DOCKERFILE_PATH.read_text(encoding="utf-8")

    assert "AS builder" in content, "Dockerfile must define a builder stage"
    assert "AS runtime" in content, "Dockerfile must define a runtime stage"
    assert "uv" in content, "Dockerfile must use uv package manager"
    assert "EXPOSE 8000" in content, "Dockerfile must expose port 8000"
    assert "useradd" in content or "USER" in content, "Dockerfile must run as non-root"
