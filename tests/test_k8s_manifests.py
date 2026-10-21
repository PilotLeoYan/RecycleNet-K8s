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


def test_service_manifest_specifications() -> None:
    """Validate service.yaml exposes port 8000 with selector matching deployment."""
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
    assert "secretName" in tls_entry
    assert len(tls_entry["hosts"]) > 0

    # Rules pointing to recyclenet-service:8000
    rules = spec["rules"]
    assert len(rules) > 0
    rule = rules[0]
    path_entry = rule["http"]["paths"][0]
    assert path_entry["backend"]["service"]["name"] == "recyclenet-service"
    assert path_entry["backend"]["service"]["port"]["number"] == 8000


def test_dockerfile_multi_stage_structure() -> None:
    """Validate Dockerfile is multi-stage and optimized with uv."""
    assert DOCKERFILE_PATH.exists(), "Dockerfile must exist at repository root"
    content = DOCKERFILE_PATH.read_text(encoding="utf-8")

    assert "AS builder" in content, "Dockerfile must define a builder stage"
    assert "AS runtime" in content, "Dockerfile must define a runtime stage"
    assert "uv" in content, "Dockerfile must use uv package manager"
    assert "EXPOSE 8000" in content, "Dockerfile must expose port 8000"
    assert "useradd" in content or "USER" in content, "Dockerfile must run as non-root"
