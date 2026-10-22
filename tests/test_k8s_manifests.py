"""Automated validation tests for Kubernetes manifests and Dockerfile.

Ensures that all declarative Kubernetes artifacts in `k8s/` and the Dockerfile
strictly adhere to the architecture and resiliency specifications of RecycleNet.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from src.k8s.orchestrator import (
    AuditReport,
    CrossResourceIntegrityValidator,
    DeploymentSecurityAndQoSValidator,
    DockerfileZeroTrustValidator,
    HpaSimulationStep,
    HpaV2ControlLoopSimulator,
    KubernetesOrchestrationAuditor,
    LabStep,
    MinikubeLabPlanBuilder,
    ResourceQuantityParser,
    YamlManifestLoader,
)
from src.k8s.orchestrator import (
    main as orchestrator_main,
)

K8S_DIR = Path("k8s")
DOCKERFILE_PATH = Path("Dockerfile")
DOCKERFILE_RUST_PATH = Path("Dockerfile.rust")


def _load_yaml(filename: str) -> dict[str, Any]:
    file_path = K8S_DIR / filename
    assert file_path.exists(), f"Manifest {filename} does not exist in k8s/"
    with file_path.open("r", encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f)
    return data


# ==============================================================================
# Existing Manifest Specification Tests
# ==============================================================================


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
    assert spec["replicas"] == 2, "Deployment must configure 2 replicas for HA"

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
    """Validate deployment-rust.yaml adheres to edge tier specifications."""
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
    """Validate hpa.yaml targets deployment with CPU threshold and behavior."""
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
    """Validate kustomization.yaml resources include all manifests in order."""
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


# ==============================================================================
# Helper Mock Loader for Edge Case Unit Testing
# ==============================================================================


class InMemoryManifestLoader:
    """In-memory manifest loader simulating custom or malformed YAML payloads."""

    def __init__(self, manifests: dict[str, dict[str, Any]]) -> None:
        self.manifests = manifests

    def load_manifest(self, filename: str) -> dict[str, Any]:
        """Return predefined manifest or raise FileNotFoundError."""
        if filename not in self.manifests:
            raise FileNotFoundError(f"Manifest '{filename}' not found in mock")
        return self.manifests[filename]

    def list_manifest_files(self) -> list[str]:
        """List simulated manifest filenames."""
        return sorted(self.manifests.keys())


# ==============================================================================
# SOLID Orchestrator & Parser Unit Tests
# ==============================================================================


def test_resource_quantity_parser_cpu() -> None:
    """Verify ResourceQuantityParser CPU parsing across valid and invalid inputs."""
    # Millicores notations
    assert ResourceQuantityParser.parse_cpu_millicores("250m") == 250
    assert ResourceQuantityParser.parse_cpu_millicores("1000m") == 1000
    assert ResourceQuantityParser.parse_cpu_millicores("100m") == 100

    # Whole and fractional core notations
    assert ResourceQuantityParser.parse_cpu_millicores("1") == 1000
    assert ResourceQuantityParser.parse_cpu_millicores(1) == 1000
    assert ResourceQuantityParser.parse_cpu_millicores(0.5) == 500
    assert ResourceQuantityParser.parse_cpu_millicores("0.5") == 500
    assert ResourceQuantityParser.parse_cpu_millicores("2.5") == 2500

    # Error handling
    with pytest.raises(ValueError, match="cannot be empty"):
        ResourceQuantityParser.parse_cpu_millicores("")

    with pytest.raises(ValueError, match="cannot be None"):
        ResourceQuantityParser.parse_cpu_millicores(None)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="must be positive"):
        ResourceQuantityParser.parse_cpu_millicores(-1)

    with pytest.raises(ValueError, match="must be positive"):
        ResourceQuantityParser.parse_cpu_millicores("0m")

    with pytest.raises(ValueError, match="Invalid numerical format"):
        ResourceQuantityParser.parse_cpu_millicores("invalid-cpu")


def test_resource_quantity_parser_memory() -> None:
    """Verify ResourceQuantityParser memory byte parsing across binary/decimal units."""
    # Binary SI multipliers (powers of 1024)
    assert ResourceQuantityParser.parse_memory_bytes("64Mi") == 64 * 1024 * 1024
    assert ResourceQuantityParser.parse_memory_bytes("128Mi") == 128 * 1024 * 1024
    assert ResourceQuantityParser.parse_memory_bytes("512Mi") == 512 * 1024 * 1024
    assert ResourceQuantityParser.parse_memory_bytes("1Gi") == 1 * 1024 * 1024 * 1024
    assert ResourceQuantityParser.parse_memory_bytes("1Ki") == 1024

    # Decimal SI multipliers (powers of 1000)
    assert ResourceQuantityParser.parse_memory_bytes("100M") == 100_000_000
    assert ResourceQuantityParser.parse_memory_bytes("1G") == 1_000_000_000

    # Plain integers and byte strings
    assert ResourceQuantityParser.parse_memory_bytes(1048576) == 1048576
    assert ResourceQuantityParser.parse_memory_bytes("1048576") == 1048576

    # Error handling
    with pytest.raises(ValueError, match="cannot be empty"):
        ResourceQuantityParser.parse_memory_bytes("")

    with pytest.raises(ValueError, match="cannot be None"):
        ResourceQuantityParser.parse_memory_bytes(None)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="must be positive"):
        ResourceQuantityParser.parse_memory_bytes(-500)

    with pytest.raises(ValueError, match="Invalid memory byte specification"):
        ResourceQuantityParser.parse_memory_bytes("not-a-number")


def test_yaml_manifest_loader() -> None:
    """Verify YamlManifestLoader behaves correctly on existing and missing files."""
    loader = YamlManifestLoader(K8S_DIR)
    manifest = loader.load_manifest("deployment.yaml")
    assert manifest["kind"] == "Deployment"

    files = loader.list_manifest_files()
    assert "deployment.yaml" in files
    assert "deployment-rust.yaml" in files

    with pytest.raises(FileNotFoundError, match="not found"):
        loader.load_manifest("nonexistent.yaml")


def test_deployment_security_and_qos_validator_on_repo() -> None:
    """Verify DeploymentSecurityAndQoSValidator passes on repo deployments."""
    validator = DeploymentSecurityAndQoSValidator()
    loader = YamlManifestLoader(K8S_DIR)
    violations = validator.validate(loader)

    assert violations == [], f"Expected 0 violations, found: {violations}"


def test_deployment_security_and_qos_validator_detects_violations() -> None:
    """Verify DeploymentSecurityAndQoSValidator detects unsafe deployment configs."""
    # Malformed deployment: replicas < 2, root user, privileged, requests > limits
    bad_deployment: dict[str, Any] = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {"name": "bad-deployment"},
        "spec": {
            "replicas": 1,  # Violation: replicas < 2
            "template": {
                "spec": {
                    "securityContext": {
                        "runAsNonRoot": False,  # Violation
                        "runAsUser": 0,  # Violation: root user
                    },
                    "containers": [
                        {
                            "name": "bad-container",
                            "securityContext": {
                                "allowPrivilegeEscalation": True,  # Violation
                                "capabilities": {"drop": []},  # Violation: missing ALL
                            },
                            "resources": {
                                "requests": {"cpu": "1000m", "memory": "2Gi"},
                                "limits": {"cpu": "500m", "memory": "1Gi"},
                                # Violations: requests > limits (CPU & Memory)
                            },
                            "livenessProbe": {
                                "httpGet": {"path": "/wrong-live", "port": 8000}
                            },
                            "readinessProbe": {
                                "httpGet": {"path": "/wrong-ready", "port": 8000}
                            },
                        }
                    ],
                }
            },
        },
    }

    mock_loader = InMemoryManifestLoader(
        {
            "deployment.yaml": bad_deployment,
            "deployment-rust.yaml": bad_deployment,
        }
    )

    validator = DeploymentSecurityAndQoSValidator()
    violations = validator.validate(mock_loader)

    assert len(violations) > 0
    violation_text = "\n".join(violations)
    assert "Replicas must be >= 2" in violation_text
    assert "runAsNonRoot: true" in violation_text
    assert "runAsUser: 10001" in violation_text
    assert "allowPrivilegeEscalation: false" in violation_text
    assert "drop 'ALL'" in violation_text
    assert "exceeds limit" in violation_text
    assert "Liveness probe must target '/healthz'" in violation_text
    assert "Readiness probe must target '/ready'" in violation_text


def test_cross_resource_integrity_validator_on_repo() -> None:
    """Verify CrossResourceIntegrityValidator passes on real repository manifests."""
    validator = CrossResourceIntegrityValidator()
    loader = YamlManifestLoader(K8S_DIR)
    violations = validator.validate(loader)

    assert violations == [], (
        f"Expected 0 cross-resource violations, found: {violations}"
    )


def test_cross_resource_integrity_validator_detects_mismatches() -> None:
    """Verify CrossResourceIntegrityValidator catches broken cross-references."""
    mock_data: dict[str, dict[str, Any]] = {
        "configmap.yaml": {
            "metadata": {"name": "recyclenet-config"},
            "data": {"PORT": "8000"},
        },
        "secret.yaml": {
            "metadata": {"name": "recyclenet-secret"},
            "data": {"API_KEY": "dummy"},
        },
        "deployment.yaml": {
            "metadata": {"name": "mismatched-dep"},
            "spec": {
                "template": {
                    "metadata": {"labels": {"app": "other-app"}},
                    "spec": {
                        "containers": [
                            {
                                "name": "c1",
                                "ports": [{"containerPort": 9999}],
                                "envFrom": [
                                    {"configMapRef": {"name": "unknown-cm"}},
                                    {"secretRef": {"name": "unknown-sec"}},
                                ],
                                "env": [
                                    {
                                        "name": "MISSING_KEY",
                                        "valueFrom": {
                                            "configMapKeyRef": {
                                                "name": "recyclenet-config",
                                                "key": "NON_EXISTENT_KEY",
                                            }
                                        },
                                    }
                                ],
                            }
                        ]
                    },
                }
            },
        },
        "deployment-rust.yaml": {
            "metadata": {"name": "rust-dep"},
            "spec": {
                "template": {
                    "metadata": {"labels": {"app": "other-app"}},
                    "spec": {"containers": []},
                }
            },
        },
        "service.yaml": {
            "metadata": {"name": "recyclenet-service"},
            "spec": {
                "selector": {"app": "recyclenet"},
                "ports": [{"port": 8000, "targetPort": 8000}],
            },
        },
        "hpa.yaml": {"spec": {"scaleTargetRef": {"name": "recyclenet-deployment"}}},
        "ingress.yaml": {
            "spec": {
                "tls": [{"secretName": "unknown-tls-secret"}],
                "rules": [
                    {
                        "http": {
                            "paths": [
                                {
                                    "backend": {
                                        "service": {
                                            "name": "wrong-service",
                                            "port": {"number": 1234},
                                        }
                                    }
                                }
                            ]
                        }
                    }
                ],
            }
        },
        "kustomization.yaml": {
            "resources": ["configmap.yaml"]  # Missing other manifests
        },
    }

    mock_loader = InMemoryManifestLoader(mock_data)
    validator = CrossResourceIntegrityValidator()
    violations = validator.validate(mock_loader)

    violation_text = "\n".join(violations)
    assert "unknown ConfigMap 'unknown-cm'" in violation_text
    assert "unknown Secret 'unknown-sec'" in violation_text
    assert "NON_EXISTENT_KEY" in violation_text
    assert "Selector 'app=recyclenet' does not match" in violation_text
    assert "targetPort 8000 does not match" in violation_text
    assert (
        "scaleTargetRef name 'recyclenet-deployment' does not match" in violation_text
    )
    assert "wrong-service" in violation_text
    assert "unknown-tls-secret" in violation_text
    assert "omitted from kustomization resources" in violation_text


def test_dockerfile_zero_trust_validator_both_dockerfiles() -> None:
    """Verify DockerfileZeroTrustValidator passes on Dockerfile and Dockerfile.rust."""
    validator = DockerfileZeroTrustValidator(
        root_dir=Path("."),
        dockerfiles={"Dockerfile": 8000, "Dockerfile.rust": 8080},
    )
    loader = YamlManifestLoader(K8S_DIR)
    violations = validator.validate(loader)

    assert violations == [], f"Expected 0 Dockerfile violations, found: {violations}"


def test_dockerfile_zero_trust_validator_detects_violations(
    tmp_path: Path,
) -> None:
    """Verify DockerfileZeroTrustValidator detects missing user or stages."""
    # Write a non-compliant Dockerfile missing stages, user, and exposed port
    bad_df = tmp_path / "Dockerfile"
    bad_df.write_text(
        "FROM ubuntu:latest\nRUN echo hello\n",
        encoding="utf-8",
    )

    validator = DockerfileZeroTrustValidator(
        root_dir=tmp_path,
        dockerfiles={"Dockerfile": 8000},
    )
    loader = YamlManifestLoader(K8S_DIR)
    violations = validator.validate(loader)

    assert len(violations) > 0
    text = "\n".join(violations)
    assert "AS builder" in text
    assert "AS runtime" in text
    assert "appuser" in text
    assert "USER appuser" in text
    assert "EXPOSE 8000" in text


def test_kubernetes_orchestration_auditor_full_audit() -> None:
    """Verify KubernetesOrchestrationAuditor execution and report generation."""
    auditor = KubernetesOrchestrationAuditor()
    report: AuditReport = auditor.run_audit()

    assert report.is_valid is True
    assert len(report.violations) == 0
    assert report.checked_components == 3

    markdown = report.format_markdown()
    assert "# Kubernetes Manifest & Zero-Trust Security Audit" in markdown
    assert "Audit Status**: `PASSED`" in markdown
    assert "Integrity Confirmed" in markdown


def test_hpa_v2_simulator_compute_desired_replicas() -> None:
    """Verify HPA v2 desired replica calculation, deadband, and boundary clamps."""
    simulator = HpaV2ControlLoopSimulator(
        min_replicas=2,
        max_replicas=5,
        target_cpu_utilization_pct=70.0,
    )

    # 1. Kubernetes 10% deadband tolerance around 1.0 ratio (63% to 77% CPU)
    # Target = 70%. At 70% CPU, ratio = 1.0 -> keep current
    assert (
        simulator.compute_desired_replicas(
            current_replicas=2, current_cpu_utilization_pct=70.0
        )
        == 2
    )
    # At 75% CPU, ratio = 75/70 = 1.071 (within 0.90..1.10) -> keep current
    assert (
        simulator.compute_desired_replicas(
            current_replicas=2, current_cpu_utilization_pct=75.0
        )
        == 2
    )
    # At 65% CPU, ratio = 65/70 = 0.928 (within 0.90..1.10) -> keep current
    assert (
        simulator.compute_desired_replicas(
            current_replicas=3, current_cpu_utilization_pct=65.0
        )
        == 3
    )

    # 2. Scale-out beyond deadband
    # At 98% CPU, ratio = 98/70 = 1.4 -> ceil(2 * 1.4) = 3
    assert (
        simulator.compute_desired_replicas(
            current_replicas=2, current_cpu_utilization_pct=98.0
        )
        == 3
    )
    # At 115% CPU, ratio = 115/70 = 1.642 -> ceil(3 * 1.642) = 5
    assert (
        simulator.compute_desired_replicas(
            current_replicas=3, current_cpu_utilization_pct=115.0
        )
        == 5
    )

    # 3. Maximum replica clamping (maxReplicas = 5)
    # At 250% CPU, raw desired = ceil(2 * 250/70) = 8 -> clamped to 5
    assert (
        simulator.compute_desired_replicas(
            current_replicas=2, current_cpu_utilization_pct=250.0
        )
        == 5
    )

    # 4. Minimum replica clamping (minReplicas = 2)
    # At 15% CPU, raw desired = ceil(5 * 15/70) = 2 -> clamped to minReplicas (2)
    assert (
        simulator.compute_desired_replicas(
            current_replicas=5, current_cpu_utilization_pct=15.0
        )
        == 2
    )
    # At 5% CPU, raw desired = ceil(2 * 5/70) = 1 -> clamped to minReplicas (2)
    assert (
        simulator.compute_desired_replicas(
            current_replicas=2, current_cpu_utilization_pct=5.0
        )
        == 2
    )

    # Error handling
    with pytest.raises(ValueError, match="current_replicas must be"):
        simulator.compute_desired_replicas(
            current_replicas=0, current_cpu_utilization_pct=50.0
        )

    with pytest.raises(ValueError, match="cannot be negative"):
        simulator.compute_desired_replicas(
            current_replicas=2, current_cpu_utilization_pct=-10.0
        )


def test_hpa_v2_simulator_k6_workload_trajectory() -> None:
    """Verify HPA simulation replicates the 4 k6 stages and scales 2 -> 3 -> 5 -> 2."""
    simulator = HpaV2ControlLoopSimulator(
        min_replicas=2,
        max_replicas=5,
        target_cpu_utilization_pct=70.0,
        manifest_path=K8S_DIR / "hpa.yaml",
    )

    steps: list[HpaSimulationStep] = simulator.simulate_k6_workload()
    assert len(steps) == 4

    # Stage 1: Warm-up (10 VUs)
    assert steps[0].stage_name == "Warm-up (10 VUs)"
    assert steps[0].target_vus == 10
    assert steps[0].current_replicas == 2
    assert steps[0].desired_replicas == 2
    assert "Stable" in steps[0].scaling_action

    # Stage 2: Ramp-up (30 VUs) -> CPU > 70% -> triggers scale-out to 3
    assert steps[1].stage_name == "Ramp-up (30 VUs)"
    assert steps[1].target_vus == 30
    assert steps[1].current_replicas == 2
    assert steps[1].desired_replicas == 3
    assert "Scale-Out (2 -> 3" in steps[1].scaling_action

    # Stage 3: Peak Load (60 VUs) -> CPU > 70% -> expands to max cap 5
    assert steps[2].stage_name == "Peak Load (60 VUs)"
    assert steps[2].target_vus == 60
    assert steps[2].current_replicas == 3
    assert steps[2].desired_replicas == 5
    assert "Scale-Out (3 -> 5" in steps[2].scaling_action

    # Stage 4: Cooldown (0 VUs) -> traffic drops -> scales back to minReplicas 2
    assert steps[3].stage_name == "Cooldown (0 VUs)"
    assert steps[3].target_vus == 0
    assert steps[3].current_replicas == 5
    assert steps[3].desired_replicas == 2
    assert "Scale-In (5 -> 2" in steps[3].scaling_action

    # Markdown table formatting
    table = simulator.format_simulation_table(steps)
    assert "Warm-up (10 VUs)" in table
    assert "Peak Load (60 VUs)" in table
    assert "Scale-Out (3 -> 5 Max Replicas)" in table


def test_minikube_lab_plan_builder() -> None:
    """Verify MinikubeLabPlanBuilder provides all 5 structured lab steps in sequence."""
    plan: list[LabStep] = MinikubeLabPlanBuilder.build_plan()

    assert len(plan) == 5
    step_numbers = [s.step_number for s in plan]
    assert step_numbers == [1, 2, 3, 4, 5]

    assert "minikube start --driver=docker" in plan[0].command
    assert "minikube addons enable metrics-server" in plan[1].command
    assert "eval $(minikube docker-env)" in plan[2].command
    assert "kubectl apply -f k8s/" in plan[3].command
    assert "k6 run benchmarks/k6_load_test.js" in plan[4].command

    table = MinikubeLabPlanBuilder.format_plan_table(plan)
    assert "### Local Minikube Cloud-Native Lab Runbook" in table
    assert "minikube start --driver=docker" in table
    assert "k6 run benchmarks/k6_load_test.js" in table


def test_orchestrator_main_entrypoint(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify CLI entrypoint main() runs successfully and produces expected output."""
    exit_code = orchestrator_main()
    assert exit_code == 0

    captured = capsys.readouterr()
    assert "Kubernetes Manifest & Zero-Trust Security Audit" in captured.out
    assert "HPA v2 Workload Trajectory Simulation" in captured.out
    assert "Local Minikube Cloud-Native Lab Runbook" in captured.out
