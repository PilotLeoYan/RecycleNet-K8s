"""Kubernetes orchestration, manifest validation, and HPA simulation engine.

Provides SOLID-compliant manifest validation, zero-trust container security auditing,
declarative cross-resource integrity checking, and an autoscaling simulation model
matching the k6 load testing stages for RecycleNet-K8s.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import yaml

# ==============================================================================
# Protocols (Interface Segregation & Dependency Inversion)
# ==============================================================================


@runtime_checkable
class ManifestLoaderProtocol(Protocol):
    """Protocol defining the interface for loading Kubernetes manifests."""

    def load_manifest(self, filename: str) -> dict[str, Any]:
        """Load and return the parsed YAML contents of a manifest file.

        Args:
            filename: Name of the YAML manifest file (e.g. 'deployment.yaml').

        Returns:
            Dictionary representation of the Kubernetes manifest.
        """
        ...


@runtime_checkable
class ValidationRuleProtocol(Protocol):
    """Protocol defining a single, decoupled Kubernetes manifest validation rule."""

    def validate(self, loader: ManifestLoaderProtocol) -> list[str]:
        """Execute validation against manifests loaded via the provided loader.

        Args:
            loader: ManifestLoaderProtocol instance providing access to YAML documents.

        Returns:
            List of violation descriptions. Empty list signifies complete compliance.
        """
        ...


# ==============================================================================
# Domain Models & Value Objects
# ==============================================================================


@dataclass(frozen=True)
class AuditReport:
    """Aggregated result of the Kubernetes orchestration and manifest audit."""

    is_valid: bool
    violations: list[str] = field(default_factory=list)
    checked_components: int = 0

    def format_markdown(self) -> str:
        """Format the audit report as a structured Markdown document."""
        status_badge = "PASSED" if self.is_valid else "FAILED"
        lines = [
            "# Kubernetes Manifest & Zero-Trust Security Audit",
            "",
            f"**Audit Status**: `{status_badge}` | "
            f"**Rule Sets Audited**: `{self.checked_components}` | "
            f"**Total Violations**: `{len(self.violations)}`",
            "",
        ]
        if self.violations:
            lines.append("### Detected Policy Violations:")
            for index, violation in enumerate(self.violations, start=1):
                lines.append(f"{index}. [VIOLATION] {violation}")
        else:
            lines.append(
                "> **Integrity Confirmed**: All manifests, QoS constraints, "
                "cross-resource bindings, and container security contexts are valid."
            )
        return "\n".join(lines)


@dataclass(frozen=True)
class HpaSimulationStep:
    """Simulation step capturing the state of an HPA v2 autoscaling control loop."""

    stage_name: str
    duration: str
    target_vus: int
    observed_cpu_pct: float
    current_replicas: int
    desired_replicas: int
    scaling_action: str
    description: str


@dataclass(frozen=True)
class LabStep:
    """Structured instruction step for executing local Minikube cluster workflows."""

    step_number: int
    name: str
    command: str
    description: str


# ==============================================================================
# Resource Quantity Parser (Single Responsibility Principle)
# ==============================================================================


class ResourceQuantityParser:
    """Parser for Kubernetes CPU millicores and memory byte quantity notations."""

    _BINARY_SI_MULTIPLIERS: dict[str, int] = {
        "Ei": 1024**6,
        "Pi": 1024**5,
        "Ti": 1024**4,
        "Gi": 1024**3,
        "Mi": 1024**2,
        "Ki": 1024,
    }

    _DECIMAL_SI_MULTIPLIERS: dict[str, int] = {
        "E": 1000**6,
        "P": 1000**5,
        "T": 1000**4,
        "G": 1000**3,
        "M": 1000**2,
        "k": 1000,
    }

    @staticmethod
    def parse_cpu_millicores(value: str | int | float) -> int:
        """Parse a Kubernetes CPU quantity into integer millicores.

        Examples:
            - "250m" -> 250
            - "1000m" -> 1000
            - "1" -> 1000
            - 1 -> 1000
            - 0.5 -> 500
            - "0.5" -> 500

        Args:
            value: CPU quantity representation.

        Returns:
            Integer millicores (1 CPU core = 1000 millicores).

        Raises:
            ValueError: If the quantity cannot be parsed or is non-positive.
        """
        if value is None:
            raise ValueError("CPU quantity cannot be None")

        if isinstance(value, (int, float)):
            if value <= 0:
                raise ValueError(f"CPU quantity must be positive, got {value}")
            return int(round(value * 1000))

        raw = str(value).strip()
        if not raw:
            raise ValueError("CPU quantity cannot be empty")

        if raw.endswith("m"):
            millicores_str = raw[:-1].strip()
            try:
                m_val = float(millicores_str)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid millicores format in CPU quantity: '{value}'"
                ) from exc
            if m_val <= 0:
                raise ValueError(f"CPU quantity must be positive, got {value}")
            return int(round(m_val))

        try:
            core_val = float(raw)
        except ValueError as exc:
            raise ValueError(
                f"Invalid numerical format in CPU quantity: '{value}'"
            ) from exc

        if core_val <= 0:
            raise ValueError(f"CPU quantity must be positive, got {value}")
        return int(round(core_val * 1000))

    @staticmethod
    def parse_memory_bytes(value: str | int) -> int:
        """Parse a Kubernetes memory quantity into exact integer bytes.

        Examples:
            - "64Mi" -> 67_108_864
            - "512Mi" -> 536_870_912
            - "1Gi" -> 1_073_741_824
            - "128Mi" -> 134_217_728
            - 1048576 -> 1048576

        Args:
            value: Memory quantity representation.

        Returns:
            Exact integer byte count.

        Raises:
            ValueError: If the quantity cannot be parsed or is non-positive.
        """
        if value is None:
            raise ValueError("Memory quantity cannot be None")

        if isinstance(value, int):
            if value <= 0:
                raise ValueError(f"Memory bytes must be positive, got {value}")
            return value

        raw = str(value).strip()
        if not raw:
            raise ValueError("Memory quantity cannot be empty")

        # Binary SI multipliers (e.g. Ki, Mi, Gi)
        for suffix, multiplier in ResourceQuantityParser._BINARY_SI_MULTIPLIERS.items():
            if raw.endswith(suffix):
                num_str = raw[: -len(suffix)].strip()
                try:
                    num_val = float(num_str)
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid memory value prefix in '{value}'"
                    ) from exc
                if num_val <= 0:
                    raise ValueError(f"Memory quantity must be positive, got {value}")
                return int(round(num_val * multiplier))

        # Decimal SI multipliers (e.g. k, M, G)
        for (
            suffix,
            multiplier,
        ) in ResourceQuantityParser._DECIMAL_SI_MULTIPLIERS.items():
            if raw.endswith(suffix):
                num_str = raw[: -len(suffix)].strip()
                try:
                    num_val = float(num_str)
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid memory value prefix in '{value}'"
                    ) from exc
                if num_val <= 0:
                    raise ValueError(f"Memory quantity must be positive, got {value}")
                return int(round(num_val * multiplier))

        # Plain integer bytes string
        try:
            int_val = int(raw)
        except ValueError as exc:
            raise ValueError(f"Invalid memory byte specification: '{value}'") from exc

        if int_val <= 0:
            raise ValueError(f"Memory bytes must be positive, got {value}")
        return int_val


# ==============================================================================
# Manifest Loader Implementation
# ==============================================================================


class YamlManifestLoader:
    """Loads declarative Kubernetes manifests from a configured local directory."""

    def __init__(self, k8s_dir: Path | str = Path("k8s")) -> None:
        self.k8s_dir = Path(k8s_dir)

    def load_manifest(self, filename: str) -> dict[str, Any]:
        """Load and parse a YAML file from the manifests directory.

        Args:
            filename: Target manifest filename.

        Returns:
            Parsed YAML dictionary.

        Raises:
            FileNotFoundError: If the manifest file is not found.
            ValueError: If the file does not parse to a valid dictionary.
        """
        manifest_path = self.k8s_dir / filename
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Manifest file '{filename}' not found at {manifest_path}"
            )
        with manifest_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError(
                f"Manifest '{filename}' must be a YAML dictionary, "
                f"got {type(data).__name__}"
            )
        return data

    def list_manifest_files(self) -> list[str]:
        """List all manifest filenames in the directory."""
        if not self.k8s_dir.is_dir():
            return []
        files = [p.name for p in self.k8s_dir.glob("*.yaml")] + [
            p.name for p in self.k8s_dir.glob("*.yml")
        ]
        return sorted(set(files))


# ==============================================================================
# Manifest Validation Rules (SRP & OCP)
# ==============================================================================


class DeploymentSecurityAndQoSValidator:
    """Validates deployment security contexts, probes, and resource constraints."""

    DEPLOYMENT_FILES: tuple[str, ...] = ("deployment.yaml", "deployment-rust.yaml")

    def validate(self, loader: ManifestLoaderProtocol) -> list[str]:
        """Audit Deployment securityContext, QoS guarantees, and health probes."""
        violations: list[str] = []

        for filename in self.DEPLOYMENT_FILES:
            try:
                manifest = loader.load_manifest(filename)
            except Exception as exc:
                violations.append(f"[{filename}] Failed to load manifest: {exc}")
                continue

            spec = manifest.get("spec", {})
            replicas = spec.get("replicas")
            if replicas is None or replicas < 2:
                violations.append(
                    f"[{filename}] Replicas must be >= 2 for HA (found: {replicas})"
                )

            template_spec = spec.get("template", {}).get("spec", {})
            pod_sc = template_spec.get("securityContext", {})

            # Pod-level security context verification
            if pod_sc.get("runAsNonRoot") is not True:
                violations.append(
                    f"[{filename}] Pod securityContext must set 'runAsNonRoot: true'"
                )
            if pod_sc.get("runAsUser") != 10001:
                violations.append(
                    f"[{filename}] Pod securityContext must set 'runAsUser: 10001' "
                    f"(found: {pod_sc.get('runAsUser')})"
                )

            containers = template_spec.get("containers", [])
            if not containers:
                violations.append(
                    f"[{filename}] Deployment specification defines no containers"
                )
                continue

            for container in containers:
                c_name = container.get("name", "unknown")
                c_sc = container.get("securityContext", {})

                if c_sc.get("allowPrivilegeEscalation") is not False:
                    violations.append(
                        f"[{filename} / {c_name}] Container must set "
                        "'allowPrivilegeEscalation: false'"
                    )

                dropped_caps = c_sc.get("capabilities", {}).get("drop", [])
                if "ALL" not in dropped_caps:
                    violations.append(
                        f"[{filename} / {c_name}] Container capabilities must "
                        "drop 'ALL'"
                    )

                # Compute QoS validation (requests <= limits)
                resources = container.get("resources", {})
                reqs = resources.get("requests", {})
                lims = resources.get("limits", {})

                if "cpu" in reqs and "cpu" in lims:
                    try:
                        req_cpu = ResourceQuantityParser.parse_cpu_millicores(
                            reqs["cpu"]
                        )
                        lim_cpu = ResourceQuantityParser.parse_cpu_millicores(
                            lims["cpu"]
                        )
                        if req_cpu > lim_cpu:
                            violations.append(
                                f"[{filename} / {c_name}] CPU request ({req_cpu}m) "
                                f"exceeds limit ({lim_cpu}m)"
                            )
                    except ValueError as exc:
                        violations.append(
                            f"[{filename} / {c_name}] Invalid CPU configuration: {exc}"
                        )
                else:
                    violations.append(
                        f"[{filename} / {c_name}] Container must specify both CPU "
                        "requests and limits"
                    )

                if "memory" in reqs and "memory" in lims:
                    try:
                        req_mem = ResourceQuantityParser.parse_memory_bytes(
                            reqs["memory"]
                        )
                        lim_mem = ResourceQuantityParser.parse_memory_bytes(
                            lims["memory"]
                        )
                        if req_mem > lim_mem:
                            violations.append(
                                f"[{filename} / {c_name}] Memory request ({req_mem}B) "
                                f"exceeds limit ({lim_mem}B), risking OOMKilled"
                            )
                    except ValueError as exc:
                        violations.append(
                            f"[{filename} / {c_name}] Invalid memory "
                            f"configuration: {exc}"
                        )
                else:
                    violations.append(
                        f"[{filename} / {c_name}] Container must specify both memory "
                        "requests and limits"
                    )

                # Health Probes
                liveness = container.get("livenessProbe", {})
                live_path = liveness.get("httpGet", {}).get("path")
                if live_path != "/healthz":
                    violations.append(
                        f"[{filename} / {c_name}] Liveness probe must target "
                        f"'/healthz' (found: {live_path})"
                    )

                readiness = container.get("readinessProbe", {})
                ready_path = readiness.get("httpGet", {}).get("path")
                if ready_path != "/ready":
                    violations.append(
                        f"[{filename} / {c_name}] Readiness probe must target "
                        f"'/ready' (found: {ready_path})"
                    )

        return violations


class CrossResourceIntegrityValidator:
    """Validates integrity across ConfigMaps, Secrets, Services, HPAs, and Ingress."""

    def validate(self, loader: ManifestLoaderProtocol) -> list[str]:
        """Audit cross-resource bindings, selectors, and reference consistency."""
        violations: list[str] = []

        # 1. Load ConfigMap & Secret definitions
        try:
            configmap = loader.load_manifest("configmap.yaml")
            cm_name = configmap.get("metadata", {}).get("name")
            cm_keys = set(configmap.get("data", {}).keys())
        except Exception as exc:
            violations.append(f"[configmap.yaml] Unable to load configmap: {exc}")
            cm_name, cm_keys = None, set()

        try:
            secret = loader.load_manifest("secret.yaml")
            secret_name = secret.get("metadata", {}).get("name")
            secret_keys = set(secret.get("data", {}).keys())
        except Exception as exc:
            violations.append(f"[secret.yaml] Unable to load secret: {exc}")
            secret_name, secret_keys = None, set()

        # 2. Check Deployments environment variable references
        deployments = ["deployment.yaml", "deployment-rust.yaml"]
        container_ports: set[int] = set()
        deployment_pod_labels: dict[str, dict[str, str]] = {}

        for dep_file in deployments:
            try:
                dep_data = loader.load_manifest(dep_file)
            except Exception as exc:
                violations.append(f"[{dep_file}] Unable to load deployment: {exc}")
                continue

            template_spec = dep_data.get("spec", {}).get("template", {})
            pod_labels = template_spec.get("metadata", {}).get("labels", {})
            deployment_pod_labels[dep_file] = pod_labels

            for container in template_spec.get("spec", {}).get("containers", []):
                for p in container.get("ports", []):
                    if "containerPort" in p:
                        container_ports.add(p["containerPort"])

                # envFrom bindings
                for env_from in container.get("envFrom", []):
                    ref_cm = env_from.get("configMapRef", {}).get("name")
                    if ref_cm and ref_cm != cm_name:
                        violations.append(
                            f"[{dep_file}] envFrom references unknown ConfigMap "
                            f"'{ref_cm}'"
                        )
                    ref_sec = env_from.get("secretRef", {}).get("name")
                    if ref_sec and ref_sec != secret_name:
                        violations.append(
                            f"[{dep_file}] envFrom references unknown Secret "
                            f"'{ref_sec}'"
                        )

                # env key references
                for env_item in container.get("env", []):
                    val_from = env_item.get("valueFrom", {})
                    cm_ref = val_from.get("configMapKeyRef")
                    if cm_ref:
                        if cm_ref.get("name") != cm_name:
                            violations.append(
                                f"[{dep_file}] configMapKeyRef references unknown "
                                f"ConfigMap '{cm_ref.get('name')}'"
                            )
                        elif cm_ref.get("key") not in cm_keys:
                            violations.append(
                                f"[{dep_file}] configMapKeyRef key "
                                f"'{cm_ref.get('key')}' not found in CM '{cm_name}'"
                            )

                    sec_ref = val_from.get("secretKeyRef")
                    if sec_ref:
                        if sec_ref.get("name") != secret_name:
                            violations.append(
                                f"[{dep_file}] secretKeyRef references unknown "
                                f"Secret '{sec_ref.get('name')}'"
                            )
                        elif sec_ref.get("key") not in secret_keys:
                            violations.append(
                                f"[{dep_file}] secretKeyRef key '{sec_ref.get('key')}' "
                                f"not found in Secret '{secret_name}'"
                            )

        # 3. Service selector and port alignment
        try:
            service = loader.load_manifest("service.yaml")
            svc_name = service.get("metadata", {}).get("name")
            svc_selector = service.get("spec", {}).get("selector", {})
            svc_ports = service.get("spec", {}).get("ports", [])
            svc_port_numbers = {p.get("port") for p in svc_ports}

            for dep_file, pod_labels in deployment_pod_labels.items():
                for sel_k, sel_v in svc_selector.items():
                    if pod_labels.get(sel_k) != sel_v:
                        violations.append(
                            f"[service.yaml] Selector '{sel_k}={sel_v}' does not match "
                            f"pod template label in '{dep_file}'"
                        )

            for p in svc_ports:
                target_port = p.get("targetPort")
                if target_port not in container_ports:
                    violations.append(
                        f"[service.yaml] targetPort {target_port} does not match "
                        f"any container port ({container_ports})"
                    )
        except Exception as exc:
            violations.append(f"[service.yaml] Service validation error: {exc}")
            svc_name, svc_port_numbers = None, set()

        # 4. HPA scaleTargetRef alignment
        try:
            hpa = loader.load_manifest("hpa.yaml")
            scale_target = hpa.get("spec", {}).get("scaleTargetRef", {})
            target_name = scale_target.get("name")
            deployment_manifest = loader.load_manifest("deployment.yaml")
            dep_name = deployment_manifest.get("metadata", {}).get("name")

            if target_name != dep_name:
                violations.append(
                    f"[hpa.yaml] scaleTargetRef name '{target_name}' does not match "
                    f"deployment metadata name '{dep_name}'"
                )
        except Exception as exc:
            violations.append(f"[hpa.yaml] HPA validation error: {exc}")

        # 5. Ingress backend routing and TLS secret
        try:
            ingress = loader.load_manifest("ingress.yaml")
            rules = ingress.get("spec", {}).get("rules", [])
            for rule in rules:
                for path_entry in rule.get("http", {}).get("paths", []):
                    backend_svc = (
                        path_entry.get("backend", {}).get("service", {}).get("name")
                    )
                    backend_port = (
                        path_entry.get("backend", {})
                        .get("service", {})
                        .get("port", {})
                        .get("number")
                    )
                    if backend_svc != svc_name:
                        violations.append(
                            f"[ingress.yaml] Backend service '{backend_svc}' does not "
                            f"match Service name '{svc_name}'"
                        )
                    if backend_port not in svc_port_numbers:
                        violations.append(
                            f"[ingress.yaml] Backend port {backend_port} not found in "
                            f"Service ports {svc_port_numbers}"
                        )

            for tls in ingress.get("spec", {}).get("tls", []):
                tls_secret = tls.get("secretName")
                if tls_secret != secret_name:
                    violations.append(
                        f"[ingress.yaml] Ingress TLS secretName '{tls_secret}' does "
                        f"not match Secret name '{secret_name}'"
                    )
        except Exception as exc:
            violations.append(f"[ingress.yaml] Ingress validation error: {exc}")

        # 6. Kustomization manifest inventory completeness
        try:
            kustomization = loader.load_manifest("kustomization.yaml")
            resources = set(kustomization.get("resources", []))

            # If loader provides list_manifest_files or has k8s_dir
            if hasattr(loader, "list_manifest_files"):
                all_manifests = set(loader.list_manifest_files())  # type: ignore[attr-defined]
            elif hasattr(loader, "k8s_dir"):
                k_dir = Path(getattr(loader, "k8s_dir"))
                all_manifests = {
                    p.name
                    for p in list(k_dir.glob("*.yaml")) + list(k_dir.glob("*.yml"))
                }
            else:
                all_manifests = {
                    "configmap.yaml",
                    "secret.yaml",
                    "deployment.yaml",
                    "deployment-rust.yaml",
                    "service.yaml",
                    "hpa.yaml",
                    "ingress.yaml",
                    "kustomization.yaml",
                }

            expected_resources = all_manifests - {"kustomization.yaml"}
            missing_resources = expected_resources - resources
            for missing in sorted(missing_resources):
                violations.append(
                    f"[kustomization.yaml] Manifest '{missing}' is in directory "
                    "but omitted from kustomization resources"
                )
        except Exception as exc:
            violations.append(
                f"[kustomization.yaml] Kustomization validation error: {exc}"
            )

        return violations


class DockerfileZeroTrustValidator:
    """Validates Dockerfiles for multi-stage build, UID, and port exposition."""

    def __init__(
        self,
        root_dir: Path | str = Path("."),
        dockerfiles: dict[str, int] | None = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.dockerfiles: dict[str, int] = (
            dockerfiles
            if dockerfiles is not None
            else {"Dockerfile": 8000, "Dockerfile.rust": 8080}
        )

    def validate(self, loader: ManifestLoaderProtocol) -> list[str]:
        """Audit Dockerfiles for zero-trust security compliance."""
        violations: list[str] = []

        # If loader has a k8s_dir, check its parent if root_dir lacks Dockerfile
        effective_dir = self.root_dir
        if hasattr(loader, "k8s_dir"):
            k8s_path = Path(getattr(loader, "k8s_dir"))
            if (
                not (effective_dir / "Dockerfile").exists()
                and (k8s_path.parent / "Dockerfile").exists()
            ):
                effective_dir = k8s_path.parent

        for filename, expected_port in self.dockerfiles.items():
            dockerfile_path = effective_dir / filename
            if not dockerfile_path.is_file():
                violations.append(
                    f"[{filename}] Dockerfile not found at {dockerfile_path}"
                )
                continue

            content = dockerfile_path.read_text(encoding="utf-8")

            # Multi-stage build stages
            if "AS builder" not in content:
                violations.append(
                    f"[{filename}] Missing multi-stage builder stage ('AS builder')"
                )
            if "AS runtime" not in content:
                violations.append(
                    f"[{filename}] Missing multi-stage runtime stage ('AS runtime')"
                )

            # Zero-trust unprivileged user
            if "appuser" not in content or "10001" not in content:
                violations.append(
                    f"[{filename}] Missing zero-trust user 'appuser' with UID 10001"
                )

            # Drop privileges
            if "USER appuser" not in content:
                violations.append(
                    f"[{filename}] Dockerfile must drop root ('USER appuser')"
                )

            # Expose target port
            if f"EXPOSE {expected_port}" not in content:
                violations.append(
                    f"[{filename}] Dockerfile must expose port {expected_port} "
                    f"('EXPOSE {expected_port}')"
                )

        return violations


# ==============================================================================
# Orchestration Auditor (Composite Aggregator)
# ==============================================================================


class KubernetesOrchestrationAuditor:
    """Aggregates and executes validation rules against manifests and build specs."""

    def __init__(
        self,
        loader: ManifestLoaderProtocol | None = None,
        rules: list[ValidationRuleProtocol] | None = None,
    ) -> None:
        self.loader = loader or YamlManifestLoader()
        self.rules: list[ValidationRuleProtocol] = (
            rules
            if rules is not None
            else [
                DeploymentSecurityAndQoSValidator(),
                CrossResourceIntegrityValidator(),
                DockerfileZeroTrustValidator(),
            ]
        )

    def run_audit(self) -> AuditReport:
        """Run all configured validation rules and compile the composite AuditReport."""
        all_violations: list[str] = []
        for rule in self.rules:
            rule_violations = rule.validate(self.loader)
            all_violations.extend(rule_violations)

        return AuditReport(
            is_valid=len(all_violations) == 0,
            violations=all_violations,
            checked_components=len(self.rules),
        )


# ==============================================================================
# HPA v2 Control Loop Simulator
# ==============================================================================


class HpaV2ControlLoopSimulator:
    """Simulates Kubernetes HPA v2 autoscaling dynamics under k6 load testing."""

    def __init__(
        self,
        min_replicas: int = 2,
        max_replicas: int = 5,
        target_cpu_utilization_pct: float = 70.0,
        scale_up_behavior: dict[str, Any] | None = None,
        scale_down_behavior: dict[str, Any] | None = None,
        manifest_path: Path | str | None = None,
    ) -> None:
        # Load from manifest if provided or discoverable
        if manifest_path is not None:
            path = Path(manifest_path)
            if path.is_file():
                with path.open("r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                spec = data.get("spec", {})
                min_replicas = spec.get("minReplicas", min_replicas)
                max_replicas = spec.get("maxReplicas", max_replicas)
                for metric in spec.get("metrics", []):
                    if metric.get("resource", {}).get("name") == "cpu":
                        target_cpu_utilization_pct = float(
                            metric.get("resource", {})
                            .get("target", {})
                            .get("averageUtilization", target_cpu_utilization_pct)
                        )
                behavior = spec.get("behavior", {})
                scale_up_behavior = behavior.get("scaleUp", scale_up_behavior)
                scale_down_behavior = behavior.get("scaleDown", scale_down_behavior)

        self.min_replicas = min_replicas
        self.max_replicas = max_replicas
        self.target_cpu_utilization_pct = target_cpu_utilization_pct
        self.scale_up_behavior = scale_up_behavior or {}
        self.scale_down_behavior = scale_down_behavior or {}

    def compute_desired_replicas(
        self,
        current_replicas: int,
        current_cpu_utilization_pct: float,
    ) -> int:
        """Compute desired replica count using Kubernetes HPA v2 control loop algorithm.

        Formula:
            ratio = current_cpu_utilization_pct / target_cpu_utilization_pct
            Tolerance deadband: if 0.90 <= ratio <= 1.10, desired = current_replicas
            Otherwise: desired = ceil(current_replicas * ratio)
            Clamped to [min_replicas, max_replicas]

        Args:
            current_replicas: Currently running pod replica count.
            current_cpu_utilization_pct: Observed cluster average CPU utilization (%).

        Returns:
            Target replica count bounded by min and max replicas.
        """
        if current_replicas <= 0:
            raise ValueError("current_replicas must be greater than or equal to 1")
        if current_cpu_utilization_pct < 0:
            raise ValueError("current_cpu_utilization_pct cannot be negative")

        ratio = current_cpu_utilization_pct / self.target_cpu_utilization_pct

        # Kubernetes standard 10% tolerance deadband
        if 0.90 <= ratio <= 1.10:
            return current_replicas

        raw_desired = math.ceil(current_replicas * ratio)
        return max(self.min_replicas, min(self.max_replicas, raw_desired))

    def simulate_k6_workload(self) -> list[HpaSimulationStep]:
        """Simulate the 4 k6 stages defined in benchmarks/k6_load_test.js.

        Stages:
            1. Warm-up (10 VUs, 15s) -> CPU ~35%, stays at minReplicas (2)
            2. Ramp-up (30 VUs, 30s) -> CPU ~98%, scale-out to 3 replicas
            3. Peak Load (60 VUs, 45s) -> CPU ~115%, scale-out to maxReplicas (5)
            4. Cooldown (0 VUs, 15s) -> CPU ~15%, scales back down to minReplicas (2)

        Returns:
            List of 4 HpaSimulationStep records tracing the scaling trajectory.
        """
        steps: list[HpaSimulationStep] = []
        current = self.min_replicas

        stages = [
            (
                "Warm-up (10 VUs)",
                "15s",
                10,
                35.0,
                "Stable (No Scale)",
                "Warm-up load absorbed by baseline pods; CPU stays below target",
            ),
            (
                "Ramp-up (30 VUs)",
                "30s",
                30,
                98.0,
                "Scale-Out (2 -> 3 Replicas)",
                "Traffic ramp pushes CPU to 98%; HPA controller triggers scale-out",
            ),
            (
                "Peak Load (60 VUs)",
                "45s",
                60,
                115.0,
                "Scale-Out (3 -> 5 Max Replicas)",
                "Peak load elevates CPU to 115%; HPA expands to max cap (5 pods)",
            ),
            (
                "Cooldown (0 VUs)",
                "15s",
                0,
                15.0,
                "Scale-In (5 -> 2 Min Replicas)",
                "Traffic drops to 0; stabilization elapses; returns to minReplicas",
            ),
        ]

        for name, duration, vus, cpu_pct, action_label, desc in stages:
            desired = self.compute_desired_replicas(current, cpu_pct)
            steps.append(
                HpaSimulationStep(
                    stage_name=name,
                    duration=duration,
                    target_vus=vus,
                    observed_cpu_pct=cpu_pct,
                    current_replicas=current,
                    desired_replicas=desired,
                    scaling_action=action_label,
                    description=desc,
                )
            )
            current = desired

        return steps

    def format_simulation_table(self, steps: list[HpaSimulationStep]) -> str:
        """Render the HPA simulation trajectory as a Markdown table."""
        lines = [
            "### HPA v2 Workload Trajectory Simulation (`benchmarks/k6_load_test.js`)",
            "",
            "| Stage | Duration | VUs | CPU Util (%) | Current | Desired | "
            "Scaling Action | Architectural Impact |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- |",
        ]
        for step in steps:
            lines.append(
                f"| **{step.stage_name}** | {step.duration} | {step.target_vus} | "
                f"{step.observed_cpu_pct:.1f}% | {step.current_replicas} | "
                f"{step.desired_replicas} | `{step.scaling_action}` | "
                f"{step.description} |"
            )
        return "\n".join(lines)


# ==============================================================================
# Local Minikube Lab Plan Builder
# ==============================================================================


class MinikubeLabPlanBuilder:
    """Builder providing structured steps to deploy and test RecycleNet on Minikube."""

    @staticmethod
    def build_plan() -> list[LabStep]:
        """Construct the 5 structured developer lab execution steps."""
        return [
            LabStep(
                step_number=1,
                name="Start Minikube Cluster",
                command="minikube start --driver=docker",
                description="Initialize local single-node cluster via Docker",
            ),
            LabStep(
                step_number=2,
                name="Enable Metrics Server",
                command="minikube addons enable metrics-server",
                description="Enable metrics-server addon required for HPA v2 metrics",
            ),
            LabStep(
                step_number=3,
                name="Build Local Docker Image",
                command="eval $(minikube docker-env) && "
                "docker build -t recyclenet-api:local .",
                description="Build multi-stage container directly into Minikube daemon",
            ),
            LabStep(
                step_number=4,
                name="Apply Declarative Manifests",
                command="kubectl apply -f k8s/",
                description="Deploy manifests: ConfigMap, Secret, Workloads, Network",
            ),
            LabStep(
                step_number=5,
                name="Execute k6 SLO & HPA Load Test",
                command="k6 run benchmarks/k6_load_test.js",
                description="Execute 4-stage k6 load test verifying SLOs and HPA scale",
            ),
        ]

    @staticmethod
    def format_plan_table(steps: list[LabStep]) -> str:
        """Render the Minikube lab execution plan as a Markdown table."""
        lines = [
            "### Local Minikube Cloud-Native Lab Runbook",
            "",
            "| Step | Action Name | Shell Command | Operational Purpose |",
            "| :---: | :--- | :--- | :--- |",
        ]
        for s in steps:
            lines.append(
                f"| {s.step_number} | **{s.name}** | `{s.command}` | {s.description} |"
            )
        return "\n".join(lines)


# ==============================================================================
# CLI Entrypoint
# ==============================================================================


def main() -> int:
    """Execute manifest auditing, simulate HPA v2, and print Markdown reports."""
    loader = YamlManifestLoader(k8s_dir=Path("k8s"))
    auditor = KubernetesOrchestrationAuditor(loader=loader)
    report = auditor.run_audit()

    print(report.format_markdown())
    print("\n" + "=" * 80 + "\n")

    hpa_path = Path("k8s/hpa.yaml")
    simulator = HpaV2ControlLoopSimulator(
        manifest_path=hpa_path if hpa_path.exists() else None
    )
    sim_steps = simulator.simulate_k6_workload()
    print(simulator.format_simulation_table(sim_steps))
    print("\n" + "=" * 80 + "\n")

    lab_plan = MinikubeLabPlanBuilder.build_plan()
    print(MinikubeLabPlanBuilder.format_plan_table(lab_plan))

    return 0 if report.is_valid else 1


if __name__ == "__main__":
    sys.exit(main())
