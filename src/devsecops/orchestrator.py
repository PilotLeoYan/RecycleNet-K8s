"""RecycleNet DevSecOps orchestration, workflow verification, and deployment engine.

Provides SOLID-compliant validation for CI/CD GitHub Actions workflows, zero-trust
Google Cloud Workload Identity Federation OIDC token exchanges, and Google Cloud Run
serverless autoscaling simulation.
"""

from __future__ import annotations

import argparse
import math
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable
from urllib.parse import urlparse

import yaml

# ==============================================================================
# Protocols (Interface Segregation & Dependency Inversion)
# ==============================================================================


@runtime_checkable
class WorkflowLoaderProtocol(Protocol):
    """Protocol defining the interface for loading CI/CD workflows and configs."""

    def load_yaml(self, relative_path: str) -> dict[str, Any]:
        """Load and parse YAML file contents with 'on' key normalization.

        Args:
            relative_path: Relative path to the YAML file from repository root.

        Returns:
            Parsed dictionary representation of the YAML document.
        """
        ...

    def load_text(self, relative_path: str) -> str:
        """Load raw text contents of a file relative to repository root.

        Args:
            relative_path: Relative path to the text file.

        Returns:
            Decoded string content of the file.
        """
        ...

    def file_exists(self, relative_path: str) -> bool:
        """Check whether a file exists at the given relative path.

        Args:
            relative_path: Relative path to test.

        Returns:
            True if file exists and is a regular file, False otherwise.
        """
        ...


@runtime_checkable
class DevSecOpsRuleProtocol(Protocol):
    """Protocol defining a single DevSecOps policy or workflow validation rule."""

    def validate(self, loader: WorkflowLoaderProtocol) -> list[str]:
        """Execute validation against repository files accessible via the loader.

        Args:
            loader: WorkflowLoaderProtocol instance for accessing workspace files.

        Returns:
            List of violation descriptions. Empty list signifies complete compliance.
        """
        ...


# ==============================================================================
# Domain Models & Value Objects
# ==============================================================================


@dataclass(frozen=True)
class DevSecOpsAuditReport:
    """Aggregated audit findings for repository DevSecOps and cloud configurations."""

    is_valid: bool
    violations: list[str] = field(default_factory=list)
    checked_components: int = 0

    def format_markdown(self) -> str:
        """Format the audit report as a structured Markdown document.

        Returns:
            Markdown-formatted summary report.
        """
        status_badge = "PASSED" if self.is_valid else "FAILED"
        lines = [
            "# DevSecOps & Cloud Deployment Security Audit",
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
                "> **Integrity Confirmed**: All CI/CD workflows, GCP Workload Identity "
                "OIDC federations, zero-trust secrets, and toolchain configurations "
                "adhere to enterprise security policies."
            )
        return "\n".join(lines)


@dataclass(frozen=True)
class OidcFederationClaimCheck:
    """Validation verdict of GitHub Actions OIDC JWT claims for GCP STS."""

    is_valid: bool
    issuer: str
    subject: str
    audience: str
    service_account: str
    violations: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class OidcExchangeStep:
    """Structured phase in the zero-trust Workload Identity Federation exchange."""

    step_number: int
    phase: str
    source_actor: str
    target_service: str
    artifact: str
    ttl_seconds: int


@dataclass(frozen=True)
class CloudRunScalingStep:
    """Simulated serverless autoscaling state under concurrent request volume."""

    stage_name: str
    concurrent_requests: int
    container_concurrency: int
    desired_instances: int
    estimated_p95_ms: float


# ==============================================================================
# Concrete Loader (Single Responsibility Principle)
# ==============================================================================


class RepositoryWorkflowLoader:
    """Concrete repository file loader implementing WorkflowLoaderProtocol."""

    def __init__(self, root_dir: Path | None = None) -> None:
        """Initialize loader with a base repository root directory.

        Args:
            root_dir: Base directory path, defaults to current working directory.
        """
        self.root_dir = root_dir if root_dir is not None else Path(".")

    def _resolve_safe_path(self, relative_path: str) -> Path:
        """Resolve a relative path and enforce repository root containment (CWE-022)."""
        base_resolved = self.root_dir.resolve()
        target_resolved = (self.root_dir / relative_path).resolve()
        if not target_resolved.is_relative_to(base_resolved):
            raise ValueError(
                f"Path traversal detected for relative path '{relative_path}'."
            )
        return self.root_dir / relative_path

    def file_exists(self, relative_path: str) -> bool:
        """Check whether a file exists at the given relative path."""
        target = self._resolve_safe_path(relative_path)
        return target.is_file()

    def load_text(self, relative_path: str) -> str:
        """Load raw text contents of a file relative to repository root."""
        target = self._resolve_safe_path(relative_path)
        if not target.is_file():
            raise FileNotFoundError(
                f"Workflow file '{relative_path}' not found in repository root"
            )
        return target.read_text(encoding="utf-8")

    def load_yaml(self, relative_path: str) -> dict[str, Any]:
        """Load and parse YAML file contents with 'on' key normalization.

        Note: PyYAML 1.1 sometimes parses unquoted `on:` as boolean True,
        so True keys are recursively normalized to "on".
        """
        content = self.load_text(relative_path)
        parsed = yaml.safe_load(content)
        if parsed is None:
            return {}
        if not isinstance(parsed, dict):
            return {"content": parsed}
        normalized = self._normalize_yaml_keys(parsed)
        if isinstance(normalized, dict):
            return normalized
        return {"content": normalized}

    @staticmethod
    def _normalize_yaml_keys(data: Any) -> Any:
        """Recursively normalize boolean True keys to 'on'."""
        if isinstance(data, dict):
            normalized: dict[str | Any, Any] = {}
            for key, val in data.items():
                norm_key = "on" if key is True else key
                normalized[norm_key] = RepositoryWorkflowLoader._normalize_yaml_keys(
                    val
                )
            return normalized
        if isinstance(data, list):
            return [
                RepositoryWorkflowLoader._normalize_yaml_keys(item) for item in data
            ]
        return data


# ==============================================================================
# SOLID Validation Rules (Open/Closed & Single Responsibility)
# ==============================================================================


class CiWorkflowValidator:
    """Validates CI GitHub Actions workflow quality, security, and coverage."""

    def validate(self, loader: WorkflowLoaderProtocol) -> list[str]:
        """Validate .github/workflows/ci.yaml (or .github/workflows/ci.yml)."""
        ci_path = ".github/workflows/ci.yaml"
        if not loader.file_exists(ci_path):
            ci_path = ".github/workflows/ci.yml"
            if not loader.file_exists(ci_path):
                return [
                    "CI workflow not found at .github/workflows/ci.yaml or "
                    ".github/workflows/ci.yml"
                ]

        violations: list[str] = []
        data = loader.load_yaml(ci_path)

        # 1. Triggers
        triggers = data.get("on", {})
        if isinstance(triggers, dict):
            # pull_request on main
            pr = triggers.get("pull_request")
            if pr is None:
                violations.append("CI workflow must declare a 'pull_request' trigger.")
            elif isinstance(pr, dict):
                branches = pr.get("branches", [])
                if isinstance(branches, str):
                    branches = [branches]
                if "main" not in branches:
                    violations.append(
                        "CI workflow 'pull_request' trigger must target branch 'main'."
                    )

            # push on main
            push = triggers.get("push")
            if push is None:
                violations.append("CI workflow must declare a 'push' trigger.")
            elif isinstance(push, dict):
                branches = push.get("branches", [])
                if isinstance(branches, str):
                    branches = [branches]
                if "main" not in branches:
                    violations.append(
                        "CI workflow 'push' trigger must target branch 'main'."
                    )
        else:
            violations.append(
                "CI workflow triggers ('on') must be configured as a dictionary."
            )

        # 2. Permissions (least-privilege)
        top_perms = data.get("permissions")
        has_contents_read = False
        if isinstance(top_perms, dict) and top_perms.get("contents") == "read":
            has_contents_read = True

        jobs = data.get("jobs", {})
        if isinstance(jobs, dict):
            for job in jobs.values():
                if isinstance(job, dict):
                    job_perms = job.get("permissions")
                    if (
                        isinstance(job_perms, dict)
                        and job_perms.get("contents") == "read"
                    ):
                        has_contents_read = True

        if not has_contents_read:
            violations.append(
                "CI workflow must enforce least-privilege permissions "
                "with 'contents: read'."
            )

        # Extract all steps from all jobs
        all_steps: list[dict[str, Any]] = []
        if isinstance(jobs, dict):
            for job in jobs.values():
                if isinstance(job, dict):
                    steps = job.get("steps", [])
                    if isinstance(steps, list):
                        for step in steps:
                            if isinstance(step, dict):
                                all_steps.append(step)

        # 3. astral-sh/setup-uv step with enable-cache: true
        uv_steps = [s for s in all_steps if "astral-sh/setup-uv" in s.get("uses", "")]
        if not uv_steps:
            violations.append("CI workflow must include 'astral-sh/setup-uv' step.")
        else:
            has_cache = any(
                s.get("with", {}).get("enable-cache") is True for s in uv_steps
            )
            if not has_cache:
                violations.append(
                    "CI workflow 'astral-sh/setup-uv' step must configure "
                    "'enable-cache: true'."
                )

        # 4. Mandatory Toolchain Commands
        run_commands = [str(s.get("run", "")) for s in all_steps if "run" in s]

        if not any("ruff check" in cmd for cmd in run_commands):
            violations.append("CI workflow must execute 'ruff check'.")

        if not any("ruff format --check" in cmd for cmd in run_commands):
            violations.append("CI workflow must execute 'ruff format --check'.")

        if not any("mypy" in cmd for cmd in run_commands):
            violations.append("CI workflow must execute 'mypy'.")

        has_bandit = any(
            "bandit" in cmd
            and "-c pyproject.toml" in cmd
            and ("-r src" in cmd or "-r src/" in cmd)
            for cmd in run_commands
        )
        if not has_bandit:
            violations.append(
                "CI workflow must execute 'bandit -c pyproject.toml -r src'."
            )

        has_pytest_cov = any(
            "pytest" in cmd
            and ("--cov=src" in cmd or "--cov src" in cmd)
            and "--cov-fail-under=80" in cmd
            for cmd in run_commands
        )
        if not has_pytest_cov:
            violations.append(
                "CI workflow must execute 'pytest' with '--cov=src' and "
                "'--cov-fail-under=80'."
            )

        return violations


class CdWorkflowGcpOidcValidator:
    """Validates CD workflow zero-trust GCP OIDC Workload Identity Federation."""

    def validate(self, loader: WorkflowLoaderProtocol) -> list[str]:
        """Validate .github/workflows/cd.yaml (or .github/workflows/cd.yml)."""
        cd_path = ".github/workflows/cd.yaml"
        if not loader.file_exists(cd_path):
            cd_path = ".github/workflows/cd.yml"
            if not loader.file_exists(cd_path):
                return [
                    "CD workflow not found at .github/workflows/cd.yaml or "
                    ".github/workflows/cd.yml"
                ]

        violations: list[str] = []
        raw_text = loader.load_text(cd_path)

        # 1. Zero-Trust Security: NO static credentials in workflow
        if "credentials_json" in raw_text or "GCP_SA_KEY" in raw_text:
            violations.append(
                "Zero-Trust violation: static credentials detected ('credentials_json' "
                "or 'GCP_SA_KEY'). Must use GCP Workload Identity Federation."
            )

        data = loader.load_yaml(cd_path)

        # 2. Triggers: push to main and workflow_dispatch
        triggers = data.get("on", {})
        if isinstance(triggers, dict):
            push = triggers.get("push")
            if push is None:
                violations.append("CD workflow must declare a 'push' trigger.")
            elif isinstance(push, dict):
                branches = push.get("branches", [])
                if isinstance(branches, str):
                    branches = [branches]
                if "main" not in branches:
                    violations.append(
                        "CD workflow 'push' trigger must target branch 'main'."
                    )

            if "workflow_dispatch" not in triggers:
                violations.append(
                    "CD workflow must declare a 'workflow_dispatch' trigger."
                )
        else:
            violations.append(
                "CD workflow triggers ('on') must be configured as a dictionary."
            )

        # 3. Permissions: id-token: write and contents: read
        top_perms = data.get("permissions", {})
        has_id_token_write = False
        has_contents_read = False
        if isinstance(top_perms, dict):
            if top_perms.get("id-token") == "write":
                has_id_token_write = True
            if top_perms.get("contents") == "read":
                has_contents_read = True

        jobs = data.get("jobs", {})
        if isinstance(jobs, dict):
            for job in jobs.values():
                if isinstance(job, dict):
                    j_perms = job.get("permissions", {})
                    if isinstance(j_perms, dict):
                        if j_perms.get("id-token") == "write":
                            has_id_token_write = True
                        if j_perms.get("contents") == "read":
                            has_contents_read = True

        if not has_id_token_write:
            violations.append(
                "CD workflow must declare 'id-token: write' permission for OIDC "
                "token federation."
            )
        if not has_contents_read:
            violations.append("CD workflow must declare 'contents: read' permission.")

        # Extract steps
        all_steps: list[dict[str, Any]] = []
        if isinstance(jobs, dict):
            for job in jobs.values():
                if isinstance(job, dict):
                    steps = job.get("steps", [])
                    if isinstance(steps, list):
                        for step in steps:
                            if isinstance(step, dict):
                                all_steps.append(step)

        # 4. google-github-actions/auth step with WIF
        auth_steps = [
            s for s in all_steps if "google-github-actions/auth" in s.get("uses", "")
        ]
        if not auth_steps:
            violations.append(
                "CD workflow must include 'google-github-actions/auth' step."
            )
        else:
            auth_step = auth_steps[0]
            step_with = auth_step.get("with", {})
            if not step_with.get("workload_identity_provider"):
                violations.append(
                    "Google auth step must specify 'workload_identity_provider'."
                )
            if not step_with.get("service_account"):
                violations.append("Google auth step must specify 'service_account'.")

        # 5. Artifact Registry (-docker.pkg.dev) & Docker build-push
        if "-docker.pkg.dev" not in raw_text:
            violations.append(
                "CD workflow must configure Google Artifact Registry "
                "('-docker.pkg.dev')."
            )

        sha_marker = "sha-${{ github.sha }}"
        if sha_marker not in raw_text and "${{ github.sha }}" not in raw_text:
            violations.append(
                "CD workflow must tag Docker images with 'sha-${{ github.sha }}'."
            )

        has_build_push = any(
            "docker/build-push-action" in s.get("uses", "") for s in all_steps
        )
        if not has_build_push:
            violations.append(
                "CD workflow must use 'docker/build-push-action' for container builds."
            )

        # 6. Continuous Deployment to Cloud Run
        deploy_steps = [
            s
            for s in all_steps
            if "google-github-actions/deploy-cloudrun" in s.get("uses", "")
        ]
        if not deploy_steps:
            violations.append(
                "CD workflow must include 'google-github-actions/deploy-cloudrun' step."
            )
        else:
            deploy_step = deploy_steps[0]
            step_with = deploy_step.get("with", {})
            flags = str(step_with.get("flags", ""))
            combined_deploy_text = f"{flags} {step_with}"

            if (
                "--port=8000" not in combined_deploy_text
                and step_with.get("port") != 8000
            ):
                violations.append("Cloud Run deployment must configure port 8000.")

            if (
                "--memory=1Gi" not in combined_deploy_text
                and step_with.get("memory") != "1Gi"
            ):
                violations.append("Cloud Run deployment must configure memory to 1Gi.")

            if "--cpu=1" not in combined_deploy_text and step_with.get("cpu") not in (
                1,
                "1",
            ):
                violations.append("Cloud Run deployment must configure CPU to 1.")

            if "--min-instances" not in combined_deploy_text:
                violations.append(
                    "Cloud Run deployment must configure '--min-instances' "
                    "scaling flag."
                )

            if "--max-instances" not in combined_deploy_text:
                violations.append(
                    "Cloud Run deployment must configure '--max-instances' "
                    "scaling flag."
                )

        return violations


class DevSecOpsToolchainValidator:
    """Validates repository toolchain configuration in pyproject.toml and workflows."""

    def validate(self, loader: WorkflowLoaderProtocol) -> list[str]:
        """Validate pyproject.toml, CodeQL, dependency review, and Dependabot."""
        violations: list[str] = []

        # 1. pyproject.toml validation
        if not loader.file_exists("pyproject.toml"):
            violations.append("Configuration file 'pyproject.toml' not found.")
        else:
            raw_toml = loader.load_text("pyproject.toml")
            try:
                toml_data = tomllib.loads(raw_toml)

                # pytest-cov in [dependency-groups].dev
                dev_deps = toml_data.get("dependency-groups", {}).get("dev", [])
                if not any("pytest-cov" in str(dep) for dep in dev_deps):
                    violations.append(
                        "pyproject.toml must declare 'pytest-cov' under "
                        "[dependency-groups].dev."
                    )

                # coverage report fail_under >= 80
                cov_report = (
                    toml_data.get("tool", {}).get("coverage", {}).get("report", {})
                )
                fail_under = cov_report.get("fail_under", 0)
                if fail_under < 80:
                    violations.append(
                        f"pyproject.toml [tool.coverage.report].fail_under must "
                        f"be >= 80 (found {fail_under})."
                    )

                # mypy disallow_untyped_defs == True
                mypy_cfg = toml_data.get("tool", {}).get("mypy", {})
                if mypy_cfg.get("disallow_untyped_defs") is not True:
                    violations.append(
                        "pyproject.toml [tool.mypy].disallow_untyped_defs must be true."
                    )

                # bandit configuration section
                if "bandit" not in toml_data.get("tool", {}):
                    violations.append(
                        "pyproject.toml must contain a [tool.bandit] configuration "
                        "section."
                    )
            except Exception as err:
                violations.append(f"Failed to parse pyproject.toml: {err}")

        # 2. Mandatory security workflows and dependabot
        required_security_files = [
            (".github/workflows/codeql.yml", "CodeQL security scan workflow"),
            (
                ".github/workflows/dependency-review.yml",
                "Dependency review PR workflow",
            ),
            (".github/dependabot.yml", "Dependabot automated dependency updater"),
        ]

        for file_path, desc in required_security_files:
            if not loader.file_exists(file_path):
                violations.append(
                    f"Missing required security configuration: {desc} ({file_path})."
                )

        return violations


# ==============================================================================
# Composite Auditor & Simulators
# ==============================================================================


class DevSecOpsAuditor:
    """Composite auditor coordinating all DevSecOps validation rules."""

    def __init__(
        self,
        loader: WorkflowLoaderProtocol | None = None,
        rules: list[DevSecOpsRuleProtocol] | None = None,
    ) -> None:
        """Initialize auditor with loader and rules."""
        self.loader = loader if loader is not None else RepositoryWorkflowLoader()
        default_rules: list[DevSecOpsRuleProtocol] = [
            CiWorkflowValidator(),
            CdWorkflowGcpOidcValidator(),
            DevSecOpsToolchainValidator(),
        ]
        self.rules: list[DevSecOpsRuleProtocol] = (
            rules if rules is not None else default_rules
        )

    def run_audit(self) -> DevSecOpsAuditReport:
        """Run all configured validation rules and aggregate violations."""
        all_violations: list[str] = []
        for rule in self.rules:
            all_violations.extend(rule.validate(self.loader))

        return DevSecOpsAuditReport(
            is_valid=len(all_violations) == 0,
            violations=all_violations,
            checked_components=len(self.rules),
        )


class GcpOidcWorkloadIdentitySimulator:
    """Simulates and verifies GCP Workload Identity Federation OIDC token exchange."""

    @staticmethod
    def validate_claims(
        claims: dict[str, Any] | None = None,
    ) -> OidcFederationClaimCheck:
        """Verify the GitHub Actions OIDC JWT claims against zero-trust policy."""
        if claims is None:
            claims = {
                "iss": "https://token.actions.githubusercontent.com",
                "sub": "repo:PilotLeoYan/RecycleNet-K8s:ref:refs/heads/main",
                "aud": (
                    "https://iam.googleapis.com/projects/1234567890/locations/global/"
                    "workloadIdentityPools/github-pool/providers/github-provider"
                ),
                "repository": "PilotLeoYan/RecycleNet-K8s",
                "ref": "refs/heads/main",
                "service_account": (
                    "recyclenet-cd-sa@PROJECT_ID.iam.gserviceaccount.com"
                ),
            }

        violations: list[str] = []

        issuer = str(claims.get("iss", ""))
        subject = str(claims.get("sub", ""))
        audience = str(claims.get("aud", ""))
        repository = str(claims.get("repository", ""))
        ref = str(claims.get("ref", ""))
        sa = str(claims.get("service_account", ""))

        if issuer != "https://token.actions.githubusercontent.com":
            violations.append(
                f"Invalid OIDC issuer: '{issuer}'. Expected "
                "'https://token.actions.githubusercontent.com'."
            )

        if subject and (
            not subject.startswith("repo:")
            or not subject.endswith("/RecycleNet-K8s:ref:refs/heads/main")
        ):
            violations.append(
                f"Untrusted OIDC subject claim: '{subject}'. "
                "Expected 'repo:<owner>/RecycleNet-K8s:ref:refs/heads/main'."
            )

        parsed_audience = urlparse(audience)
        if (
            parsed_audience.scheme != "https"
            or parsed_audience.hostname != "iam.googleapis.com"
            or not parsed_audience.path.startswith("/projects/")
        ):
            violations.append(
                f"Invalid OIDC audience: '{audience}'. Must target Google Cloud STS "
                "('https://iam.googleapis.com/projects/...')."
            )

        repo_owner, repo_sep, repo_name = repository.partition("/")
        if not repo_owner or repo_sep != "/" or repo_name != "RecycleNet-K8s":
            violations.append(
                f"Untrusted GitHub repository in claim: '{repository}'. "
                "Expected '<owner>/RecycleNet-K8s'."
            )

        if ref != "refs/heads/main":
            violations.append(
                f"Untrusted git ref: '{ref}'. CD deployments are restricted "
                "to 'refs/heads/main'."
            )

        sa_local, sa_sep, sa_domain = sa.partition("@")
        sa_domain_parts = sa_domain.split(".")
        is_sa_valid = (
            bool(sa_local)
            and sa_sep == "@"
            and "@" not in sa_domain
            and "/" not in sa
            and ":" not in sa
            and len(sa_domain_parts) == 4
            and bool(sa_domain_parts[0])
            and sa_domain_parts[1:] == ["iam", "gserviceaccount", "com"]
        )
        if not is_sa_valid:
            violations.append(
                f"Invalid GCP service account: '{sa}'. Must be a valid GCP IAM "
                "service account."
            )

        return OidcFederationClaimCheck(
            is_valid=len(violations) == 0,
            issuer=issuer,
            subject=subject,
            audience=audience,
            service_account=sa,
            violations=violations,
        )

    @staticmethod
    def simulate_exchange() -> list[OidcExchangeStep]:
        """Generate the 4-step OIDC zero-trust token exchange sequence."""
        return [
            OidcExchangeStep(
                step_number=1,
                phase="OIDC Token Issuance",
                source_actor="GitHub Actions Runner",
                target_service="GitHub Token Service",
                artifact="Signed JWT (OIDC Token, RS256)",
                ttl_seconds=300,
            ),
            OidcExchangeStep(
                step_number=2,
                phase="Security Token Exchange",
                source_actor="GitHub Actions Runner",
                target_service="GCP Security Token Service (STS)",
                artifact="Federated STS Access Token",
                ttl_seconds=3600,
            ),
            OidcExchangeStep(
                step_number=3,
                phase="IAM Impersonation",
                source_actor="GCP STS",
                target_service="GCP IAM Service Account Credentials API",
                artifact="Short-lived GCP Access Token",
                ttl_seconds=3600,
            ),
            OidcExchangeStep(
                step_number=4,
                phase="Workload Deployment Authorization",
                source_actor="GCP Cloud SDK / Runner",
                target_service="Google Artifact Registry & Cloud Run",
                artifact="OAuth2 Bearer Deployment Session",
                ttl_seconds=3600,
            ),
        ]

    @staticmethod
    def format_simulation_table(steps: list[OidcExchangeStep]) -> str:
        """Render the 4-step OIDC exchange sequence as a Markdown table."""
        lines = [
            "### GCP Workload Identity Federation OIDC 4-Step Exchange",
            "",
            "| Step | Phase | Source Actor | Target Service | "
            "Generated Artifact | TTL |",
            "| :---: | :--- | :--- | :--- | :--- | :---: |",
        ]
        for step in steps:
            lines.append(
                f"| {step.step_number} | **{step.phase}** | `{step.source_actor}` | "
                f"`{step.target_service}` | `{step.artifact}` | {step.ttl_seconds}s |"
            )
        return "\n".join(lines)


class CloudRunDeploymentSimulator:
    """Simulates serverless horizontal autoscaling for Google Cloud Run."""

    def __init__(
        self,
        min_instances: int = 1,
        max_instances: int = 5,
        container_concurrency: int = 80,
    ) -> None:
        """Initialize Cloud Run autoscaling parameters."""
        self.min_instances = min_instances
        self.max_instances = max_instances
        self.container_concurrency = container_concurrency

    def compute_desired_instances(self, concurrent_requests: int) -> int:
        """Compute desired serverless container instances based on traffic volume."""
        if concurrent_requests <= 0:
            return self.min_instances
        raw = math.ceil(concurrent_requests / self.container_concurrency)
        return max(self.min_instances, min(self.max_instances, raw))

    def simulate_traffic_stages(self) -> list[CloudRunScalingStep]:
        """Simulate autoscaling trajectory across 4 standard traffic stages."""
        stages = [
            ("Idle / Warm Baseline", 10, 45.0),
            ("Moderate Traffic", 120, 72.0),
            ("Peak Burst", 360, 128.0),
            ("Cooldown", 0, 42.0),
        ]
        steps: list[CloudRunScalingStep] = []
        for stage_name, reqs, p95 in stages:
            desired = self.compute_desired_instances(reqs)
            steps.append(
                CloudRunScalingStep(
                    stage_name=stage_name,
                    concurrent_requests=reqs,
                    container_concurrency=self.container_concurrency,
                    desired_instances=desired,
                    estimated_p95_ms=p95,
                )
            )
        return steps

    @staticmethod
    def format_scaling_table(steps: list[CloudRunScalingStep]) -> str:
        """Render the Cloud Run scaling trajectory as a Markdown table."""
        lines = [
            "### Google Cloud Run Serverless Autoscaling Simulation",
            "",
            "| Traffic Stage | Concurrency Cap | In-Flight Requests | "
            "Desired Instances | Estimated P95 Latency |",
            "| :--- | :---: | :---: | :---: | :---: |",
        ]
        for step in steps:
            lines.append(
                f"| **{step.stage_name}** | {step.container_concurrency} "
                f"req/container | {step.concurrent_requests} | "
                f"{step.desired_instances} | {step.estimated_p95_ms:.1f} ms |"
            )
        return "\n".join(lines)


# ==============================================================================
# CLI Entrypoint
# ==============================================================================


def main(argv: list[str] | None = None) -> int:
    """Execute DevSecOps workflow audit and cloud deployment simulation."""
    parser = argparse.ArgumentParser(
        description=(
            "RecycleNet DevSecOps workflow audit and GCP cloud deployment simulator."
        )
    )
    parser.add_argument(
        "--ci-only",
        action="store_true",
        help="Execute only CI quality gate rule validation",
    )
    parser.add_argument(
        "--cd-only",
        action="store_true",
        help="Execute only CD GCP OIDC and deployment rule validation",
    )
    parser.add_argument(
        "--format-md",
        action="store_true",
        help="Export execution summary as GitHub Flavored Markdown",
    )
    args = parser.parse_args(argv if argv is not None else [])

    loader = RepositoryWorkflowLoader(root_dir=Path("."))
    rules: list[DevSecOpsRuleProtocol] | None = None
    if args.ci_only and not args.cd_only:
        rules = [CiWorkflowValidator(), DevSecOpsToolchainValidator()]
    elif args.cd_only and not args.ci_only:
        rules = [CdWorkflowGcpOidcValidator()]

    auditor = DevSecOpsAuditor(loader=loader, rules=rules)
    report = auditor.run_audit()

    oidc_sim = GcpOidcWorkloadIdentitySimulator()
    claim_check = oidc_sim.validate_claims()
    exchange_steps = oidc_sim.simulate_exchange()

    cloudrun_sim = CloudRunDeploymentSimulator()
    scaling_steps = cloudrun_sim.simulate_traffic_stages()

    separator = "\n\n---\n\n" if args.format_md else "\n\n" + "=" * 80 + "\n\n"
    sections = [
        report.format_markdown(),
        oidc_sim.format_simulation_table(exchange_steps),
        cloudrun_sim.format_scaling_table(scaling_steps),
    ]
    print(separator.join(sections))

    return 0 if report.is_valid and claim_check.is_valid else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
