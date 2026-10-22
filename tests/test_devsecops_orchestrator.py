"""Unit and integration tests for DevSecOps orchestrator, rules, and simulators."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import src.devsecops
from src.devsecops.orchestrator import (
    CdWorkflowGcpOidcValidator,
    CiWorkflowValidator,
    CloudRunDeploymentSimulator,
    CloudRunScalingStep,
    DevSecOpsAuditor,
    DevSecOpsAuditReport,
    DevSecOpsRuleProtocol,
    DevSecOpsToolchainValidator,
    GcpOidcWorkloadIdentitySimulator,
    OidcExchangeStep,
    OidcFederationClaimCheck,
    RepositoryWorkflowLoader,
    WorkflowLoaderProtocol,
)
from src.devsecops.orchestrator import (
    main as devsecops_main,
)


class MockWorkflowLoader:
    """In-memory mock loader for isolated unit testing of validation rules."""

    def __init__(
        self,
        files: dict[str, str] | None = None,
        yaml_files: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.files = files or {}
        self.yaml_files = yaml_files or {}

    def file_exists(self, relative_path: str) -> bool:
        return relative_path in self.files or relative_path in self.yaml_files

    def load_text(self, relative_path: str) -> str:
        if relative_path in self.files:
            return self.files[relative_path]
        raise FileNotFoundError(f"Mock file '{relative_path}' not found")

    def load_yaml(self, relative_path: str) -> dict[str, Any]:
        if relative_path in self.yaml_files:
            return self.yaml_files[relative_path]
        raise FileNotFoundError(f"Mock YAML '{relative_path}' not found")


# ==============================================================================
# Protocol & Loader Tests
# ==============================================================================


def test_repository_workflow_loader_real_repo() -> None:
    """Verify repository loader accurately reads files and parses YAML."""
    loader = RepositoryWorkflowLoader(root_dir=Path("."))
    assert isinstance(loader, WorkflowLoaderProtocol)
    assert loader.file_exists("pyproject.toml")
    assert not loader.file_exists("non_existent_file_xyz.abc")

    text = loader.load_text("pyproject.toml")
    assert "[project]" in text

    ci_yaml = loader.load_yaml(".github/workflows/ci.yaml")
    assert "name" in ci_yaml
    assert "on" in ci_yaml

    with pytest.raises(FileNotFoundError):
        loader.load_text("path/that/does/not/exist.txt")


def test_repository_workflow_loader_yaml_normalization(tmp_path: Path) -> None:
    """Verify boolean True keys in YAML are normalized to 'on'."""
    yaml_content = "on:\n  push:\n    branches: [main]\n"
    test_file = tmp_path / "test.yaml"
    test_file.write_text(yaml_content, encoding="utf-8")

    loader = RepositoryWorkflowLoader(root_dir=tmp_path)
    data = loader.load_yaml("test.yaml")
    assert "on" in data
    assert True not in data

    empty_file = tmp_path / "empty.yaml"
    empty_file.write_text("", encoding="utf-8")
    assert loader.load_yaml("empty.yaml") == {}

    scalar_file = tmp_path / "scalar.yaml"
    scalar_file.write_text("hello world", encoding="utf-8")
    assert loader.load_yaml("scalar.yaml") == {"content": "hello world"}


# ==============================================================================
# Domain Model Tests
# ==============================================================================


def test_devsecops_audit_report_formatting() -> None:
    """Verify markdown output of DevSecOpsAuditReport for pass and fail states."""
    passed_report = DevSecOpsAuditReport(
        is_valid=True,
        violations=[],
        checked_components=3,
    )
    md_passed = passed_report.format_markdown()
    assert "PASSED" in md_passed
    assert "Integrity Confirmed" in md_passed

    failed_report = DevSecOpsAuditReport(
        is_valid=False,
        violations=["Missing CI security scanner", "Static secret found"],
        checked_components=3,
    )
    md_failed = failed_report.format_markdown()
    assert "FAILED" in md_failed
    assert "1. [VIOLATION] Missing CI security scanner" in md_failed
    assert "2. [VIOLATION] Static secret found" in md_failed


def test_oidc_dataclasses() -> None:
    """Verify dataclasses initialization and field types."""
    claim = OidcFederationClaimCheck(
        is_valid=True,
        issuer="https://token.actions.githubusercontent.com",
        subject="repo:org/repo:ref:refs/heads/main",
        audience="https://iam.googleapis.com",
        service_account="sa@project.iam.gserviceaccount.com",
    )
    assert claim.is_valid is True
    assert claim.violations == []

    step = OidcExchangeStep(
        step_number=1,
        phase="OIDC Issuance",
        source_actor="Runner",
        target_service="GitHub",
        artifact="Token",
        ttl_seconds=300,
    )
    assert step.step_number == 1
    assert step.ttl_seconds == 300

    scaling_step = CloudRunScalingStep(
        stage_name="Burst",
        concurrent_requests=100,
        container_concurrency=80,
        desired_instances=2,
        estimated_p95_ms=65.5,
    )
    assert scaling_step.desired_instances == 2


# ==============================================================================
# CI Workflow Validator Tests
# ==============================================================================


def test_ci_workflow_validator_on_real_repo() -> None:
    """Verify repository CI workflow passes validation with zero violations."""
    loader = RepositoryWorkflowLoader(root_dir=Path("."))
    validator = CiWorkflowValidator()
    assert isinstance(validator, DevSecOpsRuleProtocol)
    violations = validator.validate(loader)
    assert violations == [], f"Unexpected CI violations: {violations}"


def test_ci_workflow_validator_missing_file() -> None:
    """Verify missing CI workflow flags a violation."""
    loader = MockWorkflowLoader()
    validator = CiWorkflowValidator()
    violations = validator.validate(loader)
    assert any("CI workflow not found" in v for v in violations)


def test_ci_workflow_validator_trigger_and_permission_violations() -> None:
    """Verify triggers and permissions are strictly enforced."""
    invalid_yaml: dict[str, Any] = {
        "on": {
            "pull_request": {"branches": ["feature"]},
            "push": {"branches": ["dev"]},
        },
        "permissions": {"contents": "write"},
        "jobs": {
            "test": {
                "steps": [
                    {
                        "uses": "astral-sh/setup-uv@v7",
                        "with": {"enable-cache": False},
                    },
                    {"run": "echo hello"},
                ]
            }
        },
    }
    loader = MockWorkflowLoader(yaml_files={".github/workflows/ci.yaml": invalid_yaml})
    validator = CiWorkflowValidator()
    violations = validator.validate(loader)

    assert any(
        "pull_request' trigger must target branch 'main'" in v for v in violations
    )
    assert any("push' trigger must target branch 'main'" in v for v in violations)
    assert any("contents: read" in v for v in violations)
    assert any("enable-cache: true" in v for v in violations)
    assert any("ruff check" in v for v in violations)
    assert any("ruff format --check" in v for v in violations)
    assert any("mypy" in v for v in violations)
    assert any("bandit -c pyproject.toml -r src" in v for v in violations)
    assert any("pytest' with '--cov=src'" in v for v in violations)


def test_ci_workflow_validator_invalid_on_type() -> None:
    """Verify non-dict trigger is flagged."""
    invalid_yaml: dict[str, Any] = {
        "on": "push",
        "permissions": {"contents": "read"},
        "jobs": {},
    }
    loader = MockWorkflowLoader(yaml_files={".github/workflows/ci.yaml": invalid_yaml})
    violations = CiWorkflowValidator().validate(loader)
    assert any(
        "triggers ('on') must be configured as a dictionary" in v for v in violations
    )


# ==============================================================================
# CD Workflow Validator Tests
# ==============================================================================


def test_cd_workflow_validator_on_real_repo() -> None:
    """Verify repository CD workflow passes validation with zero violations."""
    loader = RepositoryWorkflowLoader(root_dir=Path("."))
    validator = CdWorkflowGcpOidcValidator()
    assert isinstance(validator, DevSecOpsRuleProtocol)
    violations = validator.validate(loader)
    assert violations == [], f"Unexpected CD violations: {violations}"


def test_cd_workflow_validator_missing_file() -> None:
    """Verify missing CD workflow flags a violation."""
    loader = MockWorkflowLoader()
    violations = CdWorkflowGcpOidcValidator().validate(loader)
    assert any("CD workflow not found" in v for v in violations)


def test_cd_workflow_validator_static_credential_detection() -> None:
    """Verify static credentials violate zero-trust policy."""
    bad_text = "env:\n  credentials_json: ${{ secrets.GCP_KEY }}\n"
    loader = MockWorkflowLoader(
        files={".github/workflows/cd.yaml": bad_text},
        yaml_files={".github/workflows/cd.yaml": {"on": {}}},
    )
    violations = CdWorkflowGcpOidcValidator().validate(loader)
    assert any(
        "Zero-Trust violation: static credentials detected" in v for v in violations
    )


def test_cd_workflow_validator_missing_steps_and_permissions() -> None:
    """Verify missing OIDC permissions and deploy steps are flagged."""
    invalid_yaml: dict[str, Any] = {
        "on": {"push": {"branches": ["main"]}},
        "permissions": {"contents": "read"},
        "jobs": {
            "deploy": {
                "steps": [
                    {
                        "uses": "google-github-actions/auth@v2",
                        "with": {},
                    }
                ]
            }
        },
    }
    raw_text = "dummy workflow"
    loader = MockWorkflowLoader(
        files={".github/workflows/cd.yaml": raw_text},
        yaml_files={".github/workflows/cd.yaml": invalid_yaml},
    )
    violations = CdWorkflowGcpOidcValidator().validate(loader)

    assert any("workflow_dispatch" in v for v in violations)
    assert any("id-token: write" in v for v in violations)
    assert any("workload_identity_provider" in v for v in violations)
    assert any("service_account" in v for v in violations)
    assert any("-docker.pkg.dev" in v for v in violations)
    assert any("sha-${{ github.sha }}" in v for v in violations)
    assert any("docker/build-push-action" in v for v in violations)
    assert any("google-github-actions/deploy-cloudrun" in v for v in violations)


def test_cd_workflow_validator_cloudrun_flag_violations() -> None:
    """Verify missing Cloud Run flags are flagged as violations."""
    yaml_data: dict[str, Any] = {
        "on": {
            "push": {"branches": ["main"]},
            "workflow_dispatch": None,
        },
        "permissions": {
            "contents": "read",
            "id-token": "write",
        },
        "jobs": {
            "deploy": {
                "steps": [
                    {
                        "uses": "google-github-actions/auth@v2",
                        "with": {
                            "workload_identity_provider": "prov",
                            "service_account": "sa@proj.iam.gserviceaccount.com",
                        },
                    },
                    {
                        "uses": "docker/build-push-action@v6",
                        "with": {},
                    },
                    {
                        "uses": "google-github-actions/deploy-cloudrun@v2",
                        "with": {
                            "service": "api",
                            "flags": "--port=9000",
                        },
                    },
                ]
            }
        },
    }
    raw_text = "-docker.pkg.dev sha-${{ github.sha }}"
    loader = MockWorkflowLoader(
        files={".github/workflows/cd.yaml": raw_text},
        yaml_files={".github/workflows/cd.yaml": yaml_data},
    )
    violations = CdWorkflowGcpOidcValidator().validate(loader)

    assert any("port 8000" in v for v in violations)
    assert any("memory to 1Gi" in v for v in violations)
    assert any("CPU to 1" in v for v in violations)
    assert any("--min-instances" in v for v in violations)
    assert any("--max-instances" in v for v in violations)


# ==============================================================================
# Toolchain Validator Tests
# ==============================================================================


def test_devsecops_toolchain_validator_on_real_repo() -> None:
    """Verify repository toolchain passes validation with zero violations."""
    loader = RepositoryWorkflowLoader(root_dir=Path("."))
    validator = DevSecOpsToolchainValidator()
    assert isinstance(validator, DevSecOpsRuleProtocol)
    violations = validator.validate(loader)
    assert violations == [], f"Unexpected toolchain violations: {violations}"


def test_devsecops_toolchain_validator_violations() -> None:
    """Verify toolchain configuration deficiencies are flagged."""
    invalid_toml = """
[dependency-groups]
dev = ["ruff"]

[tool.coverage.report]
fail_under = 60

[tool.mypy]
disallow_untyped_defs = false
"""
    loader = MockWorkflowLoader(
        files={"pyproject.toml": invalid_toml},
    )
    violations = DevSecOpsToolchainValidator().validate(loader)

    assert any("pytest-cov" in v for v in violations)
    assert any("fail_under must be >= 80" in v for v in violations)
    assert any("disallow_untyped_defs must be true" in v for v in violations)
    assert any("[tool.bandit]" in v for v in violations)
    assert any("CodeQL" in v for v in violations)
    assert any("Dependency review" in v for v in violations)
    assert any("Dependabot" in v for v in violations)


def test_devsecops_toolchain_missing_toml() -> None:
    """Verify missing pyproject.toml is flagged."""
    loader = MockWorkflowLoader()
    violations = DevSecOpsToolchainValidator().validate(loader)
    assert any("pyproject.toml' not found" in v for v in violations)


# ==============================================================================
# Auditor & Simulators Tests
# ==============================================================================


def test_devsecops_auditor_execution() -> None:
    """Verify composite auditor aggregates report from all configured rules."""
    loader = RepositoryWorkflowLoader(root_dir=Path("."))
    auditor = DevSecOpsAuditor(loader=loader)
    report = auditor.run_audit()
    assert report.is_valid is True
    assert report.violations == []
    assert report.checked_components == 3


def test_gcp_oidc_workload_identity_simulator() -> None:
    """Verify OIDC claim validation and exchange sequence generation."""
    sim = GcpOidcWorkloadIdentitySimulator()

    # Valid default claims
    valid_check = sim.validate_claims()
    assert valid_check.is_valid is True
    assert valid_check.violations == []

    # Invalid claims
    bad_claims: dict[str, Any] = {
        "iss": "https://evil.token.issuer.com",
        "aud": "https://attacker.site",
        "repository": "untrusted/repo",
        "ref": "refs/heads/feature-branch",
        "service_account": "invalid-sa@gmail.com",
    }
    invalid_check = sim.validate_claims(bad_claims)
    assert invalid_check.is_valid is False
    assert len(invalid_check.violations) == 5

    # Spoofed URL / domain substring attacks (CodeQL CWE-020 regression test)
    spoofed_claims: dict[str, Any] = {
        "iss": "https://token.actions.githubusercontent.com",
        "aud": "https://iam.googleapis.com.attacker.com/projects/123",
        "repository": "PilotLeoYan/RecycleNet-K8s",
        "ref": "refs/heads/main",
        "service_account": "deployer@proj.iam.gserviceaccount.com.attacker.com",
    }
    spoofed_check = sim.validate_claims(spoofed_claims)
    assert spoofed_check.is_valid is False
    assert len(spoofed_check.violations) == 2
    assert any("Invalid OIDC audience" in v for v in spoofed_check.violations)
    assert any("Invalid GCP service account" in v for v in spoofed_check.violations)

    # Exchange steps
    steps = sim.simulate_exchange()
    assert len(steps) == 4
    assert steps[0].phase == "OIDC Token Issuance"
    assert steps[1].phase == "Security Token Exchange"
    assert steps[2].phase == "IAM Impersonation"
    assert steps[3].phase == "Workload Deployment Authorization"

    table = sim.format_simulation_table(steps)
    assert "### GCP Workload Identity Federation OIDC 4-Step Exchange" in table
    assert "Signed JWT (OIDC Token, RS256)" in table


def test_cloud_run_deployment_simulator() -> None:
    """Verify Cloud Run autoscaling logic and traffic stage trajectory."""
    sim = CloudRunDeploymentSimulator(
        min_instances=1,
        max_instances=5,
        container_concurrency=80,
    )

    assert sim.compute_desired_instances(0) == 1
    assert sim.compute_desired_instances(-10) == 1
    assert sim.compute_desired_instances(10) == 1
    assert sim.compute_desired_instances(80) == 1
    assert sim.compute_desired_instances(81) == 2
    assert sim.compute_desired_instances(360) == 5
    assert sim.compute_desired_instances(1000) == 5

    stages = sim.simulate_traffic_stages()
    assert len(stages) == 4
    assert stages[0].stage_name == "Idle / Warm Baseline"
    assert stages[0].desired_instances == 1
    assert stages[1].stage_name == "Moderate Traffic"
    assert stages[1].desired_instances == 2
    assert stages[2].stage_name == "Peak Burst"
    assert stages[2].desired_instances == 5
    assert stages[3].stage_name == "Cooldown"
    assert stages[3].desired_instances == 1

    table = sim.format_scaling_table(stages)
    assert "### Google Cloud Run Serverless Autoscaling Simulation" in table
    assert "Idle / Warm Baseline" in table


# ==============================================================================
# Package __getattr__ & CLI Main Tests
# ==============================================================================


def test_package_exports() -> None:
    """Verify lazy-loading exports and error handling in src.devsecops."""
    assert src.devsecops.DevSecOpsAuditor is DevSecOpsAuditor
    assert src.devsecops.CiWorkflowValidator is CiWorkflowValidator

    with pytest.raises(AttributeError, match="has no attribute 'UnknownSymbol'"):
        _ = getattr(src.devsecops, "UnknownSymbol")


def test_orchestrator_main(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify CLI main entrypoint executes cleanly and prints reports."""
    exit_code = devsecops_main()
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "DevSecOps & Cloud Deployment Security Audit" in captured.out
    assert "GCP Workload Identity Federation OIDC 4-Step Exchange" in captured.out
    assert "Google Cloud Run Serverless Autoscaling Simulation" in captured.out
