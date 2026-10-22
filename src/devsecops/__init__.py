"""RecycleNet DevSecOps orchestration, workflow verification, and deployment package."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
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
        main,
    )

__all__ = [
    "CdWorkflowGcpOidcValidator",
    "CiWorkflowValidator",
    "CloudRunDeploymentSimulator",
    "CloudRunScalingStep",
    "DevSecOpsAuditor",
    "DevSecOpsAuditReport",
    "DevSecOpsRuleProtocol",
    "DevSecOpsToolchainValidator",
    "GcpOidcWorkloadIdentitySimulator",
    "OidcExchangeStep",
    "OidcFederationClaimCheck",
    "RepositoryWorkflowLoader",
    "WorkflowLoaderProtocol",
    "main",
]


def __getattr__(name: str) -> Any:
    """Lazily resolve package exports to prevent runpy module import warnings."""
    if name in __all__:
        from src.devsecops import orchestrator

        return getattr(orchestrator, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
