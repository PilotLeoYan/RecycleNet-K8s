"""RecycleNet Kubernetes orchestration and autoscaling simulation package."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.k8s.orchestrator import (
        AuditReport,
        CrossResourceIntegrityValidator,
        DeploymentSecurityAndQoSValidator,
        DockerfileZeroTrustValidator,
        HpaSimulationStep,
        HpaV2ControlLoopSimulator,
        KubernetesOrchestrationAuditor,
        LabStep,
        ManifestLoaderProtocol,
        MinikubeLabPlanBuilder,
        ResourceQuantityParser,
        ValidationRuleProtocol,
        YamlManifestLoader,
        main,
    )

__all__ = [
    "AuditReport",
    "CrossResourceIntegrityValidator",
    "DeploymentSecurityAndQoSValidator",
    "DockerfileZeroTrustValidator",
    "HpaSimulationStep",
    "HpaV2ControlLoopSimulator",
    "KubernetesOrchestrationAuditor",
    "LabStep",
    "ManifestLoaderProtocol",
    "MinikubeLabPlanBuilder",
    "ResourceQuantityParser",
    "ValidationRuleProtocol",
    "YamlManifestLoader",
    "main",
]


def __getattr__(name: str) -> Any:
    """Lazily resolve package exports to prevent runpy module import warnings."""
    if name in __all__:
        from src.k8s import orchestrator

        return getattr(orchestrator, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
