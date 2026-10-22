"""RecycleNet-K8s Scientific Benchmarking Suite and Harness."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from benchmarks.benchmark_suite import (
        BenchmarkRunner,
        BenchmarkTargetProtocol,
        HttpBenchmarkTarget,
        InProcessFastApiTarget,
        LatencyMetrics,
        MockBenchmarkTarget,
        ParityMetrics,
        generate_synthetic_image,
        generate_synthetic_samples,
    )

__all__ = [
    "BenchmarkRunner",
    "BenchmarkTargetProtocol",
    "HttpBenchmarkTarget",
    "InProcessFastApiTarget",
    "LatencyMetrics",
    "MockBenchmarkTarget",
    "ParityMetrics",
    "generate_synthetic_image",
    "generate_synthetic_samples",
]


def __getattr__(name: str) -> object:
    """Lazily load benchmark symbols to prevent premature circular runpy imports."""
    if name in __all__:
        import benchmarks.benchmark_suite as suite

        return getattr(suite, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
