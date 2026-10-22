"""Unit and integration tests for the scientific benchmarking harness and k6 script.

Tests verify:
- k6 load test script syntax, stages, thresholds, and endpoints.
- BenchmarkTargetProtocol implementations (Mock, InProcess, HTTP).
- BenchmarkRunner latency percentile calculations (p50 <= p95 <= p99) and RPS.
- Numerical parity verification (top-1 agreement, probability deltas).
- Scientific Markdown report generation.
- End-to-end in-process benchmarking against RecycleNet FastAPI application.
"""

from __future__ import annotations

import base64
import io
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

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
    main,
)
from src.api.main import app


def test_k6_load_test_script_structure() -> None:
    """Verify benchmarks/k6_load_test.js exists and satisfies all requirement SLOs."""
    k6_path = Path("benchmarks/k6_load_test.js")
    assert k6_path.is_file(), "benchmarks/k6_load_test.js must exist"

    content = k6_path.read_text(encoding="utf-8")

    # Target URL configurable with default :8000
    assert "TARGET_URL" in content
    assert "http://localhost:8000" in content

    # Stages defined (warm-up, ramp-up, peak load, cooldown)
    assert "stages:" in content
    assert "Warm-up" in content or "target: 10" in content
    assert "Ramp-up" in content or "target: 30" in content
    assert "Peak load" in content or "target: 60" in content
    assert "Cooldown" in content or "target: 0" in content

    # Strict SLO thresholds
    assert "thresholds:" in content
    assert "http_req_failed" in content
    assert "rate<0.01" in content
    assert "p(95)<300" in content
    assert "p(99)<600" in content

    # Endpoints tested
    assert "/healthz" in content
    assert "/info" in content
    assert "/predict" in content

    # Telemetry and response validation
    assert "latitude" in content
    assert "longitude" in content
    assert "predicted_class" in content

    # Valid base64 JPEG embedded
    b64_match = re.search(r"SAMPLE_BASE64_JPEG\s*=\s*'([^']+)'", content)
    assert b64_match is not None, "SAMPLE_BASE64_JPEG not found in k6 script"
    raw_bytes = base64.b64decode(b64_match.group(1))
    img = Image.open(io.BytesIO(raw_bytes))
    assert img.format in ("JPEG", "JPG")
    assert img.size[0] > 0 and img.size[1] > 0


def test_synthetic_image_generation() -> None:
    """Verify synthetic image generator yields decodable RGB JPEG bytes."""
    sample = generate_synthetic_image(color=(100, 150, 200), size=(224, 224))
    assert isinstance(sample, bytes)
    assert len(sample) > 0

    img = Image.open(io.BytesIO(sample))
    assert img.size == (224, 224)
    assert img.mode == "RGB"

    samples = generate_synthetic_samples(count=5)
    assert len(samples) == 5
    for s in samples:
        decoded = Image.open(io.BytesIO(s))
        assert decoded.size == (224, 224)


def test_mock_benchmark_target_protocol() -> None:
    """Verify MockBenchmarkTarget satisfies BenchmarkTargetProtocol."""
    target = MockBenchmarkTarget(name="TestMock", latency_ms=0.0, rss_mb=55.5)
    assert isinstance(target, BenchmarkTargetProtocol)
    assert target.name == "TestMock"
    assert target.measure_rss_mb() == 55.5

    img_bytes = generate_synthetic_image()
    output = target.predict(img_bytes)
    assert "predicted_class" in output
    assert "confidence" in output
    assert "class_probabilities" in output
    assert output["predicted_class"] == "cardboard"
    assert output["confidence"] == 0.85


def test_benchmark_runner_latency_percentiles_and_rps() -> None:
    """Verify BenchmarkRunner accurately calculates p50, p95, p99, and RPS."""
    target = MockBenchmarkTarget(name="FastMock", latency_ms=1.0, rss_mb=40.0)
    runner = BenchmarkRunner(default_warmup=2, default_requests=10)

    metrics = runner.run_latency_benchmark(target)

    assert isinstance(metrics, LatencyMetrics)
    assert metrics.total_requests == 10
    assert metrics.error_rate == 0.0
    assert metrics.rss_mb == 40.0
    assert metrics.rps > 0.0

    # Strict percentile ordering invariant
    assert metrics.min_ms <= metrics.p50_ms
    assert metrics.p50_ms <= metrics.p95_ms
    assert metrics.p95_ms <= metrics.p99_ms
    assert metrics.p99_ms <= metrics.max_ms


def test_benchmark_runner_failure_handling() -> None:
    """Verify BenchmarkRunner gracefully tracks target error rate."""
    failing_target = MockBenchmarkTarget(name="FailMock", should_fail=True)
    runner = BenchmarkRunner(default_warmup=1, default_requests=5)

    metrics = runner.run_latency_benchmark(failing_target)
    assert metrics.total_requests == 5
    assert metrics.error_rate == 1.0
    assert metrics.p50_ms == 0.0
    assert metrics.rps == 0.0


def test_compare_parity_identical_targets() -> None:
    """Verify compare_parity passes when predictions and probabilities match."""
    probs = {
        "cardboard": 0.90,
        "glass": 0.04,
        "metal": 0.03,
        "paper": 0.01,
        "plastic": 0.01,
        "trash": 0.01,
    }
    target_a = MockBenchmarkTarget(name="TierA", probabilities=probs)
    target_b = MockBenchmarkTarget(name="TierB", probabilities=probs)

    runner = BenchmarkRunner(parity_tolerance=1e-3)
    parity = runner.compare_parity(target_a, target_b, num_samples=5)

    assert isinstance(parity, ParityMetrics)
    assert parity.passed is True
    assert parity.top1_agreement_rate == 1.0
    assert parity.max_prob_diff == 0.0
    assert parity.num_samples == 5


def test_compare_parity_diverging_targets() -> None:
    """Verify compare_parity fails when top-1 classes or probabilities diverge."""
    probs_a = {"cardboard": 0.85, "glass": 0.15}
    probs_b = {"cardboard": 0.15, "glass": 0.85}

    target_a = MockBenchmarkTarget(name="TierA", probabilities=probs_a)
    target_b = MockBenchmarkTarget(name="TierB", probabilities=probs_b)

    runner = BenchmarkRunner(parity_tolerance=1e-3)
    parity = runner.compare_parity(target_a, target_b, num_samples=5)

    assert parity.passed is False
    assert parity.top1_agreement_rate == 0.0
    assert parity.max_prob_diff == 0.70


def test_format_markdown_report() -> None:
    """Verify Markdown report generation formats tables and parity sections."""
    runner = BenchmarkRunner()
    latency_mock = LatencyMetrics(
        p50_ms=4.12,
        p95_ms=6.84,
        p99_ms=8.50,
        mean_ms=4.55,
        min_ms=3.80,
        max_ms=9.10,
        rps=220.5,
        total_requests=100,
        error_rate=0.0,
        rss_mb=42.0,
    )
    parity_mock = ParityMetrics(
        max_prob_diff=0.00012,
        top1_agreement_rate=1.0,
        num_samples=10,
        passed=True,
    )

    report = runner.format_markdown_report(
        results=[("FastAPI", latency_mock), ("Axum", latency_mock)],
        parity=parity_mock,
    )

    assert "# RecycleNet-K8s" in report
    assert "| Serving Target |" in report
    assert "FastAPI" in report
    assert "Axum" in report
    assert "4.12" in report
    assert "220.50" in report
    assert "Top-1 Class Agreement" in report
    assert "PASS (Zero Skew)" in report


def test_in_process_fastapi_target_predict_and_rss() -> None:
    """Verify InProcessFastApiTarget executes inference against the FastAPI app."""
    sample_img = generate_synthetic_image(color=(80, 160, 240))

    with InProcessFastApiTarget(app) as target:
        assert isinstance(target, BenchmarkTargetProtocol)
        assert "FastAPI" in target.name

        resp = target.predict(sample_img)
        assert "predicted_class" in resp
        assert "confidence" in resp
        assert "class_probabilities" in resp
        assert isinstance(resp["predicted_class"], str)
        assert 0.0 <= resp["confidence"] <= 1.0

        rss = target.measure_rss_mb()
        assert rss >= 0.0


def test_benchmark_runner_with_in_process_target() -> None:
    """Verify BenchmarkRunner executes end-to-end benchmark on in-process FastAPI."""
    runner = BenchmarkRunner(default_warmup=1, default_requests=3)

    with InProcessFastApiTarget(app) as target:
        metrics = runner.run_latency_benchmark(target, num_requests=3, warmup=1)

        assert metrics.total_requests == 3
        assert metrics.error_rate == 0.0
        assert metrics.p50_ms > 0.0
        assert metrics.rps > 0.0


def test_cli_main_in_process_flag(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify CLI main entrypoint executes in-process mode cleanly."""
    with patch(
        "sys.argv",
        ["benchmark_suite.py", "--in-process", "--requests", "2", "--warmup", "1"],
    ):
        main()

    captured = capsys.readouterr()
    assert "Running in-process benchmark" in captured.out
    assert "FastAPI (Python, In-Process)" in captured.out
    assert "Latency p50 (ms)" in captured.out


def test_cli_main_fallback_when_servers_offline(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify CLI main falls back to in-process when external ports are offline."""
    with patch(
        "sys.argv",
        [
            "benchmark_suite.py",
            "--python-url",
            "http://127.0.0.1:59998",
            "--rust-url",
            "http://127.0.0.1:59999",
            "--requests",
            "2",
            "--warmup",
            "1",
        ],
    ):
        main()

    captured = capsys.readouterr()
    assert "Falling back to in-process FastAPI benchmark" in captured.out
    assert "Latency p50 (ms)" in captured.out


def test_http_benchmark_target_mocked() -> None:
    """Verify HttpBenchmarkTarget handles requests, responses, and closures."""
    target = HttpBenchmarkTarget(base_url="http://mock-service:8080")
    assert target.name == "HTTP Target (http://mock-service:8080)"

    mock_client = MagicMock()
    mock_health = MagicMock()
    mock_health.status_code = 200
    mock_client.get.return_value = mock_health

    mock_pred = MagicMock()
    mock_pred.json.return_value = {
        "predicted_class": "plastic",
        "confidence": 0.92,
        "class_probabilities": {"plastic": 0.92},
    }
    mock_client.post.return_value = mock_pred

    with patch.object(target, "_client", mock_client):
        assert target.is_healthy() is True
        res = target.predict(b"fake_image_bytes")
        assert res["predicted_class"] == "plastic"
        target.close()
        mock_client.close.assert_called_once()
