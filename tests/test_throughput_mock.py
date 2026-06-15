"""Tests for eval.throughput: benchmark_sentinel against a MockTransport.

Confirms the async fan-out, the throughput/latency dict shape, and the
percentile math at >=2 concurrency levels — all locally, no live Sentinel and no
matplotlib/torch.
"""
from __future__ import annotations

import pytest

from eval.throughput import benchmark_sentinel, make_mock_transport, percentiles
from verifier.types import ArtifactKind, ResourceLimits, VerifySpec


def _spec(kind=ArtifactKind.DOCKERFILE) -> VerifySpec:
    return VerifySpec(
        spec_id="bench-t",
        kind=kind,
        smoke={"port": 8000, "must_contain": ["FROM", "CMD"], "health_path": "/health"},
        limits=ResourceLimits(wall_s=5, mem_mb=128),
    )


GOOD_DOCKERFILE = (
    "FROM python:3.11-slim\nWORKDIR /app\nCOPY . .\n"
    'EXPOSE 8000\nCMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]\n'
)


# --- percentiles helper ----------------------------------------------------
def test_percentiles_basic() -> None:
    samples = [float(i) for i in range(1, 101)]  # 1..100
    p = percentiles(samples, (50, 90, 99))
    # nearest-rank: p50 -> rank 50 -> value 50; p90 -> 90; p99 -> 99.
    assert p["p50"] == 50.0
    assert p["p90"] == 90.0
    assert p["p99"] == 99.0


def test_percentiles_empty() -> None:
    p = percentiles([])
    assert p == {"p50": 0.0, "p90": 0.0, "p99": 0.0}


def test_percentiles_single() -> None:
    p = percentiles([0.5])
    assert p["p50"] == 0.5 and p["p90"] == 0.5 and p["p99"] == 0.5


# --- benchmark_sentinel with MockTransport ---------------------------------
async def test_benchmark_returns_sane_dict_multi_level() -> None:
    artifacts = [GOOD_DOCKERFILE for _ in range(12)]
    transport = make_mock_transport(latency_s=0.002)
    report = await benchmark_sentinel(
        artifacts,
        _spec(),
        concurrency_levels=(1, 4, 8),
        transport=transport,
    )
    assert report["n_artifacts"] == 12
    assert report["spec_kind"] == "dockerfile"
    levels = report["levels"]
    assert len(levels) == 3  # >= 2 concurrency levels
    for lv in levels:
        assert lv["count"] == 12
        assert lv["wall_s"] > 0.0
        assert lv["throughput_per_s"] > 0.0
        lat = lv["latency_s"]
        for key in ("min", "mean", "max", "p50", "p90", "p99"):
            assert key in lat
            assert lat[key] >= 0.0
        # latency ordering sanity
        assert lat["min"] <= lat["p50"] <= lat["max"] + 1e-9
        assert lat["p50"] <= lat["p90"] + 1e-9 <= lat["p99"] + 1e-9
        # mock returns build_ok True for all
        assert lv["build_ok_count"] == 12
    assert report["best_throughput_per_s"] > 0.0


async def test_higher_concurrency_not_slower_wall() -> None:
    # With injected per-request latency, more concurrency should not increase
    # wall time (and typically reduces it). Assert the weak, robust inequality.
    artifacts = [GOOD_DOCKERFILE for _ in range(16)]
    transport = make_mock_transport(latency_s=0.005)
    report = await benchmark_sentinel(
        artifacts, _spec(), concurrency_levels=(1, 8), transport=transport
    )
    wall_c1 = report["levels"][0]["wall_s"]
    wall_c8 = report["levels"][1]["wall_s"]
    assert wall_c8 <= wall_c1 + 1e-6


async def test_python_kind_artifacts() -> None:
    artifacts = ["print(1)\n" for _ in range(6)]
    transport = make_mock_transport(latency_s=0.001)
    report = await benchmark_sentinel(
        artifacts,
        _spec(kind=ArtifactKind.PYTHON),
        concurrency_levels=(2, 4),
        transport=transport,
    )
    assert report["spec_kind"] == "python"
    assert len(report["levels"]) == 2
    # raw-python path: exit_code 0 -> build_ok True
    assert all(lv["build_ok_count"] == 6 for lv in report["levels"])


async def test_throughput_consistent_with_count_and_wall() -> None:
    artifacts = [GOOD_DOCKERFILE for _ in range(10)]
    transport = make_mock_transport(latency_s=0.001)
    report = await benchmark_sentinel(
        artifacts, _spec(), concurrency_levels=(4,), transport=transport
    )
    lv = report["levels"][0]
    assert lv["throughput_per_s"] == pytest.approx(lv["count"] / lv["wall_s"], rel=1e-6)
