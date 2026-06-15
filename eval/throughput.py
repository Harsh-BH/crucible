"""M2 deliverable: Sentinel throughput + latency benchmark.

Fires a batch of artifact verifications through the **hardened Sentinel sandbox**
(via :class:`verifier.SentinelVerifier` / :class:`verifier.SentinelClient`) at a
range of concurrency levels and measures, per level:

* wall time for the whole batch,
* throughput (verifications / second),
* latency percentiles p50 / p90 / p99 (and min / max / mean).

This is the "throughput measured" half of M2 (the parity half is
:mod:`eval.parity`). It motivates routing the *training-loop* reward through
Sentinel's queue + worker pool (KEDA-autoscaled) rather than blocking the trainer
on local subprocess builds — see ``DESIGN.md`` ("verifier-backend strategy").

Injectable transport
--------------------
``benchmark_sentinel`` accepts an ``httpx`` ``transport``. With an
``httpx.MockTransport`` it runs **entirely locally** (CI / unit tests:
``tests/test_throughput_mock.py``) and still exercises the full async fan-out and
the percentile math. With ``transport=None`` and a real ``base_url`` it hits a
live Sentinel (which requires ``make up`` in the Sentinel repo; default port
8080). Either way the measured wall time / throughput are real for that backend.

Concurrency is bounded with an :class:`asyncio.Semaphore`; the harness submits
all artifacts and gathers them, so throughput reflects the sandbox's queue +
worker parallelism at each level. stdlib + ``httpx`` + ``verifier`` only.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from typing import Any

import httpx

from verifier.sentinel_client import SentinelClient, SentinelVerifier
from verifier.types import ArtifactKind, ResourceLimits, VerifySpec

__all__ = ["benchmark_sentinel", "percentiles", "make_mock_transport"]


def percentiles(samples: list[float], ps: tuple[float, ...] = (50, 90, 99)) -> dict[str, float]:
    """Return ``{"pNN": value}`` for the requested percentiles of ``samples``.

    Uses the "nearest-rank" method (no interpolation) on the sorted samples,
    which is robust for small/large batches alike. Returns zeros for an empty
    input.
    """
    if not samples:
        return {f"p{int(p)}": 0.0 for p in ps}
    ordered = sorted(samples)
    n = len(ordered)
    out: dict[str, float] = {}
    for p in ps:
        # nearest-rank: rank = ceil(p/100 * n), 1-indexed, clamped to [1, n].
        rank = max(1, min(n, -(-int(p) * n // 100)))
        out[f"p{int(p)}"] = ordered[rank - 1]
    return out


def make_mock_transport(*, latency_s: float = 0.0, exit_code: int = 0) -> httpx.MockTransport:
    """A self-contained Sentinel mock: ``POST`` 202 then ``GET`` SUCCESS.

    Returns a harness-style SUCCESS job so :class:`SentinelVerifier` maps it to a
    real :class:`~verifier.types.VerifyResult`. ``latency_s`` injects a small per-
    request sleep so the benchmark's percentile/throughput math has non-trivial
    numbers in tests; the handler is async so the sleep does not block the loop.

    Useful for tests and for ``--mock`` on the CLI (no live Sentinel needed).
    """
    job_id = "bench-0001"
    # A minimal harness JSON line: build_ok/smoke_ok True so SUCCESS maps cleanly.
    harness_json = json.dumps(
        {"build_ok": True, "smoke_ok": True, "signals": {}, "reasons": []}
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        if latency_s:
            await asyncio.sleep(latency_s)
        if request.method == "POST" and request.url.path.endswith("/submissions"):
            return httpx.Response(202, json={"job_id": job_id, "status": "QUEUED"})
        if request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "status": "SUCCESS",
                    "exit_code": exit_code,
                    "stdout": harness_json + "\n",
                    "stderr": "",
                    "time_used_ms": int(latency_s * 1000) or 5,
                    "memory_used_kb": 12000,
                },
            )
        return httpx.Response(404, json={"error": "not found"})

    return httpx.MockTransport(handler)


async def _time_one(
    verifier: SentinelVerifier,
    artifact: str,
    spec: VerifySpec,
    sem: asyncio.Semaphore,
) -> tuple[float, bool]:
    """Verify one artifact under the semaphore; return (latency_s, build_ok)."""
    async with sem:
        t0 = time.perf_counter()
        result = await verifier.verify(artifact, spec)
        return time.perf_counter() - t0, bool(result.build_ok)


async def _run_level(
    artifacts: list[str],
    spec: VerifySpec,
    *,
    base_url: str,
    concurrency: int,
    transport: httpx.AsyncBaseTransport | None,
) -> dict[str, Any]:
    """Run the whole batch at one concurrency level; return its metrics."""
    # One shared client/verifier per level so connection reuse is realistic.
    client = SentinelClient(base_url=base_url, transport=transport, timeout=60.0)
    verifier = SentinelVerifier(client=client, poll_interval=0.001, deadline_s=60.0)
    sem = asyncio.Semaphore(max(1, concurrency))
    try:
        t0 = time.perf_counter()
        results = await asyncio.gather(
            *(_time_one(verifier, art, spec, sem) for art in artifacts)
        )
        wall = time.perf_counter() - t0
    finally:
        await client.aclose()

    latencies = [lat for lat, _ in results]
    n_ok = sum(1 for _, ok in results)
    count = len(artifacts)
    pcts = percentiles(latencies)
    return {
        "concurrency": concurrency,
        "count": count,
        "wall_s": wall,
        "throughput_per_s": (count / wall) if wall > 0 else 0.0,
        "build_ok_count": n_ok,
        "latency_s": {
            "min": min(latencies) if latencies else 0.0,
            "mean": (sum(latencies) / len(latencies)) if latencies else 0.0,
            "max": max(latencies) if latencies else 0.0,
            **pcts,
        },
    }


async def benchmark_sentinel(
    artifacts: list[str],
    spec: VerifySpec,
    *,
    base_url: str = "http://localhost:8080",
    concurrency_levels: tuple[int, ...] = (1, 4, 16, 64),
    transport: httpx.AsyncBaseTransport | None = None,
) -> dict[str, Any]:
    """Benchmark Sentinel verification throughput at several concurrency levels.

    Parameters
    ----------
    artifacts:
        The artifacts (Dockerfiles, or raw Python for ``ArtifactKind.PYTHON``) to
        verify — the same batch is replayed at every concurrency level so levels
        are comparable.
    spec:
        The :class:`VerifySpec` to verify each artifact against. For raw-code
        throughput use ``spec.kind = ArtifactKind.PYTHON``.
    base_url:
        Sentinel base URL (default ``http://localhost:8080``). Ignored for
        routing when a ``MockTransport`` is supplied, but still set on the client.
    concurrency_levels:
        Semaphore bounds to sweep.
    transport:
        Optional injected ``httpx`` transport. Pass an ``httpx.MockTransport``
        (e.g. :func:`make_mock_transport`) to run locally; pass ``None`` to hit a
        live Sentinel at ``base_url``.

    Returns
    -------
    dict
        ``{"base_url", "n_artifacts", "spec_kind", "levels": [<per-level>...],
        "best_throughput_per_s"}`` where each per-level dict has ``concurrency``,
        ``wall_s``, ``throughput_per_s``, and a ``latency_s`` block with
        ``min/mean/max/p50/p90/p99``.
    """
    levels: list[dict[str, Any]] = []
    for c in concurrency_levels:
        levels.append(
            await _run_level(
                artifacts,
                spec,
                base_url=base_url,
                concurrency=int(c),
                transport=transport,
            )
        )
    best = max((lv["throughput_per_s"] for lv in levels), default=0.0)
    return {
        "base_url": base_url,
        "n_artifacts": len(artifacts),
        "spec_kind": spec.kind.value if hasattr(spec.kind, "value") else str(spec.kind),
        "concurrency_levels": list(concurrency_levels),
        "levels": levels,
        "best_throughput_per_s": best,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m eval.throughput",
        description=(
            "Sentinel throughput/latency benchmark (M2). Running against a LIVE "
            "Sentinel requires `make up` in the Sentinel repo (default port "
            "8080). Use --mock to run with no server."
        ),
    )
    p.add_argument("--base-url", default="http://localhost:8080", help="Sentinel base URL")
    p.add_argument("--n", type=int, default=64, help="Number of artifacts to fire")
    p.add_argument(
        "--concurrency", type=int, nargs="+", default=[1, 4, 16, 64],
        help="Concurrency levels to sweep",
    )
    p.add_argument(
        "--mock", action="store_true",
        help="Use an in-process MockTransport instead of a live Sentinel",
    )
    p.add_argument(
        "--mock-latency", type=float, default=0.002,
        help="Per-request latency injected by the mock (seconds)",
    )
    p.add_argument("--python", action="store_true", help="Send raw-Python artifacts (kind=python)")
    p.add_argument("--out", default=None, help="Write the JSON report here")
    return p


def _demo_artifacts(n: int, *, python: bool) -> tuple[list[str], VerifySpec]:
    if python:
        spec = VerifySpec(
            spec_id="bench-py",
            kind=ArtifactKind.PYTHON,
            smoke={},
            limits=ResourceLimits(wall_s=5, mem_mb=128),
        )
        artifacts = [f"print({i})\n" for i in range(n)]
    else:
        spec = VerifySpec(
            spec_id="bench-df",
            kind=ArtifactKind.DOCKERFILE,
            smoke={"port": 8000, "must_contain": ["FROM", "CMD"], "health_path": "/health"},
            limits=ResourceLimits(wall_s=5, mem_mb=128),
        )
        df = (
            "FROM python:3.11-slim\nWORKDIR /app\nCOPY . .\n"
            'EXPOSE 8000\nCMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]\n'
        )
        artifacts = [df for _ in range(n)]
    return artifacts, spec


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    artifacts, spec = _demo_artifacts(args.n, python=args.python)
    transport = make_mock_transport(latency_s=args.mock_latency) if args.mock else None
    report = asyncio.run(
        benchmark_sentinel(
            artifacts,
            spec,
            base_url=args.base_url,
            concurrency_levels=tuple(args.concurrency),
            transport=transport,
        )
    )
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, sort_keys=True)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
