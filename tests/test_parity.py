"""Tests for eval.parity: local-py vs sentinel agreement (MockTransport).

We simulate Sentinel with an httpx.MockTransport whose handler actually EXECUTES
the submitted harness source in a subprocess (exactly as the real sandbox would
run it) and returns its real stdout as a SUCCESS job. This makes the parity check
meaningful — both backends run the identical harness produced by
verifier.smoke.checks.build_python_harness — rather than asserting against a
hard-coded verdict. No live Sentinel needed; no matplotlib/torch.
"""
from __future__ import annotations

import json
import subprocess
import sys

import httpx
import pytest

from eval.parity import parity_report
from verifier.types import ArtifactKind, ResourceLimits, VerifySpec

# Good Dockerfile -> build_ok & smoke_ok True on both backends.
GOOD_DOCKERFILE = """\
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir fastapi uvicorn
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
"""

# Trivial Dockerfile that builds (FROM/WORKDIR/COPY/EXPOSE/CMD present) but does
# NOT launch a recognised server -> build_ok True, smoke_ok False on both.
TRIVIAL_DOCKERFILE = """\
FROM python:3.11-slim
WORKDIR /app
COPY . .
EXPOSE 8000
CMD ["true"]
"""

# Garbage -> build_ok False, smoke_ok False on both.
GARBAGE = "this is not a dockerfile at all\n"


def _spec(**smoke) -> VerifySpec:
    smoke.setdefault("port", 8000)
    smoke.setdefault("must_contain", ["FROM", "CMD"])
    smoke.setdefault("health_path", "/health")
    return VerifySpec(
        spec_id="parity-t",
        kind=ArtifactKind.DOCKERFILE,
        smoke=smoke,
        limits=ResourceLimits(wall_s=10, mem_mb=256),
    )


def _executing_mock_transport() -> httpx.MockTransport:
    """MockTransport that RUNS the submitted Python harness and returns its stdout.

    This mirrors what the real Sentinel sandbox does (execute the submission,
    capture stdout/exit code), so the SentinelVerifier maps a genuine verdict.
    """
    # Map job_id -> captured run output, so POST runs the code and GET returns it.
    jobs: dict[str, dict] = {}
    counter = {"i": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path.endswith("/submissions"):
            body = json.loads(request.content)
            src = body["source_code"]
            proc = subprocess.run(
                [sys.executable, "-I", "-c", src],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            counter["i"] += 1
            jid = f"job-{counter['i']:04d}"
            jobs[jid] = {
                "status": "SUCCESS",
                "exit_code": proc.returncode,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "time_used_ms": 7,
                "memory_used_kb": 10000,
            }
            return httpx.Response(202, json={"job_id": jid, "status": "QUEUED"})
        if request.method == "GET":
            jid = request.url.path.rsplit("/", 1)[-1]
            return httpx.Response(200, json=jobs.get(jid, {"status": "INTERNAL_ERROR"}))
        return httpx.Response(404, json={"error": "not found"})

    return httpx.MockTransport(handler)


async def test_parity_good_and_trivial_all_agree() -> None:
    spec = _spec()
    artifacts = [GOOD_DOCKERFILE, TRIVIAL_DOCKERFILE, GARBAGE, GOOD_DOCKERFILE]
    report = await parity_report(
        artifacts, spec, sentinel_transport=_executing_mock_transport()
    )
    assert report["n"] == 4
    assert report["agreement_rate"] == 1.0
    assert report["build_agreement_rate"] == 1.0
    assert report["smoke_agreement_rate"] == 1.0
    assert report["disagreements"] == []


async def test_parity_records_capture_both_verdicts() -> None:
    spec = _spec()
    report = await parity_report(
        [GOOD_DOCKERFILE], spec, sentinel_transport=_executing_mock_transport()
    )
    rec = report["records"][0]
    # Good Dockerfile: both build & smoke True under both backends.
    assert rec["local_py"]["build_ok"] is True
    assert rec["local_py"]["smoke_ok"] is True
    assert rec["sentinel"]["build_ok"] is True
    assert rec["sentinel"]["smoke_ok"] is True
    assert rec["sentinel"]["status"] == "SUCCESS"


async def test_parity_trivial_builds_but_no_smoke_on_both() -> None:
    spec = _spec()
    report = await parity_report(
        [TRIVIAL_DOCKERFILE], spec, sentinel_transport=_executing_mock_transport()
    )
    rec = report["records"][0]
    assert rec["local_py"]["build_ok"] is True
    assert rec["local_py"]["smoke_ok"] is False
    assert rec["sentinel"]["build_ok"] is True
    assert rec["sentinel"]["smoke_ok"] is False
    assert rec["agree"] is True


async def test_parity_detects_disagreement() -> None:
    # A pathological mock that always reports build_ok/smoke_ok False (e.g. the
    # sandbox killed the harness) must surface as a disagreement on a GOOD file.
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(202, json={"job_id": "j1", "status": "QUEUED"})
        # RUNTIME_ERROR -> build_ok/smoke_ok stay False.
        return httpx.Response(200, json={"status": "RUNTIME_ERROR", "exit_code": 1})

    report = await parity_report(
        [GOOD_DOCKERFILE], _spec(), sentinel_transport=httpx.MockTransport(handler)
    )
    # local-py says build/smoke True; sentinel says False -> disagreement.
    assert report["agreement_rate"] == 0.0
    assert len(report["disagreements"]) == 1
    d = report["disagreements"][0]
    assert d["local_py"]["build_ok"] is True
    assert d["sentinel"]["build_ok"] is False


@pytest.mark.parametrize("n", [2, 6])
async def test_parity_agreement_rate_arithmetic(n: int) -> None:
    spec = _spec()
    artifacts = [GOOD_DOCKERFILE if i % 2 == 0 else GARBAGE for i in range(n)]
    report = await parity_report(
        artifacts, spec, sentinel_transport=_executing_mock_transport()
    )
    assert report["n"] == n
    assert report["n_agree"] == n
    assert report["agreement_rate"] == pytest.approx(1.0)
