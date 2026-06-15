"""Tests for verifier.sentinel_client using httpx.MockTransport (no live server)."""
from __future__ import annotations

import json

import httpx
import pytest

from verifier.sentinel_client import SentinelClient, SentinelVerifier
from verifier.smoke.checks import build_python_harness
from verifier.types import ArtifactKind, ResourceLimits, VerifySpec

GOOD_DOCKERFILE = """\
FROM python:3.12-slim
WORKDIR /app
COPY . .
RUN pip install fastapi uvicorn
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
"""

JOB_ID = "01912345-6789-7abc-def0-123456789abc"


def _spec(kind=ArtifactKind.DOCKERFILE, **smoke) -> VerifySpec:
    return VerifySpec(
        spec_id="t", kind=kind, smoke=smoke, limits=ResourceLimits(wall_s=10, mem_mb=128)
    )


def _make_transport(final_job: dict, *, queued_polls: int = 1) -> httpx.MockTransport:
    """Transport: POST -> 202 {job_id}; first N GETs -> RUNNING; then final_job."""
    state = {"polls": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/api/v1/submissions":
            body = json.loads(request.content)
            assert body["language"] == "python"
            assert "source_code" in body
            return httpx.Response(202, json={"job_id": JOB_ID, "status": "QUEUED"})
        if request.method == "GET" and request.url.path.endswith(f"/{JOB_ID}"):
            state["polls"] += 1
            if state["polls"] <= queued_polls:
                return httpx.Response(200, json={"status": "RUNNING"})
            return httpx.Response(200, json=final_job)
        return httpx.Response(404, json={"error": "not found"})

    return httpx.MockTransport(handler)


def _verifier_with(final_job: dict, **kw) -> SentinelVerifier:
    client = SentinelClient(transport=_make_transport(final_job, **kw))
    return SentinelVerifier(client=client, poll_interval=0.001, deadline_s=5)


# --- client basics ---------------------------------------------------------
async def test_client_submit_and_get_roundtrip() -> None:
    job = {"status": "SUCCESS", "exit_code": 0, "stdout": "hi\n"}
    async with SentinelClient(transport=_make_transport(job, queued_polls=0)) as c:
        jid = await c.submit("print('hi')")
        assert jid == JOB_ID
        got = await c.get(jid)
        assert got["status"] == "SUCCESS"


async def test_client_run_polls_until_terminal() -> None:
    job = {"status": "SUCCESS", "exit_code": 0, "stdout": "x"}
    async with SentinelClient(transport=_make_transport(job, queued_polls=2)) as c:
        result = await c.run("print(1)", poll_interval=0.001, deadline_s=5)
        assert result["status"] == "SUCCESS"


async def test_client_run_deadline_raises() -> None:
    # never reaches terminal -> TimeoutError
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(202, json={"job_id": JOB_ID, "status": "QUEUED"})
        return httpx.Response(200, json={"status": "RUNNING"})

    async with SentinelClient(transport=httpx.MockTransport(handler)) as c:
        with pytest.raises(TimeoutError):
            await c.run("x", poll_interval=0.001, deadline_s=0.05)


async def test_client_submit_non_202_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    async with SentinelClient(transport=httpx.MockTransport(handler)) as c:
        with pytest.raises(httpx.HTTPStatusError):
            await c.submit("x")


async def test_client_sends_limits_and_bearer() -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            seen["body"] = json.loads(request.content)
            seen["auth"] = request.headers.get("authorization")
            return httpx.Response(202, json={"job_id": JOB_ID, "status": "QUEUED"})
        return httpx.Response(200, json={"status": "SUCCESS", "exit_code": 0})

    async with SentinelClient(api_key="secret", transport=httpx.MockTransport(handler)) as c:
        await c.run("x", time_limit_ms=5000, memory_limit_kb=2048, poll_interval=0.001)
    assert seen["body"]["time_limit_ms"] == 5000
    assert seen["body"]["memory_limit_kb"] == 2048
    assert seen["auth"] == "Bearer secret"


# --- SentinelVerifier mapping ----------------------------------------------
async def test_verifier_dockerfile_success_parses_harness() -> None:
    spec = _spec(must_contain=["FROM", "CMD"], base_image_prefix="python:3.12", port=8000)
    # Sentinel ran our harness; harness printed the contract JSON to stdout.
    harness_json = json.dumps(
        {"build_ok": True, "smoke_ok": True, "signals": {"spec_gaming": False}, "reasons": []}
    )
    job = {
        "status": "SUCCESS",
        "exit_code": 0,
        "stdout": harness_json + "\n",
        "stderr": "",
        "time_used_ms": 1500,
        "memory_used_kb": 20480,
    }
    v = _verifier_with(job)
    res = await v.verify(GOOD_DOCKERFILE, spec)
    assert res.backend == "sentinel"
    assert res.status == "SUCCESS"
    assert res.build_ok is True
    assert res.smoke_ok is True
    assert res.exit_code == 0
    assert res.wall_s == pytest.approx(1.5)
    assert res.mem_mb == pytest.approx(20.0)
    # documented gap: these stay False
    assert res.hack_flags.seccomp_violation is False
    assert res.hack_flags.network_attempt is False


async def test_verifier_dockerfile_spec_gaming_signal_propagates() -> None:
    spec = _spec(must_contain=["FROM"], port=8000)
    harness_json = json.dumps(
        {"build_ok": True, "smoke_ok": False, "signals": {"spec_gaming": True}, "reasons": ["x"]}
    )
    job = {"status": "SUCCESS", "exit_code": 0, "stdout": harness_json}
    res = await _verifier_with(job).verify("FROM python:3.12\n", spec)
    assert res.hack_flags.spec_gaming is True
    assert res.raw["signals"]["spec_gaming"] is True


async def test_verifier_python_kind_submits_artifact_directly() -> None:
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            captured["src"] = json.loads(request.content)["source_code"]
            return httpx.Response(202, json={"job_id": JOB_ID, "status": "QUEUED"})
        return httpx.Response(200, json={"status": "SUCCESS", "exit_code": 0, "stdout": ""})

    client = SentinelClient(transport=httpx.MockTransport(handler))
    v = SentinelVerifier(client=client, poll_interval=0.001)
    code = "print('hello from python')\n"
    res = await v.verify(code, _spec(kind=ArtifactKind.PYTHON))
    # PYTHON kind submits the artifact verbatim (NOT the harness).
    assert captured["src"] == code
    assert res.build_ok is True  # ran clean
    assert res.smoke_ok is True  # exit 0


async def test_verifier_dockerfile_kind_submits_harness_not_artifact() -> None:
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            captured["src"] = json.loads(request.content)["source_code"]
            return httpx.Response(202, json={"job_id": JOB_ID, "status": "QUEUED"})
        return httpx.Response(200, json={"status": "SUCCESS", "exit_code": 0, "stdout": "{}"})

    client = SentinelClient(transport=httpx.MockTransport(handler))
    v = SentinelVerifier(client=client, poll_interval=0.001)
    spec = _spec(must_contain=["FROM"], port=8000)
    await v.verify(GOOD_DOCKERFILE, spec)
    expected = build_python_harness(GOOD_DOCKERFILE, spec)
    assert captured["src"] == expected
    assert captured["src"] != GOOD_DOCKERFILE


async def test_verifier_timeout_maps_flags() -> None:
    job = {"status": "TIMEOUT", "exit_code": None, "stdout": "", "stderr": "killed"}
    res = await _verifier_with(job).verify(GOOD_DOCKERFILE, _spec(port=8000))
    assert res.status == "TIMEOUT"
    assert res.build_ok is False
    assert res.smoke_ok is False
    assert res.hack_flags.timed_out is True
    assert res.hack_flags.resource_exhaustion is True


async def test_verifier_memory_limit_maps_oom() -> None:
    job = {"status": "MEMORY_LIMIT_EXCEEDED", "exit_code": None}
    res = await _verifier_with(job).verify(GOOD_DOCKERFILE, _spec(port=8000))
    assert res.hack_flags.oom_killed is True
    assert res.hack_flags.resource_exhaustion is True


async def test_verifier_exit_137_maps_oom() -> None:
    job = {"status": "SUCCESS", "exit_code": 137, "stdout": ""}
    res = await _verifier_with(job).verify(GOOD_DOCKERFILE, _spec(port=8000))
    assert res.exit_code == 137
    assert res.hack_flags.oom_killed is True
    assert res.hack_flags.resource_exhaustion is True


async def test_verifier_runtime_error_maps_false() -> None:
    job = {"status": "RUNTIME_ERROR", "exit_code": 1, "stderr": "Traceback ..."}
    res = await _verifier_with(job).verify(GOOD_DOCKERFILE, _spec(port=8000))
    assert res.status == "RUNTIME_ERROR"
    assert res.build_ok is False
    assert res.smoke_ok is False
    assert "Traceback" in res.stderr_tail


async def test_verifier_client_timeout_maps_flags() -> None:
    # Server never reaches terminal -> SentinelClient.run raises TimeoutError ->
    # verify() maps it to timed_out/resource_exhaustion (status client-timeout).
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(202, json={"job_id": JOB_ID, "status": "QUEUED"})
        return httpx.Response(200, json={"status": "RUNNING"})

    client = SentinelClient(transport=httpx.MockTransport(handler))
    v = SentinelVerifier(client=client, poll_interval=0.001, deadline_s=0.03)
    res = await v.verify(GOOD_DOCKERFILE, _spec(port=8000))
    assert res.status == "client-timeout"
    assert res.hack_flags.timed_out is True


async def test_verifier_maps_limits_from_spec() -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            seen["body"] = json.loads(request.content)
            return httpx.Response(202, json={"job_id": JOB_ID, "status": "QUEUED"})
        return httpx.Response(200, json={"status": "SUCCESS", "exit_code": 0, "stdout": "{}"})

    client = SentinelClient(transport=httpx.MockTransport(handler))
    v = SentinelVerifier(client=client, poll_interval=0.001)
    spec = _spec(port=8000)  # wall_s=10, mem_mb=128
    await v.verify(GOOD_DOCKERFILE, spec)
    assert seen["body"]["time_limit_ms"] == 10_000  # 10 * 1000
    assert seen["body"]["memory_limit_kb"] == 128 * 1024
