"""Unit tests for verifier.smoke.checks (stdlib-only check logic)."""
from __future__ import annotations

import json
import subprocess
import sys

from verifier.smoke.checks import (
    build_python_harness,
    check_dockerfile,
    parse_harness_output,
)
from verifier.types import ArtifactKind, VerifySpec

GOOD_DOCKERFILE = """\
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
"""

# Trivial artifact that merely parrots the must_contain tokens.
TRIVIAL_GAMING = """\
FROM python:3.12-slim
EXPOSE 8000
CMD ["true"]
"""

NO_FROM = """\
RUN echo hi
CMD ["python", "app.py"]
"""


def _spec(**smoke) -> VerifySpec:
    return VerifySpec(spec_id="t", kind=ArtifactKind.DOCKERFILE, smoke=smoke)


def test_good_dockerfile_build_and_smoke_ok() -> None:
    spec = _spec(
        must_contain=["FROM", "CMD", "EXPOSE 8000"],
        base_image_prefix="python:3.12",
        port=8000,
        health_path="/health",
    )
    out = check_dockerfile(GOOD_DOCKERFILE, spec)
    assert out["build_ok"] is True
    assert out["smoke_ok"] is True
    sig = out["signals"]
    assert sig["base_image"] == "python:3.12-slim"
    assert sig["exposed_ports"] == [8000]
    assert sig["launches_server"] is True
    assert sig["spec_gaming"] is False


def test_missing_required_token_fails_build() -> None:
    spec = _spec(must_contain=["FROM", "HEALTHCHECK"], port=8000)
    out = check_dockerfile(GOOD_DOCKERFILE, spec)
    assert out["build_ok"] is False
    assert "HEALTHCHECK" in out["signals"]["missing_required"]


def test_wrong_base_prefix_fails_build() -> None:
    spec = _spec(must_contain=["FROM"], base_image_prefix="node:20")
    out = check_dockerfile(GOOD_DOCKERFILE, spec)
    assert out["build_ok"] is False
    assert out["signals"]["base_prefix_ok"] is False


def test_no_from_fails() -> None:
    spec = _spec(must_contain=["CMD"])
    out = check_dockerfile(NO_FROM, spec)
    assert out["build_ok"] is False
    assert out["signals"]["from_count"] == 0


def test_scratch_base_is_weaker_and_not_build_ok() -> None:
    spec = _spec(must_contain=["FROM"])
    out = check_dockerfile("FROM scratch\nCMD [\"/app\"]\n", spec)
    assert out["signals"]["base_image_scratch"] is True
    assert out["build_ok"] is False


def test_port_not_exposed_fails_smoke_only() -> None:
    df = """\
FROM python:3.12-slim
RUN pip install fastapi uvicorn
COPY . .
CMD ["uvicorn", "app:app"]
"""
    spec = _spec(must_contain=["FROM", "CMD"], port=9000)
    out = check_dockerfile(df, spec)
    assert out["build_ok"] is True  # builds fine
    assert out["smoke_ok"] is False  # port 9000 not exposed
    assert out["signals"]["port_ok"] is False


def test_no_server_launch_fails_smoke() -> None:
    df = """\
FROM python:3.12-slim
RUN pip install requests
COPY . .
EXPOSE 8000
CMD ["python", "compute_once.py"]
"""
    spec = _spec(must_contain=["FROM", "CMD"], port=8000)
    out = check_dockerfile(df, spec)
    assert out["build_ok"] is True
    assert out["smoke_ok"] is False
    assert out["signals"]["launches_server"] is False


def test_spec_gaming_detected_on_trivial_artifact() -> None:
    # Tokens present, build may even pass, but no real setup -> spec_gaming.
    spec = _spec(must_contain=["FROM", "EXPOSE 8000"], port=8000)
    out = check_dockerfile(TRIVIAL_GAMING, spec)
    assert out["signals"]["spec_gaming"] is True
    # CMD ["true"] is not a server -> smoke must NOT pass.
    assert out["smoke_ok"] is False


def test_real_app_not_flagged_as_gaming() -> None:
    spec = _spec(must_contain=["FROM", "CMD"], port=8000)
    out = check_dockerfile(GOOD_DOCKERFILE, spec)
    assert out["signals"]["spec_gaming"] is False
    assert out["signals"]["has_real_setup"] is True


def test_line_continuation_joined() -> None:
    df = (
        "FROM python:3.12-slim\n"
        "RUN apt-get update \\\n"
        "  && apt-get install -y curl\n"
        "EXPOSE 8000\n"
        'CMD ["uvicorn","a:a"]\n'
    )
    spec = _spec(must_contain=["FROM"], port=8000)
    out = check_dockerfile(df, spec)
    # The continued RUN should count as one logical line / real setup.
    assert out["signals"]["has_real_setup"] is True
    assert out["build_ok"] is True


# --- harness round-trip ----------------------------------------------------
def test_build_python_harness_is_stdlib_only_source() -> None:
    spec = _spec(must_contain=["FROM", "CMD"], port=8000, health_path="/health")
    src = build_python_harness(GOOD_DOCKERFILE, spec)
    assert "import json" in src
    assert "import re" in src
    # No third-party / heavy imports leaked into the harness.
    for forbidden in ("import httpx", "import verifier", "import requests", "import os"):
        assert forbidden not in src


def test_harness_runs_in_subprocess_and_roundtrips() -> None:
    spec = _spec(
        must_contain=["FROM", "CMD", "EXPOSE 8000"],
        base_image_prefix="python:3.12",
        port=8000,
        health_path="/health",
    )
    src = build_python_harness(GOOD_DOCKERFILE, spec)
    proc = subprocess.run(
        [sys.executable, "-I", "-c", src],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    parsed = parse_harness_output(proc.stdout)
    assert parsed is not None
    assert parsed["build_ok"] is True
    assert parsed["smoke_ok"] is True
    # Matches the in-process check exactly (same source of truth).
    inproc = check_dockerfile(GOOD_DOCKERFILE, spec)
    assert parsed["build_ok"] == inproc["build_ok"]
    assert parsed["smoke_ok"] == inproc["smoke_ok"]


def test_harness_failing_artifact_roundtrips_false() -> None:
    spec = _spec(must_contain=["FROM"], port=8000)
    src = build_python_harness(NO_FROM, spec)
    proc = subprocess.run(
        [sys.executable, "-I", "-c", src], capture_output=True, text=True, timeout=30,
        check=False,
    )
    assert proc.returncode == 0
    parsed = parse_harness_output(proc.stdout)
    assert parsed is not None
    assert parsed["build_ok"] is False


def test_parse_harness_output_ignores_noise() -> None:
    noisy = "warning: something\nrandom line\n" + json.dumps(
        {"build_ok": True, "smoke_ok": False, "signals": {}, "reasons": []}
    ) + "\n"
    parsed = parse_harness_output(noisy)
    assert parsed == {"build_ok": True, "smoke_ok": False, "signals": {}, "reasons": []}


def test_parse_harness_output_none_when_absent() -> None:
    assert parse_harness_output("no json here\n") is None
    assert parse_harness_output("") is None
    # A JSON line without the build_ok marker is not the contract line.
    assert parse_harness_output('{"foo": 1}\n') is None
