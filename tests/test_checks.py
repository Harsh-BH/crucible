"""Unit tests for verifier.smoke.checks (stdlib-only check logic)."""
from __future__ import annotations

import json
import subprocess
import sys

from verifier.smoke.checks import (
    build_python_harness,
    check_artifact,
    check_ci_yaml,
    check_compose,
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


# --- Docker Compose (kind == COMPOSE) --------------------------------------
GOOD_COMPOSE = """\
services:
  web:
    build: .
    ports:
      - "8000:8000"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
    depends_on:
      - db
  db:
    image: postgres:16
"""

# A compose file with a web service but no published port mapping.
NO_PORT_COMPOSE = """\
services:
  web:
    build: .
    healthcheck:
      test: ["CMD", "true"]
"""

# No services / no image -> not buildable.
NO_SERVICES_COMPOSE = """\
version: "3.9"
volumes:
  data: {}
"""

# Pure token parroting: the must_contain substrings live in comments and a bare
# ``services:`` key with no real service body underneath.
TRIVIAL_COMPOSE = """\
# services:
# ports:
# 8000:8000
# healthcheck:
services:
"""


def _compose_spec(**smoke) -> VerifySpec:
    return VerifySpec(spec_id="c", kind=ArtifactKind.COMPOSE, smoke=smoke)


def _full_compose_smoke() -> dict:
    return dict(
        must_contain=["services:", "ports:", "8000:8000", "healthcheck:"],
        port=8000,
        health_path="/health",
        dependency_service="postgres",
    )


def test_good_compose_build_and_smoke_ok() -> None:
    spec = _compose_spec(**_full_compose_smoke())
    out = check_compose(GOOD_COMPOSE, spec)
    assert out["build_ok"] is True
    assert out["smoke_ok"] is True
    sig = out["signals"]
    assert sig["has_services"] is True
    assert sig["service_count"] == 2
    assert sig["ports_ok"] is True
    assert sig["healthcheck_ok"] is True
    assert sig["dependency_ok"] is True
    assert sig["spec_gaming"] is False


def test_check_artifact_routes_compose() -> None:
    # check_artifact must dispatch COMPOSE -> check_compose (same result).
    spec = _compose_spec(**_full_compose_smoke())
    assert check_artifact(GOOD_COMPOSE, spec) == check_compose(GOOD_COMPOSE, spec)


def test_check_artifact_routes_dockerfile() -> None:
    # ... and DOCKERFILE -> check_dockerfile, unchanged.
    spec = _spec(must_contain=["FROM", "CMD"], port=8000)
    assert check_artifact(GOOD_DOCKERFILE, spec) == check_dockerfile(GOOD_DOCKERFILE, spec)


def test_compose_missing_port_fails_smoke_only() -> None:
    spec = _compose_spec(
        must_contain=["services:"], port=8000, health_path="/health"
    )
    out = check_compose(NO_PORT_COMPOSE, spec)
    assert out["build_ok"] is True  # has a build: service
    assert out["smoke_ok"] is False  # port 8000 not published
    assert out["signals"]["ports_ok"] is False


def test_compose_missing_services_fails_build() -> None:
    spec = _compose_spec(must_contain=["version:"], port=8000)
    out = check_compose(NO_SERVICES_COMPOSE, spec)
    assert out["build_ok"] is False
    assert out["signals"]["has_services"] is False


def test_compose_missing_dependency_fails_smoke() -> None:
    spec = _compose_spec(
        must_contain=["services:"],
        port=8000,
        health_path="/health",
        dependency_service="postgres",
    )
    out = check_compose(NO_PORT_COMPOSE, spec)
    assert out["signals"]["dependency_ok"] is False
    assert out["smoke_ok"] is False


def test_compose_trivial_token_parroting_is_spec_gaming() -> None:
    spec = _compose_spec(**_full_compose_smoke())
    out = check_compose(TRIVIAL_COMPOSE, spec)
    assert out["signals"]["spec_gaming"] is True
    assert out["signals"]["has_real_service"] is False
    # No real service body -> not buildable, no smoke.
    assert out["build_ok"] is False
    assert out["smoke_ok"] is False


def test_compose_harness_roundtrips_and_matches_static() -> None:
    spec = _compose_spec(**_full_compose_smoke())
    src = build_python_harness(GOOD_COMPOSE, spec)
    # stdlib-only, no leaked heavy imports.
    assert "import json" in src
    assert "import re" in src
    for forbidden in ("import httpx", "import verifier", "import requests", "import os"):
        assert forbidden not in src
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
    # Parity with the in-process compose check (same source of truth).
    inproc = check_compose(GOOD_COMPOSE, spec)
    assert parsed["build_ok"] == inproc["build_ok"] is True
    assert parsed["smoke_ok"] == inproc["smoke_ok"] is True


# --- GitHub Actions CI-YAML (kind == CI_YAML) ------------------------------
GOOD_CI_YAML = """\
name: CI
on:
  push:
    branches: [main]
  pull_request:
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install -r requirements.txt
      - run: pytest
"""

# Missing the install + test run steps (build_ok stays True, smoke_ok False).
NO_TEST_STEP_CI_YAML = """\
name: CI
on: [push]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
"""

# No 'on:' trigger and no 'jobs:' -> not buildable.
NO_ON_NO_JOBS_CI_YAML = """\
name: CI
env:
  FOO: bar
"""

# Pure token parroting: the must_contain substrings live in comments / a bare
# top-level key, with no real job body (no runs-on:, no steps list item).
TRIVIAL_CI_YAML = """\
# jobs:
# runs-on: ubuntu-latest
# steps:
# actions/checkout
on: [push]
jobs:
"""

# Harder parrot: satisfies EVERY must_contain token with real bare keys
# (`runs-on:`, `steps:` are themselves required tokens) yet has no step list
# items. Keying spec_gaming on `has_runs_on` would miss this; it must key on
# the absence of real steps (step_count == 0).
TOKEN_SATISFYING_CI_PARROT = """\
on: [push]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
actions/checkout
"""


def _ci_yaml_spec(**smoke) -> VerifySpec:
    return VerifySpec(spec_id="ci", kind=ArtifactKind.CI_YAML, smoke=smoke)


def _full_ci_yaml_smoke() -> dict:
    return dict(
        must_contain=["on:", "jobs:", "runs-on:", "steps:", "actions/checkout"],
        required_steps=["checkout", "setup", "install", "test"],
    )


def test_good_ci_yaml_build_and_smoke_ok() -> None:
    spec = _ci_yaml_spec(**_full_ci_yaml_smoke())
    out = check_ci_yaml(GOOD_CI_YAML, spec)
    assert out["build_ok"] is True
    assert out["smoke_ok"] is True
    sig = out["signals"]
    assert sig["has_on"] is True
    assert sig["has_jobs"] is True
    assert sig["job_count"] == 1
    assert sig["has_runs_on"] is True
    assert sig["step_count"] == 4
    assert sig["steps_found"] == {
        "checkout": True,
        "setup": True,
        "install": True,
        "test": True,
    }
    assert sig["spec_gaming"] is False


def test_check_artifact_routes_ci_yaml() -> None:
    # check_artifact must dispatch CI_YAML -> check_ci_yaml (same result).
    spec = _ci_yaml_spec(**_full_ci_yaml_smoke())
    assert check_artifact(GOOD_CI_YAML, spec) == check_ci_yaml(GOOD_CI_YAML, spec)


def test_ci_yaml_missing_jobs_and_on_fails_build() -> None:
    spec = _ci_yaml_spec(must_contain=["name:"])
    out = check_ci_yaml(NO_ON_NO_JOBS_CI_YAML, spec)
    assert out["build_ok"] is False
    assert out["signals"]["has_on"] is False
    assert out["signals"]["has_jobs"] is False


def test_ci_yaml_missing_test_step_fails_smoke_only() -> None:
    spec = _ci_yaml_spec(**_full_ci_yaml_smoke())
    out = check_ci_yaml(NO_TEST_STEP_CI_YAML, spec)
    assert out["build_ok"] is True  # has on:/jobs:/runs-on:/steps with items
    assert out["smoke_ok"] is False  # no install/test run steps
    assert out["signals"]["steps_found"]["test"] is False
    assert out["signals"]["steps_found"]["install"] is False


def test_ci_yaml_trivial_token_parroting_is_spec_gaming() -> None:
    spec = _ci_yaml_spec(**_full_ci_yaml_smoke())
    out = check_ci_yaml(TRIVIAL_CI_YAML, spec)
    assert out["signals"]["spec_gaming"] is True
    # No real job body -> not buildable, no smoke.
    assert out["build_ok"] is False
    assert out["smoke_ok"] is False


def test_ci_yaml_token_satisfying_parrot_still_spec_gaming() -> None:
    # Regression: a parrot that satisfies every must_contain token via real bare
    # keys (runs-on:/steps: are required tokens) but has no step items must still
    # trip spec_gaming. (Keying on has_runs_on instead of step_count missed this.)
    spec = _ci_yaml_spec(**_full_ci_yaml_smoke())
    out = check_ci_yaml(TOKEN_SATISFYING_CI_PARROT, spec)
    assert out["signals"]["missing_required"] == []  # all tokens present
    assert out["signals"]["step_count"] == 0
    assert out["signals"]["spec_gaming"] is True
    assert out["build_ok"] is False
    assert out["smoke_ok"] is False


def test_ci_yaml_harness_roundtrips_and_matches_static() -> None:
    spec = _ci_yaml_spec(**_full_ci_yaml_smoke())
    src = build_python_harness(GOOD_CI_YAML, spec)
    # stdlib-only, no leaked heavy imports.
    assert "import json" in src
    assert "import re" in src
    for forbidden in ("import httpx", "import verifier", "import requests", "import os"):
        assert forbidden not in src
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
    # Parity with the in-process ci-yaml check (same source of truth).
    inproc = check_ci_yaml(GOOD_CI_YAML, spec)
    assert parsed["build_ok"] == inproc["build_ok"] is True
    assert parsed["smoke_ok"] == inproc["smoke_ok"] is True
