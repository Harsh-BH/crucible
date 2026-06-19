"""Tests for verifier.backends (StaticVerifier, LocalPyVerifier,
LocalDockerVerifier mapping) and the get_verifier factory."""
from __future__ import annotations

import shutil
import subprocess

import pytest

from verifier import (
    LocalComposeVerifier,
    LocalDockerVerifier,
    LocalGenuineVerifier,
    LocalK8sVerifier,
    LocalPyVerifier,
    LocalTerraformVerifier,
    SentinelVerifier,
    StaticVerifier,
    Verifier,
    get_verifier,
)
from verifier.types import ArtifactKind, ResourceLimits, VerifyResult, VerifySpec

GOOD_DOCKERFILE = """\
FROM python:3.12-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir fastapi uvicorn
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
"""

BAD_DOCKERFILE = "RUN echo nope\n"

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

# Token-parroting compose: must_contain substrings present (in comments) but no
# real service body underneath -> spec_gaming.
TRIVIAL_COMPOSE = """\
# services:
# ports:
# 8000:8000
# healthcheck:
services:
"""

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

# Token-parroting CI-YAML: must_contain substrings present (in comments) but no
# real job body underneath (no runs-on:, no steps list item) -> spec_gaming.
TRIVIAL_CI_YAML = """\
# jobs:
# runs-on: ubuntu-latest
# steps:
# actions/checkout
on: [push]
jobs:
"""


def _spec(**smoke) -> VerifySpec:
    return VerifySpec(
        spec_id="t",
        kind=ArtifactKind.DOCKERFILE,
        smoke=smoke,
        limits=ResourceLimits(wall_s=10, mem_mb=256),
    )


def _compose_spec(**smoke) -> VerifySpec:
    return VerifySpec(
        spec_id="t-compose",
        kind=ArtifactKind.COMPOSE,
        smoke=smoke,
        limits=ResourceLimits(wall_s=10, mem_mb=256),
    )


def _full_compose_smoke() -> dict:
    return dict(
        must_contain=["services:", "ports:", "8000:8000", "healthcheck:"],
        port=8000,
        health_path="/health",
        dependency_service="postgres",
    )


def _ci_yaml_spec(**smoke) -> VerifySpec:
    return VerifySpec(
        spec_id="t-ci",
        kind=ArtifactKind.CI_YAML,
        smoke=smoke,
        limits=ResourceLimits(wall_s=10, mem_mb=256),
    )


def _full_ci_yaml_smoke() -> dict:
    return dict(
        must_contain=["on:", "jobs:", "runs-on:", "steps:", "actions/checkout"],
        required_steps=["checkout", "setup", "install", "test"],
    )


def _py_spec(wall_s: int = 10) -> VerifySpec:
    return VerifySpec(
        spec_id="t-py",
        kind=ArtifactKind.PYTHON,
        smoke={},
        limits=ResourceLimits(wall_s=wall_s, mem_mb=256),
    )


# --- StaticVerifier --------------------------------------------------------
async def test_static_verifier_good() -> None:
    v = StaticVerifier()
    assert isinstance(v, Verifier)
    spec = _spec(must_contain=["FROM", "CMD"], base_image_prefix="python:3.12", port=8000)
    res = await v.verify(GOOD_DOCKERFILE, spec)
    assert res.backend == "static"
    assert res.build_ok is True
    assert res.smoke_ok is True
    assert res.exit_code == 0
    assert res.reward is None


async def test_static_verifier_bad() -> None:
    v = StaticVerifier()
    res = await v.verify(BAD_DOCKERFILE, _spec(must_contain=["FROM"]))
    assert res.build_ok is False


# --- StaticVerifier: kind dispatch to compose ------------------------------
async def test_static_verifier_compose_good() -> None:
    # StaticVerifier.verify -> check_artifact -> check_compose for a COMPOSE spec.
    v = StaticVerifier()
    res = await v.verify(GOOD_COMPOSE, _compose_spec(**_full_compose_smoke()))
    assert res.backend == "static"
    assert res.build_ok is True
    assert res.smoke_ok is True
    assert res.exit_code == 0
    assert res.hack_flags.spec_gaming is False
    assert res.reward is None


async def test_static_verifier_compose_spec_gaming_flag() -> None:
    # A trivial token-parroting compose trips hack_flags.spec_gaming.
    v = StaticVerifier()
    res = await v.verify(TRIVIAL_COMPOSE, _compose_spec(**_full_compose_smoke()))
    assert res.build_ok is False
    assert res.smoke_ok is False
    assert res.hack_flags.spec_gaming is True


# --- StaticVerifier: kind dispatch to ci-yaml ------------------------------
async def test_static_verifier_ci_yaml_good() -> None:
    # StaticVerifier.verify -> check_artifact -> check_ci_yaml for a CI_YAML spec.
    v = StaticVerifier()
    res = await v.verify(GOOD_CI_YAML, _ci_yaml_spec(**_full_ci_yaml_smoke()))
    assert res.backend == "static"
    assert res.build_ok is True
    assert res.smoke_ok is True
    assert res.exit_code == 0
    assert res.hack_flags.spec_gaming is False
    assert res.reward is None


async def test_static_verifier_ci_yaml_spec_gaming_flag() -> None:
    # A trivial token-parroting workflow trips hack_flags.spec_gaming.
    v = StaticVerifier()
    res = await v.verify(TRIVIAL_CI_YAML, _ci_yaml_spec(**_full_ci_yaml_smoke()))
    assert res.build_ok is False
    assert res.smoke_ok is False
    assert res.hack_flags.spec_gaming is True


# --- LocalPyVerifier (real subprocess) -------------------------------------
async def test_local_py_matches_static() -> None:
    spec = _spec(must_contain=["FROM", "CMD"], base_image_prefix="python:3.12", port=8000)
    static_res = await StaticVerifier().verify(GOOD_DOCKERFILE, spec)
    py_res = await LocalPyVerifier().verify(GOOD_DOCKERFILE, spec)
    assert py_res.backend == "local-py"
    assert py_res.exit_code == 0
    assert py_res.status == "ok"
    assert py_res.build_ok == static_res.build_ok is True
    assert py_res.smoke_ok == static_res.smoke_ok is True
    assert py_res.wall_s > 0.0
    assert py_res.reward is None


async def test_local_py_bad_dockerfile() -> None:
    py_res = await LocalPyVerifier().verify(BAD_DOCKERFILE, _spec(must_contain=["FROM"]))
    assert py_res.exit_code == 0  # harness itself exits 0
    assert py_res.build_ok is False


async def test_local_py_compose_matches_static() -> None:
    # COMPOSE harness inlines the compose check -> same (build_ok, smoke_ok) as
    # the in-process static path (mirrors test_local_py_matches_static).
    spec = _compose_spec(**_full_compose_smoke())
    static_res = await StaticVerifier().verify(GOOD_COMPOSE, spec)
    py_res = await LocalPyVerifier().verify(GOOD_COMPOSE, spec)
    assert py_res.backend == "local-py"
    assert py_res.exit_code == 0
    assert py_res.status == "ok"
    assert py_res.build_ok == static_res.build_ok is True
    assert py_res.smoke_ok == static_res.smoke_ok is True
    assert py_res.reward is None


async def test_local_py_ci_yaml_matches_static() -> None:
    # CI_YAML harness inlines the ci-yaml check -> same (build_ok, smoke_ok) as
    # the in-process static path (mirrors the compose + Dockerfile parity tests).
    spec = _ci_yaml_spec(**_full_ci_yaml_smoke())
    static_res = await StaticVerifier().verify(GOOD_CI_YAML, spec)
    py_res = await LocalPyVerifier().verify(GOOD_CI_YAML, spec)
    assert py_res.backend == "local-py"
    assert py_res.exit_code == 0
    assert py_res.status == "ok"
    assert py_res.build_ok == static_res.build_ok is True
    assert py_res.smoke_ok == static_res.smoke_ok is True
    assert py_res.reward is None


async def test_local_py_timeout_maps_flags(monkeypatch) -> None:
    v = LocalPyVerifier()

    def boom(*a, **k):
        raise subprocess.TimeoutExpired(cmd="python3", timeout=1, output="", stderr="")

    monkeypatch.setattr(subprocess, "run", boom)
    res = await v.verify(GOOD_DOCKERFILE, _spec(must_contain=["FROM"]))
    assert res.status == "timeout"
    assert res.hack_flags.timed_out is True
    assert res.hack_flags.resource_exhaustion is True


async def test_local_py_oom_maps_flags(monkeypatch) -> None:
    v = LocalPyVerifier()

    def fake(*a, **k):
        return subprocess.CompletedProcess(a[0], 137, stdout="", stderr="boom")

    monkeypatch.setattr(subprocess, "run", fake)
    res = await v.verify(GOOD_DOCKERFILE, _spec(must_contain=["FROM"]))
    assert res.status == "oom"
    assert res.hack_flags.oom_killed is True
    assert res.hack_flags.resource_exhaustion is True


# --- LocalPyVerifier PYTHON path (raw code, no harness) --------------------
async def test_local_py_python_exit_zero() -> None:
    # ArtifactKind.PYTHON runs the artifact itself; clean exit -> build+smoke ok
    # (mirrors SentinelVerifier's raw-code mapping).
    res = await LocalPyVerifier().verify("import sys; sys.exit(0)", _py_spec())
    assert res.backend == "local-py"
    assert res.exit_code == 0
    assert res.status == "ok"
    assert res.build_ok is True
    assert res.smoke_ok is True
    assert res.hack_flags.any() is False


async def test_local_py_python_nonzero_exit() -> None:
    # Ran (so "built") but exited non-zero -> build_ok True, smoke_ok False.
    res = await LocalPyVerifier().verify("import sys; sys.exit(3)", _py_spec())
    assert res.exit_code == 3
    assert res.status == "nonzero-exit"
    assert res.build_ok is True
    assert res.smoke_ok is False
    assert res.hack_flags.any() is False


async def test_local_py_python_timeout_maps_flags() -> None:
    # A real (tiny) wall-clock timeout: 5s sleep vs a 300ms limit.
    v = LocalPyVerifier(time_limit_ms=300)
    res = await v.verify("import time; time.sleep(5)", _py_spec())
    assert res.status == "timeout"
    assert res.hack_flags.timed_out is True
    assert res.hack_flags.resource_exhaustion is True


# --- get_verifier factory --------------------------------------------------
def test_factory_types() -> None:
    assert isinstance(get_verifier("static"), StaticVerifier)
    assert isinstance(get_verifier("local-py"), LocalPyVerifier)
    assert isinstance(get_verifier("local-docker"), LocalDockerVerifier)
    assert isinstance(get_verifier("local-compose"), LocalComposeVerifier)
    assert isinstance(get_verifier("local-terraform"), LocalTerraformVerifier)
    assert isinstance(get_verifier("local-k8s"), LocalK8sVerifier)
    assert isinstance(get_verifier("local"), LocalGenuineVerifier)
    assert isinstance(get_verifier("sentinel"), SentinelVerifier)
    # default
    assert isinstance(get_verifier(), StaticVerifier)


def test_factory_default_is_static() -> None:
    assert get_verifier().name == "static"


def test_factory_passes_kwargs() -> None:
    v = get_verifier("local-py", time_limit_ms=1234, mem_mb=99)
    assert isinstance(v, LocalPyVerifier)
    assert v.time_limit_ms == 1234
    assert v.mem_mb == 99


def test_factory_sentinel_base_url() -> None:
    v = get_verifier("sentinel", base_url="http://example:9999")
    assert isinstance(v, SentinelVerifier)
    assert v._client.base_url == "http://example:9999"


def test_factory_unknown_raises() -> None:
    with pytest.raises(ValueError, match="unknown verifier name"):
        get_verifier("does-not-exist")


def test_factory_returns_protocol() -> None:
    for name in (
        "static", "local-py", "local-docker", "local-compose",
        "local-terraform", "local-k8s", "local", "sentinel",
    ):
        assert isinstance(get_verifier(name), Verifier)
        assert get_verifier(name).name == name


# --- LocalDockerVerifier: mapping unit-tested with faked steps -------------
async def test_local_docker_unavailable(monkeypatch) -> None:
    v = LocalDockerVerifier()
    monkeypatch.setattr(v, "_docker_available", lambda: False)
    res = await v.verify(GOOD_DOCKERFILE, _spec(port=8000))
    assert res.backend == "local-docker"
    assert res.status == "docker-unavailable"
    assert res.build_ok is False


def _ok_proc(stdout="", stderr="", rc=0):
    return subprocess.CompletedProcess(["docker"], rc, stdout=stdout, stderr=stderr)


async def test_local_docker_full_success_mapping(monkeypatch) -> None:
    v = LocalDockerVerifier()
    monkeypatch.setattr(v, "_docker_available", lambda: True)
    monkeypatch.setattr(v, "_docker_build", lambda *a, **k: _ok_proc("built"))
    monkeypatch.setattr(v, "_docker_run", lambda *a, **k: _ok_proc("container123\n"))
    monkeypatch.setattr(v, "_probe", lambda url, status, dl: (True, "HTTP 200"))
    stopped = {}
    monkeypatch.setattr(v, "_docker_stop", lambda cid: stopped.setdefault("id", cid))
    monkeypatch.setattr(v, "_docker_rmi", lambda tag: None)

    res = await v.verify(GOOD_DOCKERFILE, _spec(port=8000, health_path="/health"))
    assert res.build_ok is True
    assert res.smoke_ok is True
    assert res.status == "smoke-ok"
    assert stopped["id"] == "container123"  # teardown happened


async def test_local_docker_build_failure_mapping(monkeypatch) -> None:
    v = LocalDockerVerifier()
    monkeypatch.setattr(v, "_docker_available", lambda: True)
    monkeypatch.setattr(v, "_docker_build", lambda *a, **k: _ok_proc("", "boom", rc=1))
    monkeypatch.setattr(v, "_docker_rmi", lambda tag: None)
    res = await v.verify(BAD_DOCKERFILE, _spec(port=8000))
    assert res.build_ok is False
    assert res.smoke_ok is False
    assert res.status == "build-failed"
    assert res.exit_code == 1


async def test_local_docker_smoke_failure_mapping(monkeypatch) -> None:
    v = LocalDockerVerifier()
    monkeypatch.setattr(v, "_docker_available", lambda: True)
    monkeypatch.setattr(v, "_docker_build", lambda *a, **k: _ok_proc("built"))
    monkeypatch.setattr(v, "_docker_run", lambda *a, **k: _ok_proc("cid\n"))
    monkeypatch.setattr(v, "_probe", lambda url, status, dl: (False, "no response"))
    monkeypatch.setattr(v, "_docker_stop", lambda cid: None)
    monkeypatch.setattr(v, "_docker_rmi", lambda tag: None)
    res = await v.verify(GOOD_DOCKERFILE, _spec(port=8000))
    assert res.build_ok is True  # built fine
    assert res.smoke_ok is False  # but never served
    assert res.status == "smoke-failed"


async def test_local_docker_build_timeout_mapping(monkeypatch) -> None:
    v = LocalDockerVerifier()
    monkeypatch.setattr(v, "_docker_available", lambda: True)

    def timeout(*a, **k):
        raise subprocess.TimeoutExpired(cmd="docker build", timeout=1)

    monkeypatch.setattr(v, "_docker_build", timeout)
    monkeypatch.setattr(v, "_docker_rmi", lambda tag: None)
    res = await v.verify(GOOD_DOCKERFILE, _spec(port=8000))
    assert res.status == "build-timeout"
    assert res.hack_flags.timed_out is True
    assert res.hack_flags.resource_exhaustion is True


# --- LocalDockerVerifier: real build (skipped if no docker) ----------------
@pytest.mark.skipif(shutil.which("docker") is None, reason="docker CLI not available")
async def test_local_docker_real_build_returns_result() -> None:
    # A genuine but tiny build that does NOT need network at run time; we only
    # assert the call completes and yields a VerifyResult with a sane status.
    # (We avoid asserting smoke_ok=True since base-image pulls may be offline.)
    df = "FROM hello-world\n"
    v = LocalDockerVerifier(remove_image=True)
    spec = _spec(port=8000)
    res = await v.verify(df, spec)
    assert isinstance(res, VerifyResult)
    assert res.backend == "local-docker"
    # Either it built, or the build failed (e.g. offline) -- both are valid
    # outcomes; we just confirm no exception and a recorded status.
    assert res.status in {
        "smoke-ok", "smoke-failed", "build-failed", "built", "run-failed",
        "build-timeout", "run-timeout",
    }


# --- LocalComposeVerifier: mapping unit-tested with faked steps ------------
async def test_local_compose_unavailable(monkeypatch) -> None:
    v = LocalComposeVerifier()
    monkeypatch.setattr(v, "_compose_available", lambda: False)
    res = await v.verify(GOOD_COMPOSE, _compose_spec(port=8000))
    assert res.backend == "local-compose"
    assert res.status == "docker-unavailable"
    assert res.build_ok is False
    assert res.stderr_tail == "docker CLI not found on PATH"


async def test_local_compose_full_success_mapping(monkeypatch) -> None:
    v = LocalComposeVerifier()
    monkeypatch.setattr(v, "_compose_available", lambda: True)
    monkeypatch.setattr(v, "_compose_up", lambda *a, **k: _ok_proc("up"))
    monkeypatch.setattr(v, "_probe", lambda url, status, dl: (True, "HTTP 200"))
    torn = {}
    monkeypatch.setattr(
        v, "_compose_down", lambda ctx, project: torn.setdefault("project", project)
    )

    res = await v.verify(GOOD_COMPOSE, _compose_spec(port=8000, health_path="/health"))
    assert res.build_ok is True
    assert res.smoke_ok is True
    assert res.status == "smoke-ok"
    assert torn["project"].startswith("crucible-compose-")  # teardown happened


async def test_local_compose_smoke_failure_mapping(monkeypatch) -> None:
    v = LocalComposeVerifier()
    monkeypatch.setattr(v, "_compose_available", lambda: True)
    monkeypatch.setattr(v, "_compose_up", lambda *a, **k: _ok_proc("up"))
    monkeypatch.setattr(v, "_probe", lambda url, status, dl: (False, "no response"))
    monkeypatch.setattr(v, "_compose_down", lambda ctx, project: None)
    res = await v.verify(GOOD_COMPOSE, _compose_spec(port=8000))
    assert res.build_ok is True  # came up fine
    assert res.smoke_ok is False  # but never served
    assert res.status == "smoke-failed"


async def test_local_compose_up_failure_mapping(monkeypatch) -> None:
    v = LocalComposeVerifier()
    monkeypatch.setattr(v, "_compose_available", lambda: True)
    monkeypatch.setattr(v, "_compose_up", lambda *a, **k: _ok_proc("", "boom", rc=1))
    monkeypatch.setattr(v, "_compose_down", lambda ctx, project: None)
    res = await v.verify(GOOD_COMPOSE, _compose_spec(port=8000))
    assert res.build_ok is False
    assert res.smoke_ok is False
    assert res.status == "compose-up-failed"
    assert res.exit_code == 1


async def test_local_compose_timeout_mapping(monkeypatch) -> None:
    v = LocalComposeVerifier()
    monkeypatch.setattr(v, "_compose_available", lambda: True)

    def timeout(*a, **k):
        raise subprocess.TimeoutExpired(cmd="docker compose up", timeout=1)

    monkeypatch.setattr(v, "_compose_up", timeout)
    monkeypatch.setattr(v, "_compose_down", lambda ctx, project: None)
    res = await v.verify(GOOD_COMPOSE, _compose_spec(port=8000))
    assert res.status == "compose-timeout"
    assert res.hack_flags.timed_out is True
    assert res.hack_flags.resource_exhaustion is True


async def test_local_compose_oom_on_up_maps_flags(monkeypatch) -> None:
    v = LocalComposeVerifier()
    monkeypatch.setattr(v, "_compose_available", lambda: True)
    monkeypatch.setattr(v, "_compose_up", lambda *a, **k: _ok_proc("", "Killed", rc=137))
    monkeypatch.setattr(v, "_compose_down", lambda ctx, project: None)
    res = await v.verify(GOOD_COMPOSE, _compose_spec(port=8000))
    assert res.status == "compose-up-failed"
    assert res.build_ok is False
    assert res.hack_flags.oom_killed is True
    assert res.hack_flags.resource_exhaustion is True


async def test_local_compose_context_files_traversal_guard(monkeypatch) -> None:
    # A context_files key escaping the build context must raise (no write
    # outside the temp dir) -- same guard as LocalDockerVerifier.
    v = LocalComposeVerifier()
    monkeypatch.setattr(v, "_compose_available", lambda: True)
    # _compose_up should never be reached; fail loudly if it is.
    monkeypatch.setattr(
        v, "_compose_up", lambda *a, **k: pytest.fail("up reached despite bad path")
    )
    down = {}
    monkeypatch.setattr(v, "_compose_down", lambda ctx, project: down.setdefault("hit", True))
    spec = _compose_spec(port=8000, context_files={"../evil": "x"})
    with pytest.raises(ValueError, match="escapes build context"):
        await v.verify(GOOD_COMPOSE, spec)
    assert down.get("hit") is True  # teardown still ran in finally


# --- spec helpers + sample artifacts for terraform/k8s ----------------------
GOOD_TERRAFORM = """\
terraform {
  required_providers {
    null = {
      source = "hashicorp/null"
    }
  }
}

resource "null_resource" "noop" {}
"""

GOOD_K8S = """\
apiVersion: apps/v1
kind: Deployment
metadata:
  name: web
spec:
  replicas: 1
  selector:
    matchLabels:
      app: web
  template:
    metadata:
      labels:
        app: web
    spec:
      containers:
        - name: web
          image: nginx:1.27
"""


def _tf_spec(**smoke) -> VerifySpec:
    return VerifySpec(
        spec_id="t-tf",
        kind=ArtifactKind.TERRAFORM,
        smoke=smoke,
        limits=ResourceLimits(wall_s=10, mem_mb=256),
    )


def _k8s_spec(**smoke) -> VerifySpec:
    return VerifySpec(
        spec_id="t-k8s",
        kind=ArtifactKind.K8S,
        smoke=smoke,
        limits=ResourceLimits(wall_s=10, mem_mb=256),
    )


# --- LocalTerraformVerifier: mapping unit-tested with faked hooks ----------
async def test_local_terraform_unavailable(monkeypatch) -> None:
    v = LocalTerraformVerifier()
    monkeypatch.setattr(v, "_terraform_available", lambda: False)
    res = await v.verify(GOOD_TERRAFORM, _tf_spec())
    assert res.backend == "local-terraform"
    assert res.status == "terraform-unavailable"
    assert res.build_ok is False
    assert res.stderr_tail == "terraform CLI not found on PATH"


async def test_local_terraform_validated_mapping(monkeypatch) -> None:
    v = LocalTerraformVerifier()
    monkeypatch.setattr(v, "_terraform_available", lambda: True)
    monkeypatch.setattr(v, "_tf_init", lambda *a, **k: _ok_proc("initialized"))
    monkeypatch.setattr(v, "_tf_validate", lambda *a, **k: _ok_proc("Success!"))
    res = await v.verify(GOOD_TERRAFORM, _tf_spec())
    assert res.build_ok is True
    assert res.smoke_ok is True
    assert res.status == "validated"
    assert res.exit_code == 0


async def test_local_terraform_validate_failed_mapping(monkeypatch) -> None:
    v = LocalTerraformVerifier()
    monkeypatch.setattr(v, "_terraform_available", lambda: True)
    monkeypatch.setattr(v, "_tf_init", lambda *a, **k: _ok_proc("initialized"))
    monkeypatch.setattr(v, "_tf_validate", lambda *a, **k: _ok_proc("", "Error: bad", rc=1))
    res = await v.verify(GOOD_TERRAFORM, _tf_spec())
    assert res.build_ok is False
    assert res.smoke_ok is False
    assert res.status == "validate-failed"
    assert res.exit_code == 1
    assert "Error: bad" in res.stderr_tail


async def test_local_terraform_init_failed_mapping(monkeypatch) -> None:
    v = LocalTerraformVerifier()
    monkeypatch.setattr(v, "_terraform_available", lambda: True)
    monkeypatch.setattr(v, "_tf_init", lambda *a, **k: _ok_proc("", "no providers", rc=1))
    # validate must never run if init failed.
    monkeypatch.setattr(
        v, "_tf_validate", lambda *a, **k: pytest.fail("validate ran despite init failure")
    )
    res = await v.verify(GOOD_TERRAFORM, _tf_spec())
    assert res.build_ok is False
    assert res.smoke_ok is False
    assert res.status == "init-failed"
    assert res.exit_code == 1


async def test_local_terraform_init_timeout_mapping(monkeypatch) -> None:
    v = LocalTerraformVerifier()
    monkeypatch.setattr(v, "_terraform_available", lambda: True)

    def timeout(*a, **k):
        raise subprocess.TimeoutExpired(cmd="terraform init", timeout=1)

    monkeypatch.setattr(v, "_tf_init", timeout)
    res = await v.verify(GOOD_TERRAFORM, _tf_spec())
    assert res.status == "terraform-timeout"
    assert res.hack_flags.timed_out is True
    assert res.hack_flags.resource_exhaustion is True


async def test_local_terraform_validate_timeout_mapping(monkeypatch) -> None:
    v = LocalTerraformVerifier()
    monkeypatch.setattr(v, "_terraform_available", lambda: True)
    monkeypatch.setattr(v, "_tf_init", lambda *a, **k: _ok_proc("initialized"))

    def timeout(*a, **k):
        raise subprocess.TimeoutExpired(cmd="terraform validate", timeout=1)

    monkeypatch.setattr(v, "_tf_validate", timeout)
    res = await v.verify(GOOD_TERRAFORM, _tf_spec())
    assert res.status == "terraform-timeout"
    assert res.hack_flags.timed_out is True
    assert res.hack_flags.resource_exhaustion is True


async def test_local_terraform_context_files_traversal_guard(monkeypatch) -> None:
    v = LocalTerraformVerifier()
    monkeypatch.setattr(v, "_terraform_available", lambda: True)
    monkeypatch.setattr(
        v, "_tf_init", lambda *a, **k: pytest.fail("init reached despite bad path")
    )
    spec = _tf_spec(context_files={"../evil.tf": "x"})
    with pytest.raises(ValueError, match="escapes build context"):
        await v.verify(GOOD_TERRAFORM, spec)


# --- LocalK8sVerifier: mapping unit-tested with faked hook -----------------
async def test_local_k8s_unavailable(monkeypatch) -> None:
    v = LocalK8sVerifier()
    monkeypatch.setattr(v, "_kubeconform_available", lambda: False)
    res = await v.verify(GOOD_K8S, _k8s_spec())
    assert res.backend == "local-k8s"
    assert res.status == "kubeconform-unavailable"
    assert res.build_ok is False
    assert res.stderr_tail == "kubeconform CLI not found on PATH"


async def test_local_k8s_validated_mapping(monkeypatch) -> None:
    v = LocalK8sVerifier()
    monkeypatch.setattr(v, "_kubeconform_available", lambda: True)
    monkeypatch.setattr(v, "_kubeconform", lambda *a, **k: _ok_proc("Valid: 1"))
    res = await v.verify(GOOD_K8S, _k8s_spec())
    assert res.build_ok is True
    assert res.smoke_ok is True
    assert res.status == "validated"
    assert res.exit_code == 0


async def test_local_k8s_invalid_mapping(monkeypatch) -> None:
    v = LocalK8sVerifier()
    monkeypatch.setattr(v, "_kubeconform_available", lambda: True)
    monkeypatch.setattr(v, "_kubeconform", lambda *a, **k: _ok_proc("", "invalid", rc=1))
    res = await v.verify(GOOD_K8S, _k8s_spec())
    assert res.build_ok is False
    assert res.smoke_ok is False
    assert res.status == "invalid"
    assert res.exit_code == 1


async def test_local_k8s_timeout_mapping(monkeypatch) -> None:
    v = LocalK8sVerifier()
    monkeypatch.setattr(v, "_kubeconform_available", lambda: True)

    def timeout(*a, **k):
        raise subprocess.TimeoutExpired(cmd="kubeconform", timeout=1)

    monkeypatch.setattr(v, "_kubeconform", timeout)
    res = await v.verify(GOOD_K8S, _k8s_spec())
    assert res.status == "k8s-timeout"
    assert res.hack_flags.timed_out is True
    assert res.hack_flags.resource_exhaustion is True


async def test_local_k8s_non_strict_omits_flag(monkeypatch) -> None:
    seen = {}

    def fake_run(cmd, *a, **k):
        seen["cmd"] = cmd
        return _ok_proc("Valid")

    v = LocalK8sVerifier(strict=False)
    monkeypatch.setattr(v, "_kubeconform_available", lambda: True)
    monkeypatch.setattr(subprocess, "run", fake_run)
    res = await v.verify(GOOD_K8S, _k8s_spec())
    assert res.status == "validated"
    assert "-strict" not in seen["cmd"]


# --- LocalGenuineVerifier: kind-aware dispatch -----------------------------
async def test_local_genuine_dispatches_by_kind(monkeypatch) -> None:
    v = LocalGenuineVerifier()
    # Force every constructed sub-verifier to report its kind via a sentinel
    # VerifyResult, so we can assert routing without touching any real CLI.
    docker_sub = v._sub_for(ArtifactKind.DOCKERFILE)
    tf_sub = v._sub_for(ArtifactKind.TERRAFORM)
    assert isinstance(docker_sub, LocalDockerVerifier)
    assert isinstance(tf_sub, LocalTerraformVerifier)

    async def fake_docker(artifact, spec):
        return VerifyResult(backend="local-docker", status="routed-docker", build_ok=True)

    async def fake_tf(artifact, spec):
        return VerifyResult(backend="local-terraform", status="routed-tf", build_ok=True)

    monkeypatch.setattr(docker_sub, "verify", fake_docker)
    monkeypatch.setattr(tf_sub, "verify", fake_tf)

    df_res = await v.verify(GOOD_DOCKERFILE, _spec(port=8000))
    tf_res = await v.verify(GOOD_TERRAFORM, _tf_spec())
    assert df_res.status == "routed-docker"
    assert df_res.backend == "local-docker"
    assert tf_res.status == "routed-tf"
    assert tf_res.backend == "local-terraform"


async def test_local_genuine_python_matches_local_py() -> None:
    # PYTHON dispatches to LocalPyVerifier: a trivial clean-exit program yields
    # build_ok & smoke_ok with no real terraform/docker involved.
    v = LocalGenuineVerifier()
    res = await v.verify("import sys; sys.exit(0)", _py_spec())
    assert res.backend == "local-py"
    assert res.build_ok is True
    assert res.smoke_ok is True
    assert res.hack_flags.any() is False


async def test_local_genuine_caches_sub_verifiers() -> None:
    v = LocalGenuineVerifier()
    first = v._sub_for(ArtifactKind.K8S)
    second = v._sub_for(ArtifactKind.K8S)
    assert first is second
    assert isinstance(first, LocalK8sVerifier)


def test_local_genuine_unknown_kind_falls_back_to_static() -> None:
    v = LocalGenuineVerifier()
    sub = v._sub_for(ArtifactKind.CI_YAML)
    assert isinstance(sub, StaticVerifier)
