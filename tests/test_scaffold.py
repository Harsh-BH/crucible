"""NS-2: app scaffold + build-context wiring for genuine local-docker build.

Covers scaffold.app_scaffold (framework branch + health path + deps),
tasks.build_verify_spec injecting smoke["context_files"], and
LocalDockerVerifier writing those files into the build context (incl. the
path-traversal guard). All vf-free and docker-free.
"""
from __future__ import annotations

import os
import subprocess

import pytest
from infra_synth import gold, scaffold, tasks

from verifier.backends import LocalDockerVerifier
from verifier.types import ArtifactKind, ResourceLimits, VerifySpec


def _info(framework: str = "fastapi", dep: str = "postgres") -> dict:
    """A task info dict for one combo (mirrors tasks._info_for output).

    Searches both splits so the test never depends on which split a combo
    lands in (each combo is in exactly one of train/test).
    """
    for split in ("train", "test"):
        for t in tasks.generate_tasks(seed=0, split=split):
            i = t["info"]
            if i["framework"] == framework and i["dependency"] == dep:
                return i
    raise AssertionError(f"no task for {framework}/{dep}")


def _compose_info(framework: str = "fastapi", dep: str = "postgres") -> dict:
    """A compose-kind task info dict for one combo (mirrors _compose_info_for)."""
    for split in ("train", "test"):
        for t in tasks.generate_tasks(seed=0, split=split, kind="compose"):
            i = t["info"]
            if i["framework"] == framework and i["dependency"] == dep:
                return i
    raise AssertionError(f"no compose task for {framework}/{dep}")


def test_app_scaffold_fastapi() -> None:
    files = scaffold.app_scaffold(_info("fastapi", "postgres"))
    assert set(files) == {"requirements.txt", "app/__init__.py", "app/main.py"}
    assert "FastAPI" in files["app/main.py"]
    # framework + dependency packages both land in requirements.txt
    assert "fastapi" in files["requirements.txt"]
    assert "uvicorn[standard]" in files["requirements.txt"]
    assert "psycopg2-binary" in files["requirements.txt"]


def test_app_scaffold_flask_serves_requested_health_path() -> None:
    info = _info("flask", "none")
    health = info["smoke"]["health_path"]
    files = scaffold.app_scaffold(info)
    assert "Flask" in files["app/main.py"]
    assert f'@app.get("{health}")' in files["app/main.py"]


def test_compose_scaffold_includes_dockerfile_and_app() -> None:
    info = _compose_info("fastapi", "postgres")
    files = scaffold.compose_scaffold(info)
    # The compose context = app scaffold PLUS a working Dockerfile.
    assert set(files) == {
        "Dockerfile",
        "requirements.txt",
        "app/__init__.py",
        "app/main.py",
    }
    df = files["Dockerfile"]
    assert df and df.startswith("FROM ")
    # The Dockerfile is exactly the build+serve-verified gold reference.
    assert df == gold.gold_dockerfile(info)
    # The app scaffold files are carried through unchanged.
    assert "FastAPI" in files["app/main.py"]
    assert "fastapi" in files["requirements.txt"]
    assert "psycopg2-binary" in files["requirements.txt"]


def test_compose_scaffold_flask_dockerfile_matches_gold() -> None:
    info = _compose_info("flask", "none")
    files = scaffold.compose_scaffold(info)
    assert files["Dockerfile"] == gold.gold_dockerfile(info)
    assert "Flask" in files["app/main.py"]


def test_build_verify_spec_injects_context_files() -> None:
    info = _info("fastapi", "redis")
    spec = tasks.build_verify_spec(info)
    cf = spec.smoke["context_files"]
    assert "requirements.txt" in cf and "app/main.py" in cf
    # smoke params still present alongside the scaffold
    assert spec.smoke["health_path"] == info["smoke"]["health_path"]


def test_build_verify_spec_respects_caller_context_files() -> None:
    info = _info("fastapi", "none")
    info["smoke"] = dict(info["smoke"], context_files={"custom.txt": "hi"})
    spec = tasks.build_verify_spec(info)
    assert spec.smoke["context_files"] == {"custom.txt": "hi"}


def test_local_docker_writes_context_files(monkeypatch) -> None:
    """The build context contains the scaffold files (nested app/ too)."""
    captured: dict[str, str] = {}

    def fake_build(context_dir, tag, mem_mb, cpus, timeout_s):
        for root, _dirs, names in os.walk(context_dir):
            for n in names:
                rel = os.path.relpath(os.path.join(root, n), context_dir)
                with open(os.path.join(root, n), encoding="utf-8") as fh:
                    captured[rel.replace(os.sep, "/")] = fh.read()
        return subprocess.CompletedProcess([], 1, "", "stop here")  # fail fast

    v = LocalDockerVerifier()
    monkeypatch.setattr(v, "_docker_available", lambda: True)
    monkeypatch.setattr(v, "_docker_build", fake_build)
    monkeypatch.setattr(v, "_docker_rmi", lambda tag: None)

    spec = tasks.build_verify_spec(_info("fastapi", "postgres"))
    v._run_blocking("FROM python:3.11-slim\n", spec)

    assert captured["Dockerfile"].startswith("FROM python")
    assert "fastapi" in captured["requirements.txt"]
    assert "FastAPI" in captured["app/main.py"]
    assert "app/__init__.py" in captured


def test_local_docker_context_files_traversal_guard() -> None:
    spec = VerifySpec(
        spec_id="evil",
        kind=ArtifactKind.DOCKERFILE,
        smoke={"context_files": {"../escape.txt": "pwned"}},
        limits=ResourceLimits(),
    )
    with pytest.raises(ValueError, match="escapes build context"):
        LocalDockerVerifier()._run_blocking("FROM python:3.11-slim\n", spec)
