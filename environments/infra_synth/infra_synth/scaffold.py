"""App scaffold written into the Docker build context for genuine build+smoke.

vf-free, **stdlib-only**. :func:`app_scaffold` returns the files a realistic
``infra_synth`` Dockerfile expects to ``COPY`` — a ``requirements.txt`` and a
minimal-but-real web app exposing the task's health endpoint — so
``LocalDockerVerifier`` can build the image and the smoke probe hits a live
server (not an empty build context). See NS-2 in ``docs/ROADMAP.md``.

The app object is ``app.main:app`` (matching ``tasks.FRAMEWORKS[*]['app_target']``).
The app only serves the health endpoint; declared service dependencies
(``psycopg2-binary`` / ``redis``) go into ``requirements.txt`` so the build
exercises the Dockerfile's system-package setup, but the app does not connect to
them — the smoke test grades "builds and serves health", not live connectivity.
"""
from __future__ import annotations

from typing import Any

from .gold import gold_dockerfile


def _fastapi_main(health: str) -> str:
    return (
        "from fastapi import FastAPI\n\n"
        "app = FastAPI()\n\n\n"
        f'@app.get("{health}")\n'
        "def health() -> dict:\n"
        '    return {"status": "ok"}\n'
    )


def _flask_main(health: str) -> str:
    return (
        "from flask import Flask\n\n"
        "app = Flask(__name__)\n\n\n"
        f'@app.get("{health}")\n'
        "def health():\n"
        '    return {"status": "ok"}, 200\n'
    )


def app_scaffold(info: dict[str, Any]) -> dict[str, str]:
    """Return a ``{relative_path: content}`` build-context map for ``info``.

    Expects the ``info`` shape from :func:`infra_synth.tasks._info_for`
    (``framework``, ``packages``, ``dep_packages``, and ``smoke.health_path``).
    Files: ``requirements.txt`` (framework + dependency packages) and an ``app``
    package (``app/__init__.py`` + ``app/main.py`` defining ``app``).
    """
    framework = info.get("framework", "fastapi")
    smoke = info.get("smoke") or {}
    health = str(smoke.get("health_path", "/health"))
    if not health.startswith("/"):
        health = "/" + health

    reqs = list(info.get("packages") or []) + list(info.get("dep_packages") or [])
    main = _flask_main(health) if framework == "flask" else _fastapi_main(health)

    return {
        "requirements.txt": "\n".join(reqs) + "\n",
        "app/__init__.py": "",
        "app/main.py": main,
    }


def compose_scaffold(info: dict[str, Any]) -> dict[str, str]:
    """Return the full build context a genuine ``docker compose up --build`` needs.

    The model's Compose task emits a ``docker-compose.yml`` whose ``web`` service
    uses ``build: .`` -- so the context it builds against must contain a
    ``Dockerfile`` (plus the app the Dockerfile ``COPY``s). This is the Compose
    analog of :func:`app_scaffold` for plain Dockerfile tasks (NS-2): with this
    context written alongside the model's compose file, the verifier can run
    ``docker compose up`` and the smoke probe hits a live server (not an empty
    context where ``build: .`` has nothing to build).

    Returns a ``{relative_path: content}`` map combining :func:`app_scaffold`
    (``requirements.txt`` + ``app/__init__.py`` + ``app/main.py``) with a
    working ``Dockerfile`` -- reusing the already build+serve-verified reference
    from :func:`infra_synth.gold.gold_dockerfile`. Keys: ``Dockerfile``,
    ``requirements.txt``, ``app/__init__.py``, ``app/main.py``.
    """
    files = app_scaffold(info)
    files["Dockerfile"] = gold_dockerfile(info)
    return files


__all__ = ["app_scaffold", "compose_scaffold"]
