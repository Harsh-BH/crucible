"""Gold (reference) Dockerfile generation for ``infra_synth``.

vf-free, **stdlib-only** (plus ``verifier.types`` indirectly via ``tasks``-shaped
``info`` dicts — but this module reads the dict directly and does not import
``verifier``).

:func:`gold_dockerfile` renders a *correct* reference Dockerfile for a task's
``info``. It is used for:

- eval reference artifacts / few-shot material, and
- a sanity check (see ``tests/test_gold.py``) that a gold artifact passes its own
  spec's static checks (pinned ``FROM``, ``WORKDIR``, dependency install,
  ``COPY``, ``EXPOSE <port>``, ``CMD`` launching the server).

The generated Dockerfile intentionally satisfies the ``smoke.must_contain``
substrings produced by :func:`infra_synth.tasks._info_for`.
"""
from __future__ import annotations

from typing import Any

# Canonical pinned base image per language (first entry of tasks.LANGUAGES tags).
_BASE_IMAGE: dict[str, str] = {
    "python": "python:3.11-slim",
}

# OS packages required by certain dependency profiles (installed via apt).
_APT_FOR_DEP: dict[str, tuple[str, ...]] = {
    "postgres": ("libpq-dev", "gcc"),
    "redis": (),
    "none": (),
}


def _server_cmd(server: str, app_target: str, port: int) -> str:
    """Return a JSON-array CMD line launching ``server`` on ``port``."""
    if server == "uvicorn":
        parts = [
            "uvicorn",
            app_target,
            "--host",
            "0.0.0.0",
            "--port",
            str(port),
        ]
    elif server == "gunicorn":
        parts = [
            "gunicorn",
            "--bind",
            f"0.0.0.0:{port}",
            app_target,
        ]
    else:  # pragma: no cover - future servers
        parts = [server, app_target, "--port", str(port)]
    inner = ", ".join(f'"{p}"' for p in parts)
    return f"CMD [{inner}]"


def gold_dockerfile(info: dict[str, Any]) -> str:
    """Render a correct reference Dockerfile for ``info``.

    Expects the ``info`` shape produced by :func:`infra_synth.tasks._info_for`
    (keys: ``language``, ``server``, ``app_target``, ``dependency``,
    ``dep_packages``, and ``smoke`` with ``port`` / ``base_image_prefix``).
    """
    language = info.get("language", "python")
    smoke = info.get("smoke", {})
    port = int(smoke.get("port", 8000))
    server = info.get("server", "uvicorn")
    app_target = info.get("app_target", "app.main:app")
    dependency = info.get("dependency", "none")

    base_image = _BASE_IMAGE.get(language, "python:3.11-slim")
    apt_pkgs = _APT_FOR_DEP.get(dependency, ())

    lines: list[str] = []
    lines.append(f"FROM {base_image}")
    lines.append("")
    lines.append("ENV PYTHONUNBUFFERED=1 \\")
    lines.append("    PIP_NO_CACHE_DIR=1 \\")
    lines.append("    PIP_DISABLE_PIP_VERSION_CHECK=1")
    lines.append("")
    lines.append("WORKDIR /app")
    lines.append("")

    if apt_pkgs:
        pkgs = " ".join(apt_pkgs)
        lines.append("RUN apt-get update \\")
        lines.append(f"    && apt-get install -y --no-install-recommends {pkgs} \\")
        lines.append("    && rm -rf /var/lib/apt/lists/*")
        lines.append("")

    # Install python deps first for better layer caching.
    lines.append("COPY requirements.txt ./")
    lines.append("RUN pip install --no-cache-dir -r requirements.txt")
    lines.append("")
    lines.append("COPY ./app ./app")
    lines.append("")

    # Drop privileges (non-root, production-ready) before exposing/serving.
    lines.append("RUN useradd --create-home --uid 10001 appuser \\")
    lines.append("    && chown -R appuser:appuser /app")
    lines.append("USER appuser")
    lines.append("")

    lines.append(f"EXPOSE {port}")
    lines.append("")
    lines.append(_server_cmd(server, app_target, port))

    return "\n".join(lines) + "\n"


__all__ = ["gold_dockerfile"]
