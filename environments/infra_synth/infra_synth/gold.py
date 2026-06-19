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

# Pinned image for each dependency service in a compose document (no floating
# ``latest``). ``none`` maps to no separate service (see :func:`gold_compose`).
_DEP_SERVICE_IMAGE: dict[str, str] = {
    "postgres": "postgres:16",
    "redis": "redis:7",
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


def gold_compose(info: dict[str, Any]) -> str:
    """Render a correct reference ``docker-compose.yml`` for ``info``.

    Expects the compose ``info`` shape produced by
    :func:`infra_synth.tasks._compose_info_for` (keys: ``dependency`` and
    ``smoke`` with ``port`` / ``health_path`` / ``dependency_service``). The
    document intentionally satisfies :func:`verifier.smoke.checks.check_compose`:

    - a top-level ``services:`` key with a real ``web`` service (``build: .``),
    - a ``ports:`` mapping publishing ``"<P>:<P>"``,
    - a ``healthcheck:`` block whose ``test`` curls ``http://localhost:<P><H>``,
    - and (when ``dependency != none``) a separate ``postgres``/``redis`` service
      plus a ``depends_on`` reference.

    The output contains every ``smoke['must_contain']`` substring
    (``services:`` / ``ports:`` / ``"<P>:<P>"`` / ``healthcheck:``) and a real
    ``build:``/``image:`` (so it is not flagged ``spec_gaming``).
    """
    smoke = info.get("smoke", {})
    port = int(smoke.get("port", 8000))
    health = smoke.get("health_path", "/health")
    dependency = info.get("dependency", "none")
    dep_service = smoke.get("dependency_service")
    if dep_service is None and dependency != "none":
        dep_service = dependency

    lines: list[str] = []
    lines.append("services:")
    lines.append("  web:")
    lines.append("    build: .")
    lines.append("    ports:")
    lines.append(f'      - "{port}:{port}"')
    lines.append("    healthcheck:")
    lines.append(
        f'      test: ["CMD", "curl", "-f", "http://localhost:{port}{health}"]'
    )
    lines.append("      interval: 10s")
    lines.append("      timeout: 3s")
    lines.append("      retries: 3")
    if dep_service:
        lines.append("    depends_on:")
        lines.append(f"      - {dep_service}")
    lines.append("    restart: unless-stopped")
    if dep_service:
        image = _DEP_SERVICE_IMAGE.get(dep_service, f"{dep_service}:latest")
        lines.append(f"  {dep_service}:")
        lines.append(f"    image: {image}")
        lines.append("    restart: unless-stopped")

    return "\n".join(lines) + "\n"


__all__ = ["gold_dockerfile", "gold_compose"]
