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

# Per-language ``actions/setup-*`` step for a GitHub Actions workflow (the token
# ``check_ci_yaml`` keys the ``setup`` step off of -> ``actions/setup-``).
_CI_SETUP: dict[str, dict[str, str]] = {
    "python": {
        "action": "actions/setup-python@v5",
        "version_key": "python-version",
        "version": "3.11",
    },
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


def gold_ci_yaml(info: dict[str, Any]) -> str:
    """Render a correct reference GitHub Actions workflow for ``info``.

    Expects the ci-yaml ``info`` shape produced by
    :func:`infra_synth.tasks._ci_yaml_info_for` (keys: ``language`` /
    ``framework`` for flavor; the structure is what matters). The document
    intentionally satisfies :func:`verifier.smoke.checks.check_ci_yaml`:

    - a top-level ``on:`` trigger (push + pull_request),
    - a top-level ``jobs:`` mapping with >= 1 job that declares ``runs-on:``,
    - a ``steps:`` list with the four semantic steps the check detects via token
      heuristics: ``actions/checkout`` (checkout), ``actions/setup-<lang>``
      (setup), a ``run:`` mentioning ``install`` (install), and a ``run:``
      mentioning ``pytest`` (test).

    The output contains every ``smoke['must_contain']`` substring (``on:`` /
    ``jobs:`` / ``runs-on:`` / ``steps:`` / ``actions/checkout``) and a real job
    body (so it is not flagged ``spec_gaming``).
    """
    language = info.get("language", "python")
    setup = _CI_SETUP.get(language, _CI_SETUP["python"])

    lines: list[str] = []
    lines.append("name: ci")
    lines.append("on:")
    lines.append("  push:")
    lines.append('    branches: ["main"]')
    lines.append("  pull_request:")
    lines.append("jobs:")
    lines.append("  test:")
    lines.append("    runs-on: ubuntu-latest")
    lines.append("    steps:")
    lines.append("      - uses: actions/checkout@v4")
    lines.append(f"      - uses: {setup['action']}")
    lines.append("        with:")
    lines.append(f'          {setup["version_key"]}: "{setup["version"]}"')
    lines.append("      - run: pip install -r requirements.txt")
    lines.append("      - run: pytest")

    return "\n".join(lines) + "\n"


def gold_terraform(info: dict[str, Any]) -> str:
    """Render a correct reference Terraform (HCL) config for ``info``.

    Expects the terraform ``info`` shape produced by
    :func:`infra_synth.tasks._terraform_info_for` (the ``smoke`` block carries
    ``port``; the rest is flavor). The config intentionally satisfies
    :func:`verifier.smoke.checks.check_terraform`:

    - a ``terraform {`` block AND a ``provider "docker" {`` block,
    - a ``resource "docker_image" "web" {`` block and a
      ``resource "docker_container" "web" {`` block (the ``resource_type``
      smoke gate), and
    - the requested ``port`` mapped under the container's ``ports`` block.

    The output contains every ``smoke['must_contain']`` substring
    (``terraform`` / ``provider "docker"`` / ``resource "docker_image"`` /
    ``resource "docker_container"``) and real resource bodies (so it is not
    flagged ``spec_gaming``).
    """
    smoke = info.get("smoke", {})
    port = int(smoke.get("port", 8000))

    lines: list[str] = []
    lines.append("terraform {")
    lines.append("  required_providers {")
    lines.append("    docker = {")
    lines.append('      source = "kreuzwerker/docker"')
    lines.append("    }")
    lines.append("  }")
    lines.append("}")
    lines.append("")
    lines.append('provider "docker" {}')
    lines.append("")
    lines.append('resource "docker_image" "web" {')
    lines.append('  name = "web-service:latest"')
    lines.append("  build {")
    lines.append('    context = "."')
    lines.append("  }")
    lines.append("}")
    lines.append("")
    lines.append('resource "docker_container" "web" {')
    lines.append('  name  = "web-service"')
    lines.append("  image = docker_image.web.image_id")
    lines.append("  ports {")
    lines.append(f"    internal = {port}")
    lines.append(f"    external = {port}")
    lines.append("  }")
    lines.append("}")

    return "\n".join(lines) + "\n"


def gold_k8s(info: dict[str, Any]) -> str:
    """Render a correct reference Kubernetes manifest set for ``info``.

    Expects the k8s ``info`` shape produced by
    :func:`infra_synth.tasks._k8s_info_for` (the ``smoke`` block carries ``port``
    and ``health_path``). The manifests intentionally satisfy
    :func:`verifier.smoke.checks.check_k8s`:

    - two documents (a ``Deployment`` and a ``Service``), each declaring
      ``apiVersion:`` / ``kind:`` / ``metadata:``,
    - the requested ``port`` under ``containerPort:`` / ``port:`` /
      ``targetPort:``, and
    - a ``livenessProbe`` whose ``httpGet`` probes the requested ``health_path``.

    The output contains every ``smoke['must_contain']`` substring
    (``apiVersion:`` / ``kind: Deployment`` / ``kind: Service`` /
    ``containerPort:``) and real ``spec:`` bodies (so it is not flagged
    ``spec_gaming``).
    """
    smoke = info.get("smoke", {})
    port = int(smoke.get("port", 8000))
    health = smoke.get("health_path", "/health")

    lines: list[str] = []
    lines.append("apiVersion: apps/v1")
    lines.append("kind: Deployment")
    lines.append("metadata:")
    lines.append("  name: web")
    lines.append("spec:")
    lines.append("  replicas: 1")
    lines.append("  selector:")
    lines.append("    matchLabels:")
    lines.append("      app: web")
    lines.append("  template:")
    lines.append("    metadata:")
    lines.append("      labels:")
    lines.append("        app: web")
    lines.append("    spec:")
    lines.append("      containers:")
    lines.append("        - name: web")
    lines.append("          image: web-service:latest")
    lines.append("          ports:")
    lines.append(f"            - containerPort: {port}")
    lines.append("          livenessProbe:")
    lines.append("            httpGet:")
    lines.append(f"              path: {health}")
    lines.append(f"              port: {port}")
    lines.append("---")
    lines.append("apiVersion: v1")
    lines.append("kind: Service")
    lines.append("metadata:")
    lines.append("  name: web")
    lines.append("spec:")
    lines.append("  selector:")
    lines.append("    app: web")
    lines.append("  ports:")
    lines.append(f"    - port: {port}")
    lines.append(f"      targetPort: {port}")

    return "\n".join(lines) + "\n"


__all__ = [
    "gold_dockerfile",
    "gold_compose",
    "gold_ci_yaml",
    "gold_terraform",
    "gold_k8s",
]
