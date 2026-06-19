"""Task generation for the ``infra_synth`` environment.

vf-free, **stdlib-only** (plus the committed ``verifier.types``) so it unit-tests
without ``verifiers`` / ``datasets``.

What this module owns
---------------------
- A small **parameter grid** (language x framework x dependencies x port x
  health path) describing infrastructure specs. We start with Python; the grid
  is structured so node/go can be added later by extending ``LANGUAGES`` and the
  per-language launch metadata.
- :func:`generate_tasks` — a deterministic, seeded list of task dicts shaped for
  a ``datasets.Dataset`` (``question`` / ``answer`` / ``info`` / ``task``).
- A **contamination-resistant** train/test split: ``train`` and ``test`` draw
  from **disjoint** parameter combinations (fresh combos are held out for test),
  so a policy cannot memorise an eval spec it saw during training.
- :func:`build_verify_spec` — converts a task's ``info`` dict into a
  :class:`verifier.types.VerifySpec` (the bridge to the verifier backends).
- :data:`SYSTEM_PROMPT` — instructs the model to emit ONLY a Dockerfile.

The ``info`` dict is the single source of truth carried through the pipeline;
it contains everything needed to (a) render the NL prompt, (b) build a gold
reference (see ``gold.py``), and (c) construct a ``VerifySpec``.
"""
from __future__ import annotations

import random
from typing import Any

from verifier.types import ArtifactKind, ResourceLimits, VerifySpec

from .scaffold import app_scaffold, compose_scaffold

# ---------------------------------------------------------------------------
# Parameter grid
# ---------------------------------------------------------------------------
# Per-language metadata. Only ``python`` is exercised today; the shape leaves
# room for node/go (each would supply its own base image, install command, and
# server launch template). Keeping this as data keeps gold.py and the prompt
# rendering free of language-specific ``if`` ladders.
LANGUAGES: dict[str, dict[str, Any]] = {
    "python": {
        "display": "Python",
        # Pinned base-image tags (we never use the floating ``latest``).
        "base_images": ("python:3.11-slim", "python:3.12-slim"),
        "base_image_prefix": "python:",
    },
}

# Frameworks, with the pip package and the import/app object used to launch.
FRAMEWORKS: dict[str, dict[str, Any]] = {
    "fastapi": {
        "display": "FastAPI",
        "packages": ("fastapi", "uvicorn[standard]"),
        # uvicorn app target -> module:attr
        "app_target": "app.main:app",
        "server": "uvicorn",
    },
    "flask": {
        "display": "Flask",
        "packages": ("flask", "gunicorn"),
        "app_target": "app.main:app",
        "server": "gunicorn",
    },
}

# Dependency profiles -> extra system/python packages the spec must mention.
DEPENDENCIES: dict[str, dict[str, Any]] = {
    "none": {"display": "no external services", "packages": ()},
    "postgres": {"display": "a PostgreSQL database", "packages": ("psycopg2-binary",)},
    "redis": {"display": "a Redis cache", "packages": ("redis",)},
}

PORTS: tuple[int, ...] = (8000, 8080, 5000, 3000, 9000)
HEALTH_PATHS: tuple[str, ...] = ("/health", "/healthz", "/livez", "/ping", "/status")

# Dependency profile -> the compose service name (and image base) that satisfies
# ``check_compose``'s ``dependency_service`` gate. ``none`` has no extra service.
DEP_SERVICES: dict[str, str | None] = {
    "none": None,
    "postgres": "postgres",
    "redis": "redis",
}

TASK_NAME = "infra_synth"

SYSTEM_PROMPT = (
    "You are an expert platform engineer. Given an infrastructure specification, "
    "produce a single, correct Dockerfile that satisfies it.\n\n"
    "Output rules (strict):\n"
    "- Respond with ONLY one Dockerfile, wrapped in a single ```dockerfile fenced "
    "code block.\n"
    "- Do NOT include any prose, explanation, commentary, or <think> blocks before "
    "or after the block.\n"
    "- Use a pinned base image tag (never `latest`).\n"
    "- Install the required dependencies, COPY the application in, EXPOSE the "
    "requested port, and set a CMD that starts the server on that port.\n"
)

COMPOSE_SYSTEM_PROMPT = (
    "You are an expert platform engineer. Given an infrastructure specification, "
    "produce a single, correct docker-compose.yml that satisfies it.\n\n"
    "Output rules (strict):\n"
    "- Respond with ONLY one docker-compose.yml, wrapped in a single ```yaml "
    "fenced code block.\n"
    "- Do NOT include any prose, explanation, commentary, or <think> blocks before "
    "or after the block.\n"
    "- Define a top-level `services:` mapping with a web service that builds the "
    "app (`build: .`).\n"
    "- Publish the requested port (map `<port>:<port>` under a `ports:` block) and "
    "add a `healthcheck:` that probes the requested health path.\n"
    "- When an external service dependency is requested, declare it as a separate "
    "service (using a pinned image) and reference it via `depends_on`.\n"
)

CI_YAML_SYSTEM_PROMPT = (
    "You are an expert platform engineer. Given a CI specification, produce a "
    "single, correct GitHub Actions workflow that satisfies it.\n\n"
    "Output rules (strict):\n"
    "- Respond with ONLY one GitHub Actions workflow YAML, wrapped in a single "
    "```yaml fenced code block.\n"
    "- Do NOT include any prose, explanation, commentary, or <think> blocks before "
    "or after the block.\n"
    "- Define an `on:` trigger and a top-level `jobs:` mapping with a job that "
    "runs-on a runner and has a `steps:` list.\n"
    "- The steps must: check out the code (`actions/checkout`), set up the "
    "language (`actions/setup-...`), install dependencies, and run the test "
    "suite (pytest).\n"
)

TERRAFORM_SYSTEM_PROMPT = (
    "You are an expert platform engineer. Given an infrastructure specification, "
    "produce a single, correct Terraform (HCL) configuration that satisfies it.\n\n"
    "Output rules (strict):\n"
    "- Respond with ONLY one Terraform configuration, wrapped in a single ```hcl "
    "fenced code block.\n"
    "- Do NOT include any prose, explanation, commentary, or <think> blocks before "
    "or after the block.\n"
    "- Declare a `terraform { ... }` block and a `provider \"docker\" {}` block.\n"
    "- Define a `resource \"docker_image\" \"...\" { ... }` that builds the app and "
    "a `resource \"docker_container\" \"...\" { ... }` that runs it.\n"
    "- Publish the requested port (map it under the container's `ports { ... }` "
    "block as `internal`/`external`).\n"
)

K8S_SYSTEM_PROMPT = (
    "You are an expert platform engineer. Given an infrastructure specification, "
    "produce correct Kubernetes manifests that satisfy it.\n\n"
    "Output rules (strict):\n"
    "- Respond with ONLY the Kubernetes manifests, wrapped in a single ```yaml "
    "fenced code block (use `---` to separate documents).\n"
    "- Do NOT include any prose, explanation, commentary, or <think> blocks before "
    "or after the block.\n"
    "- Each document must declare `apiVersion:`, `kind:`, and `metadata:`.\n"
    "- Provide a `Deployment` (with a container that sets `containerPort:` to the "
    "requested port and a `livenessProbe` probing the health path) and a "
    "matching `Service`.\n"
)


# ---------------------------------------------------------------------------
# Grid enumeration + split
# ---------------------------------------------------------------------------
def _all_combos() -> list[tuple[str, str, str, int, str]]:
    """Cartesian product of the grid, in a stable, deterministic order.

    Returns tuples of ``(language, framework, dependency, port, health_path)``.
    """
    combos: list[tuple[str, str, str, int, str]] = []
    for language in LANGUAGES:
        for framework in FRAMEWORKS:
            for dependency in DEPENDENCIES:
                for port in PORTS:
                    for health in HEALTH_PATHS:
                        combos.append((language, framework, dependency, port, health))
    return combos


def _split_combos(split: str) -> list[tuple[str, str, str, int, str]]:
    """Partition the grid into DISJOINT train / test subsets.

    Contamination resistance: we hold out a fixed fraction of combinations for
    ``test`` using a deterministic hash on the combo itself (independent of the
    sampling ``seed``), so the train and test pools never overlap regardless of
    ``n`` or ``seed``. Roughly 25% of combos are reserved for test.
    """
    combos = _all_combos()
    # Deterministic, seed-independent partition. ``hash()`` is salted per-process,
    # so use a stable hash over the tuple's repr instead.
    held_out: list[tuple[str, str, str, int, str]] = []
    train: list[tuple[str, str, str, int, str]] = []
    for combo in combos:
        digest = _stable_hash(repr(combo))
        if digest % 4 == 0:  # ~25% reserved for test
            held_out.append(combo)
        else:
            train.append(combo)

    if split == "test":
        return held_out
    if split == "train":
        return train
    raise ValueError(f"unknown split {split!r} (expected 'train' or 'test')")


def _stable_hash(s: str) -> int:
    """A small, process-stable hash (FNV-1a, 32-bit). Stdlib-only."""
    h = 0x811C9DC5
    for ch in s.encode("utf-8"):
        h ^= ch
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def _render_question(combo: tuple[str, str, str, int, str]) -> str:
    """Render a natural-language infra spec for one parameter combo."""
    language, framework, dependency, port, health = combo
    lang_meta = LANGUAGES[language]
    fw_meta = FRAMEWORKS[framework]
    dep_meta = DEPENDENCIES[dependency]

    dep_clause = (
        "The service has no external service dependencies."
        if dependency == "none"
        else f"The service connects to {dep_meta['display']}."
    )
    return (
        f"Write a Dockerfile to containerize a {lang_meta['display']} web service "
        f"built with {fw_meta['display']}. "
        f"{dep_clause} "
        f"The application code lives under ./app (entrypoint module "
        f"`{fw_meta['app_target']}`) and requirements are in ./requirements.txt. "
        f"The server must listen on port {port} and expose a health-check endpoint "
        f"at `{health}` returning HTTP 200. "
        f"Use a pinned, slim base image and a non-root, production-ready setup."
    )


def _info_for(combo: tuple[str, str, str, int, str], spec_id: str) -> dict[str, Any]:
    """Build the ``info`` dict (the pipeline's source of truth) for a combo.

    Contains the grid params, plus the ``smoke`` block consumed by the verifier
    backends and a ``must_contain`` list of substrings a correct Dockerfile must
    include (a cheap static gate the ``static`` backend can enforce).
    """
    language, framework, dependency, port, health = combo
    lang_meta = LANGUAGES[language]
    fw_meta = FRAMEWORKS[framework]

    must_contain = [
        "FROM",
        "WORKDIR",
        "COPY",
        "CMD",
        f"EXPOSE {port}",
    ]
    return {
        "spec_id": spec_id,
        "kind": ArtifactKind.DOCKERFILE.value,  # "dockerfile"
        # Grid parameters (also used by gold.py to render a reference).
        "language": language,
        "framework": framework,
        "dependency": dependency,
        "packages": list(fw_meta["packages"]),
        "dep_packages": list(DEPENDENCIES[dependency]["packages"]),
        "app_target": fw_meta["app_target"],
        "server": fw_meta["server"],
        # Smoke-test parameters consumed when constructing a VerifySpec.
        "smoke": {
            "port": port,
            "health_path": health,
            "expect_status": 200,
            "must_contain": must_contain,
            "base_image_prefix": lang_meta["base_image_prefix"],
        },
    }


def _render_compose_question(combo: tuple[str, str, str, int, str]) -> str:
    """Render a natural-language Docker Compose spec for one parameter combo."""
    language, framework, dependency, port, health = combo
    lang_meta = LANGUAGES[language]
    fw_meta = FRAMEWORKS[framework]
    dep_service = DEP_SERVICES[dependency]

    dep_clause = (
        "The service has no external service dependencies."
        if dep_service is None
        else (
            f"It connects to a {dep_service} backing service, which must be "
            f"defined as a separate service in the compose file and referenced "
            f"via depends_on."
        )
    )
    return (
        f"Write a docker-compose.yml that runs a {fw_meta['display']} "
        f"{lang_meta['display']} web service built from the local Dockerfile "
        f"(`build: .`). "
        f"{dep_clause} "
        f"The web service must publish port {port} (map {port}:{port}) and "
        f"declare a health-check that probes `{health}` and expects HTTP 200. "
        f"Pin any service images to explicit tags (never `latest`)."
    )


def _compose_info_for(
    combo: tuple[str, str, str, int, str], spec_id: str
) -> dict[str, Any]:
    """Build the compose ``info`` dict (the pipeline's source of truth) for a combo.

    Mirrors :func:`_info_for` but targets ``ArtifactKind.COMPOSE`` and the
    ``check_compose`` smoke contract: ``port`` / ``health_path`` / ``expect_status``
    / ``must_contain`` (the case-sensitive substring gate) / ``dependency_service``
    (``"postgres"``/``"redis"``, or ``None`` when ``dependency == "none"``).
    """
    language, framework, dependency, port, health = combo
    fw_meta = FRAMEWORKS[framework]
    dep_service = DEP_SERVICES[dependency]

    must_contain = [
        "services:",
        "ports:",
        f"{port}:{port}",
        "healthcheck:",
    ]
    return {
        "spec_id": spec_id,
        "kind": ArtifactKind.COMPOSE.value,  # "compose"
        # Grid parameters (also used by gold.py to render a reference).
        "language": language,
        "framework": framework,
        "dependency": dependency,
        "packages": list(fw_meta["packages"]),
        "dep_packages": list(DEPENDENCIES[dependency]["packages"]),
        "app_target": fw_meta["app_target"],
        "server": fw_meta["server"],
        # Smoke-test parameters consumed when constructing a VerifySpec.
        "smoke": {
            "port": port,
            "health_path": health,
            "expect_status": 200,
            "must_contain": must_contain,
            "dependency_service": dep_service,
        },
    }


def _render_ci_yaml_question(combo: tuple[str, str, str, int, str]) -> str:
    """Render a natural-language GitHub Actions CI spec for one parameter combo.

    A CI workflow has no server to probe, so the port / health-path fields of the
    combo are unused here (the grid is shared with the other kinds for split
    parity); only language / framework drive the rendered flavor.
    """
    language, framework, _dependency, _port, _health = combo
    lang_meta = LANGUAGES[language]
    fw_meta = FRAMEWORKS[framework]
    return (
        f"Write a GitHub Actions workflow (.github/workflows/ci.yml) that runs CI "
        f"for a {fw_meta['display']} {lang_meta['display']} project: trigger on "
        f"push and pull_request, check out the code, set up "
        f"{lang_meta['display']}, install dependencies from requirements.txt, and "
        f"run the test suite (pytest)."
    )


def _ci_yaml_info_for(
    combo: tuple[str, str, str, int, str], spec_id: str
) -> dict[str, Any]:
    """Build the ci-yaml ``info`` dict (the pipeline's source of truth) for a combo.

    Mirrors :func:`_compose_info_for` but targets ``ArtifactKind.CI_YAML`` and the
    ``check_ci_yaml`` smoke contract: ``must_contain`` (the case-sensitive
    substring gate) and ``required_steps`` (the semantic steps detected via token
    heuristics). A CI workflow has no server to probe, so there is intentionally
    NO ``port`` / ``health_path`` in the smoke block.
    """
    language, framework, dependency, _port, _health = combo
    fw_meta = FRAMEWORKS[framework]

    return {
        "spec_id": spec_id,
        "kind": ArtifactKind.CI_YAML.value,  # "ci-yaml"
        # Grid parameters (also used by gold.py to render a reference flavor).
        "language": language,
        "framework": framework,
        "dependency": dependency,
        "packages": list(fw_meta["packages"]),
        "dep_packages": list(DEPENDENCIES[dependency]["packages"]),
        "app_target": fw_meta["app_target"],
        "server": fw_meta["server"],
        # Smoke-test parameters consumed when constructing a VerifySpec. The
        # locked ci-yaml contract: must_contain + required_steps, no port/health.
        "smoke": {
            "must_contain": [
                "on:",
                "jobs:",
                "runs-on:",
                "steps:",
                "actions/checkout",
            ],
            "required_steps": ["checkout", "setup", "install", "test"],
        },
    }


def _render_terraform_question(combo: tuple[str, str, str, int, str]) -> str:
    """Render a natural-language Terraform (HCL) spec for one parameter combo.

    Terraform here provisions a Docker image + container for the web service via
    the ``kreuzwerker/docker`` provider; the language / framework drive only the
    rendered flavor (the structure is what ``check_terraform`` grades). The combo
    health-path is unused (no probe in this stand-in); only ``port`` matters.
    """
    language, framework, _dependency, port, _health = combo
    lang_meta = LANGUAGES[language]
    fw_meta = FRAMEWORKS[framework]
    return (
        f"Write a Terraform (HCL) configuration that provisions a "
        f"{fw_meta['display']} {lang_meta['display']} web service as a Docker "
        f"container using the `kreuzwerker/docker` provider. Declare a "
        f"`terraform` block with the required provider, a `provider \"docker\"` "
        f"block, a `docker_image` resource that builds the app image from the "
        f"local context, and a `docker_container` resource that runs it and "
        f"publishes port {port} (map {port}:{port})."
    )


def _terraform_info_for(
    combo: tuple[str, str, str, int, str], spec_id: str
) -> dict[str, Any]:
    """Build the terraform ``info`` dict (the pipeline's source of truth).

    Mirrors :func:`_compose_info_for` but targets ``ArtifactKind.TERRAFORM`` and
    the ``check_terraform`` smoke contract: ``must_contain`` (the case-sensitive
    substring gate), ``resource_type`` (the smoke-gated resource type), and
    ``port`` (the published port). Terraform ignores framework-server specifics,
    so there is intentionally NO ``health_path`` in the smoke block.
    """
    language, framework, dependency, port, _health = combo
    fw_meta = FRAMEWORKS[framework]

    return {
        "spec_id": spec_id,
        "kind": ArtifactKind.TERRAFORM.value,  # "terraform"
        # Grid parameters (also used by gold.py to render a reference flavor).
        "language": language,
        "framework": framework,
        "dependency": dependency,
        "packages": list(fw_meta["packages"]),
        "dep_packages": list(DEPENDENCIES[dependency]["packages"]),
        "app_target": fw_meta["app_target"],
        "server": fw_meta["server"],
        # Smoke-test parameters consumed when constructing a VerifySpec. The
        # locked terraform contract: must_contain + resource_type + port.
        "smoke": {
            "must_contain": [
                "terraform",
                'provider "docker"',
                'resource "docker_image"',
                'resource "docker_container"',
            ],
            "resource_type": "docker_container",
            "port": port,
        },
    }


def _render_k8s_question(combo: tuple[str, str, str, int, str]) -> str:
    """Render a natural-language Kubernetes spec for one parameter combo.

    The language / framework drive the rendered flavor; the structure (a
    Deployment + Service, the container port, and a probe on the health path) is
    what ``check_k8s`` grades.
    """
    language, framework, _dependency, port, health = combo
    lang_meta = LANGUAGES[language]
    fw_meta = FRAMEWORKS[framework]
    return (
        f"Write Kubernetes manifests that deploy a {fw_meta['display']} "
        f"{lang_meta['display']} web service: a `Deployment` whose container "
        f"listens on port {port} (`containerPort: {port}`) and declares a "
        f"`livenessProbe` that probes `{health}`, plus a matching `Service` that "
        f"exposes port {port}. Each manifest must declare apiVersion, kind, and "
        f"metadata; separate documents with `---`."
    )


def _k8s_info_for(
    combo: tuple[str, str, str, int, str], spec_id: str
) -> dict[str, Any]:
    """Build the k8s ``info`` dict (the pipeline's source of truth).

    Mirrors :func:`_compose_info_for` but targets ``ArtifactKind.K8S`` and the
    ``check_k8s`` smoke contract: ``must_contain`` (the case-sensitive substring
    gate), ``port`` / ``health_path`` (the probe gate), and ``kinds_required``
    (the document kinds the manifest set must declare).
    """
    language, framework, dependency, port, health = combo
    fw_meta = FRAMEWORKS[framework]

    return {
        "spec_id": spec_id,
        "kind": ArtifactKind.K8S.value,  # "k8s"
        # Grid parameters (also used by gold.py to render a reference flavor).
        "language": language,
        "framework": framework,
        "dependency": dependency,
        "packages": list(fw_meta["packages"]),
        "dep_packages": list(DEPENDENCIES[dependency]["packages"]),
        "app_target": fw_meta["app_target"],
        "server": fw_meta["server"],
        # Smoke-test parameters consumed when constructing a VerifySpec. The
        # locked k8s contract: must_contain + port + health_path + kinds_required.
        "smoke": {
            "must_contain": [
                "apiVersion:",
                "kind: Deployment",
                "kind: Service",
                "containerPort:",
            ],
            "port": port,
            "health_path": health,
            "kinds_required": ["Deployment", "Service"],
        },
    }


def _gold_hint(combo: tuple[str, str, str, int, str]) -> str:
    """Short 'answer' hint stored per task — the canonical base image tag."""
    language = combo[0]
    return LANGUAGES[language]["base_images"][0]


def generate_tasks(
    n: int | None = None,
    seed: int = 0,
    split: str = "train",
    kind: str = "dockerfile",
) -> list[dict[str, Any]]:
    """Return a deterministic, seeded list of task dicts for ``split``.

    Each dict has the dataset shape::

        {"question": <NL spec>, "answer": <gold hint>, "info": {...}, "task": "infra_synth"}

    Determinism: for a given ``(seed, split, kind)`` the selection and order are
    stable. ``train`` and ``test`` draw from disjoint combination pools (see
    :func:`_split_combos`), so the splits are contamination-resistant; ``kind``
    does NOT change the split (it only changes the rendered artifact).

    ``kind`` selects the target artifact: ``"dockerfile"`` (the default — output
    is byte-for-byte identical to the historical no-``kind`` behavior),
    ``"compose"`` (a ``docker-compose.yml`` NL spec + the ``check_compose`` smoke
    contract), ``"ci-yaml"`` (a GitHub Actions workflow NL spec + the
    ``check_ci_yaml`` smoke contract), ``"terraform"`` (an HCL config NL spec +
    the ``check_terraform`` smoke contract), or ``"k8s"`` (Kubernetes manifests
    NL spec + the ``check_k8s`` smoke contract). The ``kind`` is folded into the
    ``spec_id`` for non-Dockerfile kinds (``compose-`` / ``ci-`` / ``tf-`` /
    ``k8s-``) so the per-kind task ids never collide.

    ``n`` caps the number of tasks (``None`` -> use the whole split pool). If
    ``n`` exceeds the pool size we return the whole pool (no duplication).
    """
    if kind not in (
        ArtifactKind.DOCKERFILE.value,
        ArtifactKind.COMPOSE.value,
        ArtifactKind.CI_YAML.value,
        ArtifactKind.TERRAFORM.value,
        ArtifactKind.K8S.value,
    ):
        raise ValueError(
            f"unknown kind {kind!r} (expected 'dockerfile', 'compose', 'ci-yaml', "
            f"'terraform', or 'k8s')"
        )

    pool = _split_combos(split)
    # Explicit, reproducible RNG keyed by seed+split (string seed avoids the
    # per-process hash salt that affects ``hash(tuple)``).
    rng = random.Random(f"{seed}:{split}")
    rng.shuffle(pool)

    if n is not None:
        pool = pool[: max(0, n)]

    is_compose = kind == ArtifactKind.COMPOSE.value
    is_ci_yaml = kind == ArtifactKind.CI_YAML.value
    is_terraform = kind == ArtifactKind.TERRAFORM.value
    is_k8s = kind == ArtifactKind.K8S.value
    tasks: list[dict[str, Any]] = []
    for i, combo in enumerate(pool):
        base_id = f"{split}-{seed}-{i:04d}-{combo[0]}-{combo[1]}-{combo[2]}-{combo[3]}"
        if is_compose:
            spec_id = f"compose-{base_id}"
            question = _render_compose_question(combo)
            info = _compose_info_for(combo, spec_id)
        elif is_ci_yaml:
            spec_id = f"ci-{base_id}"
            question = _render_ci_yaml_question(combo)
            info = _ci_yaml_info_for(combo, spec_id)
        elif is_terraform:
            spec_id = f"tf-{base_id}"
            question = _render_terraform_question(combo)
            info = _terraform_info_for(combo, spec_id)
        elif is_k8s:
            spec_id = f"k8s-{base_id}"
            question = _render_k8s_question(combo)
            info = _k8s_info_for(combo, spec_id)
        else:
            # Dockerfile path: unchanged spec_id / question / info (byte-for-byte).
            spec_id = base_id
            question = _render_question(combo)
            info = _info_for(combo, spec_id)
        tasks.append(
            {
                "question": question,
                "answer": _gold_hint(combo),
                "info": info,
                "task": TASK_NAME,
            }
        )
    return tasks


# ---------------------------------------------------------------------------
# info dict -> VerifySpec
# ---------------------------------------------------------------------------
def build_verify_spec(info: dict[str, Any]) -> VerifySpec:
    """Convert a task ``info`` dict into a :class:`VerifySpec`.

    - ``kind`` comes from ``info['kind']`` (defaults to ``dockerfile``).
    - The ``smoke`` block is passed through verbatim (the verifier backend reads
      ``port`` / ``health_path`` / ``expect_status`` / ``must_contain`` /
      ``base_image_prefix`` from it). For Dockerfile tasks we attach a
      ``context_files`` map (the app scaffold the Dockerfile ``COPY``s) so
      ``local-docker`` builds against a real app, not an empty context. For
      Compose tasks we attach the compose scaffold (the app scaffold PLUS a
      working ``Dockerfile``) so a genuine ``docker compose up`` can resolve the
      service's ``build: .``. A caller-supplied ``context_files`` is left
      untouched.
    - :class:`ResourceLimits` is built from optional ``info['limits']`` fields,
      falling back to the contract defaults.
    """
    kind_raw = info.get("kind", ArtifactKind.DOCKERFILE.value)
    kind = ArtifactKind(kind_raw) if not isinstance(kind_raw, ArtifactKind) else kind_raw

    limits_in = info.get("limits") or {}
    # ResourceLimits uses slots, so read defaults from a fresh instance (the
    # class attributes are member descriptors, not the default values).
    defaults = ResourceLimits()
    limits = ResourceLimits(
        wall_s=int(limits_in.get("wall_s", defaults.wall_s)),
        mem_mb=int(limits_in.get("mem_mb", defaults.mem_mb)),
        pids=int(limits_in.get("pids", defaults.pids)),
        cpus=float(limits_in.get("cpus", defaults.cpus)),
    )

    spec_id = info.get("spec_id")
    if not spec_id:
        raise ValueError("info dict missing required 'spec_id'")

    smoke = dict(info.get("smoke", {}))
    if kind == ArtifactKind.DOCKERFILE and "context_files" not in smoke:
        smoke["context_files"] = app_scaffold(info)
    if kind == ArtifactKind.COMPOSE and "context_files" not in smoke:
        smoke["context_files"] = compose_scaffold(info)

    return VerifySpec(
        spec_id=str(spec_id),
        kind=kind,
        smoke=smoke,
        limits=limits,
    )


__all__ = [
    "SYSTEM_PROMPT",
    "COMPOSE_SYSTEM_PROMPT",
    "CI_YAML_SYSTEM_PROMPT",
    "TERRAFORM_SYSTEM_PROMPT",
    "K8S_SYSTEM_PROMPT",
    "TASK_NAME",
    "LANGUAGES",
    "FRAMEWORKS",
    "DEPENDENCIES",
    "DEP_SERVICES",
    "PORTS",
    "HEALTH_PATHS",
    "generate_tasks",
    "build_verify_spec",
]
