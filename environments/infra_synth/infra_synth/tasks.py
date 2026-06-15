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


def _gold_hint(combo: tuple[str, str, str, int, str]) -> str:
    """Short 'answer' hint stored per task — the canonical base image tag."""
    language = combo[0]
    return LANGUAGES[language]["base_images"][0]


def generate_tasks(
    n: int | None = None,
    seed: int = 0,
    split: str = "train",
) -> list[dict[str, Any]]:
    """Return a deterministic, seeded list of task dicts for ``split``.

    Each dict has the dataset shape::

        {"question": <NL spec>, "answer": <gold hint>, "info": {...}, "task": "infra_synth"}

    Determinism: for a given ``(seed, split)`` the selection and order are stable.
    ``train`` and ``test`` draw from disjoint combination pools (see
    :func:`_split_combos`), so the splits are contamination-resistant.

    ``n`` caps the number of tasks (``None`` -> use the whole split pool). If
    ``n`` exceeds the pool size we return the whole pool (no duplication).
    """
    pool = _split_combos(split)
    # Explicit, reproducible RNG keyed by seed+split (string seed avoids the
    # per-process hash salt that affects ``hash(tuple)``).
    rng = random.Random(f"{seed}:{split}")
    rng.shuffle(pool)

    if n is not None:
        pool = pool[: max(0, n)]

    tasks: list[dict[str, Any]] = []
    for i, combo in enumerate(pool):
        spec_id = f"{split}-{seed}-{i:04d}-{combo[0]}-{combo[1]}-{combo[2]}-{combo[3]}"
        tasks.append(
            {
                "question": _render_question(combo),
                "answer": _gold_hint(combo),
                "info": _info_for(combo, spec_id),
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
      ``base_image_prefix`` from it).
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

    return VerifySpec(
        spec_id=str(spec_id),
        kind=kind,
        smoke=dict(info.get("smoke", {})),
        limits=limits,
    )


__all__ = [
    "SYSTEM_PROMPT",
    "TASK_NAME",
    "LANGUAGES",
    "FRAMEWORKS",
    "DEPENDENCIES",
    "PORTS",
    "HEALTH_PATHS",
    "generate_tasks",
    "build_verify_spec",
]
