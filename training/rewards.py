"""Reward functions for Crucible GRPO training (torch-free).

These are **TRL-compatible** reward callables. TRL's ``GRPOTrainer`` calls a
reward function *synchronously* once per generation step with the signature::

    def reward_fn(prompts, completions, **dataset_columns) -> list[float]

where ``completions`` is a list (one decoded string, or a chat message list, per
sampled rollout) and each extra dataset column (e.g. ``answer``, ``info``) is
passed as a list keyword argument aligned with ``completions``.

Two reward families live here:

- :func:`gsm8k_reward` / :func:`format_reward` — pure, stdlib-only math/format
  rewards for the **M1** reproduction.
- :func:`make_infra_synth_reward` — a **factory** returning a sync reward fn that
  bridges into the project's async verifier layer (``verifier.get_verifier`` +
  ``verifier.shape_reward``). This is the single bridge used for BOTH
  ``infra_synth`` training (``verifier_backend="static"`` / ``"local-docker"``)
  and the **M2 Sentinel path** (``verifier_backend="sentinel"``).

Import discipline: ``verifier`` / ``infra_synth`` are imported **lazily** inside
the factory's closure, so this module imports with only stdlib present and the
pure rewards unit-test without the verifier package, ``datasets`` or ``torch``.
"""
from __future__ import annotations

import asyncio
import re
from typing import TYPE_CHECKING, Any, Callable

from .data import extract_boxed_or_final_number

if TYPE_CHECKING:  # type-only; never imported at runtime
    from verifier.types import VerifySpec, Verifier, VerifyResult

__all__ = [
    "gsm8k_reward",
    "format_reward",
    "make_infra_synth_reward",
    "normalize_number",
    "completion_to_text",
]

# A boxed final answer (the format we ask GSM8K completions to end with).
_BOXED_RE = re.compile(r"\\boxed\{[^{}]*\}")


def _import_infra_env() -> tuple[Any, Any]:
    """Lazily import ``(tasks_module, extract_dockerfile)`` from the env package.

    The ``infra_synth`` env is a proper nested package
    (``pip install -e ./environments/infra_synth`` or ``prime env install``), so
    its modules import cleanly as ``infra_synth.tasks`` / ``infra_synth.parser``.
    """
    from infra_synth import tasks as infra_tasks
    from infra_synth.parser import extract_dockerfile

    return infra_tasks, extract_dockerfile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def completion_to_text(completion: Any) -> str:
    """Coerce a TRL completion into a plain string.

    TRL hands back either a decoded string (non-conversational) or a list of
    chat messages ``[{"role": "assistant", "content": ...}, ...]``
    (conversational). We concatenate the assistant content in the latter case.
    """
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        parts: list[str] = []
        for msg in completion:
            if isinstance(msg, dict):
                content = msg.get("content", "")
                if isinstance(content, str):
                    parts.append(content)
                elif isinstance(content, list):
                    # multimodal content blocks -> keep text parts
                    for block in content:
                        if isinstance(block, dict) and isinstance(
                            block.get("text"), str
                        ):
                            parts.append(block["text"])
            elif isinstance(msg, str):
                parts.append(msg)
        return "\n".join(parts)
    return str(completion)


def normalize_number(s: str | None) -> str | None:
    """Normalise a numeric answer string for robust equality comparison.

    Strips ``$``/``,``/whitespace, drops a trailing ``.0`` / trailing zeros on a
    decimal, and canonicalises ``-0`` -> ``0``. Returns ``None`` for non-numeric
    input (the caller then falls back to a string compare).
    """
    if s is None:
        return None
    t = s.strip().replace("$", "").replace(",", "").replace("%", "")
    if not t:
        return None
    try:
        val = float(t)
    except ValueError:
        return None
    if val == int(val):
        return str(int(val))
    # Trim trailing zeros on the fractional part for a canonical form.
    return repr(val).rstrip("0").rstrip(".")


# ---------------------------------------------------------------------------
# GSM8K rewards
# ---------------------------------------------------------------------------
def gsm8k_reward(
    prompts: list[Any] | None,
    completions: list[Any],
    answer: list[str] | None = None,
    **kwargs: Any,
) -> list[float]:
    """Correctness reward for GSM8K: ``1.0`` if the extracted answer matches gold.

    For each completion, extract the final number (boxed / ``####`` / last
    number) and compare to the gold ``answer`` (numeric-normalised, with a string
    fallback for symbolic golds). Returns a list of ``1.0``/``0.0`` floats.

    Signature matches TRL: ``answer`` arrives as a per-sample list (the dataset's
    ``answer`` column). Tolerant if ``answer`` is missing (returns all zeros).
    """
    n = len(completions)
    golds: list[str] = list(answer) if answer is not None else [""] * n
    rewards: list[float] = []
    for i in range(n):
        text = completion_to_text(completions[i])
        gold = golds[i] if i < len(golds) else ""
        pred = extract_boxed_or_final_number(text)
        rewards.append(_match(pred, gold))
    return rewards


def _match(pred: str | None, gold: str) -> float:
    if pred is None:
        return 0.0
    gold_extracted = extract_boxed_or_final_number(gold) or gold
    pn, gn = normalize_number(pred), normalize_number(gold_extracted)
    if pn is not None and gn is not None:
        return 1.0 if pn == gn else 0.0
    # Symbolic / non-numeric fallback: case-insensitive trimmed string compare.
    return 1.0 if pred.strip().lower() == str(gold_extracted).strip().lower() else 0.0


def format_reward(
    prompts: list[Any] | None,
    completions: list[Any],
    **kwargs: Any,
) -> list[float]:
    """Small shaping reward (``0.0``/``1.0``) for emitting a ``\\boxed{...}``.

    Encourages the model to produce a parseable final answer. Intended as a tiny
    auxiliary reward (TRL averages multiple ``reward_funcs``), never the primary
    signal. Independent of the gold answer.
    """
    rewards: list[float] = []
    for comp in completions:
        text = completion_to_text(comp)
        rewards.append(1.0 if _BOXED_RE.search(text) else 0.0)
    return rewards


# ---------------------------------------------------------------------------
# infra_synth / Sentinel bridge
# ---------------------------------------------------------------------------
def make_infra_synth_reward(
    verifier_backend: str = "static",
    *,
    build_weight: float = 0.3,
    smoke_weight: float = 0.7,
    hack_penalty: float = 0.0,
    binary: bool = False,
    sentinel_base_url: str | None = None,
    verifier: "Verifier | None" = None,
    time_limit_ms: int | None = None,
    mem_mb: int | None = None,
) -> Callable[..., list[float]]:
    """Build a TRL-compatible SYNC reward fn that verifies each Dockerfile.

    The returned ``reward_fn(prompts, completions, info=..., **kw)`` does, per
    generation step:

    1. parse the Dockerfile from each completion
       (:func:`infra_synth.parser.extract_dockerfile`),
    2. build a :class:`~verifier.types.VerifySpec` from the row's ``info`` dict
       (:func:`infra_synth.tasks.build_verify_spec`),
    3. run the **async** verifier (``verifier.get_verifier(backend).verify``)
       over the whole batch **concurrently** (``asyncio.gather``) inside a single
       ``asyncio.run`` call, and
    4. shape each :class:`~verifier.types.VerifyResult` into a scalar via
       :func:`verifier.reward.shape_reward`.

    Backends (``verifier_backend``): ``"static"`` (in-process, default — the
    fast/safe path for code-path checks and CI), ``"local-docker"`` (genuine
    build + smoke), or ``"sentinel"`` (the hardened sandbox — **M2**; needs a
    running Sentinel at ``sentinel_base_url``). A ``verifier`` instance may be
    **injected directly** (used by tests with a stub — no live Sentinel/docker
    required); when given it overrides ``verifier_backend``.

    ``verifier`` / ``infra_synth`` are imported lazily inside the closure so this
    factory's *module* stays torch/verifier-free at import time.
    """

    def _resolve_verifier() -> "Verifier":
        if verifier is not None:
            return verifier
        from verifier import get_verifier  # lazy

        kwargs: dict[str, Any] = {}
        if time_limit_ms is not None:
            kwargs["time_limit_ms"] = time_limit_ms
        if mem_mb is not None:
            kwargs["mem_mb"] = mem_mb
        if verifier_backend == "sentinel" and sentinel_base_url is not None:
            kwargs["base_url"] = sentinel_base_url
        return get_verifier(verifier_backend, **kwargs)

    def reward_fn(
        prompts: list[Any] | None,
        completions: list[Any],
        info: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> list[float]:
        # Lazy imports keep the module torch/verifier-free at import time.
        infra_tasks, extract_dockerfile = _import_infra_env()  # lazy
        from verifier import shape_reward  # lazy

        n = len(completions)
        infos: list[dict[str, Any]] = list(info) if info is not None else [{}] * n

        # Parse artifacts + build specs up front (pure / stdlib).
        artifacts: list[str] = [completion_to_text(completions[i]) for i in range(n)]
        dockerfiles: list[str] = [extract_dockerfile(a) for a in artifacts]
        specs: list["VerifySpec"] = []
        for i in range(n):
            row_info = infos[i] if i < len(infos) else {}
            specs.append(infra_tasks.build_verify_spec(row_info))

        backend = _resolve_verifier()

        async def _run_all() -> list["VerifyResult"]:
            # Verify the whole batch concurrently on one event loop.
            return await asyncio.gather(
                *(backend.verify(dockerfiles[i], specs[i]) for i in range(n))
            )

        results = _run_one_loop(_run_all())

        return [
            shape_reward(
                r,
                build_weight=build_weight,
                smoke_weight=smoke_weight,
                hack_penalty=hack_penalty,
                binary=binary,
            )
            for r in results
        ]

    # A stable, introspectable name (TRL uses ``__name__`` for metric logging).
    reward_fn.__name__ = f"infra_synth_reward_{verifier_backend}"
    return reward_fn


def _run_one_loop(coro: Any) -> Any:
    """Run ``coro`` to completion on a fresh event loop, robust to context.

    Uses ``asyncio.run`` when no loop is running (the normal TRL training
    thread). If a loop is already running in this thread (rare — e.g. called
    from inside async test infra), falls back to a dedicated new loop in a worker
    thread so we never raise ``RuntimeError: asyncio.run() cannot be called from
    a running event loop``.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    # A loop is already running: execute the coroutine on its own loop in a
    # separate thread and block for the result.
    import concurrent.futures

    def _target() -> Any:
        return asyncio.run(coro)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(_target).result()
