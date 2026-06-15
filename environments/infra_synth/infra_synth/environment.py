"""infra_synth — RLVR environment for infrastructure-as-code synthesis.

Single-turn `verifiers` Hub-spec environment. Given a natural-language infra
spec, the model emits a Dockerfile (in a ```dockerfile fenced block); the reward
comes from verifying that artifact (build + smoke test) through a pluggable
verifier backend (``verifier.Verifier``).

This module is the **thin** vf-wiring layer. The substance lives in vf-free,
stdlib-only helpers so they unit-test without ``verifiers`` / ``datasets``:

- :mod:`infra_synth.parser` — extract the Dockerfile from a completion.
- :mod:`infra_synth.tasks`  — generate tasks + build a ``VerifySpec``.
- :mod:`infra_synth.gold`   — render a reference Dockerfile.

The verifier integration (``get_verifier`` / ``shape_reward`` /
``result_to_metrics``) is provided by a parallel ``verifier`` package and is
imported **lazily inside functions** so this module imports even before that
code lands.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable

from verifier.types import VerifyResult

from . import gold as gold_mod
from . import parser as parser_mod
from . import tasks as tasks_mod

if TYPE_CHECKING:  # avoid importing heavy deps at module load
    import verifiers as vf

    from verifier.types import Verifier

logger = logging.getLogger("infra_synth")


# ---------------------------------------------------------------------------
# vf-free reward core (unit-tested directly in tests/test_infra_reward.py)
# ---------------------------------------------------------------------------
def _score(
    result: VerifyResult,
    *,
    build_weight: float = 0.3,
    smoke_weight: float = 0.7,
    hack_penalty: float = 0.0,
    binary: bool = False,
) -> float:
    """Pure-Python mirror of ``verifier.shape_reward`` (the documented contract).

    Used as a **fallback** when ``verifier.shape_reward`` is not importable yet
    and as the unit under test for the shaping math (so the reward logic is
    verifiable without the parallel verifier package or ``verifiers``):

        reward = build_weight*build_ok + smoke_weight*smoke_ok - hack_penalty*hack

    With ``binary=True`` the reward is ``1.0`` only when both build and smoke
    pass (still minus the hack penalty), else ``0.0``. The result is clamped to
    ``[0.0, 1.0]`` after applying the penalty.
    """
    hack = 1.0 if result.hack_flags.any() else 0.0
    if binary:
        base = 1.0 if (result.build_ok and result.smoke_ok) else 0.0
    else:
        base = build_weight * (1.0 if result.build_ok else 0.0) + smoke_weight * (
            1.0 if result.smoke_ok else 0.0
        )
    reward = base - hack_penalty * hack
    return max(0.0, min(1.0, reward))


def _metrics(result: VerifyResult) -> dict[str, float]:
    """Pure-Python mirror of ``verifier.result_to_metrics`` (fallback)."""
    return {
        "build_ok": 1.0 if result.build_ok else 0.0,
        "smoke_ok": 1.0 if result.smoke_ok else 0.0,
        "hack_any": 1.0 if result.hack_flags.any() else 0.0,
    }


def _resolve_verifier(
    verifier: "Verifier | None",
    verifier_backend: str,
    sentinel_base_url: str | None,
) -> "Verifier":
    """Return the verifier to use: the injected one, else ``get_verifier(...)``.

    ``get_verifier`` is imported lazily so this module imports even before the
    parallel ``verifier.backends`` / ``verifier.reward`` code lands.
    """
    if verifier is not None:
        return verifier
    from verifier import get_verifier  # lazy: provided by the parallel agent

    return get_verifier(verifier_backend, base_url=sentinel_base_url)


def _shape_reward(result: VerifyResult, **kw: float) -> float:
    """Call ``verifier.shape_reward`` if available, else the local mirror."""
    try:
        from verifier import shape_reward  # lazy
    except Exception:  # pragma: no cover - depends on parallel agent
        return _score(result, **kw)
    return shape_reward(result, **kw)


def _result_to_metrics(result: VerifyResult) -> dict[str, float]:
    """Call ``verifier.result_to_metrics`` if available, else the local mirror."""
    try:
        from verifier import result_to_metrics  # lazy
    except Exception:  # pragma: no cover - depends on parallel agent
        return _metrics(result)
    return result_to_metrics(result)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
def load_environment(
    *,
    verifier_backend: str = "static",
    verifier: "Verifier | None" = None,
    build_weight: float = 0.3,
    smoke_weight: float = 0.7,
    hack_penalty: float = 0.0,
    num_tasks: int | None = None,
    seed: int = 0,
    split: str = "train",
    sentinel_base_url: str | None = None,
    **kwargs: Any,
) -> "vf.Environment":
    """Construct and return the ``infra_synth`` single-turn RLVR environment.

    Parameters
    ----------
    verifier_backend:
        Which verifier to resolve via ``verifier.get_verifier`` when ``verifier``
        is not injected. ``"static"`` (zero-dependency static checks, the
        default), ``"local-py"``, ``"local-docker"`` (genuine build + smoke), or
        ``"sentinel"`` (hardened sandbox).
    verifier:
        An explicit ``Verifier`` instance. If given, it overrides
        ``verifier_backend`` (used for testing with a stub).
    build_weight, smoke_weight, hack_penalty:
        Reward-shaping weights forwarded to ``verifier.shape_reward``.
    num_tasks, seed, split:
        Dataset sizing / determinism. ``split="train"`` and ``split="test"`` use
        DISJOINT parameter combinations (contamination-resistant).
    sentinel_base_url:
        Base URL for the Sentinel backend (ignored by other backends).
    """
    import verifiers as vf  # heavy import, only when actually constructing the env

    parser = vf.Parser(extract_fn=parser_mod.extract_dockerfile)

    # Resolve the verifier once and close over it.
    active_verifier = _resolve_verifier(verifier, verifier_backend, sentinel_base_url)

    # Zero-arg dataset builders (verifiers accepts a DatasetBuilder callable).
    def _train_builder() -> "Any":
        from datasets import Dataset

        return Dataset.from_list(
            tasks_mod.generate_tasks(n=num_tasks, seed=seed, split=split)
        )

    def _eval_builder() -> "Any":
        from datasets import Dataset

        return Dataset.from_list(
            tasks_mod.generate_tasks(n=num_tasks, seed=seed, split="test")
        )

    # ------------------------------------------------------------------
    # Reward functions (verifiers passes kwargs by NAME; always take **kwargs).
    # ------------------------------------------------------------------
    async def build_smoke_reward(
        completion: Any = None,
        info: dict | None = None,
        state: dict | None = None,
        parser: "vf.Parser" = parser,  # noqa: B008 - default is the env parser
        **_kw: Any,
    ) -> float:
        """Primary reward: verify the emitted Dockerfile (build + smoke).

        Registered FIRST (weight 1.0). The weight-0 metric fns rely on this
        having run first to populate ``state['infra_synth']``.

        verifiers SILENTLY catches reward-fn exceptions and scores 0.0, so we
        catch and log our own errors explicitly to avoid masking failures as
        legitimate zeros.
        """
        info = info or {}
        state = state if state is not None else {}
        try:
            artifact = parser.parse_answer(completion) or ""
        except Exception as e:  # pragma: no cover - defensive
            logger.exception("infra_synth: failed to parse artifact: %s", e)
            artifact = ""

        try:
            spec = tasks_mod.build_verify_spec(info)
        except Exception as e:
            logger.exception("infra_synth: failed to build VerifySpec: %s", e)
            result = VerifyResult(status=f"spec-error:{e}", backend=verifier_backend)
            state["infra_synth"] = {"result": result, "metrics": _metrics(result)}
            return 0.0

        try:
            result = await active_verifier.verify(artifact, spec)
        except Exception as e:
            logger.exception("infra_synth: verifier.verify raised: %s", e)
            result = VerifyResult(status=f"verify-error:{e}", backend=verifier_backend)

        try:
            r = _shape_reward(
                result,
                build_weight=build_weight,
                smoke_weight=smoke_weight,
                hack_penalty=hack_penalty,
            )
            metrics = _result_to_metrics(result)
        except Exception as e:  # pragma: no cover - defensive
            logger.exception("infra_synth: reward shaping raised: %s", e)
            r, metrics = 0.0, _metrics(result)

        # Stash for the weight-0 metric fns + post-hoc analysis.
        state["infra_synth"] = {"result": result, "metrics": metrics}
        return float(r)

    def format_reward(
        completion: Any = None,
        parser: "vf.Parser" = parser,  # noqa: B008
        **_kw: Any,
    ) -> float:
        """Small dense format signal: 1.0 if a non-empty Dockerfile was emitted.

        Weight is kept small (0.1) so the policy cannot game reward by emitting a
        well-formatted but non-building Dockerfile; it exists to fight early
        zero-advantage when nothing builds yet.
        """
        try:
            artifact = parser.parse_answer(completion) or ""
        except Exception:  # pragma: no cover - defensive
            return 0.0
        return 1.0 if artifact.strip() else 0.0

    # ------------------------------------------------------------------
    # Weight-0 metrics. They READ state['infra_synth']['metrics'], which is
    # populated by build_smoke_reward (registered first => runs first).
    # ------------------------------------------------------------------
    def _metric_reader(key: str) -> Callable[..., float]:
        def _fn(state: dict | None = None, **_kw: Any) -> float:
            data = (state or {}).get("infra_synth") or {}
            return float((data.get("metrics") or {}).get(key, 0.0))

        return _fn

    build_ok_metric = _metric_reader("build_ok")
    smoke_ok_metric = _metric_reader("smoke_ok")
    hack_any_metric = _metric_reader("hack_any")
    # Stable names in logged metrics.
    build_ok_metric.__name__ = "build_ok_metric"
    smoke_ok_metric.__name__ = "smoke_ok_metric"
    hack_any_metric.__name__ = "hack_any_metric"

    rubric = vf.Rubric(
        funcs=[build_smoke_reward, format_reward],
        weights=[1.0, 0.1],
        parser=parser,
    )
    rubric.add_metric(build_ok_metric)
    rubric.add_metric(smoke_ok_metric)
    rubric.add_metric(hack_any_metric)

    return vf.SingleTurnEnv(
        dataset=_train_builder,
        eval_dataset=_eval_builder,
        system_prompt=tasks_mod.SYSTEM_PROMPT,
        parser=parser,
        rubric=rubric,
    )


# Re-export the gold helper for convenience (eval references / sanity checks).
gold_dockerfile = gold_mod.gold_dockerfile

__all__ = ["load_environment", "gold_dockerfile"]
