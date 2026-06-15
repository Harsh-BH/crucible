"""Reward shaping for ``VerifyResult`` -> scalar.

Consumed by the ``infra_synth`` environment to turn a backend-agnostic
:class:`verifier.types.VerifyResult` into a training reward.

Design rationale (per the project's RLVR research notes)
--------------------------------------------------------
Pure pass/fail rewards are *sparse* and, when most rollouts in a group score 0,
yield zero advantage (GRPO degenerates). Awarding **partial credit** for a
Dockerfile that *builds* but does not pass the smoke test (default 0.3) keeps a
useful gradient while still rewarding the full build+serve outcome the most
(default 1.0). The optional ``hack_penalty`` lets the verifier-exploitation
study (C3) subtract reward when any :class:`~verifier.types.HackFlags` tripped.
"""
from __future__ import annotations

from .types import VerifyResult

__all__ = ["shape_reward", "result_to_metrics"]


def shape_reward(
    result: VerifyResult,
    *,
    build_weight: float = 0.3,
    smoke_weight: float = 0.7,
    hack_penalty: float = 0.0,
    binary: bool = False,
) -> float:
    """Map a :class:`VerifyResult` to a scalar reward.

    Parameters
    ----------
    result:
        The verifier outcome to score.
    build_weight, smoke_weight:
        Weights for ``build_ok`` and ``smoke_ok`` in the *shaped* mode. With the
        defaults (0.3 / 0.7) a build-only result scores ``0.3`` and a full
        build+smoke result scores ``1.0``.
    hack_penalty:
        Non-negative amount subtracted from the (shaped) reward when
        ``result.hack_flags.any()`` is true. Ignored in ``binary`` mode.
    binary:
        If true, return ``1.0`` iff (``build_ok`` and ``smoke_ok``) else ``0.0``
        (no partial credit, no penalty).

    Returns
    -------
    float
        In ``binary`` mode: ``0.0`` or ``1.0``.

        In shaped mode: ``build_weight*build_ok + smoke_weight*smoke_ok`` minus
        ``hack_penalty`` if any hack flag tripped, then **clamped** to the closed
        interval ``[-hack_penalty, 1.0]``. The clamp rule lets the penalty push
        the reward negative (down to ``-hack_penalty``) so an exploit can be made
        strictly worse than doing nothing, while the positive side never exceeds
        ``1.0`` regardless of weight choices. When ``hack_penalty == 0`` the
        floor is ``0.0`` (rewards are non-negative).
    """
    if binary:
        return 1.0 if (result.build_ok and result.smoke_ok) else 0.0

    r = build_weight * float(bool(result.build_ok)) + smoke_weight * float(
        bool(result.smoke_ok)
    )
    if hack_penalty and result.hack_flags.any():
        r -= hack_penalty

    floor = -abs(hack_penalty)
    # clamp to [floor, 1.0]
    if r < floor:
        r = floor
    if r > 1.0:
        r = 1.0
    return float(r)


def result_to_metrics(result: VerifyResult) -> dict[str, float]:
    """Return a flat, log-friendly metrics dict for one verification.

    Keys: ``build_ok``, ``smoke_ok``, ``hack_any`` (all 0.0/1.0) and ``wall_s``.
    """
    return {
        "build_ok": float(bool(result.build_ok)),
        "smoke_ok": float(bool(result.smoke_ok)),
        "hack_any": float(bool(result.hack_flags.any())),
        "wall_s": float(result.wall_s),
    }
