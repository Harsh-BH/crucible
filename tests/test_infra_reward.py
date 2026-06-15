"""Tests for the vf-free reward core in ``infra_synth.environment`` (no backend).

We do NOT depend on the parallel ``verifier`` reward implementation here: we
inject a STUB ``Verifier`` returning a canned ``VerifyResult`` and assert the
shaping math via the factored-out ``_score`` helper (the pure-Python mirror of
``verifier.shape_reward``).

Importing ``infra_synth.environment`` is ``verifiers``-free (``verifiers`` is
imported lazily inside ``load_environment``), so this stays vf-free.
"""
from __future__ import annotations

import asyncio

import pytest

from infra_synth.environment import _metrics, _score
from verifier.types import HackFlags, Verifier, VerifySpec, VerifyResult


class StubVerifier:
    """Canned ``Verifier`` (implements the protocol) for reward-path testing."""

    name = "stub"

    def __init__(self, result: VerifyResult) -> None:
        self._result = result
        self.calls: list[tuple[str, VerifySpec]] = []

    async def verify(self, artifact: str, spec: VerifySpec) -> VerifyResult:
        self.calls.append((artifact, spec))
        return self._result


def test_stub_satisfies_verifier_protocol() -> None:
    stub = StubVerifier(VerifyResult())
    assert isinstance(stub, Verifier)


@pytest.mark.parametrize(
    "build_ok,smoke_ok,bw,sw,expected",
    [
        (False, False, 0.3, 0.7, 0.0),
        (True, False, 0.3, 0.7, 0.3),
        (False, True, 0.3, 0.7, 0.7),
        (True, True, 0.3, 0.7, 1.0),
        (True, False, 0.5, 0.5, 0.5),
        (False, True, 0.5, 0.5, 0.5),
    ],
)
def test_score_matches_weighted_sum(build_ok, smoke_ok, bw, sw, expected) -> None:
    r = VerifyResult(build_ok=build_ok, smoke_ok=smoke_ok)
    got = _score(r, build_weight=bw, smoke_weight=sw)
    assert got == pytest.approx(bw * build_ok + sw * smoke_ok)
    assert got == pytest.approx(expected)


def test_score_hack_penalty_and_clamp() -> None:
    r = VerifyResult(build_ok=True, smoke_ok=True, hack_flags=HackFlags(network_attempt=True))
    # 1.0 base - 0.5 penalty = 0.5
    assert _score(r, build_weight=0.3, smoke_weight=0.7, hack_penalty=0.5) == pytest.approx(0.5)
    # A penalty larger than the base clamps to 0.0 (never negative).
    assert _score(r, build_weight=0.3, smoke_weight=0.7, hack_penalty=5.0) == 0.0


def test_score_binary_mode() -> None:
    both = VerifyResult(build_ok=True, smoke_ok=True)
    one = VerifyResult(build_ok=True, smoke_ok=False)
    assert _score(both, binary=True) == 1.0
    assert _score(one, binary=True) == 0.0


def test_metrics_reflect_result() -> None:
    r = VerifyResult(build_ok=True, smoke_ok=False, hack_flags=HackFlags(oom_killed=True))
    m = _metrics(r)
    assert m == {"build_ok": 1.0, "smoke_ok": 0.0, "hack_any": 1.0}


def test_full_reward_path_with_stub() -> None:
    """End-to-end: stub.verify() result -> _score, matching the reward fn logic."""
    canned = VerifyResult(build_ok=True, smoke_ok=True, backend="stub")
    stub = StubVerifier(canned)
    spec = VerifySpec(spec_id="t", kind="dockerfile")  # type: ignore[arg-type]

    result = asyncio.run(stub.verify("FROM python:3.11-slim\nEXPOSE 8000", spec))
    assert stub.calls, "verifier should have been invoked"
    reward = _score(result, build_weight=0.3, smoke_weight=0.7)
    assert reward == pytest.approx(1.0)
    # The metric snapshot the env stashes in state mirrors _metrics(result).
    assert _metrics(result) == {"build_ok": 1.0, "smoke_ok": 1.0, "hack_any": 0.0}


def test_score_mirrors_real_shape_reward_if_available() -> None:
    """If the parallel verifier.shape_reward exists, our mirror must agree."""
    shape_reward = pytest.importorskip("verifier").__dict__.get("shape_reward")
    if shape_reward is None:
        pytest.skip("verifier.shape_reward not available yet")
    for bo in (False, True):
        for so in (False, True):
            r = VerifyResult(build_ok=bo, smoke_ok=so)
            assert _score(r, build_weight=0.3, smoke_weight=0.7) == pytest.approx(
                shape_reward(r, build_weight=0.3, smoke_weight=0.7)
            )
