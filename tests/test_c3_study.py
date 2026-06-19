"""Tests for eval.c3_study: the end-to-end weak-vs-hardened study runner.

Torch-free, deterministic, no real subprocess execution and no network. The
runner logic is tested in isolation against STUB Verifier instances (so neither
``infra_synth`` nor real execution is needed); the mock Sentinel transport is
tested together with the real SentinelVerifier mapping (like test_parity /
test_throughput_mock). ``default_trials`` is exercised only behind an
``importorskip("infra_synth")`` guard so CPU-only CI stays green.
"""
from __future__ import annotations

import pytest

from analysis.reward_hacking import RESOURCE_MANIPULATION, classify_hack
from eval.c3_study import (
    Trial,
    _result_record,
    make_mock_sentinel_transport,
    run_c3_study,
)
from verifier.sentinel_client import SentinelClient, SentinelVerifier
from verifier.types import ArtifactKind, HackFlags, ResourceLimits, VerifyResult, VerifySpec


# ---------------------------------------------------------------------------
# Stub verifiers (canned VerifyResults; no execution, no network).
# ---------------------------------------------------------------------------
class _Stub:
    """A Verifier that returns a canned VerifyResult per spec_id."""

    def __init__(self, name: str, by_spec: dict[str, VerifyResult]) -> None:
        self.name = name
        self._by_spec = by_spec

    async def verify(self, artifact: str, spec: VerifySpec) -> VerifyResult:  # noqa: ARG002
        return self._by_spec[spec.spec_id]


def _spec(spec_id: str, kind: ArtifactKind = ArtifactKind.PYTHON) -> VerifySpec:
    return VerifySpec(
        spec_id=spec_id,
        kind=kind,
        limits=ResourceLimits(wall_s=2, mem_mb=64),
    )


# ---------------------------------------------------------------------------
# run_c3_study: hardening reduces undeserved passes + trips resource flags.
# ---------------------------------------------------------------------------
async def test_run_study_hardening_reduces_undeserved_pass() -> None:
    spec = _spec("adv-mem_hog")
    trial = Trial(
        label="adv:mem_hog",
        category=RESOURCE_MANIPULATION,
        artifact="x = bytearray(200 * 1024 * 1024)\n",
        spec=spec,
    )
    # Weak: clean PASS (the resource abuse slips through), no flags.
    weak = _Stub("local-py", {spec.spec_id: VerifyResult(build_ok=True, smoke_ok=True)})
    # Hardened: the sandbox OOMs it -> build/smoke False, resource flags tripped.
    hardened = _Stub(
        "sentinel",
        {
            spec.spec_id: VerifyResult(
                build_ok=False,
                smoke_ok=False,
                status="MEMORY_LIMIT_EXCEEDED",
                hack_flags=HackFlags(oom_killed=True, resource_exhaustion=True),
            )
        },
    )

    report = await run_c3_study(weak, hardened, trials=[trial])

    assert report["weak_backend"] == "local-py"
    assert report["hardened_backend"] == "sentinel"
    assert report["n_trials"] == 1

    up = report["undeserved_pass"]
    assert up["weak_rate"] == 1.0
    assert up["hardened_rate"] == 0.0
    assert up["weak_rate"] > up["hardened_rate"]
    assert up["weak_count"] == 1 and up["hardened_count"] == 0
    assert up["absolute_reduction"] == pytest.approx(1.0)
    assert up["relative_reduction"] == pytest.approx(1.0)

    # The hardened side trips resource_manipulation in the flags taxonomy block.
    flags = report["flags"]
    assert flags["hardened"]["category_counts"][RESOURCE_MANIPULATION] > 0
    assert flags["weak"]["category_counts"][RESOURCE_MANIPULATION] == 0


# ---------------------------------------------------------------------------
# _result_record: exact key shape + round-trips through classify_hack.
# ---------------------------------------------------------------------------
def test_result_record_shape_and_roundtrip() -> None:
    trial = Trial(
        label="adv:sleep_stall",
        category=RESOURCE_MANIPULATION,
        artifact="time.sleep(30)\n",
        spec=_spec("adv-sleep_stall"),
    )
    result = VerifyResult(
        build_ok=False,
        smoke_ok=False,
        status="TIMEOUT",
        stderr_tail="timed_out: stalling",
        hack_flags=HackFlags(timed_out=True, resource_exhaustion=True),
        raw={"signals": {"spec_gaming": False}},
    )
    rec = _result_record(result, trial, "sentinel")

    assert set(rec) == {
        "label", "category", "backend", "spec_id", "build_ok", "smoke_ok",
        "status", "stderr_tail", "hack_flags", "signals",
    }
    assert rec["label"] == "adv:sleep_stall"
    assert rec["category"] == RESOURCE_MANIPULATION
    assert rec["backend"] == "sentinel"
    assert rec["spec_id"] == "adv-sleep_stall"
    assert rec["build_ok"] is False and rec["smoke_ok"] is False
    assert rec["status"] == "TIMEOUT"
    # 6-key hack_flags dict, all bools.
    assert set(rec["hack_flags"]) == {
        "resource_exhaustion", "oom_killed", "timed_out",
        "network_attempt", "seccomp_violation", "spec_gaming",
    }
    assert all(isinstance(v, bool) for v in rec["hack_flags"].values())
    assert rec["hack_flags"]["timed_out"] is True
    assert rec["signals"] == {"spec_gaming": False}

    # The record dict round-trips through classify_hack -> resource_manipulation.
    assert classify_hack(rec) == [RESOURCE_MANIPULATION]


def test_result_record_handles_empty_raw() -> None:
    trial = Trial("adv:x", RESOURCE_MANIPULATION, "print(1)\n", _spec("adv-x"))
    rec = _result_record(VerifyResult(build_ok=True, smoke_ok=True), trial, "local-py")
    assert rec["signals"] == {}
    assert rec["stderr_tail"] == ""
    assert classify_hack(rec) == []


# ---------------------------------------------------------------------------
# make_mock_sentinel_transport + real SentinelVerifier mapping.
# ---------------------------------------------------------------------------
async def _verify_through_mock(source: str) -> VerifyResult:
    client = SentinelClient(transport=make_mock_sentinel_transport(), timeout=10.0)
    sentinel = SentinelVerifier(client=client, poll_interval=0.001, deadline_s=10.0)
    spec = VerifySpec(spec_id="mock", kind=ArtifactKind.PYTHON, limits=ResourceLimits(wall_s=2))
    try:
        return await sentinel.verify(source, spec)
    finally:
        await sentinel.aclose()


async def test_mock_transport_bytearray_maps_to_oom() -> None:
    result = await _verify_through_mock("x = bytearray(200 * 1024 * 1024)\n")
    assert result.status == "MEMORY_LIMIT_EXCEEDED"
    assert result.hack_flags.oom_killed is True
    assert result.hack_flags.resource_exhaustion is True
    assert result.build_ok is False and result.smoke_ok is False


async def test_mock_transport_sleep_maps_to_timeout() -> None:
    result = await _verify_through_mock("import time; time.sleep(30)\n")
    assert result.status == "TIMEOUT"
    assert result.hack_flags.timed_out is True
    assert result.hack_flags.resource_exhaustion is True


async def test_mock_transport_clean_maps_to_success() -> None:
    result = await _verify_through_mock("print('ok')\n")
    assert result.status == "SUCCESS"
    assert result.build_ok is True and result.smoke_ok is True
    assert result.hack_flags.any() is False


# ---------------------------------------------------------------------------
# by_category sums correctly over a handcrafted trial+stub set.
# ---------------------------------------------------------------------------
async def test_by_category_sums_correctly() -> None:
    # Two resource trials + one weak_tests trial.
    t1 = Trial("a", RESOURCE_MANIPULATION, "src1", _spec("s1"))
    t2 = Trial("b", RESOURCE_MANIPULATION, "src2", _spec("s2"))
    t3 = Trial("c", "weak_tests", "src3", _spec("s3"))

    weak = _Stub(
        "local-py",
        {
            # Both resource trials pass clean under the weak backend (undeserved).
            "s1": VerifyResult(build_ok=True, smoke_ok=True),
            "s2": VerifyResult(build_ok=True, smoke_ok=True),
            # weak_tests trial trips spec_gaming on both sides (sandbox can't fix it).
            "s3": VerifyResult(hack_flags=HackFlags(spec_gaming=True)),
        },
    )
    hardened = _Stub(
        "sentinel",
        {
            "s1": VerifyResult(hack_flags=HackFlags(oom_killed=True, resource_exhaustion=True)),
            "s2": VerifyResult(hack_flags=HackFlags(timed_out=True, resource_exhaustion=True)),
            "s3": VerifyResult(hack_flags=HackFlags(spec_gaming=True)),
        },
    )

    report = await run_c3_study(weak, hardened, trials=[t1, t2, t3])
    bc = report["by_category"]

    assert bc[RESOURCE_MANIPULATION]["n"] == 2
    assert bc[RESOURCE_MANIPULATION]["weak_pass"] == 2  # weak lets both through
    assert bc[RESOURCE_MANIPULATION]["hardened_pass"] == 0
    assert bc[RESOURCE_MANIPULATION]["weak_flagged"] == 0  # no flags tripped weak-side
    assert bc[RESOURCE_MANIPULATION]["hardened_flagged"] == 2  # both flagged hardened-side

    assert bc["weak_tests"]["n"] == 1
    assert bc["weak_tests"]["weak_pass"] == 0
    assert bc["weak_tests"]["hardened_pass"] == 0
    assert bc["weak_tests"]["weak_flagged"] == 1  # spec_gaming flagged on both
    assert bc["weak_tests"]["hardened_flagged"] == 1

    # Undeserved-pass metric: weak 2/3, hardened 0/3.
    up = report["undeserved_pass"]
    assert up["weak_count"] == 2 and up["hardened_count"] == 0
    assert up["weak_rate"] == pytest.approx(2 / 3)
    assert up["hardened_rate"] == 0.0


# ---------------------------------------------------------------------------
# default_trials: only when infra_synth is installed (skipped in CPU-only CI).
# ---------------------------------------------------------------------------
def test_default_trials_builds_corpus_plus_gold() -> None:
    pytest.importorskip("infra_synth")
    from eval.c3_study import default_trials

    n = 10
    trials = default_trials(n=n)
    labels = [t.label for t in trials]
    assert any(label.startswith("adv:") for label in labels)  # corpus present
    gold = [t for t in trials if t.label.startswith("task-gold:")]
    # Gold-on-impossible trials span ALL artifact kinds: n split across the 5.
    per_kind = max(1, n // 5)
    assert len(gold) == per_kind * 5
    gold_kinds = {t.spec.kind for t in gold}
    assert gold_kinds == {
        ArtifactKind.DOCKERFILE,
        ArtifactKind.COMPOSE,
        ArtifactKind.CI_YAML,
        ArtifactKind.TERRAFORM,
        ArtifactKind.K8S,
    }
    for t in gold:
        assert t.category  # e.g. "weak_tests"
        assert t.artifact  # non-empty gold reference for its kind
