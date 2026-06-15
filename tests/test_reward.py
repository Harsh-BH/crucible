"""Tests for verifier.reward.shape_reward and result_to_metrics."""
from __future__ import annotations

import pytest

from verifier.reward import result_to_metrics, shape_reward
from verifier.types import HackFlags, VerifyResult


def _res(build: bool, smoke: bool, *, hack: bool = False) -> VerifyResult:
    hf = HackFlags(spec_gaming=hack)
    return VerifyResult(build_ok=build, smoke_ok=smoke, hack_flags=hf)


# --- binary mode -----------------------------------------------------------
def test_binary_full_pass() -> None:
    assert shape_reward(_res(True, True), binary=True) == 1.0


def test_binary_build_only_is_zero() -> None:
    assert shape_reward(_res(True, False), binary=True) == 0.0


def test_binary_fail_is_zero() -> None:
    assert shape_reward(_res(False, False), binary=True) == 0.0


def test_binary_ignores_hack_penalty() -> None:
    assert shape_reward(_res(True, True, hack=True), binary=True, hack_penalty=5.0) == 1.0


# --- shaped mode -----------------------------------------------------------
def test_shaped_full() -> None:
    assert shape_reward(_res(True, True)) == pytest.approx(1.0)


def test_shaped_build_only_partial_credit() -> None:
    # default build_weight=0.3
    assert shape_reward(_res(True, False)) == pytest.approx(0.3)


def test_shaped_smoke_only_weight() -> None:
    # smoke without build is degenerate but weights still apply (0.7).
    assert shape_reward(_res(False, True)) == pytest.approx(0.7)


def test_shaped_fail_is_zero() -> None:
    assert shape_reward(_res(False, False)) == pytest.approx(0.0)


def test_custom_weights() -> None:
    r = shape_reward(_res(True, False), build_weight=0.5, smoke_weight=0.5)
    assert r == pytest.approx(0.5)


def test_weights_over_one_clamped_to_one() -> None:
    r = shape_reward(_res(True, True), build_weight=0.8, smoke_weight=0.8)
    assert r == 1.0


# --- hack penalty ----------------------------------------------------------
def test_hack_penalty_subtracts_only_when_flag_set() -> None:
    no_hack = shape_reward(_res(True, True, hack=False), hack_penalty=0.5)
    assert no_hack == pytest.approx(1.0)
    with_hack = shape_reward(_res(True, True, hack=True), hack_penalty=0.5)
    assert with_hack == pytest.approx(0.5)


def test_hack_penalty_can_push_negative_to_floor() -> None:
    # build-only (0.3) minus 1.0 penalty -> -0.7, but floor is -hack_penalty=-1.0
    r = shape_reward(_res(True, False, hack=True), hack_penalty=1.0)
    assert r == pytest.approx(-0.7)


def test_hack_penalty_floor_clamp() -> None:
    # fail (0.0) minus huge penalty clamps to -hack_penalty.
    r = shape_reward(_res(False, False, hack=True), hack_penalty=2.0)
    assert r == pytest.approx(-2.0)


def test_zero_penalty_floor_is_zero() -> None:
    r = shape_reward(_res(False, False, hack=True), hack_penalty=0.0)
    assert r == 0.0


# --- metrics ---------------------------------------------------------------
def test_result_to_metrics() -> None:
    res = _res(True, False, hack=True)
    res.wall_s = 2.5
    m = result_to_metrics(res)
    assert m == {"build_ok": 1.0, "smoke_ok": 0.0, "hack_any": 1.0, "wall_s": 2.5}
