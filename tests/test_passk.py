"""Tests for eval.passk: the unbiased pass@k estimator.

Covers the edge cases (c=0 -> 0, c > n-k -> 1, monotonicity in k), that the
stable product form matches the direct binomial C(n-c,k)/C(n,k) on small n, and
the corpus aggregation / curve helpers. Stdlib + pytest only.
"""
from __future__ import annotations

import pytest

from eval.passk import (
    _pass_at_k_binomial,
    pass_at_k,
    pass_at_k_corpus,
    passk_curve,
)


# --- edge cases ------------------------------------------------------------
def test_c_zero_is_zero() -> None:
    # No correct sample -> pass@k is 0 for every k.
    for k in (1, 2, 5, 10):
        assert pass_at_k(10, 0, k) == 0.0


def test_c_greater_than_n_minus_k_is_one() -> None:
    # When n - c < k, every size-k subset must contain a correct sample -> 1.0.
    assert pass_at_k(10, 9, 2) == 1.0  # n-c = 1 < 2
    assert pass_at_k(10, 10, 1) == 1.0  # all correct
    assert pass_at_k(5, 5, 5) == 1.0
    assert pass_at_k(8, 7, 3) == 1.0  # n-c = 1 < 3


def test_all_correct_one_sample() -> None:
    assert pass_at_k(1, 1, 1) == 1.0
    assert pass_at_k(1, 0, 1) == 0.0


def test_k_clamped_to_n() -> None:
    # k > n is clamped to n (cannot draw more than you have).
    assert pass_at_k(4, 2, 10) == pass_at_k(4, 2, 4)


def test_monotonic_in_k() -> None:
    # pass@k is non-decreasing in k for fixed (n, c).
    n, c = 50, 7
    prev = -1.0
    for k in range(1, n + 1):
        val = pass_at_k(n, c, k)
        assert val >= prev - 1e-12
        prev = val
    assert pass_at_k(n, c, n) == 1.0  # with k=n you always include all correct


def test_bounded_unit_interval() -> None:
    for n in (1, 5, 20, 200):
        for c in (0, 1, n // 3, n):
            for k in (1, 2, 5):
                v = pass_at_k(n, c, k)
                assert 0.0 <= v <= 1.0


# --- stable product == direct binomial on small n -------------------------
@pytest.mark.parametrize("n", [1, 2, 3, 5, 8, 12, 20])
def test_product_form_matches_binomial(n: int) -> None:
    for c in range(0, n + 1):
        for k in range(1, n + 1):
            stable = pass_at_k(n, c, k)
            direct = _pass_at_k_binomial(n, c, k)
            assert stable == pytest.approx(direct, abs=1e-9), (n, c, k, stable, direct)


def test_binomial_reference_sanity() -> None:
    # Spot-check against a hand computation: n=4, c=1, k=2.
    #   1 - C(3,2)/C(4,2) = 1 - 3/6 = 0.5
    assert _pass_at_k_binomial(4, 1, 2) == pytest.approx(0.5)
    assert pass_at_k(4, 1, 2) == pytest.approx(0.5)
    # n=4, c=2, k=2 -> 1 - C(2,2)/C(4,2) = 1 - 1/6
    assert pass_at_k(4, 2, 2) == pytest.approx(1 - 1 / 6)


def test_large_n_stable_no_overflow() -> None:
    # The product form must stay finite/sane where the raw binomials are huge.
    v = pass_at_k(200, 3, 128)
    assert 0.0 <= v <= 1.0
    # Compare to the exact binomial (math.comb handles big ints exactly).
    assert v == pytest.approx(_pass_at_k_binomial(200, 3, 128), abs=1e-9)


# --- input validation ------------------------------------------------------
def test_invalid_inputs_raise() -> None:
    with pytest.raises(ValueError):
        pass_at_k(0, 0, 1)
    with pytest.raises(ValueError):
        pass_at_k(5, 6, 1)  # c > n
    with pytest.raises(ValueError):
        pass_at_k(5, -1, 1)
    with pytest.raises(ValueError):
        pass_at_k(5, 2, 0)  # k < 1


# --- corpus + curve --------------------------------------------------------
def test_corpus_is_mean_over_problems() -> None:
    results = [(10, 0), (10, 10), (10, 5)]
    expected = (
        pass_at_k(10, 0, 1) + pass_at_k(10, 10, 1) + pass_at_k(10, 5, 1)
    ) / 3
    assert pass_at_k_corpus(results, 1) == pytest.approx(expected)


def test_corpus_empty_is_zero() -> None:
    assert pass_at_k_corpus([], 1) == 0.0


def test_curve_keys_and_monotonic() -> None:
    results = [(50, 3), (50, 1), (50, 8), (50, 0)]
    ks = [1, 2, 4, 8, 16, 32]
    curve = passk_curve(results, ks)
    assert set(curve) == set(ks)
    vals = [curve[k] for k in ks]
    # Corpus pass@k is non-decreasing in k (each problem's is).
    assert all(vals[i] <= vals[i + 1] + 1e-12 for i in range(len(vals) - 1))
    assert all(0.0 <= v <= 1.0 for v in vals)
