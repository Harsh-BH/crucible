"""Unbiased ``pass@k`` estimator (Chen et al., HumanEval; arXiv:2107.03374).

`pass@k` is the probability that at least one of ``k`` i.i.d. samples drawn from
the policy for a problem is correct. Estimating it as "fraction of problems with
>=1 correct in the first k samples" is **biased** (high variance, and undefined
when you draw ``n != k`` samples). The HumanEval paper's fix is to draw ``n >> k``
samples per problem, count ``c`` correct, and use the *unbiased* combinatorial
estimator::

    pass@k = 1 - C(n - c, k) / C(n, k)

i.e. one minus the probability that a size-``k`` subset of the ``n`` samples
contains *no* correct sample. We use the **numerically stable product form**

    pass@k = 1 - prod_{i = n-c+1}^{n} (1 - k / i)

which avoids overflow in the binomials for large ``n`` (we evaluate to k≈128–256
for the "search-compression" sweep — see ``DESIGN.md`` and arXiv:2504.13837).

This module is **stdlib-only** so it imports anywhere (no numpy/torch). The
corpus aggregate is the mean of per-problem ``pass@k`` (the standard HumanEval
reduction), which is what unbiasedness is defined against.
"""
from __future__ import annotations

import math

__all__ = ["pass_at_k", "pass_at_k_corpus", "passk_curve"]


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased ``pass@k`` for one problem with ``n`` samples, ``c`` correct.

    Parameters
    ----------
    n:
        Total samples drawn for the problem (must be ``>= 1``).
    c:
        Number of those samples that were correct (``0 <= c <= n``).
    k:
        The ``k`` in ``pass@k`` (``k >= 1``). ``k`` is clamped to ``n`` (you
        cannot select more than you drew), matching the HumanEval convention.

    Returns
    -------
    float
        The probability in ``[0.0, 1.0]`` that a uniformly random size-``k``
        subset of the ``n`` samples contains at least one correct sample.

    Notes
    -----
    Edge cases (all consistent with the closed form):

    * ``c == 0``           -> ``0.0`` (no correct sample can ever be selected).
    * ``c > n - k``        -> ``1.0`` (every size-``k`` subset must include a
      correct one — there are too few incorrect samples to fill it).
    * otherwise the stable product ``1 - prod_{i=n-c+1}^{n} (1 - k/i)``.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if not (0 <= c <= n):
        raise ValueError(f"c must satisfy 0 <= c <= n, got c={c}, n={n}")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    # Cannot select more samples than were drawn.
    if k > n:
        k = n
    if c == 0:
        return 0.0
    if c > n - k:
        # Not enough incorrect samples (n - c) to fill a size-k subset, so every
        # subset contains >=1 correct sample.
        return 1.0
    # Stable product form: prod_{i=n-c+1}^{n} (1 - k/i) == C(n-c, k) / C(n, k).
    prob_all_wrong = 1.0
    for i in range(n - c + 1, n + 1):
        prob_all_wrong *= 1.0 - k / i
    return 1.0 - prob_all_wrong


def _pass_at_k_binomial(n: int, c: int, k: int) -> float:
    """Reference implementation via :func:`math.comb` (exact; for tests/docs).

    ``1 - C(n-c, k) / C(n, k)``. Numerically fine for small ``n`` but can lose
    precision / overflow for large ``n`` — :func:`pass_at_k` is the production
    path. Kept here so the equivalence is documented and unit-tested.
    """
    if k > n:
        k = n
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def pass_at_k_corpus(results: list[tuple[int, int]], k: int) -> float:
    """Corpus-level ``pass@k``: the mean of per-problem ``pass@k``.

    Parameters
    ----------
    results:
        One ``(n_i, c_i)`` tuple per problem — total samples drawn and number
        correct. This is the contamination-resistant unit of aggregation used by
        HumanEval (unbiasedness is defined per problem, then averaged).
    k:
        The ``k`` in ``pass@k``.

    Returns
    -------
    float
        ``mean_i pass_at_k(n_i, c_i, k)``, or ``0.0`` for an empty corpus.
    """
    if not results:
        return 0.0
    total = 0.0
    for n_i, c_i in results:
        total += pass_at_k(n_i, c_i, k)
    return total / len(results)


def passk_curve(results: list[tuple[int, int]], ks: list[int]) -> dict[int, float]:
    """Corpus ``pass@k`` at several ``k`` — the data for a pass@k curve.

    Parameters
    ----------
    results:
        ``[(n_i, c_i), ...]`` as in :func:`pass_at_k_corpus`.
    ks:
        The ``k`` values to evaluate (e.g. ``[1, 2, 4, 8, 16, 32, 64, 128]``
        for the base-vs-RL search-compression sweep).

    Returns
    -------
    dict[int, float]
        ``{k: corpus_pass_at_k}``. ``pass@k`` is monotonically non-decreasing in
        ``k`` for a fixed corpus, so the returned values are non-decreasing in
        ``k`` (a useful invariant to assert in tests / plots).
    """
    return {int(k): pass_at_k_corpus(results, int(k)) for k in ks}
