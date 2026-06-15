"""Unit tests for ``training.seeds`` — the multi-seed aggregation math.

Pure stdlib (no torch / subprocess launched here): we exercise
:func:`mean_std_ci` and :func:`aggregate_seed_metrics` on synthetic per-seed
dicts and assert the mean / sample-std / 95% CI arithmetic, including the
``CI = mean ± 1.96 * std / sqrt(n)`` rule the M1 definition-of-done relies on.
"""
from __future__ import annotations

import math

import pytest

from training.seeds import aggregate_seed_metrics, mean_std_ci, summarize_runs

_Z95 = 1.96


# ---------------------------------------------------------------------------
# mean_std_ci
# ---------------------------------------------------------------------------
def test_mean_std_ci_basic() -> None:
    vals = [0.2, 0.4, 0.6]
    out = mean_std_ci(vals)
    mean = sum(vals) / 3
    # Sample std (ddof=1).
    var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
    std = math.sqrt(var)
    ci = _Z95 * std / math.sqrt(3)

    assert out["n"] == 3
    assert out["mean"] == pytest.approx(0.4)
    assert out["std"] == pytest.approx(std)
    assert out["ci95"] == pytest.approx(ci)
    assert out["ci_low"] == pytest.approx(0.4 - ci)
    assert out["ci_high"] == pytest.approx(0.4 + ci)
    # The defining relationship.
    assert out["ci95"] == pytest.approx(_Z95 * out["std"] / math.sqrt(out["n"]))


def test_mean_std_ci_uses_sample_std_ddof1() -> None:
    # [2, 4, 4, 4, 5, 5, 7, 9]: population std = 2.0, sample std = 2.13809...
    vals = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
    out = mean_std_ci(vals)
    n = len(vals)
    mean = sum(vals) / n
    sample_std = math.sqrt(sum((v - mean) ** 2 for v in vals) / (n - 1))
    assert out["mean"] == pytest.approx(5.0)
    assert out["std"] == pytest.approx(sample_std)
    assert out["std"] != pytest.approx(2.0)  # NOT the population std (ddof=0)


def test_mean_std_ci_single_value_zero_spread() -> None:
    out = mean_std_ci([0.5])
    assert out["n"] == 1
    assert out["mean"] == pytest.approx(0.5)
    assert out["std"] == 0.0  # a single seed has no spread
    assert out["ci95"] == 0.0
    assert out["ci_low"] == pytest.approx(0.5)
    assert out["ci_high"] == pytest.approx(0.5)


def test_mean_std_ci_empty_is_nan() -> None:
    out = mean_std_ci([])
    assert out["n"] == 0
    assert math.isnan(out["mean"])
    assert math.isnan(out["std"])
    assert math.isnan(out["ci95"])


def test_mean_std_ci_constant_values() -> None:
    out = mean_std_ci([0.7, 0.7, 0.7])
    assert out["mean"] == pytest.approx(0.7)
    assert out["std"] == pytest.approx(0.0)
    assert out["ci95"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# aggregate_seed_metrics
# ---------------------------------------------------------------------------
def test_aggregate_seed_metrics_per_key() -> None:
    runs = [
        {"reward_mean": 0.2, "kl": 0.01},
        {"reward_mean": 0.4, "kl": 0.02},
        {"reward_mean": 0.6, "kl": 0.03},
    ]
    agg = aggregate_seed_metrics(runs)

    assert set(agg) == {"reward_mean", "kl"}
    assert agg["reward_mean"]["mean"] == pytest.approx(0.4)
    assert agg["reward_mean"]["n"] == 3
    # CI math holds per metric.
    rm = agg["reward_mean"]
    assert rm["ci95"] == pytest.approx(_Z95 * rm["std"] / math.sqrt(rm["n"]))
    assert agg["kl"]["mean"] == pytest.approx(0.02)


def test_aggregate_seed_metrics_union_of_keys() -> None:
    # A metric present in only some runs still aggregates over what's available.
    runs = [
        {"reward_mean": 0.2, "pass@1": 0.1},
        {"reward_mean": 0.4},  # no pass@1 here
        {"reward_mean": 0.6, "pass@1": 0.3},
    ]
    agg = aggregate_seed_metrics(runs)
    assert agg["reward_mean"]["n"] == 3
    assert agg["pass@1"]["n"] == 2  # only the two runs that had it
    assert agg["pass@1"]["mean"] == pytest.approx(0.2)


def test_aggregate_seed_metrics_skips_non_numeric() -> None:
    # A string label (e.g. a "seed" tag) is ignored; bools count as numeric.
    runs = [
        {"seed": "s0", "reward_mean": 1.0, "converged": True},
        {"seed": "s1", "reward_mean": 0.0, "converged": False},
    ]
    agg = aggregate_seed_metrics(runs)
    assert "seed" not in agg
    assert agg["reward_mean"]["mean"] == pytest.approx(0.5)
    assert agg["converged"]["mean"] == pytest.approx(0.5)  # True/False -> 1/0


def test_aggregate_seed_metrics_empty() -> None:
    assert aggregate_seed_metrics([]) == {}


# ---------------------------------------------------------------------------
# summarize_runs — JSONL reduction + aggregation (torch-free, file-based)
# ---------------------------------------------------------------------------
def test_summarize_runs_reads_last_value_and_aggregates(tmp_path) -> None:
    # Each per-seed JSONL keeps the LAST logged value of every metric.
    seed_dirs = []
    for i, final_reward in enumerate([0.2, 0.4, 0.6]):
        p = tmp_path / f"metrics-{i}.jsonl"
        p.write_text(
            f'{{"step": 1, "reward": 0.0}}\n{{"step": 2, "reward": {final_reward}}}\n',
            encoding="utf-8",
        )
        seed_dirs.append(str(p))

    out = summarize_runs(seed_dirs)
    assert out["n_runs"] == 3
    assert [r["reward"] for r in out["per_run"]] == pytest.approx([0.2, 0.4, 0.6])
    assert out["aggregate"]["reward"]["mean"] == pytest.approx(0.4)


def test_summarize_runs_records_missing(tmp_path) -> None:
    out = summarize_runs([str(tmp_path / "does-not-exist.jsonl")])
    assert out["n_runs"] == 0
    assert out["missing"] == [str(tmp_path / "does-not-exist.jsonl")]
