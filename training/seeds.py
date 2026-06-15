"""Multi-seed aggregation + launch helpers for GRPO runs (torch-free).

The **M1 definition-of-done** requires reward to rise *stably across >=3 seeds*.
This module provides the stdlib-only statistics to back that claim and a helper
to launch :mod:`training.run` across seeds ``{0, 1, 2}`` and collect their
per-run JSON summaries.

Functions
---------
- :func:`aggregate_seed_metrics` — given a list of per-seed metric dicts, compute
  ``mean``, ``std`` (sample std, ``ddof=1``) and a normal-approx 95% confidence
  interval (``mean ± 1.96 * std / sqrt(n)``) for every shared metric key.
- :func:`summarize_runs` — read per-step JSONL metric logs (one JSON object per
  line) and reduce each to a final-value dict per run, then aggregate.
- :func:`launch_seeds` — run ``python -m training.run`` (or ``training/run.py``)
  for each seed in a subprocess and load the resulting per-run summaries.

Everything here is pure stdlib (``json`` / ``math`` / ``subprocess``); no torch,
trl, datasets or verifier import. Statistics are unit-tested directly.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from collections.abc import Iterable, Sequence
from typing import Any

__all__ = [
    "aggregate_seed_metrics",
    "summarize_runs",
    "launch_seeds",
    "mean_std_ci",
]

_Z95 = 1.96  # standard-normal two-sided 95% critical value


def mean_std_ci(values: Sequence[float]) -> dict[str, float]:
    """Return ``{mean, std, ci95, ci_low, ci_high, n}`` for ``values``.

    - ``std`` is the **sample** standard deviation (``ddof=1``); it is ``0.0``
      when ``n < 2`` (a single seed has no spread).
    - ``ci95`` is the half-width ``1.96 * std / sqrt(n)``; ``ci_low``/``ci_high``
      are ``mean ∓ ci95``. With ``n == 1`` the interval collapses to the point.

    Empty input yields all-``nan`` (so callers can detect "no data").
    """
    n = len(values)
    if n == 0:
        nan = float("nan")
        return {"mean": nan, "std": nan, "ci95": nan, "ci_low": nan, "ci_high": nan, "n": 0}

    vals = [float(v) for v in values]
    mean = math.fsum(vals) / n
    if n < 2:
        std = 0.0
    else:
        var = math.fsum((v - mean) ** 2 for v in vals) / (n - 1)
        std = math.sqrt(var)
    ci95 = _Z95 * std / math.sqrt(n)
    return {
        "mean": mean,
        "std": std,
        "ci95": ci95,
        "ci_low": mean - ci95,
        "ci_high": mean + ci95,
        "n": n,
    }


def aggregate_seed_metrics(runs: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Aggregate per-seed metric dicts into ``{metric: {mean, std, ci95, ...}}``.

    ``runs`` is a list of dicts (one per seed), each mapping a metric name to a
    scalar (e.g. ``{"reward_mean": 0.42, "kl": 0.01, "pass@1": 0.4}``). For each
    metric key present in **any** run we collect the (numeric) values across
    runs and reduce via :func:`mean_std_ci`. Metric keys are the **union** across
    runs; a run missing a key simply does not contribute a value for it.

    Non-numeric values (e.g. a string ``seed`` label) are skipped per metric.
    Returns a mapping from metric name to its summary stats. Empty ``runs`` ->
    empty dict.
    """
    if not runs:
        return {}

    keys: list[str] = []
    seen: set[str] = set()
    for run in runs:
        for k in run:
            if k not in seen:
                seen.add(k)
                keys.append(k)

    summary: dict[str, dict[str, float]] = {}
    for key in keys:
        values: list[float] = []
        for run in runs:
            if key not in run:
                continue
            v = run[key]
            if isinstance(v, bool):  # bools are ints in Python; treat as numeric
                values.append(float(v))
            elif isinstance(v, (int, float)):
                values.append(float(v))
            # else: skip non-numeric (e.g. nested dicts / strings)
        if values:
            summary[key] = mean_std_ci(values)
    return summary


def _final_from_jsonl(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Reduce a per-step JSONL metric log to a dict of last-seen numeric values.

    Each line is a JSON object of step metrics (e.g. ``{"step": 5, "reward":
    0.3, ...}``). We keep, per key, the value from the **last** line that carried
    it (the final logged value). Blank/malformed lines are ignored.
    """
    final: dict[str, Any] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                final.update(obj)
    return final


def summarize_runs(jsonl_paths: Iterable[str | os.PathLike[str]]) -> dict[str, Any]:
    """Read per-step JSONL logs (one per seed) and aggregate their final metrics.

    For each path we take the last logged value of every metric
    (:func:`_final_from_jsonl`), then aggregate across seeds with
    :func:`aggregate_seed_metrics`. Returns ``{"per_run": [...], "aggregate":
    {...}, "n_runs": int}``. Missing files are skipped (recorded under
    ``"missing"``).
    """
    per_run: list[dict[str, Any]] = []
    missing: list[str] = []
    for path in jsonl_paths:
        if not os.path.exists(path):
            missing.append(str(path))
            continue
        per_run.append(_final_from_jsonl(path))

    out: dict[str, Any] = {
        "per_run": per_run,
        "aggregate": aggregate_seed_metrics(per_run),
        "n_runs": len(per_run),
    }
    if missing:
        out["missing"] = missing
    return out


def launch_seeds(
    *,
    env: str = "gsm8k",
    seeds: Sequence[int] = (0, 1, 2),
    output_dir: str = "outputs/m1",
    extra_args: Sequence[str] | None = None,
    python_exe: str | None = None,
    run_module: str = "training.run",
    check: bool = True,
) -> dict[str, Any]:
    """Launch ``training.run`` once per seed and collect the per-run summaries.

    For each ``seed`` we invoke (in a subprocess)::

        <python> -m training.run --env <env> --seed <seed>
            --output-dir <output_dir>/seed-<seed> <extra_args...>

    then read ``<output_dir>/seed-<seed>/summary.json`` (the per-run summary
    :func:`training.run.main` writes). Results are aggregated across seeds with
    :func:`aggregate_seed_metrics`.

    Parameters
    ----------
    extra_args:
        Additional CLI flags forwarded to every run (e.g.
        ``["--model", "Qwen/Qwen3-1.7B", "--num-generations", "8"]``).
    check:
        If ``True`` (default), raise ``CalledProcessError`` on a non-zero run.

    Returns ``{"per_run": [...], "aggregate": {...}, "seeds": [...],
    "summary_paths": [...]}``. **This actually spawns training** — for the
    metric math alone use :func:`aggregate_seed_metrics` / :func:`summarize_runs`
    directly. Heavy training imports live in ``training.run`` (the subprocess),
    not here, so this module stays torch-free.
    """
    python_exe = python_exe or sys.executable
    extra = list(extra_args or [])
    per_run: list[dict[str, Any]] = []
    summary_paths: list[str] = []

    for seed in seeds:
        seed_dir = os.path.join(output_dir, f"seed-{seed}")
        os.makedirs(seed_dir, exist_ok=True)
        cmd = [
            python_exe, "-m", run_module,
            "--env", env,
            "--seed", str(seed),
            "--output-dir", seed_dir,
            *extra,
        ]
        subprocess.run(cmd, check=check)  # noqa: S603 - trusted local invocation

        summary_path = os.path.join(seed_dir, "summary.json")
        summary_paths.append(summary_path)
        if os.path.exists(summary_path):
            with open(summary_path, encoding="utf-8") as fh:
                data = json.load(fh)
            # The summary stores final metrics under "final_metrics" (see run.py).
            metrics = data.get("final_metrics", data) if isinstance(data, dict) else {}
            per_run.append({"seed": seed, **metrics})

    return {
        "per_run": per_run,
        "aggregate": aggregate_seed_metrics(per_run),
        "seeds": list(seeds),
        "summary_paths": summary_paths,
    }
