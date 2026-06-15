"""Training/eval curve plots for the RLVR run (the standard GRPO dashboard).

Reads **per-step JSONL logs** (one JSON object per optimisation step) and plots
the diagnostics that matter for verifiable-reward RL. The expected per-step
schema (all keys optional — missing series are simply skipped, so this tolerates
whatever subset the trainer emits)::

    {
      "step": 12,
      "reward_mean": 0.41,            # mean shaped reward over the batch
      "reward_std": 0.22,             # WITHIN-GROUP std (the GRPO advantage scale)
      "kl": 0.013,                    # KL(policy || ref)
      "entropy": 1.84,                # token entropy (watch for collapse)
      "completion_length": 213.0,     # mean completion length (tokens)
      "grad_norm": 0.77,
      "frac_zero_advantage": 0.18,    # fraction of groups with ~0 advantage
      "pass_at_1": 0.33,              # optional eval overlays
      "pass_at_k": {"8": 0.55, "64": 0.71}
    }

Why these series (research grounding, cited in ``DESIGN.md``):

* **within-group reward std** and **frac_zero_advantage** track GRPO's failure
  mode — when every rollout in a group gets the same reward the advantage is 0
  and the gradient vanishes (GRPO, arXiv:2402.03300; partial-credit shaping and
  DAPO's dynamic sampling, arXiv:2503.14476, exist to keep this > 0).
* **entropy** guards the entropy-collapse pathology (arXiv:2505.22617).
* **kl / grad_norm / completion_length** are the usual stability tells; Dr.GRPO
  (arXiv:2503.20783) and GSPO (arXiv:2507.18071) motivate watching length and
  the optimisation signal.
* :func:`passk_base_vs_rl` plots base-vs-RL ``pass@k`` to large ``k`` — the
  **search-compression** test (arXiv:2504.13837): curves that cross mean RL is
  *sharpening* the base model's existing search; RL dominating at all ``k``
  (incl. ``k`` where base ``pass@k`` is ~0) means genuine capability expansion.

matplotlib is imported **locally** in every plotting function so importing this
module (and unit-testing the data helpers) never requires it; the functions
raise a clear error if it is missing. Each accepts data + an output path and
returns the path written.
"""
from __future__ import annotations

import json
from typing import Any

__all__ = [
    "load_step_logs",
    "extract_series",
    "plot_training_dashboard",
    "plot_passk_overlay",
    "passk_base_vs_rl",
    "STANDARD_PANELS",
]

#: (json key, human title, y-axis label) for the standard dashboard panels.
STANDARD_PANELS: tuple[tuple[str, str, str], ...] = (
    ("reward_mean", "Reward (mean)", "reward"),
    ("reward_std", "Within-group reward std", "std"),
    ("kl", "KL(policy || ref)", "KL"),
    ("entropy", "Policy entropy", "nats"),
    ("completion_length", "Completion length", "tokens"),
    ("grad_norm", "Gradient norm", "||g||"),
    ("frac_zero_advantage", "Fraction zero-advantage groups", "fraction"),
    ("pass_at_1", "pass@1 (eval)", "pass@1"),
)


def _require_mpl():  # noqa: ANN202 - returns the pyplot module
    """Import matplotlib (Agg backend) or raise a clear error."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - only without mpl
        raise RuntimeError(
            "analysis.curves plotting requires matplotlib; install it or use the "
            "extracted series (extract_series) directly"
        ) from exc
    return plt


def load_step_logs(jsonl_path: str) -> list[dict[str, Any]]:
    """Load per-step records from a JSONL file (skips blank/unparseable lines).

    Records are returned in file order and (stably) sorted by ``step`` when that
    key is present, so out-of-order log flushes still plot left-to-right.
    """
    rows: list[dict[str, Any]] = []
    with open(jsonl_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    if all("step" in r for r in rows) and rows:
        rows.sort(key=lambda r: r["step"])
    return rows


def extract_series(rows: list[dict[str, Any]], key: str) -> tuple[list[float], list[float]]:
    """Return ``(steps, values)`` for ``key``, over rows where it is present.

    ``steps`` uses the row's ``step`` field when present, else the row index.
    Rows missing ``key`` (or carrying a non-numeric value) are skipped, so a
    series that only starts being logged partway through still plots correctly.
    """
    steps: list[float] = []
    values: list[float] = []
    for i, row in enumerate(rows):
        if key not in row:
            continue
        val = row[key]
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            continue
        steps.append(float(row.get("step", i)))
        values.append(float(val))
    return steps, values


def plot_training_dashboard(
    rows_or_path: list[dict[str, Any]] | str,
    out_path: str,
    *,
    panels: tuple[tuple[str, str, str], ...] = STANDARD_PANELS,
    title: str = "Crucible RLVR training dashboard",
) -> str:
    """Plot the standard multi-panel RLVR dashboard to ``out_path``.

    ``rows_or_path`` is either the loaded step rows or a path to the JSONL log.
    Panels whose key never appears get an "(no data)" placeholder so the grid
    layout is stable across runs. Returns the path written.
    """
    plt = _require_mpl()
    rows = load_step_logs(rows_or_path) if isinstance(rows_or_path, str) else rows_or_path

    n = len(panels)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3 * nrows), squeeze=False)
    flat = [ax for row in axes for ax in row]

    for ax, (key, panel_title, ylabel) in zip(flat, panels, strict=False):
        xs, ys = extract_series(rows, key)
        if xs:
            ax.plot(xs, ys, marker=".", linewidth=1.2)
        else:
            ax.text(0.5, 0.5, "(no data)", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(panel_title)
        ax.set_xlabel("step")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    # Hide any unused axes.
    for ax in flat[n:]:
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_passk_overlay(
    rows_or_path: list[dict[str, Any]] | str,
    out_path: str,
    *,
    ks: tuple[int, ...] | None = None,
    title: str = "pass@1 / pass@k over training",
) -> str:
    """Overlay ``pass@1`` and ``pass@k`` (from each row's ``pass_at_k`` dict).

    Each row may carry ``pass_at_1`` and/or a ``pass_at_k`` dict ``{k: value}``.
    ``ks`` selects which ``k`` curves to overlay (default: the union of keys seen
    across rows). Returns the path written.
    """
    plt = _require_mpl()
    rows = load_step_logs(rows_or_path) if isinstance(rows_or_path, str) else rows_or_path

    fig, ax = plt.subplots(figsize=(9, 5))

    xs1, ys1 = extract_series(rows, "pass_at_1")
    if xs1:
        ax.plot(xs1, ys1, marker="o", label="pass@1")

    # Discover the k's present in pass_at_k dicts.
    seen_ks: set[int] = set()
    for r in rows:
        d = r.get("pass_at_k")
        if isinstance(d, dict):
            for k in d:
                try:
                    seen_ks.add(int(k))
                except (TypeError, ValueError):
                    continue
    chosen = sorted(seen_ks) if ks is None else [int(k) for k in ks]

    for k in chosen:
        xs: list[float] = []
        ys: list[float] = []
        for i, r in enumerate(rows):
            d = r.get("pass_at_k")
            if not isinstance(d, dict):
                continue
            val = d.get(str(k), d.get(k))
            if val is None:
                continue
            xs.append(float(r.get("step", i)))
            ys.append(float(val))
        if xs:
            ax.plot(xs, ys, marker=".", label=f"pass@{k}")

    ax.set_xlabel("step")
    ax.set_ylabel("pass rate")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def passk_base_vs_rl(
    base_curve: dict[int, float],
    rl_curve: dict[int, float],
    out_path: str,
    *,
    title: str = "pass@k: base vs RL (search-compression test)",
) -> str:
    """Plot base-vs-RL ``pass@k`` curves on a log-x axis (the C-critique test).

    Parameters
    ----------
    base_curve, rl_curve:
        ``{k: corpus_pass_at_k}`` for the base and RL-trained models (build these
        with :func:`eval.passk.passk_curve`, sweeping ``k`` to ~128–256).
    out_path:
        Where to write the figure.

    Interpretation (annotated on the plot): if the curves **cross** (RL higher at
    small ``k``, base catching up at large ``k``) RL is *sharpening* the base
    model's search; if RL **dominates at all ``k``** — especially where base
    ``pass@k`` ≈ 0 — RL has genuinely **expanded** capability (arXiv:2504.13837).
    Returns the path written.
    """
    plt = _require_mpl()
    fig, ax = plt.subplots(figsize=(9, 5))

    bks = sorted(base_curve)
    rks = sorted(rl_curve)
    if bks:
        ax.plot(bks, [base_curve[k] for k in bks], marker="o", label="base")
    if rks:
        ax.plot(rks, [rl_curve[k] for k in rks], marker="s", label="RL")

    all_ks = sorted(set(bks) | set(rks))
    if all_ks and min(all_ks) > 0:
        ax.set_xscale("log", base=2)
    ax.set_xlabel("k (samples)")
    ax.set_ylabel("pass@k")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
