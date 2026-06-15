# analysis

Reward-hacking study (contribution C3) and training diagnostics. See
[`../docs/DESIGN.md`](../docs/DESIGN.md) for the full study design.

- **`reward_hacking.py`** — C3 scaffolding. The 6-category reward-hacking
  taxonomy (`CATEGORIES` + `classify_hack`), an ImpossibleBench-style
  `cheating_rate`, and `compare_weak_vs_hardened(weak, hardened)` (cheating rate,
  per-category counts, reduction) for the **weak (`local-py`) vs hardened
  (`sentinel`)** comparison. `load_rollouts(jsonl)` reads logged
  rollouts/`HackFlags`; `plot_taxonomy(...)` charts it (matplotlib, local
  import). Citations: ImpossibleBench (arXiv:2510.20270), pass@k
  (arXiv:2107.03374). Note: Sentinel does not yet surface seccomp/network
  signals (seccomp currently disabled), so taxonomy category 6 is under-counted.
- **`curves.py`** — the standard GRPO/RLVR dashboard from per-step JSONL logs
  (reward mean, within-group reward std, KL, entropy, completion length, grad
  norm, fraction zero-advantage, pass@1/pass@k overlays) and
  `passk_base_vs_rl(...)` for the search-compression crossover test
  (arXiv:2504.13837). The expected per-step log schema is documented in the
  module docstring.

matplotlib is imported **locally** in every plotting function, so importing
these modules (and the unit tests) never requires it; the plotters raise a clear
error if it is missing. Stdlib + `verifier` otherwise. Tests:
`tests/test_reward_hacking.py`.
