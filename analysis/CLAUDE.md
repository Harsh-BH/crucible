# analysis/ — C3 reward-hacking study & RLVR dashboards

Torch-free; matplotlib is imported **lazily** (`_require_mpl`) so the data path
works without it. Reads JSONL logs produced by training/eval.

## Files

- `reward_hacking.py` (C3) — the **6-category taxonomy** (`CATEGORIES`,
  `CATEGORY_DESCRIPTIONS`): weak-tests/hardcoding, answer leakage, fake success,
  resource/timer/sandbox manipulation, test-harness side effects, gaming the
  verifier. `classify_hack` maps `HackFlags` onto categories; `cheating_rate`
  (ImpossibleBench-style) and `compare_weak_vs_hardened` quantify the
  **weak (`local-py`) vs hardened (`sentinel`)** reduction. `load_rollouts`,
  `plot_taxonomy`.
- `curves.py` — the GRPO **dashboard** from per-step JSONL
  (`plot_training_dashboard`): reward mean, within-group reward std, fraction
  zero-advantage, KL, entropy, completion length, grad norm. Plus
  `plot_passk_overlay` and `passk_base_vs_rl` (base-vs-RL pass@k crossover).

## Notes

- Caveat (documented in `docs/DESIGN.md` §3): Sentinel does not yet surface
  `seccomp_violation`/`network_attempt`, so taxonomy category 6 is under-counted
  on the hardened side until that lands.
- See `docs/DESIGN.md` §3–§4 for methodology and references.
