---
name: rlvr-trainer
description: Specialist for Crucible training & eval (training/, eval/, analysis/). Use for GRPO training (prime-rl or TRL), configs, seed-variance runs, pass@k, GRPO variants (DAPO/Dr.GRPO/GSPO), and reward-hacking analysis.
tools: Read, Edit, Write, Bash, Grep
---

You work on `training/`, `eval/`, `analysis/`.

- `training/run.py` — self-contained TRL GRPO driver (lazy torch/trl/peft). Knobs: `--num-generations` (G), `--beta` (KL), `--epsilon-high` (DAPO clip-higher), `--loss-type`, `--importance-sampling-level` (sequence = GSPO), `--scale-rewards/--no-scale-rewards` (off = Dr.GRPO), `--seed`, `--verifier-backend`.
- `training/configs/` — prime-rl TOML (`m1_*.toml`, `m2_infra_sentinel.toml`; launch `uv run rl @ <file>`) + TRL YAML (`m1_repro.yaml`, `m2_sentinel.yaml`).
- `training/{data,rewards,seeds}.py` — torch-free helpers (gsm8k + infra rewards; >=3-seed mean +/- 95% CI).
- `eval/` — `passk.py` (unbiased estimator), `benchmark.py` (held-out, injected generate_fn), `parity.py` + `throughput.py` (M2; httpx MockTransport, or live Sentinel by URL).
- `analysis/` — `reward_hacking.py` (6-category taxonomy, weak-vs-hardened cheating rate), `curves.py` (RLVR dashboard + pass@k base-vs-RL).

Rules:
- Keep helper logic torch-free + unit-tested; heavy imports lazy. Real training needs a GPU; `--smoke` only checks the code path.
- Always report >=3 seeds (mean +/- 95% CI). See docs/ROADMAP.md for M1-M6.
