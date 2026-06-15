# training/ — M1/M2 GRPO trainers

Two trainer paths share the configs in `configs/`. Heavy imports
(`trl`/`torch`/`verifiers`) are **lazy** — the helpers (`data.py`, `rewards.py`,
`seeds.py`) and the `RunConfig` dataclass stay torch-free so they unit-test
without a GPU.

## Files

- `run.py` — self-contained **TRL GRPO** driver. `RunConfig` is the single
  source of truth; precedence is dataclass defaults < `--config` YAML < CLI flags.
- `data.py` · `rewards.py` · `seeds.py` — dataset builders, reward wiring, and
  the ≥3-seed aggregation harness (mean ± 95% CI).
- `configs/` — prime-rl TOML (`m1_gsm8k.toml`, `m1_reverse_text.toml`,
  `m2_infra_sentinel.toml`) + TRL YAML (`m1_repro.yaml`, `m2_sentinel.yaml`).

## Two paths

```bash
# prime-rl (primary stack), single-GPU colocated:
uv run rl @ training/configs/m1_gsm8k.toml \
  --trainer-gpu-ids 0 --inference-gpu-ids 0 --inference.gpu-memory-utilization 0.5
# TRL baseline (hackable; exposes the ablation knobs), one seed:
python training/run.py --env gsm8k --model Qwen/Qwen3-1.7B --num-generations 8 --seed 0
```

## GRPO-variant knob mapping (in `run.py`; see `training/README.md`)

- `--epsilon-high` → **DAPO** clip-higher
- `--scale-rewards` (default OFF) → off = **Dr.GRPO** (no within-group std norm)
- `--importance-sampling-level sequence` → **GSPO** (TRL only)
- `--loss-type {grpo,bnpo,dr_grpo}`

## Notes

- **Single GPU** is the default target (Qwen3-1.7B + LoRA, ~24–40 GB). Scales up
  via config. `python training/run.py --smoke` checks the code path without a GPU.
- **M1** = reward rises, stable across ≥3 seeds. **M2** = route reward through
  Sentinel (`--verifier-backend sentinel --sentinel-base-url ...`). See `docs/ROADMAP.md`.
