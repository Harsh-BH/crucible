# M1 loop-sanity on gsm8k, Qwen2.5-0.5B, RTX 3050 6 GB — 3-seed real run

This directory is the missing evidence for the "GSM8K reward 0.525 -> 0.662" claim that
previously existed only as prose in `docs/ROADMAP.md` (no committed config, no raw log,
task name and step count disagreeing with the one config that existed). It replaces that
prose with an actual run, against `training/configs/m1_gsm8k_6gb.yaml` (new, committed
alongside this directory), following `training/README.md`'s M1 definition-of-done
(>=3 seeds, report mean/spread, not a single curve).

Command, per seed (`N` in `{0,1,2}`):

```
PYTHONPATH=. .venv-train/bin/python -m training.run \
    --config training/configs/m1_gsm8k_6gb.yaml --seed N \
    --output-dir results/m1_gsm8k_6gb/seedN
```

`.venv-train` (not committed, see `.gitignore`): Python 3.13.13, `torch==2.13.0` (CUDA),
`transformers==5.16.1`, `trl==1.12.0`, `peft==0.20.0`, `accelerate==1.14.0`,
`datasets==5.0.1`. GPU: RTX 3050 Laptop, 6144 MiB.

## What's in each `seedN/`

- `summary.json` — final-step metrics + the exact `RunConfig` used.
- `metrics.ndjson` — one JSON object per logged step (41 rows: step 0 = pre-train baseline
  bookkeeping line HF/TRL emits, steps 1-40 = training). Renamed from TRL's `metrics.jsonl`
  only because `*.jsonl` is repo-wide gitignored; content is identical.
- `stdout.txt` — full captured stdout (includes TRL/rich's per-step sample-completion
  tables — this is the actual model output at each step, not summarized).
- `completions/*.parquet` — TRL's per-step raw completion dumps. Gitignored (`*.parquet`);
  not committed. Regenerate by re-running the command above.

## Measured results (real, this run, 2026-08-31)

| seed | step_time (s) | train_runtime (s) | gsm8k_reward first10 -> last10 | delta | final combined reward |
|---|---|---|---|---|---|
| 0 | 14.77 | 588.9 | 0.263 -> 0.287 | **+0.025** | 0.375 |
| 1 | 14.95 | 597.1 | 0.388 -> 0.325 | **-0.062** | 0.250 |
| 2 | 14.64 | 603.9 | 0.450 -> 0.312 | **-0.138** | 0.500 |

Mean per-seed delta (last-10-step avg minus first-10-step avg of `rewards/gsm8k_reward/mean`,
the correctness proxy): **-0.058**, stdev 0.066. Two of three seeds end LOWER than they
started; the one positive seed (+0.025) is inside the noise band (per-step std within a run
is 0.23-0.26 against a step mean of ~0.3). `frac_reward_zero_std` (fraction of steps where
every rollout in the group scored identically, i.e. GRPO's advantage is zero and there is no
gradient that step — the exact failure mode `training/README.md`'s "Metrics to watch" section
names) is 0.5 / 0.0 / 0.0 for seeds 0/1/2.

Peak GPU memory: measured directly (3 s polling) for seed 0 only: **4856 MiB / 6144 MiB**
(~4.74 of 6 GB), not the previously-claimed "5.4 of 6 GB" — that number does not reproduce
here. Seeds 1-2 used an identical model/batch/LoRA config, so peak VRAM is expected to be
close, but was **not independently measured** for those two runs (no live poll was running) —
do not cite a peak-VRAM number for seed 1 or 2 specifically.

`completions/clipped_ratio` (fraction of completions that hit the 256-token cap rather than
terminating naturally) is 0.625 on the final step of all three seeds — a majority of
completions are being cut off before the model necessarily finishes its reasoning, which is
itself a plausible partial explanation for the flat/declining reward (a truncated completion
before `\boxed{...}` scores 0 on both `format_reward` and `gsm8k_reward` regardless of whether
the reasoning was on track).

## Honest verdict

**Reward does not rise stably across seeds at this scale.** The correctness reward
(`rewards/gsm8k_reward/mean`) is flat-to-declining across all three seeds within run, and the
cross-seed mean trend is slightly negative. This is a real, measured, negative result at
Qwen2.5-0.5B / 40 steps / group=4 / LoRA rank 8 — it does not corroborate the previously
CV-stated "0.525 -> 0.662" curve (which was never traceable to any committed config or raw
log; see the round-2 findings above this file's directory in the audit trail). It also does
not meet `training/README.md`'s own M1 bar ("reward rises, stable across >=3 seeds"), even at
this much smaller-than-M1 scale. This is consistent with — not contradicting — the repo's own
prior "Partially exercised... noisy, single seed" framing: it turns out the noise was real,
and a single seed was not enough to see that the mean trend is flat-to-negative.
