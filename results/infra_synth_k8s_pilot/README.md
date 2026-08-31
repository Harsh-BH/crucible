# GRPO on infra_synth/k8s — the actual experiment, round 2

Round 1 (`results/infra_synth_eval/`) measured a pass rate with no training.
This directory is the experiment infra_synth was built to answer: does GRPO
reward rise on the project's OWN task, and does that translate into a real
held-out pass-rate improvement? `results/m1_gsm8k_6gb/` answered the analogous
question for the gsm8k reproduction (reward flat/declining); this is the
infra_synth answer.

## Why k8s, why temperature=0.7

GRPO needs reward variance WITHIN a group (same prompt, `num_generations`
samples) or the advantage is exactly 0 and there is no gradient. Round-1's
per-kind pass rates (before the kubeconform-cache correction below) suggested
k8s was the only kind with headroom; `scripts/infra_synth_k8s_variance_check.py`
checked that directly on the TRAIN split with real GRPO-style sampling
(`do_sample=True`, NOT the greedy eval decoding):

| temperature | frac_reward_zero_std | mean reward |
|---|---|---|
| 1.0 (the gsm8k config's default) | 0.75 (6/8 groups all-zero) | 0.06 |
| 0.7 | **0.0** (0/8) | 0.41 |

So the pilot config (`training/configs/infra_synth_k8s_6gb.yaml`) uses
`temperature: 0.7`, a deliberate, measured deviation from the gsm8k default.

## Two real bugs found running this (both fixed, both committed)

1. **`training.run.train()` never saved a model.** With the historical
   `save_steps=0` default, `GRPOConfig.save_strategy="no"` means a *completed*
   run left no checkpoint at all — no way to ever reload the trained weights
   for a downstream eval. Fixed: unconditional `trainer.save_model(cfg.output_dir)`
   at the end of `train()`.
2. **OOM on this 6 GB card at `max_completion_length` values a short smoke test
   didn't catch**, because completion length (and therefore peak activation
   memory) varies step to step: 768 OOM'd at step 3, 512 (+
   `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`) OOM'd at step 6. 384
   survived a 10-step burn-in and the full 40-step run.

## CORRECTION: the round-1 k8s baseline does not reproduce

`results/infra_synth_eval/README.md` now carries the full correction. Short
version: `LocalK8sVerifier` had no `-cache` for kubeconform before this round,
so schema-fetch network flakiness was silently scoring some valid manifests as
failures — not just the 48s/call slowness reported in round 1, but the 53.3%
pass-rate number itself. Re-running the *unmodified* round-1 script after the
cache fix, same 15 test-split tasks: **14/15 (93.3%)**.

This matters for the experiment: the base model was already close to ceiling
on k8s once measured correctly, so GRPO has very little room to show a
pass-rate *improvement* (max possible gain: +6.7pp to 100%). The TRAIN-split
sampled-reward result below is unaffected by this correction (measured after
the cache fix already); the held-out PASS RATE comparison below is the number
this correction actually constrains.

## Pilot: seed 0, 40 steps

Run: `PYTHONPATH=. .venv-train/bin/python -m training.run --config
training/configs/infra_synth_k8s_6gb.yaml --seed 0 --output-dir
results/infra_synth_k8s_pilot/seed0` (with
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` set).

`.venv-train`: Python 3.13.13, `torch==2.13.0+cu130` (CUDA), `transformers==5.16.1`,
`trl` + `peft==0.20.0`. GPU: RTX 3050 Laptop, 6144 MiB. ~21s/step, 40 steps in
14m13s.

**TRAIN-split reward, seen by the trainer** (`metrics.ndjson`, temperature=0.7,
sampled, num_generations=4): mean reward first-10 steps -> last-10 steps
**0.3125 -> 0.55** (reproduced twice: an earlier run before the save-model fix
existed hit 0.3125 -> 0.5625 on identical config/seed; not committed since it
predates the fix and has no adapter). `frac_reward_zero_std` averaged 0.2 over
the run (mostly non-degenerate groups, matching the variance-check prediction).
This is the reward the TRAINER sees — a different claim from the held-out pass
rate below (the gsm8k runs showed exactly this kind of gap: total reward rose
while a format term moved and correctness stayed flat. Here there is no
separate format term for k8s — `shape_reward` is effectively binary since
`LocalK8sVerifier` sets `build_ok == smoke_ok` always — so there's less room
for that specific failure mode, but the TRAIN/sampled vs TEST/greedy axes are
still distinct claims).

**Held-out PASS RATE, test split, same 15 tasks/same harness as round 1**
(`posttrain_eval.json`, greedy n=1, `scripts/infra_synth_k8s_posttrain_eval.py`):

| | before (base model) | after (seed 0 adapter) |
|---|---|---|
| pass rate | 14/15 (93.3%) | **15/15 (100%)** |

Exactly one task flipped: `k8s-test-0-0021-python-flask-postgres-9000`. Before,
the generated manifest's `livenessProbe.httpGet` omitted the required `port:`
field (kubeconform-strict rejects it: `HTTPGetAction` requires `port`); after
training, the model added `port: 9000` to that block. A legible, genuine
schema-correctness fix, not a degenerate trick (both before/after manifests are
structurally the same Deployment+Service pair; diff was inspected by hand).

**Caveat on N: this is a 1-in-15 flip.** With the baseline already at 93.3%,
there is exactly one task per seed that CAN flip, and greedy decoding is
deterministic (no sampling noise to average out) — so "14/15 -> 15/15" is real
and reproducible for this seed/config, but it is not strong evidence on its
own that training reliably fixes this class of error; that needs the
multi-seed run this pilot was gating.

## What's NOT in this directory

`adapter_model.safetensors`, `tokenizer.json`, `training_args.bin`, and
`completions/*.parquet` are produced by the run but not committed
(`*.safetensors`/`*.parquet` are repo-gitignored, same as
`results/m1_gsm8k_6gb/`'s parquet dumps) — regenerate with the command above.
`summary.json` / `metrics.ndjson` (renamed from TRL's `metrics.jsonl`, same
reason as the gsm8k run) / `posttrain_eval.json` are committed and are what
every number above traces to.
