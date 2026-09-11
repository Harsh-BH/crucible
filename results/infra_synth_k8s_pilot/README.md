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
own that training reliably fixes this class of error. The eval-axis change and
3-seed run below settle it.

## Eval-axis correction: greedy has no headroom, sampling does

Base-model comparison, same 15 test tasks, `results/infra_synth_eval/k8s_variance_check_t0.7.json`
(TRAIN split) vs a fresh TEST-split sampled measurement:

| axis | rate |
|---|---|
| greedy, test split | 14/15 = 93.3% |
| sampled t=0.7, TRAIN split (variance check) | mean reward 0.406 |
| sampled t=0.7, **TEST split** (`baseline_sampled_eval.json`, n=4/task) | pass@1 (mean) 43.3%, pass@4 (any-of-4) 100% |

The model writes valid k8s manifests 93% of the time greedily but only ~43% of
the time per-sample when sampled at t=0.7 (t=1.0 is worse still — variance
check mean reward 0.06). **The base model is not weak at this task, it is
fragile under sampling** — and GRPO trains on exactly that sampling
distribution (`num_generations=4`, `temperature=0.7`). So the held-out eval
below reports BOTH axes: greedy (comparable to round 1's number, expected to
barely move since it's near ceiling) and sampled t=0.7 n=4/task (the axis the
training signal actually operates on, with real headroom).

## 3 seeds, byte-identical config (`training/configs/infra_synth_k8s_6gb.yaml`, only `--seed` differs)

Command per seed: `PYTHONPATH=. .venv-train/bin/python -m training.run --config
training/configs/infra_synth_k8s_6gb.yaml --seed {0,1,2} --output-dir
results/infra_synth_k8s_pilot/seed{0,1,2}`, with
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` set. `.venv-train`: Python
3.13.13, `torch==2.13.0+cu130` (CUDA), `transformers==5.16.1`, `trl` +
`peft==0.20.0`. GPU: RTX 3050 Laptop, 6144 MiB. 40 steps each: seed0 14m13s
(21.3s/it), seed1 11m51s (17.8s/it), seed2 12m32s (18.8s/it).

**TRAIN-split reward, seen by the trainer** (`metrics.ndjson`, temperature=0.7,
sampled, num_generations=4), first-10 -> last-10 steps:

| seed | first10 | last10 | delta | mean `frac_reward_zero_std` |
|---|---|---|---|---|
| 0 | 0.3125 | 0.5500 | +0.2375 | 0.200 |
| 1 | 0.5125 | 0.8375 | +0.3250 | 0.313 |
| 2 | 0.4125 | 0.5500 | +0.1375 | 0.163 |
| **mean ± std** | | | **+0.233 ± 0.077** | |

**All 3 seeds rose** (contrast with the gsm8k m1 run: seed 0 +0.025, seed 2
−0.138 — mixed signs, no real trend). Groups stayed mostly non-degenerate
throughout (`frac_reward_zero_std` 0.16-0.31, well under the 0.75 the t=1.0
variance check predicted for a bad temperature choice). This is the reward the
TRAINER sees, on sampled TRAIN-split rollouts — a different claim from the
held-out pass rates below.

**Held-out pass rate, same 15 test tasks/harness as round 1**
(`scripts/infra_synth_k8s_posttrain_eval.py`; `posttrain_eval.json` = greedy
n=1).

*Superseded first pass, n=4/task* (`sampled_eval.json`): baseline pass@1 43.3%,
"any-of-4" 100%; seed0 38.3%/80.0%, seed1 60.0%/100%, seed2 50.0%/93.3%. At
n=k=4 "any-of-4" is just the empirical outcome of the 4 draws actually taken —
noisy, not the unbiased estimator — and it read as pass@4 *regressing* for 2 of
3 seeds. That reading did not survive higher resolution; see below.

**Corrected: n=8/task, `eval/passk.py`'s unbiased combinatorial pass@k
estimator** (`sampled_eval_n8.json` / `baseline_sampled_eval_n8.json`;
i.i.d. samples at fixed temperature=0.7 — the estimator's assumption holds
here, each `generate()` call is an independent draw with no shared state):

| k | baseline | seed0 | seed1 | seed2 | seed3 | mean ± std (4 seeds) | Δ vs baseline |
|---|---|---|---|---|---|---|---|
| 1 | 0.433 | 0.392 | 0.608 | 0.533 | 0.550 | 0.521 ± 0.092 | **+0.088** |
| 2 | 0.671 | 0.629 | 0.855 | 0.776 | 0.814 | 0.769 ± 0.099 | **+0.097** |
| 4 | 0.883 | 0.869 | 0.987 | 0.941 | 0.978 | 0.944 ± 0.054 | **+0.061** |
| 8 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 ± 0.000 | 0.000 (ceiling) |

Greedy: baseline 93.3% (14/15); all 3 seeds 100% (15/15) — one task flipped per
seed (see above), report as that, not as "100% pass rate."

**Retracting the n=4 "pass@4 regressed" reading.** At proper resolution the
curve shape is: **2 of 3 seeds (seed1, seed2) sit ABOVE baseline at every k**
(the "starts higher, flattens earlier" signature of a genuinely improved
policy, not a narrowed-but-net-worse one), and **seed0 sits AT OR SLIGHTLY
BELOW baseline at every k** (a small, roughly uniform decline, not a
lucky-escape-hatches-removed pattern). The corpus-mean curve is above baseline
at k=1,2,4 and ties it at k=8 (both already at ceiling). This is a real,
if seed-variable, effect — not the "GRPO traded diversity for consistency and
net-lost" story the noisier n=4 read suggested. The 3-seed spread (std 4.9-9.4pp
across k=1..4) is still large relative to the mean delta (4.9-8.2pp), which is
exactly why seed 3 was run next (below) rather than stopping here.

### seed 3 (added 2026-09-11)

seed 3 had been trained back with the others but its post-training eval was never run,
so it sat uncommitted and out of the table above. Run now with the same harness and the
same 15 test tasks; its six result files are committed alongside seeds 0-2.

It lands **above baseline at every k** (0.550 / 0.814 / 0.978 / 1.000) and above the
old 3-seed mean at k=1, 2 and 4, which moves the corpus-mean delta up at every k:
**+0.078 -> +0.088** at k=1, **+0.082 -> +0.097** at k=2, **+0.049 -> +0.061** at k=4.
Greedy is 15/15, same as the other three. Its trainer-side reward is the highest of the
four (0.750 mean vs seed2's 0.625).

**It does not resolve the variance concern, which is why it was run.** The spread is
essentially unchanged (k=1 std 0.090 -> 0.092, k=2 0.094 -> 0.099, k=4 0.049 -> 0.054),
so the delta-to-spread ratio improves only from roughly 0.9 to roughly 1.0 -- the effect
is still about the same size as the seed-to-seed spread. What changed is the sign
consistency: **3 of 4 seeds now sit above baseline at every k**, with seed0 still the
lone seed at or slightly below. A 4th seed narrows nothing on its own; it takes more
seeds, and seed 4 was never trained.

## The `save_model` bug is a bigger finding than the pilot itself

`training.run.train()` never called `trainer.save_model()`. With the
historical `save_steps=0` default, `GRPOConfig.save_strategy="no"` means every
completed GRPO run in this project's history — gsm8k included — **discarded
its own trained weights**. No post-training evaluation of ANY prior run was
ever possible; every claim about training "working" could only ever be checked
against the trainer's own logged reward, never against what the trained model
actually does on held-out data. The first pilot run of this seed (before the
fix existed) is a direct demonstration: it trained cleanly, its reward curve
(0.3125 -> 0.5625) is real, and its weights are gone — unrecoverable,
re-trained from scratch as seed0 above once the fix landed. Fixed now
(`training/run.py`, unconditional `trainer.save_model(cfg.output_dir)`).

## What's NOT in this directory

`adapter_model.safetensors`, `tokenizer.json`, `training_args.bin`, and
`completions/*.parquet` are produced by the run but not committed
(`*.safetensors`/`*.parquet` are repo-gitignored, same as
`results/m1_gsm8k_6gb/`'s parquet dumps) — regenerate with the command above.
`summary.json` / `metrics.ndjson` (renamed from TRL's `metrics.jsonl`, same
reason as the gsm8k run) / `posttrain_eval.json` are committed and are what
every number above traces to.
