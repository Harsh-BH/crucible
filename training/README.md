# training

GRPO training for Crucible. Two interchangeable trainer paths share the same
torch-free building blocks (`data.py`, `rewards.py`, `seeds.py`):

| Path | Driver | When to use |
| --- | --- | --- |
| **prime-rl** (primary) | `uv run rl @ <toml>` | scale-out runs; multi-GPU; the configs in [`configs/*.toml`](./configs/) |
| **TRL** (hackable baseline) | `python training/run.py …` | a self-contained, editable GRPO loop in this repo; the configs in [`configs/*.yaml`](./configs/) |

Heavy / GPU deps (`verifiers`, `vllm`, `wandb`, `datasets`, `torch`, `trl`,
`peft`) are **not** in the core install — they come from the root `--extra
train` group. The modules here import torch-free (heavy imports are lazy,
inside functions) so the unit tests run on CPU with only stdlib + `pyyaml` +
`datasets`.

> **Honest scope note.** The configs and code are *runnable*, but real training
> needs a GPU. On CPU you can only check the **code path**:
> `python training/run.py --env gsm8k --smoke` swaps in a tiny model, 2 steps,
> a 4-row dataset and a group of 2 — it proves dataset → rewards →
> `GRPOTrainer.train()` wires up, **not** that reward rises. Reward-curve claims
> (the M1 DoD below) require a GPU run.

---

## prime-rl (primary stack)

```bash
# M1: the known GRPO result on GSM8K (Qwen3-1.7B + LoRA), single GPU:
uv run rl @ training/configs/m1_gsm8k.toml \
    --trainer-gpu-ids 0 --inference-gpu-ids 0 \
    --inference.gpu-memory-utilization 0.5 --inference.model.max-model-len 2048

# Cheap loop smoke test first (tiny SFT'd model, reverse-text toy task):
uv run rl @ training/configs/m1_reverse_text.toml --trainer-gpu-ids 0 --inference-gpu-ids 0
```

Configs:

- **`m1_reverse_text.toml`** — smallest end-to-end loop; confirms orchestrator +
  trainer + inference are wired before spending GPU on the real run.
- **`m1_gsm8k.toml`** — the M1 reproduction: `Qwen/Qwen3-1.7B` + LoRA on GSM8K
  (`math-env` / `gsm8k`), single-GPU-friendly. Online difficulty filtering on.
- **`m2_infra_sentinel.toml`** — M2: same recipe, env swapped to `infra-synth`
  with `verifier_backend="sentinel"`. Requires the env installed into the
  prime-rl workspace **and** a running Sentinel (see M2 below).

## TRL (`training/run.py`)

```bash
# Equivalent M1 run, single seed, from the YAML config (CLI flags override):
python training/run.py --config training/configs/m1_repro.yaml --seed 0

# Or fully from flags:
python training/run.py --env gsm8k --model Qwen/Qwen3-1.7B \
    --num-generations 8 --beta 0.0 --lr 1e-6 --max-steps 100 --seed 0
```

`run.py` reads a YAML config of **flat keys named exactly like the CLI flags /
`RunConfig` fields** (dashes → underscores). Precedence is dataclass defaults <
`--config` YAML < CLI flags actually passed. A `seeds:` list in the YAML is
ignored by `run.py` (a single run uses one seed) and consumed by `seeds.py`.

Configs: **`m1_repro.yaml`** (GSM8K M1) and **`m2_sentinel.yaml`** (infra_synth,
`verifier_backend: sentinel`).

---

## M1 — definition of done

Reproduce a **known** GRPO result so the harness is trusted before we point it
at our own env:

1. **Reward rises** over training on GSM8K (Qwen3-1.7B + LoRA).
2. **Stable across ≥3 seeds.** Launch seeds `{0,1,2}` and report `mean ± 95% CI`:

   ```python
   from training.seeds import launch_seeds
   res = launch_seeds(
       env="gsm8k", seeds=(0, 1, 2), output_dir="outputs/m1",
       extra_args=["--config", "training/configs/m1_repro.yaml"],
   )
   # res["aggregate"] maps each logged metric (the reward key name is set by the
   # TRL version, e.g. "reward" / "rewards/<func>") to its summary stats:
   #   {mean, std, ci95, ci_low, ci_high, n}
   ```

   `seeds.py` is pure stdlib statistics: `mean_std_ci` /
   `aggregate_seed_metrics` compute the mean, **sample** std (ddof=1) and a
   normal-approx 95% CI, **`CI = mean ± 1.96·std/√n`**. `summarize_runs` reduces
   the per-step `metrics.jsonl` logs `run.py` writes (last value per key) and
   aggregates across seeds.

### Single-GPU tuning

- `--inference.gpu-memory-utilization 0.5` and
  `--inference.model.max-model-len 2048` (prime-rl) keep the vLLM rollout server
  and the trainer on one GPU.
- LoRA (`rank=8, alpha=16`) instead of full fine-tuning.
- Shrink `batch_size` / `group_size` (`--num-generations` in TRL) and
  `max_completion_tokens` if you OOM. The TRL path also exposes
  `--gradient-accumulation-steps` to trade memory for step time.
- Smoke first: prime-rl `m1_reverse_text.toml`; TRL `--smoke`.

---

## Algorithm → flag mapping (TRL `run.py`)

The defaults bake in the modern GRPO recipe (KL off, no reward scaling); flip
individual flags to recover named variants:

| Variant | Flags |
| --- | --- |
| **DAPO** clip-higher | `--epsilon-high 0.28` (decouples the upper PPO clip) |
| **Dr.GRPO** | `--no-scale-rewards` (**default**) — no within-group std normalisation, removes the length/difficulty bias |
| **GSPO** | `--importance-sampling-level sequence` — sequence-level (not token-level) importance ratio |
| KL regularisation | `--beta 0.04` (default `0.0` = KL off) |
| loss aggregation | `--loss-type {grpo,bnpo,dr_grpo}` |

These compose, e.g. `--epsilon-high 0.28 --no-scale-rewards
--importance-sampling-level sequence`. Unknown/forward-compat knobs in YAML are
forwarded to `GRPOConfig` and silently dropped (with a warning) if the installed
TRL doesn't declare them.

---

## M2 — reward via Sentinel

Swap the **reward backend** to route each rollout's Dockerfile through the
hardened Sentinel sandbox (build + smoke probe):

```bash
# TRL:
python training/run.py --env infra_synth \
    --verifier-backend sentinel --sentinel-base-url http://localhost:8080 \
    --model Qwen/Qwen3-1.7B --num-generations 8 --beta 0.0 --max-steps 100
# (or: --config training/configs/m2_sentinel.yaml)

# prime-rl:
uv run rl @ training/configs/m2_infra_sentinel.toml --trainer-gpu-ids 0 --inference-gpu-ids 0
```

`rewards.make_infra_synth_reward` parses the Dockerfile, builds a `VerifySpec`
from each row's `info`, runs the async verifier over the batch concurrently, and
shapes each `VerifyResult` with `verifier.shape_reward` (default
`build_weight=0.3`, `smoke_weight=0.7` → build-only `0.3`, build+smoke `1.0`).
Other backends: `static` (in-process, default — CI/code-path), `local-docker`
(genuine local build). **`sentinel` requires a running Sentinel at the base URL**
(`make up` in the Sentinel repo, `:8080`) and the `infra_synth` env package
installed (`pip install -e environments/infra_synth`, or `prime env install` for
the prime-rl workspace).

---

## Metrics to watch (failure modes)

- **Entropy collapse** — completion entropy → 0: the policy stops exploring;
  reward plateaus and the run is effectively dead. Lower LR / add a little KL.
- **Within-group std → 0** — when all rollouts in a group score the same,
  GRPO's advantage is zero and there is **no gradient** (the motivation for
  partial-credit reward shaping and online difficulty filtering).
- **KL divergence** — spiking KL from the reference signals the policy running
  away (reward hacking / degenerate text); raise `--beta` or clip tighter.
- Also: mean reward (should rise), completion length (watch for runaway length),
  grad norm, and `clip_ratio` (how often the PPO clip binds).
