# Completion-cap ablation: does 62.5%-clipped completions explain the flat reward?

Motivation: `results/m1_gsm8k_6gb/README.md` measured 62.5% of completions hitting the
256-token cap on the final step of every 0.5B seed, and a truncated completion scores 0 on
both reward terms regardless of reasoning quality — so the flat/declining reward measured
there could be an artifact of the cap, not evidence the method doesn't work.

## Attempt 1: max_completion_length=512, everything else identical to m1_gsm8k_6gb.yaml

`training/configs/m1_gsm8k_6gb_cap512.yaml` — same `env=gsm8k`, `model=Qwen2.5-0.5B-Instruct`,
`lora_rank=8`, `num_generations=4`, `per_device_train_batch_size=4`,
`gradient_accumulation_steps=2` as the round-3 baseline, only `max_completion_length` raised
256 -> 512.

**Result: OOMs on the RTX 3050 6 GB, at step 0, during the backward pass.**
`results/m1_gsm8k_6gb_cap512/seed0/stdout.txt`:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.16 GiB. GPU 0 has a total
capacity of 5.67 GiB of which 444.12 MiB is free. Including non-PyTorch memory, this process
has 5.14 GiB memory in use. Of the allocated memory 4.65 GiB is allocated by PyTorch, and
386.52 MiB is reserved by PyTorch but unallocated.
```

`results/m1_gsm8k_6gb_cap512/gpu_mem_seed0.txt` (3 s polling) shows memory climbing from
model-load (~2.3-2.4 GB) to 5150-5360 MiB just before the crash — the crash itself happens
inside a CUDA allocation that never gets a clean `nvidia-smi` sample. Real, citable finding:
**at this group/batch size, this GPU cannot hold 512-token completions for a 0.5B model with
LoRA+bf16+gradient-checkpointing.**

## Attempt 2: same cap, half the group and batch size (fits, one seed only)

`training/configs/m1_gsm8k_6gb_cap512_g2.yaml` — `num_generations` 4->2,
`per_device_train_batch_size` 4->2, `gradient_accumulation_steps` 2->4 (holds the effective
batch size at 8, same as before), `max_completion_length=512` unchanged. This is NOT
"everything else identical" any more — flagging that plainly, it is a genuinely different
config, run for **one seed only** (seed 0), not the 3-seed standard used elsewhere in this
audit. GPU/time budget was prioritized toward the Qwen3-1.7B feasibility test instead of a
full 3-seed replication here; re-run seeds 1-2 if this ablation needs to stand on its own.

**Result: fits. 40/40 steps, 27.74 s/step (~2x the 256-cap baseline's 14.7-15.0 s/step),
18:29 wall-clock.** `completions/clipped_ratio` is **0.0** (down from 0.625 at the 256 cap) —
confirms the cap really was truncating completions before, and raising it does stop that.
`completions/mean_length` 285.75 tokens (well under the new 512 cap, so generations are now
terminating naturally rather than being cut off).

`rewards/gsm8k_reward/mean`: first-10-step avg **0.4375**, last-10-step avg **0.4375**,
**delta 0.000** — exactly flat. (first-5 0.325 -> last-5 0.350, also no meaningful move.)

## Honest verdict

The completion cap was real (62.5% clipped -> 0% clipped after doubling it) but **it was not
suppressing a rising-reward signal that the cap was hiding** — removing it produces an exactly
flat trend, if anything slightly flatter than the noisy 256-cap baseline's per-seed deltas
(+0.025 / -0.062 / -0.138). The absolute reward level here (~0.44) is higher than the 256-cap
runs' first-10 averages (0.26-0.45), consistent with untruncated completions scoring better on
average — but "scores higher on average" and "reward rises over training" are different
claims, and only the second one matters for the CV's "GRPO reward" claim. This makes the
round-3 null result **harder to explain away**, not weaker: the obvious confound (truncation)
has been tested and ruled out, on real measured data, in one seed.
