# training (WIP)

The trainer for Crucible is [`prime-rl`](https://github.com/PrimeIntellect-ai/prime-rl)
/ TRL-style GRPO, driven by the `verifiers` library and a vLLM rollout server.
Run configurations (e.g. the M1 reproduction on Qwen3-1.7B and the M2 run that
routes reward through Sentinel) land in [`configs/`](./configs/) as they are
built. Heavy/GPU dependencies (`verifiers`, `vllm`, `wandb`, `datasets`) are
installed via the root `--extra train` group, not the core install.

**Status:** scaffolding only — training entrypoints and configs are added by a
later subagent.
