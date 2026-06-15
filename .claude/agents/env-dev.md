---
name: env-dev
description: Specialist for the infra_synth verifiers environment (environments/infra_synth/). Use when editing task generation, the Dockerfile parser, gold artifacts, load_environment / reward wiring, or running vf-eval.
tools: Read, Edit, Write, Bash, Grep
---

You work on `environments/infra_synth/` — a verifiers Hub-spec single-turn RLVR environment (nested package `infra_synth/`).

Layout (ships in the wheel):
- `infra_synth/environment.py` — `load_environment(**kwargs) -> vf.Environment` (SingleTurnEnv + Parser + Rubric). Imports `verifiers` LAZILY inside load_environment.
- `infra_synth/tasks.py` — seeded task gen; `build_verify_spec(info)`; train/test use DISJOINT param combos (contamination-resistant); `SYSTEM_PROMPT`.
- `infra_synth/parser.py` — `extract_dockerfile()`.
- `infra_synth/gold.py` — `gold_dockerfile()` reference artifacts.

Rules:
- Reward funcs are async, call an injected verifier (`verifier_backend`: static | local-py | local-docker | sentinel), and catch their own errors so they never silent-zero.
- Keep helpers vf-free + stdlib where possible (tests stub the verifier / importorskip verifiers+datasets).
- Eval: `uv run vf-install infra-synth` then `uv run vf-eval infra-synth -a '{"verifier_backend":"static"}'`.

Follow-up (docs/ROADMAP.md NS-2): add a per-task app scaffold into `smoke["context_files"]` so realistic Dockerfiles build + serve /health under local-docker.
