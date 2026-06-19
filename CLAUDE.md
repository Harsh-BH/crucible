# Crucible — Project Guide

**Crucible is a security-hardened RLVR (Reinforcement Learning with Verifiable
Rewards) environment framework.** A small code model emits infrastructure-as-code
(Dockerfile first) and each rollout is graded by *actually building and
smoke-testing* the artifact inside a sandboxed verifier — not by a string match.
The reward comes from executing untrusted model output, so the project also
studies what happens when the policy learns to attack the grader (the C3
weak-vs-hardened reward-hacking study).

## Repository layout

This is a **uv workspace**. The root is the `crucible` dev package
(`training/`, `eval/`, `analysis/`); `verifier/` and `environments/infra_synth/`
are separate distributions wired in.

```
RLF-VRTP/
├── verifier/                  dist `crucible-verifier` (nested pkg; pyproject one level up)
│   └── verifier/              import `verifier`
│       ├── types.py           FROZEN CONTRACTS — do not edit (see below)
│       ├── backends.py        StaticVerifier / LocalPyVerifier / LocalDockerVerifier / get_verifier()
│       ├── sentinel_client.py async SentinelClient + SentinelVerifier (Sentinel API :8080)
│       ├── reward.py          shape_reward(), result_to_metrics()
│       └── smoke/checks.py    Dockerfile checks + build_python_harness()
├── environments/infra_synth/  dist `infra-synth`, import `infra_synth` (nested pkg)
│   ├── pyproject.toml          verifiers Hub spec (+ [tool.verifiers.eval])
│   └── infra_synth/            __init__.py · environment.py (load_environment) · tasks.py · parser.py · gold.py · scaffold.py
├── training/                  run.py (TRL GRPO) · data.py · rewards.py · seeds.py · configs/
├── eval/                      passk.py · benchmark.py · parity.py · throughput.py
├── analysis/                  reward_hacking.py (C3 taxonomy) · curves.py (RLVR dashboard)
├── docs/                      DESIGN.md · ROADMAP.md
└── tests/                     249 tests (verifier · env · training · eval · analysis)
```

Each package below has its own `CLAUDE.md` with package-specific detail; Claude
Code loads it on demand when you work in that subtree.

## Commands

| Task | Command |
| --- | --- |
| Sync deps (core + dev) | `uv sync --extra dev` |
| Run all tests | `uv run pytest -q` (249 pass, 1 skipped — needs an HF download) |
| Lint | `uv run ruff check .` |
| Types | `uv run mypy verifier training eval` |
| Train — TRL baseline | `python training/run.py --env gsm8k --model Qwen/Qwen3-1.7B --num-generations 8 --seed 0` |
| Train — prime-rl (primary) | `uv run rl @ training/configs/m1_gsm8k.toml --trainer-gpu-ids 0 --inference-gpu-ids 0 --inference.gpu-memory-utilization 0.5` |
| Install Hub env | `uv run vf-install infra-synth` |
| Eval the env (static) | `uv run vf-eval infra-synth -a '{"verifier_backend":"static"}'` |
| M2 parity (no GPU) | `python -m eval.parity --base-url http://localhost:8080` |
| M2 throughput (mock) | `python -m eval.throughput --mock` |

> The training stack (`uv sync --extra train`: verifiers/vllm/wandb/datasets) is
> GPU-oriented and will not resolve cleanly on a CPU-only box. Core + dev is
> CPU-only and light.

## Conventions

- **Python ≥ 3.11** (`requires-python = ">=3.11,<3.14"`).
- **ruff**, line-length **100**.
- **Torch-free helpers + lazy heavy imports.** `verifier/`, `eval/`, and
  `analysis/` import with only `httpx` + stdlib (and matplotlib lazily). Heavy
  deps (`verifiers`/`trl`/`torch`/`vllm`) are imported *inside functions* so the
  test suite stays torch-free. Keep it that way.
- **Reward functions are async** (the `Verifier.verify` protocol and env reward
  funcs are coroutines).
- **≥3 seeds, config-driven.** Every run is seeded; report mean ± 95% CI
  (`training/seeds.py`, `eval.benchmark.evaluate_multi_seed`).

## Frozen contract (read before editing the verifier layer)

`verifier/verifier/types.py` is **FROZEN**. Backends, the environment, reward
shaping, training, and eval all depend on its exact names and fields: `Verifier`
(protocol), `VerifyResult`, `VerifySpec`, `HackFlags`, `ResourceLimits`,
`ArtifactKind`. **Never edit it without a coordinated update across every
consumer.** `tests/test_contracts.py` pins it and will fail if it drifts.

## Verifier backends

Four backends implement the one `Verifier` protocol and return one
`VerifyResult` shape, so swapping is a config change (`get_verifier()`):

- `static` — in-process Dockerfile heuristics, no sandbox; universal fallback / M1 reward.
- `local-py` — harness as a local `python3 -I` subprocess; the deliberately **weak** baseline.
- `local-docker` — genuine `docker build` + run + HTTP probe; the **eval-time** verifier.
- `sentinel` — same harness inside Sentinel's nsjail/cgroups-v2 sandbox; the **hardened**, scalable training reward.

**Training reward uses the fast backends** (`static`/`local-py`/`sentinel`);
**eval uses genuine `local-docker`**. The C3 study is **weak (`local-py`) vs
hardened (`sentinel`)** running the *same* harness — the `HackFlags` difference
*is* the experiment.

## Pointers

- **Design / rationale** (verifier strategy, C3, eval protocol): `docs/DESIGN.md`.
- **Forward plan** (next steps NS-1..NS-4, milestones M3–M6, C1 Sentinel
  extension): `docs/ROADMAP.md`. **NS-1** (wheel packaging) and **NS-2** (app
  scaffold for genuine build+smoke) are **done**; nearest remaining steps need
  real infra: **NS-3** GPU GRPO run (M1) and **NS-4** live Sentinel (M2).
