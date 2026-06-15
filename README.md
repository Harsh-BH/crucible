# Crucible

**A security-hardened RL-environment framework for RLVR (Reinforcement Learning
with Verifiable Rewards) training.** Crucible trains language models to produce
*verifiable* infrastructure-as-code, grading each rollout by actually building
and smoke-testing the artifact inside a sandbox — and studies what happens when
a policy learns to attack the verifier instead of solving the task.

> **Status:** the foundation is built and tested end-to-end (**243 tests
> passing**). The verifier layer, the `infra_synth` environment, the M1/M2
> training scaffolding (prime-rl configs + a self-contained TRL trainer +
> a ≥3-seed harness), the M2 proof (Sentinel↔local parity + a throughput
> benchmark), the eval/`pass@k` tooling, the C3 reward-hacking analysis, and a
> full [`docs/DESIGN.md`](./docs/DESIGN.md) all exist now. The heavy *execution*
> steps — real GPU training and throughput against a live Sentinel — are
> documented, runnable commands (a GPU and a running Sentinel are required; they
> are not exercised in CI). Remaining gaps are tracked under
> [Known limitations](#known-limitations--next-steps).

---

## Contributions

- **C1 — Hardened verifier-as-a-service.** A *pluggable* verifier layer with a
  hardened execution backend built on **Sentinel** (`github.com/Harsh-BH/Sentinel`:
  nsjail, cgroups v2, no network). Sentinel runs untrusted Python/C++ today; the
  C1 roadmap is to *extend Sentinel with an infrastructure job type* so even
  Docker-style builds run inside the same hardened sandbox.
- **C2 — `infra_synth` environment.** A `verifiers`-Hub-spec single-turn RLVR
  environment where the model emits infrastructure-as-code (Dockerfile first) and
  is graded by a real `build + smoke test` rather than a string match. It ships
  as a proper wheel-installable package.
- **C3 — Reward-hacking-via-verifier-exploitation study.** Because the reward
  comes from *executing* untrusted model output, the policy can be incentivized
  to attack the grader (resource exhaustion, OOM, timeouts, network egress,
  seccomp violations, spec gaming). Crucible records these as `HackFlags` on every
  result and studies the **weak-verifier vs hardened-verifier** axis directly.
- **C4 — Reproducible benchmarks.** Seeded, config-driven runs; ≥3-seed variance
  reporting (mean ± 95% CI); unbiased `pass@k`; and the `pass@k` base-vs-RL
  crossover that honestly separates capability gain from search compression.

---

## Architecture

```
                 +-----------------------------------------------+
                 |                  Trainer                      |
                 |   prime-rl (uv run rl)  |  TRL (training/run) |
                 +-----------------------------------------------+
                        |  prompts                ^  rewards
                        v                         |
                 +----------------+               |
                 | Rollout server |  completions  |
                 |     (vLLM)      |---------------+
                 +----------------+               |
                        |  artifact (text)        |
                        v                         |
                 +-----------------------------------------------+
                 |          infra_synth environment              |
                 |   (verifiers.SingleTurnEnv; parse + grade)    |
                 +-----------------------------------------------+
                        |  (artifact, VerifySpec)
                        v
        +-----------------------------------------------------------+
        |              Pluggable verifier (Verifier protocol)       |
        |                                                           |
        |  static       local-py        local-docker     sentinel  |
        |  (in-proc,    (weak: subproc   (genuine build   (hardened |
        |   fast        + rlimits,        + smoke; the      Python   |
        |   default)    C3 baseline)      eval verifier)    sandbox) |
        +-----------------------------------------------------------+
                        |  VerifyResult (+ HackFlags)
                        v
                 +-----------------------------------------------+
                 |   verifier.reward.shape_reward(...) -> float  |
                 +-----------------------------------------------+
```

All four backends implement one `verifier.types.Verifier` protocol and return
one `VerifyResult` shape, so swapping them is a config change. **Training-loop
reward** uses the *fast* backends (`static` / `local-py` / Sentinel-sandboxed
harness — sub-second, scalable via Sentinel's queue + worker pool + KEDA);
**eval-time reward** uses the *genuine* `local-docker` build + smoke test. The
C3 study's apples-to-apples comparison is **`local-py` (weak) vs `sentinel`
(hardened)** running the *same* check harness — the difference in `HackFlags`
*is* the experiment. See [`docs/DESIGN.md`](./docs/DESIGN.md) for the full
rationale.

---

## Repository layout

```
RLF-VRTP/
├── README.md                       # this file — project source of truth
├── LICENSE                         # Apache-2.0
├── pyproject.toml                  # uv workspace root + the `crucible` dev package
├── docs/
│   └── DESIGN.md                   # architecture, verifier strategy, C3 + eval protocol
├── verifier/                       # publishable, dependency-light verifier layer
│   ├── types.py                    #   FROZEN CONTRACTS (Verifier protocol + dataclasses)
│   ├── backends.py                 #   Static / LocalPy / LocalDocker verifiers + get_verifier()
│   ├── sentinel_client.py          #   async Sentinel API client + SentinelVerifier
│   ├── reward.py                   #   shape_reward() + result_to_metrics()
│   └── smoke/checks.py             #   DRY Dockerfile checks + Python harness builder
├── environments/
│   └── infra_synth/                # verifiers Hub-spec env (dist: infra-synth)
│       ├── pyproject.toml          #   Hub spec + [tool.verifiers.eval] defaults
│       └── infra_synth/            #   the package (ships in the wheel)
│           ├── __init__.py         #     re-exports load_environment + helpers
│           ├── environment.py      #     load_environment(**kwargs) -> vf.Environment
│           ├── tasks.py            #     seeded task generation + build_verify_spec()
│           ├── parser.py           #     extract the Dockerfile from a completion
│           └── gold.py             #     reference (gold) Dockerfile generation
├── training/                       # M1/M2 trainers + configs
│   ├── run.py                      #   self-contained TRL GRPO driver (DAPO/Dr.GRPO/GSPO knobs)
│   ├── data.py · rewards.py · seeds.py
│   └── configs/                    #   prime-rl TOML (m1_*, m2_*) + TRL YAML
├── eval/                           # passk.py · benchmark.py · parity.py · throughput.py
├── analysis/                       # reward_hacking.py (C3 taxonomy) · curves.py (RLVR dashboard)
├── scripts/setup.sh                # bootstrap: install uv, uv sync, next steps
└── tests/                          # 243 tests (verifier, env, training, eval, analysis)
```

`crucible-verifier` (import `verifier`) and `infra-synth` (import `infra_synth`)
are separate distributions wired together as a **uv workspace**; the top-level
`training/`, `eval/`, and `analysis/` dirs make up the `crucible` dev package.

---

## Quickstart

Crucible uses [`uv`](https://docs.astral.sh/uv/). The core install is
**CPU-only** and light; the training stack is opt-in.

```bash
bash scripts/setup.sh          # installs uv if needed, syncs core + dev, prints next steps
# --- or manually ---
uv sync --extra dev            # core (crucible-verifier, httpx, pyyaml, rich) + dev tooling
uv run pytest -q               # 243 passed, 1 skipped (the skip needs an HF network download)
uv run ruff check .
```

### Evaluate the `infra_synth` environment

```bash
uv run vf-install infra-synth                                   # install the Hub env
# Fast, dependency-free static grading (no Docker/Sentinel needed):
uv run vf-eval infra-synth -m <model> -b <vllm-url> -k OPENAI_API_KEY \
  -a '{"verifier_backend": "static"}'
# Genuine build + smoke test (needs a Docker daemon):
uv run vf-eval infra-synth ... -a '{"verifier_backend": "local-docker"}'
# Hardened sandboxed checks (needs a running Sentinel at :8080):
uv run vf-eval infra-synth ... -a '{"verifier_backend": "sentinel", "sentinel_base_url": "http://localhost:8080"}'
```

### Train (M1 / M2)

```bash
# M1 — prime-rl (primary stack), single-GPU colocated, Qwen3-1.7B + LoRA:
uv run rl @ training/configs/m1_gsm8k.toml \
  --trainer-gpu-ids 0 --inference-gpu-ids 0 --inference.gpu-memory-utilization 0.5

# M1 — self-contained TRL baseline (hackable; supports the ablation knobs):
python training/run.py --env gsm8k --model Qwen/Qwen3-1.7B --num-generations 8 --seed 0

# M2 — route the reward through the hardened Sentinel sandbox (needs Sentinel up):
python training/run.py --env infra_synth --verifier-backend sentinel \
  --sentinel-base-url http://localhost:8080
```

> **GPU / training deps:** `uv sync --extra train` pulls `verifiers`, `vllm`,
> `wandb`, `datasets`. This group is **GPU-oriented** and intentionally excluded
> from the core install — it will not resolve cleanly on a CPU-only box. Real
> training needs a GPU; `python training/run.py --smoke` only checks the code
> path. See [`training/README.md`](./training/README.md) for both trainer paths,
> the DAPO/Dr.GRPO/GSPO knob mapping, and single-GPU tuning.

### Prove M2 without a GPU (parity + throughput, against a mock or live Sentinel)

```bash
python -m eval.parity --base-url http://localhost:8080      # Sentinel verdicts == local baseline
python -m eval.throughput --mock                            # throughput + p50/p90/p99 (mock transport)
```

---

## Milestones

| Milestone | Goal | Status |
| --------- | ---- | ------ |
| **M1** | Reproduce GRPO on a built-in `verifiers` env with Qwen3-1.7B; reward rises; stable across ≥3 seeds. | **Scaffolded** — prime-rl configs (`m1_gsm8k`, `m1_reverse_text`) + TRL `run.py` + seed harness are built and unit-tested; the GPU run is the remaining execution step. |
| **M2** | Route the verifiable reward through **Sentinel** and capture `HackFlags`; match a local baseline; measure throughput. | **Scaffolded** — `SentinelVerifier` + `m2_*` configs built; parity (local-py vs Sentinel = 1.0 agreement on a mock that runs the real harness) and a throughput benchmark are implemented; numbers against a live Sentinel are the remaining step. |
| **M3** | `infra_synth` v1 — Dockerfile synthesis graded by build + smoke test, end to end, with a measurable ablated gain. | **Next** — the env + the genuine `local-docker` build path exist; the per-task *app scaffold* (so realistic FastAPI/Flask Dockerfiles actually build and serve `/health`) is the next piece (see below). |

Downstream, the **C3** study (`analysis/reward_hacking.py`) compares the weak
`local-py` verifier against the hardened `sentinel` verifier on reward-hacking
incidence, using a 6-category taxonomy and an ImpossibleBench-style cheating rate.

**→ The full forward plan** — prioritized next steps, the M3–M6 detail, the C1
Sentinel-extension plan, and an execution playbook — lives in
[`docs/ROADMAP.md`](./docs/ROADMAP.md).

---

## Known limitations & next steps

- **`crucible-verifier` is not yet wheel-installable.** The `verifier/` package
  uses a flat layout (like `infra_synth` did before its fix), so a *built wheel*
  ships its modules at the top level instead of under a `verifier/` namespace; it
  works in editable/dev mode (and all tests pass) but not as a standalone wheel.
  Since `infra_synth` depends on `crucible-verifier`, this must be fixed (the same
  nested-package move) before publishing either to the Hub. Tracked as the next
  packaging task.
- **Genuine build+smoke needs a per-task app scaffold.** `LocalDockerVerifier`
  builds a context containing only the model's `Dockerfile`, so a realistic
  Dockerfile that `COPY`s app code fails to build. M3 adds an app scaffold
  (`requirements.txt` + a minimal server exposing `/health`) into the build
  context; the frozen `VerifySpec.smoke` dict already accommodates it via
  `context_files`, so this is non-breaking.
- **Real runs need real infra.** GRPO training requires a GPU; live throughput
  numbers require a running Sentinel (`make up` in the Sentinel repo). Neither is
  exercised in CI.
- **Sentinel seccomp is currently disabled** (a kafel `fstat` issue upstream);
  namespaces + cgroups v2 + no-network are enforced, but syscall-violation
  `HackFlags` are not produced until that is restored.

---

## Interface contract (for contributors)

The verifier contract in [`verifier/types.py`](./verifier/types.py) is
**frozen** — backends, the environment, reward shaping, training, and eval all
depend on its exact names and fields (`Verifier`, `VerifyResult`, `VerifySpec`,
`HackFlags`, `ResourceLimits`, `ArtifactKind`). Do not edit it without a
coordinated update across consumers; `tests/test_contracts.py` pins it.

---

## License

[Apache-2.0](./LICENSE).
