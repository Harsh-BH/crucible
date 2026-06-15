# Crucible

**A security-hardened RL-environment framework for RLVR (Reinforcement Learning
with Verifiable Rewards) training.** Crucible trains language models to produce
*verifiable* infrastructure-as-code, grading each rollout by actually building
and smoke-testing the artifact inside a sandbox — and studies what happens when
a policy learns to attack the verifier instead of solving the task.

> **Status:** base scaffolding is in place. The frozen interface contracts,
> packaging, docs, and test/lint infra exist now; the environment logic,
> verifier backends, the Sentinel client, reward shaping, and training configs
> land next (all marked **WIP** below).

---

## Contributions

- **C1 — Hardened verifier-as-a-service.** A pluggable verifier layer with a
  hardened execution backend built on **Sentinel** (nsjail, cgroups v2, no
  network). Sentinel runs untrusted Python/C++ today; the C1 roadmap is to
  *extend Sentinel with an infrastructure job type* so even Docker-style builds
  run under the same hardened sandbox.
- **C2 — `infra_synth` environment.** A `verifiers`-Hub-spec RLVR environment
  where the model emits infrastructure-as-code (Dockerfile first) and is graded
  by a real `build + smoke test` rather than a string match.
- **C3 — Reward-hacking-via-verifier-exploitation study.** Because reward comes
  from *executing* untrusted model output, the policy can be incentivized to
  attack the grader (resource exhaustion, OOM, timeouts, network egress, seccomp
  violations, spec gaming). Crucible records these as `HackFlags` on every
  result and studies the weak-verifier-vs-hardened-verifier axis directly.

---

## Architecture

```
                 +-----------------------------------------------+
                 |                  Trainer                      |
                 |        (prime-rl / TRL-style GRPO)            |
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
                 |   (verifiers.Environment; single-turn task)   |
                 +-----------------------------------------------+
                        |  (artifact, VerifySpec)
                        v
                 +-----------------------------------------------+
                 |        Pluggable verifier (Verifier)          |
                 |          verifier.types.Verifier              |
                 |                                               |
                 |   +---------------------+   +---------------+ |
                 |   |  LocalDockerVerifier|   | SentinelVerif.| |
                 |   |  (weak baseline:    |   | (hardened:    | |
                 |   |   docker build +    |   |  nsjail /     | |
                 |   |   smoke test)       |   |  cgroups v2 / | |
                 |   |                     |   |  no network)  | |
                 |   +---------------------+   +---------------+ |
                 +-----------------------------------------------+
                        |  VerifyResult (+ HackFlags)
                        v
                 +-----------------------------------------------+
                 |   verifier.reward.shape_reward(...) -> float  |
                 +-----------------------------------------------+
```

The verifier is chosen at config time. The **weak** (local Docker) backend and
the **hardened** (Sentinel) backend share one `Verifier` protocol and one
`VerifyResult` shape, which is exactly what makes the C3 study an apples-to-apples
comparison.

---

## Repository layout

```
RLF-VRTP/
├── README.md                  # this file — project source of truth
├── LICENSE                    # Apache-2.0
├── pyproject.toml             # uv workspace root + the `crucible` dev package
├── verifier/                  # publishable, dependency-light verifier layer
│   ├── types.py               #   FROZEN CONTRACTS (Verifier protocol + dataclasses)
│   ├── __init__.py            #   re-exports the public contract
│   ├── README.md              #   interface notes for the backends/client/reward
│   └── smoke/                 #   per-task smoke-test specs (WIP)
├── environments/
│   └── infra_synth/           # verifiers Hub-spec env package (dist: infra-synth)
│       ├── infra_synth.py     #   load_environment(**kwargs) -> vf.Environment (STUB)
│       ├── pyproject.toml     #   Hub spec + [tool.verifiers.eval] defaults
│       └── README.md          #   env card (WIP)
├── training/                  # GRPO/RLVR training entrypoints + configs (WIP)
├── eval/                      # benchmarks + ablations (WIP)
├── analysis/                  # reward-hacking study + training curves (WIP)
├── scripts/setup.sh           # bootstrap: install uv, uv sync, next steps
└── tests/test_contracts.py    # contract tests (PASS today)
```

`crucible-verifier` (import `verifier`) and `infra-synth` (import `infra_synth`)
are separate distributions wired together as a **uv workspace**; the top-level
`training/`, `eval/`, and `analysis/` dirs make up the `crucible` dev package.

---

## Verifier backends (read this — it is an honest design note)

**Sentinel** (the hardened backend, `github.com/Harsh-BH/Sentinel`) is a real
async Python/C++ code-execution engine. It runs a **single source file** in
nsjail with cgroups v2 limits and no network, and exposes:

- `POST /api/v1/submissions` → `202 {job_id, status: "QUEUED"}`
- poll `GET /api/v1/submissions/:id` until a terminal status
  (`SUCCESS` / `COMPILATION_ERROR` / `RUNTIME_ERROR` / `TIMEOUT` /
  `MEMORY_LIMIT_EXCEEDED` / `INTERNAL_ERROR`)
- default base URL `http://localhost:8080`, prefix `/api/v1`.

**Sentinel does not build Docker images.** So Crucible ships a **pluggable**
verifier layer rather than pretending one backend does everything:

- **Local Docker verifier** (`verifier.backends.LocalDockerVerifier`, WIP) —
  builds the generated `Dockerfile` and runs a smoke test *locally*. This is the
  **weak baseline**: it is the natural grader for the Dockerfile task today, but
  it executes untrusted build steps on the host's Docker daemon, which is
  exactly the attack surface C3 probes.
- **Sentinel verifier** (`verifier.sentinel_client.SentinelVerifier`, WIP) —
  routes untrusted **Python/C++ execution checks** through the hardened sandbox.
  This is the path used for `ArtifactKind.PYTHON` and for hardened acceptance
  checks (M2).

**C1 roadmap:** extend Sentinel with an *infra job type* so that Dockerfile-style
builds and infra smoke tests also run inside the hardened sandbox, closing the
gap between "weak local Docker" and "hardened Sentinel" for every artifact kind.

Both backends implement the same `verifier.types.Verifier` protocol and return
the same `VerifyResult`, so swapping them is a config change — and the difference
in `HackFlags` between them *is* the experiment.

---

## Quickstart

Crucible uses [`uv`](https://docs.astral.sh/uv/). The core install is
**CPU-only** and light; the training stack is opt-in.

```bash
# 0. bootstrap (installs uv if needed, syncs core + dev, prints next steps)
bash scripts/setup.sh

# --- or manually ---

# 1. core install (crucible-verifier, httpx, pyyaml, rich)
uv sync

# 2. add developer tooling (pytest, pytest-asyncio, ruff, mypy)
uv sync --extra dev

# 3. run the contract tests
uv run pytest -q

# 4. lint
uv run ruff check .
```

> **GPU / training:** `uv sync --extra train` pulls `verifiers`, `vllm`,
> `wandb`, and `datasets`. This group is **GPU-oriented** and is intentionally
> excluded from the core install — do not expect it to resolve cleanly on a
> CPU-only box.

Once the `infra_synth` environment logic is implemented, evaluate it with the
`verifiers` CLI:

```bash
uv run vf-install infra-synth     # install the Hub env into the active venv
uv run vf-eval infra-synth        # eval (defaults: 10 examples x 4 rollouts)
```

---

## Milestones

| Milestone | Goal | Status |
| --------- | ---- | ------ |
| **M1** | Reproduce GRPO on a built-in `verifiers` environment with Qwen3-1.7B (pipeline sanity: trainer ↔ vLLM ↔ env ↔ reward). | WIP |
| **M2** | Route the verifiable reward through **Sentinel** (hardened sandboxed execution checks) and capture `HackFlags`. | WIP |
| **M3** | `infra_synth` v1 — Dockerfile synthesis graded by build + smoke test, end to end. | WIP |

Downstream of the milestones, the **C3** study (analysis/) compares the weak
local-Docker verifier against the hardened Sentinel verifier on reward-hacking
incidence.

---

## Interface contract (for contributors)

The verifier contract in [`verifier/types.py`](./verifier/types.py) is
**frozen** — backends, the environment, and reward shaping all depend on its
exact names and fields. See [`verifier/README.md`](./verifier/README.md) for the
signatures later modules must implement (`verifier.backends`,
`verifier.sentinel_client`, `verifier.reward`). Do not edit `types.py` without a
coordinated update across consumers.

---

## License

[Apache-2.0](./LICENSE).
