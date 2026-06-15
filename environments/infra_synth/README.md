# infra_synth

A single-turn **RLVR** (Reinforcement Learning with Verifiable Rewards)
environment for the [Crucible](../../README.md) framework, packaged per the
Prime Intellect `verifiers` Hub spec.

- **Distribution:** `infra-synth` · **Import / dir:** `infra_synth`
- **Entrypoint:** `load_environment(**kwargs) -> vf.Environment`

> **Training: WIP.** The environment, dataset, parser, gold references, and
> reward wiring are implemented and unit-tested. End-to-end RL training is still
> work in progress.

## The task

Given a natural-language infrastructure spec, the model must emit a **Dockerfile**
inside a single ```` ```dockerfile ```` fenced block (no prose, no `<think>`).
Specs are parameterized over a grid (language × framework × dependencies × port
× health path); the first language is Python (FastAPI / Flask). Later artifact
kinds (`compose`, `terraform`, `k8s`, `ci-yaml`) reuse the same `ArtifactKind`
contract.

The **train** and **test** splits draw from **disjoint** parameter combinations
(fresh combos are held out for test), so the split is contamination-resistant.

## The reward (verifiable, shaped)

The emitted Dockerfile is extracted (last fenced block) and graded by a
**pluggable verifier** (`verifier.Verifier`), producing a `verifier.VerifyResult`:

1. **build_ok** — the artifact builds (e.g. `docker build` succeeds).
2. **smoke_ok** — a smoke test against the built artifact passes (port +
   health-check endpoint return the expected status).

The scalar reward is `verifier.shape_reward(result, ...)`:

```
reward ≈ build_weight * build_ok + smoke_weight * smoke_ok − hack_penalty * hack
```

with defaults **build 0.3 + smoke 0.7** (clamped to `[0, 1]`). A small dense
**format** reward (weight 0.1) gives non-zero signal when a well-formed
Dockerfile is emitted, to fight early zero-advantage; its weight is kept small so
the policy cannot game reward by producing well-formatted but non-building output.
Weight-0 **metrics** (`build_ok_metric`, `smoke_ok_metric`, `hack_any_metric`)
are logged for analysis (they read state stashed by the primary reward fn, which
runs first).

### Verifier backends

| backend         | what it does                                            |
| --------------- | ------------------------------------------------------- |
| `static`        | **zero-dependency default** — static checks (pinned `FROM`, `EXPOSE <port>`, `CMD`, required substrings). No Docker required. |
| `local-docker`  | the genuine **build + smoke** reward — builds the Dockerfile and hits the health endpoint locally. |
| `local-py`      | local Python execution check.                           |
| `sentinel`      | hardened sandboxed checks (nsjail, cgroups v2, no network). |

The backend is resolved via the shared `verifier.get_verifier(name, base_url=...)`
contract; pass `sentinel_base_url=...` for the Sentinel backend.

## `load_environment` kwargs

| kwarg                | default    | meaning                                            |
| -------------------- | ---------- | -------------------------------------------------- |
| `verifier_backend`   | `"static"` | which `Verifier` to resolve (see table above).     |
| `verifier`           | `None`     | inject an explicit `Verifier` (overrides backend). |
| `build_weight`       | `0.3`      | weight on `build_ok`.                              |
| `smoke_weight`       | `0.7`      | weight on `smoke_ok`.                              |
| `hack_penalty`       | `0.0`      | penalty when any `HackFlags` trip.                 |
| `num_tasks`          | `None`     | cap on tasks (None → whole split pool).            |
| `seed`               | `0`        | deterministic sampling seed.                       |
| `split`              | `"train"`  | training split; eval uses the disjoint `test` split. |
| `sentinel_base_url`  | `None`     | base URL for the Sentinel backend.                 |

## How to eval

```bash
# install the env package into the active environment
vf-install infra-synth

# evaluate a model served at a vLLM-compatible URL
vf-eval infra-synth -m <model> -b <vllm-url> -a '{"verifier_backend":"local-docker"}'
```

`static` is the zero-dependency default (great for smoke-testing the harness);
`local-docker` gives the genuine build + smoke reward; `sentinel` runs the
hardened sandboxed checks. Eval defaults are declared in
[`pyproject.toml`](./pyproject.toml) under `[tool.verifiers.eval]`.
