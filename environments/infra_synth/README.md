# infra_synth

> **Status: WIP.** The environment logic is a documented stub; `load_environment`
> currently raises `NotImplementedError`. Datasets, verifier wiring, and reward
> shaping land with a later subagent.

A single-turn **RLVR** (Reinforcement Learning with Verifiable Rewards)
environment for the [Crucible](../../README.md) framework, packaged per the
Prime Intellect `verifiers` Hub spec.

- **Distribution:** `infra-synth` · **Import / dir:** `infra_synth`
- **Entrypoint:** `load_environment(**kwargs) -> vf.Environment`

## The task

The model is given an infrastructure spec and must emit the corresponding
**infrastructure-as-code artifact** as text. The first supported artifact is a
**Dockerfile**; later kinds (`compose`, `terraform`, `k8s`, `ci-yaml`) reuse the
same `ArtifactKind` contract.

## The reward (verifiable)

The emitted artifact is graded by a **pluggable verifier**
(`verifier.Verifier`), producing a `verifier.VerifyResult`:

1. **build_ok** — the artifact builds (e.g. `docker build` succeeds).
2. **smoke_ok** — a smoke test against the built artifact passes.

`verifier.reward.shape_reward(result, ...)` then maps the result to a scalar
(build + smoke weighted, with an optional penalty driven by
`VerifyResult.hack_flags`).

Two backends (see the root README "Verifier backends"):
- **local Docker** — builds + smoke-tests the Dockerfile locally (weak baseline).
- **Sentinel** — hardened sandboxed execution checks (nsjail, cgroups v2, no
  network); used for the reward-hacking study (M2).

## How to eval (once implemented)

```bash
# install the env package into the active environment
vf-install infra-synth

# run an eval (defaults from [tool.verifiers.eval]: 10 examples x 4 rollouts)
vf-eval infra-synth
```

Eval defaults are declared in [`pyproject.toml`](./pyproject.toml) under
`[tool.verifiers.eval]`.
