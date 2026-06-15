# crucible-verifier

Dependency-light, **backend-agnostic** verifier layer for Crucible.

- **Distribution name:** `crucible-verifier`
- **Import name:** `verifier`
- **Runtime deps:** `httpx` only (the contracts themselves are stdlib-only).

## The stable contract

The frozen contract lives in [`verifier/types.py`](./types.py) and is
re-exported from `verifier/__init__.py`:

- `ArtifactKind` — kind of artifact under test (`dockerfile`, `compose`,
  `terraform`, `k8s`, `ci-yaml`, `python`).
- `ResourceLimits` — wall time / memory / pids / cpus budget.
- `VerifySpec` — one acceptance check (`spec_id`, `kind`, `smoke`, `limits`).
- `HackFlags` — verifier-exploitation signals for the reward-hacking study.
- `VerifyResult` — backend-agnostic outcome (`build_ok`, `smoke_ok`, `reward`, …).
- `Verifier` — `runtime_checkable` `Protocol`: `name: str` +
  `async def verify(self, artifact: str, spec: VerifySpec) -> VerifyResult`.

**Do not change names or fields in `types.py`.** Backends and the `infra_synth`
environment depend on this exact shape.

## Interface notes for later subagents (NOT yet implemented)

Conform to these signatures when you build them:

### `verifier/backends.py`
- `LocalDockerVerifier` — builds a generated `Dockerfile` and runs a smoke test
  locally (the "weak" verifier baseline). Implements `Verifier`.
- `LocalPyVerifier` — runs a Python execution check locally. Implements `Verifier`.

### `verifier/sentinel_client.py`
- `SentinelClient` — thin async HTTP client for the Sentinel sandbox.
- `SentinelVerifier` — wraps `SentinelClient` as a `Verifier` (the "hardened"
  execution path).

Sentinel targets the **real async API**:
- `POST /api/v1/submissions` → `202 {job_id, status: "QUEUED"}`
- poll `GET /api/v1/submissions/:id` until a terminal status
- base URL default `http://localhost:8080`, prefix `/api/v1`
- request body: `{language, source_code, stdin, time_limit_ms?, memory_limit_kb?}`
- terminal statuses: `SUCCESS` / `COMPILATION_ERROR` / `RUNTIME_ERROR` /
  `TIMEOUT` / `MEMORY_LIMIT_EXCEEDED` / `INTERNAL_ERROR`

Sentinel runs a single Python/C++ source file in nsjail (cgroups v2, no
network); it does **not** build Docker images. See the root README "Verifier
backends" section for the weak-vs-hardened split.

### `verifier/reward.py`
```python
def shape_reward(
    result: VerifyResult,
    *,
    build_weight: float = 0.3,
    smoke_weight: float = 0.7,
    hack_penalty: float = 0.0,
    binary: bool = False,
) -> float: ...
```
Populates `VerifyResult.reward` (which is `None` until shaped).

### `verifier/smoke/`
Concrete per-task smoke-test specs land in this package.
