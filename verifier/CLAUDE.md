# verifier/ — the pluggable verifier layer

Distribution `crucible-verifier`, import `verifier`. Dependency-light: only
`httpx` + stdlib (no torch/verifiers). Everything else in the repo depends on
the contract here.

## Files

- `types.py` — **FROZEN CONTRACTS.** `Verifier` protocol, `VerifyResult`,
  `VerifySpec`, `HackFlags`, `ResourceLimits`, `ArtifactKind`. **Do not edit**
  without coordinated updates to every consumer; `tests/test_contracts.py` pins it.
- `backends.py` — `StaticVerifier`, `LocalPyVerifier`, `LocalDockerVerifier`,
  and `get_verifier(...)` (the factory that selects a backend by name).
- `sentinel_client.py` — async `SentinelClient` + `SentinelVerifier` (talks to
  the Sentinel API on :8080; accepts an injectable `httpx` transport for tests).
- `reward.py` — `shape_reward()` (partial credit via `build_weight`, optional
  `hack_penalty`) and `result_to_metrics()`.
- `smoke/checks.py` — DRY Dockerfile checks (`check_dockerfile`) and
  `build_python_harness()` — the single harness string `local-py` and `sentinel`
  both execute, keeping weak and hardened honest.

## Backends (one protocol, one result)

`static` (in-proc heuristics) · `local-py` (weak subprocess `python3 -I` +
rlimits, C3 baseline) · `local-docker` (genuine build + smoke, eval verifier) ·
`sentinel` (hardened sandbox, scalable training reward). Train uses the fast
ones; eval uses `local-docker`; C3 compares `local-py` vs `sentinel`.

## Tests

```bash
uv run pytest tests/test_backends.py tests/test_sentinel_client.py \
  tests/test_reward.py tests/test_checks.py
```

(`tests/test_contracts.py` guards `types.py`.)

## Note (NS-1)

`verifier/` uses a flat layout, so a *built wheel* ships modules at top level and
`import verifier` fails non-editable. NS-1 in `docs/ROADMAP.md` is the
nested-package fix needed before Hub publish (`infra-synth` depends on this).
