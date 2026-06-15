# environments/infra_synth/ — the RLVR environment (C2)

Distribution `infra-synth`, import `infra_synth`. A **nested package**: the
importable code lives in `environments/infra_synth/infra_synth/`, and the Hub
spec `pyproject.toml` sits one level up. Depends on `crucible-verifier`.

## Files

- `infra_synth/environment.py` — `load_environment(**kwargs) -> vf.Environment`,
  a single-turn `verifiers` env. `verifier_backend` selects the grader; the
  reward is an async `build_smoke_reward` + an optional `format_reward`.
- `infra_synth/tasks.py` — seeded task generation over a parameter grid
  (language × framework × dependency × port × health-path); `generate_tasks()`
  and `build_verify_spec(info)`. Train/test draw **disjoint** combos
  (seed-independent FNV-1a hash) for contamination resistance.
- `infra_synth/parser.py` — `extract_dockerfile()` pulls the last fenced
  ```dockerfile block from a completion.
- `infra_synth/gold.py` — `gold_dockerfile(info)` renders a correct reference
  (eval references + the gold-passes-its-own-spec test).

## Hub spec

`pyproject.toml` declares the verifiers Hub spec, `[tool.verifiers.eval]`
defaults, and force-includes itself into the wheel as `infra_synth/pyproject.toml`.

## Commands

```bash
uv run vf-install infra-synth
uv run vf-eval infra-synth -a '{"verifier_backend":"static"}'
```

## Note (NS-2 / M3)

`load_environment` and the env helpers are built; the genuine `local-docker`
build needs a per-task **app scaffold** (`requirements.txt` + a minimal server
exposing `/health`) written into the build context. `VerifySpec.smoke` already
accommodates `context_files`, so this is non-breaking. See NS-2 in `docs/ROADMAP.md`.
