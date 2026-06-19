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
  (seed-independent FNV-1a hash) for contamination resistance. A second
  `ArtifactKind` is wired end-to-end through the static path: pass
  `artifact_kind="compose"` to `load_environment` (or `kind="compose"` to
  `generate_tasks`) to emit/grade a `docker-compose.yml` instead of a Dockerfile
  (graded by `check_compose`); `"dockerfile"` stays the default. A third
  `ArtifactKind` is wired the same way: `artifact_kind="ci-yaml"` (or
  `kind="ci-yaml"`) emits/grades a GitHub Actions workflow (graded by
  `check_ci_yaml`).
- `infra_synth/parser.py` — `extract_dockerfile()` pulls the last fenced
  ```dockerfile block from a completion; `extract_compose()` pulls the last
  ```yaml/```yml/```compose block (Docker Compose support); `extract_ci_yaml()`
  pulls the last ```yaml/```yml block (GitHub Actions workflow support).
- `infra_synth/gold.py` — `gold_dockerfile(info)` renders a correct reference
  (eval references + the gold-passes-its-own-spec test); `gold_compose(info)`
  renders a correct reference `docker-compose.yml` that passes `check_compose`;
  `gold_ci_yaml(info)` renders a correct reference workflow that passes
  `check_ci_yaml`.
- `infra_synth/scaffold.py` — `app_scaffold(info)` returns the build-context
  files a realistic Dockerfile `COPY`s (`requirements.txt` + a minimal
  FastAPI/Flask `app` serving the task's health path). `build_verify_spec`
  attaches it as `smoke["context_files"]`; `LocalDockerVerifier` writes it into
  the build context for genuine build+smoke.
- `infra_synth/impossible.py` — C3 ImpossibleBench-style generator
  (`impossible_tasks` + `MUTATIONS`: spec mutations no faithful Dockerfile, gold
  included, can pass — all `weak_tests`) plus `adversarial_corpus`
  (category-tagged, safe `PYTHON`-kind grader-subversion artifacts for the
  weak-vs-hardened study; dangerous `fork_bomb` excluded by default). Stdlib +
  `verifier.types` only; does NOT import `analysis/`.

## Hub spec

`pyproject.toml` declares the verifiers Hub spec, `[tool.verifiers.eval]`
defaults, and force-includes itself into the wheel as `infra_synth/pyproject.toml`.

## Commands

```bash
uv run vf-install infra-synth
uv run vf-eval infra-synth -a '{"verifier_backend":"static"}'
```

## Note (NS-2 — done)

NS-2 is complete: the per-task **app scaffold** (`scaffold.py`) ships
`requirements.txt` + a minimal FastAPI/Flask `app` (serving the task's health
path) into the build context via `smoke["context_files"]`, and
`LocalDockerVerifier` writes those files before building. Verified end-to-end
under a real Docker daemon — gold Dockerfile **builds + serves health → HTTP
200** for both frameworks. Non-breaking (the frozen `VerifySpec.smoke` dict
already accommodated `context_files`). M3 next: a measurable, ablated model gain.
