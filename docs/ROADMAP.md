# Crucible — Roadmap & Next Steps

> Forward plan and execution checklist. For *why* the system is shaped the way it
> is (verifier strategy, weak-vs-hardened axis, eval protocol), see
> [`DESIGN.md`](./DESIGN.md). For the high-level pitch and layout, see the
> [root README](../README.md).
>
> **Snapshot:** 2026-06-16 · branch `main` · 249 tests passing · ruff-clean ·
> mypy-clean. **NS-1 and NS-2 are done** (see §1); NS-3/NS-4 remain (need GPU /
> live Sentinel).

---

## 0. Where we are

**Built, tested, pushed:**
- Project base — uv workspace, frozen `verifier/types.py` contracts, Hub-spec packaging.
- **Verifier layer (C1, in-repo part):** four backends behind one `Verifier` protocol —
  `static`, `local-py` (weak baseline), `local-docker` (genuine build+smoke),
  `sentinel` (hardened) — plus the async Sentinel client and `shape_reward`.
- **`infra_synth` environment (C2):** parser, seeded task-gen, gold refs,
  `load_environment`; wheel-installable.
- **M1 scaffolding:** prime-rl TOML configs + self-contained TRL `run.py` + ≥3-seed harness.
- **M2 scaffolding:** Sentinel-routed reward + parity test + throughput benchmark.
- **C3/C4 scaffolding:** reward-hacking taxonomy/analysis, unbiased `pass@k`, eval harness, `DESIGN.md`.

**Done since (NS-1, NS-2):**
- **NS-1 — `crucible-verifier` wheel packaging.** `verifier/` moved to a nested
  layout (`verifier/verifier/`); a built wheel installs as `import verifier`
  non-editable (verified in a clean venv). Unblocks Hub publish (M6).
- **NS-2 — genuine `local-docker` build+smoke.** `infra_synth/scaffold.py`
  ships a `requirements.txt` + minimal FastAPI/Flask app into the build context
  (`smoke["context_files"]`); `LocalDockerVerifier` writes it in. Verified
  end-to-end under a real Docker daemon: gold Dockerfile **builds + serves
  health → HTTP 200** (both frameworks).
- **C3 end-to-end study harness (M5 enabler).** `infra_synth/impossible.py`
  (`impossible_tasks` + `MUTATIONS` that make a spec unsatisfiable — gold fails
  its own static check, verified — plus a safe category-tagged `adversarial_corpus`),
  `LocalPyVerifier` now executes `ArtifactKind.PYTHON` directly (weak↔hardened
  symmetry), and `eval/c3_study.py` (`run_c3_study`) grades the same trials
  through **weak (`local-py`) vs hardened (`sentinel`)** → `compare_weak_vs_hardened`
  taxonomy + an undeserved-pass metric. Demonstrated with the simulated sandbox
  (`--mock`): the hardened side closes `resource_manipulation` (mem/timer) that
  weak lets pass. The **live weak-vs-hardened numbers** still await NS-4.
- **M4 — all declared `ArtifactKind`s wired end-to-end (static path).** Compose,
  GitHub Actions CI-YAML, Terraform (HCL), and k8s (YAML) now join Dockerfile and
  Python. The static checker is **kind-dispatched** (`verifier.smoke.checks.
  check_artifact` + a per-kind harness builder); `check_compose`/`check_ci_yaml`/
  `check_terraform`/`check_k8s` are stdlib-only heuristic stand-ins (no PyYAML/HCL
  libs). The env gained per-kind `extract_*`, `gold_*`, `generate_tasks(kind=...)`,
  `*_SYSTEM_PROMPT`s, and `load_environment(artifact_kind=...)`. Verified per kind:
  gold passes its own spec via the in-process static check **and** the inlined
  `local-py` harness (0 parity mismatches), and `spec_gaming` fires on a
  token-parroting artifact. **The C3 study spans all kinds** —
  `impossible_tasks(kind=...)` + `eval.c3_study.default_trials` grade gold-on-
  impossible across every kind (gold always fails → 0 undeserved passes, verified).
- **M4 — genuine `docker compose` verifier (`LocalComposeVerifier`).** Compose is
  no longer only a static stand-in: `get_verifier("local-compose")` writes the
  model's `docker-compose.yml` + a build context (`infra_synth.scaffold.
  compose_scaffold` = gold `Dockerfile` + `requirements.txt` + `app/`), runs
  `docker compose up --build`, HTTP-probes the health path, and tears down.
  **Verified against a live Docker daemon (29.4.2 / Compose 5.1.3):** gold compose
  for FastAPI + Flask **builds + serves health → 200**; a path the server doesn't
  serve correctly yields `smoke_ok=False` (the probe discriminates). The existing
  `local-docker` Dockerfile verifier was re-confirmed end-to-end the same way.
  Remaining genuine verifiers: `terraform validate`, `kubeconform` — need those
  CLIs (heuristic stand-ins exist today).

**Scaffolded but NOT yet executed (need real infra):**
- A real GPU GRPO run (M1 "reward rises, ≥3 seeds").
- Throughput/parity against a *live* Sentinel (M2 numbers).
- Real C3 cheating-rate figures: run `eval.c3_study` against a live Sentinel (NS-4).

---

## 1. Immediate next steps (prioritized)

| # | Task | Why | Acceptance criteria | Effort | Blocks |
|---|------|-----|---------------------|--------|--------|
| ~~**NS-1**~~ ✅ | ~~**Fix `crucible-verifier` wheel packaging**~~ **DONE** | `verifier/` was a flat package; a built wheel shipped modules at top level, so `import verifier` failed from a wheel. | **Met.** `verifier/` moved to a nested layout (`verifier/verifier/`); a built wheel installs into a clean venv and `import verifier` / `from verifier.types import VerifyResult` work *non-editable*; 249 tests green; ruff + mypy clean. | S | Hub publish (M6) |
| ~~**NS-2**~~ ✅ | ~~**M3 app scaffold for genuine build+smoke**~~ **DONE** | `LocalDockerVerifier` built a context with only the `Dockerfile`; realistic Dockerfiles that `COPY` app code failed. | **Met.** `infra_synth/scaffold.py: app_scaffold(info) -> dict[str,str]` (requirements + a minimal FastAPI/Flask server exposing the health path); `build_verify_spec` puts it in `smoke["context_files"]`; `LocalDockerVerifier` writes those files into the build context (with a path-traversal guard); gold Dockerfile **builds + serves health → 200** under a real Docker daemon (both frameworks). | M | M3 |
| **NS-3** | **Execute M1 on a GPU** | Prove the loop: reward rises and is stable across ≥3 seeds. | `reverse-text` climbs ~0.05→0.8 (loop sanity), then `gsm8k` on Qwen3-1.7B shows rising reward across seeds {0,1,2}; report mean±95% CI via `training/seeds.py`. | M | M3+ |
| **NS-4** | **Stand up Sentinel + execute M2** | Real parity + throughput numbers (the systems story). | `make up` in the Sentinel repo (:8080); `python -m eval.parity --base-url ...` shows local-py↔sentinel agreement; `python -m eval.throughput --base-url ...` reports throughput + p50/p90/p99. | S–M | M5 |

**Recommended order:** ~~NS-1~~ → ~~NS-2~~ (both done) → **NS-3/NS-4 in parallel
once compute + a Sentinel host are available** (the remaining work; both need
real infra this repo can't self-provision).

---

## 2. Milestone roadmap (M3 → M6)

**M3 — `infra_synth` v1 (Dockerfile, end-to-end).**
Done when: NS-2 lands (genuine build+smoke); a small model (Qwen3-1.7B+LoRA) shows a
**measurable, ablated** gain on held-out tasks (build+smoke pass-rate up vs base),
reported with pass@1 + pass@k and ≥3 seeds. Ablations: verifier strength
(static vs local-docker), reward shaping on/off.

**M4 — Expand & harden.**
Add `ArtifactKind` coverage beyond Dockerfile (compose → Terraform `validate` →
k8s `kubeconform` → CI-YAML), each with gold refs + smoke specs. Add **shaped
rewards** (already supported via `build_weight`/`smoke_weight`), **difficulty
scheduling** (easy→hard; prime-rl `[orchestrator.buffer]` filtering / TRL dynamic
sampling), and address **exploration collapse** on this env (entropy bonus,
temperature, DAPO clip-higher).

**M5 — Reward-hacking study (C3).**
Run the **weak (`local-py`) vs hardened (`sentinel`)** comparison; catalog exploits
into the 6-category taxonomy (`analysis/reward_hacking.py`); compute the
ImpossibleBench-style **cheating rate** on impossible/mutated tasks; ship a
**verifier-hardening recipe** with numbers (cheating-rate reduction *without*
hurting true accuracy). Tie in pass@k base-vs-RL (search-compression).

**M6 — Ship.**
Publish `infra-synth` to the **Environments Hub** (`prime login → prime env push
infra-synth`; needs NS-1 first), a one-command training script, a benchmark
table, a launch write-up, and a demo (dial "steps" → watch accuracy rise).

---

## 3. C1 — extend Sentinel with an infra job type (the systems headline)

Sentinel today runs a single Python/C++ source file in nsjail; it cannot build
Docker images. To run the *genuine* infra reward inside the hardened sandbox
(closing the gap between `local-docker` and `sentinel`):

1. **New job type / `Language` value** (`dockerfile`/`infra`) carrying a Dockerfile
   + a file map (build context) + a smoke spec (commands + an HTTP probe).
2. **Build+run executor** — rootless **buildkit/kaniko** to build, then run the
   image (gVisor/Kata/Sysbox) with a **loopback-enabled** netns so the smoke test
   can `curl` the started container.
3. **Structured `hack_flags` in the result** — add `OOMKilled`, `TimedOut`,
   `Signal`, `SeccompViolation`, `NetworkAttempt`, `BuildFailed`,
   `SmokeTestPassed` (+ json tags on `ExecutionResult`).
4. **Re-enable seccomp** — fix the kafel `fstat` identifier so syscall-violation
   flags are actually produced (currently disabled upstream).
5. **Auth** — an API-key header on `/api/v1/submissions*`.

Acceptance: a Dockerfile build + smoke test runs entirely inside Sentinel and
returns the same `VerifyResult`+`HackFlags` shape, so `infra_synth` can target
`sentinel` for *every* artifact kind. (Note: this work lives in the **Sentinel
repo**, not here.)

---

## 4. Execution playbook (turn scaffolding into results)

```bash
# --- M1: GPU GRPO run ---
# prime-rl (primary), single-GPU colocated:
uv run rl @ training/configs/m1_gsm8k.toml \
  --trainer-gpu-ids 0 --inference-gpu-ids 0 \
  --inference.gpu-memory-utilization 0.5 --inference.model.max-model-len 2048
# TRL baseline (hackable; ablation knobs), one seed:
python training/run.py --env gsm8k --model Qwen/Qwen3-1.7B --num-generations 8 --seed 0
# ≥3 seeds + variance: drive run.py over seeds {0,1,2}, aggregate with training/seeds.py

# --- M2: live Sentinel ---
#   (in the Sentinel repo) make up        # API on :8080
python -m eval.parity     --base-url http://localhost:8080   # local-py == sentinel verdicts
python -m eval.throughput --base-url http://localhost:8080   # throughput + p50/p90/p99
python training/run.py --env infra_synth --verifier-backend sentinel \
  --sentinel-base-url http://localhost:8080
```

What's required: a GPU (24–40 GB is enough for 1.7B+LoRA) for M1; a running
Sentinel for M2; a Docker daemon for genuine `local-docker` eval.

---

## 5. Open decisions (confirm before big runs)

- **Compute (the one still open):** personal/college GPU, rented H100s by the hour,
  or Prime Intellect hosted? This fixes model size + LoRA vs full fine-tune. Defaults
  in the configs: **Qwen3-1.7B + LoRA, single GPU** (scales up via config).
- Already decided: keep the name **Crucible**; domain **`infra_synth`**; **framework
  GRPO only** (no hand-rolled optimizer).

---

## 6. Risks & things to watch

- **Entropy / exploration collapse** — watch entropy + within-group reward std → 0
  (zero-advantage = no gradient). Mitigate with reward shaping (partial credit is
  already on), difficulty scheduling, clip-higher, temperature.
- **Search-compression critique** — measure **pass@k base-vs-RL** to large k; if
  curves cross, RL is sharpening, not expanding capability. `analysis/curves.py`
  has `passk_base_vs_rl`.
- **Verifier throughput** — genuine `docker build` per rollout is too slow for the
  loop; keep the training reward on `static`/`local-py`/Sentinel-harness and reserve
  `local-docker` for eval. Sentinel's queue + worker pool + KEDA is the scale path.
- **Contamination** — `infra_synth` train/test splits already use disjoint parameter
  combos; keep new task families contamination-resistant.
- **Reproducibility** — every run seeded + config-driven; always report ≥3 seeds
  with mean ± 95% CI.
