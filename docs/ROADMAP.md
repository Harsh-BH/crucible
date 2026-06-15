# Crucible — Roadmap & Next Steps

> Forward plan and execution checklist. For *why* the system is shaped the way it
> is (verifier strategy, weak-vs-hardened axis, eval protocol), see
> [`DESIGN.md`](./DESIGN.md). For the high-level pitch and layout, see the
> [root README](../README.md).
>
> **Snapshot:** 2026-06-15 · branch `claude/wizardly-mayer-cl2h56` · commit `67afe73`
> · 243 tests passing · ruff-clean.

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

**Scaffolded but NOT yet executed (need real infra):**
- A real GPU GRPO run (M1 "reward rises, ≥3 seeds").
- Throughput/parity against a *live* Sentinel (M2 numbers).
- Genuine `local-docker` build+smoke for realistic apps (needs the app scaffold, below).

---

## 1. Immediate next steps (prioritized)

| # | Task | Why | Acceptance criteria | Effort | Blocks |
|---|------|-----|---------------------|--------|--------|
| **NS-1** | **Fix `crucible-verifier` wheel packaging** | `verifier/` is a flat package; a built wheel ships modules at top level, so `import verifier` fails from a wheel. It's `infra_synth`'s dependency → Hub publish is blocked until fixed. | A built `crucible-verifier` wheel installs into a clean venv and `import verifier`, `from verifier.types import VerifyResult` work *non-editable*; 243 tests still green. | S | Hub publish (M6) |
| **NS-2** | **M3 app scaffold for genuine build+smoke** | `LocalDockerVerifier` builds a context with only the `Dockerfile`; realistic Dockerfiles that `COPY` app code fail. | `environments/infra_synth/.../scaffold.py: app_scaffold(info) -> dict[str,str]` (requirements + a minimal server exposing `/health`); `build_verify_spec` puts it in `smoke["context_files"]`; `LocalDockerVerifier` writes those files into the build context; a gold Dockerfile **builds + serves `/health` → 200** under a real Docker daemon. | M | M3 |
| **NS-3** | **Execute M1 on a GPU** | Prove the loop: reward rises and is stable across ≥3 seeds. | `reverse-text` climbs ~0.05→0.8 (loop sanity), then `gsm8k` on Qwen3-1.7B shows rising reward across seeds {0,1,2}; report mean±95% CI via `training/seeds.py`. | M | M3+ |
| **NS-4** | **Stand up Sentinel + execute M2** | Real parity + throughput numbers (the systems story). | `make up` in the Sentinel repo (:8080); `python -m eval.parity --base-url ...` shows local-py↔sentinel agreement; `python -m eval.throughput --base-url ...` reports throughput + p50/p90/p99. | S–M | M5 |

**Recommended order:** NS-1 (small, unblocks publishing) → NS-2 (completes C2's "genuine reward") → then NS-3/NS-4 in parallel once compute + a Sentinel host are available.

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
