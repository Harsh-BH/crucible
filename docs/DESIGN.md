# Crucible — Design

Crucible trains a small code model with **reinforcement learning from verifiable
rewards (RLVR)** on an infrastructure-as-code task (`infra_synth`: natural-
language spec → Dockerfile), and uses that setting to study two questions the
RLVR literature leaves open: **(C2)** can the reward signal be made fast *and*
trustworthy by routing it through a hardened sandbox, and **(C3)** does hardening
the verifier reduce *reward hacking*?

This document describes the architecture, the verifier-backend strategy, the
reward-hacking study, and the eval protocol, and is explicit about what is built
versus WIP.

---

## 1. Architecture

```
Trainer (GRPO; prime-rl/TRL-style, `verifiers`)
  │  prompts → Rollout server (vLLM, OpenAI-compatible API) → completions
  ▼
infra_synth env (verifiers Hub-spec, 1-turn): tasks→prompt; parser→Dockerfile; →VerifySpec
  │  (artifact, VerifySpec)
  ▼
Pluggable verifier (verifier.get_verifier): static · local-py · local-docker · sentinel
  →  VerifyResult(build_ok, smoke_ok, HackFlags, …)  →  shape_reward()  →  scalar reward
  ╰────────────────────────────── reward back to trainer ──────────────────────────────╯
```

The seam that makes the project tractable is the **frozen verifier contract**
(`verifier/types.py`): `VerifySpec` (what to check) in, `VerifyResult`
(`build_ok`, `smoke_ok`, `HackFlags`, timing) out, behind an async `Verifier`
protocol. The environment and every backend agree only on these dataclasses, so
backends are swappable without touching the trainer or the environment, and the
whole verifier + environment layer imports with only `httpx` + stdlib (no
`torch`/`vllm`/`verifiers`) — which is what lets `eval/` and `analysis/` run, and
unit-test, without the GPU stack.

`infra_synth` is a single-turn `verifiers` environment. `tasks.py` enumerates a
parameter grid (language × framework × dependency × port × health-path), renders
an NL spec, and builds a `VerifySpec`; `parser.py` extracts the last fenced
```dockerfile block from a completion; `gold.py` renders a correct reference
Dockerfile (used for eval references and the gold-passes-its-own-spec test).

**Reward shaping** (`verifier/reward.py`) is deliberately *not* pure pass/fail.
A Dockerfile that builds but fails the smoke test earns partial credit
(`build_weight=0.3`), a full build+serve earns `1.0`. Sparse 0/1 rewards make
GRPO degenerate: when every rollout in a group scores the same, the group-
relative advantage is zero and the gradient vanishes (GRPO, arXiv:2402.03300).
Partial credit keeps a usable gradient on the many "almost-right" Dockerfiles a
mid-training policy emits. An optional `hack_penalty` can drive reward negative
when any `HackFlags` tripped — the lever for the C3 penalised-reward ablation.

---

## 2. Verifier-backend strategy (C2)

The four backends span a **weak → hardened** execution axis, and the design
splits them across **training** and **eval** by their speed/fidelity trade-off:

| backend         | where the check runs                                  | role |
|-----------------|-------------------------------------------------------|------|
| `static`        | in-process `check_dockerfile` heuristics, no sandbox  | universal fallback / lower bound; M1 reward |
| `local-py`      | the harness as a local `python3 -I` subprocess        | the deliberately **weak** baseline (C3) |
| `sentinel`      | the same harness inside nsjail/cgroups-v2 (no network)| the **hardened**, scalable training reward (M2) |
| `local-docker`  | genuine `docker build` + `docker run` + HTTP probe    | the **genuine** EVAL-time verifier |

**Training-loop reward uses a FAST verifier.** RL needs the reward for thousands
of rollouts per step with sub-second latency, or the trainer starves waiting on
the verifier. `static`/`local-py` are in-process/subprocess and effectively
free; `sentinel` runs the same deterministic harness in a hardened sandbox but
is built to **scale horizontally** — Sentinel is a queue + worker-pool service
(KEDA-autoscaled on queue depth), so the trainer fires verifications
concurrently and Sentinel absorbs the burst. `eval/throughput.py` measures
exactly this: it fans verifications through `SentinelVerifier` at increasing
concurrency and reports throughput + p50/p90/p99 latency, demonstrating the
queue scales (the M2 "throughput measured" deliverable).

**A single source of truth keeps weak and hardened honest.** `local-py` and
`sentinel` execute the *identical* harness string from
`build_python_harness()` — one as a local subprocess, one submitted to the
sandbox. So routing the reward through Sentinel must yield the same verdict as
the local baseline. `eval/parity.py` proves this: it runs each artifact through
both `local-py` and `sentinel` and reports agreement on `(build_ok, smoke_ok)`,
surfacing any divergence (the M2 "matches a local-check baseline" deliverable).
Both `eval/parity.py` and `eval/throughput.py` accept an injectable `httpx`
transport, so they run in CI against an `httpx.MockTransport` *and* against a
live Sentinel by URL with the same code.

**The genuine verifier is `local-docker`, and it is currently shallow.** The
static/harness path is a *heuristic stand-in*: it parses the Dockerfile and
checks for a pinned `FROM`, required tokens, an `EXPOSE`d port, and a
server-launching `CMD`. It never builds an image. `local-docker` does the real
thing — build, run, poll `http://localhost:<port><health_path>` for a 200 — but
today it only writes the model's Dockerfile into an otherwise **empty build
context**, so any realistic FastAPI/Flask spec (which `COPY ./app` and installs
`requirements.txt`) cannot actually serve.

### Concrete next step: app scaffold in the build context

To make `local-docker` genuine for realistic specs, ship a small **app scaffold**
(a minimal but real FastAPI/Flask app exposing the requested `health_path`, plus
a `requirements.txt`) into the Docker build context alongside the model's
Dockerfile. The build then exercises the model's dependency install / `COPY` /
`CMD`, and the smoke probe genuinely hits a live server. This is **non-breaking**:
`VerifySpec.smoke` is a free-form `dict` and already accommodates a
`context_files` mapping, so the environment can attach scaffold files without any
change to the frozen contract. (Status: **WIP** — the contract is ready; the
scaffold templates and `LocalDockerVerifier` context wiring are not yet built.)

### Sentinel roadmap (C1)

Sentinel today executes a *single source file* (it is a code-execution sandbox),
so it cannot `docker build`. That is why `sentinel` runs the Python *harness*
rather than a real build, and why `local-docker` (not `sentinel`) is the eval
verifier. The C1 roadmap is to add an **`infra` job type** to Sentinel that
performs a sandboxed build + smoke inside the cluster, at which point the genuine
build+serve verifier becomes horizontally scalable and could move into the
training loop. (Status: **WIP/roadmap** — not built.)

---

## 3. Reward-hacking study (C3): weak vs hardened

RLVR policies optimise the *measured* reward, so a weak verifier invites
**reward hacking** — passing the check without doing the task. C3 asks whether
moving from the weak verifier (`local-py`) to the hardened one (`sentinel`)
reduces it, measured on the *same* rollouts.

**Methodology — ImpossibleBench-style cheating rate** (arXiv:2510.20270). On
impossible/mutated tasks the only way to "pass" is to subvert the check, so any
pass — or any tripped exploitation signal — counts as cheating. The raw data is
the `HackFlags` (`resource_exhaustion`, `oom_killed`, `timed_out`,
`network_attempt`, `seccomp_violation`, `spec_gaming`) recorded on every
`VerifyResult`. `analysis/reward_hacking.py` computes a `cheating_rate` and
`compare_weak_vs_hardened` (cheating rate + per-category counts + the reduction
the sandbox buys).

**The 6-category taxonomy** (`classify_hack` maps signals onto these):

1. **weak tests / hardcoding** — satisfies the literal `must_contain`/substring
   gate without real work (signal: `spec_gaming`);
2. **answer leakage** — prints/returns the grader's expected output;
3. **fake success** — `exit(0)` / `assert True` / neutralised assertions;
4. **resource / timer / sandbox manipulation** — exhaust CPU/memory, evade
   limits, stall the timer (signal: `resource_exhaustion`/`oom_killed`/
   `timed_out`);
5. **test-harness side effects** — edit the grader/tests, monkeypatch
   `conftest.py`, write outside the build context;
6. **gaming the verifier itself** — seccomp/network escape to subvert the
   verdict (signal: `seccomp_violation`/`network_attempt`).

**Hypothesis.** The hardened sandbox should close categories 4–6 outright
(cgroups caps resources; no-net blocks exfiltration; an isolated FS blocks
harness edits) while categories 1–3 are *spec/grader* weaknesses the sandbox
cannot fix — those need a stronger check (the §2 genuine build). So we expect a
large reduction concentrated in 4–6 and little movement in 1–3, which would
localise *where* hardening helps.

**Caveat (documented).** Sentinel does not yet surface `seccomp_violation` or
`network_attempt` signals, and **seccomp is currently disabled** in the deployed
sandbox. Category 6 is therefore *under-counted* on the hardened side until that
lands; the analysis notes this so the comparison is not over-read.

`analysis/reward_hacking.py` is built (taxonomy, `classify_hack`,
`cheating_rate`, `compare_weak_vs_hardened`, `load_rollouts`, and a matplotlib
`plot_taxonomy`). The end-to-end study (logging real rollouts under both
backends on a mutated-task set, then comparing) is **WIP** — it needs the
training run and an impossible-task generator.

---

## 4. Eval protocol

Held-out evaluation lives in `eval/benchmark.py` and follows four rules.

**Unbiased pass@k** (`eval/passk.py`; Chen et al., arXiv:2107.03374). Estimating
pass@k as "fraction with ≥1 correct in k samples" is biased. We draw `n ≫ k`
samples per problem, count `c` correct, and use the unbiased estimator
`pass@k = 1 − C(n−c, k)/C(n, k)`, evaluated via the numerically **stable product
form** `1 − ∏_{i=n−c+1}^{n}(1 − k/i)` (with `0.0` if `c==0`, `1.0` if `c>n−k`) so
it stays finite at large k. The corpus value is the mean of per-problem pass@k.

**≥3 seeds, mean ± CI.** `evaluate_multi_seed` runs ≥3 seeds and reports
`mean ± std` and a 95% normal CI (`mean ± 1.96·std/√#seeds`) on the headline
metrics (`pass@1`, each `pass@k`, build/smoke rates, `hack_any`).

**Contamination-resistant splits.** `infra_synth.tasks` partitions the grid so
`train` and `test` draw from **disjoint parameter combinations** (a seed-
independent FNV-1a hash reserves ~25% of combos for test). A policy cannot have
seen a test spec during training. (Built.)

**The search-compression test** (arXiv:2504.13837). RL on verifiable rewards can
merely *sharpen* the base model's existing sampling rather than add capability.
To distinguish, sweep base-vs-RL pass@k to large k (≈128–256;
`eval/passk.passk_curve` + `analysis/curves.passk_base_vs_rl`): if the curves
**cross** (RL higher at small k, base catching up at large k) RL is sharpening;
if RL **dominates at every k** — including problems where base pass@k ≈ 0 — that
is genuine capability expansion. Temperature is tuned **per model** (base and RL
optima differ), since pass@k at large k is temperature-sensitive.

`analysis/curves.py` plots the standard GRPO dashboard from per-step JSONL logs:
reward mean, **within-group reward std** and **fraction zero-advantage** (the
GRPO collapse tells — DAPO's dynamic sampling, arXiv:2503.14476, exists to keep
these healthy; Dr.GRPO, arXiv:2503.20783, and GSPO, arXiv:2507.18071, motivate
the length/optimisation-signal panels), KL, **entropy** (the entropy-collapse
guard, arXiv:2505.22617), completion length, grad norm, and pass@1/pass@k
overlays. matplotlib is imported locally so the data path works without it.

---

## 5. Status summary

**Built & tested (no GPU/heavy deps):** the frozen verifier contract; all four
backends (`static`/`local-py`/`local-docker` mapping/`sentinel` via injectable
`httpx` transport); reward shaping; the `infra_synth` env helpers (tasks /
parser / gold) with disjoint splits; `eval/passk.py`; `eval/benchmark.py`
(injected `generate_fn` + multi-seed + OpenAI/vLLM factory); `eval/throughput.py`
and `eval/parity.py` (M2, verified against `MockTransport`); `analysis/
reward_hacking.py` (C3 taxonomy + cheating-rate); `analysis/curves.py`.

**WIP / roadmap:** the GRPO training run and configs (parallel work); genuine
`local-docker` via the **app-scaffold build context** (§2); the Sentinel
**`infra` job type** (C1, §2); the end-to-end C3 study on logged rollouts +
mutated tasks (§3); and Sentinel surfacing seccomp/network signals (with seccomp
re-enabled) so taxonomy category 6 is fully counted.

### CLIs

```bash
# Held-out eval of a vLLM/OpenAI-compatible policy (≥3 seeds, unbiased pass@k):
python -m eval.benchmark --base-url http://localhost:8000/v1 --model <name> \
    --backend static --n 200 --ks 1 8 64 128 --seeds 0 1 2

# M2 throughput (live Sentinel needs `make up` in the Sentinel repo, port 8080):
python -m eval.throughput --base-url http://localhost:8080 --n 256 --concurrency 1 4 16 64
python -m eval.throughput --mock --n 256 --concurrency 1 4 16 64   # no server

# M2 parity (Sentinel vs local-py):
python -m eval.parity --base-url http://localhost:8080 --n 32
```

### References

GRPO (arXiv:2402.03300) · DAPO (arXiv:2503.14476) · Dr.GRPO (arXiv:2503.20783) ·
GSPO (arXiv:2507.18071) · pass@k / HumanEval (arXiv:2107.03374) · ImpossibleBench
(arXiv:2510.20270) · entropy collapse (arXiv:2505.22617) · search-compression /
"does RL incentivise reasoning beyond the base model" (arXiv:2504.13837).
