# infra_synth pass rate: Qwen2.5-0.5B-Instruct, REAL local verifier

This measures the project's actual thesis metric — not the M1 gsm8k reproduction
(`results/m1_gsm8k_6gb/`, which showed reward does not rise on a task infra_synth
doesn't care about). README.md's pitch is a policy that produces *verifiable*
infra-as-code, graded by actually building/running/validating the artifact. This
directory is that grading, run for real, with **no training** — a single-shot
(greedy, n=1) generation + verification pass over held-out `infra_synth` test-split
tasks, scored by the genuine local backends (`verifier.get_verifier("local")`, the
kind-aware dispatcher in `verifier/verifier/backends.py:900`):

| kind | genuine check | verifies |
|---|---|---|
| DOCKERFILE | `docker build` + `docker run` + HTTP smoke probe | `LocalDockerVerifier` |
| COMPOSE | `docker compose up` + HTTP smoke probe | `LocalComposeVerifier` |
| TERRAFORM | `terraform init` + `terraform validate` | `LocalTerraformVerifier` |
| K8S | `kubeconform -strict` | `LocalK8sVerifier` |
| CI_YAML | **no genuine backend exists** — falls back to `StaticVerifier` (must_contain + step-token heuristics only) | not a real build/run check |

Commands (repo root, `.venv-train` for generation — has torch+transformers+CUDA but
not `verifier`/`infra_synth` on `sys.path` by default, hence the explicit
`PYTHONPATH`):

```
PYTHONPATH=verifier:environments/infra_synth:. .venv/bin/python scripts/infra_synth_gold_sanity.py
PYTHONPATH=verifier:environments/infra_synth:. .venv-train/bin/python scripts/infra_synth_pass_rate.py \
    --kinds dockerfile compose terraform k8s ci-yaml --n 8 4 15 15 15
PYTHONPATH=verifier:environments/infra_synth:. .venv/bin/python scripts/infra_synth_impossible_probe.py
```

Model: `Qwen/Qwen2.5-0.5B-Instruct` (HF cache, no download). Greedy decoding
(`do_sample=False`), `max_new_tokens=768`, one sample per task (real docker /
terraform / kubeconform runs are too slow on a 6 GB laptop GPU + ~3 GB free disk
for a pass@k sweep — see TIMING below). Ports 8080/3000 are excluded from the
DOCKERFILE/COMPOSE task pool because the live `sentinel-api`/`sentinel-frontend`
containers already bind those host ports (unrelated to model/gold quality — do not
touch that stack per the task's hard constraint).

## Gold sanity (`gold_sanity.json`)

Does the gold (honest) reference pass its OWN spec under the real verifier, per
kind? **5/5 kinds pass** (dockerfile, compose, terraform, ci-yaml all first-try;
k8s passed on 2 of 3 attempts — one run hit a 120s `kubeconform` timeout, most
likely its default network schema fetch with no `-cache`/offline flag configured
in `LocalK8sVerifier._kubeconform`; a clean retry on the same gold artifact passed
in 0.65s). This validates the harness: the genuine backends are not spuriously
failing correct artifacts.

## Model pass rate (`summary.json`, per-kind `*.json`)

n=1 (greedy) per task, test split, seed 0:

| kind | N | build_pass_rate | smoke_pass_rate | mean wall/sample |
|---|---|---|---|---|
| dockerfile | 8 | 0.25 (2/8) | **0.0** (0/8) | 9.3s |
| compose | 4 | 0.0 (0/4) | **0.0** (0/4) | 0.05s |
| terraform | 15 | 0.0 (0/15) | **0.0** (0/15) | 0.68s |
| k8s | 15 | 0.533 (8/15) | **0.533** (8/15) | 48.3s* |
| ci-yaml | 15 | 1.0 (15/15) | 0.0 (0/15) | ~0s |

\* k8s wall time is anomalously high vs. the 0.5-1s gold-check baseline and is
network-bound (unpinned `kubeconform` schema fetch, see above) — not comparable to
the other kinds' wall time.

**Genuine-backend (dockerfile+compose+terraform+k8s) combined: 8/42 smoke-pass =
19.0%.** ci-yaml is excluded from that figure — its `StaticVerifier` fallback is
not a real build/run/validate check (see table above) and its `build_ok=1.0` /
`smoke_ok=0.0` split reflects two different static heuristics (`must_contain`
passed; the separate `required_steps` token-matcher did not — see WHAT CANNOT BE
CLAIMED in the probe report for a concrete false-negative in that heuristic).

k8s (53.3%) is the standout: kubeconform-strict schema validation is a much lower
bar for a 0.5B model to clear than "produce HCL that satisfies `terraform
validate`'s full semantic model" or "produce a Dockerfile that builds AND actually
serves HTTP 200 on the right port." Terraform and (smoke-wise) compose/dockerfile
are effectively 0% at this model scale, n=1, greedy.

## Impossible-task probe (`impossible_gold.json`)

`infra_synth.impossible.impossible_tasks()` mutates a spec's `must_contain` list to
make it unsatisfiable. Ran the GOLD (honest) reference for each kind through the
mutated spec via the same genuine `local` backend (Sentinel untouched, no `--mock`,
no training). Result: **12/15 "undeserved passes."** This is NOT a security hole —
it is scope mismatch, confirmed by grepping `verifier/verifier/backends.py` for
`must_contain` (zero hits): the genuine dynamic backends (docker/compose/terraform/
k8s) never check `must_contain` at all — only `StaticVerifier` does (`ci-yaml`
correctly rejected 3/3). `must_contain` is documented in
`environments/infra_synth/infra_synth/tasks.py:246` as "a cheap static gate **the
static backend** can enforce" — by design, not by omission. Practically: the
`impossible_tasks()` false-positive probe, as currently written, only tests
whether `StaticVerifier` can be fooled by token-parroting; pointed at the genuine
backends it can't produce a real unsatisfiable task for 4 of 5 kinds, because those
backends grade real behavior that the mutation doesn't touch (e.g.
`_mutate_contradictory_port` demands a decoy `EXPOSE` line while the real HTTP
probe still hits the true port regardless of what the Dockerfile's `EXPOSE`
documents). `run_c3_study`/`default_trials` (`eval/c3_study.py`) sidesteps this —
it pairs `local-py` (weak) vs `sentinel` (hardened), never the genuine
docker/terraform/k8s backends vs a mutated spec — so this specific gap was never
previously exercised.
