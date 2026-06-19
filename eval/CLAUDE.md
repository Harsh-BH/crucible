# eval/ — held-out evaluation & M2 proof

Torch-free: only `httpx` + stdlib. Both M2 tools accept an injectable `httpx`
transport, so they run in CI against a mock *and* against a live Sentinel by URL
with the same code.

## Files

- `passk.py` — **unbiased pass@k** (`pass_at_k`, Chen et al.): draw `n ≫ k`,
  count `c` correct, use `1 − C(n−c,k)/C(n,k)` via the stable product form.
  `passk_curve` sweeps a list of k.
- `benchmark.py` — held-out eval. `evaluate(...)` takes an **injected
  `generate_fn`** (`make_openai_generate_fn` builds one for vLLM/OpenAI);
  `evaluate_multi_seed` runs ≥3 seeds and reports mean ± 95% CI on pass@1,
  pass@k, build/smoke rates, `hack_any`. Resolves tasks from `infra_synth.tasks`.
- `parity.py` (M2) — `parity_report` runs each artifact through **`local-py` vs
  `sentinel`** and reports agreement on `(build_ok, smoke_ok)`. Run:
  `python -m eval.parity --base-url http://localhost:8080`.
- `throughput.py` (M2) — `benchmark_sentinel` fans verifications at increasing
  concurrency and reports throughput + p50/p90/p99. `make_mock_transport()`
  (`httpx.MockTransport`) runs it entirely locally:
  `python -m eval.throughput --mock`.
- `c3_study.py` (C3) — end-to-end weak-vs-hardened reward-hacking study runner.
  `run_c3_study(weak, hardened, trials=...)` grades impossible tasks + the
  adversarial corpus through **`local-py` vs `sentinel`** on the same trials and
  returns the `compare_weak_vs_hardened` taxonomy, an undeserved-pass metric, and
  a per-category breakdown. `make_verifiers` wires the pair (injectable Sentinel
  transport); `default_trials` builds trials from `infra_synth` (imported
  lazily); `make_mock_sentinel_transport` simulates the hardened sandbox. Run:
  `python -m eval.c3_study --mock --n 12`.

## Notes

- pass@k base-vs-RL crossover (search-compression) is plotted from
  `passk.passk_curve` + `analysis.curves.passk_base_vs_rl`.
- See `docs/DESIGN.md` §4 (eval protocol) and `docs/ROADMAP.md` (M2 = NS-4).
