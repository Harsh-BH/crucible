# eval

Benchmarking and the M2 proof for Crucible. See [`../docs/DESIGN.md`](../docs/DESIGN.md)
for the full design (verifier-backend strategy + eval protocol).

- **`passk.py`** — the unbiased `pass@k` estimator (Chen et al.,
  arXiv:2107.03374), numerically stable product form. Stdlib only.
- **`benchmark.py`** — `evaluate(...)` / `evaluate_multi_seed(...)` over the
  contamination-resistant `infra_synth` **test** split: build/smoke pass rates,
  `pass@1`/`pass@k`, and aggregate `HackFlags` rates. `generate_fn(prompt, n)` is
  injected; `make_openai_generate_fn(...)` wraps a vLLM/OpenAI-compatible server.
- **`throughput.py`** — M2 "throughput measured": fans verifications through
  `SentinelVerifier` at several concurrency levels and reports throughput +
  p50/p90/p99 latency. Injectable `httpx` transport (runs against a
  `MockTransport` in CI or a live Sentinel by URL).
- **`parity.py`** — M2 "matches a local-check baseline": runs each artifact
  through `local-py` **and** `sentinel` and reports `(build_ok, smoke_ok)`
  agreement + disagreements.

CLIs: `python -m eval.benchmark --base-url … --model …`,
`python -m eval.throughput --mock --n 256 --concurrency 1 4 16 64`,
`python -m eval.parity --base-url http://localhost:8080`. A **live** Sentinel
requires `make up` in the Sentinel repo (port 8080); `--mock` needs no server.

Imports only stdlib + `verifier` + (lazily) `httpx`; matplotlib is never imported
here. Tests: `tests/test_passk.py`, `tests/test_parity.py`,
`tests/test_throughput_mock.py`.
