# tests/ — 433 tests, torch-free

Baseline: **433 pass** (one GSM8K test skips if the HF download is offline). Run
the whole suite with `uv run pytest -q`.

## Torch-free discipline

The suite must run without the GPU stack. Keep it that way:

- **Stub the `Verifier`** — tests construct fake verifiers/results against the
  frozen `verifier.types` contract rather than running real backends.
- **`importorskip` heavy optional deps** — gate any test that needs `verifiers`
  / `datasets` behind `pytest.importorskip(...)` so a CPU-only box still passes.
- Never import `torch`/`trl`/`vllm` at module top level in a test.

## Coverage map

- Contract: `test_contracts.py` (pins the FROZEN `verifier/types.py`).
- Verifier: `test_backends.py` · `test_sentinel_client.py` · `test_reward.py` ·
  `test_checks.py`.
- Env: `test_tasks.py` · `test_parser.py` · `test_gold.py` · `test_scaffold.py`
  (NS-2 app scaffold + build-context wiring) · `test_infra_env_integration.py` ·
  `test_infra_reward.py`.
- Training: `test_training_data.py` · `test_training_rewards.py` ·
  `test_training_seeds.py`.
- Eval: `test_passk.py` · `test_parity.py` · `test_throughput_mock.py`.
- Analysis: `test_reward_hacking.py`.

## Run subsets

```bash
uv run pytest tests/test_backends.py -q          # one file
uv run pytest tests/test_passk.py::<name> -q     # one test
uv run pytest -q -k reward                        # by keyword
```
