---
description: Run the Crucible test suite (433 tests; torch-free).
argument-hint: "[pytest args, e.g. -k pattern or a path]"
allowed-tools: Bash
---

Run the suite with uv:

```
uv run pytest -q $ARGUMENTS
```

Baseline: 433 passed (one GSM8K test skips if the HF download is offline). If uv isn't synced yet, run `uv sync --extra dev` first.
