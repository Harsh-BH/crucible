---
description: Run the Crucible test suite (243 tests; torch-free).
argument-hint: "[pytest args, e.g. -k pattern or a path]"
allowed-tools: Bash
---

Run the suite with uv:

```
uv run pytest -q $ARGUMENTS
```

Baseline: 243 passed, 1 skipped (the skip needs an HF network download). If uv isn't synced yet, run `uv sync --extra dev` first.
