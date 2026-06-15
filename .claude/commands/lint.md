---
description: Lint and type-check Crucible (ruff --fix, then mypy).
allowed-tools: Bash
---

Run ruff (autofix) then mypy:

```
uv run ruff check . --fix && uv run mypy verifier training eval
```

Keep the tree ruff-clean. `verifier/types.py` is a frozen contract excluded from rewrites in pyproject — don't fight it.
