#!/usr/bin/env bash
# Crucible SessionStart hook — make the uv workspace test-ready on session start.
# Non-blocking by design: always exits 0 so a sync hiccup never blocks the session.
set -uo pipefail

cd "${CLAUDE_PROJECT_DIR:-.}" || exit 0

if command -v uv >/dev/null 2>&1; then
  if uv sync --extra dev >/dev/null 2>&1; then
    echo "Crucible: uv workspace synced (dev extras) — pytest/ruff/mypy ready."
  else
    echo "Crucible: 'uv sync --extra dev' did not complete; run it manually if tests fail." >&2
  fi
else
  echo "Crucible: uv not found — install from https://astral.sh/uv, then run 'uv sync --extra dev'." >&2
fi

exit 0
