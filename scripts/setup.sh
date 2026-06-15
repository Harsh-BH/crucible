#!/usr/bin/env bash
# Crucible bootstrap.
#
# Installs uv (if missing), syncs the core + dev dependency groups, and prints
# next-step commands. The heavy/GPU training stack is intentionally NOT synced
# here (see `--extra train` below).
#
# Usage:   bash scripts/setup.sh
# Tip:     chmod +x scripts/setup.sh   # to run it as ./scripts/setup.sh
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

# 1. Ensure uv is available.
if ! command -v uv >/dev/null 2>&1; then
  echo ">> uv not found; installing..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # uv installs to ~/.local/bin by default; make it available for this shell.
  export PATH="$HOME/.local/bin:$PATH"
fi
echo ">> using uv: $(command -v uv) ($(uv --version))"

# 2. Sync the workspace with developer tooling (CPU-only).
echo ">> uv sync --extra dev"
uv sync --extra dev

# 3. Next steps.
cat <<'EOF'

==============================================================================
Crucible is set up. Next steps:

  # Run the contract tests
  uv run pytest -q

  # Lint
  uv run ruff check .

  # GPU training stack (pulls vllm/verifiers/wandb/datasets — GPU-oriented):
  uv sync --extra train

  # Work with the infra_synth Hub environment (once its logic is implemented):
  uv run vf-install infra-synth     # install the env into the active venv
  uv run vf-eval infra-synth        # run an eval (defaults: 10 examples x 4)
==============================================================================
EOF
