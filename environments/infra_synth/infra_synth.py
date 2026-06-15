"""infra_synth — RLVR environment for infrastructure-as-code synthesis.

STATUS: STUB. The environment logic is **work in progress** and is implemented
by a later subagent. This module only defines the Hub-discovery entrypoint
(`load_environment`) per the Prime Intellect `verifiers` Hub spec.

Intended behavior (once implemented)
------------------------------------
`load_environment` returns a `verifiers.Environment` for a single-turn task:

1. **Prompt.** The model is given an infrastructure spec (Dockerfile first; later
   `compose` / `terraform` / `k8s` / `ci-yaml`) and must emit the corresponding
   artifact as text.
2. **Verification.** The emitted artifact is graded by a pluggable verifier
   (`verifier.Verifier`):
     - the **local Docker** backend builds the Dockerfile and runs a smoke test
       (the weak baseline), or
     - the **Sentinel** backend runs hardened sandboxed execution checks
       (M2; nsjail, cgroups v2, no network).
   Verification produces a `verifier.VerifyResult`.
3. **Reward.** `verifier.reward.shape_reward(result, ...)` maps the
   `VerifyResult` to a scalar (build + smoke weighted, optional hack penalty).
   `VerifyResult.hack_flags` feeds the reward-hacking-via-verifier-exploitation
   study (contribution C3).

Discovery follows the bare `load_environment` function-name convention (no
`entry_points`). Module/dir name uses underscores (`infra_synth`); the
distribution name uses hyphens (`infra-synth`).

Expected kwargs (subject to change as the env is built)
-------------------------------------------------------
- ``backend``: which verifier to use ("docker" | "sentinel").
- ``num_examples`` / ``rollouts_per_example``: eval sizing (see
  ``[tool.verifiers.eval]`` defaults in this package's ``pyproject.toml``).
- ``dataset_split``, ``seed``, reward-shaping weights, etc.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # avoid importing the heavy `verifiers` dep at module load
    import verifiers as vf


def load_environment(**kwargs: Any) -> "vf.Environment":
    """Construct and return the `infra_synth` RLVR environment.

    WIP stub. Raises ``NotImplementedError`` until the environment logic is
    implemented by a later subagent. See the module docstring for the intended
    behavior and the expected ``kwargs``.
    """
    raise NotImplementedError(
        "infra_synth.load_environment is not implemented yet (WIP). "
        "The environment logic, datasets, and verifier wiring are built by a "
        "later subagent. See the module docstring for the intended design."
    )
