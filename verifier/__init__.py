"""Crucible verifier layer — public API.

This package defines the *stable contract* that every verifier backend and the
`infra_synth` environment's reward functions agree on. The contract lives in
:mod:`verifier.types` and is re-exported here.

WHAT EXISTS NOW
---------------
Only the frozen data contracts (:mod:`verifier.types`). Everything else is
added by later work and MUST conform to the ``Verifier`` protocol below.

WHAT LANDS LATER (do not assume these import yet)
-------------------------------------------------
- ``verifier.backends``        : ``LocalDockerVerifier`` (builds a Dockerfile +
                                 runs a smoke test locally) and ``LocalPyVerifier``.
                                 Each implements the ``Verifier`` protocol.
- ``verifier.sentinel_client`` : ``SentinelClient`` (thin async HTTP client for
                                 the Sentinel sandbox) + ``SentinelVerifier``
                                 (wraps it as a ``Verifier``).
- ``verifier.reward``          : ``shape_reward(result, *, build_weight=0.3,
                                 smoke_weight=0.7, hack_penalty=0.0,
                                 binary=False) -> float``.

Sentinel API targeted by ``sentinel_client`` (see README "Verifier backends"):
``POST /api/v1/submissions`` -> 202 ``{job_id, status}``; poll
``GET /api/v1/submissions/:id``; base URL default ``http://localhost:8080``,
prefix ``/api/v1``; request body ``{language, source_code, stdin,
time_limit_ms?, memory_limit_kb?}``; terminal statuses SUCCESS /
COMPILATION_ERROR / RUNTIME_ERROR / TIMEOUT / MEMORY_LIMIT_EXCEEDED /
INTERNAL_ERROR.

Import :mod:`verifier.types` directly if you only need the contracts and want to
avoid touching the backend modules.
"""
from __future__ import annotations

from .types import (
    ArtifactKind,
    HackFlags,
    ResourceLimits,
    Verifier,
    VerifyResult,
    VerifySpec,
)

__all__ = [
    "ArtifactKind",
    "ResourceLimits",
    "VerifySpec",
    "HackFlags",
    "VerifyResult",
    "Verifier",
]
