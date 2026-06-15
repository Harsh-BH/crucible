"""Crucible verifier layer — public API.

This package defines the *stable contract* that every verifier backend and the
`infra_synth` environment's reward functions agree on. The contract lives in
:mod:`verifier.types` and is re-exported here.

PUBLIC API
----------
- :mod:`verifier.types`        : the frozen data contracts (re-exported here).
- ``get_verifier(name, ...)``  : factory -> a backend implementing ``Verifier``.
                                 Names: ``static`` | ``local-py`` |
                                 ``local-docker`` | ``sentinel``.
- ``verifier.backends``        : ``StaticVerifier`` (in-process static analysis,
                                 the fallback), ``LocalPyVerifier`` (weak local
                                 subprocess baseline), ``LocalDockerVerifier``
                                 (genuine build + smoke probe).
- ``verifier.sentinel_client`` : ``SentinelClient`` (async HTTP client for the
                                 Sentinel sandbox) + ``SentinelVerifier`` (the
                                 hardened execution path).
- ``verifier.reward``          : ``shape_reward(result, *, build_weight=0.3,
                                 smoke_weight=0.7, hack_penalty=0.0,
                                 binary=False) -> float`` and
                                 ``result_to_metrics(result)``.
- ``verifier.smoke.checks``    : ``check_dockerfile`` / ``build_python_harness``
                                 / ``parse_harness_output`` (stdlib-only check
                                 logic shared by every backend).

This package stays importable with only ``httpx`` + stdlib (no
``verifiers``/``torch``/``vllm``/``datasets``).

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

from .backends import (
    LocalDockerVerifier,
    LocalPyVerifier,
    StaticVerifier,
    get_verifier,
)
from .reward import result_to_metrics, shape_reward
from .sentinel_client import SentinelClient, SentinelVerifier
from .smoke.checks import build_python_harness, check_dockerfile, parse_harness_output
from .types import (
    ArtifactKind,
    HackFlags,
    ResourceLimits,
    Verifier,
    VerifyResult,
    VerifySpec,
)

__all__ = [
    # frozen contracts
    "ArtifactKind",
    "ResourceLimits",
    "VerifySpec",
    "HackFlags",
    "VerifyResult",
    "Verifier",
    # factory + backends
    "get_verifier",
    "StaticVerifier",
    "LocalPyVerifier",
    "LocalDockerVerifier",
    "SentinelClient",
    "SentinelVerifier",
    # reward shaping
    "shape_reward",
    "result_to_metrics",
    # smoke check helpers
    "check_dockerfile",
    "build_python_harness",
    "parse_harness_output",
]
