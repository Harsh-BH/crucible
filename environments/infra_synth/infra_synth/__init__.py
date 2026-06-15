"""infra_synth — Hub-spec RLVR environment package.

The Hub-discovery entrypoint is ``infra_synth.load_environment`` (wired in
``infra_synth.environment``). The vf-free helper modules (``tasks`` / ``parser``
/ ``gold``) are re-exported here so consumers get a clean public API without
reaching into submodules.

Importing this package is ``verifiers``-free: ``environment`` only imports
``verifiers`` lazily inside ``load_environment``.
"""
from __future__ import annotations

from . import gold, parser, tasks
from .environment import load_environment
from .gold import gold_dockerfile
from .parser import extract_dockerfile
from .tasks import SYSTEM_PROMPT, build_verify_spec, generate_tasks

__all__ = [
    "load_environment",
    "tasks",
    "parser",
    "gold",
    "generate_tasks",
    "build_verify_spec",
    "gold_dockerfile",
    "extract_dockerfile",
    "SYSTEM_PROMPT",
]
