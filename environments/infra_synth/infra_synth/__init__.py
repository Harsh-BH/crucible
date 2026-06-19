"""infra_synth — Hub-spec RLVR environment package.

The Hub-discovery entrypoint is ``infra_synth.load_environment`` (wired in
``infra_synth.environment``). The vf-free helper modules (``tasks`` / ``parser``
/ ``gold`` / ``scaffold``) are re-exported here so consumers get a clean public
API without reaching into submodules.

Importing this package is ``verifiers``-free: ``environment`` only imports
``verifiers`` lazily inside ``load_environment``.
"""
from __future__ import annotations

from . import gold, impossible, parser, scaffold, tasks
from .environment import load_environment
from .gold import gold_ci_yaml, gold_compose, gold_dockerfile, gold_k8s, gold_terraform
from .impossible import MUTATIONS, Adversary, adversarial_corpus, impossible_tasks
from .parser import (
    extract_ci_yaml,
    extract_compose,
    extract_dockerfile,
    extract_k8s,
    extract_terraform,
)
from .scaffold import app_scaffold
from .tasks import (
    CI_YAML_SYSTEM_PROMPT,
    COMPOSE_SYSTEM_PROMPT,
    K8S_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    TERRAFORM_SYSTEM_PROMPT,
    build_verify_spec,
    generate_tasks,
)

__all__ = [
    "load_environment",
    "tasks",
    "parser",
    "gold",
    "scaffold",
    "impossible",
    "generate_tasks",
    "build_verify_spec",
    "gold_dockerfile",
    "gold_compose",
    "gold_ci_yaml",
    "gold_terraform",
    "gold_k8s",
    "extract_dockerfile",
    "extract_compose",
    "extract_ci_yaml",
    "extract_terraform",
    "extract_k8s",
    "app_scaffold",
    "impossible_tasks",
    "adversarial_corpus",
    "Adversary",
    "MUTATIONS",
    "SYSTEM_PROMPT",
    "COMPOSE_SYSTEM_PROMPT",
    "CI_YAML_SYSTEM_PROMPT",
    "TERRAFORM_SYSTEM_PROMPT",
    "K8S_SYSTEM_PROMPT",
]
