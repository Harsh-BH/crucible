"""Frozen data contracts for the Crucible verifier layer.

Shared by verifier backends (which IMPLEMENT `Verifier`) and the `infra_synth`
environment reward functions (which CONSUME these types). Keep this stdlib-only
so it can be imported anywhere without heavy dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable


class ArtifactKind(str, Enum):
    DOCKERFILE = "dockerfile"
    COMPOSE = "compose"
    TERRAFORM = "terraform"
    K8S = "k8s"
    CI_YAML = "ci-yaml"
    PYTHON = "python"  # routing raw code execution through Sentinel (M2)


@dataclass(slots=True)
class ResourceLimits:
    wall_s: int = 30
    mem_mb: int = 512
    pids: int = 64
    cpus: float = 1.0


@dataclass(slots=True)
class VerifySpec:
    """Describes a single acceptance check for one task."""

    spec_id: str
    kind: ArtifactKind
    smoke: dict[str, Any] = field(default_factory=dict)
    limits: ResourceLimits = field(default_factory=ResourceLimits)


@dataclass(slots=True)
class HackFlags:
    """Verifier-exploitation signals (raw data for the reward-hacking study)."""

    resource_exhaustion: bool = False
    oom_killed: bool = False
    timed_out: bool = False
    network_attempt: bool = False
    seccomp_violation: bool = False
    spec_gaming: bool = False

    def any(self) -> bool:
        return any(
            (
                self.resource_exhaustion,
                self.oom_killed,
                self.timed_out,
                self.network_attempt,
                self.seccomp_violation,
                self.spec_gaming,
            )
        )


@dataclass(slots=True)
class VerifyResult:
    """Backend-agnostic outcome of verifying one artifact."""

    build_ok: bool = False
    smoke_ok: bool = False
    exit_code: int | None = None
    status: str = ""
    stdout_tail: str = ""
    stderr_tail: str = ""
    wall_s: float = 0.0
    mem_mb: float = 0.0
    backend: str = ""
    hack_flags: HackFlags = field(default_factory=HackFlags)
    reward: float | None = None  # filled by verifier.reward.shape_reward(); None until then
    raw: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Verifier(Protocol):
    """Any verifier backend. Implementations live in verifier.backends / verifier.sentinel_client."""

    name: str

    async def verify(self, artifact: str, spec: VerifySpec) -> VerifyResult: ...
