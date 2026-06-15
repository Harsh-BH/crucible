"""Contract tests for the frozen `verifier.types` data shapes.

Dependency-light on purpose: imports only from `verifier` / `verifier.types`
(no `verifiers` / `vllm` / `torch`). These assertions pin the contract that
later subagents depend on.
"""
from __future__ import annotations

from verifier import (
    ArtifactKind,
    HackFlags,
    ResourceLimits,
    Verifier,
    VerifyResult,
    VerifySpec,
)
from verifier import types as vtypes


def test_reexports_match_types_module() -> None:
    # The public re-exports are the same objects as in verifier.types.
    assert ArtifactKind is vtypes.ArtifactKind
    assert ResourceLimits is vtypes.ResourceLimits
    assert VerifySpec is vtypes.VerifySpec
    assert HackFlags is vtypes.HackFlags
    assert VerifyResult is vtypes.VerifyResult
    assert Verifier is vtypes.Verifier


def test_artifact_kind_values() -> None:
    # str-Enum members compare equal to their string values.
    assert ArtifactKind.DOCKERFILE == "dockerfile"
    assert ArtifactKind.COMPOSE == "compose"
    assert ArtifactKind.TERRAFORM == "terraform"
    assert ArtifactKind.K8S == "k8s"
    assert ArtifactKind.CI_YAML == "ci-yaml"
    assert ArtifactKind.PYTHON == "python"


def test_resource_limits_defaults() -> None:
    rl = ResourceLimits()
    assert rl.wall_s == 30
    assert rl.mem_mb == 512
    assert rl.pids == 64
    assert rl.cpus == 1.0


def test_verify_spec_defaults() -> None:
    spec = VerifySpec(spec_id="t1", kind=ArtifactKind.DOCKERFILE)
    assert spec.spec_id == "t1"
    assert spec.kind is ArtifactKind.DOCKERFILE
    assert spec.smoke == {}
    assert isinstance(spec.limits, ResourceLimits)
    assert spec.limits.wall_s == 30


def test_hack_flags_defaults_and_any() -> None:
    hf = HackFlags()
    assert hf.any() is False
    assert hf.resource_exhaustion is False
    assert hf.network_attempt is False
    # Tripping any single flag flips .any().
    hf.spec_gaming = True
    assert hf.any() is True


def test_verify_result_defaults() -> None:
    res = VerifyResult()
    assert res.build_ok is False
    assert res.smoke_ok is False
    assert res.exit_code is None
    assert res.status == ""
    assert res.wall_s == 0.0
    assert res.mem_mb == 0.0
    assert res.backend == ""
    assert res.reward is None
    assert res.raw == {}
    assert isinstance(res.hack_flags, HackFlags)
    assert res.hack_flags.any() is False


def test_verifier_runtime_checkable_protocol() -> None:
    class DummyVerifier:
        name = "dummy"

        async def verify(self, artifact: str, spec: VerifySpec) -> VerifyResult:
            return VerifyResult(backend=self.name, build_ok=True)

    obj = DummyVerifier()
    assert isinstance(obj, Verifier)

    # A class missing `verify` must NOT satisfy the protocol.
    class NotAVerifier:
        name = "nope"

    assert not isinstance(NotAVerifier(), Verifier)
