"""Unit tests for ``training.rewards`` — torch-free GRPO reward callables.

Covers:
- :func:`gsm8k_reward` correctness (boxed / ``####`` -> 1.0, wrong -> 0.0),
- :func:`format_reward` boxed-format detection,
- :func:`make_infra_synth_reward` via an INJECTED stub Verifier (no live
  Sentinel / docker), including concurrent verification of a batch.
"""
from __future__ import annotations

import pytest

from training.rewards import (
    format_reward,
    gsm8k_reward,
    make_infra_synth_reward,
)
from verifier.types import VerifyResult, VerifySpec

# ``make_infra_synth_reward``'s closure imports ``infra_synth.tasks`` /
# ``infra_synth.parser`` from the installed env package (``pip install -e
# ./environments/infra_synth``); these tests skip cleanly if it is absent.


# ---------------------------------------------------------------------------
# gsm8k_reward
# ---------------------------------------------------------------------------
def test_gsm8k_reward_correct_boxed() -> None:
    out = gsm8k_reward(["q"], [r"reasoning... \boxed{42}"], answer=["42"])
    assert out == [1.0]


def test_gsm8k_reward_correct_gsm8k_marker() -> None:
    out = gsm8k_reward(["q"], ["chain of thought\n#### 42"], answer=["42"])
    assert out == [1.0]


def test_gsm8k_reward_wrong() -> None:
    out = gsm8k_reward(["q"], [r"\boxed{41}"], answer=["42"])
    assert out == [0.0]


def test_gsm8k_reward_numeric_equality_tolerant() -> None:
    # 42.0 / $42 / 42 all match a gold of "42".
    out = gsm8k_reward(
        ["q", "q", "q"],
        [r"\boxed{42.0}", r"\boxed{$42}", "the answer is 42"],
        answer=["42", "42", "42"],
    )
    assert out == [1.0, 1.0, 1.0]


def test_gsm8k_reward_gold_with_marker() -> None:
    # The gold itself may be a raw GSM8K answer string ("... #### 42").
    out = gsm8k_reward(["q"], [r"\boxed{42}"], answer=["solution text\n#### 42"])
    assert out == [1.0]


def test_gsm8k_reward_missing_answer_all_zeros() -> None:
    # Tolerant when the answer column is absent.
    out = gsm8k_reward(["q", "q"], [r"\boxed{1}", r"\boxed{2}"])
    assert out == [0.0, 0.0]


def test_gsm8k_reward_batch_mixed() -> None:
    out = gsm8k_reward(
        ["q"] * 3,
        [r"\boxed{7}", r"\boxed{8}", "no answer here"],
        answer=["7", "9", "5"],
    )
    assert out == [1.0, 0.0, 0.0]


# ---------------------------------------------------------------------------
# format_reward
# ---------------------------------------------------------------------------
def test_format_reward_detects_boxed() -> None:
    out = format_reward(["q", "q"], [r"final: \boxed{42}", "no box at all"])
    assert out == [1.0, 0.0]


# ---------------------------------------------------------------------------
# make_infra_synth_reward — injected stub Verifier (no Sentinel / docker)
# ---------------------------------------------------------------------------
class _StubVerifier:
    """A canned Verifier (satisfies the protocol) for reward-path testing.

    Records every (artifact, spec) it was asked to verify and always returns the
    same canned VerifyResult.
    """

    name = "stub"

    def __init__(self, result: VerifyResult) -> None:
        self._result = result
        self.calls: list[tuple[str, VerifySpec]] = []

    async def verify(self, artifact: str, spec: VerifySpec) -> VerifyResult:
        self.calls.append((artifact, spec))
        return self._result


# A minimal completion carrying a fenced Dockerfile the parser can extract.
_DOCKERFILE_COMPLETION = (
    "Here is the Dockerfile:\n"
    "```dockerfile\n"
    "FROM python:3.11-slim\n"
    "WORKDIR /app\n"
    "COPY . .\n"
    "EXPOSE 8000\n"
    'CMD ["uvicorn", "app.main:app", "--port", "8000"]\n'
    "```\n"
)


def _one_info() -> dict:
    """A valid infra_synth `info` dict (has the spec_id build_verify_spec needs)."""
    from infra_synth.tasks import generate_tasks

    return generate_tasks(n=1, seed=0, split="train")[0]["info"]


def test_infra_reward_shaped_score_build_only() -> None:
    """build_ok=True, smoke_ok=False with default weights -> 0.3*1 + 0.7*0 = 0.3."""
    pytest.importorskip("infra_synth")
    stub = _StubVerifier(VerifyResult(build_ok=True, smoke_ok=False))
    reward_fn = make_infra_synth_reward(verifier=stub)

    info = _one_info()
    out = reward_fn(["spec"], [_DOCKERFILE_COMPLETION], info=[info])

    assert out == pytest.approx([0.3])
    assert stub.calls, "the injected verifier should have been invoked"
    # The factory should have parsed the Dockerfile out of the fenced block.
    assert "FROM python:3.11-slim" in stub.calls[0][0]


def test_infra_reward_full_pass_is_one() -> None:
    pytest.importorskip("infra_synth")
    stub = _StubVerifier(VerifyResult(build_ok=True, smoke_ok=True))
    reward_fn = make_infra_synth_reward(verifier=stub)
    out = reward_fn(["spec"], [_DOCKERFILE_COMPLETION], info=[_one_info()])
    assert out == pytest.approx([1.0])


def test_infra_reward_custom_weights() -> None:
    pytest.importorskip("infra_synth")
    stub = _StubVerifier(VerifyResult(build_ok=True, smoke_ok=False))
    reward_fn = make_infra_synth_reward(verifier=stub, build_weight=0.5, smoke_weight=0.5)
    out = reward_fn(["spec"], [_DOCKERFILE_COMPLETION], info=[_one_info()])
    assert out == pytest.approx([0.5])


def test_infra_reward_batch_concurrent() -> None:
    """A batch of N completions is verified (concurrently) -> N shaped scores."""
    pytest.importorskip("infra_synth")
    from infra_synth.tasks import generate_tasks

    n = 4
    stub = _StubVerifier(VerifyResult(build_ok=True, smoke_ok=False))
    reward_fn = make_infra_synth_reward(verifier=stub)

    tasks = generate_tasks(n=n, seed=0, split="train")
    completions = [_DOCKERFILE_COMPLETION] * n
    infos = [t["info"] for t in tasks]

    out = reward_fn(["spec"] * n, completions, info=infos)

    assert out == pytest.approx([0.3] * n)
    assert len(stub.calls) == n, "every completion in the batch should be verified"


def test_infra_reward_name_reflects_backend() -> None:
    # TRL logs reward metrics by __name__; it should encode the backend.
    fn = make_infra_synth_reward(verifier_backend="sentinel", verifier=_StubVerifier(VerifyResult()))
    assert fn.__name__ == "infra_synth_reward_sentinel"
