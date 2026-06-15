"""Unit tests for ``training.data`` — the torch-free dataset / answer layer.

The answer-extraction and prompt helpers are pure stdlib, so they test with no
heavy deps. The dataset *builders* (:func:`build_gsm8k` / :func:`build_infra_synth`)
need ``datasets`` (and, for infra, the env package), so they are guarded with
``pytest.importorskip`` and skip cleanly when those are unavailable.
"""
from __future__ import annotations

import pytest

from training.data import (
    GSM8K_SYSTEM_PROMPT,
    build_gsm8k,
    build_infra_synth,
    extract_boxed_or_final_number,
)

# ``build_infra_synth`` imports ``infra_synth.tasks`` from the installed env
# package (``pip install -e ./environments/infra_synth``); the builder tests are
# guarded with ``pytest.importorskip`` and skip cleanly when it is absent.


# ---------------------------------------------------------------------------
# extract_boxed_or_final_number — the core, pure extractor
# ---------------------------------------------------------------------------
def test_extract_boxed() -> None:
    assert extract_boxed_or_final_number(r"The work shows \boxed{42}.") == "42"


def test_extract_gsm8k_gold_marker() -> None:
    # GSM8K gold answers end with a "#### <number>" marker.
    assert extract_boxed_or_final_number("Long rationale...\n#### 42") == "42"


def test_extract_answer_is_phrase() -> None:
    assert extract_boxed_or_final_number("so the answer is 42") == "42"


def test_extract_no_number_returns_none() -> None:
    assert extract_boxed_or_final_number("there is no numeric answer here") is None
    assert extract_boxed_or_final_number("") is None


def test_extract_strips_currency_and_thousands_separators() -> None:
    # A bare numeric box is normalised ($ and , stripped).
    assert extract_boxed_or_final_number(r"\boxed{1,000}") == "1000"
    assert extract_boxed_or_final_number(r"\boxed{$1,000}") == "1000"
    # ...and so is the last-number fallback.
    assert extract_boxed_or_final_number("It costs $1,234 total") == "1234"


def test_extract_negative_and_decimal() -> None:
    assert extract_boxed_or_final_number(r"\boxed{-3.5}") == "-3.5"


def test_extract_boxed_wins_over_trailing_number() -> None:
    # The boxed value takes precedence over a stray later number.
    assert extract_boxed_or_final_number(r"\boxed{7} (computed in 2 steps)") == "7"


def test_extract_symbolic_box_returned_raw() -> None:
    # A non-numeric box is returned verbatim (for a symbolic string compare).
    assert extract_boxed_or_final_number(r"\boxed{\frac{1}{2}}") == r"\frac{1}{2}"


def test_extract_last_boxed_when_multiple() -> None:
    assert extract_boxed_or_final_number(r"\boxed{1} then \boxed{2}") == "2"


# ---------------------------------------------------------------------------
# normalize_number / numeric-equality helper (lives in training.rewards)
# ---------------------------------------------------------------------------
def test_normalize_number_canonicalises() -> None:
    from training.rewards import normalize_number

    assert normalize_number("42") == "42"
    assert normalize_number("42.0") == "42"  # trailing .0 dropped
    assert normalize_number("$1,000") == "1000"
    assert normalize_number("3.50") == "3.5"  # trailing zero trimmed
    assert normalize_number("-0") == "0"  # -0 canonicalised
    assert normalize_number("12%") == "12"


def test_normalize_number_non_numeric_is_none() -> None:
    from training.rewards import normalize_number

    assert normalize_number(None) is None
    assert normalize_number("") is None
    assert normalize_number("frac") is None


def test_numeric_equality_via_normalisation() -> None:
    # Two textual forms of the same value normalise equal.
    from training.rewards import normalize_number

    assert normalize_number("1,000.0") == normalize_number("$1000")
    assert normalize_number("7") != normalize_number("8")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
def test_system_prompt_requests_boxed() -> None:
    assert "boxed" in GSM8K_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Dataset builders — guarded by importorskip so they skip without deps
# ---------------------------------------------------------------------------
def test_build_gsm8k_shape() -> None:
    pytest.importorskip("datasets")
    try:
        ds = build_gsm8k(split="train", n=4, seed=0)
    except Exception as exc:  # offline / no network for the HF download
        pytest.skip(f"GSM8K download unavailable: {exc}")
    assert len(ds) == 4
    assert set(ds.column_names) == {"prompt", "answer", "question"}
    row = ds[0]
    # Chat-style prompt by default: a list of role/content messages.
    assert isinstance(row["prompt"], list)
    assert row["prompt"][0]["role"] == "system"
    assert isinstance(row["answer"], str)


def test_build_infra_synth_shape() -> None:
    pytest.importorskip("datasets")
    # The installed env package exposes the ``infra_synth.tasks`` builder.
    pytest.importorskip("infra_synth")
    ds = build_infra_synth(split="train", n=5, seed=0)
    assert len(ds) == 5
    assert set(ds.column_names) == {"prompt", "question", "answer", "info", "task"}
    row = ds[0]
    assert row["task"] == "infra_synth"
    # `info` must carry the spec_id used to build a VerifySpec downstream.
    assert "spec_id" in row["info"]
    assert isinstance(row["prompt"], list)
