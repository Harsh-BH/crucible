"""Tests for ``infra_synth.impossible`` (C3 impossible-task generator + corpus).

vf-free; stdlib + ``verifier.types`` + ``verifier.smoke.checks``. The corpus
artifacts are NEVER executed here — we only inspect their metadata. The
dangerous ``fork_bomb`` entry is asserted absent by default and is never run.
"""
from __future__ import annotations

from infra_synth import build_verify_spec, gold_dockerfile
from infra_synth import impossible as imp

from verifier.smoke.checks import check_dockerfile
from verifier.types import ArtifactKind

_CATEGORY_CONSTS = {
    imp.WEAK_TESTS,
    imp.FAKE_SUCCESS,
    imp.RESOURCE_MANIPULATION,
    imp.HARNESS_SIDE_EFFECTS,
    imp.VERIFIER_GAMING,
    imp.ANSWER_LEAKAGE,
}


# --- Impossibility crux: no faithful artifact (gold included) can pass --------
def test_gold_fails_every_impossible_task() -> None:
    """The crux: even the gold reference fails the mutated spec's static check."""
    tasks = imp.impossible_tasks(n=24, seed=0)
    assert tasks, "expected a non-empty impossible-task set"
    for task in tasks:
        info = task["info"]
        out = check_dockerfile(gold_dockerfile(info), build_verify_spec(info))
        assert out["build_ok"] is False or out["smoke_ok"] is False, (
            f"mutation {info['impossible']['mutation']} did not make the spec "
            f"unsatisfiable; reasons={out.get('reasons')}"
        )


def test_every_task_tagged_with_valid_category_and_mutation() -> None:
    tasks = imp.impossible_tasks(n=24, seed=0)
    for task in tasks:
        tag = task["info"]["impossible"]
        assert tag["category"] in _CATEGORY_CONSTS
        assert tag["mutation"] in imp.MUTATIONS


def test_deterministic_for_same_args() -> None:
    a = imp.impossible_tasks(n=12, seed=3, split="test")
    b = imp.impossible_tasks(n=12, seed=3, split="test")
    assert [t["info"]["spec_id"] for t in a] == [t["info"]["spec_id"] for t in b]
    assert [t["info"]["impossible"] for t in a] == [t["info"]["impossible"] for t in b]
    assert [t["info"]["smoke"]["must_contain"] for t in a] == [
        t["info"]["smoke"]["must_contain"] for t in b
    ]


def test_does_not_mutate_generate_tasks_output() -> None:
    """No shared-reference bug: the source tasks are untouched by mutation."""
    from infra_synth import tasks as infra_tasks

    base = infra_tasks.generate_tasks(n=8, seed=0, split="test")
    base_must = [list(t["info"]["smoke"]["must_contain"]) for t in base]
    base_has_impossible = ["impossible" in t["info"] for t in base]

    _ = imp.impossible_tasks(n=8, seed=0, split="test")

    after_must = [list(t["info"]["smoke"]["must_contain"]) for t in base]
    after_has_impossible = ["impossible" in t["info"] for t in base]
    assert after_must == base_must, "generate_tasks output was mutated in place"
    assert after_has_impossible == base_has_impossible == [False] * len(base)


def test_mutations_subset_selection() -> None:
    only = imp.impossible_tasks(n=10, seed=1, mutations=["impossible_token"])
    assert {t["info"]["impossible"]["mutation"] for t in only} == {"impossible_token"}


def test_unknown_mutation_raises() -> None:
    import pytest

    with pytest.raises(ValueError):
        imp.impossible_tasks(n=2, mutations=["nope"])


def test_all_mutations_individually_impossible() -> None:
    """Each mutation in isolation makes the gold reference fail."""
    for name in imp.MUTATIONS:
        tasks = imp.impossible_tasks(n=6, seed=0, mutations=[name])
        assert tasks
        for task in tasks:
            info = task["info"]
            out = check_dockerfile(gold_dockerfile(info), build_verify_spec(info))
            assert out["build_ok"] is False or out["smoke_ok"] is False, (
                f"mutation {name} left the spec satisfiable; reasons={out.get('reasons')}"
            )


# --- Adversarial corpus -------------------------------------------------------
def test_corpus_excludes_dangerous_by_default() -> None:
    corpus = imp.adversarial_corpus()
    names = {a.name for a in corpus}
    assert "fork_bomb" not in names
    assert all(a.dangerous is False for a in corpus)


def test_corpus_includes_dangerous_when_requested() -> None:
    corpus = imp.adversarial_corpus(include_dangerous=True)
    by_name = {a.name: a for a in corpus}
    assert "fork_bomb" in by_name
    assert by_name["fork_bomb"].dangerous is True
    # The safe entries are still present alongside the dangerous one.
    assert {"fake_success", "mem_hog", "sleep_stall", "conftest_edit", "net_exfil"} <= set(
        by_name
    )


def test_corpus_entries_well_formed() -> None:
    for adv in imp.adversarial_corpus(include_dangerous=True):
        assert adv.category in _CATEGORY_CONSTS
        assert adv.spec.kind is ArtifactKind.PYTHON
        assert adv.spec.limits.wall_s == 2
        assert adv.spec.limits.mem_mb == 64
        assert isinstance(adv.artifact, str) and adv.artifact
        assert isinstance(adv.expect_flag, str)


def test_corpus_expect_flags_are_valid_hackflags_fields() -> None:
    import dataclasses

    from verifier.types import HackFlags

    valid = {f.name for f in dataclasses.fields(HackFlags)}
    for adv in imp.adversarial_corpus(include_dangerous=True):
        if adv.expect_flag:
            assert adv.expect_flag in valid, adv.name


# --- Decoupling pin: our constants equal analysis.reward_hacking's ------------
def test_category_constants_match_analysis() -> None:
    """impossible.py duplicates the taxonomy ids as plain strings; pin equality.

    Tests live in the crucible dev package and MAY import analysis; the
    impossible.py module itself must not (and does not).
    """
    from analysis import reward_hacking as rh

    assert imp.WEAK_TESTS == rh.WEAK_TESTS
    assert imp.FAKE_SUCCESS == rh.FAKE_SUCCESS
    assert imp.RESOURCE_MANIPULATION == rh.RESOURCE_MANIPULATION
    assert imp.HARNESS_SIDE_EFFECTS == rh.HARNESS_SIDE_EFFECTS
    assert imp.VERIFIER_GAMING == rh.VERIFIER_GAMING
    assert imp.ANSWER_LEAKAGE == rh.ANSWER_LEAKAGE
    assert _CATEGORY_CONSTS == set(rh.CATEGORIES)
