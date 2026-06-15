"""Tests for ``infra_synth.tasks`` (vf-free; uses only stdlib + verifier.types)."""
from __future__ import annotations

from infra_synth import tasks as infra_tasks
from verifier.types import ArtifactKind, ResourceLimits, VerifySpec


def _combo_key(info: dict) -> tuple:
    s = info["smoke"]
    return (
        info["language"],
        info["framework"],
        info["dependency"],
        s["port"],
        s["health_path"],
    )


def test_deterministic_for_seed() -> None:
    a = infra_tasks.generate_tasks(n=8, seed=7, split="train")
    b = infra_tasks.generate_tasks(n=8, seed=7, split="train")
    assert [t["question"] for t in a] == [t["question"] for t in b]
    assert [t["info"]["spec_id"] for t in a] == [t["info"]["spec_id"] for t in b]


def test_different_seed_differs() -> None:
    a = infra_tasks.generate_tasks(n=12, seed=1, split="train")
    b = infra_tasks.generate_tasks(n=12, seed=2, split="train")
    # Ordering (and thus spec_ids) should differ for different seeds.
    assert [t["info"]["spec_id"] for t in a] != [t["info"]["spec_id"] for t in b]


def test_train_test_param_combos_disjoint() -> None:
    train = infra_tasks.generate_tasks(seed=0, split="train")
    test = infra_tasks.generate_tasks(seed=0, split="test")
    train_combos = {_combo_key(t["info"]) for t in train}
    test_combos = {_combo_key(t["info"]) for t in test}
    assert train_combos, "train split must be non-empty"
    assert test_combos, "test split must be non-empty"
    assert train_combos.isdisjoint(test_combos), "splits must be contamination-resistant"


def test_split_partition_is_stable_and_total() -> None:
    # Train + test should partition the whole grid with no overlap or gaps.
    train = infra_tasks.generate_tasks(seed=0, split="train")
    test = infra_tasks.generate_tasks(seed=0, split="test")
    train_combos = {_combo_key(t["info"]) for t in train}
    test_combos = {_combo_key(t["info"]) for t in test}
    total = len(infra_tasks._all_combos())
    assert len(train_combos) + len(test_combos) == total
    assert train_combos.isdisjoint(test_combos)


def test_task_shape_and_required_fields() -> None:
    tasks = infra_tasks.generate_tasks(n=5, seed=3, split="train")
    for t in tasks:
        assert set(t) >= {"question", "answer", "info", "task"}
        assert isinstance(t["question"], str) and t["question"]
        assert isinstance(t["answer"], str) and t["answer"]
        assert t["task"] == "infra_synth"
        info = t["info"]
        assert info["kind"] == ArtifactKind.DOCKERFILE.value == "dockerfile"
        assert info["spec_id"]
        smoke = info["smoke"]
        assert isinstance(smoke["port"], int)
        assert smoke["health_path"].startswith("/")
        assert smoke["expect_status"] == 200
        assert isinstance(smoke["must_contain"], list)
        assert "FROM" in smoke["must_contain"]
        assert "CMD" in smoke["must_contain"]
        assert f"EXPOSE {smoke['port']}" in smoke["must_contain"]
        assert smoke["base_image_prefix"]


def test_build_verify_spec_from_every_task() -> None:
    tasks = infra_tasks.generate_tasks(n=20, seed=4, split="train")
    assert tasks
    for t in tasks:
        spec = infra_tasks.build_verify_spec(t["info"])
        assert isinstance(spec, VerifySpec)
        assert spec.spec_id == t["info"]["spec_id"]
        assert spec.kind is ArtifactKind.DOCKERFILE
        assert spec.smoke["port"] == t["info"]["smoke"]["port"]
        assert isinstance(spec.limits, ResourceLimits)
        assert spec.limits.wall_s == 30  # default carried through


def test_build_verify_spec_honors_custom_limits() -> None:
    tasks = infra_tasks.generate_tasks(n=1, seed=0, split="train")
    info = dict(tasks[0]["info"])
    info["limits"] = {"wall_s": 99, "mem_mb": 256, "cpus": 2.0}
    spec = infra_tasks.build_verify_spec(info)
    assert spec.limits.wall_s == 99
    assert spec.limits.mem_mb == 256
    assert spec.limits.cpus == 2.0
    assert spec.limits.pids == 64  # untouched default


def test_n_caps_and_overflow_is_safe() -> None:
    capped = infra_tasks.generate_tasks(n=3, seed=0, split="train")
    assert len(capped) == 3
    full = infra_tasks.generate_tasks(seed=0, split="train")
    huge = infra_tasks.generate_tasks(n=10_000, seed=0, split="train")
    assert len(huge) == len(full)  # no duplication beyond the pool


def test_system_prompt_constrains_output() -> None:
    sp = infra_tasks.SYSTEM_PROMPT
    assert "```dockerfile" in sp
    assert "ONLY" in sp
    assert "<think>" in sp  # explicitly forbids think blocks


def test_unknown_split_raises() -> None:
    import pytest

    with pytest.raises(ValueError):
        infra_tasks.generate_tasks(split="validation")
