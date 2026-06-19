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


# --- compose kind ----------------------------------------------------------
def test_dockerfile_kind_default_unchanged() -> None:
    # The default kwarg must produce byte-for-byte identical tasks to the
    # historical no-kwarg call (questions, ids, and full info dicts).
    no_kwarg = infra_tasks.generate_tasks(n=10, seed=5, split="train")
    explicit = infra_tasks.generate_tasks(n=10, seed=5, split="train", kind="dockerfile")
    assert no_kwarg == explicit


def test_compose_task_shape_and_locked_smoke_keys() -> None:
    tasks = infra_tasks.generate_tasks(n=16, seed=0, split="test", kind="compose")
    assert tasks
    for t in tasks:
        info = t["info"]
        assert info["kind"] == ArtifactKind.COMPOSE.value == "compose"
        assert info["spec_id"].startswith("compose-")
        smoke = info["smoke"]
        # Locked compose smoke contract.
        assert set(smoke) >= {
            "port",
            "health_path",
            "expect_status",
            "must_contain",
            "dependency_service",
        }
        port = smoke["port"]
        assert isinstance(port, int)
        assert smoke["health_path"].startswith("/")
        assert smoke["expect_status"] == 200
        assert smoke["must_contain"] == [
            "services:",
            "ports:",
            f"{port}:{port}",
            "healthcheck:",
        ]
        # dependency_service is None iff dependency == "none".
        dep = info["dependency"]
        if dep == "none":
            assert smoke["dependency_service"] is None
        else:
            assert smoke["dependency_service"] == dep
            assert smoke["dependency_service"] in ("postgres", "redis")


def test_compose_dependency_service_none_iff_none() -> None:
    tasks = infra_tasks.generate_tasks(seed=0, split="train", kind="compose")
    for t in tasks:
        info = t["info"]
        is_none = info["smoke"]["dependency_service"] is None
        assert is_none == (info["dependency"] == "none")


def test_compose_deterministic_for_seed() -> None:
    a = infra_tasks.generate_tasks(n=8, seed=7, split="train", kind="compose")
    b = infra_tasks.generate_tasks(n=8, seed=7, split="train", kind="compose")
    assert [t["question"] for t in a] == [t["question"] for t in b]
    assert [t["info"]["spec_id"] for t in a] == [t["info"]["spec_id"] for t in b]


def test_compose_ids_disjoint_from_dockerfile_ids() -> None:
    df = infra_tasks.generate_tasks(n=12, seed=0, split="train")
    comp = infra_tasks.generate_tasks(n=12, seed=0, split="train", kind="compose")
    df_ids = {t["info"]["spec_id"] for t in df}
    comp_ids = {t["info"]["spec_id"] for t in comp}
    assert df_ids.isdisjoint(comp_ids)


def test_compose_build_verify_spec_kind() -> None:
    tasks = infra_tasks.generate_tasks(n=6, seed=2, split="train", kind="compose")
    assert tasks
    for t in tasks:
        spec = infra_tasks.build_verify_spec(t["info"])
        assert isinstance(spec, VerifySpec)
        assert spec.kind is ArtifactKind.COMPOSE
        # No Dockerfile scaffold is attached on the compose path.
        assert "context_files" not in spec.smoke
        assert spec.smoke["dependency_service"] == t["info"]["smoke"]["dependency_service"]


def test_compose_system_prompt_constrains_output() -> None:
    sp = infra_tasks.COMPOSE_SYSTEM_PROMPT
    assert "```yaml" in sp
    assert "ONLY" in sp
    assert "services:" in sp
    assert "<think>" in sp  # explicitly forbids think blocks


def test_compose_unknown_kind_raises() -> None:
    import pytest

    with pytest.raises(ValueError):
        infra_tasks.generate_tasks(kind="terraform")


# --- ci-yaml kind ----------------------------------------------------------
def test_ci_yaml_task_shape_and_locked_smoke_keys() -> None:
    tasks = infra_tasks.generate_tasks(n=12, seed=0, split="test", kind="ci-yaml")
    assert tasks
    for t in tasks:
        info = t["info"]
        assert info["kind"] == ArtifactKind.CI_YAML.value == "ci-yaml"
        assert info["spec_id"].startswith("ci-")
        smoke = info["smoke"]
        # Locked ci-yaml smoke contract: must_contain + required_steps only.
        assert set(smoke) == {"must_contain", "required_steps"}
        assert smoke["must_contain"] == [
            "on:",
            "jobs:",
            "runs-on:",
            "steps:",
            "actions/checkout",
        ]
        assert smoke["required_steps"] == ["checkout", "setup", "install", "test"]
        # A CI workflow has no server to probe -> no port/health keys.
        assert "port" not in smoke
        assert "health_path" not in smoke


def test_ci_yaml_deterministic_for_seed() -> None:
    a = infra_tasks.generate_tasks(n=8, seed=7, split="train", kind="ci-yaml")
    b = infra_tasks.generate_tasks(n=8, seed=7, split="train", kind="ci-yaml")
    assert [t["question"] for t in a] == [t["question"] for t in b]
    assert [t["info"]["spec_id"] for t in a] == [t["info"]["spec_id"] for t in b]


def test_ci_yaml_ids_disjoint_from_other_kinds() -> None:
    df = infra_tasks.generate_tasks(n=12, seed=0, split="train")
    comp = infra_tasks.generate_tasks(n=12, seed=0, split="train", kind="compose")
    ci = infra_tasks.generate_tasks(n=12, seed=0, split="train", kind="ci-yaml")
    df_ids = {t["info"]["spec_id"] for t in df}
    comp_ids = {t["info"]["spec_id"] for t in comp}
    ci_ids = {t["info"]["spec_id"] for t in ci}
    assert ci_ids.isdisjoint(df_ids)
    assert ci_ids.isdisjoint(comp_ids)


def test_ci_yaml_build_verify_spec_kind() -> None:
    tasks = infra_tasks.generate_tasks(n=6, seed=2, split="train", kind="ci-yaml")
    assert tasks
    for t in tasks:
        spec = infra_tasks.build_verify_spec(t["info"])
        assert isinstance(spec, VerifySpec)
        assert spec.kind is ArtifactKind.CI_YAML
        # No Dockerfile scaffold is attached on the ci-yaml path.
        assert "context_files" not in spec.smoke
        assert spec.smoke["required_steps"] == ["checkout", "setup", "install", "test"]


def test_ci_yaml_system_prompt_constrains_output() -> None:
    sp = infra_tasks.CI_YAML_SYSTEM_PROMPT
    assert "```yaml" in sp
    assert "ONLY" in sp
    assert "jobs:" in sp
    assert "<think>" in sp  # explicitly forbids think blocks


def test_dockerfile_and_compose_unchanged_by_ci_yaml_addition() -> None:
    # Adding the ci-yaml kind must not perturb the other kinds' output.
    df = infra_tasks.generate_tasks(n=10, seed=5, split="train")
    df_explicit = infra_tasks.generate_tasks(n=10, seed=5, split="train", kind="dockerfile")
    comp = infra_tasks.generate_tasks(n=10, seed=5, split="train", kind="compose")
    comp2 = infra_tasks.generate_tasks(n=10, seed=5, split="train", kind="compose")
    assert df == df_explicit
    assert comp == comp2
    # ci-yaml shares the same split pool/order, only the rendered artifact differs.
    # (ci-yaml smoke has no port/health, so compare the shared grid fields + order
    # via the spec_id suffix the base_id encodes.)
    ci = infra_tasks.generate_tasks(n=10, seed=5, split="train", kind="ci-yaml")
    df_lfd = [(t["info"]["language"], t["info"]["framework"], t["info"]["dependency"]) for t in df]
    ci_lfd = [(t["info"]["language"], t["info"]["framework"], t["info"]["dependency"]) for t in ci]
    assert df_lfd == ci_lfd
    # The ci-<base_id> suffix must equal the Dockerfile spec_id (same pool order).
    assert [t["info"]["spec_id"] for t in df] == [
        t["info"]["spec_id"].removeprefix("ci-") for t in ci
    ]
