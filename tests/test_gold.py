"""Tests for ``infra_synth.gold.gold_dockerfile`` (vf-free; stdlib + verifier.types).

The gold reference Dockerfile must (a) have the right structural properties and
(b) pass its own spec's static check. The static-check assertion uses
``verifier.smoke.checks.check_dockerfile`` if available, else is skipped.
"""
from __future__ import annotations

import pytest

from infra_synth import gold as infra_gold
from infra_synth import tasks as infra_tasks

# A representative sample across the grid (covers both frameworks and all deps).
_SAMPLE = infra_tasks.generate_tasks(n=24, seed=0, split="train")


@pytest.mark.parametrize("task", _SAMPLE, ids=[t["info"]["spec_id"] for t in _SAMPLE])
def test_gold_structural_properties(task: dict) -> None:
    info = task["info"]
    df = infra_gold.gold_dockerfile(info)
    port = info["smoke"]["port"]
    prefix = info["smoke"]["base_image_prefix"]

    # Pinned FROM (has a tag, not floating `latest`, not `scratch`).
    from_lines = [ln for ln in df.splitlines() if ln.strip().startswith("FROM ")]
    assert from_lines, "must declare a base image"
    base = from_lines[0].split(None, 1)[1].strip()
    assert ":" in base, "base image must be pinned with a tag"
    assert not base.endswith(":latest")
    assert base.lower() != "scratch"
    assert base.startswith(prefix)

    assert "WORKDIR" in df
    assert "COPY" in df
    assert f"EXPOSE {port}" in df

    cmd_lines = [ln for ln in df.splitlines() if ln.strip().startswith("CMD ")]
    assert cmd_lines, "must launch the server via CMD"
    # The launched server must reference the requested port.
    assert str(port) in cmd_lines[0]

    # Every must_contain substring is present.
    for needle in info["smoke"]["must_contain"]:
        assert needle in df, f"gold missing required substring {needle!r}"


def test_gold_contains_dependency_install_for_postgres() -> None:
    # Find a postgres task and confirm its OS build deps are installed.
    pg = next(t for t in _SAMPLE if t["info"]["dependency"] == "postgres")
    df = infra_gold.gold_dockerfile(pg["info"])
    assert "apt-get install" in df
    assert "libpq-dev" in df


# --- static-check assertion (skipped if the verifier check is not available) ---
checks = pytest.importorskip(
    "verifier.smoke.checks",
    reason="verifier.smoke.checks not available yet (parallel agent WIP)",
)


@pytest.mark.parametrize("task", _SAMPLE, ids=[t["info"]["spec_id"] for t in _SAMPLE])
def test_gold_passes_static_check(task: dict) -> None:
    info = task["info"]
    df = infra_gold.gold_dockerfile(info)
    spec = infra_tasks.build_verify_spec(info)
    result = checks.check_dockerfile(df, spec)
    assert result["build_ok"] is True, result.get("reasons")
    assert result["smoke_ok"] is True, result.get("reasons")
