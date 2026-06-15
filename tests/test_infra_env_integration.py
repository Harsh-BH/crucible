"""Integration test for ``infra_synth.load_environment`` with the ``verifiers`` lib.

Skipped entirely if ``verifiers`` / ``datasets`` are not installable on this box.
When present, this builds the real ``vf.SingleTurnEnv`` with the zero-dependency
``static`` backend and (best-effort) scores one fake completion through the rubric.
"""
from __future__ import annotations

import asyncio

import pytest

vf = pytest.importorskip("verifiers", reason="verifiers not installed on this box")
pytest.importorskip("datasets", reason="datasets not installed on this box")

from infra_synth import load_environment  # noqa: E402


def _get_dataset(env):
    if hasattr(env, "get_dataset"):
        try:
            return env.get_dataset()
        except Exception:
            pass
    return getattr(env, "dataset", None)


def test_load_environment_returns_environment() -> None:
    env = load_environment(verifier_backend="static", num_tasks=4, seed=0)
    assert isinstance(env, vf.Environment)
    assert env.system_prompt and "```dockerfile" in env.system_prompt


def test_dataset_non_empty_and_shaped() -> None:
    env = load_environment(verifier_backend="static", num_tasks=4, seed=0)
    ds = _get_dataset(env)
    assert ds is not None and len(ds) == 4
    cols = set(ds.column_names)
    assert {"question", "answer", "info", "task"} <= cols
    assert ds[0]["task"] == "infra_synth"


def test_eval_dataset_disjoint_from_train() -> None:
    env = load_environment(verifier_backend="static", num_tasks=8, seed=0)
    train = _get_dataset(env)
    eval_ds = env.get_eval_dataset() if hasattr(env, "get_eval_dataset") else None
    if eval_ds is None:
        pytest.skip("eval dataset not exposed by this verifiers version")
    train_ids = {r["info"]["spec_id"].split("-", 1)[1] for r in train}
    # spec_ids embed the split; just confirm the eval split is the 'test' pool.
    assert all(r["info"]["spec_id"].startswith("test-") for r in eval_ds)
    assert all(not sid.startswith("test-") for sid in
               (r["info"]["spec_id"] for r in train))


def test_score_one_completion_through_rubric_static_backend() -> None:
    """Score a correct fake Dockerfile via the rubric with the static backend."""
    env = load_environment(verifier_backend="static", num_tasks=4, seed=0)
    ds = _get_dataset(env)
    row = ds[0]
    port = row["info"]["smoke"]["port"]
    df = (
        "```dockerfile\n"
        "FROM python:3.11-slim\n"
        "WORKDIR /app\n"
        "COPY requirements.txt ./\n"
        "RUN pip install --no-cache-dir -r requirements.txt\n"
        "COPY ./app ./app\n"
        f"EXPOSE {port}\n"
        f'CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "{port}"]\n'
        "```"
    )
    completion = [{"role": "assistant", "content": df}]
    state = {
        "prompt": [{"role": "user", "content": row["question"]}],
        "completion": completion,
        "answer": row["answer"],
        "info": row["info"],
        "task": row["task"],
    }
    try:
        asyncio.run(env.rubric.score_rollout(state))
    except Exception as e:  # pragma: no cover - version-specific State plumbing
        pytest.skip(f"score_rollout needs more state plumbing on this version: {e}")

    metrics = state.get("metrics", {})
    # The primary reward + metric snapshot should reflect a correct artifact.
    assert metrics.get("build_smoke_reward") == pytest.approx(1.0)
    assert metrics.get("format_reward") == pytest.approx(1.0)
    assert metrics.get("build_ok_metric") == pytest.approx(1.0)
    assert metrics.get("smoke_ok_metric") == pytest.approx(1.0)
    assert metrics.get("hack_any_metric") == pytest.approx(0.0)
    # Weighted reward = 1.0*1.0 (build_smoke) + 0.1*1.0 (format) = 1.1
    assert state.get("reward") == pytest.approx(1.1)
    # The env stashes the raw result + metrics for post-hoc analysis.
    assert "infra_synth" in state
    assert state["infra_synth"]["metrics"]["build_ok"] == 1.0


def test_empty_completion_scores_zero_build_smoke() -> None:
    env = load_environment(verifier_backend="static", num_tasks=2, seed=0)
    ds = _get_dataset(env)
    row = ds[0]
    completion = [{"role": "assistant", "content": "I cannot help with that."}]
    state = {
        "prompt": [{"role": "user", "content": row["question"]}],
        "completion": completion,
        "answer": row["answer"],
        "info": row["info"],
        "task": row["task"],
    }
    try:
        asyncio.run(env.rubric.score_rollout(state))
    except Exception as e:  # pragma: no cover
        pytest.skip(f"score_rollout needs more state plumbing on this version: {e}")
    metrics = state.get("metrics", {})
    assert metrics.get("build_smoke_reward") == pytest.approx(0.0)
    assert metrics.get("format_reward") == pytest.approx(0.0)  # no fenced block
