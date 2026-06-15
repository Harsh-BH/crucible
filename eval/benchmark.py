"""Held-out evaluation harness for the ``infra_synth`` environment.

``evaluate`` runs a (generation, verification) loop over the **contamination-
resistant test split** (``infra_synth.tasks.generate_tasks(split="test")``, whose
parameter combos are disjoint from train — see ``DESIGN.md`` and
``infra_synth.tasks._split_combos``) and reports:

* build-pass-rate and smoke-pass-rate (any-of-``n`` and mean-over-samples),
* ``pass@1`` and ``pass@k`` via the unbiased estimator in :mod:`eval.passk`,
* aggregate :class:`verifier.types.HackFlags` rates (the C3 raw signal).

The policy is injected as ``generate_fn(prompt, n) -> list[str]`` (n candidate
completions). Tests pass a stub returning gold/garbage Dockerfiles; real use
wraps an OpenAI-compatible / vLLM server via :func:`make_openai_generate_fn`.

Eval rigor (per ``DESIGN.md`` "eval protocol"): run with ``n >> k`` (e.g.
``n=200``) and average ``pass@k`` over problems; sweep several seeds and report
mean ± 95% CI across seeds with :func:`evaluate_multi_seed`.

The module imports only stdlib + ``verifier`` + (optionally, lazily) ``httpx``.
``infra_synth.tasks`` is imported vf-free (we never trigger the package
``__init__`` that pulls ``verifiers``), so this runs without the heavy training
stack. matplotlib is never imported here.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections.abc import Callable
from typing import Any

from eval.passk import pass_at_k_corpus, passk_curve
from verifier import get_verifier, shape_reward
from verifier.types import HackFlags, VerifyResult, VerifySpec

__all__ = [
    "evaluate",
    "evaluate_multi_seed",
    "make_openai_generate_fn",
    "GenerateFn",
]

#: A generation function: ``generate_fn(prompt, n) -> list[str]`` of ``n``
#: candidate completions (raw model text, possibly wrapping a fenced Dockerfile).
GenerateFn = Callable[[str, int], list[str]]

# Hack-flag fields aggregated into rates (mirrors verifier.types.HackFlags).
_HACK_FIELDS = (
    "resource_exhaustion",
    "oom_killed",
    "timed_out",
    "network_attempt",
    "seccomp_violation",
    "spec_gaming",
)


# ---------------------------------------------------------------------------
# vf-free access to infra_synth.tasks
# ---------------------------------------------------------------------------
def _load_tasks_module() -> Any:
    """Import the vf-free ``infra_synth.tasks`` (task generation).

    Importing ``infra_synth`` is ``verifiers``-free: the package ``__init__``
    only pulls in the vf-wiring layer, which imports ``verifiers`` lazily inside
    ``load_environment`` (never at module load). So this runs without the heavy
    training stack.
    """
    from infra_synth import tasks as tasks_mod

    return tasks_mod


def _load_parser_module() -> Any:
    """Import the vf-free ``infra_synth.parser`` (Dockerfile extraction)."""
    from infra_synth import parser as parser_mod

    return parser_mod


def _resolve_tasks(env_or_tasks: Any, *, seed: int) -> list[dict[str, Any]]:
    """Normalize the ``env_or_tasks`` argument into a list of task dicts.

    Accepts:
      * ``None`` -> load the held-out test split for ``seed``;
      * a list of task dicts (each with an ``info`` dict) -> used as-is;
      * an object exposing ``generate_tasks`` / ``get_dataset`` -> best-effort.
    """
    if env_or_tasks is None:
        tasks_mod = _load_tasks_module()
        return tasks_mod.generate_tasks(split="test", seed=seed)
    if isinstance(env_or_tasks, list):
        return env_or_tasks
    # Duck-typed: an environment-like object.
    if hasattr(env_or_tasks, "generate_tasks"):
        return list(env_or_tasks.generate_tasks(split="test", seed=seed))
    raise TypeError(
        "env_or_tasks must be None, a list of task dicts, or expose "
        f"generate_tasks(); got {type(env_or_tasks)!r}"
    )


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------
async def _verify_candidate(
    verifier: Any,
    artifact: str,
    spec: VerifySpec,
    *,
    hack_penalty: float,
) -> VerifyResult:
    result = await verifier.verify(artifact, spec)
    # Fill the (otherwise None) reward so logs/metrics carry it.
    result.reward = shape_reward(result, hack_penalty=hack_penalty)
    return result


async def _evaluate_async(
    env_or_tasks: Any,
    generate_fn: GenerateFn,
    *,
    verifier_backend: str,
    n: int,
    ks: tuple[int, ...],
    seed: int,
    sentinel_base_url: str | None,
    hack_penalty: float,
    out_path: str | None,
    max_tasks: int | None,
) -> dict[str, Any]:
    tasks = _resolve_tasks(env_or_tasks, seed=seed)
    if max_tasks is not None:
        tasks = tasks[: max(0, max_tasks)]
    tasks_mod = _load_tasks_module()
    parser_mod = _load_parser_module()

    verifier = get_verifier(verifier_backend, base_url=sentinel_base_url)

    per_problem: list[tuple[int, int]] = []  # (n_samples, c_correct) for pass@k
    build_correct = 0  # count of (problem) with >=1 build_ok
    smoke_correct = 0  # count of (problem) with >=1 smoke_ok (== "correct")
    sample_build_ok = 0  # over all samples
    sample_smoke_ok = 0
    total_samples = 0
    reward_sum = 0.0
    wall_sum = 0.0
    hack_counts = {f: 0 for f in _HACK_FIELDS}
    hack_any_count = 0
    per_task_records: list[dict[str, Any]] = []

    for task in tasks:
        info = task["info"]
        spec = tasks_mod.build_verify_spec(info)
        prompt = task.get("question", "")
        completions = list(generate_fn(prompt, n))
        # Be tolerant of a generate_fn that returns fewer/more than n.
        if not completions:
            completions = [""]
        c_correct = 0  # smoke_ok defines "correct" for pass@k
        task_build_hits = 0
        for raw in completions:
            artifact = parser_mod.extract_dockerfile(raw) or raw
            result = await _verify_candidate(
                verifier, artifact, spec, hack_penalty=hack_penalty
            )
            total_samples += 1
            reward_sum += float(result.reward or 0.0)
            wall_sum += float(result.wall_s)
            if result.build_ok:
                sample_build_ok += 1
                task_build_hits += 1
            if result.smoke_ok:
                sample_smoke_ok += 1
                c_correct += 1
            hf: HackFlags = result.hack_flags
            if hf.any():
                hack_any_count += 1
            for f in _HACK_FIELDS:
                if getattr(hf, f):
                    hack_counts[f] += 1

        n_samples = len(completions)
        per_problem.append((n_samples, c_correct))
        if task_build_hits > 0:
            build_correct += 1
        if c_correct > 0:
            smoke_correct += 1
        per_task_records.append(
            {
                "spec_id": info.get("spec_id"),
                "n": n_samples,
                "c_smoke_ok": c_correct,
                "build_hits": task_build_hits,
            }
        )

    n_tasks = len(tasks)
    curve = passk_curve(per_problem, list(ks))
    pass_at_1 = pass_at_k_corpus(per_problem, 1)
    denom_samples = max(1, total_samples)

    report: dict[str, Any] = {
        "config": {
            "verifier_backend": verifier_backend,
            "sentinel_base_url": sentinel_base_url,
            "n": n,
            "ks": list(ks),
            "seed": seed,
            "hack_penalty": hack_penalty,
            "split": "test",
        },
        "n_tasks": n_tasks,
        "n_samples_total": total_samples,
        # Problem-level "any of n" rates.
        "build_pass_rate": build_correct / max(1, n_tasks),
        "smoke_pass_rate": smoke_correct / max(1, n_tasks),
        # Sample-level mean rates.
        "sample_build_rate": sample_build_ok / denom_samples,
        "sample_smoke_rate": sample_smoke_ok / denom_samples,
        "mean_reward": reward_sum / denom_samples,
        "mean_wall_s": wall_sum / denom_samples,
        # Unbiased pass@k.
        "pass_at_1": pass_at_1,
        "pass_at_k": curve,
        # C3 raw hack-flag rates (per sample).
        "hack_any_rate": hack_any_count / denom_samples,
        "hack_flag_rates": {f: hack_counts[f] / denom_samples for f in _HACK_FIELDS},
        "hack_flag_counts": hack_counts,
        "per_task": per_task_records,
    }

    # Close the Sentinel client if the backend owns one.
    aclose = getattr(verifier, "aclose", None)
    if callable(aclose):
        await aclose()

    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, sort_keys=True)
    return report


def evaluate(
    env_or_tasks: Any,
    generate_fn: GenerateFn,
    *,
    verifier_backend: str = "static",
    n: int = 4,
    ks: tuple[int, ...] = (1,),
    seed: int = 0,
    sentinel_base_url: str | None = None,
    hack_penalty: float = 0.0,
    out_path: str | None = None,
    max_tasks: int | None = None,
) -> dict[str, Any]:
    """Evaluate ``generate_fn`` on the held-out ``infra_synth`` test split.

    Parameters
    ----------
    env_or_tasks:
        ``None`` to load ``generate_tasks(split="test", seed=seed)``; a list of
        task dicts; or an env-like object exposing ``generate_tasks``.
    generate_fn:
        ``generate_fn(prompt, n) -> list[str]`` returning ``n`` candidate
        completions. INJECTED so tests can pass a deterministic stub.
    verifier_backend:
        ``"static" | "local-py" | "local-docker" | "sentinel"`` (see
        :func:`verifier.get_verifier`).
    n:
        Samples requested per problem. For a rigorous ``pass@k`` use ``n >> k``
        (e.g. ``n=200``).
    ks:
        ``k`` values for the ``pass@k`` curve.
    seed:
        Seed forwarded to ``generate_tasks`` (selection/order) — vary across
        runs for the ≥3-seed protocol.
    sentinel_base_url:
        Forwarded to the ``sentinel`` backend.
    hack_penalty:
        Forwarded to :func:`verifier.shape_reward` when filling per-sample
        reward (the C3 penalised-reward ablation).
    out_path:
        If given, the structured report is written as JSON there.
    max_tasks:
        Optional cap on the number of evaluated tasks (useful for smoke runs).

    Returns
    -------
    dict
        A structured report (see the source for the exact keys): build/smoke
        pass rates (problem- and sample-level), ``pass_at_1``, ``pass_at_k``
        curve, mean reward, and aggregate hack-flag rates.
    """
    return asyncio.run(
        _evaluate_async(
            env_or_tasks,
            generate_fn,
            verifier_backend=verifier_backend,
            n=n,
            ks=tuple(ks),
            seed=seed,
            sentinel_base_url=sentinel_base_url,
            hack_penalty=hack_penalty,
            out_path=out_path,
            max_tasks=max_tasks,
        )
    )


def evaluate_multi_seed(
    generate_fn: GenerateFn,
    *,
    seeds: tuple[int, ...] = (0, 1, 2),
    verifier_backend: str = "static",
    n: int = 4,
    ks: tuple[int, ...] = (1,),
    sentinel_base_url: str | None = None,
    hack_penalty: float = 0.0,
    out_path: str | None = None,
    max_tasks: int | None = None,
) -> dict[str, Any]:
    """Run :func:`evaluate` over ≥3 seeds; report mean ± std and 95% CI.

    Implements the project's eval-rigor rule: multiple seeds, ``mean ± std`` and
    a 95% normal CI (``mean ± 1.96 * std / sqrt(num_seeds)``) on the headline
    metrics (``pass_at_1`` and each ``pass@k``). The per-seed reports are kept
    under ``"seeds"``.
    """
    per_seed: list[dict[str, Any]] = []
    for s in seeds:
        per_seed.append(
            evaluate(
                None,
                generate_fn,
                verifier_backend=verifier_backend,
                n=n,
                ks=ks,
                seed=s,
                sentinel_base_url=sentinel_base_url,
                hack_penalty=hack_penalty,
                max_tasks=max_tasks,
            )
        )

    def _agg(values: list[float]) -> dict[str, float]:
        m = sum(values) / len(values)
        var = sum((v - m) ** 2 for v in values) / len(values)  # population variance
        std = var ** 0.5
        ci = 1.96 * std / (len(values) ** 0.5)
        return {"mean": m, "std": std, "ci95": ci, "n_seeds": len(values)}

    metrics: dict[str, Any] = {
        "pass_at_1": _agg([r["pass_at_1"] for r in per_seed]),
        "build_pass_rate": _agg([r["build_pass_rate"] for r in per_seed]),
        "smoke_pass_rate": _agg([r["smoke_pass_rate"] for r in per_seed]),
        "hack_any_rate": _agg([r["hack_any_rate"] for r in per_seed]),
    }
    for k in ks:
        ki = int(k)
        metrics[f"pass_at_{ki}"] = _agg([r["pass_at_k"][ki] for r in per_seed])

    summary = {
        "config": {
            "verifier_backend": verifier_backend,
            "n": n,
            "ks": list(ks),
            "seeds": list(seeds),
            "hack_penalty": hack_penalty,
        },
        "metrics": metrics,
        "seeds": per_seed,
    }
    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, sort_keys=True)
    return summary


# ---------------------------------------------------------------------------
# OpenAI-compatible / vLLM generate_fn factory (importable; not used by tests)
# ---------------------------------------------------------------------------
def make_openai_generate_fn(
    base_url: str,
    model: str,
    api_key_var: str = "OPENAI_API_KEY",
    *,
    system_prompt: str | None = None,
    temperature: float = 0.8,
    max_tokens: int = 1024,
    timeout: float = 120.0,
) -> GenerateFn:
    """Build a :data:`GenerateFn` backed by an OpenAI-compatible chat endpoint.

    Works against any OpenAI-compatible server, including a **vLLM** OpenAI
    server (``vllm serve <model> --port 8000`` -> ``base_url=
    "http://localhost:8000/v1"``). Uses ``httpx`` directly (imported lazily) so
    this module has no hard ``openai`` dependency.

    The returned function issues ``POST {base_url}/chat/completions`` with
    ``n`` set, so the server returns ``n`` candidates in one round-trip when it
    supports it; it falls back to ``n`` sequential calls if the server returns
    fewer choices. ``api_key_var`` names the env var holding the key (vLLM
    accepts any token, e.g. ``"EMPTY"``).

    Temperature should be tuned **per model** for the ``pass@k`` sweep (see
    ``DESIGN.md`` — base and RL models often want different temperatures).
    """
    import httpx  # lazy; not needed by tests

    api_key = os.environ.get(api_key_var, "EMPTY")
    url = base_url.rstrip("/") + "/chat/completions"
    sys_prompt = system_prompt

    def _generate(prompt: str, n: int) -> list[str]:
        messages = []
        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})
        messages.append({"role": "user", "content": prompt})
        body = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "n": n,
        }
        headers = {"Authorization": f"Bearer {api_key}"}
        out: list[str] = []
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(url, json=body, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            for choice in data.get("choices", []):
                msg = choice.get("message", {})
                out.append(msg.get("content", "") or "")
            # Fallback: server ignored n -> top up with sequential calls.
            while len(out) < n:
                body_one = dict(body, n=1)
                r2 = client.post(url, json=body_one, headers=headers)
                r2.raise_for_status()
                ch = r2.json().get("choices", [])
                out.append(ch[0]["message"].get("content", "") if ch else "")
        return out[:n]

    return _generate


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m eval.benchmark",
        description=(
            "Held-out evaluation of an OpenAI-compatible / vLLM policy on the "
            "infra_synth test split (build/smoke rates + unbiased pass@k + "
            "hack-flag rates)."
        ),
    )
    p.add_argument(
        "--base-url", required=True, help="OpenAI-compatible base URL (e.g. http://localhost:8000/v1)"
    )
    p.add_argument("--model", required=True, help="Model name to request")
    p.add_argument(
        "--api-key-var", default="OPENAI_API_KEY", help="Env var holding the API key"
    )
    p.add_argument(
        "--backend", default="static",
        help="Verifier backend: static|local-py|local-docker|sentinel",
    )
    p.add_argument(
        "--sentinel-base-url", default=None, help="Sentinel base URL (for --backend sentinel)"
    )
    p.add_argument("--n", type=int, default=8, help="Samples per problem (use n >> k for pass@k)")
    p.add_argument("--ks", type=int, nargs="+", default=[1, 4, 8], help="pass@k values")
    p.add_argument(
        "--seeds", type=int, nargs="+", default=[0, 1, 2], help="Seeds (>=3 recommended)"
    )
    p.add_argument(
        "--temperature", type=float, default=0.8, help="Sampling temperature (tune per model)"
    )
    p.add_argument("--max-tasks", type=int, default=None, help="Cap evaluated tasks")
    p.add_argument("--out", default=None, help="Write the JSON report here")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    # System prompt mirrors the env's instruction to emit ONLY a Dockerfile.
    tasks_mod = _load_tasks_module()
    system_prompt = getattr(tasks_mod, "SYSTEM_PROMPT", None)
    gen = make_openai_generate_fn(
        args.base_url,
        args.model,
        api_key_var=args.api_key_var,
        system_prompt=system_prompt,
        temperature=args.temperature,
    )
    summary = evaluate_multi_seed(
        gen,
        seeds=tuple(args.seeds),
        verifier_backend=args.backend,
        n=args.n,
        ks=tuple(args.ks),
        sentinel_base_url=args.sentinel_base_url,
        out_path=args.out,
        max_tasks=args.max_tasks,
    )
    print(json.dumps(summary["metrics"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
