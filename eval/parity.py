"""M2 deliverable: Sentinel ↔ local-check parity report.

Runs each artifact through **both** ``get_verifier("local-py")`` (the weak local
subprocess baseline) and ``get_verifier("sentinel")`` (the hardened sandbox) and
reports the agreement rate on the ``(build_ok, smoke_ok)`` verdict pair, plus a
list of any disagreements. This proves that **routing the reward through Sentinel
produces the same verdict as the local baseline** — the correctness half of M2
(the throughput half is :mod:`eval.throughput`).

Why this holds by construction: both backends execute the *same* deterministic
harness string (:func:`verifier.smoke.checks.build_python_harness`) — ``local-py``
as a local ``python3 -I`` subprocess, ``sentinel`` by submitting that identical
source to the nsjail/cgroups sandbox. The harness is stdlib-only and
deterministic, so for any well-behaved artifact the two ``(build_ok, smoke_ok)``
verdicts must match; this report *measures* that and surfaces any environment-
specific divergence (e.g. a sandbox that kills the harness on resource limits).

Injectable transport / base_url
-------------------------------
``parity_report`` accepts a ``sentinel_transport`` (``httpx`` transport). Pass an
``httpx.MockTransport`` to run **entirely locally** against a simulated Sentinel
(CI / ``tests/test_parity.py``); pass ``None`` with a real ``base_url`` to check
parity against a live Sentinel (requires ``make up`` in the Sentinel repo, port
8080). stdlib + ``httpx`` + ``verifier`` only.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any

import httpx

from verifier import get_verifier
from verifier.sentinel_client import SentinelClient, SentinelVerifier
from verifier.types import ArtifactKind, ResourceLimits, VerifySpec

__all__ = ["parity_report"]


async def parity_report(
    artifacts: list[str],
    spec: VerifySpec,
    *,
    sentinel_transport: httpx.AsyncBaseTransport | None = None,
    base_url: str = "http://localhost:8080",
) -> dict[str, Any]:
    """Compare ``local-py`` and ``sentinel`` verdicts over ``artifacts``.

    Parameters
    ----------
    artifacts:
        The artifacts to verify with both backends.
    spec:
        The :class:`VerifySpec` to verify each artifact against.
    sentinel_transport:
        Optional ``httpx`` transport injected into the Sentinel client. Use an
        ``httpx.MockTransport`` to simulate Sentinel locally; ``None`` hits a
        live Sentinel at ``base_url``.
    base_url:
        Sentinel base URL (default ``http://localhost:8080``).

    Returns
    -------
    dict
        ``{"n", "n_agree", "agreement_rate", "build_agreement_rate",
        "smoke_agreement_rate", "disagreements": [...], "records": [...]}``.
        ``agreement_rate`` is on the full ``(build_ok, smoke_ok)`` pair;
        ``disagreements`` lists each mismatch with both backends' verdicts and
        statuses for debugging.
    """
    local = get_verifier("local-py")
    sentinel_client = SentinelClient(base_url=base_url, transport=sentinel_transport, timeout=60.0)
    sentinel: SentinelVerifier = SentinelVerifier(
        client=sentinel_client, poll_interval=0.001, deadline_s=60.0
    )

    records: list[dict[str, Any]] = []
    disagreements: list[dict[str, Any]] = []
    n_agree = 0
    build_agree = 0
    smoke_agree = 0

    try:
        for idx, artifact in enumerate(artifacts):
            # Run both backends concurrently for each artifact.
            local_res, sent_res = await asyncio.gather(
                local.verify(artifact, spec),
                sentinel.verify(artifact, spec),
            )
            lb, ls = bool(local_res.build_ok), bool(local_res.smoke_ok)
            sb, ss = bool(sent_res.build_ok), bool(sent_res.smoke_ok)
            build_ok_match = lb == sb
            smoke_ok_match = ls == ss
            full_match = build_ok_match and smoke_ok_match
            if build_ok_match:
                build_agree += 1
            if smoke_ok_match:
                smoke_agree += 1
            if full_match:
                n_agree += 1

            rec = {
                "index": idx,
                "spec_id": spec.spec_id,
                "local_py": {
                    "build_ok": lb, "smoke_ok": ls, "status": local_res.status,
                },
                "sentinel": {
                    "build_ok": sb, "smoke_ok": ss, "status": sent_res.status,
                },
                "agree": full_match,
            }
            records.append(rec)
            if not full_match:
                disagreements.append(rec)
    finally:
        await sentinel.aclose()

    n = len(artifacts)
    denom = max(1, n)
    return {
        "n": n,
        "n_agree": n_agree,
        "agreement_rate": n_agree / denom,
        "build_agreement_rate": build_agree / denom,
        "smoke_agreement_rate": smoke_agree / denom,
        "disagreements": disagreements,
        "records": records,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m eval.parity",
        description=(
            "Sentinel<->local-py parity report (M2). Running against a LIVE "
            "Sentinel requires `make up` in the Sentinel repo (default port "
            "8080)."
        ),
    )
    p.add_argument("--base-url", default="http://localhost:8080", help="Sentinel base URL")
    p.add_argument("--n", type=int, default=8, help="Number of demo artifacts to compare")
    p.add_argument("--out", default=None, help="Write the JSON report here")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    spec = VerifySpec(
        spec_id="parity-df",
        kind=ArtifactKind.DOCKERFILE,
        smoke={"port": 8000, "must_contain": ["FROM", "CMD"], "health_path": "/health"},
        limits=ResourceLimits(wall_s=5, mem_mb=128),
    )
    good = (
        "FROM python:3.11-slim\nWORKDIR /app\nCOPY . .\n"
        "RUN pip install --no-cache-dir fastapi uvicorn\n"
        'EXPOSE 8000\nCMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]\n'
    )
    trivial = 'FROM python:3.11-slim\nEXPOSE 8000\nCMD ["true"]\n'
    artifacts = [good if i % 2 == 0 else trivial for i in range(args.n)]
    report = asyncio.run(parity_report(artifacts, spec, base_url=args.base_url))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, sort_keys=True)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
