"""Does the REAL local verifier correctly FAIL the gold (honest) reference on
an impossible_tasks()-mutated spec, for each kind? local-* backends only (no
Sentinel touch) -- this is the C3 false-positive probe run for real instead of
--mock, restricted to what local-* can grade.
"""
import asyncio
import json

from infra_synth import tasks as tasks_mod
from infra_synth.gold import (
    gold_ci_yaml,
    gold_compose,
    gold_dockerfile,
    gold_k8s,
    gold_terraform,
)
from infra_synth.impossible import impossible_tasks
from verifier import get_verifier

GOLDS = {
    "dockerfile": gold_dockerfile,
    "compose": gold_compose,
    "terraform": gold_terraform,
    "k8s": gold_k8s,
    "ci-yaml": gold_ci_yaml,
}
_BAD_PORTS = {8080, 3000}


async def main() -> None:
    v = get_verifier("local")
    out = []
    for kind, gold_fn in GOLDS.items():
        candidates = impossible_tasks(n=20, seed=0, split="test", kind=kind)
        picked = [c for c in candidates if c["info"]["smoke"].get("port") not in _BAD_PORTS][:3]
        for task in picked:
            info = task["info"]
            spec = tasks_mod.build_verify_spec(info)
            artifact = gold_fn(info)
            r = await v.verify(artifact, spec)
            undeserved_pass = bool(r.build_ok and r.smoke_ok)
            rec = {
                "kind": kind,
                "mutation": info["impossible"]["mutation"],
                "spec_id": info["spec_id"],
                "build_ok": r.build_ok,
                "smoke_ok": r.smoke_ok,
                "undeserved_pass": undeserved_pass,
                "status": r.status,
            }
            out.append(rec)
            print(json.dumps(rec))
    n = len(out)
    undeserved = sum(1 for r in out if r["undeserved_pass"])
    print(f"\nN={n} impossible-task gold trials; undeserved passes (verifier fooled) = {undeserved}")
    with open("results/infra_synth_eval/impossible_gold.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    asyncio.run(main())
