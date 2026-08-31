"""Gold sanity check: does the gold reference pass its own REAL verifier, per kind?

Uses the kind-aware "local" dispatcher: genuine docker build+smoke for
DOCKERFILE/COMPOSE, `terraform validate` for TERRAFORM, `kubeconform -strict`
for K8S. CI_YAML has no genuine local verifier -> LocalGenuineVerifier falls
back to StaticVerifier for it (checked explicitly below).
"""
import asyncio
import json
import time

from infra_synth import tasks as tasks_mod
from infra_synth.gold import (
    gold_ci_yaml,
    gold_compose,
    gold_dockerfile,
    gold_k8s,
    gold_terraform,
)
from verifier import get_verifier

GOLDS = {
    "dockerfile": gold_dockerfile,
    "compose": gold_compose,
    "terraform": gold_terraform,
    "k8s": gold_k8s,
    "ci-yaml": gold_ci_yaml,
}


async def main() -> None:
    v = get_verifier("local")  # kind-aware dispatcher over genuine backends
    # Ports 8080/3000 collide with the live sentinel-api/sentinel-frontend
    # containers (docker host-port bind) -- do not touch that stack, so pick a
    # task whose port avoids those two.
    _BAD_PORTS = {8080, 3000}

    out = []
    for kind, gold_fn in GOLDS.items():
        candidates = tasks_mod.generate_tasks(n=20, seed=0, split="test", kind=kind)
        t = next(
            (c for c in candidates if c["info"]["smoke"].get("port") not in _BAD_PORTS),
            candidates[0],
        )
        info = t["info"]
        spec = tasks_mod.build_verify_spec(info)
        artifact = gold_fn(info)
        t0 = time.time()
        result = await v.verify(artifact, spec)
        dt = time.time() - t0
        rec = {
            "kind": kind,
            "spec_id": info["spec_id"],
            "build_ok": result.build_ok,
            "smoke_ok": result.smoke_ok,
            "status": result.status,
            "wall_s": round(dt, 2),
            "stderr_tail": (result.stderr_tail or "")[-300:],
        }
        out.append(rec)
        print(json.dumps(rec, indent=2))
    with open("results/infra_synth_eval/gold_sanity.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    asyncio.run(main())
