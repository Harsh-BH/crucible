"""Measure Qwen2.5-0.5B-Instruct's pass rate on infra_synth (test split),
graded by the REAL local verifier per artifact kind:
  DOCKERFILE/COMPOSE -> genuine docker build + run + HTTP smoke probe
  TERRAFORM          -> genuine `terraform validate`
  K8S                -> genuine `kubeconform -strict`
  CI_YAML             -> StaticVerifier fallback (no genuine local backend exists)

Uses the repo's sanctioned eval entrypoint (eval.benchmark.evaluate) with a
custom generate_fn wrapping the LOCAL HF model (no vLLM server needed, no
model download -- Qwen2.5-0.5B-Instruct is already in the HF cache).

n=1 (single greedy sample) per task: real docker/terraform/kubeconform runs
are too slow on a 6GB laptop GPU + this disk budget for a pass@k sweep.

Ports 8080/3000 are excluded from DOCKERFILE/COMPOSE task pools -- they
collide with the live sentinel-api/sentinel-frontend containers (a real
docker host-port bind conflict, unrelated to model or gold quality).
"""
from __future__ import annotations

import argparse
import json
import os
import time

from eval.benchmark import evaluate
from infra_synth import tasks as tasks_mod

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

_SYSTEM_PROMPTS = {
    "dockerfile": tasks_mod.SYSTEM_PROMPT,
    "compose": tasks_mod.COMPOSE_SYSTEM_PROMPT,
    "ci-yaml": tasks_mod.CI_YAML_SYSTEM_PROMPT,
    "terraform": tasks_mod.TERRAFORM_SYSTEM_PROMPT,
    "k8s": tasks_mod.K8S_SYSTEM_PROMPT,
}

# Live sentinel-api (8080) / sentinel-frontend (3000) containers already bind
# these host ports -- do not touch that stack, so skip tasks that land on them.
_BAD_PORTS = {8080, 3000}

_OUT_DIR = "results/infra_synth_eval"


def _load_model():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    return tok, model


def _make_generate_fn(tok, model, system_prompt: str, max_new_tokens: int):
    import torch

    def _gen(prompt: str, n: int) -> list[str]:
        msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt").to(model.device)
        out_texts = []
        for _ in range(n):
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,  # greedy: single sample, no noise from temperature
                    pad_token_id=tok.eos_token_id,
                )
            completion = tok.decode(
                out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )
            out_texts.append(completion)
        return out_texts

    return _gen


def _task_pool(kind: str, n: int, *, seed: int = 0) -> list[dict]:
    """First ``n`` test-split tasks for ``kind``, skipping _BAD_PORTS collisions."""
    pool = tasks_mod.generate_tasks(n=max(n * 3, 30), seed=seed, split="test", kind=kind)
    filtered = [t for t in pool if t["info"]["smoke"].get("port") not in _BAD_PORTS]
    return filtered[:n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", nargs="+", default=["dockerfile", "compose", "terraform", "k8s", "ci-yaml"])
    ap.add_argument("--n", type=int, nargs="+", required=True, help="one N per --kinds entry")
    ap.add_argument("--max-new-tokens", type=int, default=768)
    args = ap.parse_args()
    assert len(args.n) == len(args.kinds)

    os.makedirs(_OUT_DIR, exist_ok=True)
    tok, model = _load_model()

    summary = {}
    for kind, n in zip(args.kinds, args.n):
        tasks = _task_pool(kind, n)
        gen_fn = _make_generate_fn(tok, model, _SYSTEM_PROMPTS[kind], args.max_new_tokens)
        t0 = time.time()
        report = evaluate(
            tasks,
            gen_fn,
            verifier_backend="local",
            n=1,
            ks=(1,),
            seed=0,
            out_path=os.path.join(_OUT_DIR, f"{kind}.json"),
        )
        dt = time.time() - t0
        summary[kind] = {
            "n_tasks": report["n_tasks"],
            "build_pass_rate": report["build_pass_rate"],
            "smoke_pass_rate": report["smoke_pass_rate"],
            "mean_wall_s_per_sample": report["mean_wall_s"],
            "total_wall_s": round(dt, 1),
        }
        print(json.dumps({kind: summary[kind]}, indent=2))

    with open(os.path.join(_OUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
