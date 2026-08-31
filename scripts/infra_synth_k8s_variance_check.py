"""Pre-training variance check for GRPO on infra_synth/k8s.

GRPO needs reward VARIANCE WITHIN A GROUP (same prompt, G samples) or the
advantage is exactly 0 and there's no gradient. Round 1's baseline found k8s
is the only kind with within-group headroom (53% pass, near the p=0.5 ideal
for a binary reward) -- dockerfile/compose/terraform were 0%, all-zero groups
guaranteed. This script checks that reasoning against the ACTUAL group-level
reward distribution (TRAIN split -- the split training will actually sample
from) before spending GPU time on a real trl run.

Uses the same generate_fn shape and hyperparameters (temperature=1.0,
top_p=1.0, do_sample=True) GRPO sampling would use -- NOT the greedy n=1 the
baseline eval used (greedy would trivially collapse every group to std=0).
"""
from __future__ import annotations

import argparse
import json

from infra_synth import tasks as tasks_mod
from verifier import get_verifier, shape_reward
import asyncio

_BAD_PORTS = {8080, 3000}  # live sentinel-api/frontend containers


def _load_model():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = "Qwen/Qwen2.5-0.5B-Instruct"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    return tok, model


def _sample_group(
    tok, model, system_prompt: str, prompt: str, g: int, max_new_tokens: int, temperature: float
) -> list[str]:
    import torch

    msgs = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(model.device)
    out = []
    for _ in range(g):
        with torch.no_grad():
            gen = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=1.0,
                pad_token_id=tok.eos_token_id,
            )
        out.append(tok.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-prompts", type=int, default=8)
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--split", default="train")
    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--out", default="results/infra_synth_eval/k8s_variance_check.json")
    args = ap.parse_args()

    from infra_synth.parser import extract_dockerfile

    tasks = tasks_mod.generate_tasks(n=args.n_prompts * 3, seed=0, split=args.split, kind="k8s")
    tasks = [t for t in tasks if t["info"]["smoke"].get("port") not in _BAD_PORTS][: args.n_prompts]

    tok, model = _load_model()
    verifier = get_verifier("local")

    groups = []
    for t in tasks:
        completions = _sample_group(
            tok, model, tasks_mod.K8S_SYSTEM_PROMPT, t["question"], args.group_size,
            args.max_new_tokens, args.temperature,
        )
        spec = tasks_mod.build_verify_spec(t["info"])
        rewards = []
        for c in completions:
            artifact = extract_dockerfile(c) or c
            r = asyncio.run(verifier.verify(artifact, spec))
            rewards.append(shape_reward(r))
        mean = sum(rewards) / len(rewards)
        var = sum((x - mean) ** 2 for x in rewards) / len(rewards)
        std = var ** 0.5
        rec = {"spec_id": t["info"]["spec_id"], "rewards": rewards, "mean": mean, "std": std}
        groups.append(rec)
        print(json.dumps(rec))

    n = len(groups)
    zero_std = sum(1 for g in groups if g["std"] == 0.0)
    frac_reward_zero_std = zero_std / n if n else 1.0
    overall_mean = sum(g["mean"] for g in groups) / n if n else 0.0
    print(f"\nN groups={n} group_size={args.group_size} split={args.split}")
    print(f"frac_reward_zero_std={frac_reward_zero_std:.3f} ({zero_std}/{n})")
    print(f"overall mean reward={overall_mean:.3f}")

    with open(args.out, "w") as f:
        json.dump(
            {
                "n_groups": n,
                "group_size": args.group_size,
                "split": args.split,
                "frac_reward_zero_std": frac_reward_zero_std,
                "overall_mean_reward": overall_mean,
                "groups": groups,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
