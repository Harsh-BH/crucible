"""Post-training held-out pass rate for a GRPO-trained infra_synth/k8s adapter.

Same harness, same test-split task SET, same greedy n=1 decoding as the
round-1 baseline (scripts/infra_synth_pass_rate.py) -- this is what makes
before/after comparable. Loads the base Qwen2.5-0.5B-Instruct + the LoRA
adapter saved by `training.run.train()` (trainer.save_model(cfg.output_dir)).
"""
from __future__ import annotations

import argparse
import json

from eval.benchmark import evaluate
from infra_synth import tasks as tasks_mod

# Reuse the EXACT task-selection logic the baseline used, so the task set is
# identical (verified: same 15 spec_ids as results/infra_synth_eval/k8s.json).
from infra_synth_pass_rate import _task_pool  # noqa: E402  (scripts/ on sys.path)


def _load_model(adapter_dir: str | None):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = "Qwen/Qwen2.5-0.5B-Instruct"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map="cuda")
    if adapter_dir:
        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return tok, model


def _make_generate_fn(tok, model, system_prompt: str, max_new_tokens: int):
    import torch

    def _gen(prompt: str, n: int) -> list[str]:
        msgs = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt").to(model.device)
        out = []
        for _ in range(n):
            with torch.no_grad():
                gen = model.generate(
                    **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                    pad_token_id=tok.eos_token_id,
                )
            out.append(tok.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
        return out

    return _gen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter-dir", default=None, help="Path to the saved LoRA adapter (None = base model)")
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tasks = _task_pool("k8s", args.n)
    tok, model = _load_model(args.adapter_dir)
    gen_fn = _make_generate_fn(tok, model, tasks_mod.K8S_SYSTEM_PROMPT, args.max_new_tokens)

    report = evaluate(tasks, gen_fn, verifier_backend="local", n=1, ks=(1,), seed=0, out_path=args.out)
    print(json.dumps(
        {
            "adapter_dir": args.adapter_dir,
            "n_tasks": report["n_tasks"],
            "build_pass_rate": report["build_pass_rate"],
            "smoke_pass_rate": report["smoke_pass_rate"],
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
