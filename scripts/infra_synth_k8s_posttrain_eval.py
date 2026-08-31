"""Held-out pass rate for infra_synth/k8s, base or GRPO-trained adapter.

Two axes, both on the SAME 15 test-split tasks the round-1 baseline used
(scripts/infra_synth_pass_rate.py's _task_pool -- verified identical spec_ids):

  greedy n=1 (--samples-per-task 1, default): comparable to the round-1 /
    corrected-baseline number (14/15, 93.3%). This is where the model already
    has little headroom (near ceiling).
  sampled t=0.7 n=4/8 (--do-sample --temperature 0.7 --samples-per-task N):
    the axis GRPO actually trains on (num_generations=4, temperature=0.7 --
    see training/configs/infra_synth_k8s_6gb.yaml and
    results/infra_synth_eval/k8s_variance_check_t0.7.json). This is where the
    base model is fragile (variance-check mean reward 0.41 on TRAIN) and where
    the training signal has real room to show up on TEST.

Loads the base Qwen2.5-0.5B-Instruct + (optionally) the LoRA adapter saved by
`training.run.train()` (trainer.save_model(cfg.output_dir)).
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


def _make_generate_fn(
    tok, model, system_prompt: str, max_new_tokens: int, *, do_sample: bool, temperature: float
):
    import torch

    def _gen(prompt: str, n: int) -> list[str]:
        msgs = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt").to(model.device)
        out = []
        for _ in range(n):
            gen_kwargs = dict(max_new_tokens=max_new_tokens, pad_token_id=tok.eos_token_id)
            if do_sample:
                gen_kwargs.update(do_sample=True, temperature=temperature, top_p=1.0)
            else:
                gen_kwargs.update(do_sample=False)
            with torch.no_grad():
                gen = model.generate(**inputs, **gen_kwargs)
            out.append(tok.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
        return out

    return _gen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter-dir", default=None, help="Path to the saved LoRA adapter (None = base model)")
    ap.add_argument("--n-tasks", type=int, default=15)
    ap.add_argument("--samples-per-task", type=int, default=1)
    ap.add_argument("--do-sample", action="store_true", help="Sample instead of greedy decoding")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tasks = _task_pool("k8s", args.n_tasks)
    tok, model = _load_model(args.adapter_dir)
    gen_fn = _make_generate_fn(
        tok, model, tasks_mod.K8S_SYSTEM_PROMPT, args.max_new_tokens,
        do_sample=args.do_sample, temperature=args.temperature,
    )

    ks = (1, args.samples_per_task) if args.samples_per_task > 1 else (1,)
    report = evaluate(
        tasks, gen_fn, verifier_backend="local", n=args.samples_per_task, ks=ks, seed=0, out_path=args.out
    )
    print(json.dumps(
        {
            "adapter_dir": args.adapter_dir,
            "do_sample": args.do_sample,
            "temperature": args.temperature if args.do_sample else None,
            "n_tasks": report["n_tasks"],
            "samples_per_task": args.samples_per_task,
            "build_pass_rate_any_of_n": report["build_pass_rate"],
            "smoke_pass_rate_any_of_n": report["smoke_pass_rate"],
            "sample_build_rate_mean": report["sample_build_rate"],
            "sample_smoke_rate_mean": report["sample_smoke_rate"],
            "pass_at_1": report["pass_at_1"],
            "pass_at_k": report["pass_at_k"],
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
