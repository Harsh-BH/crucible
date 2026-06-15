"""TRL GRPO driver for Crucible — the self-contained, hackable baseline.

This is the **TRL** training path (the project's *primary* stack is prime-rl;
see ``training/configs/*.toml`` and ``training/README.md`` for when to use
each). It wires the project's datasets (:mod:`training.data`) and rewards
(:mod:`training.rewards`) into ``trl.GRPOTrainer`` so the whole GRPO loop is
runnable and editable from this repo.

Import discipline (IMPORTANT)
-----------------------------
``torch`` / ``trl`` / ``transformers`` / ``peft`` are imported **lazily inside
functions**, never at module top level, so ``import training.run`` works in a
plain stdlib + pyyaml environment (CI / unit tests). All flag-parsing, config
loading and reward/dataset construction lives in torch-free helpers. The heavy
imports happen only when :func:`train` actually runs.

Usage
-----
    python training/run.py --env gsm8k --model Qwen/Qwen3-1.7B \\
        --num-generations 8 --beta 0.0 --lr 1e-6 --max-steps 100 --seed 0

    # Fast CPU code-path check (tiny model, 2 steps, tiny batch):
    python training/run.py --env gsm8k --smoke

    # Read defaults from a YAML config, override on the CLI:
    python training/run.py --config training/configs/m1_repro.yaml --seed 1

The **M2 Sentinel path** is just a backend swap::

    python training/run.py --env infra_synth --verifier-backend sentinel \\
        --sentinel-base-url http://localhost:8080

Real training needs a GPU; ``--smoke`` only proves the code path on CPU.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable

__all__ = ["main", "train", "build_config", "RunConfig", "parse_args"]

# A deliberately tiny model for the CPU ``--smoke`` code-path check. It is the
# smallest causal-LM that exercises the full TRL GRPO loop quickly without a GPU.
_SMOKE_MODEL = "trl-internal-testing/tiny-Qwen3ForCausalLM"


# ---------------------------------------------------------------------------
# Config dataclass (torch-free) — the single source of truth for a run
# ---------------------------------------------------------------------------
@dataclass
class RunConfig:
    """All knobs for one GRPO run. Populated from CLI flags (+ optional YAML).

    The field names mirror the CLI flags; :func:`build_config` maps them onto a
    ``trl.GRPOConfig`` + ``peft.LoraConfig``.
    """

    env: str = "gsm8k"
    model: str = "Qwen/Qwen3-1.7B"
    # LoRA
    lora: bool = True
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    # GRPO core
    num_generations: int = 8
    beta: float = 0.0
    epsilon: float = 0.2
    epsilon_high: float | None = None  # set -> DAPO clip-higher
    loss_type: str = "grpo"  # {grpo, bnpo, dr_grpo}
    importance_sampling_level: str = "token"  # {token, sequence}  (sequence = GSPO)
    scale_rewards: bool = False  # False = Dr.GRPO (no within-group std normalisation)
    temperature: float = 1.0
    top_p: float = 1.0
    max_steps: int = 100
    max_completion_length: int = 1024
    num_iterations: int = 1
    # optim / batching
    lr: float = 1e-6
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    # data
    dataset_split: str = "train"
    dataset_size: int | None = None
    # reward (infra_synth / M2)
    verifier_backend: str = "static"
    sentinel_base_url: str | None = None
    build_weight: float = 0.3
    smoke_weight: float = 0.7
    hack_penalty: float = 0.0
    use_format_reward: bool = True
    # bookkeeping
    seed: int = 0
    wandb_project: str | None = None
    wandb_name: str | None = None
    output_dir: str = "outputs/run"
    logging_steps: int = 1
    save_steps: int = 0  # 0 -> no intermediate checkpoints
    smoke: bool = False
    extra_grpo: dict[str, Any] = field(default_factory=dict)

    def resolved_model(self) -> str:
        """The model to actually load (tiny override under ``--smoke``)."""
        return _SMOKE_MODEL if self.smoke else self.model


# ---------------------------------------------------------------------------
# CLI + YAML
# ---------------------------------------------------------------------------
def _add_bool_flag(parser: argparse.ArgumentParser, name: str, default: bool, help_: str) -> None:
    """Add a paired ``--name`` / ``--no-name`` boolean flag (default ``None``).

    Default is ``None`` so we can tell "user did not pass it" from "user set it",
    which lets YAML/dataclass defaults win when the flag is omitted.
    """
    dest = name.replace("-", "_")
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument(f"--{name}", dest=dest, action="store_true", default=None, help=help_)
    grp.add_argument(f"--no-{name}", dest=dest, action="store_false", default=None)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="training.run",
        description="TRL GRPO driver for Crucible (gsm8k / infra_synth / reverse_text).",
    )
    p.add_argument("--config", default=None, help="Optional YAML config of defaults.")
    p.add_argument("--env", choices=["gsm8k", "infra_synth", "reverse_text"], default=None)
    p.add_argument("--model", default=None, help="HF model id (default Qwen/Qwen3-1.7B).")
    # LoRA
    _add_bool_flag(p, "lora", True, "Use LoRA/PEFT (default on); --no-lora for full FT.")
    p.add_argument("--lora-rank", type=int, default=None)
    p.add_argument("--lora-alpha", type=int, default=None)
    p.add_argument("--lora-dropout", type=float, default=None)
    # GRPO
    p.add_argument("--num-generations", type=int, default=None, help="Group size G.")
    p.add_argument("--beta", type=float, default=None, help="KL coefficient (0 = KL off).")
    p.add_argument("--epsilon", type=float, default=None, help="PPO clip epsilon.")
    p.add_argument(
        "--epsilon-high", type=float, default=None,
        help="Upper clip (enables DAPO clip-higher; e.g. 0.28).",
    )
    p.add_argument("--loss-type", choices=["grpo", "bnpo", "dr_grpo"], default=None)
    p.add_argument(
        "--importance-sampling-level", choices=["token", "sequence"], default=None,
        help="'sequence' = GSPO (TRL only).",
    )
    _add_bool_flag(
        p, "scale-rewards", False,
        "Scale rewards by within-group std (default OFF = Dr.GRPO).",
    )
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--max-completion-length", type=int, default=None)
    p.add_argument("--num-iterations", type=int, default=None)
    # optim / batch
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--per-device-train-batch-size", type=int, default=None)
    p.add_argument("--gradient-accumulation-steps", type=int, default=None)
    # data
    p.add_argument("--dataset-split", default=None)
    p.add_argument("--dataset-size", type=int, default=None)
    # reward / M2
    p.add_argument(
        "--verifier-backend",
        choices=["static", "local-py", "local-docker", "sentinel"], default=None,
        help="infra_synth reward backend ('sentinel' = M2).",
    )
    p.add_argument("--sentinel-base-url", default=None)
    p.add_argument("--build-weight", type=float, default=None)
    p.add_argument("--smoke-weight", type=float, default=None)
    p.add_argument("--hack-penalty", type=float, default=None)
    _add_bool_flag(p, "format-reward", True, "Add the auxiliary boxed-format reward.")
    # bookkeeping
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--wandb-project", default=None)
    p.add_argument("--wandb-name", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--logging-steps", type=int, default=None)
    p.add_argument("--save-steps", type=int, default=None)
    p.add_argument(
        "--smoke", action="store_true",
        help="CPU code-path check: tiny model + 2 steps + tiny batch.",
    )
    return p


# Map dataclass field -> the YAML/CLI dest name (they match 1:1 here).
_FIELD_NAMES = {
    "env", "model", "lora", "lora_rank", "lora_alpha", "lora_dropout",
    "num_generations", "beta", "epsilon", "epsilon_high", "loss_type",
    "importance_sampling_level", "scale_rewards", "temperature", "top_p",
    "max_steps", "max_completion_length", "num_iterations", "lr",
    "per_device_train_batch_size", "gradient_accumulation_steps",
    "dataset_split", "dataset_size", "verifier_backend", "sentinel_base_url",
    "build_weight", "smoke_weight", "hack_penalty", "use_format_reward",
    "seed", "wandb_project", "wandb_name", "output_dir", "logging_steps",
    "save_steps", "smoke",
}


def _load_yaml(path: str) -> dict[str, Any]:
    import yaml  # pyyaml is a core dependency

    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config {path!r} must be a YAML mapping, got {type(data).__name__}")
    return data


def config_from_args(args: argparse.Namespace) -> RunConfig:
    """Build a :class:`RunConfig` from parsed args, layering YAML under the CLI.

    Precedence (low -> high): dataclass defaults < YAML ``--config`` < CLI flags
    that were actually provided (non-``None``). The YAML may use either the
    dataclass field names or the ``format_reward``/``use_format_reward`` alias.
    """
    cfg = RunConfig()

    # Layer 1: YAML config (if any).
    if getattr(args, "config", None):
        raw = _load_yaml(args.config)
        # Accept a "seeds" list in YAML (multi-seed plans) but ignore it here —
        # a single run uses one seed; seeds.py consumes the list.
        raw.pop("seeds", None)
        # Alias: YAML may say `format_reward:` for the dataclass `use_format_reward`.
        if "format_reward" in raw and "use_format_reward" not in raw:
            raw["use_format_reward"] = raw.pop("format_reward")
        for k, v in raw.items():
            key = k.replace("-", "_")
            if key in _FIELD_NAMES or key == "use_format_reward":
                setattr(cfg, "use_format_reward" if key == "format_reward" else key, v)
            else:
                cfg.extra_grpo[key] = v

    # Layer 2: CLI overrides (only fields the user actually set, i.e. not None).
    overrides: dict[str, Any] = {
        "env": args.env, "model": args.model, "lora": args.lora,
        "lora_rank": args.lora_rank, "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout, "num_generations": args.num_generations,
        "beta": args.beta, "epsilon": args.epsilon, "epsilon_high": args.epsilon_high,
        "loss_type": args.loss_type,
        "importance_sampling_level": args.importance_sampling_level,
        "scale_rewards": args.scale_rewards, "temperature": args.temperature,
        "top_p": args.top_p, "max_steps": args.max_steps,
        "max_completion_length": args.max_completion_length,
        "num_iterations": args.num_iterations, "lr": args.lr,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "dataset_split": args.dataset_split, "dataset_size": args.dataset_size,
        "verifier_backend": args.verifier_backend,
        "sentinel_base_url": args.sentinel_base_url,
        "build_weight": args.build_weight, "smoke_weight": args.smoke_weight,
        "hack_penalty": args.hack_penalty, "use_format_reward": args.format_reward,
        "seed": args.seed, "wandb_project": args.wandb_project,
        "wandb_name": args.wandb_name, "output_dir": args.output_dir,
        "logging_steps": args.logging_steps, "save_steps": args.save_steps,
    }
    for key, val in overrides.items():
        if val is not None:
            setattr(cfg, key, val)
    if args.smoke:
        cfg.smoke = True

    _apply_smoke_overrides(cfg)
    return cfg


def _apply_smoke_overrides(cfg: RunConfig) -> None:
    """Shrink a config for the CPU ``--smoke`` code-path check.

    Keeps the full code path (dataset -> rewards -> GRPOTrainer.train) but makes
    it finish in seconds on CPU: tiny model, 2 steps, a 4-sample dataset and a
    small group. We do NOT touch user-chosen reward/verifier settings.
    """
    if not cfg.smoke:
        return
    cfg.max_steps = 2
    cfg.num_generations = 2
    cfg.per_device_train_batch_size = 2
    cfg.gradient_accumulation_steps = 1
    cfg.max_completion_length = 16
    cfg.dataset_size = 4
    cfg.lora_rank = min(cfg.lora_rank, 4)
    if cfg.wandb_project is None:
        os.environ.setdefault("WANDB_DISABLED", "true")


def parse_args(argv: list[str] | None = None) -> RunConfig:
    """Parse ``argv`` (default ``sys.argv``) into a :class:`RunConfig`."""
    parser = build_arg_parser()
    return config_from_args(parser.parse_args(argv))


# ---------------------------------------------------------------------------
# Dataset + rewards (delegates to torch-free helpers)
# ---------------------------------------------------------------------------
def build_dataset(cfg: RunConfig) -> Any:
    """Build the training dataset for ``cfg.env`` via :mod:`training.data`."""
    from . import data as data_mod

    if cfg.env == "gsm8k":
        return data_mod.build_gsm8k(
            split=cfg.dataset_split, n=cfg.dataset_size, seed=cfg.seed
        )
    if cfg.env == "infra_synth":
        return data_mod.build_infra_synth(
            split=cfg.dataset_split, n=cfg.dataset_size, seed=cfg.seed
        )
    if cfg.env == "reverse_text":
        return _build_reverse_text(cfg)
    raise ValueError(f"unknown env {cfg.env!r}")


def _build_reverse_text(cfg: RunConfig) -> Any:
    """Tiny synthetic 'reverse the text' dataset (no external download).

    Mirrors prime-rl's reverse-text toy task so the TRL path has a matching,
    download-free sanity env. Columns: ``prompt`` (chat) + ``answer`` (reversed).
    """
    import random as _random

    from datasets import Dataset

    rng = _random.Random(cfg.seed)
    words = [
        "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel",
        "india", "juliet", "kilo", "lima", "mike", "november", "oscar", "papa",
    ]
    n = cfg.dataset_size or 256
    prompts, answers = [], []
    for _ in range(n):
        k = rng.randint(2, 5)
        phrase = " ".join(rng.choice(words) for _ in range(k))
        prompts.append(
            [
                {"role": "system", "content": "Reverse the characters of the user's text. Output only the reversed text."},
                {"role": "user", "content": phrase},
            ]
        )
        answers.append(phrase[::-1])
    return Dataset.from_dict({"prompt": prompts, "answer": answers})


def build_reward_funcs(cfg: RunConfig) -> list[Callable[..., list[float]]]:
    """Build the list of TRL reward callables for ``cfg.env``."""
    from . import rewards as rewards_mod

    if cfg.env in ("gsm8k", "reverse_text"):
        funcs: list[Callable[..., list[float]]] = [rewards_mod.gsm8k_reward]
        if cfg.use_format_reward and cfg.env == "gsm8k":
            funcs.append(rewards_mod.format_reward)
        return funcs
    if cfg.env == "infra_synth":
        return [
            rewards_mod.make_infra_synth_reward(
                verifier_backend=cfg.verifier_backend,
                build_weight=cfg.build_weight,
                smoke_weight=cfg.smoke_weight,
                hack_penalty=cfg.hack_penalty,
                sentinel_base_url=cfg.sentinel_base_url,
            )
        ]
    raise ValueError(f"unknown env {cfg.env!r}")


# ---------------------------------------------------------------------------
# GRPOConfig / LoraConfig construction (lazy heavy imports)
# ---------------------------------------------------------------------------
def build_config(cfg: RunConfig) -> Any:
    """Construct a ``trl.GRPOConfig`` from ``cfg`` (imports trl lazily)."""
    from trl import GRPOConfig

    report_to = "wandb" if cfg.wandb_project else "none"
    if cfg.wandb_project:
        os.environ.setdefault("WANDB_PROJECT", cfg.wandb_project)

    kwargs: dict[str, Any] = dict(
        output_dir=cfg.output_dir,
        seed=cfg.seed,
        learning_rate=cfg.lr,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_generations=cfg.num_generations,
        max_completion_length=cfg.max_completion_length,
        max_steps=cfg.max_steps,
        num_iterations=cfg.num_iterations,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        beta=cfg.beta,
        epsilon=cfg.epsilon,
        loss_type=cfg.loss_type,
        scale_rewards=cfg.scale_rewards,
        importance_sampling_level=cfg.importance_sampling_level,
        logging_steps=cfg.logging_steps,
        report_to=report_to,
        log_completions=True,
    )
    if cfg.epsilon_high is not None:
        kwargs["epsilon_high"] = cfg.epsilon_high  # DAPO clip-higher
    if cfg.wandb_name:
        kwargs["run_name"] = cfg.wandb_name
    if cfg.save_steps and cfg.save_steps > 0:
        kwargs["save_steps"] = cfg.save_steps
    else:
        kwargs["save_strategy"] = "no"
    kwargs.update(cfg.extra_grpo)  # escape hatch for forward-compat knobs

    return _construct_filtering_unknown(GRPOConfig, kwargs)


def build_lora_config(cfg: RunConfig) -> Any | None:
    """Construct a ``peft.LoraConfig`` (or ``None`` if ``--no-lora``)."""
    if not cfg.lora:
        return None
    from peft import LoraConfig

    return LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        task_type="CAUSAL_LM",
    )


def _construct_filtering_unknown(cls: Any, kwargs: dict[str, Any]) -> Any:
    """Instantiate ``cls(**kwargs)``, dropping keys the class does not accept.

    TRL's ``GRPOConfig`` signature drifts across versions; rather than pin a
    version we filter to the fields the installed ``GRPOConfig`` actually
    declares (it is a ``@dataclass``), warning on anything dropped.
    """
    import dataclasses

    valid = {f.name for f in dataclasses.fields(cls)} if dataclasses.is_dataclass(cls) else None
    if valid is None:
        return cls(**kwargs)
    accepted = {k: v for k, v in kwargs.items() if k in valid}
    dropped = sorted(set(kwargs) - set(accepted))
    if dropped:
        import warnings

        warnings.warn(
            f"{cls.__name__} does not accept {dropped}; dropping "
            "(TRL version skew). Update training.run if a knob is missing.",
            stacklevel=2,
        )
    return cls(**accepted)


# ---------------------------------------------------------------------------
# Summary writing (torch-free)
# ---------------------------------------------------------------------------
def _metrics_from_log_history(log_history: list[dict[str, Any]]) -> dict[str, Any]:
    """Reduce a HF ``Trainer.state.log_history`` to final numeric metrics.

    Keeps the last logged value of each numeric key (reward, kl, entropy,
    completion length, grad norm, loss, ...). Robust to missing keys.
    """
    final: dict[str, Any] = {}
    for entry in log_history:
        for k, v in entry.items():
            if isinstance(v, bool):
                final[k] = float(v)
            elif isinstance(v, (int, float)):
                final[k] = float(v)
    return final


def write_summary(cfg: RunConfig, log_history: list[dict[str, Any]]) -> str:
    """Write ``<output_dir>/summary.json`` and the raw step log; return its path.

    The summary records the resolved config, the final metrics and the full
    per-step log so :func:`training.seeds.summarize_runs` /
    :func:`training.seeds.launch_seeds` can aggregate across seeds.
    """
    os.makedirs(cfg.output_dir, exist_ok=True)
    final_metrics = _metrics_from_log_history(log_history)

    # A JSONL step log (one object per line) for seeds.summarize_runs().
    metrics_path = os.path.join(cfg.output_dir, "metrics.jsonl")
    with open(metrics_path, "w", encoding="utf-8") as fh:
        for entry in log_history:
            fh.write(json.dumps(entry) + "\n")

    summary = {
        "env": cfg.env,
        "model": cfg.resolved_model(),
        "seed": cfg.seed,
        "config": {k: getattr(cfg, k) for k in sorted(_FIELD_NAMES)},
        "final_metrics": final_metrics,
        "n_log_entries": len(log_history),
        "metrics_jsonl": metrics_path,
    }
    summary_path = os.path.join(cfg.output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    return summary_path


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
def train(cfg: RunConfig) -> dict[str, Any]:
    """Run one GRPO training job described by ``cfg``; return its summary dict.

    Heavy deps (``transformers``/``trl``/``torch``/``peft``) are imported here,
    not at module load. Seeds everything via ``transformers.set_seed`` for
    reproducibility, builds the dataset + rewards + ``GRPOTrainer``, trains, then
    writes a per-run JSON summary (consumed by :mod:`training.seeds`).
    """
    import transformers
    from trl import GRPOTrainer

    transformers.set_seed(cfg.seed)

    dataset = build_dataset(cfg)
    reward_funcs = build_reward_funcs(cfg)
    grpo_config = build_config(cfg)
    peft_config = build_lora_config(cfg)

    trainer = GRPOTrainer(
        model=cfg.resolved_model(),
        reward_funcs=reward_funcs,
        args=grpo_config,
        train_dataset=dataset,
        peft_config=peft_config,
    )
    trainer.train()

    log_history = list(getattr(trainer.state, "log_history", []) or [])
    summary_path = write_summary(cfg, log_history)
    print(f"[training.run] wrote summary -> {summary_path}")
    return {
        "summary_path": summary_path,
        "final_metrics": _metrics_from_log_history(log_history),
    }


def main(argv: list[str] | None = None) -> dict[str, Any]:
    """Entrypoint: parse args -> :class:`RunConfig` -> :func:`train`."""
    cfg = parse_args(argv)
    return train(cfg)


if __name__ == "__main__":
    main()
