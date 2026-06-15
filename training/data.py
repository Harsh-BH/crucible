"""Dataset builders for Crucible GRPO training (torch-free).

This module turns the project's tasks into Hugging Face ``datasets.Dataset``
objects shaped for **TRL's** ``GRPOTrainer`` (and re-usable by the prime-rl /
``verifiers`` path). Two environments are supported:

- :func:`build_gsm8k` — the standard GSM8K math benchmark, used for the **M1
  reproduction** (a well-understood task where GRPO reward is known to rise).
- :func:`build_infra_synth` — the project's own ``infra_synth`` IaC-synthesis
  environment, used for **M2** (reward routed through the verifier / Sentinel).

Import discipline
-----------------
- ``datasets`` is imported **lazily** inside the builder functions so the pure
  string helper :func:`extract_boxed_or_final_number` (and this module's import)
  works with no heavy deps installed — its unit tests need only stdlib.
- The ``infra_synth`` env package is imported lazily inside
  :func:`build_infra_synth`; it is installed via ``pip install -e
  ./environments/infra_synth`` and pulls in only the stdlib-light
  ``verifier.types``.

Dataset column contract (consumed by ``training.rewards`` + ``training.run``)
----------------------------------------------------------------------------
- GSM8K rows: ``prompt`` (a chat-style list of messages OR a plain string,
  selectable), ``answer`` (the gold final number as a string), and ``question``
  (the raw problem text).
- infra_synth rows: ``prompt`` (chat or string), ``question`` (the NL spec),
  ``answer`` (the gold base-image hint), ``info`` (the pipeline source-of-truth
  dict used to build a ``VerifySpec``) and ``task``.

TRL passes every non-``prompt``/``completion`` dataset column through to the
reward functions as a keyword arg (a list, one entry per sample), which is how
``answer`` / ``info`` reach ``training.rewards``.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # only for type hints; no runtime import
    from datasets import Dataset

__all__ = [
    "extract_boxed_or_final_number",
    "build_gsm8k",
    "build_infra_synth",
    "GSM8K_SYSTEM_PROMPT",
]


def _import_infra_tasks() -> Any:
    """Lazily import the ``infra_synth`` task module.

    The env ships as a proper nested package (installed via ``pip install -e
    ./environments/infra_synth`` or ``prime env install``), so its task module
    imports cleanly as ``infra_synth.tasks``.
    """
    from infra_synth import tasks as infra_tasks

    return infra_tasks

# A short instruction nudging the model to end with a boxed / explicit final
# answer so the extractor (and the reward) is well-defined.
GSM8K_SYSTEM_PROMPT = (
    "You are a careful math tutor. Solve the problem step by step, then give the "
    "final numeric answer on its own line in the form \\boxed{<number>}."
)


# ---------------------------------------------------------------------------
# Answer extraction (pure, stdlib-only — torch-free AND datasets-free)
# ---------------------------------------------------------------------------
# A signed number, optionally with thousands separators and a decimal part.
# We deliberately keep this permissive then normalise in the reward layer.
_NUMBER_RE = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?")

# GSM8K gold answers end with a "#### <number>" marker.
_GSM8K_GOLD_RE = re.compile(r"####\s*(.+?)\s*$", re.MULTILINE)


def _last_number(text: str) -> str | None:
    """Return the LAST number-looking token in ``text`` (stripped of $ and ,)."""
    matches = _NUMBER_RE.findall(text or "")
    if not matches:
        return None
    raw = matches[-1]
    return raw.replace("$", "").replace(",", "").strip()


def _extract_boxed(text: str) -> str | None:
    r"""Return the contents of the LAST ``\boxed{...}`` in ``text``, else ``None``.

    Handles nested braces inside the box (e.g. ``\boxed{\frac{1}{2}}``) by
    scanning for the matching closing brace rather than using a naive regex.
    """
    if not text:
        return None
    idx = text.rfind(r"\boxed{")
    if idx == -1:
        return None
    start = idx + len(r"\boxed{")
    depth = 1
    i = start
    while i < len(text):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i].strip()
        i += 1
    return None  # unbalanced braces


def extract_boxed_or_final_number(text: str) -> str | None:
    r"""Extract a final answer from a model completion (or gold string).

    Resolution order:

    1. The contents of the last ``\boxed{...}`` (LaTeX answer convention). If the
       box itself contains a number we return that number (normalised); otherwise
       we return the raw box contents (e.g. a symbolic expression).
    2. The ``#### <answer>`` GSM8K gold marker.
    3. The last number-looking token anywhere in the text.

    Returns the normalised string (``$``/``,`` stripped) or ``None`` if nothing
    answer-like is present. Pure/stdlib — safe to unit-test without any deps.
    """
    if not text:
        return None

    boxed = _extract_boxed(text)
    if boxed is not None:
        # If the box is JUST a number (e.g. \boxed{42} or \boxed{1,000}), return
        # it normalised; for a symbolic box (e.g. \boxed{\frac{1}{2}}) return the
        # raw contents so a symbolic string compare can still match.
        stripped = boxed.strip()
        m = _NUMBER_RE.fullmatch(stripped)
        if m:
            return stripped.replace("$", "").replace(",", "")
        return stripped

    gold = _GSM8K_GOLD_RE.search(text)
    if gold:
        marked = gold.group(1).strip()
        num = _last_number(marked)
        return num if num is not None else marked

    return _last_number(text)


# ---------------------------------------------------------------------------
# Prompt shaping helpers
# ---------------------------------------------------------------------------
def _as_prompt(
    user_text: str,
    system_text: str | None,
    *,
    chat: bool,
) -> Any:
    """Return either a chat-style message list or a plain string prompt.

    TRL's ``GRPOTrainer`` accepts both a ``"prompt"`` string column and a
    conversational list-of-messages column (it applies the chat template
    itself). We default to chat for instruct models.
    """
    if chat:
        messages: list[dict[str, str]] = []
        if system_text:
            messages.append({"role": "system", "content": system_text})
        messages.append({"role": "user", "content": user_text})
        return messages
    if system_text:
        return f"{system_text}\n\n{user_text}"
    return user_text


# ---------------------------------------------------------------------------
# GSM8K
# ---------------------------------------------------------------------------
def build_gsm8k(
    split: str = "train",
    n: int | None = None,
    seed: int = 0,
    *,
    chat: bool = True,
    dataset_name: str = "openai/gsm8k",
    dataset_subset: str = "main",
) -> "Dataset":
    """Build a GSM8K dataset with ``prompt`` + ``answer`` (+ ``question``).

    Parameters
    ----------
    split:
        ``"train"`` or ``"test"`` (mapped to the HF GSM8K splits).
    n:
        Optional cap on the number of examples (after a seeded shuffle), for fast
        smoke runs. ``None`` -> the whole split.
    seed:
        Shuffle seed (reproducible subsetting).
    chat:
        If ``True`` (default) ``prompt`` is a chat message list with
        :data:`GSM8K_SYSTEM_PROMPT`; otherwise a plain string.

    Returns a ``datasets.Dataset`` with columns ``prompt``, ``answer`` (the gold
    final number as a string) and ``question`` (the raw problem). ``datasets`` is
    imported lazily so this module stays torch/datasets-free at import time.
    """
    from datasets import load_dataset  # lazy

    hf_split = "test" if split == "test" else "train"
    ds = load_dataset(dataset_name, dataset_subset, split=hf_split)

    if seed is not None:
        ds = ds.shuffle(seed=seed)
    if n is not None:
        ds = ds.select(range(min(n, len(ds))))

    def _map(example: dict[str, Any]) -> dict[str, Any]:
        question = example["question"]
        gold = extract_boxed_or_final_number(example["answer"])
        return {
            "prompt": _as_prompt(question, GSM8K_SYSTEM_PROMPT, chat=chat),
            "answer": gold if gold is not None else "",
            "question": question,
        }

    # Drop the original columns so the dataset matches our contract exactly.
    return ds.map(_map, remove_columns=ds.column_names)


# ---------------------------------------------------------------------------
# infra_synth
# ---------------------------------------------------------------------------
def build_infra_synth(
    split: str = "train",
    n: int | None = None,
    seed: int = 0,
    *,
    chat: bool = True,
) -> "Dataset":
    """Build an ``infra_synth`` dataset from :func:`infra_synth.tasks.generate_tasks`.

    Columns: ``prompt`` (chat or string, prefixed with the env
    :data:`infra_synth.tasks.SYSTEM_PROMPT`), ``question`` (the NL spec),
    ``answer`` (the gold base-image hint), ``info`` (the pipeline source of truth
    used by :func:`infra_synth.tasks.build_verify_spec`) and ``task``.

    The ``infra_synth`` env package + ``datasets`` are imported lazily so this
    module imports torch/datasets-free.
    """
    from datasets import Dataset  # lazy

    infra_tasks = _import_infra_tasks()  # lazy (installed env package)

    raw = infra_tasks.generate_tasks(n=n, seed=seed, split=split)
    system = infra_tasks.SYSTEM_PROMPT

    rows: dict[str, list[Any]] = {
        "prompt": [],
        "question": [],
        "answer": [],
        "info": [],
        "task": [],
    }
    for t in raw:
        question = t["question"]
        rows["prompt"].append(_as_prompt(question, system, chat=chat))
        rows["question"].append(question)
        rows["answer"].append(t["answer"])
        rows["info"].append(t["info"])
        rows["task"].append(t["task"])

    return Dataset.from_dict(rows)
