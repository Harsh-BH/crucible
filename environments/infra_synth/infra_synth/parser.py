"""Artifact extraction for the ``infra_synth`` environment.

vf-free, **stdlib-only** so it unit-tests without ``verifiers`` / ``datasets``.
The single public entrypoint used by the environment is
:func:`extract_dockerfile`; :func:`extract_fenced` is the generic helper it
delegates to (kept public for future artifact kinds such as compose / terraform).

Design notes
------------
- We pull the **LAST** fenced code block. Models often reason in prose and emit
  the final artifact last; taking the last block is the most robust heuristic
  and matches the system prompt ("output ONLY a Dockerfile in a single block").
- A block whose info-string names a Dockerfile dialect (``dockerfile`` /
  ``Dockerfile`` / ``docker``) is preferred. If none is tagged that way we fall
  back to the last bare ```` ``` ```` block. This tolerates models that forget
  the language tag.
- Returns ``""`` when nothing usable is found (the reward layer treats empty
  artifacts as a build failure / zero format reward).
"""
from __future__ import annotations

import re

# A fenced code block: opening fence (>=3 backticks OR tildes), optional
# info-string on the same line, body, then a closing fence of the same kind.
# We capture the info-string (group "lang") and the body (group "body").
#
# ``re.DOTALL`` lets ``.`` span newlines for the body; ``re.MULTILINE`` anchors
# the fences to line starts. Non-greedy body so adjacent blocks don't merge.
_FENCE_RE = re.compile(
    r"^[ \t]*(?P<fence>`{3,}|~{3,})[ \t]*(?P<lang>[^\n`~]*)\n"
    r"(?P<body>.*?)"
    r"^[ \t]*(?P=fence)[ \t]*$",
    re.DOTALL | re.MULTILINE,
)

# Default info-strings that mark a Dockerfile block (compared case-insensitively).
_DOCKERFILE_LANGS: tuple[str, ...] = ("dockerfile", "docker")


def _iter_blocks(text: str) -> list[tuple[str, str]]:
    """Return ``[(lang, body), ...]`` for every fenced block, in document order.

    ``lang`` is the lower-cased first token of the info-string (e.g. ``dockerfile``
    from ``dockerfile linenums="1"``); ``body`` is the raw (un-stripped) block body.
    """
    blocks: list[tuple[str, str]] = []
    for m in _FENCE_RE.finditer(text):
        info = (m.group("lang") or "").strip()
        # The info-string may carry attributes after the language token.
        lang = info.split()[0].lower() if info else ""
        blocks.append((lang, m.group("body")))
    return blocks


def extract_fenced(text: str, langs: tuple[str, ...] = _DOCKERFILE_LANGS) -> str:
    """Extract the body of the most relevant fenced code block.

    Preference order:

    1. The **last** block whose language tag is in ``langs``.
    2. Otherwise the **last** block with *any* (or no) language tag.

    Returns the stripped block body, or ``""`` if there are no fenced blocks.
    """
    if not text:
        return ""
    blocks = _iter_blocks(text)
    if not blocks:
        return ""

    wanted = {lang.lower() for lang in langs}
    # Prefer the last language-tagged block.
    for lang, body in reversed(blocks):
        if lang in wanted:
            return body.strip()
    # Fall back to the last block of any kind.
    return blocks[-1][1].strip()


def extract_dockerfile(text: str) -> str:
    """Pull the Dockerfile artifact out of a model completion.

    Prefers ```` ```dockerfile ```` / ```` ```Dockerfile ```` fenced blocks but
    accepts a bare ```` ``` ```` block; takes the **last** matching block and
    returns its stripped contents (``""`` if none). Robust to surrounding prose
    and to multiple code blocks in the same completion.
    """
    return extract_fenced(text, _DOCKERFILE_LANGS)


__all__ = ["extract_dockerfile", "extract_fenced"]
