"""Tests for ``infra_synth.parser.extract_dockerfile`` (vf-free, stdlib-only)."""
from __future__ import annotations

import os
import sys

# The env package lives under environments/infra_synth and is not importable as
# a top-level package; put its directory on sys.path so we can import the
# vf-free helper modules directly (no `verifiers` / `datasets` needed).
_PKG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "environments",
    "infra_synth",
)
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)

import parser as infra_parser  # noqa: E402


def test_single_dockerfile_block() -> None:
    text = "Here you go:\n```dockerfile\nFROM python:3.11-slim\nCMD [\"x\"]\n```\n"
    assert infra_parser.extract_dockerfile(text) == 'FROM python:3.11-slim\nCMD ["x"]'


def test_capital_dockerfile_tag() -> None:
    text = "```Dockerfile\nFROM alpine:3.19\n```"
    assert infra_parser.extract_dockerfile(text) == "FROM alpine:3.19"


def test_bare_block() -> None:
    text = "```\nFROM busybox:1.36\nEXPOSE 8080\n```"
    assert infra_parser.extract_dockerfile(text) == "FROM busybox:1.36\nEXPOSE 8080"


def test_multiple_blocks_takes_last() -> None:
    text = (
        "First a bad draft:\n```dockerfile\nFROM scratch\n```\n"
        "Actually here is the final answer:\n"
        "```dockerfile\nFROM python:3.12-slim\nEXPOSE 8000\n```\n"
    )
    assert infra_parser.extract_dockerfile(text) == "FROM python:3.12-slim\nEXPOSE 8000"


def test_prefers_tagged_block_over_later_bare_unrelated() -> None:
    # A dockerfile-tagged block should win even if a later bare block exists.
    text = (
        "```dockerfile\nFROM python:3.11-slim\nCMD [\"run\"]\n```\n"
        "```\nsome unrelated shell output\n```\n"
    )
    assert infra_parser.extract_dockerfile(text) == 'FROM python:3.11-slim\nCMD ["run"]'


def test_prose_wrapped_with_leading_thinking() -> None:
    text = (
        "Let me think about the base image and dependencies...\n\n"
        "I'll use a slim Python base.\n\n"
        "```dockerfile\nFROM python:3.11-slim\nWORKDIR /app\nEXPOSE 5000\n```\n\n"
        "That satisfies the spec."
    )
    out = infra_parser.extract_dockerfile(text)
    assert out.startswith("FROM python:3.11-slim")
    assert "EXPOSE 5000" in out
    assert "Let me think" not in out


def test_no_block_returns_empty() -> None:
    assert infra_parser.extract_dockerfile("no code here at all") == ""
    assert infra_parser.extract_dockerfile("") == ""


def test_whitespace_stripped() -> None:
    text = "```dockerfile\n\n   FROM python:3.11-slim   \n\n```"
    assert infra_parser.extract_dockerfile(text) == "FROM python:3.11-slim"


def test_extract_fenced_language_filter() -> None:
    # The generic helper can target other languages.
    text = "```yaml\nversion: '3'\n```\n```python\nprint(1)\n```"
    assert infra_parser.extract_fenced(text, langs=("yaml",)) == "version: '3'"
    assert infra_parser.extract_fenced(text, langs=("python",)) == "print(1)"


def test_tilde_fences_supported() -> None:
    text = "~~~dockerfile\nFROM alpine:3.19\n~~~"
    assert infra_parser.extract_dockerfile(text) == "FROM alpine:3.19"
