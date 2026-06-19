"""Tests for ``infra_synth.parser.extract_dockerfile`` (vf-free, stdlib-only)."""
from __future__ import annotations

from infra_synth import parser as infra_parser


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


# --- extract_compose -------------------------------------------------------
def test_compose_single_yaml_block() -> None:
    text = "Here:\n```yaml\nservices:\n  web:\n    build: .\n```\n"
    assert infra_parser.extract_compose(text) == "services:\n  web:\n    build: ."


def test_compose_yml_tag() -> None:
    text = "```yml\nservices:\n  web:\n    image: nginx:1.27\n```"
    assert infra_parser.extract_compose(text) == "services:\n  web:\n    image: nginx:1.27"


def test_compose_takes_last_yaml_block() -> None:
    text = (
        "Draft:\n```yaml\nservices:\n  bad: {}\n```\n"
        "Final:\n```yaml\nservices:\n  web:\n    build: .\n```\n"
    )
    assert infra_parser.extract_compose(text) == "services:\n  web:\n    build: ."


def test_compose_tolerates_prose() -> None:
    text = (
        "Let me think about the services and ports...\n\n"
        "```yaml\nservices:\n  web:\n    build: .\n    ports:\n      - \"8000:8000\"\n```\n\n"
        "Done."
    )
    out = infra_parser.extract_compose(text)
    assert out.startswith("services:")
    assert "Let me think" not in out
    assert '"8000:8000"' in out


def test_compose_no_block_returns_empty() -> None:
    assert infra_parser.extract_compose("no code here at all") == ""
    assert infra_parser.extract_compose("") == ""


# --- extract_ci_yaml -------------------------------------------------------
def test_ci_yaml_single_yaml_block() -> None:
    text = "Here:\n```yaml\non:\n  push:\njobs:\n  test:\n```\n"
    assert infra_parser.extract_ci_yaml(text) == "on:\n  push:\njobs:\n  test:"


def test_ci_yaml_yml_tag() -> None:
    text = "```yml\non: [push]\njobs:\n  test:\n    runs-on: ubuntu-latest\n```"
    out = infra_parser.extract_ci_yaml(text)
    assert out.startswith("on: [push]")
    assert "runs-on: ubuntu-latest" in out


def test_ci_yaml_takes_last_yaml_block() -> None:
    text = (
        "Draft:\n```yaml\non: [push]\njobs: {}\n```\n"
        "Final:\n```yaml\non:\n  push:\njobs:\n  test:\n    runs-on: ubuntu-latest\n```\n"
    )
    out = infra_parser.extract_ci_yaml(text)
    assert out.startswith("on:\n  push:")
    assert "runs-on: ubuntu-latest" in out


def test_ci_yaml_tolerates_prose() -> None:
    text = (
        "Let me think about the triggers and jobs...\n\n"
        "```yaml\nname: ci\non:\n  push:\njobs:\n  test:\n    runs-on: ubuntu-latest\n"
        "    steps:\n      - uses: actions/checkout@v4\n```\n\n"
        "Done."
    )
    out = infra_parser.extract_ci_yaml(text)
    assert out.startswith("name: ci")
    assert "Let me think" not in out
    assert "actions/checkout@v4" in out


def test_ci_yaml_no_block_returns_empty() -> None:
    assert infra_parser.extract_ci_yaml("no code here at all") == ""
    assert infra_parser.extract_ci_yaml("") == ""


# --- extract_terraform -----------------------------------------------------
def test_terraform_single_hcl_block() -> None:
    text = "Here:\n```hcl\nprovider \"docker\" {}\n```\n"
    assert infra_parser.extract_terraform(text) == 'provider "docker" {}'


def test_terraform_tf_tag() -> None:
    text = '```tf\nresource "docker_container" "web" {}\n```'
    assert infra_parser.extract_terraform(text) == 'resource "docker_container" "web" {}'


def test_terraform_terraform_tag() -> None:
    text = "```terraform\nterraform {\n}\n```"
    assert infra_parser.extract_terraform(text) == "terraform {\n}"


def test_terraform_takes_last_hcl_block() -> None:
    text = (
        'Draft:\n```hcl\nprovider "aws" {}\n```\n'
        'Final:\n```hcl\nprovider "docker" {}\n```\n'
    )
    assert infra_parser.extract_terraform(text) == 'provider "docker" {}'


def test_terraform_tolerates_prose() -> None:
    text = (
        "Let me think about the provider and resources...\n\n"
        '```hcl\nterraform {}\nprovider "docker" {}\n'
        'resource "docker_container" "web" {\n  ports {\n'
        "    internal = 8000\n  }\n}\n```\n\n"
        "Done."
    )
    out = infra_parser.extract_terraform(text)
    assert out.startswith("terraform {}")
    assert "Let me think" not in out
    assert "internal = 8000" in out


def test_terraform_no_block_returns_empty() -> None:
    assert infra_parser.extract_terraform("no code here at all") == ""
    assert infra_parser.extract_terraform("") == ""


# --- extract_k8s -----------------------------------------------------------
def test_k8s_single_yaml_block() -> None:
    text = "Here:\n```yaml\napiVersion: apps/v1\nkind: Deployment\n```\n"
    assert infra_parser.extract_k8s(text) == "apiVersion: apps/v1\nkind: Deployment"


def test_k8s_yml_tag() -> None:
    text = "```yml\napiVersion: v1\nkind: Service\n```"
    assert infra_parser.extract_k8s(text) == "apiVersion: v1\nkind: Service"


def test_k8s_takes_last_yaml_block() -> None:
    text = (
        "Draft:\n```yaml\nkind: Pod\n```\n"
        "Final:\n```yaml\napiVersion: apps/v1\nkind: Deployment\n```\n"
    )
    assert infra_parser.extract_k8s(text) == "apiVersion: apps/v1\nkind: Deployment"


def test_k8s_tolerates_prose() -> None:
    text = (
        "Let me think about the deployment and service...\n\n"
        "```yaml\napiVersion: apps/v1\nkind: Deployment\nspec:\n"
        "  template:\n    spec:\n      containers:\n"
        "        - containerPort: 8000\n```\n\n"
        "Done."
    )
    out = infra_parser.extract_k8s(text)
    assert out.startswith("apiVersion: apps/v1")
    assert "Let me think" not in out
    assert "containerPort: 8000" in out


def test_k8s_no_block_returns_empty() -> None:
    assert infra_parser.extract_k8s("no code here at all") == ""
    assert infra_parser.extract_k8s("") == ""
