from __future__ import annotations

from pathlib import Path

import pytest

from twelve_six.integration.workflow_policy import (
    WorkflowPolicyError,
    validate_repository_workflows,
    validate_workflow_files,
    validate_workflow_text,
)

PIN = "0123456789abcdef0123456789abcdef01234567"
DIGEST = "a" * 64


def _messages(text: str) -> list[str]:
    return [item.message for item in validate_workflow_text("workflow.yml", text)]


def test_pinned_actions_exact_python_and_non_persistent_checkout_pass() -> None:
    workflow = f"""
steps:
  - uses: actions/checkout@{PIN}
    with:
      persist-credentials: false
  - uses: actions/setup-python@{PIN}
    with:
      python-version: "3.11.16"
  - uses: owner/reusable/.github/workflows/test.yml@{PIN}
  - uses: ./local-action
  - uses: docker://example.invalid/tool@sha256:{DIGEST}
  - run: python -m pip install --disable-pip-version-check -e .[dev]
"""
    assert validate_workflow_text("workflow.yml", workflow) == ()


@pytest.mark.parametrize(
    ("uses", "expected"),
    [
        ("actions/checkout@v7", "immutable 40-hex"),
        ("owner/action@main", "immutable 40-hex"),
        ("owner/reusable/.github/workflows/test.yml@release", "immutable 40-hex"),
        ("docker://example.invalid/tool:latest", "sha256 digest"),
    ],
)
def test_mutable_external_inputs_fail(uses: str, expected: str) -> None:
    messages = _messages(f"steps:\n  - uses: {uses}\n")
    assert any(expected in message for message in messages)


def test_checkout_requires_persist_credentials_false() -> None:
    messages = _messages(f"steps:\n  - uses: actions/checkout@{PIN}\n")
    assert "actions/checkout must set persist-credentials: false" in messages


def test_setup_python_requires_exact_patch_version() -> None:
    workflow = f"""
steps:
  - uses: actions/setup-python@{PIN}
    with:
      python-version: "3.11"
"""
    messages = _messages(workflow)
    assert "actions/setup-python must select one exact X.Y.Z python-version" in messages


@pytest.mark.parametrize(
    "command",
    [
        "python -m pip install --upgrade pip",
        "pip install --upgrade pip",
        "python -m pip install -U pip",
    ],
)
def test_floating_pip_self_upgrade_fails(command: str) -> None:
    messages = _messages(f"steps:\n  - run: {command}\n")
    assert "workflow must not float pip via pip install --upgrade/-U pip" in messages


def test_repository_workflow_policy_passes_current_checkout() -> None:
    validate_repository_workflows(Path(__file__).resolve().parents[1])


def test_workflow_outside_repository_root_fails(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    outside = tmp_path / "outside.yml"
    outside.write_text("steps: []\n", encoding="utf-8")

    with pytest.raises(WorkflowPolicyError, match="outside repository root"):
        validate_workflow_files((outside,), repo)
