from __future__ import annotations

from pathlib import Path

import pytest

from twelve_six.integration.workflow_policy import (
    WorkflowPolicyError,
    validate_repository_workflows,
    validate_workflow_files,
    validate_workflow_text,
)

ROOT = Path(__file__).resolve().parents[1]
PIN = "1" * 40
DIGEST = "2" * 64


def _authoritative_ci() -> str:
    return f"""name: CI
on:
  pull_request:
permissions:
  contents: read
jobs:
  authority:
    runs-on: ubuntu-24.04
    timeout-minutes: 20
    steps:
      - uses: actions/checkout@{PIN}
        with:
          fetch-depth: 0
          persist-credentials: false
      - name: Secret history
        env:
          GITLEAKS_ARCHIVE_SHA256: {DIGEST}
        run: |
          sha256sum --check checksum.txt
          gitleaks git --log-opts="--full-history --all --diff-filter=tuxdb" .
      - name: Secret-gate negative fixture
        run: echo fixture
      - uses: actions/setup-python@{PIN}
        with:
          python-version: "3.11.16"
          cache: pip
          cache-dependency-path: |
            requirements/locks/linux-x86_64/runtime.lock.txt
"""


def _fast_ci() -> str:
    return f"""name: Fast CI
on:
  pull_request:
permissions:
  contents: read
concurrency:
  group: ${{{{ github.workflow }}}}-${{{{ github.event.pull_request.number }}}}
  cancel-in-progress: true
jobs:
  fast:
    runs-on: ubuntu-24.04
    timeout-minutes: 10
    steps:
      - uses: actions/checkout@{PIN}
        with:
          fetch-depth: 1
          persist-credentials: false
      - uses: actions/setup-python@{PIN}
        with:
          python-version: "3.11.16"
"""


def _messages(text: str) -> str:
    return "\n".join(item.message for item in validate_workflow_text("workflow.yml", text))


def test_authoritative_and_fast_fixtures_pass() -> None:
    assert validate_workflow_text("ci.yml", _authoritative_ci()) == ()
    assert validate_workflow_text("fast-ci.yml", _fast_ci()) == ()


def test_committed_repository_workflows_pass() -> None:
    validate_repository_workflows(ROOT)


def test_mutable_action_ref_is_rejected() -> None:
    text = _fast_ci().replace(f"actions/checkout@{PIN}", "actions/checkout@v7", 1)
    assert "immutable 40-hex" in _messages(text)


def test_docker_tag_is_rejected() -> None:
    text = _fast_ci().replace(
        f"- uses: actions/setup-python@{PIN}", "- uses: docker://example/tool:latest"
    )
    assert "sha256 digest" in _messages(text)


def test_checkout_must_disable_persisted_credentials() -> None:
    text = _fast_ci().replace("          persist-credentials: false\n", "", 1)
    assert "persist-credentials" in _messages(text)


def test_setup_python_must_be_exact() -> None:
    text = _fast_ci().replace('python-version: "3.11.16"', 'python-version: "3.11"')
    assert "exact X.Y.Z" in _messages(text)


def test_write_permission_is_rejected() -> None:
    text = _fast_ci().replace("contents: read", "contents: write")
    assert "read or none" in _messages(text)


def test_every_job_needs_bounded_timeout() -> None:
    text = _fast_ci().replace("    timeout-minutes: 10\n", "")
    assert "every job" in _messages(text)


def test_authoritative_ci_requires_explicit_full_history_raw_scan() -> None:
    text = _authoritative_ci().replace("--full-history --all", "--first-parent")
    assert "all full history" in _messages(text)


def test_authoritative_ci_cannot_be_auto_canceled() -> None:
    text = _authoritative_ci().replace(
        "permissions:\n",
        "concurrency:\n  group: authority\n  cancel-in-progress: true\npermissions:\n",
    )
    messages = _messages(text)
    assert "allowed only" in messages
    assert "must never be auto-canceled" in messages


def test_fast_cancellation_must_be_pr_scoped() -> None:
    text = _fast_ci().replace("github.event.pull_request.number", "github.ref")
    assert "pull_request.number" in _messages(text)


def test_cache_must_be_lock_keyed() -> None:
    text = _authoritative_ci().replace(
        "requirements/locks/linux-x86_64/runtime.lock.txt", "pyproject.toml"
    )
    assert "committed requirements/locks" in _messages(text)


def test_floating_pip_upgrade_is_rejected() -> None:
    text = _fast_ci().replace(
        f"      - uses: actions/setup-python@{PIN}",
        "      - run: python -m pip install --upgrade pip\n"
        f"      - uses: actions/setup-python@{PIN}",
    )
    assert "must not float pip" in _messages(text)


def test_violations_raise_deterministically(tmp_path: Path) -> None:
    workflow = tmp_path / "bad.yml"
    workflow.write_text(_fast_ci().replace("contents: read", "contents: write"), encoding="utf-8")
    with pytest.raises(WorkflowPolicyError, match="permission"):
        validate_workflow_files((workflow,), tmp_path)
