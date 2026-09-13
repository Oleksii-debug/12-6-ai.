from pathlib import Path

import pytest

from tools.validate_workflow_hygiene import WorkflowPolicyError, validate_workflow_text


VALID_WORKFLOW = """\
name: Scoped proof
on:
  pull_request:
concurrency:
  group: proof-${{ github.workflow }}-${{ github.event.pull_request.number || github.ref }}
  cancel-in-progress: true
permissions:
  contents: read
jobs:
  verify:
    runs-on: ubuntu-latest
    timeout-minutes: 20
    steps:
      - run: python -V
"""


def test_compliant_workflow_passes() -> None:
    validate_workflow_text(VALID_WORKFLOW)


def test_repository_ci_is_policy_compliant() -> None:
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    validate_workflow_text(workflow, source=".github/workflows/ci.yml")


@pytest.mark.parametrize(
    ("needle", "replacement", "message"),
    [
        ("concurrency:\n", "", "missing top-level concurrency block"),
        (
            "  cancel-in-progress: true\n",
            "  cancel-in-progress: false\n",
            "cancel-in-progress must be true",
        ),
        (
            "  group: proof-${{ github.workflow }}-${{ github.event.pull_request.number || github.ref }}\n",
            "  group: proof-static\n",
            "group must include github.workflow",
        ),
        ("permissions:\n  contents: read\n", "permissions: write-all\n", "write-all is forbidden"),
        ("    timeout-minutes: 20\n", "", "is missing timeout-minutes"),
    ],
)
def test_policy_violations_fail_closed(needle: str, replacement: str, message: str) -> None:
    candidate = VALID_WORKFLOW.replace(needle, replacement)
    with pytest.raises(WorkflowPolicyError, match=message):
        validate_workflow_text(candidate)


def test_each_normal_job_requires_its_own_timeout() -> None:
    candidate = VALID_WORKFLOW + """\
  second:
    runs-on: ubuntu-latest
    steps:
      - run: echo missing-timeout
"""
    with pytest.raises(WorkflowPolicyError, match="second.*missing timeout-minutes"):
        validate_workflow_text(candidate)


def test_reusable_workflow_job_is_exempt_from_caller_timeout() -> None:
    candidate = VALID_WORKFLOW + """\
  delegated:
    uses: owner/repository/.github/workflows/reusable.yml@immutable-ref
"""
    validate_workflow_text(candidate)


def test_malformed_tab_indentation_fails_closed() -> None:
    candidate = VALID_WORKFLOW.replace("  contents: read", "\tcontents: read")
    with pytest.raises(WorkflowPolicyError, match="tab indentation"):
        validate_workflow_text(candidate)


def test_timeout_has_bounded_positive_range() -> None:
    candidate = VALID_WORKFLOW.replace("timeout-minutes: 20", "timeout-minutes: 0")
    with pytest.raises(WorkflowPolicyError, match="between 1 and 360"):
        validate_workflow_text(candidate)
