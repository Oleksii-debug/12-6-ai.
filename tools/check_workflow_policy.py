"""Validate every repository workflow against CI supply-chain policy."""

from __future__ import annotations

from pathlib import Path

from twelve_six.integration.workflow_policy import WorkflowPolicyError, validate_repository_workflows


def main() -> int:
    try:
        validate_repository_workflows(Path("."))
    except WorkflowPolicyError as exc:
        print(str(exc))
        return 1
    print("workflow_policy=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
