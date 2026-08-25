"""Validate every repository workflow against CI supply-chain policy."""

from __future__ import annotations

from pathlib import Path

from _integration_bootstrap import load_integration_module

ROOT = Path(__file__).resolve().parents[1]
_WORKFLOW_POLICY = load_integration_module(ROOT, "workflow_policy")


def main() -> int:
    try:
        _WORKFLOW_POLICY.validate_repository_workflows(ROOT)
    except _WORKFLOW_POLICY.WorkflowPolicyError as exc:
        print(str(exc))
        return 1
    print("workflow_policy=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
