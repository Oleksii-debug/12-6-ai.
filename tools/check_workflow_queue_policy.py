"""Validate changed GitHub Actions workflows against queue-control policy."""

from __future__ import annotations

import argparse
from pathlib import Path

from twelve_six.integration.workflow_queue_policy import (
    WorkflowQueuePolicyError,
    validate_changed_workflows,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-sha", required=True)
    parser.add_argument("--head-sha", required=True)
    parser.add_argument("--repo-root", default=".")
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        checked = validate_changed_workflows(
            Path(args.repo_root),
            base_sha=args.base_sha,
            head_sha=args.head_sha,
        )
    except (OSError, UnicodeError, WorkflowQueuePolicyError) as exc:
        print(str(exc))
        return 1
    print(f"workflow_queue_policy=pass checked={checked}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
