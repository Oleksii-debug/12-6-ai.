"""Fail-closed policy for GitHub Actions queue growth.

The repository uses one shared automatic pull-request CI workflow. New task-specific
workflows must be explicitly invoked instead of creating another automatic queue for
every pull request. Existing automatic workflows may be maintained, but any modified
automatic workflow must retain cancellation semantics.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

AUTOMATIC_TRIGGERS = frozenset({"push", "pull_request", "pull_request_target"})
EXPLICIT_TRIGGERS = frozenset({"workflow_call", "workflow_dispatch"})
SHARED_AUTOMATIC_WORKFLOWS = frozenset({"ci.yml", "ci.yaml"})
_WORKFLOW_SUFFIXES = frozenset({".yml", ".yaml"})


class WorkflowQueuePolicyError(RuntimeError):
    """Raised when a changed workflow would increase uncontrolled Actions pressure."""


@dataclass(frozen=True)
class ChangedWorkflow:
    """One added or modified GitHub Actions workflow."""

    status: str
    path: str


def _block_triggers(text: str) -> set[str]:
    triggers: set[str] = set()
    lines = text.splitlines()
    in_on_block = False
    for line in lines:
        if re.fullmatch(r"on:\s*", line):
            in_on_block = True
            continue
        if in_on_block:
            if line and not line.startswith((" ", "\t")):
                break
            match = re.match(r"^\s{2}([A-Za-z_][A-Za-z0-9_-]*):", line)
            if match:
                triggers.add(match.group(1))
    return triggers


def _inline_triggers(text: str) -> set[str]:
    match = re.search(r"(?m)^on:\s*\[([^\]]*)\]\s*$", text)
    if not match:
        return set()
    return {
        item.strip().strip("'\"")
        for item in match.group(1).split(",")
        if item.strip()
    }


def workflow_triggers(text: str) -> frozenset[str]:
    """Return top-level trigger names for common block and inline-list YAML forms."""

    return frozenset(_block_triggers(text) | _inline_triggers(text))


def validate_workflow_text(path: str, status: str, text: str) -> tuple[str, ...]:
    """Return deterministic policy violations for one changed workflow."""

    candidate = Path(path)
    if candidate.suffix.lower() not in _WORKFLOW_SUFFIXES:
        return ()

    normalized_status = status.upper().strip()
    triggers = workflow_triggers(text)
    automatic = sorted(triggers & AUTOMATIC_TRIGGERS)
    explicit = sorted(triggers & EXPLICIT_TRIGGERS)
    violations: list[str] = []

    if normalized_status == "A" and candidate.name not in SHARED_AUTOMATIC_WORKFLOWS:
        if automatic:
            violations.append(
                f"{path}: new dedicated workflow may not auto-trigger on "
                f"{','.join(automatic)}; reuse ci.yml or use workflow_dispatch/workflow_call"
            )
        if not explicit:
            violations.append(
                f"{path}: new dedicated workflow requires workflow_dispatch or workflow_call"
            )

    if automatic:
        if not re.search(r"(?m)^concurrency:\s*$", text):
            violations.append(f"{path}: automatic workflow requires top-level concurrency")
        if not re.search(r"(?m)^\s{2}cancel-in-progress:\s*true\s*$", text):
            violations.append(
                f"{path}: automatic workflow requires concurrency.cancel-in-progress: true"
            )

    if normalized_status == "A" and "runs-on:" in text and "timeout-minutes:" not in text:
        violations.append(f"{path}: new runnable workflow requires timeout-minutes")

    return tuple(sorted(violations))


def changed_workflows(repo_root: Path, base_sha: str, head_sha: str) -> tuple[ChangedWorkflow, ...]:
    """Resolve added/modified workflow paths from an exact Git comparison."""

    result = subprocess.run(
        [
            "git",
            "diff",
            "--name-status",
            "--diff-filter=AM",
            base_sha,
            head_sha,
            "--",
            ".github/workflows",
        ],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or f"exit {result.returncode}"
        raise WorkflowQueuePolicyError(f"cannot resolve changed workflows: {detail}")

    items: list[ChangedWorkflow] = []
    for raw_line in result.stdout.splitlines():
        if not raw_line.strip():
            continue
        parts = raw_line.split("\t", maxsplit=1)
        if len(parts) != 2 or parts[0] not in {"A", "M"}:
            raise WorkflowQueuePolicyError(f"unexpected git diff record: {raw_line}")
        items.append(ChangedWorkflow(status=parts[0], path=parts[1]))
    return tuple(items)


def validate_changed_workflows(repo_root: Path, base_sha: str, head_sha: str) -> int:
    """Validate all added/modified workflows and return the checked file count."""

    root = repo_root.resolve()
    changed = changed_workflows(root, base_sha, head_sha)
    violations: list[str] = []
    for item in changed:
        path = root / item.path
        if not path.is_file():
            violations.append(f"{item.path}: changed workflow is missing from checkout")
            continue
        text = path.read_text(encoding="utf-8")
        violations.extend(validate_workflow_text(item.path, item.status, text))

    if violations:
        raise WorkflowQueuePolicyError("\n".join(sorted(set(violations))))
    return len(changed)
