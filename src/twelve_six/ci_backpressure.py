from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

WORKFLOW_GLOBS = ("*.yml", "*.yaml")


@dataclass(frozen=True)
class WorkflowBackpressure:
    path: str
    pull_request: bool
    paths_scoped: bool
    concurrency: bool
    cancel_in_progress: bool
    pr_scoped_group: bool


def _pull_request_paths_scoped(lines: list[str]) -> bool:
    try:
        start = lines.index("  pull_request:") + 1
    except ValueError:
        return False

    for line in lines[start:]:
        if line and not line.startswith(" "):
            break
        if line.startswith("  ") and not line.startswith("    ") and line.strip().endswith(":"):
            break
        if line == "    paths:":
            return True
    return False


def inspect_workflow(path: Path, *, repo_root: Path | None = None) -> WorkflowBackpressure:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    root = repo_root or path.parents[2]
    relative = path.relative_to(root).as_posix()

    pull_request = "  pull_request:" in lines
    concurrency = "concurrency:" in lines
    cancel_in_progress = "  cancel-in-progress: true" in lines
    group_lines = [line for line in lines if line.startswith("  group:")]
    pr_scoped_group = any(
        "github.event.pull_request.number" in line and "github.ref" in line
        for line in group_lines
    )

    return WorkflowBackpressure(
        path=relative,
        pull_request=pull_request,
        paths_scoped=_pull_request_paths_scoped(lines),
        concurrency=concurrency,
        cancel_in_progress=cancel_in_progress,
        pr_scoped_group=pr_scoped_group,
    )


def _active_workflows(repo_root: Path) -> list[Path]:
    workflow_dir = repo_root / ".github" / "workflows"
    paths: set[Path] = set()
    for pattern in WORKFLOW_GLOBS:
        paths.update(workflow_dir.glob(pattern))
    return sorted(path for path in paths if path.is_file())


def load_inventory(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "12-6.ci-legacy-workflow-backpressure.v1":
        raise ValueError("unsupported CI backpressure inventory schema")
    workflows = payload.get("workflows")
    if not isinstance(workflows, list) or not workflows:
        raise ValueError("inventory workflows must be a non-empty list")
    return payload


def validate_inventory(repo_root: Path, inventory_path: Path) -> list[str]:
    payload = load_inventory(inventory_path)
    declared = payload["workflows"]
    declared_by_path: dict[str, dict[str, Any]] = {}
    violations: list[str] = []

    for item in declared:
        path = item.get("path")
        if not isinstance(path, str) or not path:
            violations.append("inventory entry has invalid path")
            continue
        if path in declared_by_path:
            violations.append(f"duplicate inventory path: {path}")
            continue
        declared_by_path[path] = item

    active_paths = _active_workflows(repo_root)
    active_rel = {path.relative_to(repo_root).as_posix() for path in active_paths}
    declared_rel = set(declared_by_path)

    for missing in sorted(active_rel - declared_rel):
        violations.append(f"active workflow missing from inventory: {missing}")
    for stale in sorted(declared_rel - active_rel):
        violations.append(f"inventory references missing workflow: {stale}")

    for workflow_path in active_paths:
        actual = inspect_workflow(workflow_path, repo_root=repo_root)
        expected = declared_by_path.get(actual.path)
        if expected is None:
            continue

        if actual.pull_request is not bool(expected.get("pull_request")):
            violations.append(f"pull_request trigger drift: {actual.path}")
        if actual.paths_scoped is not bool(expected.get("paths_scoped")):
            violations.append(f"path scope drift requires inventory review: {actual.path}")

        if actual.pull_request:
            if not actual.concurrency:
                violations.append(f"missing concurrency block: {actual.path}")
            if not actual.cancel_in_progress:
                violations.append(f"cancel-in-progress must be true: {actual.path}")
            if not actual.pr_scoped_group:
                violations.append(f"concurrency group is not PR scoped with ref fallback: {actual.path}")

    return violations


def build_report(repo_root: Path, inventory_path: Path) -> dict[str, Any]:
    inventory = load_inventory(inventory_path)
    rows = [inspect_workflow(path, repo_root=repo_root) for path in _active_workflows(repo_root)]
    violations = validate_inventory(repo_root, inventory_path)
    return {
        "schema": "12-6.ci-backpressure-report.v1",
        "inventory_schema": inventory["schema"],
        "workflow_count": len(rows),
        "pull_request_workflow_count": sum(row.pull_request for row in rows),
        "path_scoped_workflow_count": sum(row.paths_scoped for row in rows),
        "concurrency_protected_workflow_count": sum(
            row.pull_request
            and row.concurrency
            and row.cancel_in_progress
            and row.pr_scoped_group
            for row in rows
        ),
        "broad_specialist_workflows_pending_dependency_audit": [
            item["path"]
            for item in inventory["workflows"]
            if item.get("scope_status") == "BROAD_PENDING_DEPENDENCY_AUDIT"
        ],
        "violations": violations,
        "passed": not violations,
    }
