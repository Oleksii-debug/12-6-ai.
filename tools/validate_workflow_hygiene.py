"""Fail-closed policy checks for repository GitHub Actions workflows.

This validator deliberately uses only the Python standard library so it can run
before project dependencies are installed. It is not a general YAML parser; it
validates the small set of structural invariants that bound Actions queue pressure
and workflow privilege in this repository.
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

WORKFLOW_GLOBS = ("*.yml", "*.yaml")
_KEY_RE = re.compile(r"^([A-Za-z0-9_.-]+):(?:\s*(.*))?$")


@dataclass(frozen=True)
class Line:
    number: int
    indent: int
    text: str


class WorkflowPolicyError(ValueError):
    """Raised when one workflow violates the repository queue-hygiene contract."""


def _meaningful_lines(text: str) -> list[Line]:
    result: list[Line] = []
    for number, raw in enumerate(text.splitlines(), start=1):
        if "\t" in raw[: len(raw) - len(raw.lstrip())]:
            raise WorkflowPolicyError(f"line {number}: tab indentation is not allowed")
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        result.append(Line(number=number, indent=indent, text=raw.strip()))
    return result


def _key(line: Line) -> tuple[str, str] | None:
    match = _KEY_RE.match(line.text)
    if not match:
        return None
    return match.group(1), (match.group(2) or "").strip()


def _top_level(lines: list[Line]) -> dict[str, int]:
    result: dict[str, int] = {}
    for index, line in enumerate(lines):
        if line.indent != 0:
            continue
        parsed = _key(line)
        if parsed is None:
            raise WorkflowPolicyError(
                f"line {line.number}: expected a top-level mapping key, got {line.text!r}"
            )
        name, _ = parsed
        if name in result:
            raise WorkflowPolicyError(f"line {line.number}: duplicate top-level key {name!r}")
        result[name] = index
    return result


def _block(lines: list[Line], start_index: int) -> list[Line]:
    block: list[Line] = []
    for line in lines[start_index + 1 :]:
        if line.indent == 0:
            break
        block.append(line)
    return block


def _require_concurrency(lines: list[Line], top: dict[str, int]) -> None:
    if "concurrency" not in top:
        raise WorkflowPolicyError("missing top-level concurrency block")
    block = _block(lines, top["concurrency"])
    cancel = False
    group_value = ""
    for line in block:
        parsed = _key(line)
        if parsed is None:
            continue
        name, value = parsed
        if name == "cancel-in-progress":
            cancel = value.lower() == "true"
        elif name == "group":
            group_value = value
    if not cancel:
        raise WorkflowPolicyError("concurrency.cancel-in-progress must be true")
    if "github.workflow" not in group_value:
        raise WorkflowPolicyError("concurrency.group must include github.workflow")
    if not any(
        marker in group_value
        for marker in ("github.event.pull_request.number", "github.ref", "github.head_ref")
    ):
        raise WorkflowPolicyError(
            "concurrency.group must distinguish the pull request, head ref, or git ref"
        )


def _require_permissions(lines: list[Line], top: dict[str, int]) -> None:
    if "permissions" not in top:
        raise WorkflowPolicyError("missing explicit top-level permissions")
    header = lines[top["permissions"]]
    parsed = _key(header)
    assert parsed is not None
    _, inline = parsed
    if inline:
        if inline.lower() == "write-all":
            raise WorkflowPolicyError("top-level permissions: write-all is forbidden")
        if inline not in {"{}", "read-all"}:
            raise WorkflowPolicyError(
                "inline permissions must be {}, read-all, or an explicit mapping"
            )
        return

    block = _block(lines, top["permissions"])
    if not block:
        raise WorkflowPolicyError("permissions mapping must not be empty")
    for line in block:
        parsed = _key(line)
        if parsed is None:
            raise WorkflowPolicyError(
                f"line {line.number}: malformed permissions entry {line.text!r}"
            )
        _, value = parsed
        if value not in {"read", "write", "none"}:
            raise WorkflowPolicyError(
                f"line {line.number}: permission level must be read, write, or none"
            )


def _direct_children(block: list[Line]) -> list[tuple[int, Line]]:
    if not block:
        return []
    child_indent = min(line.indent for line in block)
    return [(index, line) for index, line in enumerate(block) if line.indent == child_indent]


def _require_job_timeouts(lines: list[Line], top: dict[str, int]) -> None:
    if "jobs" not in top:
        raise WorkflowPolicyError("missing top-level jobs block")
    jobs = _block(lines, top["jobs"])
    children = _direct_children(jobs)
    if not children:
        raise WorkflowPolicyError("jobs mapping must contain at least one job")

    for child_pos, (block_index, job_line) in enumerate(children):
        parsed = _key(job_line)
        if parsed is None:
            raise WorkflowPolicyError(
                f"line {job_line.number}: malformed job definition {job_line.text!r}"
            )
        job_name, inline = parsed
        if inline:
            raise WorkflowPolicyError(
                f"line {job_line.number}: job {job_name!r} must be a mapping"
            )
        next_index = children[child_pos + 1][0] if child_pos + 1 < len(children) else len(jobs)
        job_block = jobs[block_index + 1 : next_index]
        direct = _direct_children(job_block)
        direct_map: dict[str, str] = {}
        for _, line in direct:
            item = _key(line)
            if item is not None:
                direct_map[item[0]] = item[1]
        if "uses" in direct_map:
            # Reusable-workflow jobs execute their own job-level timeout policy.
            continue
        raw_timeout = direct_map.get("timeout-minutes")
        if raw_timeout is None:
            raise WorkflowPolicyError(f"job {job_name!r} is missing timeout-minutes")
        try:
            timeout = int(raw_timeout)
        except ValueError as exc:
            raise WorkflowPolicyError(
                f"job {job_name!r} timeout-minutes must be an integer"
            ) from exc
        if timeout <= 0 or timeout > 360:
            raise WorkflowPolicyError(
                f"job {job_name!r} timeout-minutes must be between 1 and 360"
            )


def validate_workflow_text(text: str, *, source: str = "<memory>") -> None:
    """Validate one workflow document and raise on the first policy violation."""

    try:
        lines = _meaningful_lines(text)
        if not lines:
            raise WorkflowPolicyError("workflow is empty")
        top = _top_level(lines)
        for required in ("name", "on", "jobs"):
            if required not in top:
                raise WorkflowPolicyError(f"missing top-level {required!r} key")
        _require_concurrency(lines, top)
        _require_permissions(lines, top)
        _require_job_timeouts(lines, top)
    except WorkflowPolicyError as exc:
        raise WorkflowPolicyError(f"{source}: {exc}") from exc


def discover_workflows(root: Path) -> list[Path]:
    workflow_dir = root / ".github" / "workflows"
    paths: list[Path] = []
    for pattern in WORKFLOW_GLOBS:
        paths.extend(workflow_dir.glob(pattern))
    return sorted(path for path in paths if path.is_file())


def validate_paths(paths: Iterable[Path]) -> list[Path]:
    checked: list[Path] = []
    for path in paths:
        validate_workflow_text(path.read_text(encoding="utf-8"), source=str(path))
        checked.append(path)
    if not checked:
        raise WorkflowPolicyError("no workflow files found")
    return checked


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, help="workflow paths to validate")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="repository root")
    args = parser.parse_args()

    paths = args.paths or discover_workflows(args.root)
    try:
        checked = validate_paths(paths)
    except (OSError, UnicodeError, WorkflowPolicyError) as exc:
        print(f"WORKFLOW_HYGIENE_FAIL: {exc}")
        return 1

    print(f"WORKFLOW_HYGIENE_PASS: {len(checked)} workflow(s)")
    for path in checked:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
