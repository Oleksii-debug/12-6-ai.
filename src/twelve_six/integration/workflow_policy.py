"""Fail-closed policy checks for GitHub Actions workflow supply-chain inputs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_GITHUB_ACTION = re.compile(
    r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.\-/]+)?@[0-9a-f]{40}$"
)
_DOCKER_DIGEST = re.compile(r"^docker://[^\s@]+@sha256:[0-9a-f]{64}$")
_USES_LINE = re.compile(r"^\s*(?:-\s*)?uses:\s*(?P<value>.+?)\s*$")
_PIP_INSTALL = re.compile(r"(?:^|\s)(?:python\s+-m\s+)?pip\s+install(?:\s|$)")
_PIP_UPGRADE = re.compile(
    r"(?:^|\s)(?:python\s+-m\s+)?pip\s+install\s+(?:--upgrade|-U)\s+pip(?:\s|$)"
)
_LOCAL_PROJECT_INSTALL = re.compile(
    r"(?:^|\s)(?:(?:-e|--editable)\s+)?\.(?:\[[^\]]+\])?(?:\s|$)"
)
_EXACT_PYTHON = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
_TIMEOUT = re.compile(r"^[1-9][0-9]?$|^1[0-1][0-9]$|^120$")
_READ_PERMISSION = re.compile(r"^(?:read|none)$")


class WorkflowPolicyError(RuntimeError):
    """Raised when a workflow uses mutable or unsafe CI inputs or semantics."""


@dataclass(frozen=True)
class WorkflowViolation:
    path: str
    line: int
    message: str

    def render(self) -> str:
        return f"{self.path}:{self.line}: {self.message}"


def _strip_yaml_scalar(value: str) -> str:
    value = value.split(" #", 1)[0].strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1].strip()
    return value


def _uses_value(line: str) -> str | None:
    match = _USES_LINE.match(line)
    if match is None:
        return None
    return _strip_yaml_scalar(match.group("value"))


def _next_non_comment_lines(lines: tuple[str, ...], start: int, limit: int = 14) -> tuple[str, ...]:
    selected: list[str] = []
    for line in lines[start : start + limit]:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        selected.append(stripped)
    return tuple(selected)


def _continued_shell_command(lines: tuple[str, ...], index: int) -> str:
    """Return one shell command, joining explicit backslash continuations only."""

    parts = [lines[index].strip()]
    cursor = index
    while parts[-1].endswith("\\") and cursor + 1 < len(lines):
        cursor += 1
        candidate = lines[cursor].strip()
        if not candidate or candidate.startswith("#"):
            continue
        parts.append(candidate)
    return " ".join(part.removesuffix("\\").strip() for part in parts)


def _workflow_name(lines: tuple[str, ...]) -> str:
    for line in lines:
        if line.startswith("name:"):
            return _strip_yaml_scalar(line.split(":", 1)[1])
    return ""


def _top_level_permissions(lines: tuple[str, ...]) -> tuple[int, dict[str, str]] | None:
    for index, line in enumerate(lines):
        if line == "permissions:":
            permissions: dict[str, str] = {}
            for child in lines[index + 1 :]:
                if child and not child.startswith((" ", "\t")):
                    break
                stripped = child.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                if ":" not in stripped:
                    continue
                key, value = stripped.split(":", 1)
                permissions[key.strip()] = _strip_yaml_scalar(value)
            return index + 1, permissions
    return None


def _contains_exact(lines: tuple[str, ...], needle: str) -> bool:
    return any(needle in line for line in lines)


def validate_workflow_text(path: str, text: str) -> tuple[WorkflowViolation, ...]:
    """Validate one workflow using deterministic line-oriented policy checks."""

    lines = tuple(text.splitlines())
    violations: list[WorkflowViolation] = []
    name = _workflow_name(lines)

    permissions_result = _top_level_permissions(lines)
    if permissions_result is None:
        violations.append(WorkflowViolation(path, 1, "top-level permissions mapping is required"))
    else:
        permission_line, permissions = permissions_result
        if permissions.get("contents") != "read":
            violations.append(
                WorkflowViolation(path, permission_line, "top-level contents permission must be read")
            )
        for scope, value in sorted(permissions.items()):
            if _READ_PERMISSION.fullmatch(value) is None:
                violations.append(
                    WorkflowViolation(
                        path,
                        permission_line,
                        f"permission {scope!r} must be read or none, not {value!r}",
                    )
                )

    job_count = 0
    timeout_count = 0
    for index, line in enumerate(lines):
        line_number = index + 1
        stripped = line.strip()
        if stripped.startswith("runs-on:"):
            job_count += 1
            runner = _strip_yaml_scalar(stripped.split(":", 1)[1])
            if "latest" in runner.lower():
                violations.append(
                    WorkflowViolation(
                        path,
                        line_number,
                        "runs-on must not use a mutable latest runner label",
                    )
                )
        if stripped.startswith("timeout-minutes:"):
            timeout_count += 1
            value = _strip_yaml_scalar(stripped.split(":", 1)[1])
            if _TIMEOUT.fullmatch(value) is None:
                violations.append(
                    WorkflowViolation(path, line_number, "timeout-minutes must be an integer 1..120")
                )

        uses = _uses_value(line)
        if uses is not None:
            if uses.startswith("./"):
                pass
            elif uses.startswith("docker://"):
                if _DOCKER_DIGEST.fullmatch(uses) is None:
                    violations.append(
                        WorkflowViolation(
                            path, line_number, "docker action must be pinned by sha256 digest"
                        )
                    )
            elif _GITHUB_ACTION.fullmatch(uses) is None:
                violations.append(
                    WorkflowViolation(
                        path,
                        line_number,
                        "external action/reusable workflow must use an immutable 40-hex commit SHA",
                    )
                )

            following = _next_non_comment_lines(lines, index + 1)
            if uses.startswith("actions/checkout@"):
                if not any(
                    item.replace(" ", "") == "persist-credentials:false" for item in following
                ):
                    violations.append(
                        WorkflowViolation(
                            path,
                            line_number,
                            "actions/checkout must set persist-credentials: false",
                        )
                    )
                depth_values = [
                    _strip_yaml_scalar(item.split(":", 1)[1])
                    for item in following
                    if item.startswith("fetch-depth:")
                ]
                if len(depth_values) != 1 or not depth_values[0].isdigit():
                    violations.append(
                        WorkflowViolation(
                            path,
                            line_number,
                            "actions/checkout must declare one explicit numeric fetch-depth",
                        )
                    )

            if uses.startswith("actions/setup-python@"):
                python_values = [
                    _strip_yaml_scalar(item.split(":", 1)[1])
                    for item in following
                    if item.startswith("python-version:")
                ]
                if len(python_values) != 1 or _EXACT_PYTHON.fullmatch(python_values[0]) is None:
                    violations.append(
                        WorkflowViolation(
                            path,
                            line_number,
                            "actions/setup-python must select one exact X.Y.Z python-version",
                        )
                    )
                cache_values = [
                    _strip_yaml_scalar(item.split(":", 1)[1])
                    for item in following
                    if item.startswith("cache:")
                ]
                if cache_values:
                    if cache_values != ["pip"]:
                        violations.append(
                            WorkflowViolation(path, line_number, "setup-python cache must be pip")
                        )
                    window = "\n".join(following)
                    if "cache-dependency-path:" not in window or "requirements/locks/" not in window:
                        violations.append(
                            WorkflowViolation(
                                path,
                                line_number,
                                "pip cache must be keyed by committed requirements/locks inputs",
                            )
                        )

        if _PIP_UPGRADE.search(line):
            violations.append(
                WorkflowViolation(
                    path,
                    line_number,
                    "workflow must not float pip via pip install --upgrade/-U pip",
                )
            )
        elif _PIP_INSTALL.search(line):
            command = _continued_shell_command(lines, index)
            hash_locked = "--require-hashes" in command and "requirements/locks/" in command
            local_no_deps = "--no-deps" in command and _LOCAL_PROJECT_INSTALL.search(command) is not None
            if not hash_locked and not local_no_deps:
                violations.append(
                    WorkflowViolation(
                        path,
                        line_number,
                        "direct pip install must use requirements/locks with --require-hashes "
                        "or install the local project with --no-deps",
                    )
                )

    if job_count == 0:
        violations.append(WorkflowViolation(path, 1, "workflow must define at least one job"))
    elif timeout_count < job_count:
        violations.append(
            WorkflowViolation(path, 1, "every job must declare an explicit timeout-minutes")
        )

    cancel_enabled = _contains_exact(lines, "cancel-in-progress: true")
    if cancel_enabled:
        if name != "Fast CI":
            violations.append(
                WorkflowViolation(
                    path,
                    1,
                    "cancel-in-progress is allowed only in the non-authoritative Fast CI workflow",
                )
            )
        if not _contains_exact(lines, "pull_request:"):
            violations.append(
                WorkflowViolation(path, 1, "cancelable Fast CI must be pull_request-only")
            )
        if not _contains_exact(lines, "github.event.pull_request.number"):
            violations.append(
                WorkflowViolation(
                    path,
                    1,
                    "cancelable Fast CI must group by github.event.pull_request.number",
                )
            )

    if name == "CI":
        required_markers = {
            "fetch-depth: 0": "authoritative CI must checkout full history",
            "gitleaks git": "authoritative CI must run raw gitleaks git",
            "--full-history --all": "authoritative secret scan must request all full history",
            "sha256sum --check": "authoritative Gitleaks download must verify SHA-256",
            "GITLEAKS_ARCHIVE_SHA256": "authoritative CI must pin Gitleaks archive SHA-256",
            "Secret-gate negative fixture": "authoritative CI must execute a secret failure fixture",
        }
        for marker, message in required_markers.items():
            if not _contains_exact(lines, marker):
                violations.append(WorkflowViolation(path, 1, message))
        if cancel_enabled:
            violations.append(
                WorkflowViolation(path, 1, "authoritative CI evidence must never be auto-canceled")
            )

    return tuple(violations)


def validate_workflow_files(workflows: tuple[Path, ...], repo_root: Path) -> None:
    violations: list[WorkflowViolation] = []
    root = repo_root.resolve()
    for workflow in sorted(workflows, key=lambda item: item.as_posix()):
        resolved = workflow.resolve()
        try:
            relative = resolved.relative_to(root).as_posix()
        except ValueError as exc:
            raise WorkflowPolicyError(f"workflow is outside repository root: {workflow}") from exc
        text = resolved.read_text(encoding="utf-8")
        violations.extend(validate_workflow_text(relative, text))

    if violations:
        raise WorkflowPolicyError("\n".join(item.render() for item in violations))


def repository_workflows(repo_root: str | Path = ".") -> tuple[Path, ...]:
    root = Path(repo_root).resolve()
    workflow_dir = root / ".github" / "workflows"
    if not workflow_dir.is_dir():
        raise WorkflowPolicyError(".github/workflows directory is missing")
    workflows = tuple(sorted((*workflow_dir.glob("*.yml"), *workflow_dir.glob("*.yaml"))))
    if not workflows:
        raise WorkflowPolicyError("no GitHub Actions workflows found")
    return workflows


def validate_repository_workflows(repo_root: str | Path = ".") -> None:
    root = Path(repo_root).resolve()
    validate_workflow_files(repository_workflows(root), root)
