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
_PIP_UPGRADE = re.compile(
    r"(?:^|\s)(?:python\s+-m\s+)?pip\s+install\s+(?:--upgrade|-U)\s+pip(?:\s|$)"
)
_SETUP_PYTHON = "actions/setup-python@"
_CHECKOUT = "actions/checkout@"
_EXACT_PYTHON = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")


class WorkflowPolicyError(RuntimeError):
    """Raised when a workflow uses mutable or unsafe supply-chain inputs."""


@dataclass(frozen=True)
class WorkflowViolation:
    path: str
    line: int
    message: str

    def render(self) -> str:
        return f"{self.path}:{self.line}: {self.message}"


def _strip_yaml_scalar(value: str) -> str:
    """Return a simple workflow scalar without an inline comment or quotes."""

    value = value.split(" #", 1)[0].strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1].strip()
    return value


def _uses_value(line: str) -> str | None:
    match = _USES_LINE.match(line)
    if match is None:
        return None
    return _strip_yaml_scalar(match.group("value"))


def _next_non_comment_lines(lines: tuple[str, ...], start: int, limit: int = 10) -> tuple[str, ...]:
    selected: list[str] = []
    for line in lines[start : start + limit]:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        selected.append(stripped)
    return tuple(selected)


def validate_workflow_text(path: str, text: str) -> tuple[WorkflowViolation, ...]:
    """Validate one workflow without requiring a YAML parser dependency.

    The scanner intentionally covers only supply-chain-sensitive constructs that
    GitHub Actions itself represents as line-oriented keys: ``uses:``, checkout
    credential persistence, setup-python runtime selection and floating pip
    self-upgrades. It does not attempt to interpret arbitrary workflow logic.
    """

    lines = tuple(text.splitlines())
    violations: list[WorkflowViolation] = []

    for index, line in enumerate(lines):
        line_number = index + 1
        uses = _uses_value(line)
        if uses is not None:
            if uses.startswith("./"):
                pass
            elif uses.startswith("docker://"):
                if _DOCKER_DIGEST.fullmatch(uses) is None:
                    violations.append(
                        WorkflowViolation(
                            path,
                            line_number,
                            "docker action must be pinned by sha256 digest",
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
            if uses.startswith(_CHECKOUT) and not any(
                item.replace(" ", "") == "persist-credentials:false" for item in following
            ):
                violations.append(
                    WorkflowViolation(
                        path,
                        line_number,
                        "actions/checkout must set persist-credentials: false",
                    )
                )

            if uses.startswith(_SETUP_PYTHON):
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

        if _PIP_UPGRADE.search(line):
            violations.append(
                WorkflowViolation(
                    path,
                    line_number,
                    "workflow must not float pip via pip install --upgrade/-U pip",
                )
            )

    return tuple(violations)


def validate_workflow_files(workflows: tuple[Path, ...], repo_root: Path) -> None:
    """Validate all supplied workflow files and raise one deterministic error."""

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
    """Return tracked workflow candidates from the checkout filesystem."""

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
