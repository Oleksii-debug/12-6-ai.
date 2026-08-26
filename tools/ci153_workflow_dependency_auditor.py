#!/usr/bin/env python3
"""Fail-fast static dependency audit for active GitHub Actions workflows.

The auditor is Python-stdlib-only so it can execute before model/runtime
packages or training artifacts are downloaded. It proves recognized command
availability from literal pip installs, lock files, local pyproject extras, and
D08 profile metadata without assuming undeclared packages from the runner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

SCHEMA = "12-6.ci153-workflow-dependency-audit.v1"
CLASSIFICATIONS = {
    "VALID",
    "MISSING_DECLARED_DEPENDENCY",
    "STALE_PROFILE_REFERENCE",
    "AMBIGUOUS_DYNAMIC_COMMAND",
    "LEGACY_EXEMPT_WITH_REASON",
}
TOOL_PACKAGE = {
    "pytest": "pytest",
    "ruff": "ruff",
    "torchrun": "torch",
    "torch.distributed.run": "torch",
    "tokenizers": "tokenizers",
    "transformers": "transformers",
    "vllm": "vllm",
}
PYTHON_STDLIB_MODULES = {
    "compileall", "ensurepip", "json.tool", "pip", "runpy", "site", "venv",
}
RUNNER_BASE_TOOLS = {
    "bash", "cat", "cd", "chmod", "cp", "curl", "cut", "date", "diff", "echo",
    "env", "find", "git", "gh", "grep", "head", "ls", "mkdir", "mv", "printf",
    "pwd", "rm", "sed", "set", "sha256sum", "sh", "sort", "tail", "tar", "tee",
    "test", "touch", "tr", "uname", "wc", "which", "xargs", "zip",
}

LOCK_RE = re.compile(r"(?:^|\s)-r\s+([^\s\\]+\.lock\.txt)", re.MULTILINE)
PURPOSE_PROFILE_PATH_RE = re.compile(
    r"requirements/profiles/([A-Za-z0-9_.-]+)(?:/profile\.json)?"
)
PROFILE_ARG_RE = re.compile(r"--profile(?:-id)?\s+[\"']?([A-Za-z0-9_.-]+)")
PYTHON_M_RE = re.compile(
    r"(?P<python>(?:[A-Za-z0-9_./-]+/)?python(?:3(?:\.\d+)?)?)\s+-m\s+"
    r"(?P<module>[A-Za-z0-9_.-]+)"
)
PYTHON_EXEC_RE = re.compile(r"(?:^|\s)((?:[A-Za-z0-9_./-]+/)?python(?:3(?:\.\d+)?)?)\b")
DIRECT_TOOL_RE = re.compile(
    r"(?m)^\s*(?:-\s*)?(?:run:\s*)?(?:\.?/[A-Za-z0-9_./-]+/)?"
    r"(pytest|ruff|torchrun|vllm)\b"
)
PIP_INSTALL_RE = re.compile(r"\b(?:python(?:3(?:\.\d+)?)?\s+-m\s+)?pip\s+install\b")
DYNAMIC_EXEC_RE = re.compile(
    r"(?m)^\s*(?:-\s*)?(?:run:\s*)?(?:eval\b|"
    r"\$\{\{[^\n]+\}\}\s+(?:-m\s+)?[A-Za-z]|"
    r"\$[A-Za-z_][A-Za-z0-9_]*\s+-m\s+)"
)
DYNAMIC_REQUIREMENT_RE = re.compile(
    r"\bpip\s+install[^\n]*(?:-r|--requirement)\s+[\"']?(?:\$|\$\{\{)"
)
LOCAL_EXTRA_RE = re.compile(r"(?:^|\s)(?:-e\s+)?\.\[([A-Za-z0-9_,.-]+)\]")


@dataclass(frozen=True)
class Invocation:
    kind: str
    command: str
    package: str
    line: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "command": self.command,
            "package": self.package,
            "line": self.line,
        }


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def _norm_package(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name.strip().lower())


def _package_from_requirement(requirement: str) -> str | None:
    value = requirement.strip()
    if not value or value.startswith(("#", "--", "-r ", "--requirement ")):
        return None
    match = re.match(r"([A-Za-z0-9][A-Za-z0-9_.-]*)", value)
    return _norm_package(match.group(1)) if match else None


def packages_in_lock(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    packages: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        package = _package_from_requirement(line)
        if package:
            packages.add(package)
    return packages


def _all_paths(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "path" and isinstance(item, str):
                yield item
            yield from _all_paths(item)
    elif isinstance(value, list):
        for item in value:
            yield from _all_paths(item)


def load_d08_profiles(repo: Path) -> dict[str, dict[str, Any]]:
    profiles: dict[str, dict[str, Any]] = {}
    index_path = repo / "requirements/profiles/index.json"
    if index_path.is_file():
        index = _json(index_path)
        for profile_id, entry in index.get("profiles", {}).items():
            profile_path = repo / str(entry.get("path", ""))
            if not profile_path.is_file():
                profiles[str(profile_id)] = {
                    "kind": "purpose",
                    "stale": True,
                    "lock_paths": [],
                    "packages": [],
                }
                continue
            payload = _json(profile_path)
            paths = list(_all_paths(payload))
            base = payload.get("base_profile")
            if isinstance(base, dict) and isinstance(base.get("path"), str):
                base_path = repo / base["path"]
                if base_path.is_file():
                    paths.extend(_all_paths(_json(base_path)))
            lock_paths = sorted({p for p in paths if p.endswith(".lock.txt")})
            packages: set[str] = set()
            for lock_path in lock_paths:
                packages.update(packages_in_lock(repo / lock_path))
            profiles[str(profile_id)] = {
                "kind": "purpose",
                "stale": False,
                "lock_paths": lock_paths,
                "packages": sorted(packages),
            }

    locks_root = repo / "requirements/locks"
    if locks_root.is_dir():
        for profile_path in locks_root.glob("*/profile.json"):
            try:
                payload = _json(profile_path)
            except (ValueError, json.JSONDecodeError):
                continue
            profile_id = str(payload.get("profile_id") or profile_path.parent.name)
            paths = list(_all_paths(payload))
            lock_paths = sorted({p for p in paths if p.endswith(".lock.txt")})
            packages: set[str] = set()
            for lock_path in lock_paths:
                packages.update(packages_in_lock(repo / lock_path))
            profiles.setdefault(
                profile_id,
                {
                    "kind": "base-lock",
                    "stale": False,
                    "lock_paths": lock_paths,
                    "packages": sorted(packages),
                },
            )
    return profiles


def line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def extract_invocations(text: str) -> list[Invocation]:
    invocations: list[Invocation] = []
    seen: set[tuple[str, int, str]] = set()
    for match in PYTHON_M_RE.finditer(text):
        module = match.group("module")
        root = module.split(".", 1)[0]
        if module in PYTHON_STDLIB_MODULES or root == "twelve_six":
            continue
        package = TOOL_PACKAGE.get(module) or TOOL_PACKAGE.get(root)
        if package:
            line = line_number(text, match.start())
            key = (package, line, match.group(0))
            if key not in seen:
                seen.add(key)
                invocations.append(
                    Invocation("python_module", match.group(0), package, line)
                )
    for match in DIRECT_TOOL_RE.finditer(text):
        tool = match.group(1)
        line = line_number(text, match.start())
        command = match.group(0).strip()
        key = (tool, line, command)
        if key not in seen:
            seen.add(key)
            invocations.append(Invocation("console_tool", command, TOOL_PACKAGE[tool], line))
    return sorted(invocations, key=lambda row: (row.line, row.command))


def direct_pip_packages(text: str) -> set[str]:
    packages: set[str] = set()
    for line in text.splitlines():
        if not PIP_INSTALL_RE.search(line) or "-r " in line or "--requirement" in line:
            continue
        try:
            tokens = shlex.split(line.strip().rstrip("\\"))
        except ValueError:
            continue
        if "install" not in tokens:
            continue
        start = tokens.index("install") + 1
        for token in tokens[start:]:
            if token.startswith("-") or token.startswith(("$", ".")) or "${{" in token:
                continue
            package = _package_from_requirement(token)
            if package:
                packages.add(package)
    return packages


def pyproject_install_packages(repo: Path, text: str) -> dict[str, Any]:
    extras: set[str] = set()
    project_install = False
    no_deps_project_install = False
    for line in text.splitlines():
        if not PIP_INSTALL_RE.search(line):
            continue
        if re.search(r"(?:^|\s)(?:-e\s+)?\.(?:\s|$)", line):
            project_install = True
            if "--no-deps" in line:
                no_deps_project_install = True
        for match in LOCAL_EXTRA_RE.finditer(line):
            project_install = True
            extras.update(x for x in match.group(1).split(",") if x)
            if "--no-deps" in line:
                no_deps_project_install = True

    packages: set[str] = set()
    missing_extras: list[str] = []
    pyproject = repo / "pyproject.toml"
    if not project_install or no_deps_project_install or not pyproject.is_file():
        return {
            "project_install": project_install,
            "no_deps": no_deps_project_install,
            "extras": sorted(extras),
            "missing_extras": missing_extras,
            "packages": [],
        }
    payload = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    project = payload.get("project", {})
    for requirement in project.get("dependencies", []):
        package = _package_from_requirement(str(requirement))
        if package:
            packages.add(package)
    optional = project.get("optional-dependencies", {})
    for extra in sorted(extras):
        requirements = optional.get(extra)
        if not isinstance(requirements, list):
            missing_extras.append(extra)
            continue
        for requirement in requirements:
            package = _package_from_requirement(str(requirement))
            if package:
                packages.add(package)
    return {
        "project_install": project_install,
        "no_deps": no_deps_project_install,
        "extras": sorted(extras),
        "missing_extras": missing_extras,
        "packages": sorted(packages),
    }


def installed_declarations(
    repo: Path, text: str, profiles: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    lock_paths = sorted(set(LOCK_RE.findall(text)))
    missing_locks = [path for path in lock_paths if not (repo / path).is_file()]
    packages = set(direct_pip_packages(text))
    for path in lock_paths:
        packages.update(packages_in_lock(repo / path))

    pyproject = pyproject_install_packages(repo, text)
    packages.update(pyproject["packages"])

    profile_refs = set(PURPOSE_PROFILE_PATH_RE.findall(text))
    profile_refs.update(PROFILE_ARG_RE.findall(text))
    stale_profiles = sorted(
        profile_id
        for profile_id in profile_refs
        if profile_id not in profiles or profiles[profile_id].get("stale")
    )
    return {
        "lock_paths": lock_paths,
        "missing_lock_paths": missing_locks,
        "profile_references": sorted(profile_refs),
        "stale_profile_references": stale_profiles,
        "pyproject_install": pyproject,
        "declared_packages": sorted(packages),
    }


def ambiguous_commands(text: str) -> list[dict[str, Any]]:
    rows = [
        {"line": line_number(text, match.start()), "text": match.group(0).strip()}
        for match in DYNAMIC_EXEC_RE.finditer(text)
    ]
    rows.extend(
        {
            "line": line_number(text, match.start()),
            "text": match.group(0).strip(),
        }
        for match in DYNAMIC_REQUIREMENT_RE.finditer(text)
    )
    return rows


def unknown_shell_tools(text: str) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    for number, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "- name:", "uses:", "with:", "env:")):
            continue
        if stripped.startswith(("${{", "if:", "on:", "jobs:", "steps:", "permissions:")):
            continue
        if stripped.startswith(("run: |", "run: >", "- run: |", "- run: >")):
            continue
        command = stripped
        if command.startswith("- run:"):
            command = command[len("- run:") :].strip()
        elif command.startswith("run:"):
            command = command[len("run:") :].strip()
        if not command:
            continue
        first = command.split()[0]
        if ":" in first and not first.startswith(("http://", "https://")):
            continue
        try:
            token = shlex.split(command.rstrip("\\"))[0]
        except (ValueError, IndexError):
            continue
        base = Path(token).name
        if base.startswith("python") or base in RUNNER_BASE_TOOLS or base in TOOL_PACKAGE:
            continue
        if base in {"import", "from", "for", "if", "else:", "elif", "while", "return", "raise"}:
            continue
        if token.startswith((".", "/")) or "/" in token:
            continue
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.+", token):
            continue
        found.append({"tool": base, "line": number})
    dedup = {(row["tool"], row["line"]): row for row in found}
    return list(dedup.values())


def audit_workflow(
    repo: Path,
    path: Path,
    profiles: dict[str, dict[str, Any]],
    exemptions: dict[str, str],
) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    rel = path.relative_to(repo).as_posix()
    invocations = extract_invocations(text)
    declarations = installed_declarations(repo, text, profiles)
    declared_packages = set(declarations["declared_packages"])
    missing = sorted(
        {
            _norm_package(invocation.package)
            for invocation in invocations
            if _norm_package(invocation.package) not in declared_packages
        }
    )
    dynamic = ambiguous_commands(text)
    stale = bool(
        declarations["stale_profile_references"]
        or declarations["missing_lock_paths"]
        or declarations["pyproject_install"]["missing_extras"]
    )

    if rel in exemptions:
        classification = "LEGACY_EXEMPT_WITH_REASON"
        reason = exemptions[rel]
    elif stale:
        classification = "STALE_PROFILE_REFERENCE"
        reason = "referenced profile, lock, or pyproject extra is absent"
    elif dynamic:
        classification = "AMBIGUOUS_DYNAMIC_COMMAND"
        reason = "dynamic executable/module/requirement cannot be resolved statically"
    elif missing:
        classification = "MISSING_DECLARED_DEPENDENCY"
        reason = "invoked tool is absent from explicitly installed declarations"
    else:
        classification = "VALID"
        reason = "all recognized invoked tools are supplied by explicit declarations"

    return {
        "path": rel,
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "classification": classification,
        "reason": reason,
        "python_executables": sorted(set(PYTHON_EXEC_RE.findall(text))),
        "pip_install_present": bool(PIP_INSTALL_RE.search(text)),
        "declarations": declarations,
        "invocations": [row.to_dict() for row in invocations],
        "missing_packages": missing,
        "ambiguous_dynamic_commands": dynamic,
        "unknown_shell_tools_review_only": unknown_shell_tools(text),
    }


def load_exemptions(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    payload = _json(path)
    result: dict[str, str] = {}
    for workflow, reason in payload.get("exemptions", {}).items():
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"legacy exemption for {workflow!r} requires a non-empty reason")
        result[str(workflow)] = reason.strip()
    return result


def audit_repository(repo: Path, exemptions_path: Path | None = None) -> dict[str, Any]:
    repo = repo.resolve()
    profiles = load_d08_profiles(repo)
    exemptions = load_exemptions(exemptions_path)
    workflows_dir = repo / ".github/workflows"
    workflows = sorted(
        [*workflows_dir.glob("*.yml"), *workflows_dir.glob("*.yaml")],
        key=lambda path: path.name,
    )
    rows = [audit_workflow(repo, path, profiles, exemptions) for path in workflows]
    counts = {name: 0 for name in sorted(CLASSIFICATIONS)}
    for row in rows:
        counts[row["classification"]] += 1
    blocking = [
        row["path"]
        for row in rows
        if row["classification"] in {
            "MISSING_DECLARED_DEPENDENCY",
            "STALE_PROFILE_REFERENCE",
            "AMBIGUOUS_DYNAMIC_COMMAND",
        }
    ]
    inventory_digest = hashlib.sha256(
        "\n".join(f"{row['path']}:{row['sha256']}" for row in rows).encode("utf-8")
    ).hexdigest()
    return {
        "schema": SCHEMA,
        "inventory_count": len(rows),
        "inventory_sha256": inventory_digest,
        "d08_profiles": profiles,
        "classification_counts": counts,
        "blocking_workflows": blocking,
        "workflows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--exemptions",
        type=Path,
        default=Path("ci/ci153_legacy_exemptions.json"),
    )
    parser.add_argument(
        "--no-fail",
        action="store_true",
        help="write the full report without failing on blocking classifications",
    )
    args = parser.parse_args(argv)

    report = audit_repository(args.repo_root, args.exemptions)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "inventory_count": report["inventory_count"],
                "inventory_sha256": report["inventory_sha256"],
                "classification_counts": report["classification_counts"],
                "blocking_workflows": report["blocking_workflows"],
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
    )
    if report["blocking_workflows"] and not args.no_fail:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
