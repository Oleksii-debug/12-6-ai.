#!/usr/bin/env python3
"""Fail-fast static dependency audit for active GitHub Actions workflows.

This auditor is deliberately stdlib-only so it can run before PyTorch, model
artifacts, tokenizers, or training data are installed.  It proves command-level
tool availability from explicit lock installs and D08 purpose-profile metadata;
it never treats whatever happens to be preinstalled in a runner Python as a
declared dependency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
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

# Python modules / executable tools whose availability must be proven by a
# declared package.  Project modules and Python stdlib modules are intentionally
# outside this table.
TOOL_PACKAGE = {
    "pytest": "pytest",
    "ruff": "ruff",
    "torchrun": "torch",
    "torch.distributed.run": "torch",
    "tokenizers": "tokenizers",
    "transformers": "transformers",
    "vllm": "vllm",
}

# Commands supplied by GitHub-hosted Ubuntu runner or shell/Python bootstrap and
# therefore not Python dependency claims.  Unknown external commands are still
# surfaced in the report for review.
RUNNER_BASE_TOOLS = {
    "bash", "cat", "cd", "chmod", "cp", "curl", "cut", "date", "diff", "echo",
    "env", "find", "git", "gh", "grep", "head", "ls", "mkdir", "mv", "printf",
    "pwd", "rm", "sed", "set", "sha256sum", "sh", "sort", "tail", "tar", "tee",
    "test", "touch", "tr", "uname", "wc", "which", "xargs", "zip",
}

PYTHON_STDLIB_MODULES = {
    "compileall", "ensurepip", "json.tool", "pip", "runpy", "site", "venv",
}

LOCK_RE = re.compile(r"(?:^|\s)-r\s+([^\s\\]+\.lock\.txt)", re.MULTILINE)
PROFILE_RE = re.compile(r"requirements/profiles/([A-Za-z0-9_.-]+)(?:/profile\.json)?")
PYTHON_M_RE = re.compile(
    r"(?P<python>(?:[A-Za-z0-9_./-]+/)?python(?:3(?:\.\d+)?)?)\s+-m\s+"
    r"(?P<module>[A-Za-z0-9_.-]+)"
)
DIRECT_TOOL_RE = re.compile(r"(?m)^\s*(?:\.?/[A-Za-z0-9_./-]+/)?(pytest|ruff|torchrun|vllm)\b")
DYNAMIC_EXEC_RE = re.compile(
    r"(?m)^\s*(?:eval\b|\$\{\{[^\n]+\}\}\s+(?:-m\s+)?[A-Za-z]|\$[A-Za-z_][A-Za-z0-9_]*\s+-m\s+)"
)
PIP_INSTALL_RE = re.compile(r"\b(?:python(?:3(?:\.\d+)?)?\s+-m\s+)?pip\s+install\b")


@dataclass(frozen=True)
class Invocation:
    kind: str
    command: str
    package: str | None
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


def _package_from_requirement(line: str) -> str | None:
    line = line.strip()
    if not line or line.startswith("#") or line.startswith("--"):
        return None
    if line.startswith(("-r ", "--requirement ")):
        return None
    match = re.match(r"([A-Za-z0-9_.-]+)", line)
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
    index_path = repo / "requirements/profiles/index.json"
    if not index_path.is_file():
        return {}
    index = _json(index_path)
    profiles: dict[str, dict[str, Any]] = {}
    for profile_id, entry in index.get("profiles", {}).items():
        path = repo / str(entry.get("path", ""))
        if not path.is_file():
            profiles[str(profile_id)] = {"stale": True, "packages": [], "paths": []}
            continue
        payload = _json(path)
        referenced_paths = list(_all_paths(payload))
        base = payload.get("base_profile")
        if isinstance(base, dict) and isinstance(base.get("path"), str):
            base_path = repo / base["path"]
            if base_path.is_file():
                base_payload = _json(base_path)
                referenced_paths.extend(_all_paths(base_payload))
        lock_paths = sorted({p for p in referenced_paths if p.endswith(".lock.txt")})
        packages: set[str] = set()
        for lock in lock_paths:
            packages.update(packages_in_lock(repo / lock))
        profiles[str(profile_id)] = {
            "stale": False,
            "packages": sorted(packages),
            "paths": lock_paths,
        }
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
            key = (module, line_number(text, match.start()), match.group(0))
            if key not in seen:
                seen.add(key)
                invocations.append(
                    Invocation("python_module", match.group(0), package, key[1])
                )
    for match in DIRECT_TOOL_RE.finditer(text):
        tool = match.group(1)
        package = TOOL_PACKAGE[tool]
        key = (tool, line_number(text, match.start()), match.group(0).strip())
        if key not in seen:
            seen.add(key)
            invocations.append(Invocation("console_tool", key[2], package, key[1]))
    return sorted(invocations, key=lambda i: (i.line, i.command))


def direct_pip_packages(text: str) -> set[str]:
    """Conservatively capture literal one-line pip package arguments."""
    packages: set[str] = set()
    for line in text.splitlines():
        if not PIP_INSTALL_RE.search(line) or "-r " in line or "--requirement" in line:
            continue
        try:
            tokens = shlex.split(line.strip().rstrip("\\"))
        except ValueError:
            continue
        try:
            install_index = tokens.index("install")
        except ValueError:
            continue
        for token in tokens[install_index + 1 :]:
            if token.startswith("-") or "${{" in token or token.startswith("$"):
                continue
            package = _package_from_requirement(token)
            if package:
                packages.add(package)
    return packages


def installed_declarations(repo: Path, text: str, profiles: dict[str, dict[str, Any]]) -> dict[str, Any]:
    lock_paths = sorted(set(LOCK_RE.findall(text)))
    missing_locks = [p for p in lock_paths if not (repo / p).is_file()]
    packages: set[str] = set(direct_pip_packages(text))
    for path in lock_paths:
        packages.update(packages_in_lock(repo / path))

    profile_refs = sorted(set(PROFILE_RE.findall(text)))
    stale_profiles = [p for p in profile_refs if p not in profiles or profiles[p].get("stale")]
    # Merely mentioning a D08 profile is not enough to claim installation.  Only
    # include profile packages when a workflow explicitly invokes the repository
    # bootstrap convention with --profile / --profile-id.
    uses_profile_bootstrap = bool(re.search(r"--profile(?:-id)?\s+", text))
    if uses_profile_bootstrap:
        for profile_id in profile_refs:
            if profile_id in profiles and not profiles[profile_id].get("stale"):
                packages.update(profiles[profile_id].get("packages", []))

    return {
        "lock_paths": lock_paths,
        "missing_lock_paths": missing_locks,
        "profile_references": profile_refs,
        "stale_profile_references": stale_profiles,
        "declared_packages": sorted(packages),
        "uses_profile_bootstrap": uses_profile_bootstrap,
    }


def unknown_shell_tools(text: str) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    for number, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "- name:", "run:", "uses:", "with:", "env:")):
            continue
        if stripped.startswith(("${{", "if:", "on:", "jobs:", "steps:", "permissions:")):
            continue
        if ":" in stripped.split()[0] and not stripped.startswith(("http://", "https://")):
            # Likely YAML key, not a shell command.
            continue
        try:
            token = shlex.split(stripped.rstrip("\\"))[0]
        except (ValueError, IndexError):
            continue
        token = token.lstrip("-")
        base = Path(token).name
        if base.startswith("python") or base in RUNNER_BASE_TOOLS or base in TOOL_PACKAGE:
            continue
        if base in {"import", "from", "for", "if", "else:", "elif", "while", "return", "raise"}:
            continue
        if token.startswith((".", "/")) or "/" in token:
            continue
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.+", token):
            continue
        # Report only; unknown shell commands do not automatically fail because
        # runner images intentionally provide a broad system-tool surface.
        found.append({"tool": base, "line": number})
    dedup = {(row["tool"], row["line"]): row for row in found}
    return list(dedup.values())


def audit_workflow(repo: Path, path: Path, profiles: dict[str, dict[str, Any]], exemptions: dict[str, str]) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    rel = path.relative_to(repo).as_posix()
    invocations = extract_invocations(text)
    declared = installed_declarations(repo, text, profiles)
    declared_packages = set(declared["declared_packages"])
    missing = sorted(
        {
            _norm_package(inv.package)
            for inv in invocations
            if inv.package and _norm_package(inv.package) not in declared_packages
        }
    )
    dynamic = [
        {"line": line_number(text, match.start()), "text": match.group(0).strip()}
        for match in DYNAMIC_EXEC_RE.finditer(text)
    ]

    if rel in exemptions:
        classification = "LEGACY_EXEMPT_WITH_REASON"
        reason = exemptions[rel]
    elif declared["stale_profile_references"] or declared["missing_lock_paths"]:
        classification = "STALE_PROFILE_REFERENCE"
        reason = "referenced D08 profile or lock path does not exist"
    elif dynamic:
        classification = "AMBIGUOUS_DYNAMIC_COMMAND"
        reason = "dynamic executable/module cannot be resolved statically"
    elif missing:
        classification = "MISSING_DECLARED_DEPENDENCY"
        reason = "invoked tool is absent from explicitly installed locks/packages"
    else:
        classification = "VALID"
        reason = "all recognized invoked tools are provided by explicit declarations"

    return {
        "path": rel,
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "classification": classification,
        "reason": reason,
        "python_executables": sorted(set(m.group("python") for m in PYTHON_M_RE.finditer(text))),
        "pip_install_present": bool(PIP_INSTALL_RE.search(text)),
        "declarations": declared,
        "invocations": [i.to_dict() for i in invocations],
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
    workflows_dir = repo / ".github/workflows"
    profiles = load_d08_profiles(repo)
    exemptions = load_exemptions(exemptions_path)
    workflows = sorted(
        [*workflows_dir.glob("*.yml"), *workflows_dir.glob("*.yaml")],
        key=lambda p: p.name,
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
    return {
        "schema": SCHEMA,
        "inventory_count": len(rows),
        "d08_profiles": profiles,
        "classification_counts": counts,
        "blocking_workflows": blocking,
        "workflows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--exemptions", type=Path, default=Path("ci/ci153_legacy_exemptions.json"))
    parser.add_argument("--no-fail", action="store_true", help="write report without failing on blocking findings")
    args = parser.parse_args(argv)

    report = audit_repository(args.repo_root, args.exemptions)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "inventory_count": report["inventory_count"],
        "classification_counts": report["classification_counts"],
        "blocking_workflows": report["blocking_workflows"],
    }, ensure_ascii=False, sort_keys=True, indent=2))
    if report["blocking_workflows"] and not args.no_fail:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
