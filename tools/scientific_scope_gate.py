#!/usr/bin/env python3
"""Fail-closed experiment-local scientific CI gate.

This gate proves only the scientific surface an experiment actually consumes. It
never substitutes for the strict repository integration/release gate.
"""

from __future__ import annotations

import argparse
import ast
import fnmatch
import hashlib
import importlib.metadata
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

LOCK_RE = re.compile(r"^([A-Za-z0-9_.-]+)==([^\s\\]+)")
INTERNAL_PREFIX = "twelve_six"


def _run(cmd: list[str], *, cwd: Path, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, check=check, text=True, capture_output=True)


def _git(root: Path, *args: str) -> str:
    return _run(["git", *args], cwd=root).stdout.strip()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _canon_dist(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def locked_requirements(root: Path) -> tuple[dict[str, str], dict[str, str]]:
    lock_dir = root / "requirements" / "locks" / "linux-x86_64"
    versions: dict[str, str] = {}
    hashes: dict[str, str] = {}
    for stem in ("toolchain", "runtime", "dev"):
        path = lock_dir / f"{stem}.lock.txt"
        if not path.is_file():
            raise RuntimeError(f"missing lock: {path.relative_to(root)}")
        hashes[str(path.relative_to(root))] = _sha256(path)
        for raw in path.read_text(encoding="utf-8").splitlines():
            match = LOCK_RE.match(raw.strip())
            if not match:
                continue
            dist, version = _canon_dist(match.group(1)), match.group(2)
            prior = versions.get(dist)
            if prior is not None and prior != version:
                raise RuntimeError(f"conflicting lock versions for {dist}: {prior} vs {version}")
            versions[dist] = version
    return versions, hashes


def verify_environment(root: Path) -> dict[str, Any]:
    if sys.version_info[:3] != (3, 11, 16):
        raise RuntimeError(f"expected Python 3.11.16, got {sys.version.split()[0]}")
    if shutil.which("git") is None:
        raise RuntimeError("required tool missing: git")
    versions, lock_hashes = locked_requirements(root)
    installed = {
        _canon_dist(dist.metadata["Name"]): dist.version
        for dist in importlib.metadata.distributions()
        if dist.metadata.get("Name")
    }
    bad = {
        name: {"expected": version, "installed": installed.get(name)}
        for name, version in versions.items()
        if installed.get(name) != version
    }
    if bad:
        raise RuntimeError(f"locked environment mismatch: {bad}")
    return {
        "python": sys.version.split()[0],
        "git": _git(root, "--version"),
        "ruff": _run([sys.executable, "-m", "ruff", "--version"], cwd=root).stdout.strip(),
        "pytest": _run([sys.executable, "-m", "pytest", "--version"], cwd=root).stdout.strip(),
        "lock_sha256": lock_hashes,
        "locked_distribution_count": len(versions),
    }


def verify_bootstrap_evidence(path: Path, source_sha: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("source_sha") != source_sha:
        raise RuntimeError("bootstrap evidence source SHA mismatch")
    return {
        "path": path.as_posix(),
        "sha256": _sha256(path),
        "schema": payload.get("schema"),
        "source_sha": payload.get("source_sha"),
    }


def changed_paths(root: Path, base_sha: str, source_sha: str) -> list[str]:
    out = _git(root, "diff", "--name-only", "--diff-filter=ACMRT", f"{base_sha}...{source_sha}")
    return sorted(line for line in out.splitlines() if line)


def _module_candidates(root: Path, module: str) -> list[Path]:
    if not module.startswith(INTERNAL_PREFIX):
        return []
    rel = module.replace(".", "/")
    paths = [root / "src" / f"{rel}.py", root / "src" / rel / "__init__.py"]
    return [path for path in paths if path.is_file()]


def _module_for_path(path: Path, root: Path) -> tuple[str, bool]:
    rel = path.relative_to(root / "src").as_posix()
    is_package = rel.endswith("/__init__.py")
    if rel.endswith(".py"):
        rel = rel[:-3]
    if is_package:
        rel = rel[: -len("/__init__")]
    return rel.replace("/", "."), is_package


def _resolve_from(current: str, is_package: bool, level: int, module: str | None) -> str:
    if level == 0:
        return module or ""
    package = current if is_package else current.rsplit(".", 1)[0]
    parts = package.split(".") if package else []
    ascend = level - 1
    if ascend > len(parts):
        return ""
    base = parts[: len(parts) - ascend]
    if module:
        base.extend(module.split("."))
    return ".".join(base)


def semantic_sources(root: Path, entrypoints: list[str]) -> list[str]:
    queue = [root / path for path in entrypoints]
    seen: set[Path] = set()
    while queue:
        path = queue.pop()
        if not path.is_file():
            raise RuntimeError(f"missing experiment entrypoint/module: {path.relative_to(root)}")
        path = path.resolve()
        if path in seen:
            continue
        seen.add(path)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        internal = str(path).startswith(str((root / "src").resolve()))
        current, is_package = _module_for_path(path, root) if internal else ("", False)
        modules: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.update(alias.name for alias in node.names if alias.name.startswith(INTERNAL_PREFIX))
            elif isinstance(node, ast.ImportFrom):
                base = _resolve_from(current, is_package, node.level, node.module)
                if base.startswith(INTERNAL_PREFIX):
                    modules.add(base)
                    for alias in node.names:
                        if alias.name != "*":
                            modules.add(f"{base}.{alias.name}")
        for module in sorted(modules):
            for candidate in _module_candidates(root, module):
                if candidate.resolve() not in seen:
                    queue.append(candidate)
    return sorted(path.relative_to(root).as_posix() for path in seen)


def owned_tests(paths: list[str], ownership: dict[str, Any]) -> tuple[list[str], dict[str, list[str]]]:
    selected: set[str] = set()
    hits: dict[str, list[str]] = {}
    unowned: list[str] = []
    for path in paths:
        matched: list[str] = []
        for rule in ownership["rules"]:
            if any(fnmatch.fnmatch(path, pattern) for pattern in rule["patterns"]):
                matched.append(rule["id"])
                selected.update(rule["tests"])
        if matched:
            hits[path] = matched
        elif path.startswith("src/twelve_six/"):
            unowned.append(path)
    if unowned:
        raise RuntimeError(f"unowned first-party scientific surface: {unowned}")
    return sorted(selected), hits


def run_gate(
    root: Path,
    manifest: dict[str, Any],
    ownership: dict[str, Any],
    *,
    source_sha: str,
    bootstrap_evidence: Path,
    evidence_out: Path,
) -> dict[str, Any]:
    actual = _git(root, "rev-parse", "HEAD")
    if actual != source_sha:
        raise RuntimeError(f"source SHA mismatch: expected {source_sha}, got {actual}")
    declared_sha = manifest.get("source_sha")
    if declared_sha is not None and declared_sha != source_sha:
        raise RuntimeError("manifest source SHA disagrees with exact checkout")
    base_sha = manifest["base_sha"]
    _git(root, "cat-file", "-e", f"{base_sha}^{{commit}}")

    environment = verify_environment(root)
    bootstrap = verify_bootstrap_evidence(bootstrap_evidence, source_sha)
    changed = changed_paths(root, base_sha, source_sha)
    consumed = semantic_sources(root, list(manifest["entrypoints"]))
    declared = sorted(set(manifest.get("declared_consumed_paths", [])))
    for path in declared:
        if not (root / path).exists():
            raise RuntimeError(f"declared consumed path does not exist: {path}")

    surface = sorted(set(consumed + declared + changed))
    tests, hits = owned_tests(surface, ownership)
    tests = sorted(set(tests + list(manifest.get("additional_tests", []))))
    for test in tests:
        if not (root / test).is_file():
            raise RuntimeError(f"selected focused test missing: {test}")

    changed_python = sorted(p for p in changed if p.endswith(".py") and (root / p).is_file())
    commands: list[dict[str, Any]] = []
    if changed_python:
        cmd = [sys.executable, "-m", "ruff", "check", *changed_python]
        result = _run(cmd, cwd=root, check=False)
        commands.append({"kind": "changed_path_lint", "argv": cmd, "returncode": result.returncode})
        if result.returncode:
            raise RuntimeError(f"changed-path Ruff failed:\n{result.stdout}{result.stderr}")

    compile_paths = sorted(set(consumed + changed_python))
    if compile_paths:
        cmd = [sys.executable, "-m", "py_compile", *compile_paths]
        result = _run(cmd, cwd=root, check=False)
        commands.append({"kind": "static_compile", "argv": cmd, "returncode": result.returncode})
        if result.returncode:
            raise RuntimeError(f"static compile failed:\n{result.stdout}{result.stderr}")

    if tests:
        cmd = [sys.executable, "-m", "pytest", "-q", *tests]
        result = _run(cmd, cwd=root, check=False)
        commands.append({"kind": "focused_regressions", "argv": cmd, "returncode": result.returncode})
        if result.returncode:
            raise RuntimeError(f"focused tests failed:\n{result.stdout}{result.stderr}")

    report = {
        "schema": "12-6.scientific-scope-gate-evidence.v1",
        "experiment_id": manifest["experiment_id"],
        "scope_status": "PASS",
        "source_sha": source_sha,
        "base_sha": base_sha,
        "environment": environment,
        "bootstrap_evidence": bootstrap,
        "changed_paths": changed,
        "changed_python_lint_paths": changed_python,
        "semantic_consumed_paths": consumed,
        "declared_consumed_paths": declared,
        "ownership_hits": hits,
        "focused_tests": tests,
        "commands": commands,
        "whole_repository_health_claimed": False,
        "integration_release_gate": {
            "status": manifest.get("integration_status", "NOT_EVALUATED"),
            "note": manifest.get(
                "integration_note",
                "Strict repository-wide integration/release status is independent of this experiment-local PASS.",
            ),
        },
    }
    evidence_out.parent.mkdir(parents=True, exist_ok=True)
    evidence_out.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--bootstrap-evidence", required=True)
    parser.add_argument("--ownership-config", default="configs/ci/scientific_scope_ownership.v1.json")
    parser.add_argument("--evidence-out", required=True)
    args = parser.parse_args(argv)

    root = Path(args.repo_root).resolve()

    def rooted(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else root / path

    manifest = json.loads(rooted(args.manifest).read_text(encoding="utf-8"))
    ownership = json.loads(rooted(args.ownership_config).read_text(encoding="utf-8"))
    report = run_gate(
        root,
        manifest,
        ownership,
        source_sha=args.source_sha,
        bootstrap_evidence=rooted(args.bootstrap_evidence),
        evidence_out=rooted(args.evidence_out),
    )
    print(json.dumps({
        "scope_status": report["scope_status"],
        "source_sha": report["source_sha"],
        "focused_tests": report["focused_tests"],
        "whole_repository_health_claimed": report["whole_repository_health_claimed"],
        "integration_release_gate": report["integration_release_gate"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
