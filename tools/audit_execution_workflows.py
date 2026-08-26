#!/usr/bin/env python3
"""Audit workflow dependency setup against the centralized research execution spine."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any

PINNED_ACTION = re.compile(r"^\s*uses:\s*[^@\s]+@([0-9a-f]{40})(?:\s+#.*)?$", re.MULTILINE)
ANY_ACTION = re.compile(r"^\s*uses:\s*[^@\s]+@([^\s#]+)", re.MULTILINE)


def audit_workflow(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    refs = ANY_ACTION.findall(text)
    pinned = PINNED_ACTION.findall(text)
    unpinned = [ref for ref in refs if not re.fullmatch(r"[0-9a-f]{40}", ref)]

    central = "tools/bootstrap_execution_spine.py" in text
    with_dev = central and "--with-dev" in text
    scientific_tools = ("pytest" in text) or ("ruff" in text)
    direct_lock_install = bool(
        re.search(
            r"pip\s+install(?:(?!\n\s*-\s+name:).)*requirements/locks/",
            text,
            flags=re.DOTALL,
        )
    )
    exact_python = 'python-version: "3.11.16"' in text or "python-version: '3.11.16'" in text

    findings: list[str] = []
    if unpinned:
        findings.append(f"unpinned_actions:{','.join(unpinned)}")
    if scientific_tools and not with_dev:
        findings.append("scientific_tools_without_central_dev_bootstrap")
    if direct_lock_install:
        findings.append("direct_lock_install_deprecated")
    if not exact_python:
        findings.append("exact_python_3_11_16_not_declared")

    return {
        "workflow": path.as_posix(),
        "central_bootstrap": central,
        "central_dev_bootstrap": with_dev,
        "scientific_tools": scientific_tools,
        "direct_lock_install": direct_lock_install,
        "action_refs": len(refs),
        "pinned_action_refs": len(pinned),
        "unpinned_action_refs": unpinned,
        "exact_python_3_11_16": exact_python,
        "status": "PASS" if not findings else "FAIL",
        "findings": findings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workflows", nargs="+", type=Path)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    records = [audit_workflow(path) for path in args.workflows]
    report = {
        "schema_version": "12-6.execution-workflow-audit.v1",
        "workflows": records,
        "status": "PASS" if all(item["status"] == "PASS" for item in records) else "FAIL",
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if args.strict and report["status"] != "PASS":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
