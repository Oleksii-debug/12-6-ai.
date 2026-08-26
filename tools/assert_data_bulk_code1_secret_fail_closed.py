#!/usr/bin/env python3
"""Hard-fail DATA-BULK-CODE-1 when a high-confidence credential scan hits.

The materializer records credential-pattern hits as exclusions.  Issue #635 requires
those hits to fail closed, not merely receive zero byte credit.  This additive CI
firewall turns any such recorded hit into a terminal failure without changing the
bundle's source selection or capacity accounting.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

REPORT_SCHEMA = "12-6.data-bulk-code1-permissive-python-bundle-report.v1"


class SecurityFirewallError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SecurityFirewallError(message)


def credential_exclusions(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    _require(report.get("schema_version") == REPORT_SCHEMA, "unexpected report schema")
    sources = report.get("sources")
    _require(isinstance(sources, list) and len(sources) == 6, "expected six source reports")
    hits: list[dict[str, Any]] = []
    for source in sources:
        _require(isinstance(source, Mapping), "source report must be an object")
        repository = source.get("repository")
        _require(isinstance(repository, str) and repository, "source repository missing")
        exclusions = source.get("exclusions")
        _require(isinstance(exclusions, list), f"{repository}: exclusions ledger missing")
        for exclusion in exclusions:
            _require(isinstance(exclusion, Mapping), f"{repository}: malformed exclusion")
            if exclusion.get("reason") != "credential_pattern":
                continue
            path = exclusion.get("path")
            patterns = exclusion.get("patterns")
            _require(isinstance(path, str) and path, f"{repository}: credential hit path missing")
            _require(isinstance(patterns, list) and patterns, f"{repository}:{path}: credential patterns missing")
            hits.append(
                {
                    "repository": repository,
                    "path": path,
                    "patterns": [str(pattern) for pattern in patterns],
                }
            )
    return hits


def validate(report: Mapping[str, Any]) -> None:
    hits = credential_exclusions(report)
    if hits:
        rendered = "; ".join(
            f"{row['repository']}:{row['path']}[{','.join(row['patterns'])}]" for row in hits
        )
        raise SecurityFirewallError(
            "high-confidence credential/secret pattern detected; fail-closed admission: " + rendered
        )


def _self_test() -> None:
    safe = {
        "schema_version": REPORT_SCHEMA,
        "sources": [
            {"repository": f"owner/repo-{index}", "exclusions": []}
            for index in range(6)
        ],
    }
    validate(safe)

    unsafe = json.loads(json.dumps(safe))
    unsafe["sources"][2]["exclusions"] = [
        {
            "path": "src/pkg/secret.py",
            "reason": "credential_pattern",
            "patterns": ["aws_access_key"],
        }
    ]
    try:
        validate(unsafe)
    except SecurityFirewallError:
        return
    raise AssertionError("credential-pattern regression was not rejected")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", type=Path, nargs="*")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        _self_test()
        print("PASS secret fail-closed firewall self-test")
        return 0

    if not args.reports:
        parser.error("at least one materialization report is required")

    try:
        for path in args.reports:
            report = json.loads(path.read_text(encoding="utf-8"))
            validate(report)
    except (OSError, json.JSONDecodeError, SecurityFirewallError, KeyError, TypeError, ValueError) as exc:
        print(f"DATA-BULK-CODE-1 SECURITY FAIL: {exc}")
        return 1

    print(f"PASS no credential-pattern exclusions in {len(args.reports)} materialization report(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
