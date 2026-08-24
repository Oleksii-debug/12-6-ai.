#!/usr/bin/env python3
"""Sanitize a Gitleaks JSON report without retaining secret material."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

_SCHEMA = "12-6.gitleaks-sanitized-findings.v1"
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_FORBIDDEN_FIELDS = {"Secret", "Match"}


def _required_text(finding: dict[str, Any], key: str, index: int) -> str:
    value = finding.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"finding {index} field {key!r} must be non-empty text")
    if _CONTROL.search(value):
        raise ValueError(f"finding {index} field {key!r} contains control characters")
    return value


def _required_line(finding: dict[str, Any], key: str, index: int) -> int:
    value = finding.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"finding {index} field {key!r} must be a positive integer")
    return value


def sanitize_findings(payload: Any) -> dict[str, Any]:
    """Return minimal location/rule metadata and reject malformed report shapes."""
    if not isinstance(payload, list):
        raise ValueError("Gitleaks report must be a JSON array")

    sanitized: list[dict[str, Any]] = []
    for index, finding in enumerate(payload):
        if not isinstance(finding, dict):
            raise ValueError(f"finding {index} must be a JSON object")
        if not _FORBIDDEN_FIELDS.issubset(finding):
            missing = sorted(_FORBIDDEN_FIELDS.difference(finding))
            raise ValueError(f"finding {index} is missing expected sensitive fields: {missing}")

        rule_id = _required_text(finding, "RuleID", index)
        file_path = _required_text(finding, "File", index)
        commit = _required_text(finding, "Commit", index)
        if _GIT_SHA.fullmatch(commit) is None:
            raise ValueError(f"finding {index} Commit must be a lowercase 40-hex Git SHA")
        start_line = _required_line(finding, "StartLine", index)
        end_line = _required_line(finding, "EndLine", index)
        if end_line < start_line:
            raise ValueError(f"finding {index} EndLine must be >= StartLine")

        sanitized.append(
            {
                "RuleID": rule_id,
                "File": file_path,
                "StartLine": start_line,
                "EndLine": end_line,
                "Commit": commit,
            }
        )

    sanitized.sort(
        key=lambda item: (
            item["Commit"],
            item["File"],
            item["StartLine"],
            item["RuleID"],
        )
    )
    return {
        "schema": _SCHEMA,
        "finding_count": len(sanitized),
        "findings": sanitized,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    sanitized = sanitize_findings(payload)
    args.output.write_text(
        json.dumps(sanitized, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(f"sanitized_gitleaks_findings={sanitized['finding_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
