"""Sanitize Gitleaks JSON findings for safe CI incident triage.

The raw Gitleaks report can contain secret material, commit messages, author
metadata, and matched text.  This module deliberately emits only the minimum
location/rule identity needed to classify a finding.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_SCHEMA = "12-6.gitleaks-sanitized-findings.v1"


class SecretHistoryReportError(ValueError):
    """Raised when a raw Gitleaks report cannot be sanitized safely."""


def _required_text(finding: Mapping[str, Any], key: str) -> str:
    value = finding.get(key)
    if not isinstance(value, str) or not value:
        raise SecretHistoryReportError(f"finding field {key!r} must be non-empty text")
    if _CONTROL.search(value):
        raise SecretHistoryReportError(f"finding field {key!r} contains control characters")
    return value


def _required_line(finding: Mapping[str, Any], key: str) -> int:
    value = finding.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise SecretHistoryReportError(f"finding field {key!r} must be a positive integer")
    return value


def sanitize_gitleaks_findings(payload: object) -> dict[str, object]:
    """Return a deterministic metadata-only representation of Gitleaks findings."""

    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes, bytearray)):
        raise SecretHistoryReportError("Gitleaks report root must be a JSON array")

    sanitized: list[dict[str, object]] = []
    for index, raw in enumerate(payload):
        if not isinstance(raw, Mapping):
            raise SecretHistoryReportError(f"finding {index} must be a JSON object")

        rule_id = _required_text(raw, "RuleID")
        file_path = _required_text(raw, "File")
        commit = _required_text(raw, "Commit")
        if _COMMIT_SHA.fullmatch(commit) is None:
            raise SecretHistoryReportError(
                f"finding {index} Commit must be a lowercase 40-hex Git SHA"
            )
        start_line = _required_line(raw, "StartLine")
        end_line = _required_line(raw, "EndLine")
        if end_line < start_line:
            raise SecretHistoryReportError(
                f"finding {index} EndLine must be greater than or equal to StartLine"
            )

        sanitized.append(
            {
                "rule_id": rule_id,
                "file": file_path,
                "start_line": start_line,
                "end_line": end_line,
                "commit": commit,
            }
        )

    sanitized.sort(
        key=lambda item: (
            str(item["commit"]),
            str(item["file"]),
            int(item["start_line"]),
            str(item["rule_id"]),
        )
    )
    return {
        "schema": _SCHEMA,
        "finding_count": len(sanitized),
        "findings": sanitized,
    }
