from __future__ import annotations

import pytest

from twelve_six.integration.secret_history import (
    SecretHistoryReportError,
    sanitize_gitleaks_findings,
)

_COMMIT_A = "a" * 40
_COMMIT_B = "b" * 40


def _raw_finding(
    *,
    commit: str = _COMMIT_A,
    file: str = "docs/example.md",
    start_line: int = 7,
    end_line: int = 7,
    rule_id: str = "generic-api-key",
) -> dict[str, object]:
    return {
        "Description": "description may contain arbitrary scanner text",
        "StartLine": start_line,
        "EndLine": end_line,
        "StartColumn": 1,
        "EndColumn": 40,
        "Match": "THIS_MATCH_MUST_NEVER_SURVIVE",
        "Secret": "THIS_SECRET_MUST_NEVER_SURVIVE",
        "File": file,
        "SymlinkFile": "",
        "Commit": commit,
        "Entropy": 4.2,
        "Author": "private author",
        "Email": "private@example.invalid",
        "Date": "2026-08-24T00:00:00Z",
        "Message": "commit message may contain sensitive text",
        "Tags": ["key"],
        "RuleID": rule_id,
        "Fingerprint": "scanner-controlled fingerprint",
    }


def test_sanitizer_keeps_only_non_secret_location_metadata() -> None:
    result = sanitize_gitleaks_findings([_raw_finding()])

    assert result == {
        "schema": "12-6.gitleaks-sanitized-findings.v1",
        "finding_count": 1,
        "findings": [
            {
                "rule_id": "generic-api-key",
                "file": "docs/example.md",
                "start_line": 7,
                "end_line": 7,
                "commit": _COMMIT_A,
            }
        ],
    }
    rendered = repr(result)
    assert "THIS_SECRET_MUST_NEVER_SURVIVE" not in rendered
    assert "THIS_MATCH_MUST_NEVER_SURVIVE" not in rendered
    assert "private@example.invalid" not in rendered
    assert "commit message may contain sensitive text" not in rendered
    assert "scanner-controlled fingerprint" not in rendered


def test_sanitizer_orders_findings_deterministically() -> None:
    result = sanitize_gitleaks_findings(
        [
            _raw_finding(commit=_COMMIT_B, file="z.txt", start_line=2, end_line=2),
            _raw_finding(commit=_COMMIT_A, file="a.txt", start_line=5, end_line=5),
            _raw_finding(commit=_COMMIT_A, file="a.txt", start_line=3, end_line=3),
        ]
    )

    assert [
        (item["commit"], item["file"], item["start_line"])
        for item in result["findings"]
    ] == [
        (_COMMIT_A, "a.txt", 3),
        (_COMMIT_A, "a.txt", 5),
        (_COMMIT_B, "z.txt", 2),
    ]


@pytest.mark.parametrize(
    "payload",
    [
        {},
        [None],
        [_raw_finding(commit="not-a-sha")],
        [_raw_finding(start_line=0)],
        [_raw_finding(start_line=8, end_line=7)],
        [_raw_finding(file="unsafe\npath.txt")],
    ],
)
def test_sanitizer_fails_closed_on_malformed_metadata(payload: object) -> None:
    with pytest.raises(SecretHistoryReportError):
        sanitize_gitleaks_findings(payload)
