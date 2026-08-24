from __future__ import annotations

import pytest

from tools.sanitize_gitleaks_report import sanitize_findings


def _finding() -> dict[str, object]:
    return {
        "RuleID": "generic-api-key",
        "Description": "fixture",
        "File": "tests/example.txt",
        "StartLine": 4,
        "EndLine": 4,
        "Commit": "1" * 40,
        "Author": "fixture",
        "Date": "2026-08-24T00:00:00Z",
        "Message": "fixture",
        "Tags": ["key"],
        "Fingerprint": f"{'1' * 40}:tests/example.txt:generic-api-key:4",
        "Secret": "do-not-retain-this-secret-value",
        "Match": "api_key=do-not-retain-this-secret-value",
        "Entropy": 4.2,
    }


def test_sanitizer_drops_secret_match_and_unknown_fields() -> None:
    result = sanitize_findings([_finding()])
    assert result["schema"] == "12-6.gitleaks-sanitized-findings.v1"
    assert result["finding_count"] == 1
    item = result["findings"][0]
    assert "Secret" not in item
    assert "Match" not in item
    assert "Entropy" not in item
    assert "do-not-retain-this-secret-value" not in repr(result)


def test_sanitizer_requires_sensitive_fields_to_prove_expected_report_shape() -> None:
    finding = _finding()
    del finding["Secret"]
    with pytest.raises(ValueError, match="missing expected sensitive fields"):
        sanitize_findings([finding])


def test_sanitizer_requires_rule_file_commit_identity() -> None:
    finding = _finding()
    finding["Commit"] = ""
    with pytest.raises(ValueError, match="lacks rule/file/commit identity"):
        sanitize_findings([finding])


def test_sanitizer_rejects_non_array_report() -> None:
    with pytest.raises(ValueError, match="JSON array"):
        sanitize_findings({"findings": []})
