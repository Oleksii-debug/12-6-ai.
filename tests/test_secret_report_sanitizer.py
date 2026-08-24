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


def test_sanitizer_retains_only_minimum_location_and_rule_metadata() -> None:
    result = sanitize_findings([_finding()])
    assert result["schema"] == "12-6.gitleaks-sanitized-findings.v1"
    assert result["finding_count"] == 1
    item = result["findings"][0]
    assert set(item) == {"RuleID", "File", "StartLine", "EndLine", "Commit"}
    assert "do-not-retain-this-secret-value" not in repr(result)
    assert "fixture" not in repr(result)


def test_sanitizer_requires_sensitive_fields_to_prove_expected_report_shape() -> None:
    finding = _finding()
    del finding["Secret"]
    with pytest.raises(ValueError, match="missing expected sensitive fields"):
        sanitize_findings([finding])


def test_sanitizer_requires_exact_commit_identity() -> None:
    finding = _finding()
    finding["Commit"] = "ABC"
    with pytest.raises(ValueError, match="lowercase 40-hex Git SHA"):
        sanitize_findings([finding])


def test_sanitizer_rejects_invalid_line_range() -> None:
    finding = _finding()
    finding["StartLine"] = 7
    finding["EndLine"] = 6
    with pytest.raises(ValueError, match="EndLine"):
        sanitize_findings([finding])


def test_sanitizer_rejects_control_characters() -> None:
    finding = _finding()
    finding["File"] = "tests/example.txt\nforged"
    with pytest.raises(ValueError, match="control characters"):
        sanitize_findings([finding])


def test_sanitizer_output_is_deterministic() -> None:
    first = _finding()
    first["File"] = "z.txt"
    second = _finding()
    second["File"] = "a.txt"
    result = sanitize_findings([first, second])
    assert [item["File"] for item in result["findings"]] == ["a.txt", "z.txt"]


def test_sanitizer_rejects_non_array_report() -> None:
    with pytest.raises(ValueError, match="JSON array"):
        sanitize_findings({"findings": []})
