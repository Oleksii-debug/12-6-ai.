from __future__ import annotations

import json
from pathlib import Path

from twelve_six.data.document_quality import assess_document, default_quality_policy
from twelve_six.data.document_quality_real import (
    candidate_quality_policies,
    evaluate_labeled_rows,
    select_policy_on_calibration,
)

ROOT = Path(__file__).resolve().parents[1]


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_real_extension_reuses_incumbent_and_fixed_candidates() -> None:
    candidates = candidate_quality_policies()
    assert candidates[0].manifest()["policy_sha256"] == default_quality_policy().manifest()["policy_sha256"]
    assert [candidate.policy_id for candidate in candidates] == [
        "d03-lightweight-uk-en-code-v1",
        "data108-real-balanced-v3",
        "data108-real-preserve-v2",
        "data108-real-strict-v2",
    ]


def test_balanced_candidate_preserves_short_legitimate_code() -> None:
    text = "x = {'a': 1}\nprint(x)"
    incumbent = assess_document("short", text, "code", policy=default_quality_policy())
    balanced = assess_document("short", text, "code", policy=candidate_quality_policies()[1])
    assert not incumbent.accepted
    assert "too_short" in incumbent.reasons
    assert balanced.accepted


def test_false_rates_are_reported_by_source_family_and_mode() -> None:
    rows = [
        {
            "id": "good",
            "source_family": "family-a",
            "mode": "en",
            "label": "ACCEPT",
            "text": "A coherent English record contains enough varied words to be a legitimate document for calibration.",
        },
        {
            "id": "bad",
            "source_family": "family-b",
            "mode": "en",
            "label": "REJECT",
            "text": "Privacy policy\nSign in\nSubscribe\nCookie policy\nAll rights reserved",
        },
    ]
    report = evaluate_labeled_rows(rows, candidate_quality_policies()[1])
    assert set(report["by_source_family"]) == {"family-a", "family-b"}
    assert set(report["by_mode"]) == {"en"}
    for bucket in [report["overall"], *report["by_source_family"].values(), *report["by_mode"].values()]:
        assert "false_accept_rate_on_rejects" in bucket
        assert "false_reject_rate_on_accepts" in bucket


def test_policy_selection_is_deterministic_and_calibration_only() -> None:
    calibration = [
        {
            "id": "short-code",
            "source_family": "PROJECT_AUTHORED_CONTROL",
            "mode": "code",
            "label": "ACCEPT",
            "text": "x = {'a': 1}\nprint(x)",
        },
        {
            "id": "nav",
            "source_family": "PROJECT_AUTHORED_CONTROL",
            "mode": "en",
            "label": "REJECT",
            "text": "Privacy policy\nSign in\nSubscribe\nCookie policy\nAll rights reserved\nSkip to content",
        },
    ]
    first, first_reports = select_policy_on_calibration(calibration)
    second, second_reports = select_policy_on_calibration(calibration)
    assert first.manifest() == second.manifest()
    assert first_reports == second_reports
    assert first.policy_id != default_quality_policy().policy_id


def test_labeled_specs_cover_real_families_controls_and_hard_cases() -> None:
    calibration = _jsonl(ROOT / "data/quality/calibration_real_sources_v1.jsonl")
    holdout = _jsonl(ROOT / "data/quality/holdout_real_sources_v1.jsonl")
    all_rows = calibration + holdout
    families = {str(row["source_family"]) for row in all_rows}
    assert {
        "ua.rada.open-data.laws-texts",
        "en.standardebooks.manual",
        "PROJECT_AUTHORED_CONTROL",
    } <= families
    rationales = " ".join(str(row["label_rationale"]).casefold() for row in all_rows)
    for concept in (
        "boilerplate",
        "legal",
        "table",
        "short",
        "repetitive",
        "mixed-language",
        "code-heavy",
        "structured",
        "ocr",
        "high-symbol",
    ):
        assert concept in rationales


def test_holdout_source_ranges_do_not_overlap_calibration_ranges() -> None:
    calibration = _jsonl(ROOT / "data/quality/calibration_real_sources_v1.jsonl")
    holdout = _jsonl(ROOT / "data/quality/holdout_real_sources_v1.jsonl")
    cal_ranges = [
        (str(row["source_record_id"]), int(row["start_nonempty_line"]), int(row["end_nonempty_line"]))
        for row in calibration
        if "source_record_id" in row
    ]
    hold_ranges = [
        (str(row["source_record_id"]), int(row["start_nonempty_line"]), int(row["end_nonempty_line"]))
        for row in holdout
        if "source_record_id" in row
    ]
    for cal_record, cal_start, cal_end in cal_ranges:
        for hold_record, hold_start, hold_end in hold_ranges:
            if cal_record != hold_record:
                continue
            assert cal_end < hold_start or hold_end < cal_start


def test_holdout_is_not_passed_into_selection_api() -> None:
    calibration = [
        {
            "id": "short-code",
            "source_family": "PROJECT_AUTHORED_CONTROL",
            "mode": "code",
            "label": "ACCEPT",
            "text": "x = {'a': 1}\nprint(x)",
        }
    ]
    selected, _ = select_policy_on_calibration(calibration)
    holdout = [
        {
            "id": "holdout-only",
            "source_family": "PROJECT_AUTHORED_CONTROL",
            "mode": "en",
            "label": "REJECT",
            "text": "This holdout label is evaluated after selection and cannot alter the selected policy.",
        }
    ]
    before = selected.manifest()["policy_sha256"]
    evaluate_labeled_rows(holdout, selected)
    assert selected.manifest()["policy_sha256"] == before
