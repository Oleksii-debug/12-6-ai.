from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from twelve_six.data.document_quality import (
    DocumentQualityError,
    assess_document,
    default_quality_policy,
    evaluate_calibration,
    run_quality_filter,
    to_d03_quality_hook,
)


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def test_project_owned_calibration_is_exact() -> None:
    path = Path("data/quality/calibration_uk_en_code_v1.jsonl")
    rows = _jsonl(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    result = evaluate_calibration(rows, calibration_manifest_sha256=digest)
    assert result["samples"] == 30
    assert result["accuracy"] == 1.0
    assert result["false_accepts"] == []
    assert result["false_rejects"] == []
    assert {row["mode"] for row in rows} == {"uk", "en", "code"}
    assert all(row["provenance"] == "PROJECT_AUTHORED_DATA32" for row in rows)


def test_code_uses_structure_not_natural_alpha_gate() -> None:
    text = "SELECT source_id, COUNT(*) FROM records\nWHERE split = 'train'\nGROUP BY source_id;"
    decision = assess_document("sql", text, "code")
    assert decision.accepted
    assert decision.features.code_structure_score >= 2


def test_repetition_template_and_bad_unicode_are_named() -> None:
    repeated = ("Repeated page navigation line.\n" * 8) + "One useful ending sentence."
    assert "high_line_repetition" in assess_document("repeat", repeated, "en").reasons

    template = (
        "Privacy policy\nSign in\nSubscribe\nCookie policy\nAll rights reserved\n"
        "The page body is almost absent."
    )
    assert "high_template_density" in assess_document("template", template, "en").reasons

    invalid = "A coherent document with enough length for the test, but it contains \ufffd damage."
    assert "invalid_replacement_character" in assess_document("bad", invalid, "en").reasons


def test_policy_identity_changes_when_threshold_changes() -> None:
    policy = default_quality_policy()
    changed = replace(policy, en=replace(policy.en, max_symbol_ratio=0.39))
    assert policy.manifest()["policy_sha256"] != changed.manifest()["policy_sha256"]


def test_run_is_deterministic_and_input_manifest_bound() -> None:
    records = [
        {
            "id": "b",
            "mode": "en",
            "text": (
                "This is a sufficiently long English quality record with varied words "
                "and useful context."
            ),
        },
        {
            "id": "a",
            "mode": "code",
            "text": "def add(a, b):\n    return a + b\n",
        },
    ]
    first = run_quality_filter(records, input_manifest_sha256="1" * 64)
    second = run_quality_filter(list(reversed(records)), input_manifest_sha256="1" * 64)
    assert first == second
    changed = run_quality_filter(records, input_manifest_sha256="2" * 64)
    assert first["run_sha256"] != changed["run_sha256"]


def test_quality_hook_uses_incumbent_d03_seam_only() -> None:
    decision = assess_document(
        "doc",
        "A sufficiently long English document with varied vocabulary and normal punctuation.",
        "en",
    )
    hook = to_d03_quality_hook(decision, executed_at="2026-08-25T14:31:04Z")
    assert hook.hook_id == "document_quality"
    assert hook.status == "PASS"
    assert len(hook.evidence_sha256) == 64
    assert not hasattr(hook, "rights_status")


def test_current_view_source_identity_and_expected_counts() -> None:
    view_path = Path("configs/data/document_quality_current_corpus_v1.json")
    view = json.loads(view_path.read_text(encoding="utf-8"))
    source = Path(view["source_path"])
    assert hashlib.sha256(source.read_bytes()).hexdigest() == view["source_sha256"]
    assert source.stat().st_size == view["source_bytes"]
    lines = source.read_text(encoding="utf-8").splitlines()
    records = [
        {
            "id": row["id"],
            "mode": row["mode"],
            "text": "\n".join(lines[row["line_start"] - 1 : row["line_end"]]),
        }
        for row in view["records"]
    ]
    result = run_quality_filter(
        records,
        input_manifest_sha256=hashlib.sha256(view_path.read_bytes()).hexdigest(),
    )
    assert result["input_documents"] == 9
    assert result["accepted_documents"] == 9
    assert result["rejected_documents"] == 0
    assert result["by_mode"] == {
        "uk": {"input": 3, "accepted": 3, "rejected": 0},
        "en": {"input": 3, "accepted": 3, "rejected": 0},
        "code": {"input": 3, "accepted": 3, "rejected": 0},
    }


def test_invalid_manifest_identity_fails_closed() -> None:
    with pytest.raises(DocumentQualityError, match="input_manifest_sha256"):
        run_quality_filter([], input_manifest_sha256="not-a-hash")
