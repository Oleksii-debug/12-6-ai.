from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from twelve_six.data.corpus_v01 import authored_text, norm
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


def test_data25_authored_modalities_pass_without_python_only_assumption() -> None:
    for mode in ("uk", "en", "code"):
        for index in (0, 1, 17, 3384, 14311):
            text = norm(authored_text(mode, index), mode == "code")
            decision = assess_document(f"{mode}-{index}", text, mode)
            assert decision.accepted, (mode, index, decision.reasons)


def test_retained_full_corpus_report_is_bound_to_data25_manifest() -> None:
    view = json.loads(
        Path("configs/data/document_quality_current_corpus_v1.json").read_text(
            encoding="utf-8"
        )
    )
    manifest = json.loads(
        Path("data/corpus/v0.1/manifest.json").read_text(encoding="utf-8")
    )
    report = json.loads(
        Path("reports/d03/document_quality_current_corpus_20260825.json").read_text(
            encoding="utf-8"
        )
    )
    assert view["corpus_identity_sha256"] == manifest["corpus_identity_sha256"]
    assert report["input_manifest_sha256"] == manifest["corpus_identity_sha256"]
    assert report["corpus_identity_sha256"] == manifest["corpus_identity_sha256"]
    assert report["input_documents"] == 46207
    assert report["accepted_documents"] == 46207
    assert report["rejected_documents"] == 0
    assert report["reproduced_shards"] == 36
    assert report["reproduced_byte_tokens"] == 21411248
    assert report["ordered_shard_hashes_match"] is True
    assert report["rebuild_verified"] is True
    assert report["external_training_eligible_sources"] == 0
    assert report["by_mode"] == {
        "uk": {"input": 13899, "accepted": 13899, "rejected": 0},
        "en": {"input": 20093, "accepted": 20093, "rejected": 0},
        "code": {"input": 12215, "accepted": 12215, "rejected": 0},
    }


def test_invalid_manifest_identity_fails_closed() -> None:
    with pytest.raises(DocumentQualityError, match="input_manifest_sha256"):
        run_quality_filter([], input_manifest_sha256="not-a-hash")
