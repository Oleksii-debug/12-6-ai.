from __future__ import annotations

import json
from pathlib import Path

import pytest

from twelve_six.data.decontamination_authority_v2 import (
    DecontaminationError,
    build_blocker_report,
    build_report,
    validate_authority_metadata,
    verify_report,
    write_immutable_report,
)


def authorities():
    return {
        "schema": "12-6.data232-reserved-authorities.v1",
        "authorities": [
            {
                "authority_id": "eval-final",
                "identity_sha256": "1" * 64,
                "role": "final_test",
                "source_sha": "a" * 40,
            },
            {
                "authority_id": "eval-aux",
                "identity_sha256": "2" * 64,
                "role": "auxiliary_reserved",
                "source_sha": "b" * 40,
            },
        ],
    }


def row(record_id, text, *, source="source-a", family="family-a", modality="en"):
    return {
        "record_id": record_id,
        "source_id": source,
        "source_family": family,
        "modality": modality,
        "text": text,
    }


def build(train, evaluation, **kwargs):
    return build_report(
        train,
        evaluation,
        training_corpus_identity="3" * 64,
        selection_validation_identity="4" * 64,
        final_test_identity="5" * 64,
        authorities=authorities(),
        quarantine_cross_source_families=False,
        **kwargs,
    )


def test_raw_exact_overlap_is_excluded():
    report = build(
        [row("t1", "The exact reserved sentence appears here.")],
        [row("e1", "The exact reserved sentence appears here.", source="eval", family="eval")],
    )
    assert report["status"] == "PASS_WITH_EXCLUSIONS"
    assert report["excluded_records"][0]["record_id_sha256"]
    assert any(item["match_type"] == "raw_exact" for item in report["match_evidence"])


def test_adversarial_unicode_whitespace_normalized_exact():
    train = "Ｆuture\u00a0evaluation\u200b text\r\nMUST remain reserved."
    evaluation = "Future evaluation text\nMUST remain reserved."
    report = build(
        [row("t1", train)],
        [row("e1", evaluation, source="eval", family="eval")],
    )
    assert any(item["match_type"] == "normalized_exact" for item in report["match_evidence"])
    assert report["counts"]["excluded_training_records"] == 1


def test_header_footer_fragment_is_caught():
    core = " ".join(f"reserved{i}" for i in range(40))
    report = build(
        [row("t1", "publisher header " + core + " footer navigation")],
        [row("e1", core, source="eval", family="eval")],
        thresholds={"natural_fragment_containment": 0.80},
    )
    assert any(item["match_type"] == "document_fragment" for item in report["match_evidence"])


def test_near_duplicate_edit_is_caught():
    base = " ".join(f"token{i}" for i in range(60))
    changed = base.replace("token20", "changed20").replace("token41", "changed41")
    report = build(
        [row("t1", changed)],
        [row("e1", base, source="eval", family="eval")],
    )
    assert any(item["match_type"] == "near_match" for item in report["match_evidence"])


def test_cross_source_mirror_chain_propagates_eval_exclusion():
    base = " ".join(f"chain{i}" for i in range(55))
    bridge = base.replace("chain10", "bridge10")
    report = build(
        [
            row("t1", base, source="publisher-a", family="family-a"),
            row("t2", bridge, source="mirror-b", family="family-b"),
        ],
        [row("e1", bridge.replace("chain30", "eval30"), source="eval", family="eval")],
    )
    assert len(report["excluded_records"]) == 2
    assert any(
        item.get("match_scope") == "training_cross_source_mirror"
        for item in report["match_evidence"]
    )


def test_code_copy_detects_identifier_comment_and_literal_changes():
    train = row(
        "t1",
        "def compute(alpha, beta):\n    # local note\n    total = alpha + beta\n    if total > 10:\n        return total * 3\n    return total - 2\n",
        source="repo-a",
        family="repo-a",
        modality="code",
    )
    evaluation = row(
        "e1",
        "def calculate(left, right):\n    # different note\n    result = left + right\n    if result > 999:\n        return result * 44\n    return result - 88\n",
        source="repo-eval",
        family="repo-eval",
        modality="code",
    )
    report = build([train], [evaluation], thresholds={"code_copy_jaccard": 0.70})
    assert any(item["match_type"] == "code_fork_copy" for item in report["match_evidence"])


def test_family_quarantine_on_contaminated_cross_source_match():
    base = " ".join(f"family{i}" for i in range(50))
    report = build_report(
        [
            row("t1", base, source="source-a", family="family-a"),
            row("t2", "independent clean sibling", source="source-a", family="family-a"),
        ],
        [row("e1", base.replace("family10", "edit10"), source="eval", family="eval")],
        training_corpus_identity="3" * 64,
        selection_validation_identity="4" * 64,
        final_test_identity="5" * 64,
        authorities=authorities(),
        quarantine_cross_source_families=True,
    )
    assert len(report["excluded_records"]) == 2
    assert report["counts"]["quarantined_source_families"] == 1


def test_final_test_outcome_metadata_is_refused():
    bad = authorities()
    bad["authorities"][0]["accuracy"] = 0.9
    with pytest.raises(DecontaminationError, match="outcome-bearing"):
        validate_authority_metadata(bad)


def test_report_is_hash_only_and_immutable(tmp_path: Path):
    report = build(
        [row("t1", "clean training text with unrelated material")],
        [row("e1", "reserved final test sentence is distinct", source="eval", family="eval")],
    )
    verify_report(report)
    serialized = json.dumps(report)
    assert "clean training text" not in serialized
    assert "reserved final test sentence" not in serialized
    assert '"t1"' not in serialized
    assert '"e1"' not in serialized
    path = tmp_path / "report.json"
    write_immutable_report(path, report)
    write_immutable_report(path, report)
    changed = dict(report)
    changed["status"] = "tampered"
    with pytest.raises(DecontaminationError):
        write_immutable_report(path, changed)


def test_blocker_binds_reserved_final_test_without_inventing_data230():
    report = build_blocker_report(authorities(), reason="DATA-230 absent")
    verify_report(report)
    assert report["status"] == "BLOCKED_MISSING_DATA230"
    assert report["training_corpus_identity"] is None
    assert report["selection_validation_identity"] is None
    assert isinstance(report["final_test_identity"], str)
    assert report["training_executed"] is False


def test_committed_adversarial_fixture_matrix():
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/data232_decontamination_adversarial_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert fixture["schema"] == "12-6.data232-adversarial-fixtures.v1"
    for case in fixture["cases"]:
        report = build(
            [case["train"]],
            [case["evaluation"]],
            thresholds=case.get("thresholds"),
        )
        assert any(
            item["match_type"] == case["expected_match_type"]
            for item in report["match_evidence"]
        ), case["case_id"]
