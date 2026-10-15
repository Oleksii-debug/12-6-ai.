from __future__ import annotations

import copy
import hashlib

import pytest

from tools.plan2_s16_structural_preflight_v1 import (
    Section16PreflightDenied,
    inspect_structure,
)


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def cohort():
    rows = []
    for split in ("train", "validation", "test"):
        for language in ("uk", "en"):
            index = f"{language}-{split}"
            rows.append({
                "record_id": index,
                "source_id": "src." + index,
                "source_family": "family." + index,
                "document_family": "document." + index,
                "source_revision": "pinned.v1",
                "language": language,
                "split": split,
                "payload_sha256": digest(index + " original"),
                "normalized_sha256": digest(index + " normalized"),
                "rights_receipt_sha256": digest(index + " rights"),
                "privacy_receipt_sha256": digest(index + " privacy"),
            })
    return rows


def test_structural_candidate_stays_non_authorizing_and_is_order_independent():
    records = cohort()
    a = inspect_structure(records)
    assert a == inspect_structure(list(reversed(records)))
    assert a["physical_record_count"] == 6
    assert a["split_language_family_counts"]["test"] == {"uk": 1, "en": 1}
    assert all(a[key] is False for key in (
        "training_corpus_authorized", "evaluation_authorized", "tokenizer_fit_authorized",
        "production_release_authorized", "rights_independently_verified",
        "privacy_independently_verified", "decontamination_independently_verified",
    ))
    assert "STRUCTURAL_CANDIDATE_ONLY" in a["status"]


@pytest.mark.parametrize(("field", "value"), [
    ("rights_receipt_sha256", "self_issued"),
    ("privacy_receipt_sha256", ""),
    ("source_revision", "../latest"),
    ("language", "sk"),
    ("split", "unknown"),
])
def test_invalid_identity_and_forged_receipt_denied(field, value):
    rows = cohort()
    rows[0][field] = value
    with pytest.raises(Section16PreflightDenied):
        inspect_structure(rows)


def test_self_declared_training_license_cannot_promote_record():
    rows = cohort()
    rows[0]["training_corpus_authorized"] = True
    with pytest.raises(Section16PreflightDenied, match="unsupported grant"):
        inspect_structure(rows)


def test_missing_ukrainian_evaluation_is_denied():
    rows = [r for r in cohort() if not (
        r["language"] == "uk" and r["split"] == "test"
    )]
    with pytest.raises(Section16PreflightDenied, match="missing uk test"):
        inspect_structure(rows)


def test_exact_or_normalized_holdout_leakage_denied():
    for field in ("payload_sha256", "normalized_sha256"):
        rows = cohort()
        rows[-1][field] = rows[0][field]
        with pytest.raises(Section16PreflightDenied, match="duplicate"):
            inspect_structure(rows)


def test_external_reserved_eval_collision_denied():
    rows = cohort()
    with pytest.raises(Section16PreflightDenied, match="reserved evaluation collision"):
        inspect_structure(rows, reserved_payload_sha256=[rows[0]["payload_sha256"]])


@pytest.mark.parametrize("family", ["source_family", "document_family"])
def test_cluster_leakage_denied(family):
    rows = cohort()
    rows[2][family] = rows[0][family]
    with pytest.raises(Section16PreflightDenied, match="leaked"):
        inspect_structure(rows)


def test_source_revision_or_language_drift_denied():
    rows = cohort()
    rows[2]["source_id"] = rows[0]["source_id"]
    with pytest.raises(Section16PreflightDenied, match="drift"):
        inspect_structure(rows)


def test_out_of_bound_language_distribution_denied():
    rows = cohort()
    for i in range(5):
        extra = copy.deepcopy(rows[0])
        extra["record_id"] += f".{i}"
        extra["payload_sha256"] = digest(f"extra payload {i}")
        extra["normalized_sha256"] = digest(f"extra normalized {i}")
        rows.append(extra)
    with pytest.raises(Section16PreflightDenied, match="out-of-bound"):
        inspect_structure(rows)


def test_no_raw_corpus_text_in_structural_receipt():
    a = inspect_structure(cohort())
    assert not any("text" in key or "payload" in key for key in a)
    assert len(a["structural_candidate_sha256"]) == 64


def test_explicitly_revoked_source_is_denied():
    rows = cohort()
    with pytest.raises(Section16PreflightDenied, match="revoked"):
        inspect_structure(rows, revoked_source_ids=[rows[0]["source_id"]])


def test_corpus_family_cannot_impersonate_two_languages():
    rows = cohort()
    rows[1]["source_family"] = rows[0]["source_family"]
    with pytest.raises(Section16PreflightDenied, match="family language drift"):
        inspect_structure(rows)


@pytest.mark.parametrize("field", [
    "record_id", "source_id", "source_family", "document_family",
    "source_revision", "payload_sha256", "normalized_sha256",
    "rights_receipt_sha256", "privacy_receipt_sha256",
])
def test_receipt_binds_exact_validated_record_membership(field):
    original = cohort()
    changed = copy.deepcopy(original)
    if field.endswith("sha256"):
        changed[0][field] = digest("different and still well-formed " + field)
    else:
        changed[0][field] += ".revision2"
    before = inspect_structure(original)
    after = inspect_structure(changed)
    assert before["structural_candidate_sha256"] != after["structural_candidate_sha256"]


def test_receipt_binds_reserved_holdout_index_membership_not_just_count():
    original = cohort()
    before = inspect_structure(original, reserved_payload_sha256=[digest("holdout A")])
    after = inspect_structure(original, reserved_payload_sha256=[digest("holdout B")])
    assert before["structural_candidate_sha256"] != after["structural_candidate_sha256"]


def test_receipt_binds_revocation_index_membership_not_just_count():
    original = cohort()
    before = inspect_structure(original, revoked_source_ids=["source.unused.A"])
    after = inspect_structure(original, revoked_source_ids=["source.unused.B"])
    assert before["structural_candidate_sha256"] != after["structural_candidate_sha256"]


def test_membership_binding_is_order_independent_for_all_three_inputs():
    rows = cohort()
    reserved = [digest("holdout A"), digest("holdout B")]
    revoked = ["source.unused.A", "source.unused.B"]
    a = inspect_structure(rows, reserved_payload_sha256=reserved, revoked_source_ids=revoked)
    b = inspect_structure(list(reversed(rows)), reserved_payload_sha256=list(reversed(reserved)),
                          revoked_source_ids=list(reversed(revoked)))
    assert a == b


def test_membership_digest_does_not_promote_candidate_to_training_or_release():
    receipt = inspect_structure(cohort())
    for field in ("rights_independently_verified", "privacy_independently_verified",
                  "decontamination_independently_verified", "training_corpus_authorized",
                  "evaluation_authorized", "tokenizer_fit_authorized",
                  "production_release_authorized"):
        assert receipt[field] is False
    assert len(receipt["physical_membership_sha256"]) == 64
    assert len(receipt["reserved_membership_sha256"]) == 64
    assert len(receipt["revoked_membership_sha256"]) == 64
