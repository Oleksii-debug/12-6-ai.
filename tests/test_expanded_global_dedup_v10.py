from __future__ import annotations

import copy
import hashlib
import json

import pytest

from twelve_six.data import expanded_global_dedup_v10 as v10


def _canonical_line(row: dict[str, object]) -> bytes:
    return (
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode()


def _pretty_report(report: dict[str, object]) -> bytes:
    return (
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode()


def _pep_row() -> dict[str, object]:
    text = "PEP example public-domain text."
    raw = text.encode()
    return {
        "record_id": "pep:peps/pep-9999.rst",
        "source_family": v10.PEP_FAMILY,
        "source_revision": v10.PEP_REVISION,
        "source_path": "peps/pep-9999.rst",
        "source_git_blob_sha1": "1" * 40,
        "text": text,
        "text_sha256": hashlib.sha256(raw).hexdigest(),
        "utf8_bytes": len(raw),
        "training_eligible": False,
    }


def _pep_report(row: dict[str, object]) -> dict[str, object]:
    inventory = [
        {
            "record_id": row["record_id"],
            "source_git_blob_sha1": row["source_git_blob_sha1"],
            "text_sha256": row["text_sha256"],
            "utf8_bytes": row["utf8_bytes"],
        }
    ]
    return {
        "schema_version": "12-6.d03-pep-public-domain-materialization.v1",
        "status": "CANDIDATE_MATERIALIZED_ZERO_CREDIT",
        "source_family": v10.PEP_FAMILY,
        "upstream_revision": v10.PEP_REVISION,
        "upstream_tree_sha1": v10.PEP_TREE_SHA1,
        "contract_identity_sha256": v10.PEP_CONTRACT_SHA256,
        "examined_documents": 1,
        "accepted_documents": 1,
        "accepted_normalized_utf8_bytes": row["utf8_bytes"],
        "candidate_inventory_identity_sha256": v10._inventory_identity(inventory),
        "rights_evidence_identity_sha256": "2" * 64,
        "known_opl_sentinels_verified_excluded": [],
        "training_authorized_bytes": 0,
        "authorized_unique_loss_positions": 0,
        "corpus_admitted": False,
        "privacy_gate": "NOT_RUN",
        "quality_gate": "NOT_RUN",
        "global_dedup_gate": "NOT_RUN",
        "reserved_evaluation_decontamination_gate": "NOT_RUN",
        "split_gate": "NOT_RUN",
        "packing_gate": "NOT_RUN",
        "model_training_executed": False,
        "paid_compute_used": False,
    }


def _loc_row() -> dict[str, object]:
    text = "Library of Congress public-domain text."
    raw = text.encode()
    return {
        "record_id": "loc:item-1",
        "source_family": v10.LOC_FAMILY,
        "source_revision": v10.LOC_REVISION,
        "source_shard_path": v10.LOC_SHARD_PATH,
        "source_shard_lfs_sha256": v10.LOC_SHARD_SHA256,
        "source_record_id": "item-1",
        "item_url": "https://www.loc.gov/item/item-1/",
        "text_file_url": "https://www.loc.gov/resource/item-1.txt",
        "text": text,
        "text_sha256": hashlib.sha256(raw).hexdigest(),
        "utf8_bytes": len(raw),
        "training_eligible": False,
        "evaluation_eligible": False,
    }


def _loc_report(row: dict[str, object]) -> dict[str, object]:
    inventory = [
        {
            "record_id": row["record_id"],
            "source_record_id": row["source_record_id"],
            "item_url": row["item_url"],
            "text_file_url": row["text_file_url"],
            "text_sha256": row["text_sha256"],
            "utf8_bytes": row["utf8_bytes"],
        }
    ]
    return {
        "schema_version": "12-6.d03-common-pile-loc-materialization.v1",
        "status": "CANDIDATE_MATERIALIZED_ZERO_CREDIT",
        "source_family": v10.LOC_FAMILY,
        "contract_identity_sha256": v10.LOC_CONTRACT_SHA256,
        "upstream_revision": v10.LOC_REVISION,
        "source_shard_path": v10.LOC_SHARD_PATH,
        "source_shard_lfs_sha256": v10.LOC_SHARD_SHA256,
        "full_shard_hash_verified": True,
        "examined_documents": 1,
        "accepted_documents": 1,
        "accepted_normalized_utf8_bytes": row["utf8_bytes"],
        "rejection_counts": {},
        "candidate_inventory_identity_sha256": v10._inventory_identity(inventory),
        "rights_provenance_evidence_identity_sha256": "3" * 64,
        "training_authorized_bytes": 0,
        "authorized_unique_loss_positions": 0,
        "canonical_capacity_credit_bytes": 0,
        "corpus_admitted": False,
        "evaluation_eligible": False,
        "privacy_gate": "NOT_RUN",
        "quality_gate": "NOT_RUN",
        "global_dedup_gate": "NOT_RUN",
        "reserved_evaluation_decontamination_gate": "NOT_RUN",
        "balance_family_caps_gate": "NOT_RUN",
        "cluster_safe_split_gate": "NOT_RUN",
        "packing_two_clean_builds_gate": "NOT_RUN",
        "positive_unique_loss_ledger_gate": "NOT_RUN",
        "model_training_executed": False,
        "final_test_payload_accessed": False,
        "paid_compute_used": False,
    }


def _patch_pep_fixture(
    monkeypatch: pytest.MonkeyPatch,
    candidate_raw: bytes,
    report_raw: bytes,
    report: dict[str, object],
) -> None:
    monkeypatch.setattr(v10, "PEP_CANDIDATE_BYTES", len(candidate_raw))
    monkeypatch.setattr(
        v10,
        "PEP_CANDIDATE_SHA256",
        hashlib.sha256(candidate_raw).hexdigest(),
    )
    monkeypatch.setattr(v10, "PEP_REPORT_BYTES", len(report_raw))
    monkeypatch.setattr(
        v10,
        "PEP_REPORT_SHA256",
        hashlib.sha256(report_raw).hexdigest(),
    )
    monkeypatch.setattr(v10, "PEP_ACCEPTED_RECORDS", report["accepted_documents"])
    monkeypatch.setattr(
        v10,
        "PEP_ACCEPTED_UTF8_BYTES",
        report["accepted_normalized_utf8_bytes"],
    )
    monkeypatch.setattr(
        v10,
        "PEP_INVENTORY_SHA256",
        report["candidate_inventory_identity_sha256"],
    )


def _patch_loc_fixture(
    monkeypatch: pytest.MonkeyPatch,
    candidate_raw: bytes,
    report_raw: bytes,
    report: dict[str, object],
) -> None:
    monkeypatch.setattr(v10, "LOC_CANDIDATE_BYTES", len(candidate_raw))
    monkeypatch.setattr(
        v10,
        "LOC_CANDIDATE_SHA256",
        hashlib.sha256(candidate_raw).hexdigest(),
    )
    monkeypatch.setattr(v10, "LOC_REPORT_BYTES", len(report_raw))
    monkeypatch.setattr(
        v10,
        "LOC_REPORT_SHA256",
        hashlib.sha256(report_raw).hexdigest(),
    )
    monkeypatch.setattr(v10, "LOC_ACCEPTED_RECORDS", report["accepted_documents"])
    monkeypatch.setattr(
        v10,
        "LOC_ACCEPTED_UTF8_BYTES",
        report["accepted_normalized_utf8_bytes"],
    )
    monkeypatch.setattr(
        v10,
        "LOC_INVENTORY_SHA256",
        report["candidate_inventory_identity_sha256"],
    )


def test_real_execution_authority_constants_are_locked() -> None:
    assert v10.PEP_CANDIDATE_SHA256 == (
        "65d1cf23354efc5a66d7132b67602a7af1715e18a0fb77e729595866c25d33c3"
    )
    assert v10.PEP_ACCEPTED_UTF8_BYTES == 4_987_775
    assert v10.PEP_EXECUTION_RUN == 34570806209
    assert v10.LOC_CANDIDATE_SHA256 == (
        "f87201d71eb8c3f2510cdf2c9b3ce1a32b83d1a11198e8a1ff0ad67d0a3fec20"
    )
    assert v10.LOC_ACCEPTED_UTF8_BYTES == 4_499_908
    assert v10.LOC_EXECUTION_RUN == 34606219695


def test_pep_materialization_crossbinds_candidate_and_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = _pep_row()
    report = _pep_report(row)
    candidate_raw = _canonical_line(row)
    report_raw = _pretty_report(report)
    _patch_pep_fixture(monkeypatch, candidate_raw, report_raw, report)

    rows, observed = v10._validate_pep(candidate_raw, report_raw)

    assert rows == [row]
    assert observed == report


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("training_eligible", 0, "preclaims training"),
        ("utf8_bytes", True, "UTF-8 bytes drift"),
        ("source_family", "other", "family drift"),
    ],
)
def test_pep_candidate_rejects_type_alias_and_authority_drift(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
    error: str,
) -> None:
    row = _pep_row()
    row[field] = value
    report = _pep_report(_pep_row())
    candidate_raw = _canonical_line(row)
    report_raw = _pretty_report(report)
    _patch_pep_fixture(monkeypatch, candidate_raw, report_raw, report)

    with pytest.raises(v10.ExpandedDedupV10Error, match=error):
        v10._validate_pep(candidate_raw, report_raw)


def test_pep_candidate_rejects_unknown_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = _pep_row()
    row["model_training_permitted"] = False
    report = _pep_report(_pep_row())
    candidate_raw = _canonical_line(row)
    report_raw = _pretty_report(report)
    _patch_pep_fixture(monkeypatch, candidate_raw, report_raw, report)

    with pytest.raises(v10.ExpandedDedupV10Error, match="row keyset drift"):
        v10._validate_pep(candidate_raw, report_raw)


def test_loc_materialization_crossbinds_candidate_and_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = _loc_row()
    report = _loc_report(row)
    candidate_raw = _canonical_line(row)
    report_raw = _pretty_report(report)
    _patch_loc_fixture(monkeypatch, candidate_raw, report_raw, report)

    rows, observed = v10._validate_loc(candidate_raw, report_raw)

    assert rows == [row]
    assert observed == report


def test_loc_rejects_bool_alias_for_evaluation_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = _loc_row()
    row["evaluation_eligible"] = 0
    pristine = _loc_row()
    report = _loc_report(pristine)
    candidate_raw = _canonical_line(row)
    report_raw = _pretty_report(report)
    _patch_loc_fixture(monkeypatch, candidate_raw, report_raw, report)

    with pytest.raises(v10.ExpandedDedupV10Error, match="preclaims evaluation"):
        v10._validate_loc(candidate_raw, report_raw)


def test_json_parser_rejects_duplicate_keys_even_when_last_value_is_safe() -> None:
    raw = b'{"training_eligible":true,"training_eligible":false}\n'

    with pytest.raises(v10.ExpandedDedupV10Error, match="duplicate JSON key"):
        v10._load_jsonl(raw, label="candidate")


def test_raw_identity_is_checked_before_semantic_parsing() -> None:
    with pytest.raises(v10.ExpandedDedupV10Error, match="SHA-256 drift"):
        v10._require_exact_raw(
            b"safe-looking",
            label="candidate",
            expected_bytes=len(b"safe-looking"),
            expected_sha256="0" * 64,
        )


def test_matcher_inputs_remain_zero_credit_metadata_only() -> None:
    pep_row = _pep_row()
    loc_row = _loc_row()

    pep_inventory, pep_payloads = v10._pep_matcher_inputs([pep_row])
    loc_inventory, loc_payloads = v10._loc_matcher_inputs([loc_row])

    assert pep_inventory[0]["source_family"] == v10.PEP_FAMILY
    assert pep_inventory[0]["authority_ref"] == f"PR1231:{v10.PEP_EVIDENCE_HEAD}"
    assert loc_inventory[0]["source_family"] == v10.LOC_FAMILY
    assert loc_inventory[0]["authority_ref"] == f"PR1276:{v10.LOC_EVIDENCE_HEAD}"
    assert pep_payloads[pep_inventory[0]["source_id"]] == pep_row["text"].encode()
    assert loc_payloads[loc_inventory[0]["source_id"]] == loc_row["text"].encode()
    assert "training_eligible" not in pep_inventory[0]
    assert "training_eligible" not in loc_inventory[0]


def test_outer_survivor_authority_cannot_promote_training() -> None:
    dedup = {"report_sha256": "4" * 64}
    projection = {
        "schema_version": "12-6.d03-expanded-global-dedup-v9-survivors.v1",
        "survivor_authority_sha256": "5" * 64,
        "pre_dedup_source_object_count": 4,
        "post_dedup_survivor_source_object_count": 3,
        "pre_dedup_declared_capacity_bytes": 100,
        "post_dedup_declared_capacity_bytes": 90,
        "duplicate_discount_bytes": 10,
        "duplicate_cluster_count": 1,
        "duplicate_clusters": [
            {
                "member_source_ids": ["a", "b"],
                "selected_source_id": "a",
                "selected_declared_capacity_bytes": 20,
            }
        ],
        "survivor_source_ids": ["a", "c", "d"],
    }

    authority = v10._outer_survivor_authority(dedup, copy.deepcopy(projection))
    boundary = authority["truth_boundary"]

    assert boundary["global_dedup_execution_complete"] is True
    assert boundary["training_authorized_bytes"] == 0
    assert boundary["unique_causal_loss_positions_authorized"] == 0
    assert boundary["tokenizer_fit_authorized"] is False
    assert boundary["optimizer_updates"] == 0
    assert boundary["model_training_executed"] is False
