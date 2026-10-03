from __future__ import annotations

import copy
import hashlib
import inspect
import json
from pathlib import Path

import pytest

from twelve_six.data import expanded_global_dedup_v10 as v10

ROOT = Path(__file__).resolve().parents[1]


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

def _clean_fixture(
    monkeypatch: pytest.MonkeyPatch,
    *,
    source_id: str = "physical-source-1",
) -> tuple[bytes, bytes, str]:
    record_id = "clean-record-1"
    text = "independently qualified clean retained text"
    payload = text.encode("utf-8")
    projection = [
        {
            "record_id": record_id,
            "source_id": source_id,
            "source_family": "clean.fixture",
            "modality": "en",
            "text_sha256": hashlib.sha256(payload).hexdigest(),
            "text_utf8_bytes": len(payload),
        }
    ]
    record = {
        "record_id": record_id,
        "source_id": source_id,
        "source_family": "clean.fixture",
        "modality": "en",
        "text": text,
    }
    records_raw = _canonical_line(record)
    handoff_core = {
        "schema_version": v10.CLEAN_HANDOFF_SCHEMA,
        "postdedup_inventory_identity_sha256": v10.CLEAN_MATERIALIZATION_IDENTITY_SHA256,
        "input_survivor_authority_sha256": v10.CLEAN_COMPOSITION_PREFLIGHT_SHA256,
        "retained_source_count": 1,
        "matcher_input_projection": projection,
        "matcher_input_projection_sha256": hashlib.sha256(
            v10._canonical(projection)
        ).hexdigest(),
        "raw_text_persisted_in_evidence": False,
        "final_test_payload_accessed": False,
        "final_test_outcomes_accessed": False,
        "authorized_training_exposure": 0,
    }
    handoff = {
        **handoff_core,
        "handoff_identity_sha256": hashlib.sha256(
            v10._canonical(handoff_core)
        ).hexdigest(),
    }
    handoff_raw = v10._canonical(handoff, newline=True)

    monkeypatch.setattr(v10, "CLEAN_TRAINING_RECORDS_BYTES", len(records_raw))
    monkeypatch.setattr(
        v10,
        "CLEAN_TRAINING_RECORDS_SHA256",
        hashlib.sha256(records_raw).hexdigest(),
    )
    monkeypatch.setattr(
        v10,
        "CLEAN_TRAINING_HANDOFF_SHA256",
        hashlib.sha256(handoff_raw).hexdigest(),
    )
    monkeypatch.setattr(v10, "CLEAN_RETAINED_SOURCE_COUNT", 1)
    monkeypatch.setattr(v10, "CLEAN_DISTINCT_PHYSICAL_SOURCE_COUNT", 1)
    monkeypatch.setattr(v10, "CLEAN_RETAINED_PAYLOAD_BYTES", len(payload))
    return records_raw, handoff_raw, record_id


def test_clean_retained_outputs_crossbind_records_projection_and_physical_origin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records_raw, handoff_raw, record_id = _clean_fixture(monkeypatch)

    inventory, payloads, handoff = v10._validate_clean_retained(
        records_raw,
        handoff_raw,
    )

    assert len(inventory) == 1
    row = inventory[0]
    assert row["source_id"] == f"clean-retained:{record_id}"
    assert row["stable_origin_id"] == "clean-data232-source:physical-source-1"
    assert row["origin_key"] == f"clean-data232-record:{record_id}"
    assert row["authority_ref"] == f"PR2183:{v10.CLEAN_PRODUCT_HEAD}"
    assert payloads[row["source_id"]] == b"independently qualified clean retained text"
    assert handoff["authorized_training_exposure"] == 0


def test_clean_retained_projection_cannot_be_coherently_resealed_to_other_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records_raw, handoff_raw, _ = _clean_fixture(monkeypatch)
    handoff = json.loads(handoff_raw)
    handoff["matcher_input_projection"][0]["source_id"] = "other-source"
    handoff["matcher_input_projection_sha256"] = hashlib.sha256(
        v10._canonical(handoff["matcher_input_projection"])
    ).hexdigest()
    core = dict(handoff)
    core.pop("handoff_identity_sha256")
    handoff["handoff_identity_sha256"] = hashlib.sha256(
        v10._canonical(core)
    ).hexdigest()
    tampered_raw = v10._canonical(handoff, newline=True)
    monkeypatch.setattr(
        v10,
        "CLEAN_TRAINING_HANDOFF_SHA256",
        hashlib.sha256(tampered_raw).hexdigest(),
    )

    with pytest.raises(v10.ExpandedDedupV10Error, match="row/projection source_id drift"):
        v10._validate_clean_retained(records_raw, tampered_raw)


def test_v10_execution_has_no_caller_matcher_or_private_v9_execution_seam() -> None:
    parameters = inspect.signature(v10.run_expanded_dedup_v10).parameters
    assert "matcher_audit" not in parameters
    assert "matcher_verify" not in parameters
    assert "reconstructed_v8_inventory" not in parameters
    assert "data526_evidence" not in parameters
    source = inspect.getsource(v10.run_expanded_dedup_v10)
    assert "v9." not in source
    assert "run_expanded_dedup(" not in source
    assert "_FROZEN_INDEXED_AUDIT(" in source
    assert not hasattr(v10, "v9")


def test_indexed_executor_is_code_bound_and_frozen_at_import() -> None:
    assert v10.INDEXED_EXECUTOR_GIT_BLOB_SHA1 == (
        "f75336008839198b6d46bea4954f120e2d81613c"
    )
    assert v10.INDEXED_CORE_GIT_BLOB_SHA1 == (
        "af7be7909501ea9d76604ebed084cec32fbd9456"
    )
    assert v10.indexed.audit_payloads_indexed is v10._FROZEN_INDEXED_AUDIT
    assert v10.indexed.attest_incumbent_runtime is v10._FROZEN_INDEXED_ATTEST


def test_local_survivor_projection_preserves_terminal_selection_rule() -> None:
    dedup = {
        "report_sha256": "a" * 64,
        "sources": [
            {"source_id": "a", "declared_capacity_bytes": 10},
            {"source_id": "b", "declared_capacity_bytes": 20},
            {"source_id": "c", "declared_capacity_bytes": 5},
        ],
        "terminal_candidates": {
            "declared_capacity_bytes_before": 35,
            "conservative_unique_capacity_bytes_after": 25,
            "duplicate_discount_bytes": 10,
            "duplicate_cluster_count": 1,
            "duplicate_clusters": [["a", "b"]],
        },
    }

    projection = v10._derive_selection_projection(dedup)

    assert projection["selection_rule"] == v10.SELECTION_RULE
    assert projection["survivor_source_ids"] == ["b", "c"]
    assert projection["post_dedup_declared_capacity_bytes"] == 25
    assert projection["duplicate_clusters"] == [
        {
            "member_source_ids": ["a", "b"],
            "selected_source_id": "b",
            "selected_declared_capacity_bytes": 20,
        }
    ]
    assert projection["truth_boundary"]["training_authorized_bytes"] == 0


def test_runner_consumes_clean_handoff_and_drops_historical_v8_data526_inputs() -> None:
    source = (ROOT / "tools/run_d03_expanded_global_dedup_v10_pep_loc.py").read_text(
        encoding="utf-8"
    )
    assert "--clean-training-records" in source
    assert "--clean-training-handoff" in source
    assert "--data526-evidence" not in source
    assert "--data526-record-inventory" not in source
    assert "--v8-report" not in source
    assert "--v8-survivors" not in source
    assert "--rada-quality-privacy-jsonl" not in source



def _load_v10_runner():
    import importlib.util

    path = ROOT / "tools/run_d03_expanded_global_dedup_v10_pep_loc.py"
    spec = importlib.util.spec_from_file_location("_test_v10_publication_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v10_publication_create_only_two_outputs(tmp_path: Path) -> None:
    runner = _load_v10_runner()
    report = tmp_path / "report.json"
    survivors = tmp_path / "survivors.json"
    runner.publish_outputs(report, survivors, {"report_sha256": "a" * 64}, {"ok": True})
    assert json.loads(report.read_text(encoding="utf-8")) == {
        "report_sha256": "a" * 64,
    }
    assert json.loads(survivors.read_text(encoding="utf-8")) == {"ok": True}
    with pytest.raises(v10.ExpandedDedupV10Error, match="refusing to overwrite"):
        runner.publish_outputs(report, survivors, {"another": 1}, {"another": 2})


def test_v10_publication_rejects_existing_second_without_creating_first(
    tmp_path: Path,
) -> None:
    runner = _load_v10_runner()
    report = tmp_path / "report.json"
    survivors = tmp_path / "survivors.json"
    survivors.write_bytes(b"existing untouched bytes")
    with pytest.raises(v10.ExpandedDedupV10Error, match="refusing to overwrite"):
        runner.publish_outputs(report, survivors, {"a": 1}, {"b": 2})
    assert not report.exists()
    assert survivors.read_bytes() == b"existing untouched bytes"


def test_v10_publication_rejects_alias_before_creating_output(tmp_path: Path) -> None:
    runner = _load_v10_runner()
    output = tmp_path / "one.json"
    with pytest.raises(v10.ExpandedDedupV10Error, match="must be distinct"):
        runner.publish_outputs(output, output, {"a": 1}, {"b": 2})
    assert not output.exists()


def test_v10_publication_rejects_nonfinite_second_before_first(
    tmp_path: Path,
) -> None:
    runner = _load_v10_runner()
    report = tmp_path / "report.json"
    survivors = tmp_path / "survivors.json"
    with pytest.raises(v10.ExpandedDedupV10Error, match="finite/serializable"):
        runner.publish_outputs(report, survivors, {"a": 1}, {"nested": float("nan")})
    assert not report.exists()
    assert not survivors.exists()


def test_v10_publication_rolls_back_first_on_second_creation_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    runner = _load_v10_runner()
    report = tmp_path / "report.json"
    survivors = tmp_path / "survivors.json"
    real_write = runner.write_json

    def injected_second_failure(path: Path, payload: bytes) -> tuple[int, int]:
        if path == survivors:
            raise PermissionError("injected survivors creation failure")
        return real_write(path, payload)

    monkeypatch.setattr(runner, "write_json", injected_second_failure)
    with pytest.raises(PermissionError, match="injected survivors"):
        runner.publish_outputs(report, survivors, {"a": 1}, {"b": 2})
    assert not report.exists()
    assert not survivors.exists()


def test_v10_publication_never_unlinks_foreign_replacement_on_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    runner = _load_v10_runner()
    report = tmp_path / "report.json"
    survivors = tmp_path / "survivors.json"
    real_write = runner.write_json

    def replace_before_failure(path: Path, payload: bytes) -> tuple[int, int]:
        if path == survivors:
            foreign = tmp_path / "foreign.json"
            foreign.write_bytes(b"foreign replacement remains")
            foreign.replace(report)
            raise PermissionError("injected second-output failure")
        return real_write(path, payload)

    monkeypatch.setattr(runner, "write_json", replace_before_failure)
    with pytest.raises(PermissionError, match="injected second-output"):
        runner.publish_outputs(report, survivors, {"a": 1}, {"b": 2})
    assert report.read_bytes() == b"foreign replacement remains"
    assert not survivors.exists()


def test_v10_input_reader_rejects_oversized_file_before_loading(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    runner = _load_v10_runner()
    monkeypatch.setattr(runner, "MAX_INPUT_BYTES", 8)
    oversized = tmp_path / "oversized.jsonl"
    oversized.write_bytes(b"x" * 9)
    with pytest.raises(v10.ExpandedDedupV10Error, match="bounded input limit"):
        runner.read_exact_bytes(oversized, label="oversized input")


def test_v10_input_reader_accepts_exact_limit_and_rejects_empty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    runner = _load_v10_runner()
    monkeypatch.setattr(runner, "MAX_INPUT_BYTES", 8)
    exact = tmp_path / "exact.bin"
    exact.write_bytes(b"12345678")
    assert runner.read_exact_bytes(exact, label="exact input") == b"12345678"
    empty = tmp_path / "empty.bin"
    empty.write_bytes(b"")
    with pytest.raises(v10.ExpandedDedupV10Error, match="empty"):
        runner.read_exact_bytes(empty, label="empty input")


def test_v10_input_reader_rejects_symlink_before_loading(
    tmp_path: Path,
) -> None:
    runner = _load_v10_runner()
    real = tmp_path / "real.bin"
    real.write_bytes(b"valid bytes")
    link = tmp_path / "input.bin"
    try:
        link.symlink_to(real)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    with pytest.raises(v10.ExpandedDedupV10Error, match="regular file"):
        runner.read_exact_bytes(link, label="symlink input")
