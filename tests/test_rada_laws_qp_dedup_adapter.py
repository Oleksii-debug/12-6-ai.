from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from twelve_six.data import rada_laws_qp_dedup_adapter as adapter


def _row(
    *,
    parent: str = "d100.htm",
    chunk: int = 3,
    text: str = "Законодавчий український текст для перевірки адаптера.",
    encoding: str = "utf-8",
) -> tuple[dict[str, object], dict[str, object]]:
    payload = text.encode("utf-8")
    record_id = f"{parent}.q{chunk:05d}"
    row: dict[str, object] = {
        "record_id": record_id,
        "parent_record_id": parent,
        "source_path": f"zak/perv/text/{parent}",
        "source_encoding": encoding,
        "chunk_index": chunk,
        "normalized_bytes": len(payload),
        "normalized_sha256": hashlib.sha256(payload).hexdigest(),
        "text": text,
    }
    metadata = {key: value for key, value in row.items() if key != "text"}
    return row, metadata


def _exact_evidence() -> dict[str, object]:
    return {
        "environment": {
            "platform": "Linux-6.17.0-1022-azure-x86_64-with-glibc2.39",
            "python": "3.11.16",
            "runner_arch": "X64",
            "runner_os": "Linux",
        },
        "evidence_identity_sha256": adapter.EXECUTION_EVIDENCE_IDENTITY_SHA256,
        "execution_class": "LOCAL_FREE_GITHUB_HOSTED_ACTIONS",
        "observed_quality_privacy": {
            "accepted_bytes_observed_not_credited": adapter.ACCEPTED_TEXT_UTF8_BYTES,
            "accepted_chunk_count": adapter.ACCEPTED_CHUNK_COUNT,
            "accepted_inventory_sha256": adapter.ACCEPTED_INVENTORY_SHA256,
            "accepted_jsonl_file_bytes": adapter.ACCEPTED_JSONL_FILE_BYTES,
            "accepted_jsonl_sha256": adapter.ACCEPTED_JSONL_SHA256,
            "accepted_source_encoding_counts": adapter.ACCEPTED_SOURCE_ENCODING_COUNTS,
            "exact_duplicate_accepted_hashes_observed_not_removed": (
                adapter.EXACT_DUPLICATE_ACCEPTED_HASHES
            ),
            "parent_normalized_bytes": 211176449,
            "parent_record_count": 3052,
            "quality_report_file_sha256": adapter.QUALITY_REPORT_FILE_SHA256,
            "quality_report_identity_sha256": adapter.QUALITY_REPORT_IDENTITY_SHA256,
            "rejected_chunk_count": 11519,
            "rejection_reasons": {
                "low_alpha_ratio": 3599,
                "pii_email": 1,
                "pii_phone": 7919,
            },
            "total_chunks": 113078,
        },
        "parent_authority": {
            "final_head_sha": "656dd4abe7bdf9c379a86ac9a19f046d5b0d8538",
            "manifest_file_sha256": (
                "8923d26024ba396db14e6afeb367a5d3c54de573019252083aec88f2082bb119"
            ),
            "manifest_identity_sha256": (
                "ee1c59dbc481b83bffb1380f751880e5fa6e64d654b5abad266c3e2c4d18293b"
            ),
            "normalized_bytes_observed_not_credited": 211176449,
            "normalized_jsonl_sha256": (
                "b46baa0f1c5087f4a9772ee273da459e87c0e82111dd5e51b22fc0b85cde840e"
            ),
            "normalized_record_inventory_sha256": (
                "bd790b51809018950fcf8d9919f62920ab79aee64f8d73e131cbe16772554495"
            ),
            "pr": 864,
            "real_execution_evidence_identity_sha256": (
                "e1633070beaff596ae11f4974bd9662785723618f502f382850f7e3bea27ab6a"
            ),
            "real_execution_head_sha": "f62670084f80041757e162743356ac16e0fd81a7",
            "real_execution_run_id": 34565921713,
            "record_count": 3052,
        },
        "repair_execution_head_sha": adapter.UPSTREAM_REPAIR_EXECUTION_HEAD,
        "safe_result": adapter.EXECUTION_SAFE_RESULT,
        "schema_version": adapter.EXECUTION_EVIDENCE_SCHEMA,
        "source_authority": {
            "archive_bytes": 46709153,
            "archive_sha256": adapter.SOURCE_ARCHIVE_SHA256,
            "entry_identity_sha256": (
                "3939c8a407a910222173bba5a2df1b9c7d2edce1503c13b31a9c293d61c3a603"
            ),
            "observation_report_sha256": (
                "e2ad0d8a4fce01d2da03fc93d636354253beaaf563db43968b11b600e26d0cdf"
            ),
            "observation_run_id": 34535608294,
        },
        "truth_boundary": {
            "authorized_optimized_target_exposure": 0,
            "canonical_capacity_credited": 0,
            "current_corpus_eligible": False,
            "external_llm_api_data_or_intelligence": False,
            "final_test_accessed": False,
            "foreign_pretrained_weights": False,
            "learned_weights_created": False,
            "model_training_executed": False,
            "optimizer_updates_executed_on_real_targets": 0,
            "paid_compute_used": False,
            "tokenizer_fit_authorized": False,
            "training_authorized_bytes": 0,
        },
        "two_clean_reproducibility": {
            "manifests_identical": True,
            "normalized_jsonl_identical": True,
            "quality_accepted_jsonl_identical": True,
            "quality_reports_identical": True,
            "source_archives_identical": True,
        },
        "workflow_run_attempt": adapter.UPSTREAM_WORKFLOW_RUN_ATTEMPT,
        "workflow_run_id": adapter.UPSTREAM_WORKFLOW_RUN_ID,
    }


def test_live_authority_constants_lock_real_rada_execution() -> None:
    assert adapter.UPSTREAM_FINAL_EVIDENCE_COMMIT == (
        "43a7d62854255da4426d6a3dc4d5ad9b1e4898e9"
    )
    assert adapter.ACCEPTED_CHUNK_COUNT == 101559
    assert adapter.ACCEPTED_TEXT_UTF8_BYTES == 192078166
    assert adapter.ACCEPTED_JSONL_FILE_BYTES == 224528099
    assert adapter.EXACT_DUPLICATE_ACCEPTED_HASHES == 764
    assert adapter.DIRECT_RADA_ALL_PAIR_DISPATCHES == 5157064461


def test_exact_execution_evidence_self_hash_and_truth_boundary_validate() -> None:
    evidence = _exact_evidence()
    unsigned = copy.deepcopy(evidence)
    unsigned.pop("evidence_identity_sha256")
    assert hashlib.sha256(adapter._canonical(unsigned)).hexdigest() == (
        adapter.EXECUTION_EVIDENCE_IDENTITY_SHA256
    )
    adapter._validate_execution_evidence(evidence)


def test_strict_json_rejects_duplicate_keys() -> None:
    with pytest.raises(adapter.RadaLawsDedupIntakeError, match="duplicate JSON key"):
        adapter._strict_json_bytes(b'{"a":1,"a":2}', label="fixture")


def test_row_projects_one_to_one_with_chunk_specific_origin_key() -> None:
    row, metadata = _row()
    matcher, source_id, payload = adapter._validate_project_row(
        row,
        metadata,
        line_number=1,
    )
    assert source_id == f"rada-laws-qp:{row['record_id']}"
    assert matcher["source_family"] == adapter.SOURCE_FAMILY
    assert matcher["modality"] == "uk"
    assert matcher["stable_origin_id"] == f"rada-laws-document:{row['parent_record_id']}"
    assert matcher["stable_object_id"] == f"sha256:{row['normalized_sha256']}"
    assert matcher["origin_key"] == f"rada-laws-unit:{row['record_id']}"
    assert matcher["expected_raw_sha256"] == row["normalized_sha256"]
    assert matcher["declared_capacity_bytes"] == len(payload)
    assert matcher["expected_raw_bytes"] == len(payload)


@pytest.mark.parametrize("field", ["chunk_index", "normalized_bytes"])
def test_row_rejects_bool_aliases_for_integer_fields(field: str) -> None:
    row, _ = _row()
    row[field] = False
    metadata = {key: value for key, value in row.items() if key != "text"}
    pattern = "chunk_index" if field == "chunk_index" else "normalized byte count"
    with pytest.raises(adapter.RadaLawsDedupIntakeError, match=pattern):
        adapter._validate_project_row(row, metadata, line_number=1)


def test_row_rejects_text_hash_and_byte_drift() -> None:
    row, metadata = _row()
    row["text"] = f"{row['text']} tampered"
    with pytest.raises(adapter.RadaLawsDedupIntakeError, match="byte count drift"):
        adapter._validate_project_row(row, metadata, line_number=1)


def test_row_rejects_report_metadata_substitution() -> None:
    row, metadata = _row()
    metadata["source_path"] = "different.htm"
    with pytest.raises(
        adapter.RadaLawsDedupIntakeError,
        match="candidate/report metadata mismatch",
    ):
        adapter._validate_project_row(row, metadata, line_number=1)


def test_exact_content_aliases_are_preserved_for_matcher() -> None:
    first, first_meta = _row(parent="d100.htm", chunk=0, text="Однаковий текст документа.")
    second, second_meta = _row(parent="d200.htm", chunk=4, text="Однаковий текст документа.")
    first_matcher, first_id, _ = adapter._validate_project_row(
        first,
        first_meta,
        line_number=1,
    )
    second_matcher, second_id, _ = adapter._validate_project_row(
        second,
        second_meta,
        line_number=2,
    )
    assert first_id != second_id
    assert first_matcher["stable_object_id"] == second_matcher["stable_object_id"]
    assert first_matcher["origin_key"] != second_matcher["origin_key"]
    assert first_matcher["stable_origin_id"] != second_matcher["stable_origin_id"]


def _install_tiny_authority(
    monkeypatch: pytest.MonkeyPatch,
    candidate_bytes: bytes,
    metadata: dict[str, object],
    *,
    candidate_sha256: str | None = None,
) -> None:
    monkeypatch.setattr(adapter, "ACCEPTED_JSONL_FILE_BYTES", len(candidate_bytes))
    monkeypatch.setattr(
        adapter,
        "ACCEPTED_JSONL_SHA256",
        candidate_sha256 or hashlib.sha256(candidate_bytes).hexdigest(),
    )
    monkeypatch.setattr(adapter, "ACCEPTED_CHUNK_COUNT", 1)
    monkeypatch.setattr(
        adapter,
        "ACCEPTED_TEXT_UTF8_BYTES",
        int(metadata["normalized_bytes"]),
    )
    monkeypatch.setattr(adapter, "EXACT_DUPLICATE_ACCEPTED_HASHES", 0)
    monkeypatch.setattr(adapter, "ACCEPTED_SOURCE_ENCODING_COUNTS", {"utf-8": 1})
    monkeypatch.setattr(adapter, "_read_json", lambda *args, **kwargs: {})
    monkeypatch.setattr(adapter, "_validate_quality_report", lambda report: [metadata])
    monkeypatch.setattr(adapter, "_validate_execution_evidence", lambda evidence: None)


def test_candidate_transport_is_authenticated_and_parsed_from_one_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row, metadata = _row(parent="d100.htm", chunk=0)
    candidate_bytes = json.dumps(
        row,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8") + b"\n"
    candidate = tmp_path / "candidate.jsonl"
    candidate.write_bytes(candidate_bytes)
    _install_tiny_authority(monkeypatch, candidate_bytes, metadata)

    real_open = Path.open
    candidate_opens = 0

    def counted_open(path: Path, *args: object, **kwargs: object):
        nonlocal candidate_opens
        if path == candidate:
            candidate_opens += 1
            if candidate_opens > 1:
                raise AssertionError("candidate JSONL was reopened")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", counted_open)

    projection = adapter.validate_and_project_rada_laws_qp(
        candidate,
        tmp_path / "quality.json",
        tmp_path / "evidence.json",
        upstream_head=adapter.UPSTREAM_FINAL_EVIDENCE_COMMIT,
    )
    assert candidate_opens == 1
    assert projection.sources is not None
    assert len(projection.sources) == 1
    assert projection.payloads is not None
    assert set(projection.payloads) == {f"rada-laws-qp:{row['record_id']}"}


def test_candidate_transport_hash_mismatch_returns_no_projection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row, metadata = _row(parent="d100.htm", chunk=0)
    candidate_bytes = json.dumps(
        row,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8") + b"\n"
    candidate = tmp_path / "candidate.jsonl"
    candidate.write_bytes(candidate_bytes)
    _install_tiny_authority(
        monkeypatch,
        candidate_bytes,
        metadata,
        candidate_sha256="0" * 64,
    )

    with pytest.raises(
        adapter.RadaLawsDedupIntakeError,
        match="candidate transport SHA drift",
    ):
        adapter.validate_and_project_rada_laws_qp(
            candidate,
            tmp_path / "quality.json",
            tmp_path / "evidence.json",
            upstream_head=adapter.UPSTREAM_FINAL_EVIDENCE_COMMIT,
        )


def test_projection_api_exposes_no_caller_supplied_base_or_matcher_seam() -> None:
    assert not hasattr(adapter, "compose_with_incumbent_inventory")
    assert not hasattr(adapter, "delegate_to_incumbent_matcher")


def test_projection_receipt_remains_zero_credit_and_match_blocked() -> None:
    receipt = adapter._receipt(
        projection_identity_sha256="a" * 64,
        source_count=adapter.ACCEPTED_CHUNK_COUNT,
        payload_bytes=adapter.ACCEPTED_TEXT_UTF8_BYTES,
        duplicate_hashes=adapter.EXACT_DUPLICATE_ACCEPTED_HASHES,
        encoding_counts=adapter.ACCEPTED_SOURCE_ENCODING_COUNTS,
    )
    assert receipt["projection"]["exact_duplicate_payload_hashes_preserved"] == 764
    assert receipt["execution_gate"]["canonical_global_dedup_executed"] is False
    assert receipt["execution_gate"]["status"] == (
        "BLOCKED_PENDING_TERMINAL_PERFORMANCE_EQUIVALENT_EXECUTOR"
    )
    assert receipt["truth_boundary"]["canonical_capacity_credited"] == 0
    assert receipt["truth_boundary"]["training_authorized_bytes"] == 0
    assert receipt["truth_boundary"]["training_executed"] is False
