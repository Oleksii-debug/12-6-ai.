from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace

import pytest

from twelve_six.data.arxiv_languk_rematerialized_replay_v1 import (
    ARXIV,
    LANGUK,
    PARENT_INTAKE_BLOB_SHA1,
    PARENT_RUNNER_BLOB_SHA1,
    RematerializationError,
    build_receipt,
    canonical_json_bytes,
    git_blob_sha1,
    verify_candidate,
    verify_git_blob,
)


def _candidate_row(
    *,
    record_id: str = "7",
    text: str = "Український текст",
    arxiv: bool = False,
) -> dict[str, object]:
    raw = text.encode("utf-8")
    row: dict[str, object] = {
        "record_id": record_id,
        "normalized_sha256": hashlib.sha256(raw).hexdigest(),
        "normalized_bytes": len(raw),
        "text": text,
    }
    if arxiv:
        row["source_key"] = "arxiv_abstracts"
        row["source_label"] = "arxiv-abstracts"
    return row


def _fixture_spec(*, arxiv: bool, rows: list[dict[str, object]]):
    raw = b"".join(canonical_json_bytes(row) for row in rows)
    template = ARXIV if arxiv else LANGUK
    return (
        replace(
            template,
            candidate_sha256=hashlib.sha256(raw).hexdigest(),
            retained_records=len(rows),
            retained_normalized_bytes=sum(int(row["normalized_bytes"]) for row in rows),
        ),
        raw,
    )


def test_terminal_program_identities_are_pinned() -> None:
    assert ARXIV.execution_commit == "37289b5af224c58f0b169f55f4e0f8b466b90f6d"
    assert ARXIV.tool_blob_sha1 == "38c244a5a2da40ed4fb67f801f02e625bc040564"
    assert ARXIV.config_blob_sha1 == "4a819b3980634c8a2ba7de55cf854d44bcfd5757"
    assert ARXIV.candidate_sha256 == (
        "21304338039306b2df175bbb71a1aed9a443c2416f3858f3cc63ca208e9c7327"
    )
    assert (ARXIV.retained_records, ARXIV.retained_normalized_bytes) == (
        1024,
        1_139_552,
    )

    assert LANGUK.execution_commit == "5d38a32b37490b11f6fa77b3bcd21855f3e28605"
    assert LANGUK.tool_blob_sha1 == "5401881e170f2d1ae70e777a0cf3e2cfa152e8f6"
    assert LANGUK.config_blob_sha1 == "b4ee7b0f698660145cb3c1db0f8ccb690c255a5e"
    assert LANGUK.candidate_sha256 == (
        "03bf5089bb6ff4c304e3a299e2480b263a161aad8abe831db0b196af7c585db3"
    )
    assert (LANGUK.retained_records, LANGUK.retained_normalized_bytes) == (
        256,
        2_809_632,
    )


def test_git_blob_verification_is_exact() -> None:
    raw = b"trusted historical program\n"
    expected = git_blob_sha1(raw)
    verify_git_blob(raw, expected, label="fixture")
    with pytest.raises(RematerializationError, match="Git blob drift"):
        verify_git_blob(raw + b"x", expected, label="fixture")


def test_verify_languk_candidate_checks_exact_identity_and_rows() -> None:
    rows = [
        _candidate_row(record_id="10", text="Український судовий документ"),
        _candidate_row(record_id="11", text="Ще один український документ"),
    ]
    spec, raw = _fixture_spec(arxiv=False, rows=rows)
    result = verify_candidate(spec, raw)
    assert result == {
        "candidate_sha256": spec.candidate_sha256,
        "retained_records": 2,
        "retained_normalized_bytes": sum(int(row["normalized_bytes"]) for row in rows),
    }


def test_verify_arxiv_candidate_checks_source_binding() -> None:
    rows = [_candidate_row(record_id="arxiv-1", text="A long abstract", arxiv=True)]
    spec, raw = _fixture_spec(arxiv=True, rows=rows)
    verify_candidate(spec, raw)

    mutated = copy.deepcopy(rows)
    mutated[0]["source_label"] = "wrong"
    bad_spec, bad_raw = _fixture_spec(arxiv=True, rows=mutated)
    with pytest.raises(RematerializationError, match="source label drift"):
        verify_candidate(bad_spec, bad_raw)


def test_verify_candidate_rejects_duplicate_ids() -> None:
    rows = [
        _candidate_row(record_id="10", text="Документ один"),
        _candidate_row(record_id="10", text="Документ два"),
    ]
    spec, raw = _fixture_spec(arxiv=False, rows=rows)
    with pytest.raises(RematerializationError, match="duplicate record id"):
        verify_candidate(spec, raw)


def test_verify_candidate_rejects_payload_hash_drift_before_parsing() -> None:
    rows = [_candidate_row(record_id="10", text="Документ")]
    spec, raw = _fixture_spec(arxiv=False, rows=rows)
    with pytest.raises(RematerializationError, match="candidate SHA-256 drift"):
        verify_candidate(spec, raw + b"\n")


def _pass_result() -> dict[str, str]:
    return {
        "arxiv_candidate_sha256": ARXIV.candidate_sha256,
        "languk_candidate_sha256": LANGUK.candidate_sha256,
        "report_file_sha256": "1" * 64,
        "survivor_file_sha256": "2" * 64,
        "report_identity_sha256": "3" * 64,
        "survivor_authority_sha256": "4" * 64,
    }


def test_receipt_requires_two_identical_passes_and_preserves_zero_truth() -> None:
    one = _pass_result()
    receipt = build_receipt(
        pass_results=[one, dict(one)],
        incumbent_runner_blob_sha1=PARENT_RUNNER_BLOB_SHA1,
        incumbent_intake_blob_sha1=PARENT_INTAKE_BLOB_SHA1,
    )
    assert receipt["status"] == (
        "PHYSICAL_REMATERIALIZATION_AND_V9_REPLAY_EXECUTED_ZERO_CREDIT"
    )
    assert receipt["reproducibility"]["report_files_byte_identical"] is True
    assert receipt["reproducibility"]["raw_payloads_retained_after_success"] is False
    boundary = receipt["truth_boundary"]
    assert boundary["global_dedup_replay_executed"] is True
    assert boundary["canonical_capacity_credited"] == 0
    assert boundary["authorized_optimized_target_exposure"] == 0
    assert boundary["tokenizer_fit_authorized"] is False
    assert boundary["optimizer_updates"] == 0
    assert boundary["training_executed"] is False
    assert boundary["learned_weights_created"] is False
    assert boundary["final_test_outcomes_read"] is False
    assert boundary["paid_compute_used"] is False
    assert boundary["foreign_pretrained_weights_used"] is False
    assert boundary["external_llm_or_api_used_for_data_or_intelligence"] is False


def test_receipt_fails_on_second_pass_drift() -> None:
    one = _pass_result()
    two = dict(one)
    two["report_file_sha256"] = "9" * 64
    with pytest.raises(RematerializationError, match="two-pass replay mismatch"):
        build_receipt(
            pass_results=[one, two],
            incumbent_runner_blob_sha1=PARENT_RUNNER_BLOB_SHA1,
            incumbent_intake_blob_sha1=PARENT_INTAKE_BLOB_SHA1,
        )


def test_receipt_fails_on_incumbent_blob_drift() -> None:
    one = _pass_result()
    with pytest.raises(RematerializationError, match="incumbent runner blob drift"):
        build_receipt(
            pass_results=[one, dict(one)],
            incumbent_runner_blob_sha1="0" * 40,
            incumbent_intake_blob_sha1=PARENT_INTAKE_BLOB_SHA1,
        )


def test_receipt_serialization_contains_no_payload_text() -> None:
    one = _pass_result()
    receipt = build_receipt(
        pass_results=[one, dict(one)],
        incumbent_runner_blob_sha1=PARENT_RUNNER_BLOB_SHA1,
        incumbent_intake_blob_sha1=PARENT_INTAKE_BLOB_SHA1,
    )
    serialized = canonical_json_bytes(receipt)
    decoded = json.loads(serialized)
    assert decoded["historical_materializers"]["arxiv"]["retained_records"] == 1024
    assert "text" not in serialized.decode("utf-8").lower()
