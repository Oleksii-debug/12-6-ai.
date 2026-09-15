from __future__ import annotations

import copy
import hashlib
import json

import pytest

from twelve_six.data.clean_g05_g06_successor_authority_v1 import (
    AUTHORITY_STATUS,
    INTEGRATED_SOURCE_MERGE_SHA,
    NOMIS_FAMILY,
    NOMIS_PAYLOAD_SHA256,
    NOMIS_RECORD_ID,
    PHYSICAL_ARTIFACT_ID,
    PHYSICAL_ARTIFACT_ZIP_SHA256,
    PHYSICAL_DATA526_EVIDENCE_IDENTITY_SHA256,
    PHYSICAL_EXECUTION_HEAD_SHA,
    PHYSICAL_RUN_ID,
    PHYSICAL_SOURCE_REPORT_SHA256,
    PHYSICAL_SURVIVOR_AUTHORITY_SHA256,
    PR462_AUTHORITY_SHA256,
    PROVENANCE_SCOPE,
    SOURCE_RELEASE_HEAD_SHA,
    CleanG05G06SuccessorError,
    build_clean_g05_g06_successor_authority,
    build_prepared_clean_g05_g06_successor_authority,
    integrated_nomis_free_source_anchor,
    verify_clean_g05_g06_successor_authority,
)


def _cjson(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _seal(document: dict[str, object], field: str) -> dict[str, object]:
    result = copy.deepcopy(document)
    result.pop(field, None)
    result[field] = hashlib.sha256(_cjson(result)).hexdigest()
    return result


def _candidate_clean() -> dict[str, object]:
    return {
        "nomis1864_admitted": False,
        "pr462_authority_admitted": False,
        "nomis1864_quarantine_enforced": True,
        "whole_corpus_external_llm_cleanliness_claimed": False,
        "current_retained_corpus_launch_authoritative": False,
        "tokenizer_fit_authorized": False,
        "training_executed": False,
        "learned_weights_created": False,
        "final_test_outcomes_read": False,
        "paid_compute_used": False,
        "foreign_pretrained_weights": False,
    }


def _candidate_replays() -> list[dict[str, object]]:
    return [
        {"covered_record_count": 1, "covered_payload_bytes": 17},
        {"covered_record_count": 1, "covered_payload_bytes": 17},
    ]


def test_prepared_authority_binds_integrated_physical_source_anchor() -> None:
    authority = build_prepared_clean_g05_g06_successor_authority()
    anchor = authority["source_anchor"]

    assert authority["status"] == AUTHORITY_STATUS == "PREPARED_NOT_EXECUTED"
    assert authority["terminal_authority_available"] is False
    assert authority["provenance_scope"] == PROVENANCE_SCOPE
    assert anchor["integrated_source_merge_sha"] == INTEGRATED_SOURCE_MERGE_SHA
    assert anchor["source_release_head_sha"] == SOURCE_RELEASE_HEAD_SHA
    assert anchor["physical_execution_head_sha"] == PHYSICAL_EXECUTION_HEAD_SHA
    assert anchor["physical_run_id"] == PHYSICAL_RUN_ID
    assert anchor["physical_artifact_id"] == PHYSICAL_ARTIFACT_ID
    assert (
        anchor["physical_artifact_zip_sha256"]
        == PHYSICAL_ARTIFACT_ZIP_SHA256
    )
    assert anchor["physical_source_report_sha256"] == PHYSICAL_SOURCE_REPORT_SHA256
    assert (
        anchor["physical_survivor_authority_sha256"]
        == PHYSICAL_SURVIVOR_AUTHORITY_SHA256
    )
    assert (
        anchor["physical_data526_evidence_identity_sha256"]
        == PHYSICAL_DATA526_EVIDENCE_IDENTITY_SHA256
    )
    assert anchor["nomis_record_id"] == NOMIS_RECORD_ID
    assert anchor["nomis_family"] == NOMIS_FAMILY
    assert anchor["nomis_payload_sha256"] == NOMIS_PAYLOAD_SHA256
    assert anchor["pr462_authority_sha256"] == PR462_AUTHORITY_SHA256
    assert anchor["nomis1864_admitted"] is False
    assert anchor["pr462_authority_admitted"] is False
    assert anchor["nomis1864_quarantine_enforced"] is True
    verify_clean_g05_g06_successor_authority(authority)


def test_prepared_authority_has_no_downstream_or_scientific_promotion() -> None:
    authority = build_prepared_clean_g05_g06_successor_authority()

    assert all(
        value is None
        for value in authority["required_downstream_roots"].values()
    )
    assert authority["whole_corpus_external_llm_cleanliness_claimed"] is False
    assert authority["current_retained_corpus_launch_authoritative"] is False
    assert authority["authorized_optimized_target_exposure"] == 0
    assert authority["tokenizer_fit_authorized"] is False
    assert authority["optimizer_updates_executed_on_real_targets"] == 0
    assert authority["training_executed"] is False
    assert authority["learned_weights_created"] is False
    assert authority["final_test_outcomes_read"] is False
    assert authority["paid_compute_used"] is False
    assert authority["foreign_pretrained_weights"] is False


def test_self_certified_candidate_bundle_cannot_emit_terminal_authority() -> None:
    with pytest.raises(
        CleanG05G06SuccessorError,
        match="terminal G05/G06 authority is unavailable",
    ):
        build_clean_g05_g06_successor_authority(
            clean_input_binding=_candidate_clean(),
            g05_g06_coverage={"self_sealed": True},
            replay_receipts=_candidate_replays(),
        )


@pytest.mark.parametrize(
    ("field", "alias"),
    [
        ("nomis1864_admitted", 0),
        ("pr462_authority_admitted", 0),
        ("nomis1864_quarantine_enforced", 1),
        ("whole_corpus_external_llm_cleanliness_claimed", 0),
    ],
)
def test_rejects_integer_aliases_for_clean_quarantine_booleans(
    field: str,
    alias: int,
) -> None:
    clean = _candidate_clean()
    clean[field] = alias

    with pytest.raises(CleanG05G06SuccessorError, match="must be a boolean"):
        build_clean_g05_g06_successor_authority(
            clean_input_binding=clean,
            g05_g06_coverage={},
            replay_receipts=_candidate_replays(),
        )


@pytest.mark.parametrize("field", ["covered_record_count", "covered_payload_bytes"])
def test_rejects_float_aliases_for_replay_counts(field: str) -> None:
    replays = _candidate_replays()
    replays[0][field] = float(replays[0][field])

    with pytest.raises(
        CleanG05G06SuccessorError,
        match="must be a positive integer",
    ):
        build_clean_g05_g06_successor_authority(
            clean_input_binding=_candidate_clean(),
            g05_g06_coverage={},
            replay_receipts=replays,
        )


def test_rejects_terminal_status_promotion_even_when_resealed() -> None:
    authority = build_prepared_clean_g05_g06_successor_authority()
    authority["status"] = "PASS_REPLAY_BOUND_ZERO_CREDIT"
    authority = _seal(authority, "successor_authority_identity_sha256")

    with pytest.raises(
        CleanG05G06SuccessorError,
        match="PREPARED_NOT_EXECUTED",
    ):
        verify_clean_g05_g06_successor_authority(authority)


def test_rejects_downstream_root_injection_even_when_resealed() -> None:
    authority = build_prepared_clean_g05_g06_successor_authority()
    authority["required_downstream_roots"][
        "decontamination_authority_sha256"
    ] = "a" * 64
    authority = _seal(authority, "successor_authority_identity_sha256")

    with pytest.raises(
        CleanG05G06SuccessorError,
        match="untrusted downstream root",
    ):
        verify_clean_g05_g06_successor_authority(authority)


def test_rejects_integrated_source_anchor_substitution_even_when_resealed() -> None:
    authority = build_prepared_clean_g05_g06_successor_authority()
    authority["source_anchor"]["physical_source_report_sha256"] = "b" * 64
    authority["source_anchor"] = _seal(
        authority["source_anchor"],
        "source_anchor_identity_sha256",
    )
    authority = _seal(authority, "successor_authority_identity_sha256")

    with pytest.raises(
        CleanG05G06SuccessorError,
        match="integrated source anchor drift",
    ):
        verify_clean_g05_g06_successor_authority(authority)


def test_rejects_bool_alias_in_integrated_anchor_even_when_resealed() -> None:
    authority = build_prepared_clean_g05_g06_successor_authority()
    authority["source_anchor"]["nomis1864_admitted"] = 0
    authority["source_anchor"] = _seal(
        authority["source_anchor"],
        "source_anchor_identity_sha256",
    )
    authority = _seal(authority, "successor_authority_identity_sha256")

    with pytest.raises(
        CleanG05G06SuccessorError,
        match="integrated source anchor drift",
    ):
        verify_clean_g05_g06_successor_authority(authority)


def test_source_anchor_is_deterministic_and_self_sealed() -> None:
    first = integrated_nomis_free_source_anchor()
    second = integrated_nomis_free_source_anchor()

    assert first == second
    claimed = first["source_anchor_identity_sha256"]
    body = copy.deepcopy(first)
    body.pop("source_anchor_identity_sha256")
    assert claimed == hashlib.sha256(_cjson(body)).hexdigest()
