from __future__ import annotations

import copy
import hashlib
import json

import pytest

from twelve_six.data.clean_g05_g06_successor_authority_v1 import (
    AUTHORITY_STATUS,
    CLEAN_INPUT_BINDING_SCHEMA,
    CLEAN_INPUT_STATUS,
    NOMIS_FAMILY,
    NOMIS_PAYLOAD_SHA256,
    NOMIS_RECORD_ID,
    PR462_AUTHORITY_SHA256,
    PROVENANCE_SCOPE,
    REPLAY_EXECUTION_POLICY,
    REPLAY_RECEIPT_SCHEMA,
    REPLAY_STATUS,
    CleanG05G06SuccessorError,
    build_clean_g05_g06_successor_authority,
    verify_clean_g05_g06_successor_authority,
)
from twelve_six.data.final_g05_g06_coverage_v1 import (
    QUALITY_GRANULARITY_IDENTITY_SHA256,
    QUALITY_POLICY_IDENTITY_SHA256,
)

H1 = "1" * 64
H2 = "2" * 64
H3 = "3" * 64
H4 = "4" * 64
H5 = "5" * 64
H6 = "6" * 64
H7 = "7" * 64
H8 = "8" * 64
HA = "a" * 64
GIT = "d" * 40


def _cjson(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha(value: object) -> str:
    payload = value if isinstance(value, bytes) else _cjson(value)
    return hashlib.sha256(payload).hexdigest()


def _seal(document: dict[str, object], field: str) -> dict[str, object]:
    result = copy.deepcopy(document)
    result.pop(field, None)
    result[field] = _sha(result)
    return result


def _clean_input() -> dict[str, object]:
    return _seal(
        {
            "schema": CLEAN_INPUT_BINDING_SCHEMA,
            "status": CLEAN_INPUT_STATUS,
            "source_rebuild_authority_sha256": H1,
            "retained_inventory_identity_sha256": H2,
            "records_jsonl_sha256": H3,
            "decontamination_authority_sha256": H4,
            "nomis_record_id": NOMIS_RECORD_ID,
            "nomis_family": NOMIS_FAMILY,
            "nomis_payload_sha256": NOMIS_PAYLOAD_SHA256,
            "pr462_authority_sha256": PR462_AUTHORITY_SHA256,
            "nomis1864_admitted": False,
            "pr462_authority_admitted": False,
            "nomis1864_quarantine_enforced": True,
            "whole_corpus_external_llm_cleanliness_claimed": False,
            "current_retained_corpus_launch_authoritative": False,
            "authorized_optimized_target_exposure": 0,
            "tokenizer_fit_authorized": False,
            "optimizer_updates_executed_on_real_targets": 0,
            "training_executed": False,
            "learned_weights_created": False,
            "final_test_outcomes_read": False,
            "paid_compute_used": False,
            "foreign_pretrained_weights": False,
        },
        "clean_input_binding_identity_sha256",
    )


def _coverage() -> dict[str, object]:
    qualification_roots = [H5]
    record = {
        "record_id_sha256": H6,
        "payload_sha256": H7,
        "payload_bytes": 17,
    }
    return _seal(
        {
            "schema": "12-6.d03-final-g05-g06-coverage.v1",
            "status": "PASS",
            "records_jsonl_sha256": H3,
            "retained_inventory_identity_sha256": H2,
            "decontamination_authority_sha256": H4,
            "quality_policy_identity_sha256": QUALITY_POLICY_IDENTITY_SHA256,
            "quality_granularity_identity_sha256": (
                QUALITY_GRANULARITY_IDENTITY_SHA256
            ),
            "privacy_policy_identity_sha256": H8,
            "privacy_implementation_git_blob_sha": GIT,
            "qualification_authority_identities_sha256": qualification_roots,
            "qualification_authority_set_identity_sha256": _sha(
                qualification_roots
            ),
            "projection_rule": "PRE_DECONTAM_UNCHANGED_PAYLOAD_SUBSET_V1",
            "predecontam_record_count": 1,
            "predecontam_payload_bytes": 17,
            "excluded_record_count": 0,
            "excluded_payload_bytes": 0,
            "covered_record_count": 1,
            "covered_payload_bytes": 17,
            "covered_records": [record],
            "final_test_outcomes_read": False,
            "training_authorized_by_this_coverage": False,
            "tokenizer_fit_authorized": False,
            "model_training_authorized": False,
            "authorized_optimized_target_exposure": 0,
            "paid_compute_used": False,
        },
        "g05_g06_coverage_identity_sha256",
    )


def _replay(
    run_id: str,
    *,
    clean_input: dict[str, object],
    coverage: dict[str, object],
) -> dict[str, object]:
    return _seal(
        {
            "schema": REPLAY_RECEIPT_SCHEMA,
            "status": REPLAY_STATUS,
            "run_id": run_id,
            "execution_policy": REPLAY_EXECUTION_POLICY,
            "clean_input_binding_identity_sha256": clean_input[
                "clean_input_binding_identity_sha256"
            ],
            "g05_g06_coverage_identity_sha256": coverage[
                "g05_g06_coverage_identity_sha256"
            ],
            "retained_inventory_identity_sha256": H2,
            "records_jsonl_sha256": H3,
            "decontamination_authority_sha256": H4,
            "quality_policy_identity_sha256": QUALITY_POLICY_IDENTITY_SHA256,
            "quality_granularity_identity_sha256": (
                QUALITY_GRANULARITY_IDENTITY_SHA256
            ),
            "privacy_policy_identity_sha256": H8,
            "privacy_implementation_git_blob_sha": GIT,
            "qualification_authority_set_identity_sha256": coverage[
                "qualification_authority_set_identity_sha256"
            ],
            "covered_record_count": 1,
            "covered_payload_bytes": 17,
            "whole_corpus_external_llm_cleanliness_claimed": False,
            "current_retained_corpus_launch_authoritative": False,
            "authorized_optimized_target_exposure": 0,
            "tokenizer_fit_authorized": False,
            "optimizer_updates_executed_on_real_targets": 0,
            "training_executed": False,
            "learned_weights_created": False,
            "final_test_outcomes_read": False,
            "paid_compute_used": False,
            "foreign_pretrained_weights": False,
        },
        "replay_receipt_identity_sha256",
    )


def _build() -> tuple[
    dict[str, object],
    dict[str, object],
    list[dict[str, object]],
    dict[str, object],
]:
    clean_input = _clean_input()
    coverage = _coverage()
    replays = [
        _replay("clean-replay-a", clean_input=clean_input, coverage=coverage),
        _replay("clean-replay-b", clean_input=clean_input, coverage=coverage),
    ]
    authority = build_clean_g05_g06_successor_authority(
        clean_input_binding=clean_input,
        expected_clean_input_binding_identity_sha256=clean_input[
            "clean_input_binding_identity_sha256"
        ],
        expected_source_rebuild_authority_sha256=H1,
        g05_g06_coverage=coverage,
        expected_g05_g06_coverage_identity_sha256=coverage[
            "g05_g06_coverage_identity_sha256"
        ],
        expected_privacy_policy_identity_sha256=H8,
        expected_privacy_implementation_git_blob_sha=GIT,
        replay_receipts=replays,
        expected_replay_receipt_identities_sha256=[
            replay["replay_receipt_identity_sha256"] for replay in replays
        ],
    )
    return clean_input, coverage, replays, authority


def test_builds_text_free_zero_credit_successor_authority() -> None:
    clean_input, coverage, replays, authority = _build()

    assert authority["status"] == AUTHORITY_STATUS
    assert authority["provenance_scope"] == PROVENANCE_SCOPE
    assert authority["whole_corpus_external_llm_cleanliness_claimed"] is False
    assert authority["current_retained_corpus_launch_authoritative"] is False
    assert authority["authorized_optimized_target_exposure"] == 0
    assert authority["training_executed"] is False
    assert authority["learned_weights_created"] is False
    assert authority["covered_record_count"] == 1
    assert authority["covered_payload_bytes"] == 17
    assert authority["replay_receipt_identities_sha256"] == sorted(
        replay["replay_receipt_identity_sha256"] for replay in replays
    )

    verify_clean_g05_g06_successor_authority(
        authority,
        expected_identity_sha256=authority[
            "successor_authority_identity_sha256"
        ],
        expected_clean_input_binding_identity_sha256=clean_input[
            "clean_input_binding_identity_sha256"
        ],
        expected_source_rebuild_authority_sha256=H1,
        expected_g05_g06_coverage_identity_sha256=coverage[
            "g05_g06_coverage_identity_sha256"
        ],
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("nomis1864_admitted", True),
        ("pr462_authority_admitted", True),
        ("nomis1864_quarantine_enforced", False),
        ("whole_corpus_external_llm_cleanliness_claimed", True),
        ("current_retained_corpus_launch_authoritative", True),
        ("authorized_optimized_target_exposure", 1),
        ("training_executed", True),
        ("learned_weights_created", True),
    ],
)
def test_rejects_dirty_or_promoted_clean_input(field: str, value: object) -> None:
    clean_input = _clean_input()
    coverage = _coverage()
    clean_input[field] = value
    clean_input = _seal(clean_input, "clean_input_binding_identity_sha256")
    replays = [
        _replay("a", clean_input=clean_input, coverage=coverage),
        _replay("b", clean_input=clean_input, coverage=coverage),
    ]

    with pytest.raises(CleanG05G06SuccessorError):
        build_clean_g05_g06_successor_authority(
            clean_input_binding=clean_input,
            expected_clean_input_binding_identity_sha256=clean_input[
                "clean_input_binding_identity_sha256"
            ],
            expected_source_rebuild_authority_sha256=H1,
            g05_g06_coverage=coverage,
            expected_g05_g06_coverage_identity_sha256=coverage[
                "g05_g06_coverage_identity_sha256"
            ],
            expected_privacy_policy_identity_sha256=H8,
            expected_privacy_implementation_git_blob_sha=GIT,
            replay_receipts=replays,
            expected_replay_receipt_identities_sha256=[
                replay["replay_receipt_identity_sha256"] for replay in replays
            ],
        )


def test_rejects_caller_self_selected_clean_root() -> None:
    clean_input = _clean_input()
    coverage = _coverage()
    replays = [
        _replay("a", clean_input=clean_input, coverage=coverage),
        _replay("b", clean_input=clean_input, coverage=coverage),
    ]

    with pytest.raises(
        CleanG05G06SuccessorError,
        match="not independently expected",
    ):
        build_clean_g05_g06_successor_authority(
            clean_input_binding=clean_input,
            expected_clean_input_binding_identity_sha256=HA,
            expected_source_rebuild_authority_sha256=H1,
            g05_g06_coverage=coverage,
            expected_g05_g06_coverage_identity_sha256=coverage[
                "g05_g06_coverage_identity_sha256"
            ],
            expected_privacy_policy_identity_sha256=H8,
            expected_privacy_implementation_git_blob_sha=GIT,
            replay_receipts=replays,
            expected_replay_receipt_identities_sha256=[
                replay["replay_receipt_identity_sha256"] for replay in replays
            ],
        )


def test_requires_distinct_replay_run_ids() -> None:
    clean_input = _clean_input()
    coverage = _coverage()
    replays = [
        _replay("same-run", clean_input=clean_input, coverage=coverage),
        _replay("same-run", clean_input=clean_input, coverage=coverage),
    ]

    with pytest.raises(
        CleanG05G06SuccessorError,
        match="expected replay roots must be distinct",
    ):
        build_clean_g05_g06_successor_authority(
            clean_input_binding=clean_input,
            expected_clean_input_binding_identity_sha256=clean_input[
                "clean_input_binding_identity_sha256"
            ],
            expected_source_rebuild_authority_sha256=H1,
            g05_g06_coverage=coverage,
            expected_g05_g06_coverage_identity_sha256=coverage[
                "g05_g06_coverage_identity_sha256"
            ],
            expected_privacy_policy_identity_sha256=H8,
            expected_privacy_implementation_git_blob_sha=GIT,
            replay_receipts=replays,
            expected_replay_receipt_identities_sha256=[
                replay["replay_receipt_identity_sha256"] for replay in replays
            ],
        )


def test_rejects_coherently_resealed_replay_drift() -> None:
    clean_input = _clean_input()
    coverage = _coverage()
    first = _replay("a", clean_input=clean_input, coverage=coverage)
    second = _replay("b", clean_input=clean_input, coverage=coverage)
    second["covered_payload_bytes"] = 18
    second = _seal(second, "replay_receipt_identity_sha256")

    with pytest.raises(
        CleanG05G06SuccessorError,
        match="replay projection drift: covered_payload_bytes",
    ):
        build_clean_g05_g06_successor_authority(
            clean_input_binding=clean_input,
            expected_clean_input_binding_identity_sha256=clean_input[
                "clean_input_binding_identity_sha256"
            ],
            expected_source_rebuild_authority_sha256=H1,
            g05_g06_coverage=coverage,
            expected_g05_g06_coverage_identity_sha256=coverage[
                "g05_g06_coverage_identity_sha256"
            ],
            expected_privacy_policy_identity_sha256=H8,
            expected_privacy_implementation_git_blob_sha=GIT,
            replay_receipts=[first, second],
            expected_replay_receipt_identities_sha256=[
                first["replay_receipt_identity_sha256"],
                second["replay_receipt_identity_sha256"],
            ],
        )


def test_rejects_replay_nonclaim_promotion_even_when_resealed() -> None:
    clean_input = _clean_input()
    coverage = _coverage()
    first = _replay("a", clean_input=clean_input, coverage=coverage)
    second = _replay("b", clean_input=clean_input, coverage=coverage)
    second["whole_corpus_external_llm_cleanliness_claimed"] = True
    second = _seal(second, "replay_receipt_identity_sha256")

    with pytest.raises(CleanG05G06SuccessorError, match="must be false"):
        build_clean_g05_g06_successor_authority(
            clean_input_binding=clean_input,
            expected_clean_input_binding_identity_sha256=clean_input[
                "clean_input_binding_identity_sha256"
            ],
            expected_source_rebuild_authority_sha256=H1,
            g05_g06_coverage=coverage,
            expected_g05_g06_coverage_identity_sha256=coverage[
                "g05_g06_coverage_identity_sha256"
            ],
            expected_privacy_policy_identity_sha256=H8,
            expected_privacy_implementation_git_blob_sha=GIT,
            replay_receipts=[first, second],
            expected_replay_receipt_identities_sha256=[
                first["replay_receipt_identity_sha256"],
                second["replay_receipt_identity_sha256"],
            ],
        )


def test_durable_verifier_rejects_scientific_promotion_after_reseal() -> None:
    clean_input, coverage, _, authority = _build()
    authority["authorized_optimized_target_exposure"] = 1
    authority = _seal(authority, "successor_authority_identity_sha256")

    with pytest.raises(CleanG05G06SuccessorError, match="must be zero"):
        verify_clean_g05_g06_successor_authority(
            authority,
            expected_identity_sha256=authority[
                "successor_authority_identity_sha256"
            ],
            expected_clean_input_binding_identity_sha256=clean_input[
                "clean_input_binding_identity_sha256"
            ],
            expected_source_rebuild_authority_sha256=H1,
            expected_g05_g06_coverage_identity_sha256=coverage[
                "g05_g06_coverage_identity_sha256"
            ],
        )


def test_rejects_extra_root_field() -> None:
    clean_input, coverage, _, authority = _build()
    authority["external_llm_or_api_used_for_data_or_intelligence"] = False
    authority = _seal(authority, "successor_authority_identity_sha256")

    with pytest.raises(CleanG05G06SuccessorError, match="root schema drift"):
        verify_clean_g05_g06_successor_authority(
            authority,
            expected_identity_sha256=authority[
                "successor_authority_identity_sha256"
            ],
            expected_clean_input_binding_identity_sha256=clean_input[
                "clean_input_binding_identity_sha256"
            ],
            expected_source_rebuild_authority_sha256=H1,
            expected_g05_g06_coverage_identity_sha256=coverage[
                "g05_g06_coverage_identity_sha256"
            ],
        )
