from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

import pytest

from twelve_six.data import balanced_split_application_v1 as split_application

VALID_BOUNDARY = {
    "training_eligible": False,
    "evaluation_eligible": False,
    "tokenizer_fit_authorized": False,
    "model_training_authorized": False,
    "paid_compute_authorized": False,
    "final_test_outcomes_read": False,
    "authorized_optimized_target_exposure": 0,
}
IDS = {
    "inventory": hashlib.sha256(b"inventory").hexdigest(),
    "decontam": hashlib.sha256(b"decontam").hexdigest(),
    "dedup": hashlib.sha256(b"dedup").hexdigest(),
    "policy": hashlib.sha256(b"policy").hexdigest(),
    "balance": hashlib.sha256(b"balance").hexdigest(),
}
SEEDS = ("split-a", "split-b", "split-c")
STALE_SPLIT_BLOB_SHA1 = "76f18ad4dea7439e20312289d1d259b773e36d26"
CURRENT_SPLIT_BLOB_SHA1 = "5a5395748bed6b666391268b605e428af18baf0c"


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _rehash(document: dict[str, Any], field: str) -> None:
    core = dict(document)
    core.pop(field, None)
    document[field] = hashlib.sha256(_canonical_bytes(core)).hexdigest()


def _raw_records() -> list[dict[str, Any]]:
    specs = [
        ("r1", "s1", "family-uk-a", "uk", "text", "c1", "Альфа один"),
        ("r2", "s2", "family-uk-b", "uk", "text", "c1", "Альфа два"),
        ("r3", "s3", "family-en-a", "en", "text", "c2", "English three"),
        ("r4", "s4", "family-en-b", "en", "text", "c3", "English four"),
        ("r5", "s5", "family-code-a", "code", "code", "c4", "def f(): return 5"),
        ("r6", "s6", "family-code-b", "code", "code", "c4", "def g(): return 6"),
        ("r7", "s7", "family-uk-a", "uk", "text", "c5", "Український сім"),
        ("r8", "s8", "family-en-a", "en", "text", "c6", "English eight"),
    ]
    return [
        {
            "record_id": record_id,
            "source_id": source_id,
            "family": family,
            "stratum": stratum,
            "modality": modality,
            "near_duplicate_cluster_id": cluster,
            "normalized_payload": payload,
            "purpose": "pretraining_eligible",
            "training_eligible": False,
            "evaluation_eligible": False,
            "evaluation_reserved": False,
        }
        for record_id, source_id, family, stratum, modality, cluster, payload in specs
    ]


def _selection(raw_records: list[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    family_bytes: dict[str, int] = {}
    stratum_bytes: dict[str, int] = {}
    source_bytes = 0
    for raw in raw_records:
        payload = raw["normalized_payload"].encode()
        payload_bytes = len(payload)
        row = {
            "record_id": raw["record_id"],
            "source_id": raw["source_id"],
            "family": raw["family"],
            "stratum": raw["stratum"],
            "modality": raw["modality"],
            "payload_sha256": hashlib.sha256(payload).hexdigest(),
            "payload_bytes": payload_bytes,
            "near_duplicate_cluster_id": raw["near_duplicate_cluster_id"],
            "purpose": raw["purpose"],
            "training_eligible": False,
            "evaluation_eligible": False,
            "evaluation_reserved": False,
        }
        rows.append(row)
        source_bytes += payload_bytes
        family_bytes[row["family"]] = family_bytes.get(row["family"], 0) + payload_bytes
        stratum_bytes[row["stratum"]] = stratum_bytes.get(row["stratum"], 0) + payload_bytes
    document: dict[str, Any] = {
        "schema": "12-6.d03-balanced-selection-authority.v1",
        "terminal": True,
        "status": "PASS",
        "balanced_selection_identity_sha256": "0" * 64,
        "retained_inventory_identity_sha256": IDS["inventory"],
        "decontamination_authority_sha256": IDS["decontam"],
        "dedup_authority_sha256": IDS["dedup"],
        "balance_policy_identity_sha256": IDS["policy"],
        "balance_result_identity_sha256": IDS["balance"],
        "records": rows,
        "totals": {
            "record_count": len(rows),
            "source_bytes": source_bytes,
            "family_source_bytes": dict(sorted(family_bytes.items())),
            "stratum_source_bytes": dict(sorted(stratum_bytes.items())),
        },
        "claim_boundary": copy.deepcopy(VALID_BOUNDARY),
    }
    _rehash(document, "balanced_selection_identity_sha256")
    return document


def _kwargs(selection: dict[str, Any]) -> dict[str, Any]:
    return {
        "expected_selection_identity_sha256": selection[
            "balanced_selection_identity_sha256"
        ],
        "expected_retained_inventory_identity_sha256": IDS["inventory"],
        "expected_decontamination_authority_sha256": IDS["decontam"],
        "expected_dedup_authority_sha256": IDS["dedup"],
        "expected_balance_policy_identity_sha256": IDS["policy"],
        "expected_balance_result_identity_sha256": IDS["balance"],
        "expected_split_git_blob_sha1": split_application.CANONICAL_SPLIT_GIT_BLOB_SHA1,
        "variant_seeds": SEEDS,
        "validation_fraction": 0.2,
    }


def _build() -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    raw = _raw_records()
    selection = _selection(raw)
    application = split_application.build_balanced_split_application(
        selection,
        raw,
        **_kwargs(selection),
    )
    return raw, selection, application


def test_zero_credit_boundary_accepts_exact_json_types() -> None:
    split_application._require_zero_credit_boundary(copy.deepcopy(VALID_BOUNDARY))


@pytest.mark.parametrize(
    "field",
    [
        "training_eligible",
        "evaluation_eligible",
        "tokenizer_fit_authorized",
        "model_training_authorized",
        "paid_compute_authorized",
        "final_test_outcomes_read",
    ],
)
def test_zero_credit_boundary_rejects_integer_alias_for_false(field: str) -> None:
    boundary = copy.deepcopy(VALID_BOUNDARY)
    boundary[field] = 0
    with pytest.raises(split_application.BalancedSplitApplicationError):
        split_application._require_zero_credit_boundary(boundary)


@pytest.mark.parametrize("alias", [False, 0.0, "0", None])
def test_zero_credit_boundary_rejects_non_integer_zero_exposure(alias: object) -> None:
    boundary = copy.deepcopy(VALID_BOUNDARY)
    boundary["authorized_optimized_target_exposure"] = alias
    with pytest.raises(split_application.BalancedSplitApplicationError):
        split_application._require_zero_credit_boundary(boundary)


def test_zero_credit_boundary_rejects_positive_exposure() -> None:
    boundary = copy.deepcopy(VALID_BOUNDARY)
    boundary["authorized_optimized_target_exposure"] = 1
    with pytest.raises(split_application.BalancedSplitApplicationError):
        split_application._require_zero_credit_boundary(boundary)


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_zero_credit_boundary_is_closed_world(mutation: str) -> None:
    boundary = copy.deepcopy(VALID_BOUNDARY)
    if mutation == "missing":
        boundary.pop("final_test_outcomes_read")
    else:
        boundary["unexpected"] = False
    with pytest.raises(split_application.BalancedSplitApplicationError):
        split_application._require_zero_credit_boundary(boundary)


def test_application_binds_current_main_split_blob_and_rejects_stale_blob() -> None:
    assert split_application.CANONICAL_SPLIT_GIT_BLOB_SHA1 == CURRENT_SPLIT_BLOB_SHA1
    raw = _raw_records()
    selection = _selection(raw)
    kwargs = _kwargs(selection)
    kwargs["expected_split_git_blob_sha1"] = STALE_SPLIT_BLOB_SHA1
    with pytest.raises(
        split_application.BalancedSplitApplicationError,
        match="canonical split mechanics blob",
    ):
        split_application.build_balanced_split_application(selection, raw, **kwargs)


@pytest.mark.parametrize(
    "field",
    [
        "training_eligible",
        "evaluation_eligible",
        "tokenizer_fit_authorized",
        "model_training_authorized",
        "paid_compute_authorized",
        "final_test_outcomes_read",
    ],
)
def test_verifier_rejects_self_rehashed_false_integer_alias(field: str) -> None:
    raw, selection, application = _build()
    application["claim_boundary"][field] = 0
    _rehash(application, "application_identity_sha256")
    with pytest.raises(split_application.BalancedSplitApplicationError):
        split_application.verify_balanced_split_application(
            application,
            selection,
            raw,
            **_kwargs(selection),
        )


def test_verifier_rejects_self_rehashed_exposure_boolean_alias() -> None:
    raw, selection, application = _build()
    application["claim_boundary"]["authorized_optimized_target_exposure"] = False
    _rehash(application, "application_identity_sha256")
    with pytest.raises(split_application.BalancedSplitApplicationError):
        split_application.verify_balanced_split_application(
            application,
            selection,
            raw,
            **_kwargs(selection),
        )


def test_verifier_rejects_self_rehashed_top_level_int_float_alias() -> None:
    raw, selection, application = _build()
    application["selected_record_count"] = float(application["selected_record_count"])
    _rehash(application, "application_identity_sha256")
    with pytest.raises(
        split_application.BalancedSplitApplicationError,
        match="semantic content mismatch",
    ):
        split_application.verify_balanced_split_application(
            application,
            selection,
            raw,
            **_kwargs(selection),
        )


def test_verifier_rejects_self_rehashed_nested_int_float_alias() -> None:
    raw, selection, application = _build()
    shared_train = application["split_family"]["shared_train_documents"]
    application["split_family"]["shared_train_documents"] = float(shared_train)
    _rehash(application, "application_identity_sha256")
    with pytest.raises(
        split_application.BalancedSplitApplicationError,
        match="semantic content mismatch",
    ):
        split_application.verify_balanced_split_application(
            application,
            selection,
            raw,
            **_kwargs(selection),
        )