from __future__ import annotations

import copy
import hashlib
import json

import pytest

from twelve_six.data.balanced_split_application_v1 import (
    CANONICAL_SPLIT_GIT_BLOB_SHA1,
    CANONICAL_SPLIT_VALIDATION_FRACTION,
    CANONICAL_SPLIT_VARIANT_SEEDS,
    BalancedSplitApplicationError,
    build_balanced_split_application,
)
from twelve_six.data.split_postpack_binding_v1 import (
    SplitPostpackBindingError,
    build_split_bound_loss_documents,
)

UPSTREAM = {
    "retained": "1" * 64,
    "decontam": "2" * 64,
    "dedup": "3" * 64,
    "balance_policy": "4" * 64,
    "balance_result": "5" * 64,
}


def _cjson(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _self_hash(document: dict[str, object], identity_field: str) -> str:
    core = dict(document)
    core.pop(identity_field, None)
    return _sha256(_cjson(core))


def _fixture() -> tuple[dict[str, object], list[dict[str, object]], dict[str, object]]:
    rows: list[dict[str, object]] = []
    raw: list[dict[str, object]] = []
    family_bytes: dict[str, int] = {}
    stratum_bytes: dict[str, int] = {}

    strata = ("uk", "en", "code")
    for index in range(12):
        record_id = f"record-{index:02d}"
        source_id = f"source-{index % 4}"
        family = f"family-{index % 3}"
        stratum = strata[index % len(strata)]
        modality = "code" if stratum == "code" else "text"
        payload = f"payload {index} with enough deterministic text"
        payload_bytes = len(payload.encode("utf-8"))
        payload_sha256 = _sha256(payload.encode("utf-8"))
        cluster = f"cluster-{index:02d}"
        row = {
            "record_id": record_id,
            "source_id": source_id,
            "family": family,
            "stratum": stratum,
            "modality": modality,
            "payload_sha256": payload_sha256,
            "payload_bytes": payload_bytes,
            "near_duplicate_cluster_id": cluster,
            "purpose": "pretraining",
            "training_eligible": False,
            "evaluation_eligible": False,
            "evaluation_reserved": False,
        }
        rows.append(row)
        raw.append(
            {
                "record_id": record_id,
                "source_id": source_id,
                "family": family,
                "stratum": stratum,
                "modality": modality,
                "near_duplicate_cluster_id": cluster,
                "purpose": "pretraining",
                "training_eligible": False,
                "evaluation_eligible": False,
                "evaluation_reserved": False,
                "normalized_payload": payload,
            }
        )
        family_bytes[family] = family_bytes.get(family, 0) + payload_bytes
        stratum_bytes[stratum] = stratum_bytes.get(stratum, 0) + payload_bytes

    selection: dict[str, object] = {
        "schema": "12-6.d03-balanced-selection-authority.v1",
        "terminal": True,
        "status": "PASS",
        "balanced_selection_identity_sha256": "0" * 64,
        "retained_inventory_identity_sha256": UPSTREAM["retained"],
        "decontamination_authority_sha256": UPSTREAM["decontam"],
        "dedup_authority_sha256": UPSTREAM["dedup"],
        "balance_policy_identity_sha256": UPSTREAM["balance_policy"],
        "balance_result_identity_sha256": UPSTREAM["balance_result"],
        "records": rows,
        "totals": {
            "record_count": len(rows),
            "source_bytes": sum(row["payload_bytes"] for row in rows),
            "family_source_bytes": dict(sorted(family_bytes.items())),
            "stratum_source_bytes": dict(sorted(stratum_bytes.items())),
        },
        "claim_boundary": {
            "training_eligible": False,
            "evaluation_eligible": False,
            "tokenizer_fit_authorized": False,
            "model_training_authorized": False,
            "paid_compute_authorized": False,
            "final_test_outcomes_read": False,
            "authorized_optimized_target_exposure": 0,
        },
    }
    selection["balanced_selection_identity_sha256"] = _self_hash(
        selection,
        "balanced_selection_identity_sha256",
    )
    application = build_balanced_split_application(
        selection,
        raw,
        expected_selection_identity_sha256=selection["balanced_selection_identity_sha256"],
        expected_retained_inventory_identity_sha256=UPSTREAM["retained"],
        expected_decontamination_authority_sha256=UPSTREAM["decontam"],
        expected_dedup_authority_sha256=UPSTREAM["dedup"],
        expected_balance_policy_identity_sha256=UPSTREAM["balance_policy"],
        expected_balance_result_identity_sha256=UPSTREAM["balance_result"],
        expected_split_git_blob_sha1=CANONICAL_SPLIT_GIT_BLOB_SHA1,
        variant_seeds=CANONICAL_SPLIT_VARIANT_SEEDS,
        validation_fraction=CANONICAL_SPLIT_VALIDATION_FRACTION,
    )
    return selection, raw, application


def _stage_bindings(application: dict[str, object]) -> dict[str, str]:
    return {
        "normalization": "6" * 64,
        "evaluation_reservations": "7" * 64,
        "dedup": "8" * 64,
        "split": application["application_identity_sha256"],
        "packing": "9" * 64,
    }


def _build(
    selection: dict[str, object],
    raw: list[dict[str, object]],
    application: dict[str, object],
    *,
    stage_bindings: dict[str, str] | None = None,
):
    return build_split_bound_loss_documents(
        application,
        selection,
        raw,
        expected_selection_identity_sha256=selection["balanced_selection_identity_sha256"],
        expected_retained_inventory_identity_sha256=UPSTREAM["retained"],
        expected_decontamination_authority_sha256=UPSTREAM["decontam"],
        expected_dedup_authority_sha256=UPSTREAM["dedup"],
        expected_balance_policy_identity_sha256=UPSTREAM["balance_policy"],
        expected_balance_result_identity_sha256=UPSTREAM["balance_result"],
        expected_split_git_blob_sha1=CANONICAL_SPLIT_GIT_BLOB_SHA1,
        variant_seeds=CANONICAL_SPLIT_VARIANT_SEEDS,
        validation_fraction=CANONICAL_SPLIT_VALIDATION_FRACTION,
        stage_bindings=stage_bindings or _stage_bindings(application),
    )


def test_projects_authenticated_shared_train_and_validation_union() -> None:
    selection, raw, application = _fixture()

    documents = _build(selection, raw, application)

    train_ids = set(application["split_family"]["shared_train_record_ids"])
    validation_ids = set(application["split_family"]["validation_union_record_ids"])
    assert {document.document_id for document in documents} == train_ids | validation_ids
    assert {
        document.document_id for document in documents if document.split == "train"
    } == train_ids
    assert {
        document.document_id for document in documents if document.split == "validation"
    } == validation_ids
    assert all(
        document.evaluation_reserved is (document.document_id in validation_ids)
        for document in documents
    )
    assert all(document.retained_after_dedup is True for document in documents)
    assert all(document.reserved_target_ranges == () for document in documents)

    selected_by_id = {row["record_id"]: row for row in selection["records"]}
    for document in documents:
        selected = selected_by_id[document.document_id]
        assert document.family_id == selected["family"]
        assert document.language == selected["stratum"]
        assert document.dedup_cluster_id == selected["near_duplicate_cluster_id"]


def test_rejects_stage_binding_not_equal_to_terminal_application() -> None:
    selection, raw, application = _fixture()
    stage_bindings = _stage_bindings(application)
    stage_bindings["split"] = "a" * 64

    with pytest.raises(
        SplitPostpackBindingError,
        match="stage_bindings.split must equal",
    ):
        _build(selection, raw, application, stage_bindings=stage_bindings)


def test_rejects_coherently_resealed_caller_authored_split_substitution() -> None:
    selection, raw, application = _fixture()
    attacked = copy.deepcopy(application)
    split_family = attacked["split_family"]
    train = list(split_family["shared_train_record_ids"])
    validation = list(split_family["validation_union_record_ids"])
    moved = train.pop()
    validation.append(moved)
    validation.sort()
    split_family["shared_train_record_ids"] = train
    split_family["shared_train_documents"] = len(train)
    split_family["validation_union_record_ids"] = validation
    split_family["validation_union_documents"] = len(validation)
    split_family["split_family_identity_sha256"] = _self_hash(
        split_family,
        "split_family_identity_sha256",
    )
    attacked["application_identity_sha256"] = _self_hash(
        attacked,
        "application_identity_sha256",
    )

    with pytest.raises(BalancedSplitApplicationError):
        _build(
            selection,
            raw,
            attacked,
            stage_bindings=_stage_bindings(attacked),
        )


@pytest.mark.parametrize("attack", ["payload", "source", "cluster", "duplicate", "missing"])
def test_rejects_physical_or_metadata_drift(attack: str) -> None:
    selection, raw, application = _fixture()
    attacked = copy.deepcopy(raw)

    if attack == "payload":
        attacked[0]["normalized_payload"] += " drift"
    elif attack == "source":
        attacked[0]["source_id"] = "different-source"
    elif attack == "cluster":
        attacked[0]["near_duplicate_cluster_id"] = "different-cluster"
    elif attack == "duplicate":
        attacked.append(copy.deepcopy(attacked[0]))
    elif attack == "missing":
        attacked.pop()

    with pytest.raises((BalancedSplitApplicationError, SplitPostpackBindingError)):
        _build(selection, attacked, application)
