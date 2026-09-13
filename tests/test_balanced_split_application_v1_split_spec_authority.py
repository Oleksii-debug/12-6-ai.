from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

import pytest

from twelve_six.data.balanced_split_application_v1 import (
    CANONICAL_SPLIT_ALGORITHM,
    CANONICAL_SPLIT_GIT_BLOB_SHA1,
    CANONICAL_SPLIT_SPEC_IDENTITY_SHA256,
    CANONICAL_SPLIT_VALIDATION_FRACTION,
    CANONICAL_SPLIT_VARIANT_SEEDS,
    SPLIT_SPEC_SCHEMA,
    BalancedSplitApplicationError,
    build_balanced_split_application,
    verify_balanced_split_application,
)
from twelve_six.split_robustness import (
    SplitFamilySpec,
    SplitRecord,
    build_split_family,
    dedup_relations_identity,
    eligible_corpus_identity,
    verify_split_family_manifest,
)

IDS = {
    "inventory": hashlib.sha256(b"inventory").hexdigest(),
    "decontam": hashlib.sha256(b"decontam").hexdigest(),
    "dedup": hashlib.sha256(b"dedup").hexdigest(),
    "policy": hashlib.sha256(b"policy").hexdigest(),
    "balance": hashlib.sha256(b"balance").hexdigest(),
}


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
    _rehash(document, "balanced_selection_identity_sha256")
    return document


def _kwargs(selection: dict[str, Any]) -> dict[str, Any]:
    return {
        "expected_selection_identity_sha256": selection["balanced_selection_identity_sha256"],
        "expected_retained_inventory_identity_sha256": IDS["inventory"],
        "expected_decontamination_authority_sha256": IDS["decontam"],
        "expected_dedup_authority_sha256": IDS["dedup"],
        "expected_balance_policy_identity_sha256": IDS["policy"],
        "expected_balance_result_identity_sha256": IDS["balance"],
        "expected_split_git_blob_sha1": CANONICAL_SPLIT_GIT_BLOB_SHA1,
        "variant_seeds": CANONICAL_SPLIT_VARIANT_SEEDS,
        "validation_fraction": CANONICAL_SPLIT_VALIDATION_FRACTION,
    }


def _project(raw_records: list[dict[str, Any]]) -> list[SplitRecord]:
    projected = []
    for raw in raw_records:
        payload = raw["normalized_payload"]
        projected.append(
            SplitRecord(
                id=raw["record_id"],
                text=payload,
                source_id=raw["source_id"],
                modality=raw["modality"],
                content_sha256=hashlib.sha256(payload.encode()).hexdigest(),
                near_duplicate_cluster_id=raw["near_duplicate_cluster_id"],
                training_eligible=True,
                purpose=raw["purpose"],
            )
        )
    return sorted(projected, key=lambda record: record.id)


def _coherently_resealed_application(
    raw: list[dict[str, Any]],
    application: dict[str, Any],
    *,
    variant_seeds: tuple[str, ...],
    validation_fraction: float,
) -> dict[str, Any]:
    projected = _project(raw)
    attacker_spec = SplitFamilySpec(
        eligible_corpus_sha256=eligible_corpus_identity(projected),
        dedup_relations_sha256=dedup_relations_identity(projected),
        variant_seeds=variant_seeds,
        validation_fraction=validation_fraction,
        algorithm=CANONICAL_SPLIT_ALGORITHM,
    )
    attacker_family = build_split_family(projected, attacker_spec)
    verify_split_family_manifest(projected, attacker_family)

    attacked = copy.deepcopy(application)
    attacked["split_family"] = attacker_family
    fake_authority = {
        "schema": SPLIT_SPEC_SCHEMA,
        "canonical_split_git_blob_sha1": CANONICAL_SPLIT_GIT_BLOB_SHA1,
        "algorithm": CANONICAL_SPLIT_ALGORITHM,
        "variant_seeds": list(variant_seeds),
        "validation_fraction": validation_fraction,
    }
    attacked["split_spec_identity_sha256"] = hashlib.sha256(
        _canonical_bytes(fake_authority)
    ).hexdigest()
    _rehash(attacked, "application_identity_sha256")
    return attacked


def test_application_binds_independent_canonical_split_spec_identity() -> None:
    raw = _raw_records()
    selection = _selection(raw)
    application = build_balanced_split_application(selection, raw, **_kwargs(selection))
    assert application["split_spec_identity_sha256"] == CANONICAL_SPLIT_SPEC_IDENTITY_SHA256


@pytest.mark.parametrize(
    ("variant_seeds", "validation_fraction", "message"),
    [
        (("attacker-a", "attacker-b", "attacker-c"), 0.2, "variant_seeds"),
        (CANONICAL_SPLIT_VARIANT_SEEDS, 0.25, "validation_fraction"),
    ],
)
def test_builder_rejects_caller_selected_split_spec_substitution(
    variant_seeds: tuple[str, ...], validation_fraction: float, message: str
) -> None:
    raw = _raw_records()
    selection = _selection(raw)
    kwargs = _kwargs(selection)
    kwargs["variant_seeds"] = variant_seeds
    kwargs["validation_fraction"] = validation_fraction
    with pytest.raises(BalancedSplitApplicationError, match=message):
        build_balanced_split_application(selection, raw, **kwargs)


@pytest.mark.parametrize(
    ("variant_seeds", "validation_fraction"),
    [
        (("attacker-a", "attacker-b", "attacker-c"), 0.2),
        (CANONICAL_SPLIT_VARIANT_SEEDS, 0.25),
    ],
)
def test_public_verifier_rejects_coherent_reseal_with_matching_attacker_args(
    variant_seeds: tuple[str, ...], validation_fraction: float
) -> None:
    raw = _raw_records()
    selection = _selection(raw)
    canonical_kwargs = _kwargs(selection)
    application = build_balanced_split_application(selection, raw, **canonical_kwargs)
    attacked = _coherently_resealed_application(
        raw,
        application,
        variant_seeds=variant_seeds,
        validation_fraction=validation_fraction,
    )
    assert attacked["application_identity_sha256"] != application["application_identity_sha256"]
    assert attacked["split_spec_identity_sha256"] != CANONICAL_SPLIT_SPEC_IDENTITY_SHA256

    attacker_kwargs = dict(canonical_kwargs)
    attacker_kwargs["variant_seeds"] = variant_seeds
    attacker_kwargs["validation_fraction"] = validation_fraction
    with pytest.raises(
        BalancedSplitApplicationError,
        match="does not bind canonical split spec authority",
    ):
        verify_balanced_split_application(
            attacked,
            selection,
            raw,
            **attacker_kwargs,
        )
