from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

import pytest

from twelve_six.data.balanced_split_application_v1 import (
    CANONICAL_SPLIT_GIT_BLOB_SHA1,
    BalancedSplitApplicationError,
    balanced_record_counts,
    build_balanced_split_application,
    verify_balanced_split_application,
)

IDS = {
    "selection": hashlib.sha256(b"selection").hexdigest(),
    "inventory": hashlib.sha256(b"inventory").hexdigest(),
    "decontam": hashlib.sha256(b"decontam").hexdigest(),
    "dedup": hashlib.sha256(b"dedup").hexdigest(),
    "policy": hashlib.sha256(b"policy").hexdigest(),
    "balance": hashlib.sha256(b"balance").hexdigest(),
}
SEEDS = ("split-a", "split-b", "split-c")


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
        "variant_seeds": SEEDS,
        "validation_fraction": 0.2,
    }


def _build() -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    raw = _raw_records()
    selection = _selection(raw)
    application = build_balanced_split_application(selection, raw, **_kwargs(selection))
    return raw, selection, application


def test_build_and_verify_round_trip_is_deterministic() -> None:
    raw, selection, application = _build()
    verify_balanced_split_application(application, selection, raw, **_kwargs(selection))
    repeated = build_balanced_split_application(selection, list(reversed(raw)), **_kwargs(selection))
    assert repeated == application


def test_output_is_text_free_and_zero_credit() -> None:
    _raw, _selection_doc, application = _build()
    encoded = json.dumps(application, ensure_ascii=False)
    assert "Альфа" not in encoded
    assert "English three" not in encoded
    assert application["status"] == "PASS_ZERO_CREDIT"
    assert application["claim_boundary"]["authorized_optimized_target_exposure"] == 0
    assert application["claim_boundary"]["model_training_authorized"] is False


def test_record_observability_helper_is_non_policy_count_only() -> None:
    raw = _raw_records()
    selection = _selection(raw)
    assert balanced_record_counts(selection) == {"uk": 3, "en": 3, "code": 2}


def test_rejects_wrong_independently_expected_selection_identity() -> None:
    raw = _raw_records()
    selection = _selection(raw)
    kwargs = _kwargs(selection)
    kwargs["expected_selection_identity_sha256"] = hashlib.sha256(b"wrong").hexdigest()
    with pytest.raises(BalancedSplitApplicationError, match="expected identity"):
        build_balanced_split_application(selection, raw, **kwargs)


def test_rejects_nonterminal_selection_even_when_rehashed() -> None:
    raw = _raw_records()
    selection = _selection(raw)
    selection["terminal"] = False
    _rehash(selection, "balanced_selection_identity_sha256")
    kwargs = _kwargs(selection)
    with pytest.raises(BalancedSplitApplicationError, match="terminal PASS"):
        build_balanced_split_application(selection, raw, **kwargs)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("training_eligible", True, "zero-credit"),
        ("evaluation_eligible", True, "evaluation_eligible"),
        ("evaluation_reserved", True, "evaluation-reserved"),
        ("purpose", "benchmark", "forbidden/non-training"),
    ],
)
def test_rejects_eligibility_or_purpose_widening(
    field: str, value: Any, message: str
) -> None:
    raw = _raw_records()
    selection = _selection(raw)
    selection["records"][0][field] = value
    _rehash(selection, "balanced_selection_identity_sha256")
    with pytest.raises(BalancedSplitApplicationError, match=message):
        build_balanced_split_application(selection, raw, **_kwargs(selection))


def test_rejects_duplicate_selected_record_even_when_totals_rehashed() -> None:
    raw = _raw_records()
    selection = _selection(raw)
    selection["records"][1]["record_id"] = selection["records"][0]["record_id"]
    _rehash(selection, "balanced_selection_identity_sha256")
    with pytest.raises(BalancedSplitApplicationError, match="duplicate balanced record_id"):
        build_balanced_split_application(selection, raw, **_kwargs(selection))


def test_rejects_missing_raw_record() -> None:
    raw = _raw_records()
    selection = _selection(raw)
    with pytest.raises(BalancedSplitApplicationError, match="missing from raw input"):
        build_balanced_split_application(selection, raw[:-1], **_kwargs(selection))


def test_rejects_extra_raw_record() -> None:
    raw = _raw_records()
    selection = _selection(raw)
    extra = dict(raw[0])
    extra["record_id"] = "extra"
    with pytest.raises(BalancedSplitApplicationError, match="outside selection"):
        build_balanced_split_application(selection, [*raw, extra], **_kwargs(selection))


def test_rejects_duplicate_raw_record() -> None:
    raw = _raw_records()
    selection = _selection(raw)
    with pytest.raises(BalancedSplitApplicationError, match="duplicate raw record_id"):
        build_balanced_split_application(selection, [*raw, dict(raw[0])], **_kwargs(selection))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("family", "other-family", "family drift"),
        ("stratum", "code", "stratum drift"),
        ("near_duplicate_cluster_id", "other-cluster", "near_duplicate_cluster_id drift"),
        ("modality", "code", "modality drift"),
    ],
)
def test_rejects_raw_metadata_drift(field: str, value: str, message: str) -> None:
    raw = _raw_records()
    selection = _selection(raw)
    raw[0][field] = value
    with pytest.raises(BalancedSplitApplicationError, match=message):
        build_balanced_split_application(selection, raw, **_kwargs(selection))


def test_rejects_payload_drift() -> None:
    raw = _raw_records()
    selection = _selection(raw)
    raw[0]["normalized_payload"] += " drift"
    with pytest.raises(BalancedSplitApplicationError, match="payload bytes/hash drift"):
        build_balanced_split_application(selection, raw, **_kwargs(selection))


def test_rejects_wrong_canonical_split_blob() -> None:
    raw = _raw_records()
    selection = _selection(raw)
    kwargs = _kwargs(selection)
    kwargs["expected_split_git_blob_sha1"] = "1" * 40
    with pytest.raises(BalancedSplitApplicationError, match="merged #938"):
        build_balanced_split_application(selection, raw, **kwargs)


def test_rejects_rehashed_semantic_split_manifest_tamper() -> None:
    raw, selection, application = _build()
    tampered = copy.deepcopy(application)
    variant = tampered["split_family"]["variants"][0]
    variant["train_record_ids"] = list(reversed(variant["train_record_ids"]))
    variant_core = dict(variant)
    variant_core.pop("split_identity_sha256")
    variant["split_identity_sha256"] = hashlib.sha256(
        (json.dumps(variant_core, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()
    ).hexdigest()
    family_core = dict(tampered["split_family"])
    family_core.pop("split_family_identity_sha256")
    tampered["split_family"]["split_family_identity_sha256"] = hashlib.sha256(
        (json.dumps(family_core, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()
    ).hexdigest()
    _rehash(tampered, "application_identity_sha256")
    with pytest.raises(BalancedSplitApplicationError, match="semantic content mismatch"):
        verify_balanced_split_application(tampered, selection, raw, **_kwargs(selection))


def test_rejects_application_self_hash_tamper() -> None:
    raw, selection, application = _build()
    application["selected_source_bytes"] += 1
    with pytest.raises(BalancedSplitApplicationError, match="self-hash mismatch"):
        verify_balanced_split_application(application, selection, raw, **_kwargs(selection))


def test_rejects_selection_family_accounting_tamper_after_rehash() -> None:
    raw = _raw_records()
    selection = _selection(raw)
    family = next(iter(selection["totals"]["family_source_bytes"]))
    selection["totals"]["family_source_bytes"][family] += 1
    _rehash(selection, "balanced_selection_identity_sha256")
    with pytest.raises(BalancedSplitApplicationError, match="family byte accounting mismatch"):
        build_balanced_split_application(selection, raw, **_kwargs(selection))


def test_rejects_widened_claim_boundary_after_rehash() -> None:
    raw = _raw_records()
    selection = _selection(raw)
    selection["claim_boundary"]["model_training_authorized"] = True
    _rehash(selection, "balanced_selection_identity_sha256")
    with pytest.raises(
        BalancedSplitApplicationError,
        match="balanced selection claim boundary model_training_authorized widened",
    ):
        build_balanced_split_application(selection, raw, **_kwargs(selection))
