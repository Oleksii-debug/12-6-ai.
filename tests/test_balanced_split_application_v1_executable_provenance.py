from __future__ import annotations

import hashlib
import json
import sys
import types
from typing import Any

import pytest

from twelve_six.data import balanced_split_application_v1 as split_application
import twelve_six.split_robustness as ambient_split

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
        "expected_selection_identity_sha256": selection[
            "balanced_selection_identity_sha256"
        ],
        "expected_retained_inventory_identity_sha256": IDS["inventory"],
        "expected_decontamination_authority_sha256": IDS["decontam"],
        "expected_dedup_authority_sha256": IDS["dedup"],
        "expected_balance_policy_identity_sha256": IDS["policy"],
        "expected_balance_result_identity_sha256": IDS["balance"],
        "expected_split_git_blob_sha1": split_application.CANONICAL_SPLIT_GIT_BLOB_SHA1,
        "variant_seeds": split_application.CANONICAL_SPLIT_VARIANT_SEEDS,
        "validation_fraction": split_application.CANONICAL_SPLIT_VALIDATION_FRACTION,
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


def _explode(*args: object, **kwargs: object) -> Any:
    raise AssertionError("mutable ambient split mechanics were executed")


def test_checkout_source_bytes_are_exact_canonical_git_blob() -> None:
    source = split_application._read_canonical_split_source()
    assert split_application._git_blob_sha1(source) == split_application.CANONICAL_SPLIT_GIT_BLOB_SHA1


def test_tampered_checkout_source_bytes_fail_before_mechanics_execute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = split_application._read_canonical_split_source()
    monkeypatch.setattr(
        split_application,
        "_read_canonical_split_source",
        lambda: original + b"\n# provenance tamper\n",
    )
    raw = _raw_records()
    selection = _selection(raw)
    with pytest.raises(
        split_application.BalancedSplitApplicationError,
        match="executable provenance mismatch",
    ):
        split_application.build_balanced_split_application(
            selection,
            raw,
            **_kwargs(selection),
        )


def test_ambient_callable_monkeypatch_cannot_change_executed_split_mechanics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ambient_split, "eligible_corpus_identity", _explode)
    monkeypatch.setattr(ambient_split, "dedup_relations_identity", _explode)
    monkeypatch.setattr(ambient_split, "build_split_family", _explode)
    monkeypatch.setattr(ambient_split, "verify_split_family_manifest", _explode)

    raw, selection, application = _build()
    assert application["status"] == "PASS_ZERO_CREDIT"
    split_application.verify_balanced_split_application(
        application,
        selection,
        raw,
        **_kwargs(selection),
    )


def test_alternate_sys_modules_split_module_cannot_supply_mechanics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = types.ModuleType("twelve_six.split_robustness")
    for name in split_application._REQUIRED_SPLIT_EXPORTS:
        setattr(fake, name, _explode)
    monkeypatch.setitem(sys.modules, "twelve_six.split_robustness", fake)

    raw, selection, application = _build()
    assert application["canonical_split_git_blob_sha1"] == (
        split_application.CANONICAL_SPLIT_GIT_BLOB_SHA1
    )
    split_application.verify_balanced_split_application(
        application,
        selection,
        raw,
        **_kwargs(selection),
    )


def test_authenticated_module_exports_close_over_private_exact_byte_namespace() -> None:
    module = split_application._load_canonical_split_module(
        split_application.CANONICAL_SPLIT_GIT_BLOB_SHA1
    )
    assert module.build_split_family is not ambient_split.build_split_family
    for name in split_application._SPLIT_FUNCTION_EXPORTS:
        assert getattr(module, name).__globals__ is module.__dict__
