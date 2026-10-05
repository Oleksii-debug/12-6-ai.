#!/usr/bin/env python3
"""Execute canonical NEXT100-106 balance on independently qualified PEP+LoC post-QP survivors.

Execution-only bridge. It authenticates the exact text-free #2788 two-clean
artifact, re-derives record/payload inventory roots and family capacities, binds
the two newly admitted English families to their pinned intake contracts, and
delegates all mixture/cap mathematics to tools.next100_106_balance_gate.

It creates no corpus, tokenizer, training, final-test, or paid-compute authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import tools.next100_106_balance_gate as gate
from twelve_six.data import common_pile_loc_intake, pep_intake
from twelve_six.data.trusted_family_authority_current import (
    trusted_family_authority_root_sha256,
    trusted_family_projection,
)

MAX_INPUT_BYTES = 2 * 1024 * 1024
EVIDENCE_SCHEMA = "12-6.d03-pep-loc-v10-decontam-g05-g06-execution.v1"
PROOF_SCHEMA = "12-6.d03-pep-loc-v10-post-qp-two-clean.v1"
FAMILY_AUTHORITY_SCHEMA = "12-6.d03-pep-loc-postqp-family-authority.v1"
RECEIPT_SCHEMA = "12-6.d03-pep-loc-postqp-balance-execution.v1"

PARENT_EXECUTION_HEAD = "0aeddc6c32917dee110a669ffea55d55a399c1c9"
PARENT_ARTIFACT_ID = 11359730581
PARENT_ARTIFACT_ZIP_SHA256 = (
    "aabb3b4e167c7bad739e39c64b29af7e9a56c1bfb72e88016f05b02ccb0fcd67"
)
PARENT_EVIDENCE_FILE_SHA256 = (
    "e535ef7525abd591dd27fd4397ce2fd160569797daba4759eeddaa3b7d3102be"
)
PARENT_EVIDENCE_IDENTITY_SHA256 = (
    "cc09b251fe1c63ff577e9d6ee52bf4005975135211d259f3ec07f7877382b3c1"
)
PARENT_PROOF_FILE_SHA256 = (
    "e6dad3687fbb566bff4d8864abfb85ed343213ba3600d4c43ed6d221d2f8e7f5"
)
PARENT_PROOF_IDENTITY_SHA256 = (
    "53348e0f124575bdbfb659ffb33c8255ed27dd868f282d24f4f71a4f6eedd811"
)
GLOBAL_DEDUP_PARENT_HEAD = "935d508130b4ba1cb5a3a81c2272fd4d19552579"
GLOBAL_DEDUP_SURVIVOR_AUTHORITY_SHA256 = (
    "01727ad4796cddd0a6c4ec6c2add9fda0c1e523155f4f9fa024d7be0956613c9"
)
GLOBAL_DEDUP_TWO_CLEAN_PROOF_SHA256 = (
    "bb0a65d958791b05df2ba47b5b108c93f39ccf59f8bf8115c00fb1f9b90d4b6c"
)
GLOBAL_DEDUP_MATCHER_REPORT_SHA256 = (
    "cf7e4b4f1758510a6f01068ac30aebb0696623664d38e13ab2290b63a6da9de0"
)
GLOBAL_DEDUP_WORKFLOW_RUN_ID = 37338684311
GLOBAL_DEDUP_PASS_ARTIFACT_ID = 11357173588
GLOBAL_DEDUP_PASS_ARTIFACT_ZIP_SHA256 = (
    "379a7984b86afde8ea3bf0d55ee9f3c19a045799b0ee106e8bb19bf4680054d5"
)
GLOBAL_DEDUP_PROOF_ARTIFACT_ID = 11357043314
GLOBAL_DEDUP_PROOF_ARTIFACT_ZIP_SHA256 = (
    "befa277e2c624b7719295907c863db383a8bc2f206271b073447f2e4d95a254c"
)

EXPECTED_RECORDS = 432
EXPECTED_SOURCE_OBJECTS = 364
EXPECTED_TOTAL_BYTES = 14_505_468
EXPECTED_RECORD_ROOT = "e9dd30db2134e771d67e253946afa85b61fcfc129164bf6890983747273f2195"
EXPECTED_PAYLOAD_ROOT = "7292b0209ef92dc45bcb06147a65562d00edf9a77e617b05ec7562cd3681a0ca"
EXPECTED_STRATUM_BYTES = {"ua": 10_632, "en": 10_830_589, "code": 3_664_247}
EXPECTED_FAMILY_COUNTS = {"ua": 2, "en": 6, "code": 12}

POLICY_IDENTITY_SHA256 = "9a9242f47981c25e754fc95e2650050da4e4195aa1ef3a78f2c293f9e25d7ff7"

PEP_FAMILY = "en.python.peps.public-domain"
LOC_FAMILY = "en.loc.selected-digitized-books"
PEP_CONFIG_PATH = Path("configs/data/d03_pep_public_domain_intake_v1.json")
LOC_CONFIG_PATH = Path("configs/data/d03_common_pile_loc_intake_v1.json")
PEP_CONFIG_BLOB_SHA1 = "0fe71942aa1f61c870a290f23a4d9b2bbb405d70"
LOC_CONFIG_BLOB_SHA1 = "958339654703f3aab4ea9cc3686e4e4fd44a7e09"
PEP_CONTRACT_IDENTITY_SHA256 = (
    "c0cd68d0b5cfd90e75993c2c9232608d22a616c7213597bb48abca5608d5578b"
)
LOC_CONTRACT_IDENTITY_SHA256 = (
    "9f6fc98e3a3b720f58963b52b3f3cf3b651bdb9e35f08fa380c99f7b7112f828"
)
ADDED_FAMILIES = {PEP_FAMILY, LOC_FAMILY}
_HEX = frozenset("0123456789abcdef")

_TOP_LEVEL_KEYS = {
    "authority_blobs",
    "content_boundary",
    "evidence_identity_sha256",
    "execution_head_sha",
    "execution_profile",
    "gate_execution",
    "parent",
    "postdedup_authority",
    "schema_version",
    "source_authority",
    "survivor_inventory",
    "truth_boundary",
}
_PARENT_KEYS = {
    "execution_head_sha",
    "matcher_report_sha256",
    "pass_artifact_id",
    "pass_artifact_zip_sha256",
    "post_global_dedup_objects",
    "post_global_dedup_payload_bytes",
    "proof_artifact_id",
    "proof_artifact_zip_sha256",
    "survivor_authority_sha256",
    "two_clean_proof_identity_sha256",
    "workflow_run_id",
}
_INVENTORY_KEYS = {
    "schema_version",
    "record_count",
    "total_payload_bytes",
    "record_inventory_digest_sha256",
    "payload_inventory_digest_sha256",
    "records",
}
_ROW_KEYS = {
    "record_id",
    "source_id",
    "family",
    "modality",
    "payload_sha256",
    "payload_bytes",
}
_PROOF_KEYS = {
    "authorized_optimized_target_exposure",
    "canonical_capacity_credited",
    "evidence_file_sha256",
    "evidence_identity_sha256",
    "execution_head_sha",
    "final_test_outcomes_read",
    "g05_partial_documents",
    "g05_partial_emitted_units",
    "input_payload_bytes",
    "input_records",
    "later_gate_loss_bytes",
    "paid_compute_used",
    "parent_execution_head_sha",
    "parent_survivor_authority_sha256",
    "proof_identity_sha256",
    "schema_version",
    "survivor_payload_bytes",
    "survivor_payload_bytes_by_modality",
    "survivor_records",
    "tokenizer_fit_authorized",
    "training_executed",
    "two_fresh_processes_byte_identical",
}


class BalanceExecutionError(ValueError):
    """Raised when exact physical or provenance authority cannot be proven."""


def _pairs_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise BalanceExecutionError("duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise BalanceExecutionError(f"non-finite JSON constant: {value}")


def _parse_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise BalanceExecutionError("non-finite JSON number")
    return parsed


def _load(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    if not raw or len(raw) > MAX_INPUT_BYTES:
        raise BalanceExecutionError(f"invalid input size: {path}")
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_pairs_without_duplicates,
            parse_constant=_reject_constant,
            parse_float=_parse_float,
        )
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise BalanceExecutionError(f"invalid strict JSON: {path}") from exc
    if not isinstance(value, dict):
        raise BalanceExecutionError(f"expected JSON object: {path}")
    return value


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError, RecursionError) as exc:
        raise BalanceExecutionError("cannot canonicalize evidence") from exc


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256(value: Any) -> str:
    return _sha256_bytes(_canonical(value))


def _git_blob_sha1(raw: bytes) -> str:
    header = f"blob {len(raw)}".encode("ascii") + b"\0"
    return hashlib.sha1(header + raw, usedforsecurity=False).hexdigest()


def _hex(value: Any, length: int, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != length
        or value != value.lower()
        or any(ch not in _HEX for ch in value)
    ):
        raise BalanceExecutionError(f"{field} must be lowercase {length}-hex")
    return value


def _positive(value: Any, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise BalanceExecutionError(f"{field} must be a positive integer")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BalanceExecutionError(message)


def _self_hash(document: Mapping[str, Any], field: str) -> str:
    body = dict(document)
    body.pop(field, None)
    return _sha256(body)


def _read_config(
    path: Path,
    *,
    expected_blob_sha1: str,
    expected_family: str,
    expected_contract_identity: str,
    validator: Any,
) -> dict[str, Any]:
    raw = path.read_bytes()
    _require(_git_blob_sha1(raw) == expected_blob_sha1, f"{path} Git blob drift")
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_pairs_without_duplicates,
            parse_constant=_reject_constant,
            parse_float=_parse_float,
        )
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise BalanceExecutionError(f"invalid source authority: {path}") from exc
    _require(isinstance(value, dict), f"source authority must be object: {path}")
    validator(value)
    _require(value.get("source_family") == expected_family, f"{path} family drift")
    _require(
        value.get("contract_identity_sha256") == expected_contract_identity,
        f"{path} contract identity drift",
    )
    return value


def _verify_proof(proof: Mapping[str, Any]) -> None:
    _require(set(proof) == _PROOF_KEYS, "two-clean proof fields drift")
    _require(proof.get("schema_version") == PROOF_SCHEMA, "two-clean proof schema drift")
    _require(proof.get("execution_head_sha") == PARENT_EXECUTION_HEAD, "proof head drift")
    _require(
        proof.get("evidence_file_sha256") == PARENT_EVIDENCE_FILE_SHA256,
        "proof evidence file digest drift",
    )
    _require(
        proof.get("evidence_identity_sha256") == PARENT_EVIDENCE_IDENTITY_SHA256,
        "proof evidence identity drift",
    )
    _require(
        proof.get("proof_identity_sha256") == PARENT_PROOF_IDENTITY_SHA256,
        "proof identity drift",
    )
    _require(
        _self_hash(proof, "proof_identity_sha256") == PARENT_PROOF_IDENTITY_SHA256,
        "proof self-hash mismatch",
    )
    _require(
        proof.get("parent_execution_head_sha") == GLOBAL_DEDUP_PARENT_HEAD,
        "proof global-dedup parent head drift",
    )
    _require(
        proof.get("parent_survivor_authority_sha256")
        == GLOBAL_DEDUP_SURVIVOR_AUTHORITY_SHA256,
        "proof global-dedup survivor authority drift",
    )
    expected = {
        "input_records": 370,
        "input_payload_bytes": 15_014_196,
        "survivor_records": EXPECTED_RECORDS,
        "survivor_payload_bytes": EXPECTED_TOTAL_BYTES,
        "g05_partial_documents": 12,
        "g05_partial_emitted_units": 80,
        "later_gate_loss_bytes": 508_728,
        "two_fresh_processes_byte_identical": True,
        "canonical_capacity_credited": 0,
        "authorized_optimized_target_exposure": 0,
        "tokenizer_fit_authorized": False,
        "training_executed": False,
        "final_test_outcomes_read": False,
        "paid_compute_used": False,
    }
    for field, wanted in expected.items():
        observed = proof.get(field)
        _require(
            type(observed) is type(wanted) and observed == wanted,
            f"proof truth drift: {field}",
        )
    _require(
        proof.get("survivor_payload_bytes_by_modality")
        == {"code": 3_664_247, "en": 10_830_589, "uk": 10_632},
        "proof modality byte vector drift",
    )


def _verify_inventory(
    inventory: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, int], int]:
    _require(set(inventory) == _INVENTORY_KEYS, "survivor inventory fields drift")
    _require(
        inventory.get("schema_version") == "12-6.data526-record-inventory.v1",
        "survivor inventory schema drift",
    )
    _require(inventory.get("record_count") == EXPECTED_RECORDS, "record count drift")
    _require(
        inventory.get("total_payload_bytes") == EXPECTED_TOTAL_BYTES,
        "survivor byte count drift",
    )
    _require(
        inventory.get("record_inventory_digest_sha256") == EXPECTED_RECORD_ROOT,
        "record inventory root drift",
    )
    _require(
        inventory.get("payload_inventory_digest_sha256") == EXPECTED_PAYLOAD_ROOT,
        "payload inventory root drift",
    )

    raw_rows = inventory.get("records")
    _require(
        isinstance(raw_rows, list) and len(raw_rows) == EXPECTED_RECORDS,
        "survivor rows missing",
    )
    rows: list[dict[str, Any]] = []
    record_ids: set[str] = set()
    payload_hashes: set[str] = set()
    source_ids: set[str] = set()
    family_bytes: defaultdict[str, int] = defaultdict(int)
    family_records: Counter[str] = Counter()
    modality_bytes: defaultdict[str, int] = defaultdict(int)
    previous_record_id: str | None = None

    for index, raw in enumerate(raw_rows):
        _require(isinstance(raw, Mapping), f"inventory row {index} is not object")
        _require(set(raw) == _ROW_KEYS, f"inventory row {index} fields drift")
        record_id = raw.get("record_id")
        source_id = raw.get("source_id")
        family = raw.get("family")
        modality = raw.get("modality")
        payload_sha = raw.get("payload_sha256")
        payload_bytes = raw.get("payload_bytes")
        for value, field in (
            (record_id, "record_id"),
            (source_id, "source_id"),
            (family, "family"),
        ):
            _require(isinstance(value, str) and bool(value), f"row {index} {field} missing")
        _require(modality in {"uk", "en", "code"}, f"row {index} modality drift")
        _hex(payload_sha, 64, f"row {index} payload_sha256")
        _positive(payload_bytes, f"row {index} payload_bytes")
        _require(record_id not in record_ids, f"duplicate record_id: {record_id}")
        _require(payload_sha not in payload_hashes, "final exact payload collision")
        _require(
            previous_record_id is None or record_id > previous_record_id,
            "survivor inventory is not strictly record_id-sorted",
        )
        previous_record_id = record_id
        record_ids.add(record_id)
        payload_hashes.add(payload_sha)
        source_ids.add(source_id)
        family_bytes[family] += payload_bytes
        family_records[family] += 1
        modality_bytes[modality] += payload_bytes
        rows.append(
            {
                "record_id": record_id,
                "source_id": source_id,
                "family": family,
                "modality": modality,
                "payload_sha256": payload_sha,
                "payload_bytes": payload_bytes,
            }
        )

    _require(len(source_ids) == EXPECTED_SOURCE_OBJECTS, "source-object count drift")
    _require(
        sum(row["payload_bytes"] for row in rows) == EXPECTED_TOTAL_BYTES,
        "row byte sum drift",
    )
    _require(
        dict(modality_bytes) == {"code": 3_664_247, "en": 10_830_589, "uk": 10_632},
        "row modality bytes drift",
    )
    payload_projection = [
        {
            "record_id": row["record_id"],
            "payload_sha256": row["payload_sha256"],
            "payload_bytes": row["payload_bytes"],
        }
        for row in rows
    ]
    _require(_sha256(rows) == EXPECTED_RECORD_ROOT, "recomputed record root drift")
    _require(
        _sha256(payload_projection) == EXPECTED_PAYLOAD_ROOT,
        "recomputed payload root drift",
    )
    return rows, dict(family_bytes), dict(family_records), len(source_ids)


def _verify_evidence(
    evidence: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, int], int]:
    _require(set(evidence) == _TOP_LEVEL_KEYS, "evidence fields drift")
    _require(evidence.get("schema_version") == EVIDENCE_SCHEMA, "evidence schema drift")
    _require(evidence.get("execution_profile") == "LOCAL_FREE", "execution profile drift")
    _require(
        evidence.get("execution_head_sha") == PARENT_EXECUTION_HEAD,
        "evidence head drift",
    )
    _require(
        evidence.get("evidence_identity_sha256") == PARENT_EVIDENCE_IDENTITY_SHA256,
        "evidence identity drift",
    )
    _require(
        _self_hash(evidence, "evidence_identity_sha256")
        == PARENT_EVIDENCE_IDENTITY_SHA256,
        "evidence self-hash mismatch",
    )

    parent = evidence.get("parent")
    _require(
        isinstance(parent, Mapping) and set(parent) == _PARENT_KEYS,
        "parent fields drift",
    )
    _require(
        parent.get("execution_head_sha") == GLOBAL_DEDUP_PARENT_HEAD,
        "parent head drift",
    )
    _require(
        parent.get("survivor_authority_sha256")
        == GLOBAL_DEDUP_SURVIVOR_AUTHORITY_SHA256,
        "parent survivor authority drift",
    )
    _require(
        parent.get("two_clean_proof_identity_sha256")
        == GLOBAL_DEDUP_TWO_CLEAN_PROOF_SHA256,
        "parent two-clean proof drift",
    )
    expected_parent = {
        "workflow_run_id": GLOBAL_DEDUP_WORKFLOW_RUN_ID,
        "pass_artifact_id": GLOBAL_DEDUP_PASS_ARTIFACT_ID,
        "pass_artifact_zip_sha256": GLOBAL_DEDUP_PASS_ARTIFACT_ZIP_SHA256,
        "proof_artifact_id": GLOBAL_DEDUP_PROOF_ARTIFACT_ID,
        "proof_artifact_zip_sha256": GLOBAL_DEDUP_PROOF_ARTIFACT_ZIP_SHA256,
        "matcher_report_sha256": GLOBAL_DEDUP_MATCHER_REPORT_SHA256,
    }
    for field, wanted in expected_parent.items():
        observed = parent.get(field)
        _require(
            type(observed) is type(wanted) and observed == wanted,
            f"parent lineage drift: {field}",
        )
    _require(
        parent.get("post_global_dedup_objects") == 370,
        "parent object count drift",
    )
    _require(
        parent.get("post_global_dedup_payload_bytes") == 15_014_196,
        "parent byte count drift",
    )

    content = evidence.get("content_boundary")
    _require(
        content
        == {
            "raw_training_text_persisted": False,
            "raw_evaluation_text_persisted": False,
            "raw_survivor_text_persisted": False,
            "durable_output_text_free": True,
        },
        "content boundary drift",
    )
    truth = evidence.get("truth_boundary")
    expected_truth = {
        "global_cross_source_dedup_complete_for_parent": True,
        "reserved_evaluation_decontamination_complete": True,
        "canonical_quality_privacy_complete": True,
        "balance_diversity_retest_complete": False,
        "family_caps_complete": False,
        "cluster_safe_split_complete": False,
        "deterministic_pack_two_clean_complete": False,
        "positive_exact_unique_loss_ledger": False,
        "canonical_capacity_credited": 0,
        "training_authorized_bytes": 0,
        "authorized_unique_loss_positions": 0,
        "authorized_optimized_target_exposure": 0,
        "tokenizer_fit_authorized": False,
        "optimizer_updates_executed_on_real_targets": 0,
        "training_executed": False,
        "learned_weights_created": False,
        "final_test_outcomes_read": False,
        "paid_compute_used": False,
        "foreign_pretrained_weights_used": False,
    }
    _require(truth == expected_truth, "evidence truth boundary drift")

    inventory = evidence.get("survivor_inventory")
    _require(isinstance(inventory, Mapping), "survivor inventory missing")
    rows, family_bytes, family_records, source_count = _verify_inventory(inventory)

    execution = evidence.get("gate_execution")
    _require(isinstance(execution, Mapping), "gate execution missing")
    expected_execution = {
        "survivor_records": EXPECTED_RECORDS,
        "survivor_source_objects": EXPECTED_SOURCE_OBJECTS,
        "survivor_payload_bytes": EXPECTED_TOTAL_BYTES,
        "later_gate_loss_bytes": 508_728,
    }
    for field, wanted in expected_execution.items():
        _require(execution.get(field) == wanted, f"gate execution drift: {field}")
    _require(
        execution.get("survivor_payload_bytes_by_modality")
        == {"code": 3_664_247, "en": 10_830_589, "uk": 10_632},
        "gate modality byte vector drift",
    )
    _require(
        execution.get("survivor_source_family_counts")
        == dict(sorted(family_records.items())),
        "gate family record-count vector drift",
    )
    receipt = execution.get("composition_receipt")
    _require(isinstance(receipt, Mapping), "composition receipt missing")
    _require(
        receipt.get("survivor_record_inventory_digest_sha256") == EXPECTED_RECORD_ROOT,
        "composition receipt record root drift",
    )
    _require(
        receipt.get("survivor_payload_inventory_digest_sha256") == EXPECTED_PAYLOAD_ROOT,
        "composition receipt payload root drift",
    )
    _require(
        receipt.get("survivor_records") == EXPECTED_RECORDS,
        "receipt record count drift",
    )
    _require(
        receipt.get("survivor_source_objects") == EXPECTED_SOURCE_OBJECTS,
        "receipt source count drift",
    )
    _require(
        receipt.get("survivor_payload_bytes") == EXPECTED_TOTAL_BYTES,
        "receipt payload bytes drift",
    )
    return rows, family_bytes, family_records, source_count


def _family_authority(
    rows: list[dict[str, Any]],
    family_bytes: Mapping[str, int],
    family_records: Mapping[str, int],
) -> dict[str, Any]:
    families = set(family_bytes)
    _require(ADDED_FAMILIES <= families, "PEP/LoC families absent from survivors")
    existing = families - ADDED_FAMILIES
    projection = trusted_family_projection(existing)
    by_family = {row["family"]: row for row in projection}
    _require(set(by_family) == existing, "trusted existing-family projection drift")

    pep_config = _read_config(
        PEP_CONFIG_PATH,
        expected_blob_sha1=PEP_CONFIG_BLOB_SHA1,
        expected_family=PEP_FAMILY,
        expected_contract_identity=PEP_CONTRACT_IDENTITY_SHA256,
        validator=pep_intake.validate_config,
    )
    loc_config = _read_config(
        LOC_CONFIG_PATH,
        expected_blob_sha1=LOC_CONFIG_BLOB_SHA1,
        expected_family=LOC_FAMILY,
        expected_contract_identity=LOC_CONTRACT_IDENTITY_SHA256,
        validator=common_pile_loc_intake.validate_config,
    )

    semantics: dict[str, str] = {
        family: str(by_family[family]["stratum"]) for family in existing
    }
    semantics[PEP_FAMILY] = "en"
    semantics[LOC_FAMILY] = "en"
    for index, row in enumerate(rows):
        expected_modality = semantics.get(row["family"])
        _require(expected_modality is not None, f"unknown family in row {index}")
        _require(
            row["modality"] == expected_modality,
            f"family/modality semantic drift: {row['family']}",
        )

    family_rows = [
        {
            "family_id": family,
            "stratum": "ua" if semantics[family] == "uk" else semantics[family],
            "record_count": family_records[family],
            "unique_bytes": family_bytes[family],
        }
        for family in sorted(families)
    ]
    base_root = trusted_family_authority_root_sha256(existing)
    core = {
        "schema": FAMILY_AUTHORITY_SCHEMA,
        "parent_execution_head_sha": PARENT_EXECUTION_HEAD,
        "parent_evidence_identity_sha256": PARENT_EVIDENCE_IDENTITY_SHA256,
        "base_trusted_family_authority_root_sha256": base_root,
        "added_source_authorities": {
            PEP_FAMILY: {
                "config_path": str(PEP_CONFIG_PATH),
                "config_git_blob_sha1": PEP_CONFIG_BLOB_SHA1,
                "contract_identity_sha256": pep_config["contract_identity_sha256"],
                "canonical_stratum": "en",
            },
            LOC_FAMILY: {
                "config_path": str(LOC_CONFIG_PATH),
                "config_git_blob_sha1": LOC_CONFIG_BLOB_SHA1,
                "contract_identity_sha256": loc_config["contract_identity_sha256"],
                "canonical_stratum": "en",
            },
        },
        "families": family_rows,
        "record_inventory_digest_sha256": EXPECTED_RECORD_ROOT,
        "payload_inventory_digest_sha256": EXPECTED_PAYLOAD_ROOT,
        "canonical_capacity_credited": 0,
        "training_authorized_bytes": 0,
    }
    return {**core, "family_authority_identity_sha256": _sha256(core)}


def execute(
    evidence: Mapping[str, Any],
    proof: Mapping[str, Any],
    *,
    source_git_sha: str,
) -> dict[str, dict[str, Any]]:
    source_git_sha = _hex(source_git_sha, 40, "source Git SHA")
    _verify_proof(proof)
    rows, family_bytes, family_records, source_count = _verify_evidence(evidence)
    family_authority = _family_authority(rows, family_bytes, family_records)

    by_stratum = {"ua": 0, "en": 0, "code": 0}
    family_count = {"ua": 0, "en": 0, "code": 0}
    vector_families: list[dict[str, Any]] = []
    for row in family_authority["families"]:
        stratum = row["stratum"]
        capacity = row["unique_bytes"]
        vector_families.append(
            {
                "family_id": row["family_id"],
                "stratum": stratum,
                "unique_bytes": capacity,
            }
        )
        by_stratum[stratum] += capacity
        family_count[stratum] += 1

    _require(by_stratum == EXPECTED_STRATUM_BYTES, "derived stratum bytes drift")
    _require(family_count == EXPECTED_FAMILY_COUNTS, "derived family counts drift")
    vector = {
        "schema_version": gate.INPUT_SCHEMA,
        "terminal": True,
        "dedup_authority": {
            "worker_id": "D03-PEP-LOC-POSTQP-TWO-CLEAN-FINAL-UNIQUE-V1",
            "head_sha": PARENT_EXECUTION_HEAD,
            "evidence_identity_sha256": PARENT_EVIDENCE_IDENTITY_SHA256,
            "terminal_verdict": "PASS",
        },
        "families": vector_families,
        "totals": {
            "total_unique_bytes": sum(by_stratum.values()),
            "by_stratum": by_stratum,
            "family_count": family_count,
        },
        "physical_authority": {
            "execution_head_sha": PARENT_EXECUTION_HEAD,
            "artifact_id": PARENT_ARTIFACT_ID,
            "artifact_zip_sha256": PARENT_ARTIFACT_ZIP_SHA256,
            "evidence_file_sha256": PARENT_EVIDENCE_FILE_SHA256,
            "evidence_identity_sha256": PARENT_EVIDENCE_IDENTITY_SHA256,
            "proof_file_sha256": PARENT_PROOF_FILE_SHA256,
            "proof_identity_sha256": PARENT_PROOF_IDENTITY_SHA256,
            "record_inventory_digest_sha256": EXPECTED_RECORD_ROOT,
            "payload_inventory_digest_sha256": EXPECTED_PAYLOAD_ROOT,
            "record_count": EXPECTED_RECORDS,
            "source_object_count": source_count,
            "total_payload_bytes": EXPECTED_TOTAL_BYTES,
            "family_authority_identity_sha256": family_authority[
                "family_authority_identity_sha256"
            ],
        },
    }
    gate.validate_vector(vector)

    policy = gate.load_json(gate.POLICY_PATH)
    gate.validate_policy(policy)
    _require(
        policy.get("policy_identity_sha256") == POLICY_IDENTITY_SHA256,
        "canonical balance policy identity drift",
    )
    balance = gate.evaluate(policy, vector)
    _require(
        balance.get("policy_identity_sha256") == POLICY_IDENTITY_SHA256,
        "balance result policy lineage drift",
    )

    next_gate = (
        "CLUSTER_SAFE_SPLIT_AND_DETERMINISTIC_PACK"
        if balance["status"] == "TARGET_20M_SOURCE_MIX_FEASIBLE"
        else "ACQUIRE_MORE_DIVERSE_LAWFUL_SOURCE_CAPACITY"
    )
    receipt_core = {
        "schema": RECEIPT_SCHEMA,
        "execution_profile": "LOCAL_FREE",
        "execution_head_sha": source_git_sha,
        "parent_execution_head_sha": PARENT_EXECUTION_HEAD,
        "parent_artifact_id": PARENT_ARTIFACT_ID,
        "parent_artifact_zip_sha256": PARENT_ARTIFACT_ZIP_SHA256,
        "parent_evidence_file_sha256": PARENT_EVIDENCE_FILE_SHA256,
        "parent_evidence_identity_sha256": PARENT_EVIDENCE_IDENTITY_SHA256,
        "parent_proof_file_sha256": PARENT_PROOF_FILE_SHA256,
        "parent_proof_identity_sha256": PARENT_PROOF_IDENTITY_SHA256,
        "family_authority_identity_sha256": family_authority[
            "family_authority_identity_sha256"
        ],
        "balance_policy_identity_sha256": POLICY_IDENTITY_SHA256,
        "balance_result_identity_sha256": balance["result_identity_sha256"],
        "balance_status": balance["status"],
        "maximum_feasible_total_source_bytes": balance[
            "maximum_feasible_total_source_bytes"
        ],
        "maximum_feasible_stratum_bytes": balance[
            "maximum_feasible_stratum_bytes"
        ],
        "raw_capacity_by_stratum": balance["raw_capacity_by_stratum"],
        "raw_gap_to_target_by_stratum": balance["raw_gap_to_target_by_stratum"],
        "family_minimum": balance["family_minimum"],
        "survivor_record_count": EXPECTED_RECORDS,
        "survivor_source_object_count": EXPECTED_SOURCE_OBJECTS,
        "survivor_total_payload_bytes": EXPECTED_TOTAL_BYTES,
        "next_scientific_gate": next_gate,
        "canonical_capacity_credited": 0,
        "training_authorized_bytes": 0,
        "authorized_unique_loss_positions": 0,
        "authorized_optimized_target_exposure": 0,
        "tokenizer_fit_authorized": False,
        "model_training_authorized": False,
        "training_executed": False,
        "learned_weights_created": False,
        "final_test_outcomes_read": False,
        "paid_compute_used": False,
    }
    receipt = {
        **receipt_core,
        "receipt_identity_sha256": _sha256(receipt_core),
    }
    return {
        "family-authority": family_authority,
        "next100-input": vector,
        "balance-result": balance,
        "execution-receipt": receipt,
    }


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.write_bytes(_canonical(value) + b"\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--evidence-json", type=Path, required=True)
    parser.add_argument("--proof-json", type=Path, required=True)
    parser.add_argument("--source-git-sha", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        evidence = _load(args.evidence_json)
        proof = _load(args.proof_json)
        result = execute(evidence, proof, source_git_sha=args.source_git_sha)
        args.output_dir.mkdir(parents=True, exist_ok=False)
        for name, value in result.items():
            _write(args.output_dir / f"{name}.json", value)
    except (BalanceExecutionError, gate.GateError, OSError, ValueError) as exc:
        print(f"BLOCKED: {exc}")
        return 2

    receipt = result["execution-receipt"]
    print("D03_PEP_LOC_POSTQP_BALANCE=" + receipt["balance_status"])
    print(
        "MAXIMUM_FEASIBLE_TOTAL_SOURCE_BYTES="
        + str(receipt["maximum_feasible_total_source_bytes"])
    )
    print(
        "FAMILY_AUTHORITY_IDENTITY_SHA256="
        + receipt["family_authority_identity_sha256"]
    )
    print("RECEIPT_IDENTITY_SHA256=" + receipt["receipt_identity_sha256"])
    print("CANONICAL_CAPACITY_CREDITED=0")
    print("TOKENIZER_FIT_AUTHORIZED=false")
    print("TRAINING_EXECUTED=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
