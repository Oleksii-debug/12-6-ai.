#!/usr/bin/env python3
"""Compose qualified #2782 balance inventory with only novel #2788 PEP+LoC bytes.

Execution-only bridge. It authenticates both text-free physical authorities,
discounts exact payload replay before union, authenticates the two new source
families from repository-pinned intake contracts, and delegates all mixture
and family-cap arithmetic to the unchanged NEXT100-106 gate.

No corpus, tokenizer, training, evaluation-outcome, or compute authority is
created by this runner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tools.next100_106_balance_gate as gate

BASE_HEAD = "ce30dd256e5b583ed6920d78930ee4b1a20429c5"
BASE_COMPOSITION_ID = "81fb1f11f7f6b47b2492a14acf2c34eb5b900f2df1ee47e0cb04a2b194f75cb1"
BASE_RECEIPT_ID = "87e68464e445850f334a19b6c69fef0345e56f16406dac784d1fc905381565e1"
BASE_TRUSTED_ROOT = "51ee6464ce0aeebdd133c939f65c8e04878a0cac7b3deefb65f10963d89add5a"
PEP_LOC_HEAD = "0aeddc6c32917dee110a669ffea55d55a399c1c9"
PEP_LOC_EVIDENCE_ID = "cc09b251fe1c63ff577e9d6ee52bf4005975135211d259f3ec07f7877382b3c1"
PEP_FAMILY = "en.python.peps.public-domain"
LOC_FAMILY = "en.loc.selected-digitized-books"
POLICY_ID = "9a9242f47981c25e754fc95e2650050da4e4195aa1ef3a78f2c293f9e25d7ff7"

PEP_AUTHORITY = {
    "path": "configs/data/d03_pep_public_domain_intake_v1.json",
    "blob_sha1": "0fe71942aa1f61c870a290f23a4d9b2bbb405d70",
    "family": PEP_FAMILY,
    "stratum": "en",
}
LOC_AUTHORITY = {
    "path": "configs/data/d03_common_pile_loc_intake_v1.json",
    "blob_sha1": "958339654703f3aab4ea9cc3686e4e4fd44a7e09",
    "family": LOC_FAMILY,
    "stratum": "en",
}

EXPECTED = {
    "base_records": 2533,
    "base_source_objects": 2520,
    "base_bytes": 10_692_800,
    "pep_loc_records": 432,
    "pep_loc_source_objects": 364,
    "pep_loc_bytes": 14_505_468,
    "overlap_records": 254,
    "overlap_bytes": 5_428_358,
    "novel_records": 178,
    "novel_source_objects": 110,
    "novel_bytes": 9_077_110,
    "pep_records": 150,
    "pep_source_objects": 82,
    "pep_bytes": 4_577_657,
    "loc_records": 28,
    "loc_source_objects": 28,
    "loc_bytes": 4_499_453,
    "combined_records": 2711,
    "combined_source_objects": 2630,
    "combined_bytes": 19_769_910,
}
EXPECTED_CAPACITY = {"ua": 205_315, "en": 15_481_867, "code": 4_082_728}
EXPECTED_FAMILY_COUNT = {"ua": 3, "en": 7, "code": 18}
EXPECTED_GAPS = {"ua": 8_794_685, "en": 0, "code": 0}
EXPECTED_MAX = 53_100
EXPECTED_MAX_STRATA = {"ua": 23_895, "en": 18_585, "code": 10_620}
MAX_INPUT_BYTES = 8 * 1024 * 1024
HEX = frozenset("0123456789abcdef")
INVENTORY_SCHEMA = "12-6.data526-record-inventory.v1"
COMPOSITION_SCHEMA = "12-6.d03-current-code-franko-ubuntu-pep-loc-composition.v1"
DEDUP_SCHEMA = "12-6.d03-current-code-franko-ubuntu-pep-loc-dedup-proof.v1"
RECEIPT_SCHEMA = "12-6.d03-current-code-franko-ubuntu-pep-loc-balance-execution.v1"


class ExecutionError(ValueError):
    """Raised when an input or derived authority is not exact."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ExecutionError(message)


def reject_constant(value: str) -> None:
    raise ExecutionError(f"non-finite JSON constant: {value}")


def parse_float(value: str) -> float:
    parsed = float(value)
    require(math.isfinite(parsed), "non-finite JSON number")
    return parsed


def strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    require(0 < len(raw) <= MAX_INPUT_BYTES, f"input size invalid: {path}")
    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=strict_object,
            parse_constant=reject_constant,
            parse_float=parse_float,
        )
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ExecutionError(f"invalid strict JSON: {path}") from exc
    require(type(value) is dict, f"expected JSON object: {path}")
    return value


def canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError, RecursionError) as exc:
        raise ExecutionError("cannot canonicalize authority") from exc


def sha256(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def self_hash(document: Mapping[str, Any], field: str) -> str:
    body = dict(document)
    body.pop(field, None)
    return sha256(body)


def sha(value: Any, field: str, length: int = 64) -> str:
    require(
        isinstance(value, str)
        and len(value) == length
        and value == value.lower()
        and set(value) <= HEX,
        f"{field} must be {length} lowercase hex characters",
    )
    return str(value)


def positive(value: Any, field: str) -> int:
    require(type(value) is int and value > 0, f"{field} must be positive integer")
    return int(value)


def zero_truth(document: Mapping[str, Any], label: str) -> None:
    exact = {
        "canonical_capacity_credited": 0,
        "training_authorized_bytes": 0,
        "authorized_optimized_target_exposure": 0,
        "tokenizer_fit_authorized": False,
        "training_executed": False,
        "learned_weights_created": False,
        "final_test_outcomes_read": False,
        "paid_compute_used": False,
    }
    for field, expected in exact.items():
        if field in document:
            observed = document[field]
            require(
                type(observed) is type(expected) and observed == expected,
                f"{label} widened truth boundary: {field}",
            )


def verify_inventory(
    inventory: Mapping[str, Any],
    *,
    label: str,
    expected_records: int,
    expected_bytes: int,
) -> list[dict[str, Any]]:
    require(inventory.get("schema_version") == INVENTORY_SCHEMA, f"{label} schema drift")
    rows = inventory.get("records")
    require(isinstance(rows, list) and rows, f"{label} rows missing")
    required = {
        "record_id",
        "source_id",
        "family",
        "modality",
        "payload_sha256",
        "payload_bytes",
    }
    normalized: list[dict[str, Any]] = []
    seen_records: set[str] = set()
    for index, raw in enumerate(rows):
        require(isinstance(raw, Mapping) and set(raw) == required, f"{label}[{index}] schema drift")
        row = dict(raw)
        for field in ("record_id", "source_id", "family", "modality"):
            require(isinstance(row[field], str) and bool(row[field]), f"{label}[{index}].{field} invalid")
        sha(row["payload_sha256"], f"{label}[{index}].payload_sha256")
        positive(row["payload_bytes"], f"{label}[{index}].payload_bytes")
        require(row["record_id"] not in seen_records, f"{label} duplicate record_id")
        seen_records.add(row["record_id"])
        normalized.append(row)
    normalized.sort(key=lambda row: row["record_id"])
    require(normalized == rows, f"{label} row order drift")
    require(len(rows) == expected_records, f"{label} record count drift")
    require(sum(int(row["payload_bytes"]) for row in rows) == expected_bytes, f"{label} byte count drift")
    require(inventory.get("record_count") == expected_records, f"{label} declared record count drift")
    require(inventory.get("total_payload_bytes") == expected_bytes, f"{label} declared bytes drift")
    record_root = sha256(rows)
    payload_projection = [
        {
            "record_id": row["record_id"],
            "payload_sha256": row["payload_sha256"],
            "payload_bytes": row["payload_bytes"],
        }
        for row in rows
    ]
    require(
        inventory.get("record_inventory_digest_sha256") == record_root,
        f"{label} record digest drift",
    )
    require(
        inventory.get("payload_inventory_digest_sha256") == sha256(payload_projection),
        f"{label} payload digest drift",
    )
    return normalized


def verify_base(composition: Mapping[str, Any], receipt: Mapping[str, Any]) -> list[dict[str, Any]]:
    require(
        composition.get("schema") == "12-6.d03-current-code-franko-ubuntu-composition.v1",
        "base composition schema drift",
    )
    require(composition.get("source_git_sha") == BASE_HEAD, "base composition head drift")
    require(
        composition.get("composition_identity_sha256") == BASE_COMPOSITION_ID
        and self_hash(composition, "composition_identity_sha256") == BASE_COMPOSITION_ID,
        "base composition identity drift",
    )
    require(
        composition.get("trusted_family_authority_root_sha256") == BASE_TRUSTED_ROOT,
        "base trusted-family root drift",
    )
    require(composition.get("status") == "COMPOSED_ZERO_CREDIT_PENDING_BALANCE", "base status drift")
    zero_truth(composition, "base composition")
    rows = verify_inventory(
        composition["combined_inventory"],
        label="base inventory",
        expected_records=EXPECTED["base_records"],
        expected_bytes=EXPECTED["base_bytes"],
    )
    require(
        len({row["source_id"] for row in rows}) == EXPECTED["base_source_objects"],
        "base source-object count drift",
    )
    require(
        receipt.get("schema") == "12-6.d03-current-code-franko-ubuntu-balance-execution.v1",
        "base receipt schema drift",
    )
    require(receipt.get("execution_head_sha") == BASE_HEAD, "base receipt head drift")
    require(
        receipt.get("receipt_identity_sha256") == BASE_RECEIPT_ID
        and self_hash(receipt, "receipt_identity_sha256") == BASE_RECEIPT_ID,
        "base receipt identity drift",
    )
    require(receipt.get("composition_identity_sha256") == BASE_COMPOSITION_ID, "base receipt composition drift")
    require(receipt.get("balance_policy_identity_sha256") == POLICY_ID, "base policy identity drift")
    require(receipt.get("combined_record_count") == EXPECTED["base_records"], "base receipt record drift")
    require(receipt.get("combined_source_object_count") == EXPECTED["base_source_objects"], "base receipt source drift")
    require(receipt.get("combined_total_payload_bytes") == EXPECTED["base_bytes"], "base receipt byte drift")
    zero_truth(receipt, "base receipt")
    return rows


def verify_pep_loc(evidence: Mapping[str, Any]) -> list[dict[str, Any]]:
    require(
        evidence.get("schema_version") == "12-6.d03-pep-loc-v10-decontam-g05-g06-execution.v1",
        "PEP+LoC evidence schema drift",
    )
    require(evidence.get("execution_head_sha") == PEP_LOC_HEAD, "PEP+LoC head drift")
    require(
        evidence.get("evidence_identity_sha256") == PEP_LOC_EVIDENCE_ID
        and self_hash(evidence, "evidence_identity_sha256") == PEP_LOC_EVIDENCE_ID,
        "PEP+LoC evidence identity drift",
    )
    content = evidence.get("content_boundary")
    require(
        content == {
            "raw_training_text_persisted": False,
            "raw_evaluation_text_persisted": False,
            "raw_survivor_text_persisted": False,
            "durable_output_text_free": True,
        },
        "PEP+LoC content boundary drift",
    )
    truth = evidence.get("truth_boundary")
    require(isinstance(truth, Mapping), "PEP+LoC truth boundary missing")
    require(
        truth.get("global_cross_source_dedup_complete_for_parent") is True
        and truth.get("reserved_evaluation_decontamination_complete") is True
        and truth.get("canonical_quality_privacy_complete") is True
        and truth.get("balance_diversity_retest_complete") is False
        and truth.get("family_caps_complete") is False,
        "PEP+LoC gate truth drift",
    )
    zero_truth(truth, "PEP+LoC evidence")
    rows = verify_inventory(
        evidence["survivor_inventory"],
        label="PEP+LoC inventory",
        expected_records=EXPECTED["pep_loc_records"],
        expected_bytes=EXPECTED["pep_loc_bytes"],
    )
    require(
        len({row["source_id"] for row in rows}) == EXPECTED["pep_loc_source_objects"],
        "PEP+LoC source-object count drift",
    )
    gate_exec = evidence.get("gate_execution")
    require(isinstance(gate_exec, Mapping), "PEP+LoC gate execution missing")
    require(gate_exec.get("survivor_records") == EXPECTED["pep_loc_records"], "PEP+LoC survivor count drift")
    require(gate_exec.get("survivor_payload_bytes") == EXPECTED["pep_loc_bytes"], "PEP+LoC survivor bytes drift")
    return rows


def git_blob_sha1(raw: bytes) -> str:
    header = f"blob {len(raw)}".encode("ascii") + bytes((0,))
    return hashlib.sha1(header + raw, usedforsecurity=False).hexdigest()


def authenticate_new_family(meta: Mapping[str, str]) -> dict[str, Any]:
    path = ROOT / meta["path"]
    require(path.is_file() and not path.is_symlink(), f"family authority not regular file: {meta['path']}")
    raw = path.read_bytes()
    require(git_blob_sha1(raw) == meta["blob_sha1"], f"family authority blob drift: {meta['path']}")
    document = json.loads(raw.decode("utf-8"))
    require(document.get("source_family") == meta["family"], f"family declaration drift: {meta['family']}")
    zero_truth(document, f"family authority {meta['family']}")
    identity = sha256(
        {
            "authority_git_blob_sha1": meta["blob_sha1"],
            "source_family": meta["family"],
            "canonical_stratum": meta["stratum"],
        }
    )
    return {
        "family": meta["family"],
        "source_family_identity_sha256": identity,
        "language": "en",
        "modalities": ["text"],
        "stratum": "en",
        "authority_path": meta["path"],
        "authority_git_blob_sha1": meta["blob_sha1"],
    }


def classify_stratum(row: Mapping[str, Any]) -> str:
    modality = row["modality"]
    family = row["family"]
    if modality == "code":
        return "code"
    if modality == "uk" or str(family).startswith("ua."):
        return "ua"
    require(modality == "en", f"unsupported non-code modality: {modality}")
    return "en"


def execute(
    *,
    base_composition: Mapping[str, Any],
    base_receipt: Mapping[str, Any],
    pep_loc_evidence: Mapping[str, Any],
    execution_head: str,
) -> dict[str, dict[str, Any]]:
    sha(execution_head, "execution head", 40)
    base_rows = verify_base(base_composition, base_receipt)
    pep_loc_rows = verify_pep_loc(pep_loc_evidence)

    base_record_ids = {row["record_id"] for row in base_rows}
    base_source_ids = {row["source_id"] for row in base_rows}
    base_payloads = {row["payload_sha256"]: row for row in base_rows}
    require(len(base_payloads) == len(base_rows), "base exact payload replay drift")

    overlap = [row for row in pep_loc_rows if row["payload_sha256"] in base_payloads]
    novel = [row for row in pep_loc_rows if row["payload_sha256"] not in base_payloads]
    require(len(overlap) == EXPECTED["overlap_records"], "PEP+LoC overlap count drift")
    require(sum(row["payload_bytes"] for row in overlap) == EXPECTED["overlap_bytes"], "PEP+LoC overlap bytes drift")
    require(len(novel) == EXPECTED["novel_records"], "PEP+LoC novel record count drift")
    require(sum(row["payload_bytes"] for row in novel) == EXPECTED["novel_bytes"], "PEP+LoC novel byte count drift")
    require(not (base_record_ids & {row["record_id"] for row in novel}), "novel record_id collision")
    require(not (base_source_ids & {row["source_id"] for row in novel}), "novel source_id collision")
    require(len({row["payload_sha256"] for row in novel}) == len(novel), "novel exact payload replay")
    require(
        {row["family"] for row in novel} == {PEP_FAMILY, LOC_FAMILY},
        "novel rows are not exactly PEP+LoC",
    )
    require(
        all(row["family"] not in {PEP_FAMILY, LOC_FAMILY} for row in overlap),
        "PEP/LoC bytes unexpectedly overlap the qualified base",
    )

    novel_facts: dict[str, tuple[int, int, int]] = {}
    for family in (PEP_FAMILY, LOC_FAMILY):
        rows = [row for row in novel if row["family"] == family]
        novel_facts[family] = (
            len(rows),
            len({row["source_id"] for row in rows}),
            sum(row["payload_bytes"] for row in rows),
        )
    require(
        novel_facts[PEP_FAMILY]
        == (EXPECTED["pep_records"], EXPECTED["pep_source_objects"], EXPECTED["pep_bytes"]),
        "PEP novel facts drift",
    )
    require(
        novel_facts[LOC_FAMILY]
        == (EXPECTED["loc_records"], EXPECTED["loc_source_objects"], EXPECTED["loc_bytes"]),
        "LoC novel facts drift",
    )

    combined_rows = sorted([*base_rows, *novel], key=lambda row: row["record_id"])
    require(len(combined_rows) == EXPECTED["combined_records"], "combined record count drift")
    require(
        len({row["source_id"] for row in combined_rows}) == EXPECTED["combined_source_objects"],
        "combined source-object count drift",
    )
    require(
        len({row["payload_sha256"] for row in combined_rows}) == len(combined_rows),
        "combined exact payload replay remains",
    )
    require(
        sum(row["payload_bytes"] for row in combined_rows) == EXPECTED["combined_bytes"],
        "combined byte count drift",
    )

    payload_projection = [
        {
            "record_id": row["record_id"],
            "payload_sha256": row["payload_sha256"],
            "payload_bytes": row["payload_bytes"],
        }
        for row in combined_rows
    ]
    combined_inventory = {
        "schema_version": INVENTORY_SCHEMA,
        "record_count": len(combined_rows),
        "total_payload_bytes": sum(row["payload_bytes"] for row in combined_rows),
        "record_inventory_digest_sha256": sha256(combined_rows),
        "payload_inventory_digest_sha256": sha256(payload_projection),
        "records": combined_rows,
    }

    base_family_rows = base_composition.get("families")
    require(isinstance(base_family_rows, list) and len(base_family_rows) == 26, "base family projection drift")
    family_map = {row["family"]: dict(row) for row in base_family_rows}
    require(len(family_map) == len(base_family_rows), "base duplicate family")
    pep_authority = authenticate_new_family(PEP_AUTHORITY)
    loc_authority = authenticate_new_family(LOC_AUTHORITY)
    for authority in (pep_authority, loc_authority):
        family = authority["family"]
        require(family not in family_map, f"new family already present in base: {family}")
        family_map[family] = {
            "family": family,
            "source_family_identity_sha256": authority["source_family_identity_sha256"],
            "stratum": "en",
            "capacity_bytes": sum(
                row["payload_bytes"] for row in novel if row["family"] == family
            ),
        }

    observed_family_bytes: dict[str, int] = defaultdict(int)
    observed_stratum_bytes = {"ua": 0, "en": 0, "code": 0}
    for row in combined_rows:
        observed_family_bytes[row["family"]] += row["payload_bytes"]
        observed_stratum_bytes[classify_stratum(row)] += row["payload_bytes"]
    require(set(observed_family_bytes) == set(family_map), "family projection/inventory coverage drift")
    for family, capacity in observed_family_bytes.items():
        require(family_map[family]["capacity_bytes"] == capacity, f"family capacity drift: {family}")
    require(observed_stratum_bytes == EXPECTED_CAPACITY, "combined stratum capacity drift")

    families = [family_map[name] for name in sorted(family_map)]
    counts = {
        stratum: sum(1 for row in families if row["stratum"] == ("uk" if stratum == "ua" else stratum))
        for stratum in ("ua", "en", "code")
    }
    require(counts == EXPECTED_FAMILY_COUNT, "combined family count drift")

    extension_core = {
        "schema": "12-6.d03-trusted-family-extension-pep-loc.v1",
        "base_trusted_family_authority_root_sha256": BASE_TRUSTED_ROOT,
        "base_family_count": len(base_family_rows),
        "added_families": [pep_authority, loc_authority],
        "family_count_total": len(families),
        "canonical_capacity_credited": 0,
        "training_authorized_bytes": 0,
        "tokenizer_fit_authorized": False,
        "training_executed": False,
    }
    family_extension = {
        **extension_core,
        "authority_root_sha256": sha256(extension_core),
    }

    dedup_core = {
        "schema": DEDUP_SCHEMA,
        "base_execution_head_sha": BASE_HEAD,
        "base_composition_identity_sha256": BASE_COMPOSITION_ID,
        "pep_loc_execution_head_sha": PEP_LOC_HEAD,
        "pep_loc_evidence_identity_sha256": PEP_LOC_EVIDENCE_ID,
        "candidate_pep_loc_records": EXPECTED["pep_loc_records"],
        "candidate_pep_loc_payload_bytes": EXPECTED["pep_loc_bytes"],
        "discounted_exact_replay_records": EXPECTED["overlap_records"],
        "discounted_exact_replay_payload_bytes": EXPECTED["overlap_bytes"],
        "novel_pep_loc_records": EXPECTED["novel_records"],
        "novel_pep_loc_payload_bytes": EXPECTED["novel_bytes"],
        "cross_inventory_record_id_collision_free_for_novel_rows": True,
        "cross_inventory_source_id_collision_free_for_novel_rows": True,
        "combined_exact_payload_collision_free": True,
        "terminal_verdict": "PASS",
        "canonical_capacity_credited": 0,
        "training_authorized_bytes": 0,
    }
    dedup_proof = {**dedup_core, "evidence_identity_sha256": sha256(dedup_core)}

    composition_core = {
        "schema": COMPOSITION_SCHEMA,
        "status": "COMPOSED_ZERO_CREDIT_PENDING_BALANCE",
        "source_git_sha": execution_head,
        "parents": {
            "base_balance_head_sha": BASE_HEAD,
            "base_composition_identity_sha256": BASE_COMPOSITION_ID,
            "base_receipt_identity_sha256": BASE_RECEIPT_ID,
            "pep_loc_post_qp_head_sha": PEP_LOC_HEAD,
            "pep_loc_evidence_identity_sha256": PEP_LOC_EVIDENCE_ID,
        },
        "combined_inventory": combined_inventory,
        "families": families,
        "stratum_capacity_bytes": {"code": EXPECTED_CAPACITY["code"], "en": EXPECTED_CAPACITY["en"], "uk": EXPECTED_CAPACITY["ua"]},
        "stratum_family_counts": {"code": EXPECTED_FAMILY_COUNT["code"], "en": EXPECTED_FAMILY_COUNT["en"], "uk": EXPECTED_FAMILY_COUNT["ua"]},
        "family_authority_extension_root_sha256": family_extension["authority_root_sha256"],
        "combined_dedup_proof_identity_sha256": dedup_proof["evidence_identity_sha256"],
        "next_gate": "NEXT100-106_BALANCE_FAMILY_CAP",
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
    composition = {
        **composition_core,
        "composition_identity_sha256": sha256(composition_core),
    }

    vector_families = [
        {
            "family_id": row["family"],
            "stratum": "ua" if row["stratum"] == "uk" else row["stratum"],
            "unique_bytes": row["capacity_bytes"],
        }
        for row in families
    ]
    vector = {
        "schema_version": gate.INPUT_SCHEMA,
        "terminal": True,
        "dedup_authority": {
            "worker_id": "D03-CURRENT-CODE-FRANKO-UBUNTU-PEP-LOC-UNION-V1",
            "head_sha": execution_head,
            "evidence_identity_sha256": dedup_proof["evidence_identity_sha256"],
            "terminal_verdict": "PASS",
        },
        "families": vector_families,
        "totals": {
            "total_unique_bytes": EXPECTED["combined_bytes"],
            "by_stratum": EXPECTED_CAPACITY,
            "family_count": EXPECTED_FAMILY_COUNT,
        },
    }
    policy = gate.load_json(gate.POLICY_PATH)
    gate.validate_policy(policy)
    require(policy.get("policy_identity_sha256") == POLICY_ID, "NEXT100 policy identity drift")
    gate.validate_vector(vector)
    balance = gate.evaluate(policy, vector)

    require(balance.get("status") == "PARTIAL_MIX_FEASIBLE_ACQUIRE_MORE_DATA", "balance status drift")
    require(balance.get("maximum_feasible_total_source_bytes") == EXPECTED_MAX, "maximum feasible mix drift")
    require(balance.get("maximum_feasible_stratum_bytes") == EXPECTED_MAX_STRATA, "maximum stratum mix drift")
    require(balance.get("raw_capacity_by_stratum") == EXPECTED_CAPACITY, "balance capacity drift")
    require(balance.get("raw_gap_to_target_by_stratum") == EXPECTED_GAPS, "balance gap drift")
    require(balance.get("family_minimum") == {
        "required_per_stratum": 2,
        "observed": EXPECTED_FAMILY_COUNT,
        "pass": True,
    }, "family minimum drift")

    receipt_core = {
        "schema": RECEIPT_SCHEMA,
        "execution_profile": "LOCAL_FREE",
        "execution_head_sha": execution_head,
        "base_composition_identity_sha256": BASE_COMPOSITION_ID,
        "base_receipt_identity_sha256": BASE_RECEIPT_ID,
        "pep_loc_evidence_identity_sha256": PEP_LOC_EVIDENCE_ID,
        "family_authority_extension_root_sha256": family_extension["authority_root_sha256"],
        "combined_dedup_proof_identity_sha256": dedup_proof["evidence_identity_sha256"],
        "composition_identity_sha256": composition["composition_identity_sha256"],
        "balance_policy_identity_sha256": POLICY_ID,
        "balance_result_identity_sha256": balance["result_identity_sha256"],
        "discounted_exact_replay_records": EXPECTED["overlap_records"],
        "discounted_exact_replay_payload_bytes": EXPECTED["overlap_bytes"],
        "novel_pep_loc_records": EXPECTED["novel_records"],
        "novel_pep_loc_payload_bytes": EXPECTED["novel_bytes"],
        "combined_record_count": EXPECTED["combined_records"],
        "combined_source_object_count": EXPECTED["combined_source_objects"],
        "combined_total_payload_bytes": EXPECTED["combined_bytes"],
        "raw_capacity_by_stratum": EXPECTED_CAPACITY,
        "family_minimum": balance["family_minimum"],
        "maximum_feasible_total_source_bytes": EXPECTED_MAX,
        "maximum_feasible_stratum_bytes": EXPECTED_MAX_STRATA,
        "raw_gap_to_target_by_stratum": EXPECTED_GAPS,
        "balance_status": balance["status"],
        "next_scientific_gate": "ACQUIRE_MORE_DIVERSE_LAWFUL_UA_SOURCE_CAPACITY",
        "canonical_capacity_credited": 0,
        "training_authorized_bytes": 0,
        "authorized_unique_loss_positions": 0,
        "authorized_optimized_target_exposure": 0,
        "tokenizer_fit_authorized": False,
        "model_training_authorized": False,
        "optimizer_updates_executed_on_real_targets": 0,
        "training_executed": False,
        "learned_weights_created": False,
        "final_test_outcomes_read": False,
        "paid_compute_used": False,
    }
    receipt = {**receipt_core, "receipt_identity_sha256": sha256(receipt_core)}
    return {
        "family-authority-extension": family_extension,
        "combined-dedup-proof": dedup_proof,
        "composition": composition,
        "next100-input": vector,
        "balance-result": balance,
        "execution-receipt": receipt,
    }


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_bytes(canonical(value) + b"\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--base-composition-json", type=Path, required=True)
    parser.add_argument("--base-receipt-json", type=Path, required=True)
    parser.add_argument("--pep-loc-evidence-json", type=Path, required=True)
    parser.add_argument("--execution-head", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = execute(
            base_composition=load(args.base_composition_json),
            base_receipt=load(args.base_receipt_json),
            pep_loc_evidence=load(args.pep_loc_evidence_json),
            execution_head=args.execution_head,
        )
        args.output_dir.mkdir(parents=True, exist_ok=False)
        for name, value in result.items():
            write_json(args.output_dir / f"{name}.json", value)
    except (ExecutionError, gate.GateError, OSError, ValueError) as exc:
        print(f"BLOCKED: {exc}")
        return 2

    receipt = result["execution-receipt"]
    print("D03_PEP_LOC_COMBINED_BALANCE=" + receipt["balance_status"])
    print("COMBINED_TOTAL_PAYLOAD_BYTES=" + str(receipt["combined_total_payload_bytes"]))
    print("NOVEL_PEP_LOC_PAYLOAD_BYTES=" + str(receipt["novel_pep_loc_payload_bytes"]))
    print("UA_RAW_GAP_BYTES=" + str(receipt["raw_gap_to_target_by_stratum"]["ua"]))
    print("RECEIPT_IDENTITY_SHA256=" + receipt["receipt_identity_sha256"])
    print("CANONICAL_CAPACITY_CREDITED=0")
    print("TOKENIZER_FIT_AUTHORIZED=false")
    print("TRAINING_EXECUTED=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
