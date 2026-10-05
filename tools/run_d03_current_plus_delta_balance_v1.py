#!/usr/bin/env python3
"""Execute the canonical NEXT100-106 balance gate on current-clean + D03 delta.

This runner is an execution bridge only. It authenticates the two physical
inventories, composes them through the existing current+delta authority, creates
one provenance-bound combined dedup proof, and delegates all balance mathematics
to tools.next100_106_balance_gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import tools.next100_106_balance_gate as gate
from twelve_six.data.current_plus_delta_inventory_composition_v1 import (
    compose_current_clean_and_delta_inventory,
    verify_current_clean_and_delta_composition,
)

MAX_INPUT_BYTES = 8 * 1024 * 1024
COMBINED_DEDUP_SCHEMA = "12-6.d03-current-plus-delta-global-unique-proof.v1"
RECEIPT_SCHEMA = "12-6.d03-current-plus-delta-balance-execution.v1"
STRATUM_MAP = {"uk": "ua", "en": "en", "code": "code"}
_HEX = frozenset("0123456789abcdef")


class ExecutionError(ValueError):
    """Raised when physical or policy authority cannot be proved."""


def _pairs_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ExecutionError("duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ExecutionError(f"non-finite JSON constant: {value}")


def _parse_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ExecutionError("non-finite JSON number")
    return parsed


def _load(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    if len(raw) > MAX_INPUT_BYTES:
        raise ExecutionError(f"input exceeds {MAX_INPUT_BYTES} bytes: {path}")
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_pairs_without_duplicates,
            parse_constant=_reject_constant,
            parse_float=_parse_float,
        )
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ExecutionError(f"invalid strict JSON input: {path}") from exc
    if not isinstance(value, dict):
        raise ExecutionError(f"expected one JSON object: {path}")
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
        raise ExecutionError("cannot canonicalize execution evidence") from exc


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(ch not in _HEX for ch in value)
    ):
        raise ExecutionError(f"{field} must be lowercase SHA-256")
    return value


def _git(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or value != value.lower()
        or any(ch not in _HEX for ch in value)
    ):
        raise ExecutionError(f"{field} must be lowercase Git SHA-1")
    return value


def _positive_int(value: Any, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise ExecutionError(f"{field} must be a positive integer")
    return value


def _require_base_authority(authority: dict[str, Any]) -> None:
    if authority.get("schema") != "12-6.d03-current-clean-balance-authority.v1":
        raise ExecutionError("base current-clean authority schema drift")
    if authority.get("status") != "AUTHORIZED_FOR_REAL_BALANCE_EXECUTION":
        raise ExecutionError("base current-clean authority is not balance-authorized")
    if authority.get("independent_qualification_verdict") != "PASS":
        raise ExecutionError("base current-clean independent qualification is not PASS")
    if authority.get("dedup_terminal_verdict") != "PASS":
        raise ExecutionError("base current-clean dedup authority is not PASS")
    exact_zero = {
        "authorized_optimized_target_exposure": 0,
        "current_corpus_launch_authoritative": False,
        "final_test_outcomes_read": False,
        "foreign_pretrained_weights": False,
        "learned_weights_created": False,
        "optimizer_updates_executed_on_real_targets": 0,
        "paid_compute_used": False,
        "tokenizer_fit_authorized": False,
        "training_executed": False,
    }
    for field, expected in exact_zero.items():
        observed = authority.get(field)
        if type(observed) is not type(expected) or observed != expected:
            raise ExecutionError(f"base current-clean truth widened: {field}")

    _positive_int(authority.get("expected_record_count"), "base record count")
    _positive_int(authority.get("expected_total_payload_bytes"), "base total bytes")
    _positive_int(authority.get("expected_source_object_count"), "base source count")
    for field in (
        "expected_composition_receipt_identity_sha256",
        "expected_record_inventory_digest_sha256",
        "expected_payload_inventory_digest_sha256",
        "dedup_evidence_identity_sha256",
    ):
        _sha(authority.get(field), f"base {field}")
    _git(authority.get("expected_execution_head_sha"), "base execution head")
    _git(authority.get("dedup_head_sha"), "base dedup head")
    worker = authority.get("dedup_worker_id")
    if not isinstance(worker, str) or not worker.strip():
        raise ExecutionError("base dedup worker_id missing")


def _write(path: Path, value: dict[str, Any]) -> None:
    path.write_bytes(_canonical(value) + b"\n")


def execute(
    *,
    base_authority: dict[str, Any],
    base_inventory: dict[str, Any],
    delta_evidence: dict[str, Any],
    expected_delta_evidence_identity_sha256: str,
    expected_delta_execution_head_sha: str,
    expected_policy_identity_sha256: str,
    source_git_sha: str,
) -> dict[str, dict[str, Any]]:
    """Execute composition and canonical balance without creating capacity credit."""

    _require_base_authority(base_authority)
    expected_delta_evidence_identity_sha256 = _sha(
        expected_delta_evidence_identity_sha256,
        "expected delta evidence identity",
    )
    expected_delta_execution_head_sha = _git(
        expected_delta_execution_head_sha,
        "expected delta execution head",
    )
    expected_policy_identity_sha256 = _sha(
        expected_policy_identity_sha256,
        "expected balance policy identity",
    )
    source_git_sha = _git(source_git_sha, "execution source Git SHA")

    composition = compose_current_clean_and_delta_inventory(
        base_inventory=base_inventory,
        delta_evidence=delta_evidence,
        expected_base_physical_authority_identity_sha256=base_authority[
            "expected_composition_receipt_identity_sha256"
        ],
        expected_base_record_count=base_authority["expected_record_count"],
        expected_base_total_payload_bytes=base_authority[
            "expected_total_payload_bytes"
        ],
        expected_base_source_object_count=base_authority[
            "expected_source_object_count"
        ],
        expected_base_record_inventory_digest_sha256=base_authority[
            "expected_record_inventory_digest_sha256"
        ],
        expected_base_payload_inventory_digest_sha256=base_authority[
            "expected_payload_inventory_digest_sha256"
        ],
        expected_delta_evidence_identity_sha256=(
            expected_delta_evidence_identity_sha256
        ),
        expected_delta_execution_head_sha=expected_delta_execution_head_sha,
        source_git_sha=source_git_sha,
    )
    composition_identity = verify_current_clean_and_delta_composition(
        composition,
        expected_composition_identity_sha256=composition[
            "composition_identity_sha256"
        ],
    )

    dedup_core: dict[str, Any] = {
        "schema": COMBINED_DEDUP_SCHEMA,
        "base_dedup_authority": {
            "worker_id": base_authority["dedup_worker_id"],
            "head_sha": base_authority["dedup_head_sha"],
            "evidence_identity_sha256": base_authority[
                "dedup_evidence_identity_sha256"
            ],
            "terminal_verdict": base_authority["dedup_terminal_verdict"],
        },
        "delta_incumbent_global_dedup": {
            "parent_execution_head_sha": composition["delta"][
                "parent_execution_head_sha"
            ],
            "parent_survivor_authority_sha256": composition["delta"][
                "parent_survivor_authority_sha256"
            ],
            "parent_two_clean_proof_identity_sha256": composition["delta"][
                "parent_two_clean_proof_identity_sha256"
            ],
            "delta_evidence_identity_sha256": composition["delta"][
                "evidence_identity_sha256"
            ],
            "terminal_verdict": "PASS",
        },
        "composition_identity_sha256": composition_identity,
        "cross_inventory_record_id_collision_free": composition[
            "cross_inventory_record_id_collision_free"
        ],
        "cross_inventory_source_id_collision_free": composition[
            "cross_inventory_source_id_collision_free"
        ],
        "cross_inventory_exact_payload_collision_free": composition[
            "cross_inventory_exact_payload_collision_free"
        ],
        "semantics": (
            "BASE_DEDUP_PASS_PLUS_DELTA_INCUMBENT_GLOBAL_DEDUP_"
            "PLUS_CROSS_INVENTORY_COLLISION_PROOF_NO_REPLAY"
        ),
        "terminal_verdict": "PASS",
    }
    if any(
        dedup_core[field] is not True
        for field in (
            "cross_inventory_record_id_collision_free",
            "cross_inventory_source_id_collision_free",
            "cross_inventory_exact_payload_collision_free",
        )
    ):
        raise ExecutionError("combined dedup collision proof is not terminal")
    dedup_proof = {
        **dedup_core,
        "evidence_identity_sha256": _sha256(dedup_core),
    }

    families: list[dict[str, Any]] = []
    by_stratum = {"ua": 0, "en": 0, "code": 0}
    family_count = {"ua": 0, "en": 0, "code": 0}
    for row in composition["families"]:
        source_stratum = row["stratum"]
        try:
            stratum = STRATUM_MAP[source_stratum]
        except KeyError as exc:
            raise ExecutionError(
                f"unsupported composed family stratum: {source_stratum}"
            ) from exc
        capacity = _positive_int(row["capacity_bytes"], "family capacity")
        families.append(
            {
                "family_id": row["family"],
                "stratum": stratum,
                "unique_bytes": capacity,
            }
        )
        by_stratum[stratum] += capacity
        family_count[stratum] += 1
    families.sort(key=lambda row: row["family_id"])

    vector = {
        "schema_version": gate.INPUT_SCHEMA,
        "terminal": True,
        "dedup_authority": {
            "worker_id": "D03-CURRENT-PLUS-DELTA-GLOBAL-UNIQUE-COMPOSITION-V1",
            "head_sha": source_git_sha,
            "evidence_identity_sha256": dedup_proof[
                "evidence_identity_sha256"
            ],
            "terminal_verdict": "PASS",
        },
        "families": families,
        "totals": {
            "total_unique_bytes": sum(by_stratum.values()),
            "by_stratum": by_stratum,
            "family_count": family_count,
        },
        "physical_authority": {
            "composition_identity_sha256": composition_identity,
            "record_inventory_digest_sha256": composition["combined_inventory"][
                "record_inventory_digest_sha256"
            ],
            "payload_inventory_digest_sha256": composition["combined_inventory"][
                "payload_inventory_digest_sha256"
            ],
            "record_count": composition["combined_inventory"]["record_count"],
            "total_payload_bytes": composition["combined_inventory"][
                "total_payload_bytes"
            ],
            "source_object_count": (
                composition["base"]["source_object_count"]
                + composition["delta"]["source_object_count"]
            ),
            "trusted_family_authority_root_sha256": composition[
                "trusted_family_authority_root_sha256"
            ],
        },
    }
    gate.validate_vector(vector)

    policy = gate.load_json(gate.POLICY_PATH)
    gate.validate_policy(policy)
    if policy.get("policy_identity_sha256") != expected_policy_identity_sha256:
        raise ExecutionError("canonical NEXT100-106 policy identity drift")
    balance = gate.evaluate(policy, vector)
    if balance.get("policy_identity_sha256") != expected_policy_identity_sha256:
        raise ExecutionError("balance result policy lineage drift")

    status = balance["status"]
    if status == "TARGET_20M_SOURCE_MIX_FEASIBLE":
        next_scientific_gate = "CLUSTER_SAFE_SPLIT_AND_DETERMINISTIC_PACK"
    else:
        next_scientific_gate = "ACQUIRE_MORE_DIVERSE_LAWFUL_SOURCE_CAPACITY"

    receipt_core: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "execution_profile": "LOCAL_FREE",
        "execution_head_sha": source_git_sha,
        "base_current_clean_execution_head_sha": base_authority[
            "expected_execution_head_sha"
        ],
        "base_current_clean_authority_identity_sha256": base_authority[
            "expected_composition_receipt_identity_sha256"
        ],
        "delta_execution_head_sha": expected_delta_execution_head_sha,
        "delta_evidence_identity_sha256": expected_delta_evidence_identity_sha256,
        "composition_identity_sha256": composition_identity,
        "combined_dedup_proof_identity_sha256": dedup_proof[
            "evidence_identity_sha256"
        ],
        "balance_policy_identity_sha256": expected_policy_identity_sha256,
        "balance_result_identity_sha256": balance["result_identity_sha256"],
        "balance_status": status,
        "maximum_feasible_total_source_bytes": balance[
            "maximum_feasible_total_source_bytes"
        ],
        "maximum_feasible_stratum_bytes": balance[
            "maximum_feasible_stratum_bytes"
        ],
        "raw_capacity_by_stratum": balance["raw_capacity_by_stratum"],
        "raw_gap_to_target_by_stratum": balance[
            "raw_gap_to_target_by_stratum"
        ],
        "family_minimum": balance["family_minimum"],
        "combined_record_count": vector["physical_authority"]["record_count"],
        "combined_source_object_count": vector["physical_authority"][
            "source_object_count"
        ],
        "combined_total_payload_bytes": vector["physical_authority"][
            "total_payload_bytes"
        ],
        "next_scientific_gate": next_scientific_gate,
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
        "composition": composition,
        "combined-dedup-proof": dedup_proof,
        "next100-input": vector,
        "balance-result": balance,
        "execution-receipt": receipt,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--base-authority-json", type=Path, required=True)
    parser.add_argument("--base-inventory-json", type=Path, required=True)
    parser.add_argument("--delta-evidence-json", type=Path, required=True)
    parser.add_argument("--expected-delta-evidence-identity-sha256", required=True)
    parser.add_argument("--expected-delta-execution-head-sha", required=True)
    parser.add_argument("--expected-policy-identity-sha256", required=True)
    parser.add_argument("--source-git-sha", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = execute(
        base_authority=_load(args.base_authority_json),
        base_inventory=_load(args.base_inventory_json),
        delta_evidence=_load(args.delta_evidence_json),
        expected_delta_evidence_identity_sha256=(
            args.expected_delta_evidence_identity_sha256
        ),
        expected_delta_execution_head_sha=args.expected_delta_execution_head_sha,
        expected_policy_identity_sha256=args.expected_policy_identity_sha256,
        source_git_sha=args.source_git_sha,
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name, value in result.items():
        _write(args.output_dir / f"{name}.json", value)

    receipt = result["execution-receipt"]
    print(
        "D03_CURRENT_PLUS_DELTA_BALANCE="
        + receipt["balance_status"]
    )
    print(
        "MAXIMUM_FEASIBLE_TOTAL_SOURCE_BYTES="
        + str(receipt["maximum_feasible_total_source_bytes"])
    )
    print(
        "COMBINED_TOTAL_PAYLOAD_BYTES="
        + str(receipt["combined_total_payload_bytes"])
    )
    print(
        "RECEIPT_IDENTITY_SHA256="
        + receipt["receipt_identity_sha256"]
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
