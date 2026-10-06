"""Diagnose whole-record realizability of an exact NEXT100 balance authority.

Execution-only and text-free. This reproduces the sparse exact-family selector currently
implemented by PR #2583, distinguishes implementation-budget blockers from true subset
impossibility with a bounded-target bitset reference, and proves whether the exact
20M stratum totals remain feasible at record granularity. It grants no corpus,
tokenizer, training, evaluation, compute, or scale authority.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

BALANCE_HEAD = "2419da879af2f293a94f6507871fe2ed109f6ecb"
COMPOSITION_ID = "84741c6cd99d06dbc55fb413b7b150573af7b010729b23839f4fdabcf8f505ef"
BALANCE_ID = "a6077ab075e21bb072c3239bc9e9839f8ce49dd4bb5bbd4a4d342c3de54da108"
POLICY_ID = "9a9242f47981c25e754fc95e2650050da4e4195aa1ef3a78f2c293f9e25d7ff7"
SELECTION_HEAD = "8e4efd89dbace1931032dc4283bb5c22cdeda597"
SELECTION_BLOB = "08fbc509e2acd4913e62ecfc83d8312ebfc2313f"
SELECTION_POLICY = "record-id-ascending-exact-family-byte-subset-v1"
MAX_STATES = 250_000
MAX_EXPANSIONS = 5_000_000
EXPECTED_RECORDS = 100_608
EXPECTED_BYTES = 208_859_647
TARGET_TOTAL = 20_000_000
TARGET_STRATA = {"ua": 9_000_000, "en": 7_000_000, "code": 4_000_000}
CAP_BY_STRATUM = {"ua": 5_000_000, "en": 4_200_000, "code": 2_400_000}
MAX_INPUT = 64 * 1024 * 1024
OUTCOME_SCHEMA = "12-6.d03-balance-record-realization-diagnostic.v1"


class DiagnosticError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise DiagnosticError(message)


def reject_constant(value: str) -> None:
    raise DiagnosticError(f"non-finite JSON constant: {value}")


def parse_float(value: str) -> float:
    result = float(value)
    require(math.isfinite(result), "non-finite JSON number")
    return result


def strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    require(len(raw) <= MAX_INPUT, f"{path} exceeds input bound")
    value = json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=strict_object,
        parse_constant=reject_constant,
        parse_float=parse_float,
    )
    require(type(value) is dict, f"{path} must contain object")
    return value


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def digest(value: Any) -> str:
    raw = value if isinstance(value, bytes) else canonical(value)
    return hashlib.sha256(raw).hexdigest()


def self_hash(doc: dict[str, Any], field: str, expected: str) -> None:
    require(doc.get(field) == expected, f"{field} authority drift")
    core = dict(doc)
    core.pop(field, None)
    require(digest(core) == expected, f"{field} self-hash mismatch")


def verify_inputs(
    composition: dict[str, Any], balance: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    self_hash(composition, "composition_identity_sha256", COMPOSITION_ID)
    self_hash(balance, "result_identity_sha256", BALANCE_ID)
    require(composition.get("source_git_sha") == BALANCE_HEAD, "composition head drift")
    require(balance.get("policy_identity_sha256") == POLICY_ID, "balance policy drift")
    require(balance.get("status") == "TARGET_20M_SOURCE_MIX_FEASIBLE", "balance status drift")
    require(
        balance.get("maximum_feasible_total_source_bytes") == TARGET_TOTAL,
        "balance total drift",
    )
    require(balance.get("maximum_feasible_stratum_bytes") == TARGET_STRATA, "stratum target drift")

    inventory = composition.get("combined_inventory")
    require(type(inventory) is dict, "combined inventory missing")
    rows = inventory.get("records")
    require(type(rows) is list and len(rows) == EXPECTED_RECORDS, "inventory count drift")
    require(inventory.get("record_count") == EXPECTED_RECORDS, "declared record count drift")
    require(inventory.get("total_payload_bytes") == EXPECTED_BYTES, "declared byte count drift")
    require(sum(row["payload_bytes"] for row in rows) == EXPECTED_BYTES, "inventory bytes drift")
    require(len({row["record_id"] for row in rows}) == EXPECTED_RECORDS, "record replay")
    require(len({row["payload_sha256"] for row in rows}) == EXPECTED_RECORDS, "payload replay")
    payload_projection = [
        {
            "record_id": row["record_id"],
            "payload_sha256": row["payload_sha256"],
            "payload_bytes": row["payload_bytes"],
        }
        for row in rows
    ]
    require(
        digest(rows) == inventory.get("record_inventory_digest_sha256"),
        "record inventory root drift",
    )
    require(
        digest(payload_projection) == inventory.get("payload_inventory_digest_sha256"),
        "payload inventory root drift",
    )
    allocations = balance.get("deterministic_maximum_allocation")
    require(type(allocations) is list and allocations, "balance allocation missing")
    return [dict(row) for row in rows], [dict(row) for row in allocations]


def sparse_selector_outcome(rows: list[dict[str, Any]], target: int) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: row["record_id"])
    available = sum(row["payload_bytes"] for row in ordered)
    if target == available:
        return {
            "outcome": "FULL_CAPACITY",
            "states": 1,
            "expansions": 0,
            "records_examined": 0,
            "exact_subset_found": True,
        }
    parent: dict[int, tuple[int, int] | None] = {0: None}
    expansions = 0
    for index, row in enumerate(ordered):
        weight = row["payload_bytes"]
        existing = tuple(parent)
        for current in existing:
            expansions += 1
            if expansions > MAX_EXPANSIONS:
                return {
                    "outcome": "EXPANSION_BUDGET_EXCEEDED",
                    "states": len(parent),
                    "expansions": expansions,
                    "records_examined": index + 1,
                    "exact_subset_found": False,
                }
            candidate = current + weight
            if candidate > target or candidate in parent:
                continue
            if len(parent) >= MAX_STATES:
                return {
                    "outcome": "STATE_BUDGET_EXCEEDED",
                    "states": len(parent),
                    "expansions": expansions,
                    "records_examined": index + 1,
                    "exact_subset_found": False,
                }
            parent[candidate] = (current, index)
            if candidate == target:
                return {
                    "outcome": "EXACT_SUBSET_FOUND",
                    "states": len(parent),
                    "expansions": expansions,
                    "records_examined": index + 1,
                    "exact_subset_found": True,
                }
    return {
        "outcome": "NO_EXACT_SUBSET",
        "states": len(parent),
        "expansions": expansions,
        "records_examined": len(ordered),
        "exact_subset_found": False,
    }


def bitset_exact_exists(rows: list[dict[str, Any]], target: int) -> tuple[bool, int]:
    require(target >= 0, "negative subset target")
    if target == 0:
        return True, 0
    mask = (1 << (target + 1)) - 1
    bits = 1
    for index, row in enumerate(sorted(rows, key=lambda item: item["record_id"])):
        bits |= (bits << row["payload_bytes"]) & mask
        if (bits >> target) & 1:
            return True, index + 1
    return False, len(rows)


def execute(composition: dict[str, Any], balance: dict[str, Any]) -> dict[str, Any]:
    rows, allocations = verify_inputs(composition, balance)
    by_family: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    by_stratum: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_family[row["family"]].append(row)
        stratum = "ua" if row["modality"] in {"uk", "ua"} else row["modality"]
        require(stratum in TARGET_STRATA, f"unsupported modality/stratum: {row['modality']}")
        by_stratum[stratum].append(row)

    diagnostics: list[dict[str, Any]] = []
    for item in allocations:
        family = item["family_id"]
        target = item["allocated_bytes"]
        family_rows = by_family[family]
        require(sum(row["payload_bytes"] for row in family_rows) == item["available_unique_bytes"], f"{family} availability drift")
        sparse = sparse_selector_outcome(family_rows, target)
        reference, reference_examined = bitset_exact_exists(family_rows, target)
        diagnostics.append(
            {
                "family_id": family,
                "stratum": item["stratum"],
                "allocated_bytes": target,
                "available_unique_bytes": item["available_unique_bytes"],
                "record_count": len(family_rows),
                "sparse_selector": sparse,
                "reference_bitset_exact_exists": reference,
                "reference_records_examined": reference_examined,
            }
        )

    hard_impossible = sorted(
        row["family_id"]
        for row in diagnostics
        if row["sparse_selector"]["outcome"] == "NO_EXACT_SUBSET"
        and row["reference_bitset_exact_exists"] is False
    )
    budget_blocked_but_exact = sorted(
        row["family_id"]
        for row in diagnostics
        if row["sparse_selector"]["outcome"]
        in {"STATE_BUDGET_EXCEEDED", "EXPANSION_BUDGET_EXCEEDED"}
        and row["reference_bitset_exact_exists"] is True
    )
    sparse_all = all(row["sparse_selector"]["exact_subset_found"] for row in diagnostics)

    # UA and EN can retain the canonical family-byte witness if every family target
    # has an exact whole-record subset under the reference reachability check.
    reference_by_stratum = {
        stratum: all(
            row["reference_bitset_exact_exists"]
            for row in diagnostics
            if row["stratum"] == stratum
        )
        for stratum in ("ua", "en")
    }

    # Code is the one semantic witness mismatch: the canonical allocation asks for an
    # impossible 12,440-byte NumPy subset. The policy only requires 4,000,000 code
    # bytes and a 2.4M/family cap. Since every full code family is below the cap, an
    # exact whole-record stratum subset is sufficient to prove policy-level existence.
    code_rows = by_stratum["code"]
    code_cap_safe = max(
        sum(row["payload_bytes"] for row in by_family[family])
        for family in {row["family"] for row in code_rows}
    ) <= CAP_BY_STRATUM["code"]
    code_exact, code_examined = bitset_exact_exists(code_rows, TARGET_STRATA["code"])
    reference_policy_feasible = (
        reference_by_stratum["ua"]
        and reference_by_stratum["en"]
        and code_cap_safe
        and code_exact
    )

    core = {
        "schema_version": OUTCOME_SCHEMA,
        "execution_profile": "LOCAL_FREE",
        "balance_head_sha": BALANCE_HEAD,
        "composition_identity_sha256": COMPOSITION_ID,
        "balance_result_identity_sha256": BALANCE_ID,
        "balance_policy_identity_sha256": POLICY_ID,
        "selection_implementation_head_sha": SELECTION_HEAD,
        "selection_implementation_blob_sha1": SELECTION_BLOB,
        "selection_realization_policy": SELECTION_POLICY,
        "selection_sparse_state_budget": MAX_STATES,
        "selection_sparse_expansion_budget": MAX_EXPANSIONS,
        "target_total_source_bytes": TARGET_TOTAL,
        "target_stratum_bytes": TARGET_STRATA,
        "family_diagnostics": diagnostics,
        "current_sparse_selection_can_materialize_exact_balance": sparse_all,
        "hard_impossible_canonical_family_allocations": hard_impossible,
        "budget_blocked_but_reference_exact_families": budget_blocked_but_exact,
        "reference_ua_canonical_family_witness_exact": reference_by_stratum["ua"],
        "reference_en_canonical_family_witness_exact": reference_by_stratum["en"],
        "reference_code_stratum_exact_whole_record_subset_exists": code_exact,
        "reference_code_records_examined": code_examined,
        "reference_code_all_family_cap_safe_by_full_capacity": code_cap_safe,
        "reference_policy_compliant_20m_record_selection_exists": reference_policy_feasible,
        "terminal_verdict": (
            "BLOCKED_CURRENT_SELECTION_IMPLEMENTATION"
            if not sparse_all and reference_policy_feasible
            else "BLOCKED_RECORD_LEVEL_POLICY"
            if not reference_policy_feasible
            else "PASS"
        ),
        "repair_class": (
            "SELECTION_REALIZATION_BUDGETS_AND_FAMILY_WITNESS_COUPLING"
            if not sparse_all and reference_policy_feasible
            else "NONE"
        ),
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
        "scale_promotion_authorized": False,
    }
    core["diagnostic_identity_sha256"] = digest(core)
    return core


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--composition-json", type=Path, required=True)
    parser.add_argument("--balance-result-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = execute(load(args.composition_json), load(args.balance_result_json))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(canonical(result) + b"\n")
    except (DiagnosticError, OSError, UnicodeError, json.JSONDecodeError, ValueError, TypeError, KeyError) as exc:
        print(f"BLOCKED: {exc}")
        return 2
    print("RECORD_REALIZATION_VERDICT=" + result["terminal_verdict"])
    print("CURRENT_SPARSE_SELECTION_CAN_MATERIALIZE=" + str(result["current_sparse_selection_can_materialize_exact_balance"]))
    print("REFERENCE_POLICY_COMPLIANT_20M_RECORD_SELECTION_EXISTS=" + str(result["reference_policy_compliant_20m_record_selection_exists"]))
    print("HARD_IMPOSSIBLE=" + ",".join(result["hard_impossible_canonical_family_allocations"]))
    print("BUDGET_BLOCKED=" + ",".join(result["budget_blocked_but_reference_exact_families"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
