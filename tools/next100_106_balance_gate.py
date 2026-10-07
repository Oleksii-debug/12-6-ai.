#!/usr/bin/env python3
"""Executable fail-closed balance/diversity gate for NEXT100-106."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "configs/data/next100_106_balance_gate_policy_v1.json"

INPUT_SCHEMA = "12-6.next100-106-post-dedup-family-vector.v1"
STRATA = ("ua", "en", "code")


class GateError(ValueError):
    """Raised when an input cannot be trusted for balance evaluation."""


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise GateError(f"expected JSON object: {path}")
    return value


def canonical_sha(data: dict[str, Any], identity_key: str) -> str:
    body = dict(data)
    body.pop(identity_key, None)
    payload = json.dumps(
        body, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise GateError(f"{field} must be a positive integer")
    return value


def _nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise GateError(f"{field} must be a non-negative integer")
    return value


def _hex(value: Any, length: int, field: str) -> str:
    if not isinstance(value, str) or len(value) != length:
        raise GateError(f"{field} must be {length} lowercase hex characters")
    if value.lower() != value or any(ch not in "0123456789abcdef" for ch in value):
        raise GateError(f"{field} must be {length} lowercase hex characters")
    return value


def validate_policy(policy: dict[str, Any]) -> None:
    if policy.get("schema_version") != "12-6.next100-106-balance-gate-policy.v1":
        raise GateError("unexpected policy schema")
    if policy.get("worker_id") != "NEXT100-106-EXECUTABLE-BALANCE-GATE-V3":
        raise GateError("unexpected worker id")
    if policy.get("execution_class") != "LOCAL_FREE":
        raise GateError("policy must remain LOCAL_FREE")
    if canonical_sha(policy, "policy_identity_sha256") != policy.get(
        "policy_identity_sha256"
    ):
        raise GateError("policy identity mismatch")

    cfg = policy["policy"]
    if cfg["target_total_source_bytes"] != 20_000_000:
        raise GateError("20M source-byte planning target drift")
    if cfg["minimum_independent_families_per_stratum"] != 2:
        raise GateError("minimum family count drift")
    if cfg["budget_quantum_bytes"] != 100:
        raise GateError("budget quantum drift")
    if cfg["replay_or_duplication_to_meet_quota"] is not False:
        raise GateError("replay must remain forbidden")
    if cfg["model_result_guided_mixture_retuning"] is not False:
        raise GateError("model-result-guided mixture retuning must remain forbidden")

    expected = {"ua": (9, 20), "en": (7, 20), "code": (1, 5)}
    observed = {
        key: (
            cfg["strata"][key]["target_numerator"],
            cfg["strata"][key]["target_denominator"],
        )
        for key in STRATA
    }
    if observed != expected:
        raise GateError("45/35/20 mixture drift")
    if cfg["max_family_fraction_total"] != {"numerator": 1, "denominator": 4}:
        raise GateError("global family cap drift")
    if cfg["max_family_fraction_own_stratum"] != {
        "numerator": 3,
        "denominator": 5,
    }:
        raise GateError("within-stratum family cap drift")

    boundary = policy["claim_boundary"]
    if boundary != {
        "computes_source_mixture_feasibility_only": True,
        "creates_corpus_identity": False,
        "creates_shard_identity": False,
        "authorizes_tokenizer_fit": False,
        "authorizes_model_training": False,
        "authorizes_paid_compute": False,
        "relabels_source_bytes_as_loss_positions": False,
    }:
        raise GateError("claim boundary drift")


def validate_vector(vector: dict[str, Any]) -> list[dict[str, Any]]:
    if vector.get("schema_version") != INPUT_SCHEMA:
        raise GateError(f"unexpected input schema; expected {INPUT_SCHEMA}")
    if vector.get("terminal") is not True:
        raise GateError("post-dedup vector is nonterminal")

    authority = vector.get("dedup_authority")
    if not isinstance(authority, dict):
        raise GateError("dedup_authority must be an object")
    worker_id = authority.get("worker_id")
    if not isinstance(worker_id, str) or not worker_id.strip():
        raise GateError("dedup authority worker_id is required")
    _hex(authority.get("head_sha"), 40, "dedup_authority.head_sha")
    _hex(
        authority.get("evidence_identity_sha256"),
        64,
        "dedup_authority.evidence_identity_sha256",
    )
    if authority.get("terminal_verdict") != "PASS":
        raise GateError("dedup authority terminal verdict must be PASS")

    families = vector.get("families")
    if not isinstance(families, list) or not families:
        raise GateError("families must be a non-empty list")

    seen: set[str] = set()
    normalized: list[dict[str, Any]] = []
    by_stratum = defaultdict(int)
    counts = defaultdict(int)
    for index, item in enumerate(families):
        if not isinstance(item, dict):
            raise GateError(f"families[{index}] must be an object")
        family_id = item.get("family_id")
        if not isinstance(family_id, str) or not family_id.strip():
            raise GateError(f"families[{index}].family_id is required")
        if family_id in seen:
            raise GateError(f"duplicate family_id would create replay-like credit: {family_id}")
        seen.add(family_id)

        stratum = item.get("stratum")
        if stratum not in STRATA:
            raise GateError(f"invalid stratum for {family_id}: {stratum!r}")
        capacity = _positive_int(
            item.get("unique_bytes"), f"families[{index}].unique_bytes"
        )
        normalized.append(
            {"family_id": family_id, "stratum": stratum, "unique_bytes": capacity}
        )
        by_stratum[stratum] += capacity
        counts[stratum] += 1

    declared = vector.get("totals")
    if not isinstance(declared, dict):
        raise GateError("totals must be an object")
    if _nonnegative_int(declared.get("total_unique_bytes"), "totals.total_unique_bytes") != sum(
        by_stratum.values()
    ):
        raise GateError("declared total_unique_bytes mismatch")
    if declared.get("by_stratum") != {key: by_stratum[key] for key in STRATA}:
        raise GateError("declared by_stratum totals mismatch")
    if declared.get("family_count") != {key: counts[key] for key in STRATA}:
        raise GateError("declared family_count mismatch")

    return sorted(normalized, key=lambda item: item["family_id"])


def _ratio(cfg: dict[str, Any], stratum: str) -> tuple[int, int]:
    row = cfg["strata"][stratum]
    return row["target_numerator"], row["target_denominator"]


def stratum_target(total: int, cfg: dict[str, Any], stratum: str) -> int:
    numerator, denominator = _ratio(cfg, stratum)
    product = total * numerator
    if product % denominator:
        raise GateError("total does not produce exact integer stratum targets")
    return product // denominator


def family_limit(total: int, stratum_bytes: int, cfg: dict[str, Any]) -> int:
    global_cap = cfg["max_family_fraction_total"]
    own_cap = cfg["max_family_fraction_own_stratum"]
    global_numerator = total * global_cap["numerator"]
    own_numerator = stratum_bytes * own_cap["numerator"]
    if global_numerator % global_cap["denominator"]:
        raise GateError("budget quantum does not preserve exact global family cap")
    if own_numerator % own_cap["denominator"]:
        raise GateError("budget quantum does not preserve exact within-stratum family cap")
    return min(
        global_numerator // global_cap["denominator"],
        own_numerator // own_cap["denominator"],
    )


def feasible(
    total: int, families: list[dict[str, Any]], cfg: dict[str, Any]
) -> bool:
    if total <= 0:
        return False
    quantum = cfg["budget_quantum_bytes"]
    if total % quantum:
        raise GateError("candidate total must align to budget quantum")

    grouped: dict[str, list[dict[str, Any]]] = {key: [] for key in STRATA}
    for family in families:
        grouped[family["stratum"]].append(family)

    minimum = cfg["minimum_independent_families_per_stratum"]
    for stratum in STRATA:
        if len(grouped[stratum]) < minimum:
            return False
        required = stratum_target(total, cfg, stratum)
        cap = family_limit(total, required, cfg)
        available_under_caps = sum(
            min(family["unique_bytes"], cap) for family in grouped[stratum]
        )
        if available_under_caps < required:
            return False
    return True


def maximum_feasible_total(
    families: list[dict[str, Any]], cfg: dict[str, Any]
) -> int:
    target = cfg["target_total_source_bytes"]
    quantum = cfg["budget_quantum_bytes"]
    high = target // quantum
    low = 0

    # Feasibility is monotone in this capped no-replay setting: once finite
    # family capacities cannot support a larger exact mixture, increasing the
    # requested budget cannot restore missing capacity.
    while low < high:
        mid = (low + high + 1) // 2
        candidate = mid * quantum
        if feasible(candidate, families, cfg):
            low = mid
        else:
            high = mid - 1
    return low * quantum


def deterministic_allocation(
    total: int, families: list[dict[str, Any]], cfg: dict[str, Any]
) -> list[dict[str, Any]]:
    if not feasible(total, families, cfg):
        raise GateError("cannot allocate an infeasible mixture")

    grouped: dict[str, list[dict[str, Any]]] = {key: [] for key in STRATA}
    for family in families:
        grouped[family["stratum"]].append(family)

    allocations: list[dict[str, Any]] = []
    for stratum in STRATA:
        required = stratum_target(total, cfg, stratum)
        per_family_cap = family_limit(total, required, cfg)
        remaining = required
        candidates = sorted(
            grouped[stratum],
            key=lambda item: (-min(item["unique_bytes"], per_family_cap), item["family_id"]),
        )
        for family in candidates:
            take = min(family["unique_bytes"], per_family_cap, remaining)
            if take:
                allocations.append(
                    {
                        "family_id": family["family_id"],
                        "stratum": stratum,
                        "allocated_bytes": take,
                        "available_unique_bytes": family["unique_bytes"],
                        "effective_family_cap_bytes": per_family_cap,
                    }
                )
                remaining -= take
            if remaining == 0:
                break
        if remaining:
            raise GateError("allocation invariant failed despite feasibility proof")
    return sorted(allocations, key=lambda item: item["family_id"])


def evaluate(
    policy: dict[str, Any], vector: dict[str, Any]
) -> dict[str, Any]:
    validate_policy(policy)
    families = validate_vector(vector)
    cfg = policy["policy"]

    by_stratum_capacity = {
        stratum: sum(
            family["unique_bytes"]
            for family in families
            if family["stratum"] == stratum
        )
        for stratum in STRATA
    }
    family_count = {
        stratum: sum(1 for family in families if family["stratum"] == stratum)
        for stratum in STRATA
    }
    minimum = cfg["minimum_independent_families_per_stratum"]
    family_minimum_pass = all(family_count[key] >= minimum for key in STRATA)

    maximum = maximum_feasible_total(families, cfg) if family_minimum_pass else 0
    target = cfg["target_total_source_bytes"]
    if maximum == target:
        status = "TARGET_20M_SOURCE_MIX_FEASIBLE"
    elif maximum > 0:
        status = "PARTIAL_MIX_FEASIBLE_ACQUIRE_MORE_DATA"
    else:
        status = "BLOCKED_NO_NONZERO_POLICY_COMPLIANT_MIXTURE"

    allocations = deterministic_allocation(maximum, families, cfg) if maximum else []
    max_targets = (
        {key: stratum_target(maximum, cfg, key) for key in STRATA}
        if maximum
        else {key: 0 for key in STRATA}
    )
    target_targets = {key: stratum_target(target, cfg, key) for key in STRATA}

    result: dict[str, Any] = {
        "schema_version": "12-6.next100-106-balance-gate-result.v1",
        "policy_identity_sha256": policy["policy_identity_sha256"],
        "dedup_authority": vector["dedup_authority"],
        "input_totals": vector["totals"],
        "family_minimum": {
            "required_per_stratum": minimum,
            "observed": family_count,
            "pass": family_minimum_pass,
        },
        "maximum_feasible_total_source_bytes": maximum,
        "maximum_feasible_stratum_bytes": max_targets,
        "target_total_source_bytes": target,
        "target_stratum_bytes": target_targets,
        "raw_capacity_by_stratum": by_stratum_capacity,
        "raw_gap_to_target_by_stratum": {
            key: max(0, target_targets[key] - by_stratum_capacity[key])
            for key in STRATA
        },
        "deterministic_maximum_allocation": allocations,
        "status": status,
        "next_step": (
            "IMMUTABLE_CORPUS_MATERIALIZATION_STILL_REQUIRES_QUALITY_PRIVACY_"
            "DECONTAMINATION_SPLIT_PACK_AND_TWO_CLEAN_BUILD_GATES"
        ),
        "claim_boundary": {
            "authorized_training_exposure_loss_positions": 0,
            "corpus_identity": None,
            "shard_identity": None,
            "tokenizer_fit_authorized": False,
            "model_training_authorized": False,
            "paid_compute_authorized": False,
            "source_bytes_are_loss_positions": False,
        },
    }
    result["result_identity_sha256"] = canonical_sha(
        result, "result_identity_sha256"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("validate-policy")

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("input", type=Path)
    evaluate_parser.add_argument("--output", type=Path)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policy = load_json(POLICY_PATH)
    validate_policy(policy)

    if args.command == "validate-policy":
        print(
            "NEXT100-106 policy PASS "
            f"identity={policy['policy_identity_sha256']}"
        )
        return

    vector = load_json(args.input)
    result = evaluate(policy, vector)
    payload = json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
