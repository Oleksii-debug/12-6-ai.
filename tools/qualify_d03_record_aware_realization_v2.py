"""Qualify the cap-aware whole-record selector on exact physical D03 authorities.

This is a zero-credit qualification runner. It consumes only the text-free record
inventory and balance result from the already qualified 20M source-mix execution.
It exercises the Product selector's real record-realization helper without granting
corpus, tokenizer, training, evaluation, compute, or scale authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from twelve_six.data.current_clean_balanced_selection_v1 import (
    SELECTION_REALIZATION_POLICY,
    _select_stratum_records,
)

BALANCE_HEAD = "2419da879af2f293a94f6507871fe2ed109f6ecb"
COMPOSITION_ID = "84741c6cd99d06dbc55fb413b7b150573af7b010729b23839f4fdabcf8f505ef"
BALANCE_ID = "a6077ab075e21bb072c3239bc9e9839f8ce49dd4bb5bbd4a4d342c3de54da108"
POLICY_ID = "9a9242f47981c25e754fc95e2650050da4e4195aa1ef3a78f2c293f9e25d7ff7"
EXPECTED_RECORDS = 100_608
EXPECTED_BYTES = 208_859_647
TARGET_TOTAL = 20_000_000
TARGET_STRATA = {"ua": 9_000_000, "en": 7_000_000, "code": 4_000_000}
EXPECTED_CAPS = {"ua": 5_000_000, "en": 4_200_000, "code": 2_400_000}
MAX_INPUT = 64 * 1024 * 1024
OUTPUT_SCHEMA = "12-6.d03-record-aware-realization-qualification.v1"
ALLOCATION_FIELDS = {
    "family_id",
    "stratum",
    "allocated_bytes",
    "available_unique_bytes",
    "effective_family_cap_bytes",
}
MODALITY_TO_BALANCE = {"uk": "ua", "ua": "ua", "en": "en", "code": "code"}


class QualificationError(RuntimeError):
    """Raised when a physical qualification invariant fails."""


def require(ok: bool, message: str) -> None:
    if not ok:
        raise QualificationError(message)


def reject_constant(value: str) -> None:
    raise QualificationError(f"non-finite JSON constant: {value}")


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
    require(type(value) is dict, f"{path} must contain an object")
    return value


def canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def digest(value: Any) -> str:
    raw = value if isinstance(value, bytes) else canonical(value)
    return hashlib.sha256(raw).hexdigest()


def self_hash(document: dict[str, Any], field: str, expected: str) -> None:
    require(document.get(field) == expected, f"{field} authority drift")
    core = dict(document)
    core.pop(field, None)
    require(digest(core) == expected, f"{field} self-hash mismatch")


def positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise QualificationError(f"{field} must be a positive integer")
    return value


def verify_inputs(
    composition: dict[str, Any],
    balance: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    self_hash(composition, "composition_identity_sha256", COMPOSITION_ID)
    self_hash(balance, "result_identity_sha256", BALANCE_ID)
    require(composition.get("source_git_sha") == BALANCE_HEAD, "composition head drift")
    require(balance.get("policy_identity_sha256") == POLICY_ID, "balance policy drift")
    require(
        balance.get("status") == "TARGET_20M_SOURCE_MIX_FEASIBLE",
        "balance status drift",
    )
    require(
        balance.get("maximum_feasible_total_source_bytes") == TARGET_TOTAL,
        "balance total drift",
    )
    require(
        balance.get("maximum_feasible_stratum_bytes") == TARGET_STRATA,
        "balance stratum target drift",
    )

    family_minimum = balance.get("family_minimum")
    require(type(family_minimum) is dict, "family minimum authority missing")
    require(
        family_minimum.get("required_per_stratum") == 2,
        "family minimum requirement drift",
    )
    require(family_minimum.get("pass") is True, "family minimum did not pass")

    inventory = composition.get("combined_inventory")
    require(type(inventory) is dict, "combined inventory missing")
    rows = inventory.get("records")
    require(type(rows) is list, "combined inventory records missing")
    require(len(rows) == EXPECTED_RECORDS, "inventory record count drift")
    require(inventory.get("record_count") == EXPECTED_RECORDS, "record count drift")
    require(inventory.get("total_payload_bytes") == EXPECTED_BYTES, "byte count drift")
    require(
        sum(positive_int(row.get("payload_bytes"), "payload_bytes") for row in rows)
        == EXPECTED_BYTES,
        "inventory payload total drift",
    )
    record_ids = [row.get("record_id") for row in rows]
    require(
        all(isinstance(record_id, str) and record_id for record_id in record_ids),
        "invalid record id",
    )
    require(len(set(record_ids)) == EXPECTED_RECORDS, "record id replay")

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

    raw_allocations = balance.get("deterministic_maximum_allocation")
    require(type(raw_allocations) is list and raw_allocations, "balance allocation missing")
    allocations: dict[str, dict[str, Any]] = {}
    caps: defaultdict[str, set[int]] = defaultdict(set)
    for index, raw in enumerate(raw_allocations):
        require(type(raw) is dict, f"allocation[{index}] must be an object")
        require(set(raw) == ALLOCATION_FIELDS, f"allocation[{index}] fields drift")
        family = raw.get("family_id")
        stratum = raw.get("stratum")
        require(isinstance(family, str) and family, f"allocation[{index}] family invalid")
        require(stratum in TARGET_STRATA, f"allocation[{index}] stratum invalid")
        require(family not in allocations, f"duplicate allocation family: {family}")
        allocated = positive_int(raw.get("allocated_bytes"), "allocated_bytes")
        available = positive_int(
            raw.get("available_unique_bytes"),
            "available_unique_bytes",
        )
        cap = positive_int(
            raw.get("effective_family_cap_bytes"),
            "effective_family_cap_bytes",
        )
        require(allocated <= available and allocated <= cap, "allocation exceeds authority")
        allocations[family] = dict(raw)
        caps[stratum].add(cap)

    for stratum, expected_cap in EXPECTED_CAPS.items():
        require(caps[stratum] == {expected_cap}, f"{stratum} family cap drift")
    return [dict(row) for row in rows], allocations


def execute(composition: dict[str, Any], balance: dict[str, Any]) -> dict[str, Any]:
    rows, allocations = verify_inputs(composition, balance)
    by_stratum: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in rows:
        family = raw.get("family")
        modality = raw.get("modality")
        require(isinstance(family, str) and family, "inventory family invalid")
        require(modality in MODALITY_TO_BALANCE, "inventory modality invalid")
        balance_stratum = MODALITY_TO_BALANCE[modality]
        runtime_stratum = "uk" if balance_stratum == "ua" else balance_stratum
        by_stratum[balance_stratum].append(
            {
                "record_id": raw["record_id"],
                "family": family,
                "stratum": runtime_stratum,
                "payload_bytes": raw["payload_bytes"],
            }
        )

    selected_all: list[dict[str, Any]] = []
    stratum_results: dict[str, dict[str, Any]] = {}
    for stratum in ("ua", "en", "code"):
        selected = _select_stratum_records(
            by_stratum[stratum],
            stratum=stratum,
            allocations=allocations,
            target_bytes=TARGET_STRATA[stratum],
            family_cap_bytes=EXPECTED_CAPS[stratum],
        )
        require(
            sum(row["payload_bytes"] for row in selected) == TARGET_STRATA[stratum],
            f"{stratum} exact target not realized",
        )
        family_bytes: defaultdict[str, int] = defaultdict(int)
        for row in selected:
            family_bytes[row["family"]] += row["payload_bytes"]
        require(len(family_bytes) >= 2, f"{stratum} family minimum violated")
        require(
            max(family_bytes.values()) <= EXPECTED_CAPS[stratum],
            f"{stratum} family cap violated",
        )

        canonical_targets = {
            family: item["allocated_bytes"]
            for family, item in allocations.items()
            if item["stratum"] == stratum
        }
        witness_matched = (
            set(family_bytes) == set(canonical_targets)
            and all(
                family_bytes[family] == target
                for family, target in canonical_targets.items()
            )
        )
        stratum_results[stratum] = {
            "target_bytes": TARGET_STRATA[stratum],
            "selected_record_count": len(selected),
            "selected_family_count": len(family_bytes),
            "maximum_selected_family_bytes": max(family_bytes.values()),
            "continuous_family_witness_matched": witness_matched,
            "selected_membership_sha256": digest(
                sorted(
                    (
                        row["record_id"],
                        row["family"],
                        row["payload_bytes"],
                    )
                    for row in selected
                )
            ),
        }
        selected_all.extend(selected)

    selected_ids = [row["record_id"] for row in selected_all]
    require(len(selected_ids) == len(set(selected_ids)), "record reused across strata")
    require(
        sum(row["payload_bytes"] for row in selected_all) == TARGET_TOTAL,
        "selected total bytes drift",
    )
    require(
        stratum_results["code"]["continuous_family_witness_matched"] is False,
        "qualification did not exercise the proven code witness mismatch",
    )

    core: dict[str, Any] = {
        "schema_version": OUTPUT_SCHEMA,
        "execution_profile": "LOCAL_FREE",
        "balance_head_sha": BALANCE_HEAD,
        "composition_identity_sha256": COMPOSITION_ID,
        "balance_result_identity_sha256": BALANCE_ID,
        "balance_policy_identity_sha256": POLICY_ID,
        "selection_realization_policy": SELECTION_REALIZATION_POLICY,
        "target_total_source_bytes": TARGET_TOTAL,
        "target_stratum_bytes": TARGET_STRATA,
        "strata": stratum_results,
        "selected_record_count": len(selected_all),
        "selected_source_bytes": TARGET_TOTAL,
        "selected_membership_sha256": digest(
            sorted(
                (
                    row["record_id"],
                    row["family"],
                    row["payload_bytes"],
                )
                for row in selected_all
            )
        ),
        "terminal_verdict": "PASS_RECORD_AWARE_REALIZATION",
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
    core["qualification_identity_sha256"] = digest(core)
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
    except (
        QualificationError,
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        ValueError,
        TypeError,
        KeyError,
    ) as exc:
        print(f"BLOCKED: {exc}")
        return 2

    print("RECORD_AWARE_REALIZATION_VERDICT=" + result["terminal_verdict"])
    print("SELECTED_SOURCE_BYTES=" + str(result["selected_source_bytes"]))
    print(
        "CODE_CONTINUOUS_WITNESS_MATCHED="
        + str(result["strata"]["code"]["continuous_family_witness_matched"])
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
