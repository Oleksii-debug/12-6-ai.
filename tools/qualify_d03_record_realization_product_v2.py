"""Qualify the exact D03 whole-record selector against the #3045 inventory.

Execution-only and text-free. The exact Product selector implementation is imported
from the checkout and exercised against authenticated record metadata. Raw normalized
training payloads are neither required nor emitted by this qualification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from twelve_six.data.current_clean_balanced_selection_v1 import (
    SELECTION_REALIZATION_POLICY,
    _exact_record_subset,
    _NoExactRecordSubset,
)

PRODUCT_HEAD = "efadbdac0c03e7c020501652f5624b96f289ec2e"
PRODUCT_PARENT = "8e4efd89dbace1931032dc4283bb5c22cdeda597"
SELECTION_BLOB = "79eed6a6d3b941e89132f072593db1c74bd1d90d"
BALANCE_HEAD = "2419da879af2f293a94f6507871fe2ed109f6ecb"
COMPOSITION_ID = "84741c6cd99d06dbc55fb413b7b150573af7b010729b23839f4fdabcf8f505ef"
BALANCE_ID = "a6077ab075e21bb072c3239bc9e9839f8ce49dd4bb5bbd4a4d342c3de54da108"
POLICY_ID = "9a9242f47981c25e754fc95e2650050da4e4195aa1ef3a78f2c293f9e25d7ff7"
EXPECTED_RECORDS = 100_608
EXPECTED_SOURCE_OBJECTS = 100_527
EXPECTED_INPUT_BYTES = 208_859_647
TARGET_TOTAL = 20_000_000
TARGET_STRATA = {"ua": 9_000_000, "en": 7_000_000, "code": 4_000_000}
EXPECTED_IMPOSSIBLE = ["github:numpy/numpy"]
EXPECTED_FALSE_BUDGET_BLOCKERS = [
    "common-pile/ubuntu_irc",
    "en.loc.selected-digitized-books",
    "ua.rada.open-data.laws-texts",
]
EXPECTED_FALLBACK_ELIGIBILITY = {"ua": False, "en": False, "code": True}
EXPECTED_SELECTED_RECORDS = 5_294
EXPECTED_SELECTED_FAMILY_COUNTS = {"ua": 5, "en": 2, "code": 18}
EXPECTED_MEMBERSHIP_SHA256 = (
    "a00572acfb1033f7bfe55e7e1f53ae9eb44b6fa039f3f378a156bec6fd13615f"
)
EXPECTED_PAYLOAD_ROOT_SHA256 = (
    "9bcfc6eb09418c93a00373df6d4c4cf3e5795d54e12c0e4bc106eaecbc89de82"
)
EXPECTED_FAMILY_BYTES_ROOT_SHA256 = (
    "f54dec34d6c86506d3321686e63af0c226f4f7adec6bffdf7a5d8d51133db280"
)
EXPECTED_POLICY = "record-id-ascending-bitset-exact-with-safe-stratum-repair-v2"
SELECTION_PATH = Path("src/twelve_six/data/current_clean_balanced_selection_v1.py")
SCHEMA = "12-6.d03-record-realization-product-qualification.v2"
MAX_INPUT_BYTES = 64 * 1024 * 1024


class QualificationError(RuntimeError):
    """Raised when exact qualification authority or behavior drifts."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise QualificationError(message)


def reject_constant(value: str) -> None:
    raise QualificationError(f"non-finite JSON constant: {value}")


def strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def load_json(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    require(len(raw) <= MAX_INPUT_BYTES, "qualification input exceeds byte bound")
    value = json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=strict_object,
        parse_constant=reject_constant,
    )
    require(type(value) is dict, "qualification input must contain an object")
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


def verify_self_hash(document: dict[str, Any], field: str, expected: str) -> None:
    require(document.get(field) == expected, f"{field} authority drift")
    core = dict(document)
    del core[field]
    require(digest(core) == expected, f"{field} self-hash mismatch")


def git_blob_sha1(path: Path) -> str:
    payload = path.read_bytes()
    header = f"blob {len(payload)}\0".encode("ascii")
    return hashlib.sha1(header + payload, usedforsecurity=False).hexdigest()


def balance_stratum(row: dict[str, Any]) -> str:
    modality = row.get("modality")
    if modality in {"uk", "ua"}:
        return "ua"
    if modality in {"en", "code"}:
        return str(modality)
    raise QualificationError("unsupported inventory modality")


def verify_inputs(
    composition: dict[str, Any],
    balance: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    require(git_blob_sha1(SELECTION_PATH) == SELECTION_BLOB, "Product selector blob drift")
    require(SELECTION_REALIZATION_POLICY == EXPECTED_POLICY, "realization policy drift")

    verify_self_hash(composition, "composition_identity_sha256", COMPOSITION_ID)
    verify_self_hash(balance, "result_identity_sha256", BALANCE_ID)
    require(composition.get("source_git_sha") == BALANCE_HEAD, "composition head drift")
    require(balance.get("policy_identity_sha256") == POLICY_ID, "balance policy drift")
    require(
        balance.get("status") == "TARGET_20M_SOURCE_MIX_FEASIBLE",
        "terminal balance status drift",
    )
    require(
        balance.get("maximum_feasible_total_source_bytes") == TARGET_TOTAL,
        "terminal balance total drift",
    )
    require(
        balance.get("maximum_feasible_stratum_bytes") == TARGET_STRATA,
        "terminal stratum target drift",
    )

    inventory = composition.get("combined_inventory")
    require(type(inventory) is dict, "combined inventory missing")
    rows = inventory.get("records")
    require(type(rows) is list, "combined inventory records missing")
    require(len(rows) == EXPECTED_RECORDS, "combined record count drift")
    require(inventory.get("record_count") == EXPECTED_RECORDS, "declared record count drift")
    require(
        len({row["source_id"] for row in rows}) == EXPECTED_SOURCE_OBJECTS,
        "physical source count drift",
    )
    require(
        inventory.get("total_payload_bytes") == EXPECTED_INPUT_BYTES,
        "declared input bytes drift",
    )
    require(
        sum(int(row["payload_bytes"]) for row in rows) == EXPECTED_INPUT_BYTES,
        "physical input bytes drift",
    )
    require(
        len({row["record_id"] for row in rows}) == EXPECTED_RECORDS,
        "record identity replay",
    )
    require(
        len({row["payload_sha256"] for row in rows}) == EXPECTED_RECORDS,
        "payload identity replay",
    )
    require(
        digest(rows) == inventory.get("record_inventory_digest_sha256"),
        "record inventory root drift",
    )
    payload_projection = [
        {
            "record_id": row["record_id"],
            "payload_sha256": row["payload_sha256"],
            "payload_bytes": row["payload_bytes"],
        }
        for row in rows
    ]
    require(
        digest(payload_projection) == inventory.get("payload_inventory_digest_sha256"),
        "payload inventory root drift",
    )

    allocations = balance.get("deterministic_maximum_allocation")
    require(type(allocations) is list and allocations, "deterministic allocation missing")
    return [dict(row) for row in rows], [dict(row) for row in allocations]


def execute(
    composition: dict[str, Any],
    balance: dict[str, Any],
) -> dict[str, Any]:
    rows, allocations = verify_inputs(composition, balance)

    by_family: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    by_stratum: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_family[str(row["family"])].append(row)
        by_stratum[balance_stratum(row)].append(row)

    exact_family_witnesses: dict[str, list[dict[str, Any]]] = {}
    impossible: list[str] = []
    for allocation in sorted(allocations, key=lambda item: str(item["family_id"])):
        family = str(allocation["family_id"])
        try:
            exact_family_witnesses[family] = _exact_record_subset(
                by_family[family],
                target_bytes=int(allocation["allocated_bytes"]),
                family=family,
            )
        except _NoExactRecordSubset:
            impossible.append(family)

    require(impossible == EXPECTED_IMPOSSIBLE, "canonical impossible-family set drift")
    for family in EXPECTED_FALSE_BUDGET_BLOCKERS:
        require(family in exact_family_witnesses, f"false budget blocker persists: {family}")

    allocations_by_stratum: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for allocation in allocations:
        allocations_by_stratum[str(allocation["stratum"])].append(allocation)

    fallback_eligibility: dict[str, bool] = {}
    stratum_caps: dict[str, int] = {}
    for stratum in ("ua", "en", "code"):
        caps = {
            int(allocation["effective_family_cap_bytes"])
            for allocation in allocations_by_stratum[stratum]
        }
        require(len(caps) == 1, f"canonical cap drift in {stratum}")
        cap = caps.pop()
        stratum_caps[stratum] = cap
        capacities = [
            sum(int(row["payload_bytes"]) for row in family_rows)
            for family, family_rows in by_family.items()
            if balance_stratum(family_rows[0]) == stratum
        ]
        fallback_eligibility[stratum] = all(value <= cap for value in capacities)
    require(
        fallback_eligibility == EXPECTED_FALLBACK_ELIGIBILITY,
        "fallback eligibility drift",
    )

    selected: list[dict[str, Any]] = []
    repaired_strata: list[str] = []
    for stratum in ("ua", "en", "code"):
        stratum_allocations = allocations_by_stratum[stratum]
        if all(
            str(allocation["family_id"]) in exact_family_witnesses
            for allocation in stratum_allocations
        ):
            for allocation in sorted(
                stratum_allocations,
                key=lambda item: str(item["family_id"]),
            ):
                selected.extend(exact_family_witnesses[str(allocation["family_id"])])
            continue

        require(
            fallback_eligibility[stratum],
            f"required stratum fallback is not cap-safe: {stratum}",
        )
        repaired = _exact_record_subset(
            by_stratum[stratum],
            target_bytes=TARGET_STRATA[stratum],
            family=f"qualification:{stratum}",
        )
        family_bytes: defaultdict[str, int] = defaultdict(int)
        for row in repaired:
            family_bytes[str(row["family"])] += int(row["payload_bytes"])
        require(
            all(value <= stratum_caps[stratum] for value in family_bytes.values()),
            f"repaired family cap exceeded: {stratum}",
        )
        require(
            len(family_bytes) >= int(balance["family_minimum"]["required_per_stratum"]),
            f"repaired family minimum violated: {stratum}",
        )
        selected.extend(repaired)
        repaired_strata.append(stratum)

    selected.sort(key=lambda row: str(row["record_id"]))
    require(repaired_strata == ["code"], "repaired stratum set drift")
    require(len(selected) == EXPECTED_SELECTED_RECORDS, "selected record count drift")
    require(
        len({row["record_id"] for row in selected}) == len(selected),
        "selected record replay",
    )

    selected_strata: defaultdict[str, int] = defaultdict(int)
    selected_families: defaultdict[str, int] = defaultdict(int)
    selected_family_sets: defaultdict[str, set[str]] = defaultdict(set)
    for row in selected:
        stratum = balance_stratum(row)
        selected_strata[stratum] += int(row["payload_bytes"])
        selected_families[str(row["family"])] += int(row["payload_bytes"])
        selected_family_sets[stratum].add(str(row["family"]))

    require(dict(selected_strata) == TARGET_STRATA, "selected stratum bytes drift")
    require(sum(selected_strata.values()) == TARGET_TOTAL, "selected total bytes drift")
    selected_family_counts = {
        stratum: len(selected_family_sets[stratum])
        for stratum in ("ua", "en", "code")
    }
    require(
        selected_family_counts == EXPECTED_SELECTED_FAMILY_COUNTS,
        "selected family counts drift",
    )

    membership = [row["record_id"] for row in selected]
    payload_projection = [
        {
            "record_id": row["record_id"],
            "payload_sha256": row["payload_sha256"],
            "payload_bytes": row["payload_bytes"],
        }
        for row in selected
    ]
    membership_root = digest(membership)
    payload_root = digest(payload_projection)
    family_bytes_root = digest(dict(sorted(selected_families.items())))
    require(membership_root == EXPECTED_MEMBERSHIP_SHA256, "membership witness drift")
    require(payload_root == EXPECTED_PAYLOAD_ROOT_SHA256, "payload witness drift")
    require(
        family_bytes_root == EXPECTED_FAMILY_BYTES_ROOT_SHA256,
        "family-byte witness drift",
    )

    core: dict[str, Any] = {
        "schema_version": SCHEMA,
        "execution_profile": "LOCAL_FREE",
        "product_head_sha": PRODUCT_HEAD,
        "product_parent_sha": PRODUCT_PARENT,
        "product_selection_blob_sha1": SELECTION_BLOB,
        "balance_head_sha": BALANCE_HEAD,
        "composition_identity_sha256": COMPOSITION_ID,
        "balance_result_identity_sha256": BALANCE_ID,
        "balance_policy_identity_sha256": POLICY_ID,
        "selection_realization_policy": SELECTION_REALIZATION_POLICY,
        "allocated_family_count": len(allocations),
        "canonical_family_exact_count": len(exact_family_witnesses),
        "hard_impossible_canonical_families": impossible,
        "budget_false_blocker_families_proven_exact": EXPECTED_FALSE_BUDGET_BLOCKERS,
        "fallback_eligibility_by_stratum": fallback_eligibility,
        "repaired_strata": repaired_strata,
        "selected_record_count": len(selected),
        "selected_total_source_bytes": sum(selected_strata.values()),
        "selected_stratum_source_bytes": dict(selected_strata),
        "selected_family_counts": selected_family_counts,
        "selected_membership_sha256": membership_root,
        "selected_payload_projection_sha256": payload_root,
        "selected_family_bytes_sha256": family_bytes_root,
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
        result = execute(
            load_json(args.composition_json),
            load_json(args.balance_result_json),
        )
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

    print("D03_PRODUCT_RECORD_REALIZATION_QUALIFIED")
    print(
        json.dumps(
            {
                "qualification_identity_sha256": result[
                    "qualification_identity_sha256"
                ],
                "selected_record_count": result["selected_record_count"],
                "selected_total_source_bytes": result["selected_total_source_bytes"],
                "repaired_strata": result["repaired_strata"],
                "hard_impossible_canonical_families": result[
                    "hard_impossible_canonical_families"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
