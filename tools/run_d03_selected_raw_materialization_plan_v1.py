"""Plan exact raw materialization for the physically qualified 20M D03 selection.

Execution-only carrier. It derives the selected record IDs from the exact #3045
text-free inventory with the Product selector helper, authenticates the raw records
already retained by #2211, and emits only a text-free missing-record plan.

No raw payload is copied to durable output and no corpus/tokenizer/training credit is
granted by this plan.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NoReturn

from twelve_six.data.current_clean_balanced_selection_v1 import _exact_record_subset

SCHEMA = "12-6.d03-selected-raw-materialization-plan.v1"
PRODUCT_HEAD = "efadbdac0c03e7c020501652f5624b96f289ec2e"
PRODUCT_SELECTOR_BLOB = "79eed6a6d3b941e89132f072593db1c74bd1d90d"
COMPOSITION_ID = "84741c6cd99d06dbc55fb413b7b150573af7b010729b23839f4fdabcf8f505ef"
BALANCE_ID = "a6077ab075e21bb072c3239bc9e9839f8ce49dd4bb5bbd4a4d342c3de54da108"
BALANCE_POLICY_ID = "9a9242f47981c25e754fc95e2650050da4e4195aa1ef3a78f2c293f9e25d7ff7"
TARGET_TOTAL = 20_000_000
TARGET_STRATA = {"ua": 9_000_000, "en": 7_000_000, "code": 4_000_000}
EXPECTED_SELECTED_RECORDS = 5_294
EXPECTED_MEMBERSHIP_SHA256 = (
    "a00572acfb1033f7bfe55e7e1f53ae9eb44b6fa039f3f378a156bec6fd13615f"
)
EXPECTED_PAYLOAD_ROOT_SHA256 = (
    "9bcfc6eb09418c93a00373df6d4c4cf3e5795d54e12c0e4bc106eaecbc89de82"
)
EXPECTED_FAMILY_BYTES_ROOT_SHA256 = (
    "f54dec34d6c86506d3321686e63af0c226f4f7adec6bffdf7a5d8d51133db280"
)
EXPECTED_BASE_RAW_RECORDS = 254
EXPECTED_BASE_RAW_BYTES = 5_428_358
EXPECTED_BASE_SELECTED_RECORDS = 211
EXPECTED_BASE_SELECTED_BYTES = 3_582_998
EXPECTED_MISSING_RECORDS = 5_083
EXPECTED_MISSING_BYTES = 16_417_002
MAX_JSON_BYTES = 64 * 1024 * 1024
MAX_JSONL_BYTES = 16 * 1024 * 1024

MATERIALIZER_AUTHORITIES: dict[str, dict[str, Any]] = {
    "code.scipy.project": {
        "lane_pr": 2752,
        "head_sha": "62dc9f487e0b3dd9eb2a8367a01061f8f55799ef",
        "role": "code-delta-post-qp",
    },
    "github:Textualize/rich": {
        "lane_pr": 2752,
        "head_sha": "62dc9f487e0b3dd9eb2a8367a01061f8f55799ef",
        "role": "code-delta-post-qp",
    },
    "github:fastapi/fastapi": {
        "lane_pr": 2752,
        "head_sha": "62dc9f487e0b3dd9eb2a8367a01061f8f55799ef",
        "role": "code-delta-post-qp",
    },
    "github:fastapi/typer": {
        "lane_pr": 2752,
        "head_sha": "62dc9f487e0b3dd9eb2a8367a01061f8f55799ef",
        "role": "code-delta-post-qp",
    },
    "github:pallets/flask": {
        "lane_pr": 2752,
        "head_sha": "62dc9f487e0b3dd9eb2a8367a01061f8f55799ef",
        "role": "code-delta-post-qp",
    },
    "github:pandas-dev/pandas": {
        "lane_pr": 2752,
        "head_sha": "62dc9f487e0b3dd9eb2a8367a01061f8f55799ef",
        "role": "code-delta-post-qp",
    },
    "github:pydantic/pydantic": {
        "lane_pr": 2752,
        "head_sha": "62dc9f487e0b3dd9eb2a8367a01061f8f55799ef",
        "role": "code-delta-post-qp",
    },
    "common-pile/ubuntu_irc": {
        "lane_pr": 2780,
        "head_sha": "fb9c3d9c137b0ccacf40dbee8d6dc75db7a2209f",
        "role": "ubuntu-post-qp",
    },
    "en.loc.selected-digitized-books": {
        "lane_pr": 2788,
        "head_sha": "0aeddc6c32917dee110a669ffea55d55a399c1c9",
        "role": "pep-loc-post-qp",
    },
    "ua.languk.supreme-court-decisions": {
        "lane_pr": 3039,
        "head_sha": "f6a2a440aacebec6a44a5f2ba955902970ff4d50",
        "role": "languk-post-dedup-current-clean",
    },
    "ua.literature.lesia-ukrainka.na-krylah-pisen.1892-lviv": {
        "lane_pr": 2810,
        "head_sha": "1d594e048365fd2121433210b83c0dc9069c50bf",
        "role": "lesia-post-qp",
    },
    "ua.nbu.official-resolutions": {
        "lane_pr": 2815,
        "head_sha": "dc4822ec1365494507bca1d67bab1b334a3fa143",
        "role": "nbu-post-qp",
    },
    "ua.rada.open-data.laws-texts": {
        "lane_pr": 2920,
        "head_sha": "a63d7c88ccb5fa380b7af4b44bde32433cf35877",
        "role": "rada-post-data232-g05-g06-two-clean",
    },
    "ua.verba.public-domain.franko1901": {
        "lane_pr": 2766,
        "head_sha": "4280ac3a6b901906b0b38b5ff0ce0e74a90dd154",
        "role": "franko-post-qp",
    },
}

EXPECTED_MISSING_BY_FAMILY = {
    "code.scipy.project": (2, 78_307),
    "common-pile/ubuntu_irc": (784, 4_200_000),
    "en.loc.selected-digitized-books": (14, 2_800_000),
    "github:Textualize/rich": (6, 46_162),
    "github:fastapi/fastapi": (3, 19_857),
    "github:fastapi/typer": (1, 7_599),
    "github:pallets/flask": (1, 15_521),
    "github:pandas-dev/pandas": (1, 15_837),
    "github:pydantic/pydantic": (4, 235_198),
    "ua.languk.supreme-court-decisions": (256, 2_880_510),
    "ua.literature.lesia-ukrainka.na-krylah-pisen.1892-lviv": (23, 32_006),
    "ua.nbu.official-resolutions": (40, 891_322),
    "ua.rada.open-data.laws-texts": (2_584, 5_000_000),
    "ua.verba.public-domain.franko1901": (1_364, 194_683),
}


class PlanError(RuntimeError):
    """Fail-closed planning error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PlanError(message)


def blocked_constant(value: str) -> NoReturn:
    raise PlanError(f"non-finite JSON constant: {value}")


def strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256(value: bytes | Any) -> str:
    payload = value if isinstance(value, bytes) else canonical(value)
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    require(0 < len(raw) <= MAX_JSON_BYTES, f"JSON input size invalid: {path}")
    value = json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=strict_object,
        parse_constant=blocked_constant,
    )
    require(type(value) is dict, f"JSON root must be object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    raw = path.read_bytes()
    require(0 < len(raw) <= MAX_JSONL_BYTES, "raw survivor JSONL size invalid")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
        require(bool(line), f"blank JSONL line: {line_number}")
        row = json.loads(
            line,
            object_pairs_hook=strict_object,
            parse_constant=blocked_constant,
        )
        require(type(row) is dict, f"JSONL row must be object: {line_number}")
        rows.append(row)
    return rows


def verify_self_hash(document: Mapping[str, Any], field: str, expected: str) -> None:
    require(document.get(field) == expected, f"{field} identity drift")
    core = dict(document)
    del core[field]
    require(sha256(core) == expected, f"{field} self-hash mismatch")


def stratum(row: Mapping[str, Any]) -> str:
    modality = row.get("modality")
    if modality == "code":
        return "code"
    if modality == "en":
        return "en"
    if modality in {"uk", "ua"}:
        return "ua"
    raise PlanError(f"unsupported modality: {modality!r}")


def derive_selected(
    composition: Mapping[str, Any],
    balance: Mapping[str, Any],
) -> list[dict[str, Any]]:
    verify_self_hash(composition, "composition_identity_sha256", COMPOSITION_ID)
    verify_self_hash(balance, "result_identity_sha256", BALANCE_ID)
    require(balance.get("policy_identity_sha256") == BALANCE_POLICY_ID, "policy drift")
    require(
        balance.get("maximum_feasible_total_source_bytes") == TARGET_TOTAL,
        "balance total drift",
    )
    require(
        balance.get("maximum_feasible_stratum_bytes") == TARGET_STRATA,
        "balance stratum drift",
    )

    inventory = composition.get("combined_inventory")
    require(isinstance(inventory, Mapping), "combined inventory missing")
    rows = inventory.get("records")
    require(type(rows) is list and rows, "combined inventory rows missing")
    normalized = [dict(row) for row in rows]

    by_family: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in normalized:
        by_family[str(row["family"])].append(row)

    allocations = balance.get("deterministic_maximum_allocation")
    require(type(allocations) is list and allocations, "allocation missing")

    selected: list[dict[str, Any]] = []
    for allocation in sorted(allocations, key=lambda item: str(item["family_id"])):
        if allocation["stratum"] == "code":
            continue
        family = str(allocation["family_id"])
        chosen = _exact_record_subset(
            by_family[family],
            target_bytes=int(allocation["allocated_bytes"]),
            family=family,
        )
        selected.extend(chosen)

    code_rows = [row for row in normalized if stratum(row) == "code"]
    selected.extend(
        _exact_record_subset(
            code_rows,
            target_bytes=TARGET_STRATA["code"],
            family="plan:code",
        )
    )
    selected.sort(key=lambda row: str(row["record_id"]))

    require(len(selected) == EXPECTED_SELECTED_RECORDS, "selected record count drift")
    require(
        sum(int(row["payload_bytes"]) for row in selected) == TARGET_TOTAL,
        "selected byte total drift",
    )
    membership = [str(row["record_id"]) for row in selected]
    require(sha256(membership) == EXPECTED_MEMBERSHIP_SHA256, "membership root drift")
    projection = [
        {
            "record_id": row["record_id"],
            "payload_sha256": row["payload_sha256"],
            "payload_bytes": row["payload_bytes"],
        }
        for row in selected
    ]
    require(
        sha256(projection) == EXPECTED_PAYLOAD_ROOT_SHA256,
        "payload projection root drift",
    )
    family_bytes: defaultdict[str, int] = defaultdict(int)
    for row in selected:
        family_bytes[str(row["family"])] += int(row["payload_bytes"])
    require(
        sha256(dict(sorted(family_bytes.items()))) == EXPECTED_FAMILY_BYTES_ROOT_SHA256,
        "family-byte root drift",
    )
    return selected


def authenticate_base_raw(
    rows: list[dict[str, Any]],
    composition: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    inventory = composition["combined_inventory"]["records"]
    by_id = {str(row["record_id"]): row for row in inventory}
    require(len(rows) == EXPECTED_BASE_RAW_RECORDS, "base raw record count drift")

    authenticated: dict[str, dict[str, Any]] = {}
    total = 0
    for row in rows:
        record_id = row.get("record_id")
        require(type(record_id) is str and record_id, "base raw record id invalid")
        require(record_id not in authenticated, f"base raw record replay: {record_id}")
        authority = by_id.get(record_id)
        require(type(authority) is dict, f"base raw record absent from #3045: {record_id}")
        payload = row.get("normalized_payload")
        require(type(payload) is str and payload, f"base raw payload missing: {record_id}")
        raw = payload.encode("utf-8")
        require(
            len(raw) == authority.get("payload_bytes"),
            f"base raw byte drift: {record_id}",
        )
        require(
            sha256(raw) == authority.get("payload_sha256"),
            f"base raw SHA drift: {record_id}",
        )
        for field in ("source_id", "family", "modality"):
            require(
                row.get(field) == authority.get(field),
                f"base raw metadata drift: {record_id}:{field}",
            )
        total += len(raw)
        authenticated[record_id] = row
    require(total == EXPECTED_BASE_RAW_BYTES, "base raw aggregate bytes drift")
    return authenticated


def build_plan(
    composition: Mapping[str, Any],
    balance: Mapping[str, Any],
    raw_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    selected = derive_selected(composition, balance)
    base = authenticate_base_raw(raw_rows, composition)

    available = [row for row in selected if str(row["record_id"]) in base]
    missing = [row for row in selected if str(row["record_id"]) not in base]
    require(
        len(available) == EXPECTED_BASE_SELECTED_RECORDS,
        "base-selected record count drift",
    )
    require(
        sum(int(row["payload_bytes"]) for row in available)
        == EXPECTED_BASE_SELECTED_BYTES,
        "base-selected bytes drift",
    )
    require(len(missing) == EXPECTED_MISSING_RECORDS, "missing record count drift")
    require(
        sum(int(row["payload_bytes"]) for row in missing) == EXPECTED_MISSING_BYTES,
        "missing bytes drift",
    )

    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in missing:
        grouped[str(row["family"])].append(row)
    observed = {
        family: (
            len(family_rows),
            sum(int(row["payload_bytes"]) for row in family_rows),
        )
        for family, family_rows in grouped.items()
    }
    require(observed == EXPECTED_MISSING_BY_FAMILY, "missing family vector drift")
    require(
        set(grouped) == set(MATERIALIZER_AUTHORITIES),
        "materializer authority coverage drift",
    )

    family_plan: list[dict[str, Any]] = []
    for family in sorted(grouped):
        family_rows = sorted(grouped[family], key=lambda row: str(row["record_id"]))
        authority = MATERIALIZER_AUTHORITIES[family]
        family_plan.append(
            {
                "family": family,
                "stratum": stratum(family_rows[0]),
                "missing_record_count": len(family_rows),
                "missing_payload_bytes": sum(
                    int(row["payload_bytes"]) for row in family_rows
                ),
                "missing_record_ids": [row["record_id"] for row in family_rows],
                "missing_payload_projection_sha256": sha256(
                    [
                        {
                            "record_id": row["record_id"],
                            "payload_sha256": row["payload_sha256"],
                            "payload_bytes": row["payload_bytes"],
                        }
                        for row in family_rows
                    ]
                ),
                "materializer_authority": authority,
            }
        )

    core: dict[str, Any] = {
        "schema_version": SCHEMA,
        "execution_profile": "LOCAL_FREE",
        "product_head_sha": PRODUCT_HEAD,
        "product_selector_blob_sha1": PRODUCT_SELECTOR_BLOB,
        "composition_identity_sha256": COMPOSITION_ID,
        "balance_result_identity_sha256": BALANCE_ID,
        "balance_policy_identity_sha256": BALANCE_POLICY_ID,
        "selected_record_count": len(selected),
        "selected_source_bytes": TARGET_TOTAL,
        "selected_membership_sha256": EXPECTED_MEMBERSHIP_SHA256,
        "selected_payload_projection_sha256": EXPECTED_PAYLOAD_ROOT_SHA256,
        "selected_family_bytes_sha256": EXPECTED_FAMILY_BYTES_ROOT_SHA256,
        "base_raw_authority": {
            "origin_pr": 2211,
            "artifact_id": 11181030848,
            "artifact_zip_sha256": (
                "3eef740b52e9db962b01b63a0a48474f735694dca4c3848e3e9c53d8cd7f1988"
            ),
            "raw_record_count": EXPECTED_BASE_RAW_RECORDS,
            "raw_payload_bytes": EXPECTED_BASE_RAW_BYTES,
            "selected_record_count": len(available),
            "selected_payload_bytes": sum(
                int(row["payload_bytes"]) for row in available
            ),
            "selected_record_ids_sha256": sha256(
                [row["record_id"] for row in available]
            ),
        },
        "missing_record_count": len(missing),
        "missing_payload_bytes": sum(int(row["payload_bytes"]) for row in missing),
        "missing_family_count": len(family_plan),
        "family_materialization_plan": family_plan,
        "durable_output_contains_raw_payload": False,
        "split_application_executed": False,
        "pack_application_executed": False,
        "tokenizer_fit_authorized": False,
        "canonical_capacity_credited": 0,
        "training_authorized_bytes": 0,
        "authorized_unique_loss_positions": 0,
        "authorized_optimized_target_exposure": 0,
        "optimizer_updates_executed_on_real_targets": 0,
        "training_executed": False,
        "learned_weights_created": False,
        "final_test_outcomes_read": False,
        "paid_compute_used": False,
        "scale_promotion_authorized": False,
    }
    core["plan_identity_sha256"] = sha256(core)
    return core


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(allow_abbrev=False)
    result.add_argument("--composition-json", type=Path, required=True)
    result.add_argument("--balance-result-json", type=Path, required=True)
    result.add_argument("--base-raw-jsonl", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        plan = build_plan(
            load_json(args.composition_json),
            load_json(args.balance_result_json),
            load_jsonl(args.base_raw_jsonl),
        )
        require(not args.output.exists(), "output already exists")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(canonical(plan) + b"\n")
    except (
        PlanError,
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"BLOCKED: {exc}")
        return 2
    print("D03_SELECTED_RAW_MATERIALIZATION_PLAN=PASS_ZERO_CREDIT")
    print(f"PLAN_IDENTITY_SHA256={plan['plan_identity_sha256']}")
    print(f"BASE_SELECTED_BYTES={plan['base_raw_authority']['selected_payload_bytes']}")
    print(f"MISSING_BYTES={plan['missing_payload_bytes']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
