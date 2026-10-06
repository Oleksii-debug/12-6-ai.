"""Assemble the exact #3068 selected raw cohort from authenticated physical rows.

Execution-only carrier helper.  It re-derives the exact Product selection from the
#3045 composition/balance authority, accepts the already-retained #2211 raw rows
plus freshly rematerialized post-QP rows, and publishes an ephemeral canonical raw
JSONL together with a text-free zero-credit receipt.

The raw JSONL is an execution intermediate only.  Workflows using this helper must
remove it before publishing durable artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NoReturn

import run_d03_selected_raw_materialization_plan_v1 as plan_api

SCHEMA = "12-6.d03-selected-raw-assembly.v1"
PLAN_IDENTITY = "ddf42773bb7ac4856e07c7b8e22b137581f02b79e231f0740b12e867c0981400"
EXPECTED_SELECTED_RECORDS = 5_294
EXPECTED_SELECTED_BYTES = 20_000_000
EXPECTED_BASE_SELECTED_RECORDS = 211
EXPECTED_BASE_SELECTED_BYTES = 3_582_998
EXPECTED_REMATERIALIZED_RECORDS = 5_083
EXPECTED_REMATERIALIZED_BYTES = 16_417_002
RAW_KEYS = frozenset({"record_id", "source_id", "family", "modality", "normalized_payload"})
_HEX = frozenset("0123456789abcdef")
MAX_PLAN_BYTES = 2 * 1024 * 1024
MAX_RAW_JSONL_BYTES = 512 * 1024 * 1024


class AssemblyError(RuntimeError):
    """Fail-closed selected-raw assembly error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssemblyError(message)


def blocked_constant(value: str) -> NoReturn:
    raise AssemblyError(f"non-finite JSON constant: {value}")


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
    raw = value if isinstance(value, bytes) else canonical(value)
    return hashlib.sha256(raw).hexdigest()


def load_json(path: Path, *, max_bytes: int = MAX_PLAN_BYTES) -> dict[str, Any]:
    require(
        not path.is_symlink() and path.is_file(),
        f"JSON input is not a regular file: {path}",
    )
    raw = path.read_bytes()
    require(0 < len(raw) <= max_bytes, f"JSON input size invalid: {path}")
    value = json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=strict_object,
        parse_constant=blocked_constant,
    )
    require(type(value) is dict, f"JSON root must be object: {path}")
    return value


def load_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    require(
        not path.is_symlink() and path.is_file(),
        f"{label} is not a regular file: {path}",
    )
    raw = path.read_bytes()
    require(0 < len(raw) <= MAX_RAW_JSONL_BYTES, f"{label} size invalid")
    require(raw.endswith(b"\n"), f"{label} must end with LF")
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
        require(bool(line), f"{label} contains blank line {number}")
        value = json.loads(
            line,
            object_pairs_hook=strict_object,
            parse_constant=blocked_constant,
        )
        require(type(value) is dict, f"{label} row {number} must be object")
        rows.append(value)
    require(bool(rows), f"{label} contains no rows")
    return rows


def verify_plan(value: Mapping[str, Any]) -> None:
    require(value.get("schema_version") == plan_api.SCHEMA, "plan schema drift")
    claimed = value.get("plan_identity_sha256")
    require(claimed == PLAN_IDENTITY, "plan identity drift")
    require(
        type(claimed) is str and len(claimed) == 64 and set(claimed) <= _HEX,
        "plan identity invalid",
    )
    core = dict(value)
    del core["plan_identity_sha256"]
    require(sha256(core) == claimed, "plan self-hash mismatch")
    require(
        value.get("selected_record_count") == EXPECTED_SELECTED_RECORDS,
        "plan selected count drift",
    )
    require(
        value.get("selected_source_bytes") == EXPECTED_SELECTED_BYTES,
        "plan selected bytes drift",
    )
    require(
        value.get("missing_record_count") == EXPECTED_REMATERIALIZED_RECORDS,
        "plan missing count drift",
    )
    require(
        value.get("missing_payload_bytes") == EXPECTED_REMATERIALIZED_BYTES,
        "plan missing bytes drift",
    )
    require(
        value.get("durable_output_contains_raw_payload") is False,
        "plan raw boundary widened",
    )
    for key in (
        "tokenizer_fit_authorized",
        "training_executed",
        "learned_weights_created",
        "final_test_outcomes_read",
        "paid_compute_used",
        "scale_promotion_authorized",
    ):
        require(value.get(key) is False, f"plan truth boundary widened: {key}")
    for key in (
        "canonical_capacity_credited",
        "training_authorized_bytes",
        "authorized_unique_loss_positions",
        "authorized_optimized_target_exposure",
        "optimizer_updates_executed_on_real_targets",
    ):
        require(
            type(value.get(key)) is int and value.get(key) == 0,
            f"plan truth boundary widened: {key}",
        )


def _validate_raw_row(
    row: Mapping[str, Any],
    authority: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, str]:
    require(set(row) == RAW_KEYS, f"{label} raw row schema drift")
    record_id = row.get("record_id")
    require(type(record_id) is str and bool(record_id), f"{label} record_id invalid")
    for key in ("source_id", "family", "modality", "normalized_payload"):
        require(
            type(row.get(key)) is str and bool(row[key]),
            f"{label} {record_id}:{key} invalid",
        )
    for key in ("source_id", "family", "modality"):
        require(
            row[key] == authority.get(key),
            f"{label} {record_id}:{key} authority drift",
        )
    payload = row["normalized_payload"]
    raw = payload.encode("utf-8")
    require(
        len(raw) == authority.get("payload_bytes"),
        f"{label} {record_id}:payload byte drift",
    )
    require(
        sha256(raw) == authority.get("payload_sha256"),
        f"{label} {record_id}:payload SHA drift",
    )
    return {key: str(row[key]) for key in RAW_KEYS}


def _expected_missing_ids(plan: Mapping[str, Any]) -> set[str]:
    family_plan = plan.get("family_materialization_plan")
    require(
        type(family_plan) is list and family_plan,
        "family materialization plan missing",
    )
    ids: set[str] = set()
    count = 0
    total = 0
    for item in family_plan:
        require(type(item) is dict, "family plan row must be object")
        family = item.get("family")
        raw_ids = item.get("missing_record_ids")
        require(type(family) is str and family, "family plan family invalid")
        require(
            type(raw_ids) is list and raw_ids,
            f"family plan IDs missing: {family}",
        )
        require(
            len(raw_ids) == item.get("missing_record_count"),
            f"family plan count drift: {family}",
        )
        for record_id in raw_ids:
            require(
                type(record_id) is str and record_id,
                f"family plan ID invalid: {family}",
            )
            require(
                record_id not in ids,
                f"family plan duplicate record ID: {record_id}",
            )
            ids.add(record_id)
        count += len(raw_ids)
        payload_bytes = item.get("missing_payload_bytes")
        require(
            type(payload_bytes) is int and payload_bytes > 0,
            f"family plan bytes invalid: {family}",
        )
        total += payload_bytes
    require(
        count == plan.get("missing_record_count"),
        "family plan aggregate count drift",
    )
    require(
        total == plan.get("missing_payload_bytes"),
        "family plan aggregate byte drift",
    )
    return ids


def assemble(
    *,
    selected_rows: Sequence[Mapping[str, Any]],
    plan: Mapping[str, Any],
    base_rows: Sequence[Mapping[str, Any]],
    rematerialized_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Authenticate and assemble one complete selected raw cohort."""
    verify_plan(plan)
    expected_selected_records = int(plan["selected_record_count"])
    expected_selected_bytes = int(plan["selected_source_bytes"])
    require(
        len(selected_rows) == expected_selected_records,
        "derived selected count drift",
    )
    selected_by_id: dict[str, Mapping[str, Any]] = {}
    for row in selected_rows:
        record_id = row.get("record_id")
        require(type(record_id) is str and record_id, "selected record ID invalid")
        require(
            record_id not in selected_by_id,
            f"selected record replay: {record_id}",
        )
        selected_by_id[record_id] = row
    require(
        sum(int(row["payload_bytes"]) for row in selected_rows)
        == expected_selected_bytes,
        "derived selected bytes drift",
    )
    require(
        sha256(sorted(selected_by_id)) == plan.get("selected_membership_sha256"),
        "derived selected membership drift",
    )

    missing_ids = _expected_missing_ids(plan)
    require(
        missing_ids <= set(selected_by_id),
        "plan missing set escapes selected cohort",
    )
    base_selected_ids = set(selected_by_id) - missing_ids
    base_authority = plan.get("base_raw_authority")
    require(type(base_authority) is dict, "base raw authority missing")
    expected_base_records = int(base_authority["selected_record_count"])
    expected_base_bytes = int(base_authority["selected_payload_bytes"])
    require(
        len(base_selected_ids) == expected_base_records,
        "base selected count drift",
    )
    require(
        sum(int(selected_by_id[r]["payload_bytes"]) for r in base_selected_ids)
        == expected_base_bytes,
        "base selected bytes drift",
    )
    require(
        sha256(sorted(base_selected_ids))
        == base_authority.get("selected_record_ids_sha256"),
        "base selected ID root drift",
    )

    collected: dict[str, dict[str, str]] = {}
    for number, row in enumerate(base_rows, start=1):
        record_id = row.get("record_id") if isinstance(row, Mapping) else None
        require(
            type(record_id) is str and bool(record_id),
            f"base raw row {number} record ID invalid",
        )
        if record_id not in base_selected_ids:
            continue
        require(
            record_id not in collected,
            f"base selected record replay: {record_id}",
        )
        collected[record_id] = _validate_raw_row(
            row,
            selected_by_id[record_id],
            label="base",
        )
    require(
        set(collected) == base_selected_ids,
        "base raw does not exactly cover base-selected IDs",
    )

    remat_seen: set[str] = set()
    for number, row in enumerate(rematerialized_rows, start=1):
        record_id = row.get("record_id") if isinstance(row, Mapping) else None
        require(
            type(record_id) is str and bool(record_id),
            f"rematerialized row {number} record ID invalid",
        )
        require(
            record_id in missing_ids,
            f"rematerialized row is not planned missing ID: {record_id}",
        )
        require(
            record_id not in remat_seen,
            f"rematerialized record replay: {record_id}",
        )
        remat_seen.add(record_id)
        collected[record_id] = _validate_raw_row(
            row,
            selected_by_id[record_id],
            label="rematerialized",
        )
    require(
        remat_seen == missing_ids,
        "rematerialized raw does not exactly cover planned missing IDs",
    )
    require(
        set(collected) == set(selected_by_id),
        "assembled selected raw coverage drift",
    )

    output = [collected[record_id] for record_id in sorted(collected)]
    raw_jsonl = b"".join(canonical(row) + b"\n" for row in output)
    require(
        len(output) == expected_selected_records,
        "assembled record count drift",
    )
    payload_bytes = sum(
        len(row["normalized_payload"].encode("utf-8")) for row in output
    )
    require(
        payload_bytes == expected_selected_bytes,
        "assembled payload bytes drift",
    )

    projection = [
        {
            "record_id": row["record_id"],
            "payload_sha256": sha256(
                row["normalized_payload"].encode("utf-8")
            ),
            "payload_bytes": len(
                row["normalized_payload"].encode("utf-8")
            ),
        }
        for row in output
    ]
    require(
        sha256(projection) == plan.get("selected_payload_projection_sha256"),
        "assembled payload projection drift",
    )
    family_bytes: defaultdict[str, int] = defaultdict(int)
    family_counts: defaultdict[str, int] = defaultdict(int)
    for row in output:
        family = row["family"]
        family_bytes[family] += len(
            row["normalized_payload"].encode("utf-8")
        )
        family_counts[family] += 1
    require(
        sha256(dict(sorted(family_bytes.items())))
        == plan.get("selected_family_bytes_sha256"),
        "assembled family-byte root drift",
    )

    core: dict[str, Any] = {
        "schema_version": SCHEMA,
        "execution_profile": "LOCAL_FREE",
        "plan_identity_sha256": PLAN_IDENTITY,
        "composition_identity_sha256": plan.get(
            "composition_identity_sha256"
        ),
        "balance_result_identity_sha256": plan.get(
            "balance_result_identity_sha256"
        ),
        "selected_record_count": len(output),
        "selected_payload_bytes": payload_bytes,
        "selected_membership_sha256": plan.get(
            "selected_membership_sha256"
        ),
        "selected_payload_projection_sha256": plan.get(
            "selected_payload_projection_sha256"
        ),
        "selected_family_bytes_sha256": plan.get(
            "selected_family_bytes_sha256"
        ),
        "selected_raw_jsonl_sha256": sha256(raw_jsonl),
        "base_selected_record_count": len(base_selected_ids),
        "base_selected_payload_bytes": sum(
            int(selected_by_id[r]["payload_bytes"])
            for r in base_selected_ids
        ),
        "rematerialized_record_count": len(remat_seen),
        "rematerialized_payload_bytes": sum(
            int(selected_by_id[r]["payload_bytes"])
            for r in remat_seen
        ),
        "family_record_counts": dict(sorted(family_counts.items())),
        "family_payload_bytes": dict(sorted(family_bytes.items())),
        "raw_output_is_ephemeral": True,
        "durable_output_contains_raw_payload": False,
        "cluster_safe_split_complete": False,
        "deterministic_pack_two_clean_complete": False,
        "tokenizer_fit_authorized": False,
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
    receipt = {
        **core,
        "assembly_identity_sha256": sha256(core),
    }
    return output, receipt


def _write_create_only(path: Path, payload: bytes) -> None:
    require(
        not path.exists() and not path.is_symlink(),
        f"output already exists: {path}",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    require(
        not path.parent.is_symlink(),
        f"output parent is symlink: {path.parent}",
    )
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(tmp, path)
        except FileExistsError as exc:
            raise AssemblyError(
                f"output appeared concurrently: {path}"
            ) from exc
    finally:
        tmp.unlink(missing_ok=True)


def execute(
    *,
    plan_json: Path,
    composition_json: Path,
    balance_result_json: Path,
    base_raw_jsonl: Path,
    rematerialized_jsonl: Sequence[Path],
    output_raw_jsonl: Path,
    output_receipt: Path,
) -> dict[str, Any]:
    require(
        output_raw_jsonl != output_receipt,
        "raw and receipt outputs must differ",
    )
    plan = load_json(plan_json)
    composition = plan_api.load_json(composition_json)
    balance = plan_api.load_json(balance_result_json)
    selected = plan_api.derive_selected(composition, balance)
    base_rows = load_jsonl(base_raw_jsonl, label="base raw")
    remat_rows: list[dict[str, Any]] = []
    for path in rematerialized_jsonl:
        remat_rows.extend(
            load_jsonl(path, label=f"rematerialized raw {path}")
        )
    output, receipt = assemble(
        selected_rows=selected,
        plan=plan,
        base_rows=base_rows,
        rematerialized_rows=remat_rows,
    )
    raw_payload = b"".join(
        canonical(row) + b"\n" for row in output
    )
    receipt_payload = canonical(receipt) + b"\n"
    _write_create_only(output_raw_jsonl, raw_payload)
    try:
        _write_create_only(output_receipt, receipt_payload)
    except Exception:
        output_raw_jsonl.unlink(missing_ok=True)
        raise
    return receipt


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(allow_abbrev=False)
    result.add_argument("--plan-json", type=Path, required=True)
    result.add_argument("--composition-json", type=Path, required=True)
    result.add_argument("--balance-result-json", type=Path, required=True)
    result.add_argument("--base-raw-jsonl", type=Path, required=True)
    result.add_argument(
        "--rematerialized-jsonl",
        type=Path,
        action="append",
        required=True,
    )
    result.add_argument("--output-raw-jsonl", type=Path, required=True)
    result.add_argument("--output-receipt", type=Path, required=True)
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        receipt = execute(
            plan_json=args.plan_json,
            composition_json=args.composition_json,
            balance_result_json=args.balance_result_json,
            base_raw_jsonl=args.base_raw_jsonl,
            rematerialized_jsonl=args.rematerialized_jsonl,
            output_raw_jsonl=args.output_raw_jsonl,
            output_receipt=args.output_receipt,
        )
    except (
        AssemblyError,
        plan_api.PlanError,
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 2
    print("D03_SELECTED_RAW_ASSEMBLY=PASS_ZERO_CREDIT")
    print(
        "ASSEMBLY_IDENTITY_SHA256="
        f"{receipt['assembly_identity_sha256']}"
    )
    print(
        "SELECTED_RAW_JSONL_SHA256="
        f"{receipt['selected_raw_jsonl_sha256']}"
    )
    print("AUTHORIZED_OPTIMIZED_TARGET_EXPOSURE=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
