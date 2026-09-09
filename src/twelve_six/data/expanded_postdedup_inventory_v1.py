from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

SCHEMA = "twelve-six.expanded-postdedup-inventory.v1"
UPSTREAM_SURVIVOR_SCHEMA = "twelve-six.expanded-global-dedup-survivors.v1"
HEX64 = frozenset("0123456789abcdef")

ZERO_TRUTH = {
    "training_eligible": False,
    "evaluation_eligible": False,
    "authorized_optimized_target_exposure": 0,
    "tokenizer_fit": False,
    "optimizer_updates": 0,
    "model_training": False,
    "final_test_outcomes_accessed": False,
    "paid_compute_used": False,
}

REQUIRED_ROW_KEYS = (
    "record_id",
    "source_id",
    "family",
    "modality",
    "payload_sha256",
    "payload_bytes",
    "comparison_policy_id",
    "comparison_sha256",
    "comparison_bytes",
    "training_eligible",
    "evaluation_eligible",
)


def _canonical_bytes(value: Any) -> bytes:
    text = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return (text + "\n").encode("utf-8")


def _self_hash(document: Mapping[str, Any], key: str) -> str:
    payload = dict(document)
    payload.pop(key, None)
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _require_hex64(value: Any, label: str) -> str:
    valid = (
        isinstance(value, str)
        and len(value) == 64
        and not any(ch not in HEX64 for ch in value)
    )
    if not valid:
        raise ValueError(f"{label} must be a lowercase 64-hex SHA-256")
    return value


def _require_nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _require_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _verify_zero_truth(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    for key, expected in ZERO_TRUTH.items():
        if value.get(key) != expected:
            raise ValueError(f"{label}.{key} must remain {expected!r}")
    return {key: ZERO_TRUTH[key] for key in ZERO_TRUTH}


def _validate_row(row: Any, index: int) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise ValueError(f"survivors[{index}] must be an object")
    missing = [key for key in REQUIRED_ROW_KEYS if key not in row]
    if missing:
        raise ValueError(f"survivors[{index}] missing keys: {', '.join(missing)}")

    prefix = f"survivors[{index}]"
    normalized = {
        "record_id": _require_nonempty_string(row["record_id"], f"{prefix}.record_id"),
        "source_id": _require_nonempty_string(row["source_id"], f"{prefix}.source_id"),
        "family": _require_nonempty_string(row["family"], f"{prefix}.family"),
        "modality": _require_nonempty_string(row["modality"], f"{prefix}.modality"),
        "payload_sha256": _require_hex64(row["payload_sha256"], f"{prefix}.payload_sha256"),
        "payload_bytes": _require_int(
            row["payload_bytes"],
            f"{prefix}.payload_bytes",
            minimum=1,
        ),
        "comparison_policy_id": _require_nonempty_string(
            row["comparison_policy_id"],
            f"{prefix}.comparison_policy_id",
        ),
        "comparison_sha256": _require_hex64(
            row["comparison_sha256"],
            f"{prefix}.comparison_sha256",
        ),
        "comparison_bytes": _require_int(
            row["comparison_bytes"],
            f"{prefix}.comparison_bytes",
            minimum=0,
        ),
        "training_eligible": row["training_eligible"],
        "evaluation_eligible": row["evaluation_eligible"],
    }
    if normalized["training_eligible"] is not False:
        raise ValueError(f"survivors[{index}] widens training eligibility")
    if normalized["evaluation_eligible"] is not False:
        raise ValueError(f"survivors[{index}] widens evaluation eligibility")
    return normalized


def freeze_expanded_inventory(
    report: Mapping[str, Any],
    survivor_authority: Mapping[str, Any],
    *,
    expected_report_sha256: str,
    expected_survivor_authority_sha256: str,
) -> dict[str, Any]:
    """Freeze an externally selected expanded-dedup survivor set."""
    expected_report_sha256 = _require_hex64(
        expected_report_sha256,
        "expected_report_sha256",
    )
    expected_survivor_authority_sha256 = _require_hex64(
        expected_survivor_authority_sha256,
        "expected_survivor_authority_sha256",
    )

    report_sha = _require_hex64(
        report.get("report_sha256"),
        "report.report_sha256",
    )
    report_hash_matches = report_sha == _self_hash(report, "report_sha256")
    if not report_hash_matches or report_sha != expected_report_sha256:
        raise ValueError("expanded-dedup report identity mismatch")
    _verify_zero_truth(report.get("truth_boundary"), "report.truth_boundary")

    survivor_sha = _require_hex64(
        survivor_authority.get("survivor_authority_sha256"),
        "survivor_authority.survivor_authority_sha256",
    )
    if survivor_sha != _self_hash(survivor_authority, "survivor_authority_sha256"):
        raise ValueError("survivor authority self-hash mismatch")
    if survivor_sha != expected_survivor_authority_sha256:
        raise ValueError("survivor authority external identity mismatch")
    if survivor_authority.get("schema") != UPSTREAM_SURVIVOR_SCHEMA:
        raise ValueError("unexpected survivor authority schema")
    if report.get("survivor_authority_sha256") != survivor_sha:
        raise ValueError("report/survivor authority binding mismatch")
    _verify_zero_truth(
        survivor_authority.get("truth_boundary"),
        "survivor_authority.truth_boundary",
    )

    raw_rows = survivor_authority.get("survivors")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError("survivor_authority.survivors must be a non-empty list")
    rows = [_validate_row(row, index) for index, row in enumerate(raw_rows)]

    record_ids = [row["record_id"] for row in rows]
    if len(set(record_ids)) != len(record_ids):
        raise ValueError("duplicate record_id in survivor authority")

    survivor_count = _require_int(
        report.get("survivor_count"),
        "report.survivor_count",
        minimum=1,
    )
    retained_bytes = _require_int(
        report.get("retained_payload_bytes"),
        "report.retained_payload_bytes",
        minimum=1,
    )
    authority_count = survivor_authority.get("survivor_count")
    if survivor_count != len(rows) or authority_count != survivor_count:
        raise ValueError("survivor count arithmetic mismatch")
    payload_sum = sum(row["payload_bytes"] for row in rows)
    authority_bytes = survivor_authority.get("retained_payload_bytes")
    if retained_bytes != payload_sum or authority_bytes != retained_bytes:
        raise ValueError("retained payload byte arithmetic mismatch")

    sorted_rows = sorted(rows, key=lambda row: row["record_id"])
    inventory: dict[str, Any] = {
        "schema": SCHEMA,
        "input_report_sha256": report_sha,
        "input_survivor_authority_sha256": survivor_sha,
        "record_count": len(sorted_rows),
        "source_count": len({row["source_id"] for row in sorted_rows}),
        "retained_payload_bytes": payload_sum,
        "records": sorted_rows,
        "truth_boundary": dict(ZERO_TRUTH),
    }
    inventory["inventory_identity_sha256"] = _self_hash(
        inventory,
        "inventory_identity_sha256",
    )
    return inventory


def verify_inventory(
    inventory: Mapping[str, Any],
    *,
    expected_inventory_identity_sha256: str,
) -> None:
    expected = _require_hex64(
        expected_inventory_identity_sha256,
        "expected_inventory_identity_sha256",
    )
    actual = _require_hex64(
        inventory.get("inventory_identity_sha256"),
        "inventory.inventory_identity_sha256",
    )
    if actual != _self_hash(inventory, "inventory_identity_sha256") or actual != expected:
        raise ValueError("retained inventory identity mismatch")
    if inventory.get("schema") != SCHEMA:
        raise ValueError("unexpected retained inventory schema")
    _verify_zero_truth(inventory.get("truth_boundary"), "inventory.truth_boundary")

    rows = inventory.get("records")
    if not isinstance(rows, list) or not rows:
        raise ValueError("inventory.records must be non-empty")
    validated = [_validate_row(row, index) for index, row in enumerate(rows)]
    if validated != sorted(validated, key=lambda row: row["record_id"]):
        raise ValueError("inventory records must be sorted by record_id")
    if len({row["record_id"] for row in validated}) != len(validated):
        raise ValueError("duplicate record_id in retained inventory")
    if inventory.get("record_count") != len(validated):
        raise ValueError("inventory record_count mismatch")
    source_count = len({row["source_id"] for row in validated})
    if inventory.get("source_count") != source_count:
        raise ValueError("inventory source_count mismatch")
    retained_payload_bytes = sum(row["payload_bytes"] for row in validated)
    if inventory.get("retained_payload_bytes") != retained_payload_bytes:
        raise ValueError("inventory retained_payload_bytes mismatch")


def prepare_ephemeral_data232_rows(
    inventory: Mapping[str, Any],
    payload_rows: Iterable[Mapping[str, Any]],
    *,
    expected_inventory_identity_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Bind ephemeral payload text to a frozen inventory with text-free evidence."""
    verify_inventory(
        inventory,
        expected_inventory_identity_sha256=expected_inventory_identity_sha256,
    )
    inventory_by_id = {row["record_id"]: row for row in inventory["records"]}

    seen: set[str] = set()
    ephemeral: list[dict[str, Any]] = []
    binding_rows: list[dict[str, Any]] = []
    for index, payload_row in enumerate(payload_rows):
        if not isinstance(payload_row, Mapping):
            raise ValueError(f"payload_rows[{index}] must be an object")
        record_id = _require_nonempty_string(
            payload_row.get("record_id"),
            f"payload_rows[{index}].record_id",
        )
        if record_id in seen:
            raise ValueError(f"duplicate payload row {record_id}")
        seen.add(record_id)
        if record_id not in inventory_by_id:
            raise ValueError(f"unexpected payload row {record_id}")
        expected = inventory_by_id[record_id]

        normalized_payload = payload_row.get("normalized_payload")
        comparison_payload = payload_row.get("comparison_payload")
        if not isinstance(normalized_payload, str):
            raise ValueError(f"payload row {record_id} must contain string payloads")
        if not isinstance(comparison_payload, str):
            raise ValueError(f"payload row {record_id} must contain string payloads")
        payload_bytes = normalized_payload.encode("utf-8")
        comparison_bytes = comparison_payload.encode("utf-8")
        payload_hash = hashlib.sha256(payload_bytes).hexdigest()
        if len(payload_bytes) != expected["payload_bytes"]:
            raise ValueError(f"payload identity mismatch for {record_id}")
        if payload_hash != expected["payload_sha256"]:
            raise ValueError(f"payload identity mismatch for {record_id}")
        comparison_hash = hashlib.sha256(comparison_bytes).hexdigest()
        if len(comparison_bytes) != expected["comparison_bytes"]:
            raise ValueError(f"comparison payload identity mismatch for {record_id}")
        if comparison_hash != expected["comparison_sha256"]:
            raise ValueError(f"comparison payload identity mismatch for {record_id}")

        ephemeral.append(
            {
                "record_id": record_id,
                "source_id": expected["source_id"],
                "family": expected["family"],
                "modality": expected["modality"],
                "normalized_payload": normalized_payload,
                "comparison_payload": comparison_payload,
                "comparison_policy_id": expected["comparison_policy_id"],
                "training_eligible": False,
                "evaluation_eligible": False,
            }
        )
        binding_rows.append(
            {
                "record_id": record_id,
                "payload_sha256": expected["payload_sha256"],
                "payload_bytes": expected["payload_bytes"],
                "comparison_sha256": expected["comparison_sha256"],
                "comparison_bytes": expected["comparison_bytes"],
            }
        )

    missing = sorted(set(inventory_by_id) - seen)
    if missing:
        raise ValueError(f"missing payload rows: {', '.join(missing[:5])}")

    ephemeral.sort(key=lambda row: row["record_id"])
    binding_rows.sort(key=lambda row: row["record_id"])
    evidence: dict[str, Any] = {
        "schema": "twelve-six.expanded-postdedup-data232-handoff.v1",
        "inventory_identity_sha256": expected_inventory_identity_sha256,
        "record_count": len(binding_rows),
        "retained_payload_bytes": sum(row["payload_bytes"] for row in binding_rows),
        "payload_binding_sha256": hashlib.sha256(
            _canonical_bytes(binding_rows)
        ).hexdigest(),
        "truth_boundary": dict(ZERO_TRUTH),
    }
    evidence["handoff_identity_sha256"] = _self_hash(
        evidence,
        "handoff_identity_sha256",
    )
    return ephemeral, evidence


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_bytes(value))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Freeze externally selected expanded post-dedup survivors"
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--survivor-authority", type=Path, required=True)
    parser.add_argument("--expected-report-sha256", required=True)
    parser.add_argument("--expected-survivor-authority-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    inventory = freeze_expanded_inventory(
        _load_json(args.report),
        _load_json(args.survivor_authority),
        expected_report_sha256=args.expected_report_sha256,
        expected_survivor_authority_sha256=args.expected_survivor_authority_sha256,
    )
    _write_json(args.output, inventory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
