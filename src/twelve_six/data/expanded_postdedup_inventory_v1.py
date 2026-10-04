from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

SCHEMA = "twelve-six.expanded-postdedup-inventory.v1"
HANDOFF_SCHEMA = "12-6.postdedup-decontam-handoff.v1"
UPSTREAM_REPORT_SCHEMA = "12-6.d03-expanded-global-dedup-v9-report.v1"
UPSTREAM_SURVIVOR_SCHEMA = "12-6.d03-expanded-global-dedup-v9-survivors.v1"
UPSTREAM_MATCHER_SCHEMA = "12-6.next100-065-cross-source-dedup-report.v3"
DATA526_RECORD_INVENTORY_SCHEMA = "12-6.data526-record-inventory.v1"
DATA526_RECORD_INVENTORY_SHA256 = (
    "766cc01d610373183e73a632e18674b1d7afc5aaa7e3888638029e0044b26103"
)
DATA526_PAYLOAD_INVENTORY_SHA256 = (
    "56ef7a457f4c4f649ad97359c2a177131632d838756d769e4b1a2ec3f3a2a278"
)
RADA_SOURCE_PREFIX = "rada-trees-quality-unit:"
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

UPSTREAM_REPORT_BOUNDARY = {
    "expanded_global_dedup_complete": True,
    "retained_inventory_freeze_complete": False,
    "reserved_evaluation_decontamination_complete": False,
    "family_caps_complete": False,
    "cluster_safe_split_complete": False,
    "two_clean_builds_complete": False,
    "training_authorized_bytes": 0,
    "unique_causal_loss_positions_authorized": 0,
    "tokenizer_fit_authorized": False,
    "optimizer_updates": 0,
    "model_training_executed": False,
    "final_test_payload_accessed": False,
    "paid_compute_used": False,
}

UPSTREAM_SURVIVOR_BOUNDARY = {
    "global_dedup_execution_complete": True,
    "retained_inventory_freeze_complete": False,
    "reserved_evaluation_decontamination_complete": False,
    "family_caps_complete": False,
    "training_authorized_bytes": 0,
    "unique_causal_loss_positions_authorized": 0,
    "tokenizer_fit_authorized": False,
    "optimizer_updates": 0,
    "model_training_executed": False,
    "final_test_payload_accessed": False,
    "paid_compute_used": False,
}

REQUIRED_RECORD_KEYS = (
    "record_id",
    "source_id",
    "family",
    "modality",
    "payload_sha256",
    "payload_bytes",
    "training_eligible",
    "evaluation_eligible",
)


def _canonical(value: Any, *, ascii_only: bool = False, newline: bool = False) -> bytes:
    rendered = json.dumps(
        value,
        ensure_ascii=ascii_only,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return rendered + (b"\n" if newline else b"")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _self_hash(
    document: Mapping[str, Any],
    key: str,
    *,
    ascii_only: bool = False,
    newline: bool = False,
) -> str:
    payload = dict(document)
    payload.pop(key, None)
    return _sha256(_canonical(payload, ascii_only=ascii_only, newline=newline))


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
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _require_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _require_exact_boundary(value: Any, expected: Mapping[str, Any], label: str) -> None:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    if set(value) != set(expected):
        raise ValueError(f"{label} must contain exactly the canonical keys")
    for key, wanted in expected.items():
        actual = value[key]
        if isinstance(wanted, bool):
            matches = actual is wanted
        elif isinstance(wanted, int):
            matches = isinstance(actual, int) and not isinstance(actual, bool) and actual == wanted
        else:
            matches = actual == wanted
        if not matches:
            raise ValueError(f"{label}.{key} must remain {wanted!r}")


def _verify_zero_truth(value: Any, label: str) -> None:
    _require_exact_boundary(value, ZERO_TRUTH, label)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _validate_record(row: Any, index: int) -> dict[str, Any]:
    if not isinstance(row, Mapping):
        raise TypeError(f"records[{index}] must be an object")
    missing = [key for key in REQUIRED_RECORD_KEYS if key not in row]
    if missing:
        raise ValueError(f"records[{index}] missing keys: {', '.join(missing)}")
    prefix = f"records[{index}]"
    normalized = {
        "record_id": _require_nonempty_string(row["record_id"], f"{prefix}.record_id"),
        "source_id": _require_nonempty_string(row["source_id"], f"{prefix}.source_id"),
        "family": _require_nonempty_string(row["family"], f"{prefix}.family"),
        "modality": _require_nonempty_string(row["modality"], f"{prefix}.modality"),
        "payload_sha256": _require_hex64(row["payload_sha256"], f"{prefix}.payload_sha256"),
        "payload_bytes": _require_int(row["payload_bytes"], f"{prefix}.payload_bytes", minimum=1),
        "training_eligible": row["training_eligible"],
        "evaluation_eligible": row["evaluation_eligible"],
    }
    if normalized["training_eligible"] is not False:
        raise ValueError(f"{prefix} widens training eligibility")
    if normalized["evaluation_eligible"] is not False:
        raise ValueError(f"{prefix} widens evaluation eligibility")
    return normalized


def _validate_data526_record_inventory(
    inventory: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if inventory.get("schema_version") != DATA526_RECORD_INVENTORY_SCHEMA:
        raise ValueError("unexpected DATA-526 record inventory schema")
    rows = inventory.get("records")
    if not isinstance(rows, list) or not rows:
        raise ValueError("DATA-526 record inventory must contain records")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    payload_projection: list[dict[str, Any]] = []
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            raise TypeError(f"DATA-526 records[{index}] must be an object")
        record_id = _require_nonempty_string(
            raw.get("record_id"), f"DATA-526 records[{index}].record_id"
        )
        if record_id in seen:
            raise ValueError(f"duplicate DATA-526 record_id: {record_id}")
        seen.add(record_id)
        item = {
            "record_id": record_id,
            "source_id": _require_nonempty_string(
                raw.get("source_id"), f"DATA-526 records[{index}].source_id"
            ),
            "family": _require_nonempty_string(
                raw.get("family"), f"DATA-526 records[{index}].family"
            ),
            "modality": _require_nonempty_string(
                raw.get("modality"), f"DATA-526 records[{index}].modality"
            ),
            "payload_sha256": _require_hex64(
                raw.get("payload_sha256"), f"DATA-526 records[{index}].payload_sha256"
            ),
            "payload_bytes": _require_int(
                raw.get("payload_bytes"), f"DATA-526 records[{index}].payload_bytes", minimum=1
            ),
        }
        normalized.append(item)
        payload_projection.append(
            {
                "record_id": record_id,
                "payload_sha256": item["payload_sha256"],
                "payload_bytes": item["payload_bytes"],
            }
        )
    normalized.sort(key=lambda row: row["record_id"])
    payload_projection.sort(key=lambda row: row["record_id"])
    if _require_int(inventory.get("record_count"), "DATA-526 record_count", minimum=1) != len(
        normalized
    ):
        raise ValueError("DATA-526 record count drift")
    claimed_record_digest = _require_hex64(
        inventory.get("record_inventory_digest_sha256"),
        "DATA-526 record_inventory_digest_sha256",
    )
    if claimed_record_digest != DATA526_RECORD_INVENTORY_SHA256:
        raise ValueError("DATA-526 record inventory authority drift")
    if _sha256(_canonical(normalized)) != claimed_record_digest:
        raise ValueError("DATA-526 record inventory digest does not reproduce")
    claimed_payload_digest = _require_hex64(
        inventory.get("payload_inventory_digest_sha256"),
        "DATA-526 payload_inventory_digest_sha256",
    )
    if claimed_payload_digest != DATA526_PAYLOAD_INVENTORY_SHA256:
        raise ValueError("DATA-526 payload inventory authority drift")
    if _sha256(_canonical(payload_projection)) != claimed_payload_digest:
        raise ValueError("DATA-526 payload inventory digest does not reproduce")
    return normalized


def _verify_upstream_report(
    report: Mapping[str, Any],
    *,
    expected_report_sha256: str,
) -> tuple[str, Mapping[str, Any], dict[str, Mapping[str, Any]]]:
    expected_report_sha256 = _require_hex64(expected_report_sha256, "expected_report_sha256")
    if report.get("schema_version") != UPSTREAM_REPORT_SCHEMA:
        raise ValueError("unexpected expanded-V9 report schema")
    if report.get("execution_profile") != "LOCAL_FREE":
        raise ValueError("expanded-V9 execution profile must remain LOCAL_FREE")
    report_sha = _require_hex64(report.get("report_sha256"), "report.report_sha256")
    if report_sha != expected_report_sha256 or report_sha != _self_hash(report, "report_sha256"):
        raise ValueError("expanded-V9 report identity mismatch")
    if report.get("raw_text_emitted") is not False:
        raise ValueError("expanded-V9 report emitted raw text")
    _require_exact_boundary(
        report.get("claim_boundary"),
        UPSTREAM_REPORT_BOUNDARY,
        "report.claim_boundary",
    )
    data526 = report.get("data526")
    if not isinstance(data526, Mapping):
        raise TypeError("report.data526 must be an object")
    if data526.get("record_inventory_digest_sha256") != DATA526_RECORD_INVENTORY_SHA256:
        raise ValueError("expanded-V9 DATA-526 record inventory binding drift")
    dedup = report.get("dedup_v3")
    if not isinstance(dedup, Mapping):
        raise TypeError("report.dedup_v3 must be an object")
    if dedup.get("schema_version") != UPSTREAM_MATCHER_SCHEMA:
        raise ValueError("unexpected expanded-V9 matcher schema")
    matcher_sha = _require_hex64(dedup.get("report_sha256"), "report.dedup_v3.report_sha256")
    if matcher_sha != _self_hash(
        dedup,
        "report_sha256",
        ascii_only=True,
        newline=True,
    ):
        raise ValueError("expanded-V9 matcher report self-hash mismatch")
    if dedup.get("local_free_only") is not True:
        raise ValueError("expanded-V9 matcher must remain LOCAL_FREE")
    if dedup.get("model_training_executed") is not False:
        raise ValueError("expanded-V9 matcher preclaims training")
    if dedup.get("raw_text_emitted") is not False:
        raise ValueError("expanded-V9 matcher emitted raw text")
    sources = dedup.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("expanded-V9 matcher sources missing")
    by_source: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(sources):
        if not isinstance(raw, Mapping):
            raise TypeError(f"matcher sources[{index}] must be an object")
        source_id = _require_nonempty_string(
            raw.get("source_id"), f"matcher sources[{index}].source_id"
        )
        if source_id in by_source:
            raise ValueError(f"duplicate matcher source_id: {source_id}")
        _require_nonempty_string(
            raw.get("source_family"), f"matcher sources[{index}].source_family"
        )
        _require_nonempty_string(raw.get("modality"), f"matcher sources[{index}].modality")
        _require_int(
            raw.get("declared_capacity_bytes"),
            f"matcher sources[{index}].declared_capacity_bytes",
            minimum=1,
        )
        _require_int(
            raw.get("verified_raw_bytes"),
            f"matcher sources[{index}].verified_raw_bytes",
            minimum=1,
        )
        _require_hex64(
            raw.get("verified_raw_sha256"),
            f"matcher sources[{index}].verified_raw_sha256",
        )
        by_source[source_id] = raw
    if _require_int(dedup.get("source_count"), "matcher source_count", minimum=1) != len(
        by_source
    ):
        raise ValueError("expanded-V9 matcher source_count drift")
    return report_sha, dedup, by_source


def _verify_survivor_authority(
    survivor_authority: Mapping[str, Any],
    *,
    expected_survivor_authority_sha256: str,
    matcher_report_sha256: str,
    matcher_sources: Mapping[str, Mapping[str, Any]],
) -> tuple[str, list[str], int]:
    expected = _require_hex64(
        expected_survivor_authority_sha256,
        "expected_survivor_authority_sha256",
    )
    if survivor_authority.get("schema_version") != UPSTREAM_SURVIVOR_SCHEMA:
        raise ValueError("unexpected expanded-V9 survivor authority schema")
    survivor_sha = _require_hex64(
        survivor_authority.get("survivor_authority_sha256"),
        "survivor_authority.survivor_authority_sha256",
    )
    if survivor_sha != expected or survivor_sha != _self_hash(
        survivor_authority,
        "survivor_authority_sha256",
    ):
        raise ValueError("expanded-V9 survivor authority identity mismatch")
    if survivor_authority.get("matcher_report_sha256") != matcher_report_sha256:
        raise ValueError("expanded-V9 survivor/matcher report binding mismatch")
    _require_exact_boundary(
        survivor_authority.get("truth_boundary"),
        UPSTREAM_SURVIVOR_BOUNDARY,
        "survivor_authority.truth_boundary",
    )
    raw_ids = survivor_authority.get("survivor_source_ids")
    if not isinstance(raw_ids, list) or not raw_ids:
        raise ValueError("survivor_authority.survivor_source_ids must be non-empty")
    survivor_ids = [_require_nonempty_string(value, "survivor_source_id") for value in raw_ids]
    if survivor_ids != sorted(survivor_ids) or len(set(survivor_ids)) != len(survivor_ids):
        raise ValueError("survivor source ids must be sorted and unique")
    if not set(survivor_ids) <= set(matcher_sources):
        raise ValueError("survivor authority references unknown matcher source")
    expected_count = _require_int(
        survivor_authority.get("post_dedup_survivor_source_object_count"),
        "survivor_authority.post_dedup_survivor_source_object_count",
        minimum=1,
    )
    if expected_count != len(survivor_ids):
        raise ValueError("survivor source count drift")
    retained_bytes = _require_int(
        survivor_authority.get("post_dedup_declared_capacity_bytes"),
        "survivor_authority.post_dedup_declared_capacity_bytes",
        minimum=1,
    )
    observed_bytes = sum(
        _require_int(
            matcher_sources[source_id].get("declared_capacity_bytes"),
            f"matcher source {source_id}.declared_capacity_bytes",
            minimum=1,
        )
        for source_id in survivor_ids
    )
    if retained_bytes != observed_bytes:
        raise ValueError("survivor source byte arithmetic mismatch")
    return survivor_sha, survivor_ids, retained_bytes


def freeze_expanded_inventory(
    report: Mapping[str, Any],
    survivor_authority: Mapping[str, Any],
    data526_record_inventory: Mapping[str, Any],
    *,
    expected_report_sha256: str,
    expected_survivor_authority_sha256: str,
) -> dict[str, Any]:
    """Freeze the exact real expanded-V9 source survivors into record-level authority."""

    report_sha, dedup, matcher_sources = _verify_upstream_report(
        report,
        expected_report_sha256=expected_report_sha256,
    )
    matcher_sha = _require_hex64(dedup.get("report_sha256"), "matcher report SHA")
    survivor_sha, survivor_ids, retained_source_bytes = _verify_survivor_authority(
        survivor_authority,
        expected_survivor_authority_sha256=expected_survivor_authority_sha256,
        matcher_report_sha256=matcher_sha,
        matcher_sources=matcher_sources,
    )
    data526_rows = _validate_data526_record_inventory(data526_record_inventory)

    survivor_id_set = set(survivor_ids)
    rada_survivor_ids = {
        source_id for source_id in survivor_ids if source_id.startswith(RADA_SOURCE_PREFIX)
    }
    base_survivor_ids = survivor_id_set - rada_survivor_ids

    rows_by_base_source: dict[str, list[dict[str, Any]]] = {}
    for row in data526_rows:
        rows_by_base_source.setdefault(row["source_id"], []).append(row)
    missing_base_sources = sorted(base_survivor_ids - set(rows_by_base_source))
    if missing_base_sources:
        raise ValueError(
            "surviving DATA-526 source missing from record inventory: "
            + ", ".join(missing_base_sources[:5])
        )

    records: list[dict[str, Any]] = []
    for source_id in sorted(base_survivor_ids):
        source = matcher_sources[source_id]
        source_rows = rows_by_base_source[source_id]
        source_payload_bytes = sum(row["payload_bytes"] for row in source_rows)
        declared = _require_int(
            source.get("declared_capacity_bytes"),
            f"matcher source {source_id}.declared_capacity_bytes",
            minimum=1,
        )
        if source_payload_bytes != declared:
            raise ValueError(f"DATA-526 record/source byte arithmetic mismatch: {source_id}")
        for row in source_rows:
            if row["family"] != source.get("source_family"):
                raise ValueError(f"DATA-526 record/source family mismatch: {source_id}")
            if row["modality"] != source.get("modality"):
                raise ValueError(f"DATA-526 record/source modality mismatch: {source_id}")
            records.append(
                {
                    **row,
                    "training_eligible": False,
                    "evaluation_eligible": False,
                }
            )

    for source_id in sorted(rada_survivor_ids):
        source = matcher_sources[source_id]
        record_id = source_id[len(RADA_SOURCE_PREFIX) :]
        if not record_id:
            raise ValueError("Rada survivor source id has empty record identity")
        raw_bytes = _require_int(
            source.get("verified_raw_bytes"),
            f"matcher source {source_id}.verified_raw_bytes",
            minimum=1,
        )
        declared = _require_int(
            source.get("declared_capacity_bytes"),
            f"matcher source {source_id}.declared_capacity_bytes",
            minimum=1,
        )
        if raw_bytes != declared:
            raise ValueError(f"Rada verified/declared byte mismatch: {source_id}")
        records.append(
            {
                "record_id": record_id,
                "source_id": source_id,
                "family": _require_nonempty_string(
                    source.get("source_family"),
                    f"matcher source {source_id}.source_family",
                ),
                "modality": _require_nonempty_string(
                    source.get("modality"),
                    f"matcher source {source_id}.modality",
                ),
                "payload_sha256": _require_hex64(
                    source.get("verified_raw_sha256"),
                    f"matcher source {source_id}.verified_raw_sha256",
                ),
                "payload_bytes": raw_bytes,
                "training_eligible": False,
                "evaluation_eligible": False,
            }
        )

    records.sort(key=lambda row: row["record_id"])
    if len({row["record_id"] for row in records}) != len(records):
        raise ValueError("duplicate record_id after expanded-V9 survivor projection")
    retained_record_bytes = sum(row["payload_bytes"] for row in records)
    if retained_record_bytes != retained_source_bytes:
        raise ValueError("record-level retained bytes do not reproduce survivor capacity")

    source_vector = report.get("source_vector")
    if not isinstance(source_vector, Mapping):
        raise TypeError("report.source_vector must be an object")
    if (
        _require_int(
            source_vector.get("post_dedup_conservative_unique_bytes"),
            "report.source_vector.post_dedup_conservative_unique_bytes",
            minimum=1,
        )
        != retained_source_bytes
    ):
        raise ValueError("report/survivor retained-byte binding mismatch")

    inventory: dict[str, Any] = {
        "schema": SCHEMA,
        "input_report_sha256": report_sha,
        "input_matcher_report_sha256": matcher_sha,
        "input_survivor_authority_sha256": survivor_sha,
        "input_data526_record_inventory_sha256": DATA526_RECORD_INVENTORY_SHA256,
        "source_count": len(survivor_ids),
        "record_count": len(records),
        "retained_payload_bytes": retained_record_bytes,
        "records": records,
        "truth_boundary": dict(ZERO_TRUTH),
    }
    inventory["inventory_identity_sha256"] = _self_hash(
        inventory,
        "inventory_identity_sha256",
        newline=True,
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
    if actual != expected or actual != _self_hash(
        inventory,
        "inventory_identity_sha256",
        newline=True,
    ):
        raise ValueError("retained inventory identity mismatch")
    if inventory.get("schema") != SCHEMA:
        raise ValueError("unexpected retained inventory schema")
    for key in (
        "input_report_sha256",
        "input_matcher_report_sha256",
        "input_survivor_authority_sha256",
        "input_data526_record_inventory_sha256",
    ):
        _require_hex64(inventory.get(key), f"inventory.{key}")
    if inventory.get("input_data526_record_inventory_sha256") != DATA526_RECORD_INVENTORY_SHA256:
        raise ValueError("inventory DATA-526 authority drift")
    _verify_zero_truth(inventory.get("truth_boundary"), "inventory.truth_boundary")

    raw_rows = inventory.get("records")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError("inventory.records must be non-empty")
    rows = [_validate_record(row, index) for index, row in enumerate(raw_rows)]
    if rows != sorted(rows, key=lambda row: row["record_id"]):
        raise ValueError("inventory records must be sorted by record_id")
    if len({row["record_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate record_id in retained inventory")
    if _require_int(inventory.get("record_count"), "inventory.record_count", minimum=1) != len(
        rows
    ):
        raise ValueError("inventory record_count mismatch")
    source_count = len({row["source_id"] for row in rows})
    if _require_int(inventory.get("source_count"), "inventory.source_count", minimum=1) != source_count:
        raise ValueError("inventory source_count mismatch")
    retained_payload_bytes = sum(row["payload_bytes"] for row in rows)
    if (
        _require_int(
            inventory.get("retained_payload_bytes"),
            "inventory.retained_payload_bytes",
            minimum=1,
        )
        != retained_payload_bytes
    ):
        raise ValueError("inventory retained_payload_bytes mismatch")


def prepare_ephemeral_data232_rows(
    inventory: Mapping[str, Any],
    payload_rows: Iterable[Mapping[str, Any]],
    *,
    expected_inventory_identity_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Bind ephemeral training text to the frozen inventory for current DATA-232."""

    verify_inventory(
        inventory,
        expected_inventory_identity_sha256=expected_inventory_identity_sha256,
    )
    by_record = {row["record_id"]: row for row in inventory["records"]}
    seen: set[str] = set()
    ephemeral: list[dict[str, Any]] = []
    projection: list[dict[str, Any]] = []

    for index, raw in enumerate(payload_rows):
        if not isinstance(raw, Mapping):
            raise TypeError(f"payload_rows[{index}] must be an object")
        record_id = _require_nonempty_string(
            raw.get("record_id"),
            f"payload_rows[{index}].record_id",
        )
        if record_id in seen:
            raise ValueError(f"duplicate payload row {record_id}")
        seen.add(record_id)
        expected = by_record.get(record_id)
        if expected is None:
            raise ValueError(f"unexpected payload row {record_id}")
        text = raw.get("text")
        if not isinstance(text, str) or not text:
            raise TypeError(f"payload row {record_id} must contain non-empty text")
        text_bytes = text.encode("utf-8")
        if len(text_bytes) != expected["payload_bytes"]:
            raise ValueError(f"payload identity mismatch for {record_id}")
        if _sha256(text_bytes) != expected["payload_sha256"]:
            raise ValueError(f"payload identity mismatch for {record_id}")
        item = {
            "record_id": record_id,
            "source_id": expected["source_id"],
            "source_family": expected["family"],
            "modality": expected["modality"],
            "text": text,
        }
        ephemeral.append(item)
        projection.append(
            {
                "record_id": record_id,
                "source_id": expected["source_id"],
                "source_family": expected["family"],
                "modality": expected["modality"],
                "text_sha256": expected["payload_sha256"],
                "text_utf8_bytes": expected["payload_bytes"],
            }
        )

    missing = sorted(set(by_record) - seen)
    if missing:
        raise ValueError(f"missing payload rows: {', '.join(missing[:5])}")
    ephemeral.sort(key=lambda row: row["record_id"])
    projection.sort(key=lambda row: row["record_id"])

    handoff: dict[str, Any] = {
        "schema_version": HANDOFF_SCHEMA,
        "postdedup_inventory_identity_sha256": expected_inventory_identity_sha256,
        "input_survivor_authority_sha256": inventory["input_survivor_authority_sha256"],
        "retained_source_count": len(projection),
        "matcher_input_projection": projection,
        "matcher_input_projection_sha256": _sha256(_canonical(projection)),
        "raw_text_persisted_in_evidence": False,
        "final_test_payload_accessed": False,
        "final_test_outcomes_accessed": False,
        "authorized_training_exposure": 0,
    }
    handoff["handoff_identity_sha256"] = _self_hash(
        handoff,
        "handoff_identity_sha256",
    )
    return ephemeral, handoff


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical(value, newline=True))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Freeze exact expanded-V9 source survivors into record-level authority"
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--survivor-authority", type=Path, required=True)
    parser.add_argument("--data526-record-inventory", type=Path, required=True)
    parser.add_argument("--expected-report-sha256", required=True)
    parser.add_argument("--expected-survivor-authority-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    inventory = freeze_expanded_inventory(
        _load_json(args.report),
        _load_json(args.survivor_authority),
        _load_json(args.data526_record_inventory),
        expected_report_sha256=args.expected_report_sha256,
        expected_survivor_authority_sha256=args.expected_survivor_authority_sha256,
    )
    _write_json(args.output, inventory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
