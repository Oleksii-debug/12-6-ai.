from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

REQUIRED_METADATA = ("source_id", "family", "modality")


def canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _require_text(record: dict[str, Any], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"record requires non-empty {key}")
    return value


def materialize(records: list[dict[str, Any]]) -> dict[str, Any]:
    inventory: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for record in records:
        record_id = _require_text(record, "record_id")
        if record_id in seen_ids:
            raise ValueError(f"duplicate record_id: {record_id}")
        seen_ids.add(record_id)
        payload = _require_text(record, "normalized_payload")
        payload_bytes = payload.encode("utf-8")
        item = {
            "record_id": record_id,
            **{key: _require_text(record, key) for key in REQUIRED_METADATA},
            "payload_sha256": hashlib.sha256(payload_bytes).hexdigest(),
            "payload_bytes": len(payload_bytes),
        }
        inventory.append(item)

    inventory.sort(key=lambda item: item["record_id"])
    inventory_bytes = canonical_json(inventory)
    payload_projection = [
        {"record_id": item["record_id"], "payload_sha256": item["payload_sha256"], "payload_bytes": item["payload_bytes"]}
        for item in inventory
    ]
    return {
        "schema_version": "12-6.data526-record-inventory.v1",
        "record_count": len(inventory),
        "total_payload_bytes": sum(item["payload_bytes"] for item in inventory),
        "record_inventory_digest_sha256": hashlib.sha256(inventory_bytes).hexdigest(),
        "payload_inventory_digest_sha256": hashlib.sha256(canonical_json(payload_projection)).hexdigest(),
        "records": inventory,
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise TypeError(f"line {line_number} is not an object")
        records.append(value)
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize a deterministic text-free DATA-526 payload-bound record inventory")
    parser.add_argument("input_jsonl", type=Path)
    parser.add_argument("output_json", type=Path)
    args = parser.parse_args()
    result = materialize(load_jsonl(args.input_jsonl))
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_bytes(canonical_json(result) + b"\n")
    print(result["record_inventory_digest_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
