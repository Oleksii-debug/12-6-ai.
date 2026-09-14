"""Prepare ephemeral DATA-232 matcher inputs from frozen DATA-526 and EVAL-303 authorities.

Raw text is written only to caller-selected ephemeral paths. Durable DATA-232 evidence
remains hash-only. Final-test payloads are neither accepted nor required by this tool.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

TRAINING_PAYLOAD_JSONL_SHA256 = "2f21f7655c9287bbc7424b410788f431f20f0936f445fefd26fcf6114b772dbd"
TRAINING_INVENTORY_DIGEST_SHA256 = "55f01d2027057f30cfc43d4ebe668779fc48413781d8146402dd53f81e2dbf31"
TRAINING_RECORDS = 48
TRAINING_PAYLOAD_BYTES = 2_215_615
EVAL290_ZIP_SHA256 = "0c5f9f8d938284a1358bfb77284814b1f0569b2d6d13938f415ba31db64a6c3b"
EVAL291_ZIP_SHA256 = "3168a1c7884b10ba7a959c859c1539f2ad4957a852ad3f9aa3517df3c5110f94"
EVAL303_MEMBERSHIP_SHA256 = "e4bb39dd7aa6a20c7ed34e093f563b5f4896ac16828151c6b375a83cd8a068c6"
EVAL303_RECORDS = 10
EVAL290_MEMBER = "tmp/eval290-a/eval290_ua_selection_validation_v1.jsonl"
EVAL291_MEMBER = "data/evaluation/eval291/selection-validation/en.jsonl"


class MatcherInputError(RuntimeError):
    pass


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_jsonl_bytes(payload: bytes, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(payload.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise MatcherInputError(f"{label}:{number}: JSONL row must be an object")
        rows.append(value)
    return rows


def _canonical_jsonl(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(
        json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8") + b"\n"
        for row in rows
    )


def _verify_training(records_path: Path, inventory_path: Path) -> list[dict[str, Any]]:
    raw = records_path.read_bytes()
    if _sha256(raw) != TRAINING_PAYLOAD_JSONL_SHA256:
        raise MatcherInputError("DATA-526 record payload JSONL identity drift")
    source_rows = _read_jsonl_bytes(raw, str(records_path))
    if len(source_rows) != TRAINING_RECORDS:
        raise MatcherInputError("DATA-526 record-count drift")
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    if inventory.get("record_inventory_digest_sha256") != TRAINING_INVENTORY_DIGEST_SHA256:
        raise MatcherInputError("DATA-526 record inventory identity drift")
    if inventory.get("record_count") != TRAINING_RECORDS:
        raise MatcherInputError("DATA-526 inventory record-count drift")
    if inventory.get("total_payload_bytes") != TRAINING_PAYLOAD_BYTES:
        raise MatcherInputError("DATA-526 inventory payload-byte drift")

    rows: list[dict[str, Any]] = []
    payload_bytes = 0
    for source in source_rows:
        required = ("record_id", "source_id", "family", "modality", "normalized_payload")
        if any(not isinstance(source.get(key), str) for key in required):
            raise MatcherInputError("DATA-526 materialized record schema drift")
        text = source["normalized_payload"]
        payload_bytes += len(text.encode("utf-8"))
        rows.append(
            {
                "record_id": source["record_id"],
                "source_id": source["source_id"],
                "source_family": source["family"],
                "modality": source["modality"],
                "text": text,
            }
        )
    if payload_bytes != TRAINING_PAYLOAD_BYTES:
        raise MatcherInputError("DATA-526 payload-byte reconstruction drift")
    if len({row["record_id"] for row in rows}) != len(rows):
        raise MatcherInputError("DATA-526 duplicate record_id")
    return sorted(rows, key=lambda row: row["record_id"])


def _load_component_zip(path: Path, expected_sha256: str, member: str, label: str) -> list[dict[str, Any]]:
    raw = path.read_bytes()
    if _sha256(raw) != expected_sha256:
        raise MatcherInputError(f"{label} artifact ZIP identity drift")
    with zipfile.ZipFile(path) as package:
        try:
            payload = package.read(member)
        except KeyError as exc:
            raise MatcherInputError(f"{label} payload member missing: {member}") from exc
    return _read_jsonl_bytes(payload, f"{label}:{member}")


def _component_record_id(row: dict[str, Any]) -> str:
    value = row.get("record_id", row.get("document_id"))
    if not isinstance(value, str) or not value:
        raise MatcherInputError("selection component record id missing")
    return value


def _verify_evaluation(eval290_zip: Path, eval291_zip: Path, membership_path: Path) -> list[dict[str, Any]]:
    membership_raw = membership_path.read_bytes()
    if _sha256(membership_raw) != EVAL303_MEMBERSHIP_SHA256:
        raise MatcherInputError("EVAL-303 membership identity drift")
    membership = _read_jsonl_bytes(membership_raw, str(membership_path))
    if len(membership) != EVAL303_RECORDS:
        raise MatcherInputError("EVAL-303 membership record-count drift")

    component_rows = _load_component_zip(eval290_zip, EVAL290_ZIP_SHA256, EVAL290_MEMBER, "EVAL-290")
    component_rows += _load_component_zip(eval291_zip, EVAL291_ZIP_SHA256, EVAL291_MEMBER, "EVAL-291")
    by_id = {_component_record_id(row): row for row in component_rows}
    if len(by_id) != EVAL303_RECORDS or set(by_id) != {row.get("record_id") for row in membership}:
        raise MatcherInputError("EVAL-303/component membership mismatch")

    output: list[dict[str, Any]] = []
    for authority in membership:
        record_id = authority.get("record_id")
        if not isinstance(record_id, str):
            raise MatcherInputError("EVAL-303 membership record_id missing")
        if authority.get("training_eligible") is not False or authority.get("tokenizer_fit_eligible") is not False:
            raise MatcherInputError("EVAL-303 reservation boundary weakened")
        if authority.get("final_test_eligible") is not False or authority.get("selection_eligible") is not True:
            raise MatcherInputError("EVAL-303 selection/final-test boundary weakened")
        payload_row = by_id[record_id]
        text = payload_row.get("text")
        if not isinstance(text, str):
            raise MatcherInputError(f"selection payload missing text: {record_id}")
        text_bytes = text.encode("utf-8")
        if _sha256(text_bytes) != authority.get("content_sha256"):
            raise MatcherInputError(f"selection content identity drift: {record_id}")
        if len(text_bytes) != authority.get("utf8_bytes"):
            raise MatcherInputError(f"selection byte-count drift: {record_id}")
        output.append(
            {
                "record_id": record_id,
                "source_id": authority["source_id"],
                "source_family": authority["source_family"],
                "modality": authority["modality"],
                "text": text,
            }
        )
    return sorted(output, key=lambda row: row["record_id"])


def _write(path: Path, rows: list[dict[str, Any]]) -> str:
    payload = _canonical_jsonl(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return _sha256(payload)


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare exact ephemeral matcher inputs for DATA-232")
    parser.add_argument("--training-jsonl", type=Path, required=True)
    parser.add_argument("--training-inventory", type=Path, required=True)
    parser.add_argument("--eval290-zip", type=Path, required=True)
    parser.add_argument("--eval291-zip", type=Path, required=True)
    parser.add_argument("--eval303-membership", type=Path, required=True)
    parser.add_argument("--training-out", type=Path, required=True)
    parser.add_argument("--evaluation-out", type=Path, required=True)
    args = parser.parse_args()

    training = _verify_training(args.training_jsonl, args.training_inventory)
    evaluation = _verify_evaluation(args.eval290_zip, args.eval291_zip, args.eval303_membership)
    training_sha = _write(args.training_out, training)
    evaluation_sha = _write(args.evaluation_out, evaluation)
    print(
        json.dumps(
            {
                "status": "PASS",
                "training_records": len(training),
                "evaluation_records": len(evaluation),
                "training_matcher_jsonl_sha256": training_sha,
                "evaluation_matcher_jsonl_sha256": evaluation_sha,
                "final_test_payload_accessed": False,
                "final_test_outcomes_accessed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
