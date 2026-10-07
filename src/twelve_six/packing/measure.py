"""CLI for binding a D03 packaged JSONL split to D04 tokenizer/packing identity."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable, Iterator
from pathlib import Path

from twelve_six.tokenization import ByteTokenizer

from .core import TextRecord
from .jsonl import load_jsonl_records
from .manifest import PackedSplitManifest, measure_packed_split


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assignment_ids_for_split(payload: dict[str, object], *, split: str) -> tuple[str, ...]:
    assignments = payload.get("document_assignments")
    if not isinstance(assignments, list):
        raise TypeError("dataset manifest document_assignments must be a list")

    selected: list[str] = []
    seen_ids: set[str] = set()
    for index, assignment in enumerate(assignments):
        if not isinstance(assignment, dict):
            raise TypeError(f"dataset manifest assignment {index} must be a mapping")
        record_id = assignment.get("id")
        assigned_split = assignment.get("split")
        if not isinstance(record_id, str):
            raise TypeError(f"dataset manifest assignment {index} id must be a string")
        if not record_id:
            raise ValueError(f"dataset manifest assignment {index} id must be non-empty")
        if not isinstance(assigned_split, str):
            raise TypeError(f"dataset manifest assignment {index} split must be a string")
        if not assigned_split:
            raise ValueError(f"dataset manifest assignment {index} split must be non-empty")
        if record_id in seen_ids:
            raise ValueError(f"dataset manifest contains duplicate assignment id {record_id!r}")
        seen_ids.add(record_id)
        if assigned_split == split:
            selected.append(record_id)
    return tuple(selected)


def _require_assignment_order(
    records: Iterable[TextRecord],
    expected_record_ids: tuple[str, ...],
) -> Iterator[TextRecord]:
    consumed = 0
    for consumed, record in enumerate(records, start=1):
        index = consumed - 1
        if index >= len(expected_record_ids):
            raise ValueError(
                f"packaged split contains unexpected record {record.record_id!r} at index {index}"
            )
        expected_id = expected_record_ids[index]
        if record.record_id != expected_id:
            raise ValueError(
                "packaged split record assignment mismatch: "
                f"index {index} has {record.record_id!r}, expected {expected_id!r}"
            )
        yield record
    if consumed != len(expected_record_ids):
        raise ValueError(
            "packaged split record count does not match manifest assignments: "
            f"consumed {consumed}, expected {len(expected_record_ids)}"
        )


def measure_d03_packaged_split(
    dataset_manifest_path: str | Path,
    jsonl_path: str | Path,
    *,
    split: str,
) -> PackedSplitManifest:
    """Verify D03 split identity/source bytes, then measure the frozen S0 byte path."""
    manifest_path = Path(dataset_manifest_path)
    split_path = Path(jsonl_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("dataset manifest must contain a JSON object")

    dataset_id = payload.get("dataset_id")
    dataset_identity = payload.get("dataset_identity_sha256")
    outputs = payload.get("outputs")
    if not isinstance(dataset_id, str):
        raise TypeError("dataset manifest dataset_id must be a string")
    if not dataset_id:
        raise ValueError("dataset manifest dataset_id must be non-empty")
    if not isinstance(dataset_identity, str):
        raise TypeError("dataset manifest dataset_identity_sha256 must be a string")
    if not isinstance(outputs, dict):
        raise TypeError("dataset manifest outputs must be a mapping")
    if not split:
        raise ValueError("split must be non-empty")

    expected_output_name = f"{split}.jsonl"
    if split_path.name != expected_output_name:
        raise ValueError(
            "split/output mismatch: "
            f"requested split {split!r} requires {expected_output_name!r}, "
            f"got {split_path.name!r}"
        )

    expected_source_hash = outputs.get(expected_output_name)
    if expected_source_hash is None:
        raise ValueError(f"dataset manifest does not bind output {expected_output_name!r}")
    if not isinstance(expected_source_hash, str):
        raise TypeError("dataset manifest output SHA-256 must be a string")
    actual_source_hash = _sha256_file(split_path)
    if actual_source_hash != expected_source_hash:
        raise ValueError(
            f"source JSONL hash mismatch: {actual_source_hash} != {expected_source_hash}"
        )

    expected_record_ids = _assignment_ids_for_split(payload, split=split)
    records = _require_assignment_order(
        load_jsonl_records(split_path, split=split),
        expected_record_ids,
    )
    return measure_packed_split(
        records,
        ByteTokenizer(),
        dataset_id=dataset_id,
        dataset_identity_sha256=dataset_identity,
        source_jsonl_sha256=actual_source_hash,
        split=split,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--jsonl", required=True, type=Path)
    parser.add_argument("--split", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = measure_d03_packaged_split(
        args.dataset_manifest,
        args.jsonl,
        split=args.split,
    )
    output = manifest.to_dict()
    output["manifest_sha256"] = manifest.manifest_sha256
    print(json.dumps(output, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
