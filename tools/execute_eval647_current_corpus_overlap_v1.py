#!/usr/bin/env python3
"""Execute the EVAL-647 current-corpus overlap gate without persisting source text."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

from twelve_six.data.eval647_current_corpus_overlap_v1 import build_report, verify_report

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "configs/evaluation/eval_code_reserve_v1.json"
MATERIALIZER = ROOT / "tools/materialize_eval_code_reserve_v1.py"
POST_G05_G06_RECORD_KEYS = frozenset(
    {"record_id", "source_id", "family", "modality", "normalized_payload"}
)

materializer_spec = importlib.util.spec_from_file_location("eval647_materializer_for_overlap", MATERIALIZER)
if materializer_spec is None or materializer_spec.loader is None:
    raise RuntimeError("cannot load canonical EVAL-647 source materializer")
source_materializer = importlib.util.module_from_spec(materializer_spec)
materializer_spec.loader.exec_module(source_materializer)


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object")
    return value


def _project_post_g05_g06_record(
    value: dict[str, Any], line_number: int
) -> dict[str, str]:
    if set(value) != POST_G05_G06_RECORD_KEYS:
        raise RuntimeError(
            f"training JSONL line {line_number} post-G05/G06 schema drift"
        )
    projected: dict[str, str] = {}
    for source_key, matcher_key in (
        ("record_id", "record_id"),
        ("source_id", "source_id"),
        ("family", "source_family"),
        ("modality", "modality"),
        ("normalized_payload", "text"),
    ):
        field = value.get(source_key)
        if not isinstance(field, str) or not field:
            raise RuntimeError(
                f"training JSONL line {line_number} {source_key} must be non-empty string"
            )
        projected[matcher_key] = field
    return projected


def _read_training_jsonl(path: Path) -> tuple[bytes, list[dict[str, str]]]:
    raw = path.read_bytes()
    text = raw.decode("utf-8", errors="strict")
    records: list[dict[str, str]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise RuntimeError(f"training JSONL line {line_number} must be an object")
        records.append(_project_post_g05_g06_record(value, line_number))
    if not records:
        raise RuntimeError("training JSONL must contain at least one record")
    return raw, records


def _fetch_reserved_payloads(manifest: dict[str, Any], timeout: int) -> list[bytes]:
    objects = manifest.get("objects")
    if not isinstance(objects, list) or len(objects) != 2:
        raise RuntimeError("EVAL-647 manifest must contain exactly two objects")
    return [
        source_materializer._fetch(source_materializer._raw_url(row), timeout)
        for row in objects
    ]


def execute(
    *,
    training_records_jsonl: Path,
    materialization_evidence_json: Path,
    manifest_json: Path,
    output_json: Path,
    expected_materialization_identity_sha256: str,
    expected_training_jsonl_sha256: str,
    timeout: int = 30,
) -> dict[str, Any]:
    """Run the gate from independently pinned corpus identities and write hash-only evidence."""
    raw_training, records = _read_training_jsonl(training_records_jsonl)
    evidence = _read_json_object(materialization_evidence_json, "materialization evidence")
    manifest = _read_json_object(manifest_json, "EVAL-647 manifest")
    reserved_payloads = _fetch_reserved_payloads(manifest, timeout)
    report = build_report(
        records,
        manifest,
        reserved_payloads,
        evidence,
        actual_training_jsonl_sha256=hashlib.sha256(raw_training).hexdigest(),
        expected_materialization_identity_sha256=expected_materialization_identity_sha256,
        expected_training_jsonl_sha256=expected_training_jsonl_sha256,
    )
    verify_report(report)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-records-jsonl", type=Path, required=True)
    parser.add_argument("--materialization-evidence-json", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-materialization-identity-sha256", required=True)
    parser.add_argument("--expected-training-jsonl-sha256", required=True)
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args()

    report = execute(
        training_records_jsonl=args.training_records_jsonl,
        materialization_evidence_json=args.materialization_evidence_json,
        manifest_json=args.manifest,
        output_json=args.output,
        expected_materialization_identity_sha256=args.expected_materialization_identity_sha256,
        expected_training_jsonl_sha256=args.expected_training_jsonl_sha256,
        timeout=args.timeout,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "current_corpus_overlap_zero": report["current_corpus_overlap_zero"],
                "match_evidence_count": report["match_evidence_count"],
                "selection_validation_records_authorized": report[
                    "selection_validation_records_authorized"
                ],
                "report_sha256": report["report_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0 if report["current_corpus_overlap_zero"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
