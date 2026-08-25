#!/usr/bin/env python3
"""Scan JSONL corpus records for high-confidence PII/secrets without logging matches."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from twelve_six.data.privacy_filter import (
    assert_no_secret_values_in_manifest,
    build_scan_manifest,
    scan_record,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--source-registry", type=Path, required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--source-version", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--sanitized-output", type=Path)
    parser.add_argument("--default-modality", choices=("natural", "code"), default="natural")
    return parser.parse_args()


def main() -> int:
    args = _args()
    results = []
    sanitized_rows: list[dict[str, Any]] = []
    with args.input.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            record_id = str(row.get("document_id") or row.get("record_id") or f"line-{line_number}")
            text = row.get("text")
            modality = row.get("modality", args.default_modality)
            result = scan_record(
                record_id=record_id,
                source_id=args.source_id,
                source_version=args.source_version,
                modality=modality,
                text=text,
            )
            results.append(result)
            if result.train_eligible_after_privacy:
                output_row = dict(row)
                output_row["text"] = result.sanitized_text
                output_row["privacy_action"] = result.action
                output_row["privacy_evidence_sha256"] = result.evidence_sha256()
                sanitized_rows.append(output_row)

    manifest = build_scan_manifest(
        results,
        input_content_sha256=_sha256_file(args.input),
        source_registry_sha256=_sha256_file(args.source_registry),
    )
    assert_no_secret_values_in_manifest(manifest)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if args.sanitized_output is not None:
        args.sanitized_output.parent.mkdir(parents=True, exist_ok=True)
        with args.sanitized_output.open("w", encoding="utf-8", newline="\n") as handle:
            for row in sanitized_rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
