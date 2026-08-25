from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from twelve_six.data.ukrainian_normalization import normalize_document, summarize_changes


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_jsonl(
    input_path: Path,
    *,
    source_id: str,
    source_version: str,
    language_hint: str | None = "uk",
) -> dict[str, Any]:
    source_sha256 = _sha256(input_path)
    results = []
    traces = []
    for line_number, line in enumerate(input_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        if language_hint is not None and record.get("language_hint") != language_hint:
            continue
        document_id = record.get("document_id")
        if not isinstance(document_id, str) or not document_id:
            raise ValueError(f"line {line_number}: document_id is required")
        text = record.get("text")
        if not isinstance(text, str):
            raise ValueError(f"line {line_number}: text must be a string")
        modality = record.get("modality", "natural")
        result = normalize_document(
            text,
            modality=modality,
            source_id=source_id,
            source_version=source_version,
            raw_document_id=document_id,
            raw_source_sha256=source_sha256,
        )
        results.append(result)
        traces.append(result.trace.as_dict())

    summary = summarize_changes(tuple(results)).as_dict()
    return {
        "schema": "12-6.ua-normalization-audit-v1",
        "input_path": input_path.as_posix(),
        "raw_source_sha256": source_sha256,
        "source_id": source_id,
        "source_version": source_version,
        "language_hint_filter": language_hint,
        "summary": summary,
        "documents": traces,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit DATA-27 normalization over a JSONL corpus without mutating source bytes."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--source-version", required=True)
    parser.add_argument(
        "--language-hint",
        default="uk",
        help="Filter by record language_hint; pass '*' to audit every record.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    language_hint = None if args.language_hint == "*" else args.language_hint
    payload = audit_jsonl(
        args.input,
        source_id=args.source_id,
        source_version=args.source_version,
        language_hint=language_hint,
    )
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
