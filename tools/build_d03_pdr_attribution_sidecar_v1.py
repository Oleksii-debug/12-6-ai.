#!/usr/bin/env python3
"""Build a fail-closed attribution sidecar for the exact executed PDR candidate."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

SCHEMA = "12-6.d03-pdr-attribution-sidecar.v1"
REPORT_SCHEMA = "12-6.d03-pdr-attribution-report.v1"
SOURCE_DATASET = "common-pile/public_domain_review"
SOURCE_REVISION = "177f70f044bd7a347616727c0f33ab4af9baa495"
SOURCE_VALUE = "public-domain-review"
SOURCE_FAMILY = "en.common-pile.public-domain-review"
SOURCE_HOST = "publicdomainreview.org"
SOURCE_KEY = "public_domain_review"
EXPECTED_LICENSE = (
    "Creative Commons - Attribution Share-Alike - "
    "https://creativecommons.org/licenses/by-sa/4.0/"
)
LICENSE_URL = "https://creativecommons.org/licenses/by-sa/4.0/"
REUSE_TERMS_URL = "https://publicdomainreview.org/reusing-material"
PUBLISHER = "The Public Domain Review"
EXPECTED_FILES = {
    "v0/00000_collections.jsonl.gz": {
        "sha256": "0a5cbe305d7f468aad95a8cfc4d5fae14132bc359ccf92fc3705072db33a46fd",
        "type": "collection",
    },
    "v0/00000_essays.jsonl.gz": {
        "sha256": "825aed6a66ab14f89e1fe3110f67265d511994c52e5aed48b866a9e8a1bdeb4d",
        "type": "essay",
    },
}
EXPECTED_EXECUTION_RUN_ID = 34478508878
EXPECTED_CANDIDATE_JSONL_BYTES = 5_630_935
EXPECTED_CANDIDATE_JSONL_SHA256 = (
    "38b0ef94062b80ae1cb8055fe897feb10a1e8d04a9b17340cbc080d4bfb21591"
)
EXPECTED_SELECTED_RECORD_COUNT = 1_166
EXPECTED_SELECTED_NORMALIZED_UTF8_BYTES = 4_795_007
EXPECTED_CANDIDATE_PROJECTION_SHA256 = (
    "bf9f7896fc2c940458916184b1b702f443d72290e286ee16b24da4088b2aa4ab"
)
EXPECTED_HISTORICAL_REPORT_ID = (
    "a295de32115a3378beecf8f5f5066fa0f35ea136631473df3ad5bd082c12a88c"
)
EXPECTED_HISTORICAL_MANIFEST_ID = (
    "49000a9dd6dbb56dd222992b5802b21f8d33d6393024f1c8d03e035372c48f7b"
)
CANDIDATE_KEYS = frozenset(
    {
        "artifact_role",
        "record_id",
        "source_id",
        "source_dataset",
        "source_revision",
        "origin_url",
        "license",
        "attribution_required",
        "language",
        "normalized_sha256",
        "normalized_utf8_bytes",
        "text",
        "training_eligible",
        "evaluation_eligible",
    }
)
# The immutable HF raw snapshot normalizes scraper fields into top-level date,
# author and type columns; metadata retains only the source URL and license.
RAW_KEYS = frozenset(
    {"id", "text", "source", "date", "author", "type", "added", "metadata"}
)
RAW_METADATA_KEYS = frozenset({"license", "url"})


class AttributionError(RuntimeError):
    """Fail-closed PDR attribution error."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AttributionError(message)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(ch in "0123456789abcdef" for ch in value)
    )


def _validate_pdr_url(value: Any) -> str:
    _require(isinstance(value, str) and value, "PDR URL missing")
    parsed = urlparse(value)
    host = (parsed.hostname or "").casefold()
    _require(parsed.scheme == "https", "PDR URL must use HTTPS")
    _require(
        host == SOURCE_HOST or host.endswith("." + SOURCE_HOST),
        "PDR URL host drift",
    )
    return value


def _validate_author(value: Any) -> str:
    _require(isinstance(value, str), "PDR author must be a string")
    author = value.strip()
    _require(bool(author), "PDR author missing")
    _require(len(author) <= 512, "PDR author exceeds safety bound")
    _require(
        not any(ord(ch) < 32 for ch in author),
        "PDR author contains control character",
    )
    return author


def _validate_config(config: dict[str, Any]) -> None:
    rights = config.get("common_pile_rights_authority")
    _require(isinstance(rights, dict), "rights authority missing")
    _require(rights.get("source_key") == SOURCE_KEY, "source-rights key drift")
    _require(
        rights.get("project_review_status") == "REVIEW_REQUIRED",
        "rights review status must remain REVIEW_REQUIRED",
    )
    source = config.get("source")
    _require(isinstance(source, dict), "source config missing")
    _require(source.get("dataset") == SOURCE_DATASET, "dataset identity drift")
    _require(source.get("revision") == SOURCE_REVISION, "source revision drift")
    _require(source.get("family_id") == SOURCE_FAMILY, "source family drift")
    _require(source.get("source_value") == SOURCE_VALUE, "source value drift")
    _require(source.get("origin_host") == SOURCE_HOST, "source host drift")
    _require(
        source.get("expected_record_license") == EXPECTED_LICENSE,
        "source license drift",
    )
    _require(source.get("attribution_required") is True, "attribution policy drift")

    files = config.get("files")
    _require(isinstance(files, list), "source file vector missing")
    observed = {entry.get("path"): entry for entry in files if isinstance(entry, dict)}
    _require(set(observed) == set(EXPECTED_FILES), "source file vector drift")
    for path, expected in EXPECTED_FILES.items():
        _require(
            observed[path].get("sha256") == expected["sha256"],
            f"source file hash drift: {path}",
        )

    boundary = config.get("truth_boundary")
    _require(isinstance(boundary, dict), "truth boundary missing")
    _require(boundary.get("canonical_corpus_admitted") is False, "corpus admission drift")
    _require(boundary.get("training_authorized_bytes") == 0, "training credit drift")
    _require(
        boundary.get("authorized_optimized_target_exposure") == 0,
        "optimized-target exposure drift",
    )
    _require(boundary.get("tokenizer_fit_authorized") is False, "tokenizer gate drift")
    _require(boundary.get("model_training_executed") is False, "training gate drift")
    _require(boundary.get("final_test_payload_accessed") is False, "final-test drift")
    _require(boundary.get("paid_compute_used") is False, "paid-compute drift")


def _raw_key(record_id: str, origin_url: str) -> tuple[str, str]:
    return record_id, origin_url


def _validate_raw_row(
    row: dict[str, Any],
    *,
    expected_type: str,
) -> tuple[tuple[str, str], str]:
    _require(set(row) == RAW_KEYS, "raw PDR row schema drift")
    record_id = row.get("id")
    _require(isinstance(record_id, str) and record_id.strip(), "raw PDR id missing")
    _require(row.get("source") == SOURCE_VALUE, "raw PDR source drift")
    _require(isinstance(row.get("text"), str), "raw PDR text missing")
    _require(isinstance(row.get("date"), str), "raw PDR date must be a string")
    _require(isinstance(row.get("added"), str), "raw PDR added must be a string")
    _require(row.get("type") == expected_type, "raw PDR type drift")
    author = _validate_author(row.get("author"))
    metadata = row.get("metadata")
    _require(isinstance(metadata, dict), "raw PDR metadata missing")
    _require(set(metadata) == RAW_METADATA_KEYS, "raw PDR metadata schema drift")
    _require(metadata.get("license") == EXPECTED_LICENSE, "raw PDR license drift")
    origin_url = _validate_pdr_url(metadata.get("url"))
    return _raw_key(record_id.strip(), origin_url), author


def _validate_candidate(row: dict[str, Any]) -> tuple[str, str]:
    _require(set(row) == CANDIDATE_KEYS, "PDR candidate schema drift")
    _require(row.get("artifact_role") == "SOURCE_CANDIDATE_ONLY", "artifact role drift")
    _require(row.get("source_id") == SOURCE_FAMILY, "candidate source family drift")
    _require(row.get("source_dataset") == SOURCE_DATASET, "candidate dataset drift")
    _require(row.get("source_revision") == SOURCE_REVISION, "candidate revision drift")
    _require(row.get("license") == EXPECTED_LICENSE, "candidate license drift")
    _require(row.get("attribution_required") is True, "candidate attribution drift")
    _require(row.get("language") == "en", "candidate language drift")
    _require(row.get("training_eligible") is False, "candidate training gate drift")
    _require(row.get("evaluation_eligible") is False, "candidate evaluation gate drift")

    record_id = row.get("record_id")
    _require(isinstance(record_id, str) and record_id.strip(), "candidate record id missing")
    origin_url = _validate_pdr_url(row.get("origin_url"))
    text = row.get("text")
    _require(isinstance(text, str), "candidate text missing")
    encoded = text.encode("utf-8")
    expected_sha = row.get("normalized_sha256")
    expected_bytes = row.get("normalized_utf8_bytes")
    _require(_valid_sha256(expected_sha), "candidate text SHA invalid")
    _require(_sha256(encoded) == expected_sha, "candidate text SHA mismatch")
    _require(
        isinstance(expected_bytes, int)
        and not isinstance(expected_bytes, bool)
        and expected_bytes > 0,
        "candidate byte count invalid",
    )
    _require(len(encoded) == expected_bytes, "candidate byte count mismatch")
    return record_id.strip(), origin_url


def _candidate_projection(candidate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "record_id": row["record_id"],
            "normalized_sha256": row["normalized_sha256"],
            "normalized_utf8_bytes": row["normalized_utf8_bytes"],
            "origin_url_sha256": _sha256(row["origin_url"].encode("utf-8")),
        }
        for row in candidate_rows
    ]


def build_sidecar(
    candidate_rows: list[dict[str, Any]],
    raw_rows_by_type: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _require(bool(candidate_rows), "candidate rows must be nonempty")
    raw_index: dict[tuple[str, str], str] = {}
    for source_type in ("collection", "essay"):
        rows = raw_rows_by_type.get(source_type)
        _require(isinstance(rows, list), f"raw {source_type} rows missing")
        for row in rows:
            _require(isinstance(row, dict), "raw PDR row must be an object")
            key, author = _validate_raw_row(row, expected_type=source_type)
            _require(key not in raw_index, "duplicate raw PDR record identity")
            raw_index[key] = author

    sidecar: list[dict[str, Any]] = []
    seen_candidate_keys: set[tuple[str, str]] = set()
    for row in candidate_rows:
        _require(isinstance(row, dict), "candidate row must be an object")
        key = _validate_candidate(row)
        _require(key not in seen_candidate_keys, "duplicate candidate record identity")
        seen_candidate_keys.add(key)
        _require(key in raw_index, "candidate has no exact raw attribution record")
        author = raw_index[key]
        sidecar.append(
            {
                "schema_version": SCHEMA,
                "record_id": row["record_id"],
                "source_dataset": SOURCE_DATASET,
                "source_revision": SOURCE_REVISION,
                "normalized_sha256": row["normalized_sha256"],
                "normalized_utf8_bytes": row["normalized_utf8_bytes"],
                "origin_url": row["origin_url"],
                "license": EXPECTED_LICENSE,
                "attribution": {
                    "author": author,
                    "publisher": PUBLISHER,
                    "source_url": row["origin_url"],
                    "license_url": LICENSE_URL,
                    "reuse_terms_url": REUSE_TERMS_URL,
                },
                "attribution_metadata_in_training_text": False,
                "training_eligible": False,
                "evaluation_eligible": False,
            }
        )

    sidecar_bytes = b"".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True).encode("utf-8") + b"\n"
        for row in sidecar
    )
    projection = _candidate_projection(candidate_rows)
    report_core = {
        "schema_version": REPORT_SCHEMA,
        "status": "ATTRIBUTION_PROVENANCE_BOUND_ZERO_CREDIT",
        "source_dataset": SOURCE_DATASET,
        "source_revision": SOURCE_REVISION,
        "historical_execution_run_id": EXPECTED_EXECUTION_RUN_ID,
        "historical_candidate_jsonl_sha256": EXPECTED_CANDIDATE_JSONL_SHA256,
        "historical_candidate_jsonl_bytes": EXPECTED_CANDIDATE_JSONL_BYTES,
        "historical_report_identity_sha256": EXPECTED_HISTORICAL_REPORT_ID,
        "historical_manifest_identity_sha256": EXPECTED_HISTORICAL_MANIFEST_ID,
        "selected_record_count": len(sidecar),
        "selected_normalized_utf8_bytes": sum(
            int(row["normalized_utf8_bytes"]) for row in candidate_rows
        ),
        "candidate_projection_identity_sha256": _sha256(_canonical_bytes(projection)),
        "attribution_sidecar_jsonl_sha256": _sha256(sidecar_bytes),
        "author_strings_persisted_in_report": False,
        "training_text_modified": False,
        "canonical_corpus_admitted": False,
        "training_authorized_bytes": 0,
        "authorized_optimized_target_exposure": 0,
        "tokenizer_fit_authorized": False,
        "model_training_executed": False,
        "final_test_payload_accessed": False,
        "paid_compute_used": False,
        "source_rights_review_status": "REVIEW_REQUIRED",
        "fresh_real_replay_required": True,
    }
    report = {
        **report_core,
        "report_identity_sha256": _sha256(_canonical_bytes(report_core)),
    }
    return sidecar, report


def _read_jsonl_bytes(payload: bytes, *, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number, raw_line in enumerate(payload.splitlines(), start=1):
        _require(len(raw_line) <= 2_500_000, f"{label} line too large: {number}")
        try:
            value = json.loads(raw_line.decode("utf-8", errors="strict"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AttributionError(f"invalid {label} JSONL row: {number}") from exc
        _require(isinstance(value, dict), f"{label} row must be object: {number}")
        rows.append(dict(value))
    return rows


def _read_exact_candidate(path: Path) -> list[dict[str, Any]]:
    payload = path.read_bytes()
    _require(
        len(payload) == EXPECTED_CANDIDATE_JSONL_BYTES,
        "historical candidate byte count mismatch",
    )
    _require(
        _sha256(payload) == EXPECTED_CANDIDATE_JSONL_SHA256,
        "historical candidate SHA-256 mismatch",
    )
    rows = _read_jsonl_bytes(payload, label="candidate")
    _require(
        len(rows) == EXPECTED_SELECTED_RECORD_COUNT,
        "historical candidate record count mismatch",
    )
    for row in rows:
        _validate_candidate(row)
    normalized_bytes = sum(int(row["normalized_utf8_bytes"]) for row in rows)
    _require(
        normalized_bytes == EXPECTED_SELECTED_NORMALIZED_UTF8_BYTES,
        "historical candidate normalized byte count mismatch",
    )
    projection = _candidate_projection(rows)
    _require(
        _sha256(_canonical_bytes(projection)) == EXPECTED_CANDIDATE_PROJECTION_SHA256,
        "historical candidate projection mismatch",
    )
    return rows


def _read_raw_rows(
    config: dict[str, Any],
    input_dir: Path,
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {"collection": [], "essay": []}
    entries = {entry["path"]: entry for entry in config["files"]}
    for path, expected in EXPECTED_FILES.items():
        source_path = input_dir / path
        _require(source_path.is_file(), f"missing exact PDR source file: {path}")
        payload = source_path.read_bytes()
        _require(_sha256(payload) == expected["sha256"], f"source payload drift: {path}")
        try:
            raw_payload = gzip.decompress(payload)
        except (OSError, EOFError) as exc:
            raise AttributionError(f"invalid PDR gzip payload: {path}") from exc
        rows = _read_jsonl_bytes(raw_payload, label=f"raw PDR {path}")
        for row in rows:
            _validate_raw_row(row, expected_type=expected["type"])
            result[expected["type"]].append(row)
        _require(
            entries[path].get("sha256") == expected["sha256"],
            "config/source hash mismatch",
        )
    return result


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    _require(isinstance(config, dict), "config root must be object")
    _validate_config(config)
    candidate_rows = _read_exact_candidate(args.candidate)
    raw_rows = _read_raw_rows(config, args.input_dir)
    sidecar, report = build_sidecar(candidate_rows, raw_rows)
    _require(
        report["candidate_projection_identity_sha256"]
        == EXPECTED_CANDIDATE_PROJECTION_SHA256,
        "sidecar projection binding mismatch",
    )
    _require(
        report["selected_record_count"] == EXPECTED_SELECTED_RECORD_COUNT,
        "sidecar record count mismatch",
    )
    _require(
        report["selected_normalized_utf8_bytes"]
        == EXPECTED_SELECTED_NORMALIZED_UTF8_BYTES,
        "sidecar normalized byte count mismatch",
    )
    _write_jsonl(args.sidecar, sidecar)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "selected_record_count": report["selected_record_count"],
                "attribution_sidecar_jsonl_sha256": report[
                    "attribution_sidecar_jsonl_sha256"
                ],
                "training_authorized_bytes": 0,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
