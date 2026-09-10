#!/usr/bin/env python3
"""Bounded zero-credit materializer for Common Pile Public Domain Review."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import re
import unicodedata
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", re.IGNORECASE)
PHONE_RE = re.compile(
    r"(?<!\d)(?:\+?\d{1,3}[\s().-]*)?(?:\d[\s().-]*){8,14}(?!\d)"
)
SECRET_RE = re.compile(
    r"(?i)(?:api[_-]?key|access[_-]?token|secret[_-]?key|authorization)"
    r"\s*[:=]\s*[A-Za-z0-9_./+=:-]{12,}"
)
WORD_RE = re.compile(r"[A-Za-z]+")
ENGLISH_WORDS = frozenset(
    {
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "in",
        "is",
        "of",
        "on",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "with",
    }
)
OUTPUT_SCHEMA = "12-6.d03-common-pile-public-domain-review-report.v1"
CONFIG_SCHEMA = "12-6.d03-common-pile-public-domain-review-source-v1"
EXPECTED_WORKER = "D03-COMMON-PILE-PUBLIC-DOMAIN-REVIEW-EN"
EXPECTED_BASE_MAIN_SHA = "a55c6ea34566faae6f82a506d8e0f8691ddea716"
EXPECTED_RIGHTS_MERGE_SHA = "45e27ac2d32008dd99338d30ec3429dc0d964477"
EXPECTED_RIGHTS_REGISTRY_ID = "COMMON-PILE-SOURCE-RIGHTS-V1"
EXPECTED_RIGHTS_REGISTRY_SHA256 = (
    "b279c4a7404e0e501acd0c77842d1f71c43ea1dbcd611acab18163c64b060d4e"
)
EXPECTED_SOURCE_REVISION = "177f70f044bd7a347616727c0f33ab4af9baa495"
EXPECTED_FAMILY_ID = "en.common-pile.public-domain-review"
EXPECTED_SOURCE_VALUE = "public-domain-review"
EXPECTED_ORIGIN_HOST = "publicdomainreview.org"
EXPECTED_RECORD_LICENSE = (
    "Creative Commons - Attribution Share-Alike - "
    "https://creativecommons.org/licenses/by-sa/4.0/"
)
EXPECTED_FILES = {
    "v0/00000_collections.jsonl.gz": {
        "sha256": "0a5cbe305d7f468aad95a8cfc4d5fae14132bc359ccf92fc3705072db33a46fd",
        "max_compressed_bytes": 2_000_000,
        "max_uncompressed_bytes": 12_000_000,
    },
    "v0/00000_essays.jsonl.gz": {
        "sha256": "825aed6a66ab14f89e1fe3110f67265d511994c52e5aed48b866a9e8a1bdeb4d",
        "max_compressed_bytes": 2_500_000,
        "max_uncompressed_bytes": 14_000_000,
    },
}
EXPECTED_SELECTION = {
    "max_normalized_utf8_bytes": 4_800_000,
    "max_record_utf8_bytes": 200_000,
    "min_record_utf8_bytes": 120,
    "min_alpha_chars": 80,
    "min_latin_alpha_ratio": 0.9,
    "min_english_lexical_hits": 3,
}
REQUIRED_ZERO_FIELDS = {
    "canonical_corpus_admitted": False,
    "family_credit": False,
    "source_capacity_bytes_credited": 0,
    "training_authorized_bytes": 0,
    "authorized_optimized_target_exposure": 0,
    "evaluation_authorized_bytes": 0,
    "global_dedup_complete": False,
    "reserved_evaluation_decontamination_complete": False,
    "post_composition_quality_privacy_complete": False,
    "balance_family_caps_complete": False,
    "cluster_safe_split_complete": False,
    "deterministic_packing_complete": False,
    "two_clean_builds_complete": False,
    "postpack_unique_loss_ledger_complete": False,
    "tokenizer_fit_authorized": False,
    "optimizer_updates": 0,
    "model_training_executed": False,
    "final_test_payload_accessed": False,
    "paid_compute_used": False,
}


class PublicDomainReviewError(RuntimeError):
    """Fail-closed source materialization error."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PublicDomainReviewError(message)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _validate_sha256(value: Any, label: str) -> str:
    _require(
        isinstance(value, str)
        and len(value) == 64
        and all(ch in "0123456789abcdef" for ch in value),
        f"{label} must be lowercase 64-hex SHA-256",
    )
    return value


def _validate_config(cfg: dict[str, Any]) -> None:
    _require(cfg.get("schema_version") == CONFIG_SCHEMA, "config schema drift")
    _require(cfg.get("worker") == EXPECTED_WORKER, "worker identity drift")
    _require(cfg.get("base_main_sha") == EXPECTED_BASE_MAIN_SHA, "base-main identity drift")
    _require(cfg.get("execution_profile") == "LOCAL_FREE", "execution profile drift")

    rights = cfg.get("common_pile_rights_authority")
    _require(isinstance(rights, dict), "Common Pile rights authority missing")
    _require(
        rights.get("merge_commit_sha") == EXPECTED_RIGHTS_MERGE_SHA,
        "Common Pile rights merge identity drift",
    )
    _require(
        rights.get("registry_id") == EXPECTED_RIGHTS_REGISTRY_ID,
        "Common Pile rights registry id drift",
    )
    _require(
        rights.get("registry_identity_sha256") == EXPECTED_RIGHTS_REGISTRY_SHA256,
        "Common Pile rights registry identity drift",
    )
    _require(rights.get("source_key") == "public_domain_review", "source-rights key drift")
    _require(rights.get("project_review_status") == "REVIEW_REQUIRED", "rights review status drift")

    source = cfg.get("source")
    _require(isinstance(source, dict), "source config missing")
    _require(source.get("dataset") == "common-pile/public_domain_review", "dataset identity drift")
    _require(source.get("revision") == EXPECTED_SOURCE_REVISION, "source revision drift")
    _require(source.get("family_id") == EXPECTED_FAMILY_ID, "source family drift")
    _require(source.get("source_value") == EXPECTED_SOURCE_VALUE, "source value drift")
    _require(source.get("origin_host") == EXPECTED_ORIGIN_HOST, "origin host drift")
    _require(
        source.get("expected_record_license") == EXPECTED_RECORD_LICENSE,
        "record license policy drift",
    )
    _require(source.get("attribution_required") is True, "attribution boundary drift")

    files = cfg.get("files")
    _require(isinstance(files, list), "source files missing")
    observed_files: dict[str, dict[str, Any]] = {}
    for entry in files:
        _require(isinstance(entry, dict), "source file entry must be object")
        path = entry.get("path")
        _require(
            isinstance(path, str) and path not in observed_files,
            "invalid/duplicate file path",
        )
        observed_files[path] = dict(entry)
    _require(set(observed_files) == set(EXPECTED_FILES), "source file set drift")
    for path, expected in EXPECTED_FILES.items():
        actual = observed_files[path]
        _validate_sha256(actual.get("sha256"), f"{path}.sha256")
        for key, expected_value in expected.items():
            _require(actual.get(key) == expected_value, f"source file policy drift: {path}:{key}")

    selection = cfg.get("selection")
    _require(isinstance(selection, dict), "selection config missing")
    _require(selection == EXPECTED_SELECTION, "selection policy drift")

    boundary = cfg.get("truth_boundary")
    _require(isinstance(boundary, dict), "truth boundary missing")
    _require(boundary.get("candidate_snapshot_only") is True, "candidate boundary drift")
    for key, expected in REQUIRED_ZERO_FIELDS.items():
        _require(boundary.get(key) == expected, f"truth boundary weakened: {key}")


def _download_url(cfg: dict[str, Any], path: str) -> str:
    revision = cfg["source"]["revision"]
    return (
        "https://huggingface.co/datasets/common-pile/public_domain_review/"
        f"resolve/{revision}/{path}?download=true"
    )


def _fetch(url: str, max_bytes: int) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "12-6-ai-D03-PDR/1.0 (bounded zero-credit source intake)",
            "Accept": "application/octet-stream",
        },
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = response.read(max_bytes + 1)
    _require(len(payload) <= max_bytes, f"compressed payload exceeds bound: {url}")
    return payload


def _read_input(
    cfg: dict[str, Any],
    entry: dict[str, Any],
    input_dir: Path | None,
) -> bytes:
    path = entry["path"]
    if input_dir is None:
        payload = _fetch(_download_url(cfg, path), int(entry["max_compressed_bytes"]))
    else:
        candidate = input_dir / path
        _require(candidate.is_file(), f"missing local source file: {path}")
        payload = candidate.read_bytes()
        _require(
            len(payload) <= int(entry["max_compressed_bytes"]),
            f"compressed payload exceeds bound: {path}",
        )
    observed = _sha256(payload)
    expected = _validate_sha256(entry["sha256"], f"{path}.sha256")
    _require(observed == expected, f"source payload identity drift: {path}")
    return payload


def _iter_jsonl_gzip(
    payload: bytes,
    *,
    path: str,
    max_uncompressed_bytes: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total = 0
    try:
        handle = gzip.GzipFile(fileobj=io.BytesIO(payload), mode="rb")
        for ordinal, raw_line in enumerate(handle):
            total += len(raw_line)
            _require(
                total <= max_uncompressed_bytes,
                f"uncompressed payload exceeds bound: {path}",
            )
            _require(len(raw_line) <= 2_000_000, f"JSONL row too large: {path}:{ordinal}")
            try:
                line = raw_line.decode("utf-8", errors="strict")
                value = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise PublicDomainReviewError(
                    f"invalid UTF-8/JSON row: {path}:{ordinal}"
                ) from exc
            _require(isinstance(value, dict), f"JSONL row must be object: {path}:{ordinal}")
            rows.append(dict(value))
    except (OSError, EOFError) as exc:
        raise PublicDomainReviewError(f"invalid gzip payload: {path}") from exc
    return rows


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFC", value.replace("\r\n", "\n").replace("\r", "\n"))
    lines = [" ".join(line.split()) for line in normalized.split("\n")]
    return "\n".join(line for line in lines if line).strip()


def _has_forbidden_control(text: str) -> bool:
    return any(ord(ch) < 32 and ch not in "\n\t" for ch in text)


def _language_evidence(text: str) -> dict[str, Any]:
    letters = [ch for ch in text if ch.isalpha()]
    latin = [ch for ch in letters if ("A" <= ch <= "Z") or ("a" <= ch <= "z")]
    words = [match.group(0).casefold() for match in WORD_RE.finditer(text)]
    return {
        "alpha_chars": len(letters),
        "latin_alpha_ratio": round(len(latin) / len(letters), 6) if letters else 0.0,
        "english_lexical_hits": len({word for word in words if word in ENGLISH_WORDS}),
    }


def _validate_origin(url: Any, expected_host: str) -> str:
    _require(isinstance(url, str) and url, "record origin URL missing")
    parsed = urlparse(url)
    host = (parsed.hostname or "").casefold()
    _require(parsed.scheme == "https", "record origin URL must use HTTPS")
    _require(
        host == expected_host or host.endswith("." + expected_host),
        "record origin host drift",
    )
    return url


def _candidate_from_row(
    cfg: dict[str, Any],
    row: dict[str, Any],
    *,
    file_path: str,
    ordinal: int,
) -> tuple[dict[str, Any] | None, str]:
    source = cfg["source"]
    record_id = row.get("id")
    text = row.get("text")
    metadata = row.get("metadata")
    if not isinstance(record_id, (str, int)) or not str(record_id).strip():
        return None, "invalid_id"
    if not isinstance(text, str) or not text.strip():
        return None, "empty_text"
    if row.get("source") != source["source_value"]:
        return None, "source_drift"
    if not isinstance(metadata, dict):
        return None, "metadata_missing"
    if metadata.get("license") != source["expected_record_license"]:
        return None, "license_drift"
    try:
        origin_url = _validate_origin(metadata.get("url"), source["origin_host"])
    except PublicDomainReviewError:
        return None, "origin_drift"

    normalized = _normalize_text(text)
    if _has_forbidden_control(normalized):
        return None, "control_character"
    if EMAIL_RE.search(normalized) or PHONE_RE.search(normalized) or SECRET_RE.search(normalized):
        return None, "privacy_or_secret"
    encoded = normalized.encode("utf-8")
    selection = cfg["selection"]
    if len(encoded) < selection["min_record_utf8_bytes"]:
        return None, "too_short"
    if len(encoded) > selection["max_record_utf8_bytes"]:
        return None, "too_large"
    language = _language_evidence(normalized)
    if language["alpha_chars"] < selection["min_alpha_chars"]:
        return None, "too_few_letters"
    if language["latin_alpha_ratio"] < float(selection["min_latin_alpha_ratio"]):
        return None, "non_english_script"
    if language["english_lexical_hits"] < selection["min_english_lexical_hits"]:
        return None, "english_lexical_gate"

    payload_sha = _sha256(encoded)
    return (
        {
            "file_path": file_path,
            "ordinal": ordinal,
            "record_id": str(record_id),
            "origin_url": origin_url,
            "license": source["expected_record_license"],
            "normalized_sha256": payload_sha,
            "normalized_utf8_bytes": len(encoded),
            "text": normalized,
        },
        "accepted",
    )


def materialize(
    cfg: dict[str, Any],
    *,
    input_dir: Path | None,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    _validate_config(cfg)
    candidates: list[dict[str, Any]] = []
    source_files: list[dict[str, Any]] = []
    rejected: dict[str, int] = {}
    source_row_count = 0

    for entry in sorted(cfg["files"], key=lambda item: item["path"]):
        payload = _read_input(cfg, entry, input_dir)
        path = entry["path"]
        rows = _iter_jsonl_gzip(
            payload,
            path=path,
            max_uncompressed_bytes=int(entry["max_uncompressed_bytes"]),
        )
        source_row_count += len(rows)
        source_files.append(
            {
                "path": path,
                "compressed_bytes": len(payload),
                "sha256": _sha256(payload),
                "row_count": len(rows),
            }
        )
        for ordinal, row in enumerate(rows):
            candidate, reason = _candidate_from_row(
                cfg,
                row,
                file_path=path,
                ordinal=ordinal,
            )
            if candidate is None:
                rejected[reason] = rejected.get(reason, 0) + 1
                continue
            candidates.append(candidate)

    candidates.sort(key=lambda row: (row["file_path"], row["record_id"], row["ordinal"]))
    unique: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    for row in candidates:
        payload_sha = row["normalized_sha256"]
        if payload_sha in seen_hashes:
            rejected["exact_duplicate"] = rejected.get("exact_duplicate", 0) + 1
            continue
        seen_hashes.add(payload_sha)
        unique.append(row)

    selected: list[dict[str, Any]] = []
    selected_text_bytes = 0
    cap = int(cfg["selection"]["max_normalized_utf8_bytes"])
    for row in unique:
        row_bytes = int(row["normalized_utf8_bytes"])
        if selected_text_bytes + row_bytes > cap:
            rejected["family_byte_cap"] = rejected.get("family_byte_cap", 0) + 1
            continue
        selected.append(row)
        selected_text_bytes += row_bytes

    public_rows = [
        {
            "artifact_role": "SOURCE_CANDIDATE_ONLY",
            "record_id": row["record_id"],
            "source_id": cfg["source"]["family_id"],
            "source_dataset": cfg["source"]["dataset"],
            "source_revision": cfg["source"]["revision"],
            "origin_url": row["origin_url"],
            "license": row["license"],
            "attribution_required": True,
            "language": "en",
            "normalized_sha256": row["normalized_sha256"],
            "normalized_utf8_bytes": row["normalized_utf8_bytes"],
            "text": row["text"],
            "training_eligible": False,
            "evaluation_eligible": False,
        }
        for row in selected
    ]
    projection = [
        {
            "record_id": row["record_id"],
            "normalized_sha256": row["normalized_sha256"],
            "normalized_utf8_bytes": row["normalized_utf8_bytes"],
            "origin_url_sha256": _sha256(row["origin_url"].encode("utf-8")),
        }
        for row in public_rows
    ]
    projection_identity = _sha256(_canonical_bytes(projection))

    report_core = {
        "schema_version": OUTPUT_SCHEMA,
        "worker": cfg["worker"],
        "status": "CANDIDATE_MATERIALIZED_ZERO_CREDIT",
        "execution_profile": "LOCAL_FREE",
        "source": {
            "dataset": cfg["source"]["dataset"],
            "revision": cfg["source"]["revision"],
            "family_id": cfg["source"]["family_id"],
            "common_pile_source_key": cfg["common_pile_rights_authority"]["source_key"],
            "rights_registry_identity_sha256": cfg["common_pile_rights_authority"][
                "registry_identity_sha256"
            ],
        },
        "source_files": source_files,
        "source_row_count": source_row_count,
        "selected_record_count": len(public_rows),
        "selected_normalized_utf8_bytes": selected_text_bytes,
        "rejected": dict(sorted(rejected.items())),
        "record_projection_identity_sha256": projection_identity,
        "raw_text_persisted_in_report": False,
        "truth_boundary": cfg["truth_boundary"],
        "next_required_gates": [
            "current_global_cross_family_dedup",
            "fresh_reserved_evaluation_decontamination",
            "post_composition_quality_privacy",
            "balance_and_family_caps",
            "cluster_safe_split",
            "deterministic_packing_and_two_clean_builds",
            "positive_exact_postpack_unique_loss_ledger",
        ],
    }
    report = {
        **report_core,
        "report_identity_sha256": _sha256(_canonical_bytes(report_core)),
    }
    candidate_bytes = (
        b"".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True).encode("utf-8") + b"\n"
            for row in public_rows
        )
    )
    manifest_core = {
        "schema_version": "12-6.d03-common-pile-public-domain-review-manifest.v1",
        "source_revision": cfg["source"]["revision"],
        "source_file_vector_sha256": _sha256(_canonical_bytes(source_files)),
        "candidate_jsonl_sha256": _sha256(candidate_bytes),
        "candidate_jsonl_bytes": len(candidate_bytes),
        "selected_record_count": len(public_rows),
        "selected_normalized_utf8_bytes": selected_text_bytes,
        "record_projection_identity_sha256": projection_identity,
        "report_identity_sha256": report["report_identity_sha256"],
        "artifact_role": "SOURCE_CANDIDATE_ONLY",
        "training_authorized_bytes": 0,
    }
    manifest = {
        **manifest_core,
        "manifest_identity_sha256": _sha256(_canonical_bytes(manifest_core)),
    }
    return public_rows, report, manifest


def _write_outputs(
    output_dir: Path,
    rows: list[dict[str, Any]],
    report: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = output_dir / "candidate.jsonl"
    with candidate_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--input-dir",
        type=Path,
        help="Optional local root containing exact bound v0/*.jsonl.gz files.",
    )
    args = parser.parse_args()

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    _require(isinstance(cfg, dict), "config root must be object")
    rows, report, manifest = materialize(cfg, input_dir=args.input_dir)
    _write_outputs(args.output, rows, report, manifest)
    print(
        json.dumps(
            {
                "status": report["status"],
                "selected_record_count": report["selected_record_count"],
                "selected_normalized_utf8_bytes": report[
                    "selected_normalized_utf8_bytes"
                ],
                "report_identity_sha256": report["report_identity_sha256"],
                "training_authorized_bytes": 0,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
