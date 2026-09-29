#!/usr/bin/env python3
"""Derive text-free PDR attributable-subset accounting from exact replay outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

SCHEMA = "12-6.d03-pdr-attributable-subset-accounting.v1"
SOURCE_DATASET = "common-pile/public_domain_review"
SOURCE_REVISION = "177f70f044bd7a347616727c0f33ab4af9baa495"
SOURCE_FAMILY = "en.common-pile.public-domain-review"
SIDECAR_SCHEMA = "12-6.d03-pdr-attribution-sidecar.v1"
PUBLISHER = "The Public Domain Review"
LICENSE_URL = "https://creativecommons.org/licenses/by-sa/4.0/"
REUSE_TERMS_URL = "https://publicdomainreview.org/reusing-material"
EXPECTED_LICENSE = (
    "Creative Commons - Attribution Share-Alike - "
    "https://creativecommons.org/licenses/by-sa/4.0/"
)
EXPECTED_CANDIDATE_SHA256 = "38b0ef94062b80ae1cb8055fe897feb10a1e8d04a9b17340cbc080d4bfb21591"
EXPECTED_CANDIDATE_BYTES = 5_630_935
EXPECTED_CANDIDATE_RECORDS = 1_166
EXPECTED_CANDIDATE_NORMALIZED_BYTES = 4_795_007
EXPECTED_CANDIDATE_PROJECTION = "bf9f7896fc2c940458916184b1b702f443d72290e286ee16b24da4088b2aa4ab"
EXPECTED_SIDECAR_SHA256 = "d4702f8c2c0fb92a9ab813c3578b91ff849ca1f98b785c1f5ae10ec86fd43535"
EXPECTED_SIDECAR_BYTES = 481_511
EXPECTED_ATTRIBUTABLE_RECORDS = 498
EXPECTED_EXCLUDED_RECORDS = 668
EXPECTED_ATTRIBUTABLE_NORMALIZED_BYTES = 3_727_864
EXPECTED_ATTRIBUTABLE_PROJECTION = "79cff661429f365c412e3113a72a0d8b35566f19f61facca3bc643d86299e32b"
EXPECTED_ATTRIBUTABLE_NORMALIZED_BYTES = 3_727_864
EXPECTED_ATTRIBUTABLE_PROJECTION = "79cff661429f365c412e3113a72a0d8b35566f19f61facca3bc643d86299e32b"
EXPECTED_AUTHORITY_IDENTITY = "0fd7388521c693028f999a58737ca40b4ce9640eedcb33dda8a9559d62d77832"
EXPECTED_EXCLUSION_IDENTITY = "5e4d99ab35bc08c3544fa99d0eb362a40e2873b75b443f74bdfc31b9eaf0dbb1"
EXPECTED_REPORT_FILE_SHA256 = "cc598d0c9765bc17c7e0e584630009ad299c909135c3442eb99b2af1b8c82dff"
EXPECTED_REPORT_IDENTITY = "8de53f55e602264d6d3401269faf270253e6f162fd99dfbdd00907c816cf47d9"

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
SIDECAR_KEYS = frozenset(
    {
        "schema_version",
        "record_id",
        "source_dataset",
        "source_revision",
        "normalized_sha256",
        "normalized_utf8_bytes",
        "origin_url",
        "license",
        "attribution",
        "attribution_metadata_in_training_text",
        "training_eligible",
        "evaluation_eligible",
    }
)
ATTRIBUTION_KEYS = frozenset(
    {"author", "publisher", "source_url", "license_url", "reuse_terms_url"}
)


class SubsetAccountingError(RuntimeError):
    """Fail-closed PDR subset-accounting error."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SubsetAccountingError(message)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _pairs_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SubsetAccountingError(f"duplicate JSON member: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise SubsetAccountingError(f"non-finite JSON number: {value}")


def _loads_strict(raw: bytes, *, label: str) -> Any:
    try:
        text = raw.decode("utf-8", errors="strict")
        return json.loads(
            text,
            object_pairs_hook=_pairs_no_duplicates,
            parse_constant=_reject_nonfinite,
        )
    except UnicodeDecodeError as exc:
        raise SubsetAccountingError(f"{label} is not UTF-8") from exc
    except json.JSONDecodeError as exc:
        raise SubsetAccountingError(f"{label} is invalid JSON") from exc


def _read_jsonl(raw: bytes, *, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(raw.splitlines(), start=1):
        _require(bool(line), f"{label} contains blank row: {number}")
        value = _loads_strict(line, label=f"{label} row {number}")
        _require(type(value) is dict, f"{label} row must be object: {number}")
        rows.append(value)
    return rows


def _positive_int(value: Any, message: str) -> int:
    _require(type(value) is int and value > 0, message)
    return value


def _zero_int(value: Any, message: str) -> None:
    _require(type(value) is int and value == 0, message)


def _validate_candidate(row: dict[str, Any]) -> tuple[str, str, str, int]:
    _require(set(row) == CANDIDATE_KEYS, "candidate schema drift")
    _require(row["artifact_role"] == "SOURCE_CANDIDATE_ONLY", "candidate role drift")
    _require(row["source_id"] == SOURCE_FAMILY, "candidate family drift")
    _require(row["source_dataset"] == SOURCE_DATASET, "candidate dataset drift")
    _require(row["source_revision"] == SOURCE_REVISION, "candidate revision drift")
    _require(row["license"] == EXPECTED_LICENSE, "candidate license drift")
    _require(row["attribution_required"] is True, "candidate attribution drift")
    _require(row["language"] == "en", "candidate language drift")
    _require(row["training_eligible"] is False, "candidate training drift")
    _require(row["evaluation_eligible"] is False, "candidate evaluation drift")
    record_id = row["record_id"]
    origin_url = row["origin_url"]
    digest = row["normalized_sha256"]
    text = row["text"]
    _require(type(record_id) is str and bool(record_id.strip()), "candidate id invalid")
    _require(type(origin_url) is str and origin_url.startswith("https://"), "candidate URL invalid")
    _require(type(digest) is str and len(digest) == 64, "candidate digest invalid")
    _require(type(text) is str, "candidate text invalid")
    encoded = text.encode("utf-8")
    byte_count = _positive_int(row["normalized_utf8_bytes"], "candidate bytes invalid")
    _require(len(encoded) == byte_count, "candidate byte count mismatch")
    _require(_sha256(encoded) == digest, "candidate text digest mismatch")
    return record_id, origin_url, digest, byte_count


def _validate_sidecar(row: dict[str, Any]) -> tuple[str, str, str, int]:
    _require(set(row) == SIDECAR_KEYS, "sidecar schema drift")
    _require(row["schema_version"] == SIDECAR_SCHEMA, "sidecar version drift")
    _require(row["source_dataset"] == SOURCE_DATASET, "sidecar dataset drift")
    _require(row["source_revision"] == SOURCE_REVISION, "sidecar revision drift")
    _require(row["license"] == EXPECTED_LICENSE, "sidecar license drift")
    _require(row["attribution_metadata_in_training_text"] is False, "sidecar training-text drift")
    _require(row["training_eligible"] is False, "sidecar training drift")
    _require(row["evaluation_eligible"] is False, "sidecar evaluation drift")
    attribution = row["attribution"]
    _require(type(attribution) is dict and set(attribution) == ATTRIBUTION_KEYS, "attribution schema drift")
    author = attribution["author"]
    _require(type(author) is str and bool(author.strip()) and len(author) <= 512, "attribution author invalid")
    _require(not any(ord(ch) < 32 for ch in author), "attribution author control character")
    _require(attribution["publisher"] == PUBLISHER, "attribution publisher drift")
    _require(attribution["source_url"] == row["origin_url"], "attribution source URL drift")
    _require(attribution["license_url"] == LICENSE_URL, "attribution license URL drift")
    _require(attribution["reuse_terms_url"] == REUSE_TERMS_URL, "attribution reuse terms drift")
    record_id = row["record_id"]
    origin_url = row["origin_url"]
    digest = row["normalized_sha256"]
    _require(type(record_id) is str and bool(record_id.strip()), "sidecar id invalid")
    _require(type(origin_url) is str and origin_url.startswith("https://"), "sidecar URL invalid")
    _require(type(digest) is str and len(digest) == 64, "sidecar digest invalid")
    byte_count = _positive_int(row["normalized_utf8_bytes"], "sidecar bytes invalid")
    return record_id, origin_url, digest, byte_count


def derive_subset(
    candidate_rows: list[dict[str, Any]],
    sidecar_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    _require(len(candidate_rows) == EXPECTED_CANDIDATE_RECORDS, "candidate count mismatch")
    _require(len(sidecar_rows) == EXPECTED_ATTRIBUTABLE_RECORDS, "attributable count mismatch")
    candidates: dict[tuple[str, str], tuple[str, int]] = {}
    for row in candidate_rows:
        record_id, origin_url, digest, byte_count = _validate_candidate(row)
        key = (record_id, origin_url)
        _require(key not in candidates, "duplicate candidate identity")
        candidates[key] = (digest, byte_count)

    projection: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    subset_bytes = 0
    for row in sidecar_rows:
        record_id, origin_url, digest, byte_count = _validate_sidecar(row)
        key = (record_id, origin_url)
        _require(key not in seen, "duplicate sidecar identity")
        seen.add(key)
        _require(key in candidates, "sidecar row absent from exact candidate")
        _require(candidates[key] == (digest, byte_count), "sidecar/candidate identity drift")
        subset_bytes += byte_count
        projection.append(
            {
                "record_id_sha256": _sha256(record_id.encode("utf-8")),
                "origin_url_sha256": _sha256(origin_url.encode("utf-8")),
                "normalized_sha256": digest,
                "normalized_utf8_bytes": byte_count,
            }
        )

    _require(len(seen) == EXPECTED_ATTRIBUTABLE_RECORDS, "subset cardinality drift")
    projection_identity = _sha256(_canonical_bytes(projection))
    _require(subset_bytes == EXPECTED_ATTRIBUTABLE_NORMALIZED_BYTES, "attributable byte authority drift")
    _require(projection_identity == EXPECTED_ATTRIBUTABLE_PROJECTION, "attributable projection authority drift")
    core = {
        "schema_version": SCHEMA,
        "status": "ATTRIBUTABLE_SUBSET_ACCOUNTED_REVIEW_REQUIRED_ZERO_CREDIT",
        "source_dataset": SOURCE_DATASET,
        "source_revision": SOURCE_REVISION,
        "historical_candidate_jsonl_sha256": EXPECTED_CANDIDATE_SHA256,
        "historical_candidate_record_count": EXPECTED_CANDIDATE_RECORDS,
        "historical_candidate_normalized_utf8_bytes": EXPECTED_CANDIDATE_NORMALIZED_BYTES,
        "historical_candidate_projection_identity_sha256": EXPECTED_CANDIDATE_PROJECTION,
        "historical_attribution_sidecar_sha256": EXPECTED_SIDECAR_SHA256,
        "historical_exclusion_identity_sha256": EXPECTED_EXCLUSION_IDENTITY,
        "attributable_record_count": len(seen),
        "excluded_record_count": EXPECTED_EXCLUDED_RECORDS,
        "attributable_normalized_utf8_bytes": subset_bytes,
        "attributable_projection_identity_sha256": projection_identity,
        "record_ids_persisted_in_authority": False,
        "origin_urls_persisted_in_authority": False,
        "author_values_persisted_in_authority": False,
        "source_rights_review_status": "REVIEW_REQUIRED",
        "pdr_source_admitted_records": 0,
        "pdr_source_admitted_bytes": 0,
        "canonical_capacity_credited": 0,
        "training_authorized_bytes": 0,
        "authorized_optimized_target_exposure": 0,
        "tokenizer_fit_authorized": False,
        "optimizer_updates_executed_on_real_targets": 0,
        "training_executed": False,
        "learned_weights_created": False,
        "final_test_outcomes_read": False,
        "paid_compute_used": False,
        "foreign_pretrained_weights_used": False,
    }
    authority_identity = _sha256(_canonical_bytes(core))
    return {**core, "authority_identity_sha256": authority_identity}


def _validate_historical_report(report: dict[str, Any]) -> None:
    _require(report.get("selected_record_count") == EXPECTED_CANDIDATE_RECORDS, "report candidate count drift")
    _require(report.get("attributable_record_count") == EXPECTED_ATTRIBUTABLE_RECORDS, "report attributable count drift")
    _require(report.get("excluded_record_count") == EXPECTED_EXCLUDED_RECORDS, "report excluded count drift")
    _require(report.get("excluded_reason_counts") == {"missing": EXPECTED_EXCLUDED_RECORDS}, "report exclusion reasons drift")
    _require(report.get("excluded_candidate_identity_sha256") == EXPECTED_EXCLUSION_IDENTITY, "report exclusion identity drift")
    _require(report.get("attribution_sidecar_jsonl_sha256") == EXPECTED_SIDECAR_SHA256, "report sidecar identity drift")
    _require(report.get("candidate_projection_identity_sha256") == EXPECTED_CANDIDATE_PROJECTION, "report candidate projection drift")
    _require(report.get("report_identity_sha256") == EXPECTED_REPORT_IDENTITY, "report identity drift")
    _require(report.get("source_rights_review_status") == "REVIEW_REQUIRED", "report rights drift")
    _zero_int(report.get("training_authorized_bytes"), "report training credit drift")
    _zero_int(report.get("authorized_optimized_target_exposure"), "report exposure drift")
    _require(report.get("canonical_corpus_admitted") is False, "report corpus admission drift")
    _require(report.get("tokenizer_fit_authorized") is False, "report tokenizer drift")
    _require(report.get("model_training_executed") is False, "report training drift")
    _require(report.get("final_test_payload_accessed") is False, "report final-test drift")
    _require(report.get("paid_compute_used") is False, "report paid-compute drift")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--attribution-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    candidate_raw = args.candidate.read_bytes()
    sidecar_raw = args.sidecar.read_bytes()
    report_raw = args.attribution_report.read_bytes()
    _require(len(candidate_raw) == EXPECTED_CANDIDATE_BYTES, "candidate file byte count mismatch")
    _require(_sha256(candidate_raw) == EXPECTED_CANDIDATE_SHA256, "candidate file identity drift")
    _require(len(sidecar_raw) == EXPECTED_SIDECAR_BYTES, "sidecar file byte count mismatch")
    _require(_sha256(sidecar_raw) == EXPECTED_SIDECAR_SHA256, "sidecar file identity drift")
    _require(_sha256(report_raw) == EXPECTED_REPORT_FILE_SHA256, "attribution report file identity drift")

    candidate_rows = _read_jsonl(candidate_raw, label="candidate")
    sidecar_rows = _read_jsonl(sidecar_raw, label="sidecar")
    report = _loads_strict(report_raw, label="attribution report")
    _require(type(report) is dict, "attribution report root must be object")
    _validate_historical_report(report)

    authority = derive_subset(candidate_rows, sidecar_rows)
    _require(
        authority["attributable_normalized_utf8_bytes"]
        == EXPECTED_ATTRIBUTABLE_NORMALIZED_BYTES,
        "attributable normalized byte count drift",
    )
    _require(
        authority["attributable_projection_identity_sha256"]
        == EXPECTED_ATTRIBUTABLE_PROJECTION,
        "attributable projection identity drift",
    )
    _require(
        authority["authority_identity_sha256"] == EXPECTED_AUTHORITY_IDENTITY,
        "subset authority identity drift",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(authority, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(authority, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
