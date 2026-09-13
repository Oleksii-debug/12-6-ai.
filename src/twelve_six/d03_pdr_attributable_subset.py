"""Exact, fail-closed accounting for the attributable Public Domain Review subset."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

SOURCE_DATASET = "common-pile/public_domain_review"
SOURCE_REVISION = "177f70f044bd7a347616727c0f33ab4af9baa495"
SOURCE_FAMILY = "en.common-pile.public-domain-review"
SOURCE_HOST = "publicdomainreview.org"
LICENSE = (
    "Creative Commons - Attribution Share-Alike - "
    "https://creativecommons.org/licenses/by-sa/4.0/"
)
LICENSE_URL = "https://creativecommons.org/licenses/by-sa/4.0/"
REUSE_TERMS_URL = "https://publicdomainreview.org/reusing-material"
PUBLISHER = "The Public Domain Review"

CANDIDATE_SHA256 = "38b0ef94062b80ae1cb8055fe897feb10a1e8d04a9b17340cbc080d4bfb21591"
CANDIDATE_FILE_BYTES = 5_630_935
CANDIDATE_RECORDS = 1_166
CANDIDATE_NORMALIZED_BYTES = 4_795_007
CANDIDATE_PROJECTION_SHA256 = (
    "bf9f7896fc2c940458916184b1b702f443d72290e286ee16b24da4088b2aa4ab"
)
SIDECAR_SHA256 = "d4702f8c2c0fb92a9ab813c3578b91ff849ca1f98b785c1f5ae10ec86fd43535"
SIDECAR_FILE_BYTES = 481_511
REPORT_FILE_SHA256 = "cc598d0c9765bc17c7e0e584630009ad299c909135c3442eb99b2af1b8c82dff"
REPORT_IDENTITY_SHA256 = "8de53f55e602264d6d3401269faf270253e6f162fd99dfbdd00907c816cf47d9"
ATTRIBUTABLE_RECORDS = 498
EXCLUDED_RECORDS = 668
EXCLUDED_IDENTITY_SHA256 = (
    "5e4d99ab35bc08c3544fa99d0eb362a40e2873b75b443f74bdfc31b9eaf0dbb1"
)
SOURCE_CONFIG_BLOB_SHA1 = "d0f88bfdc5dd8b4947f3f3d4ea462862be93db65"
RIGHTS_REGISTRY_IDENTITY_SHA256 = (
    "b279c4a7404e0e501acd0c77842d1f71c43ea1dbcd611acab18163c64b060d4e"
)

CANDIDATE_KEYS = {
    "artifact_role", "record_id", "source_id", "source_dataset", "source_revision",
    "origin_url", "license", "attribution_required", "language", "normalized_sha256",
    "normalized_utf8_bytes", "text", "training_eligible", "evaluation_eligible",
}
SIDECAR_KEYS = {
    "schema_version", "record_id", "source_dataset", "source_revision", "normalized_sha256",
    "normalized_utf8_bytes", "origin_url", "license", "attribution",
    "attribution_metadata_in_training_text", "training_eligible", "evaluation_eligible",
}
ATTRIBUTION_KEYS = {"author", "publisher", "source_url", "license_url", "reuse_terms_url"}
ZERO_BOUNDARY = {
    "canonical_capacity_credited": 0,
    "training_authorized_bytes": 0,
    "authorized_unique_loss_positions": 0,
    "authorized_optimized_target_exposure": 0,
    "tokenizer_fit_authorized": False,
    "optimizer_updates": 0,
    "model_training_executed": False,
    "learned_weights_created": False,
    "final_test_payload_accessed": False,
    "final_test_outcomes_read": False,
    "paid_compute_used": False,
    "foreign_pretrained_weights_used": False,
    "external_llm_or_api_used_for_data_or_intelligence": False,
}


class PdrSubsetError(RuntimeError):
    """Exact replay or accounting did not satisfy the bound contract."""


def _require(value: bool, message: str) -> None:
    if not value:
        raise PdrSubsetError(message)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _positive_int(value: Any, label: str) -> int:
    _require(type(value) is int and value > 0, f"{label} must be positive int")
    return value


def _sha256(value: Any, label: str) -> str:
    _require(
        isinstance(value, str)
        and len(value) == 64
        and all(ch in "0123456789abcdef" for ch in value),
        f"{label} must be lowercase SHA-256",
    )
    return value


def _url(value: Any, label: str) -> str:
    _require(isinstance(value, str) and value, f"{label} missing")
    parsed = urlparse(value)
    host = (parsed.hostname or "").casefold()
    _require(parsed.scheme == "https", f"{label} must use HTTPS")
    _require(host == SOURCE_HOST or host.endswith("." + SOURCE_HOST), f"{label} host drift")
    return value


def _candidate(row: dict[str, Any]) -> tuple[str, str]:
    _require(set(row) == CANDIDATE_KEYS, "candidate schema drift")
    fixed = {
        "artifact_role": "SOURCE_CANDIDATE_ONLY",
        "source_id": SOURCE_FAMILY,
        "source_dataset": SOURCE_DATASET,
        "source_revision": SOURCE_REVISION,
        "license": LICENSE,
        "attribution_required": True,
        "language": "en",
        "training_eligible": False,
        "evaluation_eligible": False,
    }
    for key, expected in fixed.items():
        _require(row[key] == expected and type(row[key]) is type(expected), f"candidate {key} drift")
    record_id = row["record_id"]
    _require(isinstance(record_id, str) and record_id.strip(), "candidate record id missing")
    origin_url = _url(row["origin_url"], "candidate origin URL")
    text = row["text"]
    _require(isinstance(text, str), "candidate text missing")
    encoded = text.encode("utf-8")
    _require(_sha(encoded) == _sha256(row["normalized_sha256"], "candidate text SHA"), "candidate text SHA mismatch")
    _require(len(encoded) == _positive_int(row["normalized_utf8_bytes"], "candidate bytes"), "candidate byte mismatch")
    return record_id.strip(), origin_url


def _sidecar(row: dict[str, Any]) -> tuple[str, str]:
    _require(set(row) == SIDECAR_KEYS, "sidecar schema drift")
    fixed = {
        "schema_version": "12-6.d03-pdr-attribution-sidecar.v1",
        "source_dataset": SOURCE_DATASET,
        "source_revision": SOURCE_REVISION,
        "license": LICENSE,
        "attribution_metadata_in_training_text": False,
        "training_eligible": False,
        "evaluation_eligible": False,
    }
    for key, expected in fixed.items():
        _require(row[key] == expected and type(row[key]) is type(expected), f"sidecar {key} drift")
    record_id = row["record_id"]
    _require(isinstance(record_id, str) and record_id.strip(), "sidecar record id missing")
    origin_url = _url(row["origin_url"], "sidecar origin URL")
    _sha256(row["normalized_sha256"], "sidecar text SHA")
    _positive_int(row["normalized_utf8_bytes"], "sidecar bytes")
    attribution = row["attribution"]
    _require(isinstance(attribution, dict) and set(attribution) == ATTRIBUTION_KEYS, "attribution schema drift")
    author = attribution["author"]
    _require(isinstance(author, str) and author.strip(), "author missing")
    _require(len(author) <= 512 and not any(ord(ch) < 32 for ch in author), "author invalid")
    _require(attribution["publisher"] == PUBLISHER, "publisher drift")
    _require(attribution["source_url"] == origin_url, "source URL drift")
    _require(attribution["license_url"] == LICENSE_URL, "license URL drift")
    _require(attribution["reuse_terms_url"] == REUSE_TERMS_URL, "reuse terms drift")
    return record_id.strip(), origin_url


def _jsonl(payload: bytes, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(payload.splitlines(), 1):
        _require(len(line) <= 2_500_000, f"{label} line too large: {number}")
        try:
            value = json.loads(line.decode("utf-8", errors="strict"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PdrSubsetError(f"invalid {label} row: {number}") from exc
        _require(isinstance(value, dict), f"{label} row must be object: {number}")
        rows.append(value)
    return rows


def _candidate_projection(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "record_id": row["record_id"],
            "normalized_sha256": row["normalized_sha256"],
            "normalized_utf8_bytes": row["normalized_utf8_bytes"],
            "origin_url_sha256": _sha(row["origin_url"].encode()),
        }
        for row in rows
    ]


def derive_subset(candidate_rows: list[dict[str, Any]], sidecar_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Derive text-free accounting; this alone does not grant source or training authority."""
    _require(bool(candidate_rows), "candidate rows empty")
    candidate: dict[tuple[str, str], dict[str, Any]] = {}
    order: dict[tuple[str, str], int] = {}
    for index, row in enumerate(candidate_rows):
        key = _candidate(row)
        _require(key not in candidate, "duplicate candidate identity")
        candidate[key] = row
        order[key] = index

    selected: list[dict[str, Any]] = []
    last = -1
    seen: set[tuple[str, str]] = set()
    for row in sidecar_rows:
        key = _sidecar(row)
        _require(key in candidate, "sidecar row absent from candidate")
        _require(key not in seen, "duplicate sidecar identity")
        _require(order[key] > last, "sidecar order drift")
        last = order[key]
        seen.add(key)
        source = candidate[key]
        _require(row["normalized_sha256"] == source["normalized_sha256"], "sidecar hash drift")
        _require(row["normalized_utf8_bytes"] == source["normalized_utf8_bytes"], "sidecar byte drift")
        selected.append(source)

    excluded = [
        {
            "record_id_sha256": _sha(row["record_id"].encode()),
            "origin_url_sha256": _sha(row["origin_url"].encode()),
            "reason": "missing",
        }
        for row in candidate_rows
        if (row["record_id"].strip(), row["origin_url"]) not in seen
    ]
    projection = [
        {
            "record_id_sha256": _sha(row["record_id"].encode()),
            "origin_url_sha256": _sha(row["origin_url"].encode()),
            "normalized_sha256": row["normalized_sha256"],
            "normalized_utf8_bytes": row["normalized_utf8_bytes"],
        }
        for row in selected
    ]
    core = {
        "schema_version": "12-6.d03-pdr-attributable-subset-receipt.v1",
        "source_dataset": SOURCE_DATASET,
        "source_revision": SOURCE_REVISION,
        "attributable_record_count": len(selected),
        "attributable_normalized_utf8_bytes": sum(row["normalized_utf8_bytes"] for row in selected),
        "attributable_projection_identity_sha256": _sha(_canonical(projection)),
        "excluded_record_count": len(excluded),
        "excluded_reason_counts": {"missing": len(excluded)},
        "excluded_candidate_identity_sha256": _sha(_canonical(excluded)),
        "truth_boundary": dict(ZERO_BOUNDARY),
    }
    return {**core, "subset_identity_sha256": _sha(_canonical(core))}


def derive_exact_replay(candidate_path: Path, sidecar_path: Path, report_path: Path) -> dict[str, Any]:
    """Bind the exact historical #1076 replay before deriving unknown subset bytes."""
    candidate_payload = candidate_path.read_bytes()
    _require(len(candidate_payload) == CANDIDATE_FILE_BYTES, "candidate file size drift")
    _require(_sha(candidate_payload) == CANDIDATE_SHA256, "candidate file SHA drift")
    candidate_rows = _jsonl(candidate_payload, "candidate")
    _require(len(candidate_rows) == CANDIDATE_RECORDS, "candidate record count drift")
    for row in candidate_rows:
        _candidate(row)
    _require(sum(row["normalized_utf8_bytes"] for row in candidate_rows) == CANDIDATE_NORMALIZED_BYTES, "candidate byte total drift")
    _require(_sha(_canonical(_candidate_projection(candidate_rows))) == CANDIDATE_PROJECTION_SHA256, "candidate projection drift")

    sidecar_payload = sidecar_path.read_bytes()
    _require(len(sidecar_payload) == SIDECAR_FILE_BYTES, "sidecar file size drift")
    _require(_sha(sidecar_payload) == SIDECAR_SHA256, "sidecar file SHA drift")
    sidecar_rows = _jsonl(sidecar_payload, "sidecar")

    report_payload = report_path.read_bytes()
    _require(_sha(report_payload) == REPORT_FILE_SHA256, "attribution report file SHA drift")
    report = json.loads(report_payload.decode("utf-8", errors="strict"))
    _require(isinstance(report, dict), "attribution report must be object")
    expected = {
        "report_identity_sha256": REPORT_IDENTITY_SHA256,
        "attributable_record_count": ATTRIBUTABLE_RECORDS,
        "excluded_record_count": EXCLUDED_RECORDS,
        "excluded_reason_counts": {"missing": EXCLUDED_RECORDS},
        "excluded_candidate_identity_sha256": EXCLUDED_IDENTITY_SHA256,
        "training_authorized_bytes": 0,
        "authorized_optimized_target_exposure": 0,
    }
    for key, value in expected.items():
        _require(report.get(key) == value and type(report.get(key)) is type(value), f"report {key} drift")

    subset = derive_subset(candidate_rows, sidecar_rows)
    _require(subset["attributable_record_count"] == ATTRIBUTABLE_RECORDS, "derived attributable count drift")
    _require(subset["excluded_record_count"] == EXCLUDED_RECORDS, "derived excluded count drift")
    _require(subset["excluded_candidate_identity_sha256"] == EXCLUDED_IDENTITY_SHA256, "derived exclusion root drift")
    binding = {
        "candidate_file_sha256": CANDIDATE_SHA256,
        "candidate_projection_identity_sha256": CANDIDATE_PROJECTION_SHA256,
        "attribution_sidecar_file_sha256": SIDECAR_SHA256,
        "attribution_report_file_sha256": REPORT_FILE_SHA256,
        "attribution_report_identity_sha256": REPORT_IDENTITY_SHA256,
        "source_config_blob_sha1": SOURCE_CONFIG_BLOB_SHA1,
        "rights_registry_identity_sha256": RIGHTS_REGISTRY_IDENTITY_SHA256,
    }
    core = {**subset, "historical_binding": binding}
    core.pop("subset_identity_sha256")
    return {**core, "receipt_identity_sha256": _sha(_canonical(core))}


def validate_blocker(payload: dict[str, Any]) -> None:
    """Validate the checked-in no-fabrication fallback as a closed-world receipt."""
    _require(
        set(payload)
        == {
            "schema_version", "status", "swarm_control_issue", "swarm_claim_issue", "lane_key",
            "execution_profile", "historical_authority", "source_authority", "retained_artifact_fact",
            "current_runtime_execution_blocker", "derived_source_admission", "truth_boundary",
        },
        "blocker schema drift",
    )
    _require(payload["schema_version"] == "12-6.d03-pdr-attributable-subset-replay-blocker.v1", "blocker version drift")
    _require(payload["status"] == "BLOCKED_ATTRIBUTABLE_SUBSET_REPLAY_REQUIRED_ZERO_CREDIT", "blocker status drift")
    _require(type(payload["swarm_control_issue"]) is int and payload["swarm_control_issue"] == 723, "control issue drift")
    _require(type(payload["swarm_claim_issue"]) is int and payload["swarm_claim_issue"] == 1728, "claim issue drift")
    _require(payload["execution_profile"] == "LOCAL_FREE", "execution profile drift")

    history = payload["historical_authority"]
    _require(history["candidate_jsonl_sha256"] == CANDIDATE_SHA256, "history candidate drift")
    _require(type(history["selected_record_count"]) is int and history["selected_record_count"] == CANDIDATE_RECORDS, "history count drift")
    _require(type(history["selected_normalized_utf8_bytes"]) is int and history["selected_normalized_utf8_bytes"] == CANDIDATE_NORMALIZED_BYTES, "history bytes drift")
    _require(type(history["attributable_record_count"]) is int and history["attributable_record_count"] == ATTRIBUTABLE_RECORDS, "history attributable drift")
    _require(type(history["excluded_record_count"]) is int and history["excluded_record_count"] == EXCLUDED_RECORDS, "history excluded drift")
    _require(history["excluded_candidate_identity_sha256"] == EXCLUDED_IDENTITY_SHA256, "history exclusion drift")

    blocker = payload["current_runtime_execution_blocker"]
    _require(blocker["exact_replay_attempted"] is True, "replay attempt drift")
    _require(blocker["exact_replay_completed"] is False, "replay completion drift")
    _require(blocker["external_dataset_api_substitution_used"] is False, "API substitution drift")
    _require(blocker["fabricated_attributable_bytes"] is False, "fabrication drift")

    admission = payload["derived_source_admission"]
    _require(
        admission
        == {
            "executed": False,
            "source_admitted_records": 0,
            "source_admitted_normalized_utf8_bytes": 0,
            "attributable_normalized_utf8_bytes": None,
            "attributable_projection_identity_sha256": None,
        },
        "blocked admission boundary drift",
    )
    _require(payload["truth_boundary"] == ZERO_BOUNDARY, "truth boundary drift")
