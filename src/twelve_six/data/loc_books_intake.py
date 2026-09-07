from __future__ import annotations

import gzip
import hashlib
import io
import json
import re
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, BinaryIO
from urllib.parse import urlparse

_HEX40_RE = re.compile(r"^[0-9a-f]{40}$")
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_RECORD_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_EMAIL_RE = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
_SECRET_RES = (
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bgh[opsu]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"(?i)\b(?:api[_ -]?key|secret[_ -]?key)\s*[:=]\s*[A-Za-z0-9_./+-]{16,}"),
)
_ALLOWED_TEXT_HOSTS = {"tile.loc.gov", "tiles.loc.gov", "www.loc.gov"}
_EXPECTED_ROW_KEYS = {"id", "text", "source", "added", "metadata"}
_EXPECTED_METADATA_KEYS = {
    "license",
    "title",
    "author",
    "year",
    "language",
    "item_url",
    "text_file_url",
}


class LocBooksIntakeError(ValueError):
    """Fail-closed Library of Congress intake contract violation."""


@dataclass(frozen=True)
class Decision:
    status: str
    reason: str
    text: str | None = None
    text_sha256: str | None = None
    utf8_bytes: int = 0


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def self_identity(value: Mapping[str, Any], field: str) -> str:
    clone = deepcopy(dict(value))
    clone.pop(field, None)
    return sha256_bytes((canonical_json(clone) + "\n").encode("utf-8"))


def _require_hex(value: object, length: int, field: str) -> str:
    if not isinstance(value, str):
        raise LocBooksIntakeError(f"{field} must be a string")
    pattern = _HEX40_RE if length == 40 else _HEX64_RE
    if not pattern.fullmatch(value):
        raise LocBooksIntakeError(f"{field} must be lowercase {length}-hex")
    return value


def validate_config(config: Mapping[str, Any]) -> None:
    expected = {
        "schema_version": "12-6.d03-loc-books-intake.v1",
        "source_family": "en.books.loc.selected-digitized.public-domain",
        "execution_profile": "LOCAL_FREE",
        "training_authorized_bytes": 0,
        "authorized_unique_loss_positions": 0,
        "canonical_capacity_credit_bytes": 0,
        "corpus_admitted": False,
        "tokenizer_fit_permitted": False,
        "model_training_permitted": False,
        "evaluation_eligible": False,
        "paid_compute_authorized": False,
    }
    for field, expected_value in expected.items():
        if config.get(field) != expected_value:
            raise LocBooksIntakeError(f"{field} drift")

    upstream = config.get("upstream")
    if not isinstance(upstream, Mapping):
        raise LocBooksIntakeError("upstream missing")
    if upstream.get("dataset") != "common-pile/library_of_congress":
        raise LocBooksIntakeError("upstream.dataset drift")
    revision = _require_hex(upstream.get("revision"), 40, "upstream.revision")
    shard_sha = _require_hex(upstream.get("shard_sha256"), 64, "upstream.shard_sha256")
    shard_path = upstream.get("shard_path")
    if shard_path != "data/00000_loc_books.jsonl.gz":
        raise LocBooksIntakeError("upstream.shard_path drift")
    if upstream.get("shard_compressed_bytes") != 358_594_502:
        raise LocBooksIntakeError("upstream.shard_compressed_bytes drift")
    expected_url = (
        "https://huggingface.co/datasets/common-pile/library_of_congress/resolve/"
        f"{revision}/{shard_path}?download=true"
    )
    if upstream.get("pinned_url") != expected_url:
        raise LocBooksIntakeError("upstream.pinned_url drift")
    if shard_sha != "a6a6023d9cae067b5531ba84537877c180ab3ff1579f99aeac446d65d465ae0f":
        raise LocBooksIntakeError("upstream shard authority drift")

    audit = config.get("common_pile_audit")
    if not isinstance(audit, Mapping):
        raise LocBooksIntakeError("common_pile_audit missing")
    audit_expected = {
        "registry_path": "configs/data/common_pile_source_rights_v1.json",
        "registry_blob_sha1": "7b4d6828288672bf25c551e85a5d7f7399e8ef0f",
        "source_key": "library_of_congress",
        "audited_collector_path": "sources/loc_books",
        "audited_code_revision": "9457f04a14cb2355ab00023420369d46ffd4a395",
        "collector_export_blob_sha1": "0f8633d9a0c12bd04140420d74a95d82a065119f",
        "collector_metadata_blob_sha1": "a219f88cbe8f6cc33bd72eed6c9d75958a8efb08",
    }
    for field, expected_value in audit_expected.items():
        if audit.get(field) != expected_value:
            raise LocBooksIntakeError(f"common_pile_audit.{field} drift")

    policy = config.get("selection_policy")
    if not isinstance(policy, Mapping):
        raise LocBooksIntakeError("selection_policy missing")
    integer_fields = (
        "max_accepted_documents",
        "max_examined_records",
        "max_total_text_bytes",
        "stop_after_min_text_bytes",
        "min_record_text_bytes",
        "max_record_text_bytes",
        "max_uncompressed_prefix_bytes",
    )
    for field in integer_fields:
        value = policy.get(field)
        if not isinstance(value, int) or value <= 0:
            raise LocBooksIntakeError(f"invalid selection_policy.{field}")
    if policy["max_total_text_bytes"] >= 5_000_000:
        raise LocBooksIntakeError("family planning cap must stay below 5,000,000 bytes")
    if policy["stop_after_min_text_bytes"] > policy["max_total_text_bytes"]:
        raise LocBooksIntakeError("stop target above byte cap")
    if policy.get("ordering") != "SOURCE_STREAM_ORDER":
        raise LocBooksIntakeError("selection ordering drift")
    if policy.get("normalization") != "STRICT_UTF8_NFC_LF_OUTER_TRIM":
        raise LocBooksIntakeError("normalization drift")

    gates = config.get("required_downstream_gates")
    if gates != [
        "global_exact_near_fragment_lineage_dedup",
        "reserved_evaluation_decontamination",
        "quality_privacy",
        "balance_family_caps",
        "cluster_safe_split",
        "deterministic_pack_two_clean_builds",
        "positive_unique_loss_ledger",
    ]:
        raise LocBooksIntakeError("required downstream gates drift")

    identity = _require_hex(config.get("contract_identity_sha256"), 64, "contract identity")
    if self_identity(config, "contract_identity_sha256") != identity:
        raise LocBooksIntakeError("contract identity mismatch")


def _strict_https(url: object, hosts: set[str]) -> bool:
    if not isinstance(url, str):
        return False
    parsed = urlparse(url)
    return (
        parsed.scheme == "https"
        and parsed.hostname in hosts
        and parsed.username is None
        and parsed.password is None
        and parsed.port in (None, 443)
        and bool(parsed.path)
        and not parsed.fragment
    )


def _normalize_text(text: object) -> Decision:
    if not isinstance(text, str):
        return Decision("REJECT", "text_not_string")
    text = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n")).strip()
    if not text:
        return Decision("REJECT", "empty_text")
    for char in text:
        codepoint = ord(char)
        if char not in "\n\t" and (codepoint < 32 or 127 <= codepoint <= 159):
            return Decision("REJECT", "disallowed_control_character")
    if _EMAIL_RE.search(text):
        return Decision("REJECT", "obvious_contact_email")
    if any(pattern.search(text) for pattern in _SECRET_RES):
        return Decision("REJECT", "obvious_secret_marker")
    encoded = text.encode("utf-8")
    return Decision(
        "NORMALIZED",
        "ok",
        text=text,
        text_sha256=sha256_bytes(encoded),
        utf8_bytes=len(encoded),
    )


def classify_row(config: Mapping[str, Any], row: object) -> Decision:
    validate_config(config)
    if not isinstance(row, Mapping):
        return Decision("REJECT", "row_not_object")
    if set(row) != _EXPECTED_ROW_KEYS:
        return Decision("REJECT", "row_schema_drift")
    record_id = row.get("id")
    if not isinstance(record_id, str) or not _RECORD_ID_RE.fullmatch(record_id):
        return Decision("REJECT", "invalid_record_id")
    if row.get("source") != "loc_books":
        return Decision("REJECT", "source_drift")
    added = row.get("added")
    if not isinstance(added, str) or "T" not in added:
        return Decision("REJECT", "invalid_added_timestamp")

    metadata = row.get("metadata")
    if not isinstance(metadata, Mapping) or set(metadata) != _EXPECTED_METADATA_KEYS:
        return Decision("REJECT", "metadata_schema_drift")
    if metadata.get("license") != "Public Domain":
        return Decision("REJECT", "license_not_public_domain")
    if metadata.get("language") != "english":
        return Decision("REJECT", "language_not_english")
    if not isinstance(metadata.get("title"), str) or not metadata["title"].strip():
        return Decision("REJECT", "missing_title")
    if not isinstance(metadata.get("author"), str) or not metadata["author"].strip():
        return Decision("REJECT", "missing_author")
    year = metadata.get("year")
    if not isinstance(year, int) or isinstance(year, bool) or not 1500 <= year <= 2024:
        return Decision("REJECT", "invalid_publication_year")
    item_url = metadata.get("item_url")
    if not _strict_https(item_url, {"www.loc.gov"}):
        return Decision("REJECT", "invalid_loc_item_url")
    if urlparse(item_url).path.rstrip("/") != f"/item/{record_id}":
        return Decision("REJECT", "item_url_id_mismatch")
    if not _strict_https(metadata.get("text_file_url"), _ALLOWED_TEXT_HOSTS):
        return Decision("REJECT", "invalid_loc_text_url")

    normalized = _normalize_text(row.get("text"))
    if normalized.status != "NORMALIZED":
        return normalized
    policy = config["selection_policy"]
    if normalized.utf8_bytes < policy["min_record_text_bytes"]:
        return Decision("REJECT", "text_too_short")
    if normalized.utf8_bytes > policy["max_record_text_bytes"]:
        return Decision("REJECT", "text_too_large")
    assert normalized.text is not None
    if sum(char.isalpha() for char in normalized.text) < 200:
        return Decision("REJECT", "insufficient_alphabetic_content")
    return Decision(
        "ACCEPT",
        "per_record_public_domain_loc_provenance",
        text=normalized.text,
        text_sha256=normalized.text_sha256,
        utf8_bytes=normalized.utf8_bytes,
    )


def _identity(rows: Sequence[Mapping[str, Any]]) -> str:
    return sha256_bytes((canonical_json(list(rows)) + "\n").encode("utf-8"))


def materialize_from_gzip_stream(
    config: Mapping[str, Any],
    compressed_stream: BinaryIO,
    *,
    full_shard_sha256_verified: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    validate_config(config)
    policy = config["selection_policy"]
    if full_shard_sha256_verified:
        raise LocBooksIntakeError(
            "streaming prefix cannot assert full-shard SHA verification; "
            "use a future full-file verifier"
        )

    candidates: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    rejection_counts: Counter[str] = Counter()
    seen_hashes: set[str] = set()
    accepted_bytes = 0
    uncompressed_bytes = 0
    examined = 0

    try:
        gz = gzip.GzipFile(fileobj=compressed_stream, mode="rb")
        for raw_line in gz:
            uncompressed_bytes += len(raw_line)
            if uncompressed_bytes > policy["max_uncompressed_prefix_bytes"]:
                break
            if examined >= policy["max_examined_records"]:
                break
            examined += 1
            try:
                line = raw_line.decode("utf-8")
            except UnicodeDecodeError:
                rejection_counts["jsonl_not_strict_utf8"] += 1
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                rejection_counts["malformed_json"] += 1
                continue
            decision = classify_row(config, row)
            record_id = row.get("id") if isinstance(row, Mapping) else None
            evidence_row: dict[str, Any] = {
                "stream_record_index": examined - 1,
                "record_id": record_id if isinstance(record_id, str) else None,
                "status": decision.status,
                "reason": decision.reason,
            }
            if decision.status != "ACCEPT":
                rejection_counts[decision.reason] += 1
                evidence.append(evidence_row)
                continue
            assert decision.text is not None and decision.text_sha256 is not None
            if decision.text_sha256 in seen_hashes:
                rejection_counts["exact_normalized_duplicate"] += 1
                evidence_row["status"] = "REJECT"
                evidence_row["reason"] = "exact_normalized_duplicate"
                evidence_row["text_sha256"] = decision.text_sha256
                evidence.append(evidence_row)
                continue
            if accepted_bytes + decision.utf8_bytes > policy["max_total_text_bytes"]:
                rejection_counts["family_byte_cap"] += 1
                evidence_row["status"] = "REJECT"
                evidence_row["reason"] = "family_byte_cap"
                evidence_row["text_sha256"] = decision.text_sha256
                evidence.append(evidence_row)
                continue

            seen_hashes.add(decision.text_sha256)
            accepted_bytes += decision.utf8_bytes
            metadata = row["metadata"]
            candidate = {
                "record_id": f"loc:{row['id']}",
                "source_family": config["source_family"],
                "source_revision": config["upstream"]["revision"],
                "source_shard": config["upstream"]["shard_path"],
                "source_record_id": row["id"],
                "item_url": metadata["item_url"],
                "text_file_url": metadata["text_file_url"],
                "text": decision.text,
                "text_sha256": decision.text_sha256,
                "utf8_bytes": decision.utf8_bytes,
                "training_eligible": False,
                "evaluation_eligible": False,
            }
            candidates.append(candidate)
            evidence_row.update(
                {
                    "text_sha256": decision.text_sha256,
                    "utf8_bytes": decision.utf8_bytes,
                }
            )
            evidence.append(evidence_row)
            if len(candidates) >= policy["max_accepted_documents"]:
                break
            if accepted_bytes >= policy["stop_after_min_text_bytes"]:
                break
    except (OSError, EOFError) as exc:
        # A deliberately stopped prefix must not hit this path. Actual gzip corruption before
        # the selected boundary is a hard failure, not a quarantined record.
        raise LocBooksIntakeError("gzip stream failed before bounded prefix completed") from exc

    if not candidates:
        raise LocBooksIntakeError("bounded materialization retained zero candidates")

    inventory_rows = [
        {
            "record_id": row["record_id"],
            "text_sha256": row["text_sha256"],
            "utf8_bytes": row["utf8_bytes"],
        }
        for row in candidates
    ]
    evidence_rows = sorted(
        evidence,
        key=lambda row: (row["stream_record_index"], str(row["record_id"])),
    )
    report = {
        "schema_version": "12-6.d03-loc-books-materialization.v1",
        "status": "CANDIDATE_MATERIALIZED_ZERO_CREDIT",
        "source_family": config["source_family"],
        "upstream_dataset": config["upstream"]["dataset"],
        "upstream_revision": config["upstream"]["revision"],
        "upstream_shard_path": config["upstream"]["shard_path"],
        "upstream_shard_expected_sha256": config["upstream"]["shard_sha256"],
        "upstream_shard_expected_compressed_bytes": config["upstream"][
            "shard_compressed_bytes"
        ],
        "full_shard_sha256_verified": False,
        "transport_scope": "IMMUTABLE_PINNED_REVISION_BOUNDED_GZIP_PREFIX",
        "contract_identity_sha256": config["contract_identity_sha256"],
        "examined_records": examined,
        "uncompressed_prefix_bytes_observed": uncompressed_bytes,
        "accepted_documents": len(candidates),
        "accepted_normalized_utf8_bytes": accepted_bytes,
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "candidate_inventory_identity_sha256": _identity(inventory_rows),
        "selection_evidence_identity_sha256": _identity(evidence_rows),
        "training_authorized_bytes": 0,
        "authorized_unique_loss_positions": 0,
        "corpus_admitted": False,
        "tokenizer_fit_permitted": False,
        "model_training_permitted": False,
        "evaluation_eligible": False,
        "final_test_payload_accessed": False,
        "paid_compute_authorized": False,
        "required_downstream_gates": list(config["required_downstream_gates"]),
    }
    report["report_identity_sha256"] = self_identity(report, "report_identity_sha256")
    return candidates, report


def materialize_from_gzip_bytes(
    config: Mapping[str, Any], compressed: bytes
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return materialize_from_gzip_stream(config, io.BytesIO(compressed))
