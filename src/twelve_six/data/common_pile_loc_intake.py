from __future__ import annotations

import gzip
import hashlib
import io
import json
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

_HEX40_RE = re.compile(r"^[0-9a-f]{40}$")
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_EMAIL_RE = re.compile(r"(?i)(?<![\w.+-])[\w.+-]+@[a-z0-9.-]+\.[a-z]{2,}(?![\w.-])")
_SECRET_RE = re.compile(
    r"(?i)(?:-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|"
    r"(?:api[_ -]?key|access[_ -]?token|password)\s*[:=]\s*\S+)"
)
_ALLOWED_TEXT_CONTROLS = {"\n", "\r", "\t"}
_ALLOWED_TILE_HOSTS = {"tile.loc.gov", "tiles.loc.gov"}
_EXPECTED_AUDIT = {
    "registry_path": "configs/data/common_pile_source_rights_v1.json",
    "registry_blob_sha1": "7b4d6828288672bf25c551e85a5d7f7399e8ef0f",
    "source_key": "library_of_congress",
    "audited_code_revision": "9457f04a14cb2355ab00023420369d46ffd4a395",
    "collector_path": "sources/loc_books/books.py",
    "collector_blob_sha1": "0f8633d9a0c12bd04140420d74a95d82a065119f",
}
_EXPECTED_UPSTREAM = {
    "dataset": "common-pile/library_of_congress",
    "revision": "d31bdba02cdad5104ccec2c02ae799c0bcb5a9a7",
    "shard_path": "data/00000_loc_books.jsonl.gz",
    "shard_lfs_sha256": (
        "a6a6023d9cae067b5531ba84537877c180ab3ff1579f99aeac446d65d465ae0f"
    ),
    "shard_compressed_bytes": 358594502,
    "source_name": "loc_books",
}
_EXPECTED_RIGHTS = {
    "dataset_package_license_is_training_authority": False,
    "expected_language": "english",
    "expected_license": "Public Domain",
    "legal_conclusion_claimed": False,
    "official_rights_url": (
        "https://www.loc.gov/collections/selected-digitized-books/"
        "about-this-collection/rights-and-access/"
    ),
    "require_exact_item_url": True,
    "require_loc_text_file_url": True,
    "source_collection": "Library of Congress Selected Digitized Books",
}
_EXPECTED_SELECTION_POLICY = {
    "max_documents": 128,
    "max_examined_documents": 256,
    "max_jsonl_line_bytes": 4_194_304,
    "max_remote_compressed_prefix_bytes": 33_554_432,
    "max_single_normalized_utf8_bytes": 1_000_000,
    "max_total_normalized_utf8_bytes": 4_500_000,
    "min_single_normalized_utf8_bytes": 1_024,
    "ordering": "SOURCE_ORDER",
}
_REQUIRED_GATES = [
    "privacy",
    "quality",
    "global_exact_near_fragment_lineage_dedup",
    "reserved_evaluation_decontamination",
    "balance_family_caps",
    "cluster_safe_split",
    "deterministic_pack_two_clean_builds",
    "positive_unique_loss_ledger",
]
_ROOT_KEYS = {
    "authorized_unique_loss_positions",
    "canonical_capacity_credit_bytes",
    "common_pile_audit",
    "contract_identity_sha256",
    "corpus_admitted",
    "evaluation_eligible",
    "execution_profile",
    "final_test_payload_accessed",
    "model_training_permitted",
    "paid_compute_authorized",
    "required_downstream_gates",
    "rights_policy",
    "schema_version",
    "selection_policy",
    "source_family",
    "tokenizer_fit_permitted",
    "training_authorized_bytes",
    "upstream",
}


class LocIntakeError(ValueError):
    """Fail-closed violation in the bounded Library of Congress source intake."""


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def self_identity(value: Mapping[str, Any], field: str) -> str:
    clone = deepcopy(dict(value))
    clone.pop(field, None)
    return sha256_bytes((canonical_json(clone) + "\n").encode("utf-8"))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LocIntakeError(message)


def _require_hex(value: object, length: int, field: str) -> str:
    _require(isinstance(value, str), f"{field} must be a string")
    pattern = _HEX40_RE if length == 40 else _HEX64_RE
    _require(bool(pattern.fullmatch(value)), f"{field} must be lowercase {length}-hex")
    return value


def _require_exact_scalar(actual: object, expected: object, field: str) -> None:
    _require(type(actual) is type(expected) and actual == expected, f"{field} drift")


def _require_exact_mapping(
    actual: object,
    expected: Mapping[str, object],
    field: str,
) -> Mapping[str, object]:
    _require(isinstance(actual, Mapping), f"{field} missing")
    _require(set(actual) == set(expected), f"{field} schema drift")
    for key, wanted in expected.items():
        _require_exact_scalar(actual[key], wanted, f"{field} {key}")
    return actual


def validate_config(config: Mapping[str, Any]) -> None:
    _require(set(config) == _ROOT_KEYS, "config schema drift")
    expected = {
        "schema_version": "12-6.d03-common-pile-loc-intake.v1",
        "source_family": "en.loc.selected-digitized-books",
        "execution_profile": "LOCAL_FREE",
        "training_authorized_bytes": 0,
        "authorized_unique_loss_positions": 0,
        "canonical_capacity_credit_bytes": 0,
        "corpus_admitted": False,
        "tokenizer_fit_permitted": False,
        "model_training_permitted": False,
        "evaluation_eligible": False,
        "paid_compute_authorized": False,
        "final_test_payload_accessed": False,
    }
    for field, value in expected.items():
        _require_exact_scalar(config.get(field), value, field)

    audit = _require_exact_mapping(
        config.get("common_pile_audit"),
        _EXPECTED_AUDIT,
        "common_pile_audit",
    )
    _require_hex(audit.get("registry_blob_sha1"), 40, "common_pile registry blob")
    _require_hex(audit.get("audited_code_revision"), 40, "audited code revision")
    _require_hex(audit.get("collector_blob_sha1"), 40, "collector blob")

    upstream = _require_exact_mapping(
        config.get("upstream"),
        {
            **_EXPECTED_UPSTREAM,
            "resolve_url": (
                "https://huggingface.co/datasets/common-pile/library_of_congress/"
                "resolve/d31bdba02cdad5104ccec2c02ae799c0bcb5a9a7/"
                "data/00000_loc_books.jsonl.gz"
            ),
        },
        "upstream",
    )
    _require_hex(upstream.get("revision"), 40, "upstream revision")
    _require_hex(upstream.get("shard_lfs_sha256"), 64, "shard LFS sha256")
    size = upstream.get("shard_compressed_bytes")
    _require(
        isinstance(size, int) and not isinstance(size, bool) and size > 0,
        "invalid shard size",
    )

    _require_exact_mapping(config.get("rights_policy"), _EXPECTED_RIGHTS, "rights_policy")
    _require_exact_mapping(
        config.get("selection_policy"),
        _EXPECTED_SELECTION_POLICY,
        "selection_policy",
    )
    _require(config.get("required_downstream_gates") == _REQUIRED_GATES, "required gates drift")
    identity = _require_hex(config.get("contract_identity_sha256"), 64, "contract identity")
    _require(
        self_identity(config, "contract_identity_sha256") == identity,
        "contract identity mismatch",
    )


def _valid_https(url: object, hosts: set[str]) -> bool:
    if not isinstance(url, str):
        return False
    parts = urlsplit(url)
    return (
        parts.scheme == "https"
        and parts.hostname in hosts
        and parts.username is None
        and parts.password is None
        and not parts.fragment
    )


def normalize_text(text: object) -> str:
    _require(isinstance(text, str), "text must be a string")
    line_normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = unicodedata.normalize("NFC", line_normalized).strip()
    _require(bool(normalized), "text is empty")
    for char in normalized:
        if unicodedata.category(char) == "Cc" and char not in _ALLOWED_TEXT_CONTROLS:
            raise LocIntakeError("text contains forbidden control characters")
    return normalized


def _record_identity(
    record: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[str, Mapping[str, Any]]:
    _require(record.get("source") == config["upstream"]["source_name"], "source drift")
    record_id = record.get("id")
    _require(isinstance(record_id, str) and 2 <= len(record_id) <= 128, "invalid record id")
    _require(
        record_id.strip() == record_id and not any(ch.isspace() for ch in record_id),
        "record id whitespace",
    )
    metadata = record.get("metadata")
    _require(isinstance(metadata, Mapping), "metadata missing")
    rights = config["rights_policy"]
    _require(
        metadata.get("license") == rights["expected_license"],
        "record license is not Public Domain",
    )
    _require(
        metadata.get("language") == rights["expected_language"],
        "record language is not english",
    )
    year = metadata.get("year")
    _require(
        isinstance(year, int) and not isinstance(year, bool) and 1500 <= year <= 2024,
        "invalid year",
    )
    expected_item_url = f"https://www.loc.gov/item/{record_id}"
    _require(metadata.get("item_url") == expected_item_url, "item_url does not bind record id")
    _require(_valid_https(metadata.get("item_url"), {"www.loc.gov"}), "invalid LoC item URL")
    _require(
        _valid_https(metadata.get("text_file_url"), _ALLOWED_TILE_HOSTS),
        "invalid LoC text file URL",
    )
    return record_id, metadata


def _quality_reason(text: str, config: Mapping[str, Any]) -> str | None:
    raw = text.encode("utf-8")
    selection = config["selection_policy"]
    if len(raw) < selection["min_single_normalized_utf8_bytes"]:
        return "too_short"
    if len(raw) > selection["max_single_normalized_utf8_bytes"]:
        return "too_large"
    if _EMAIL_RE.search(text):
        return "email_like_contact"
    if _SECRET_RE.search(text):
        return "secret_like_text"
    visible = [ch for ch in text if not ch.isspace()]
    alpha = sum(ch.isalpha() for ch in visible)
    if not visible or alpha < 100 or alpha / len(visible) < 0.20:
        return "low_alpha_content"
    return None


def _inventory_identity(rows: Sequence[Mapping[str, Any]]) -> str:
    return sha256_bytes((canonical_json(list(rows)) + "\n").encode("utf-8"))


def materialize(
    config: Mapping[str, Any],
    records: Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Materialize an arbitrary record stream without claiming full-shard verification."""
    validate_config(config)
    selection = config["selection_policy"]
    max_docs = selection["max_documents"]
    max_examined = selection["max_examined_documents"]
    max_total = selection["max_total_normalized_utf8_bytes"]

    candidates: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    rejection_counts: Counter[str] = Counter()
    seen_ids: set[str] = set()
    seen_text_hashes: set[str] = set()
    accepted_bytes = 0
    examined = 0

    for record in records:
        if examined >= max_examined or len(candidates) >= max_docs or accepted_bytes >= max_total:
            break
        _require(isinstance(record, Mapping), "JSONL row must be an object")
        examined += 1
        record_id, metadata = _record_identity(record, config)
        _require(record_id not in seen_ids, f"duplicate record id: {record_id}")
        seen_ids.add(record_id)
        text = normalize_text(record.get("text"))
        text_bytes = text.encode("utf-8")
        text_hash = sha256_bytes(text_bytes)
        reason = _quality_reason(text, config)
        if reason is None and text_hash in seen_text_hashes:
            reason = "exact_normalized_duplicate"
        if reason is None and accepted_bytes + len(text_bytes) > max_total:
            reason = "family_byte_cap"

        evidence_row = {
            "record_id": record_id,
            "item_url": metadata["item_url"],
            "text_file_url": metadata["text_file_url"],
            "text_sha256": text_hash,
            "normalized_utf8_bytes": len(text_bytes),
            "decision": "ACCEPT" if reason is None else "REJECT",
            "reason": "candidate_zero_credit" if reason is None else reason,
        }
        evidence.append(evidence_row)
        if reason is not None:
            rejection_counts[reason] += 1
            continue

        seen_text_hashes.add(text_hash)
        candidate = {
            "record_id": f"loc:{record_id}",
            "source_family": config["source_family"],
            "source_revision": config["upstream"]["revision"],
            "source_shard_path": config["upstream"]["shard_path"],
            "source_shard_lfs_sha256": config["upstream"]["shard_lfs_sha256"],
            "source_record_id": record_id,
            "item_url": metadata["item_url"],
            "text_file_url": metadata["text_file_url"],
            "text": text,
            "text_sha256": text_hash,
            "utf8_bytes": len(text_bytes),
            "training_eligible": False,
            "evaluation_eligible": False,
        }
        candidates.append(candidate)
        accepted_bytes += len(text_bytes)

    _require(bool(candidates), "bounded materialization retained zero candidates")
    inventory_rows = [
        {
            "record_id": row["record_id"],
            "source_record_id": row["source_record_id"],
            "item_url": row["item_url"],
            "text_file_url": row["text_file_url"],
            "text_sha256": row["text_sha256"],
            "utf8_bytes": row["utf8_bytes"],
        }
        for row in candidates
    ]
    report = {
        "schema_version": "12-6.d03-common-pile-loc-materialization.v1",
        "status": "CANDIDATE_MATERIALIZED_ZERO_CREDIT",
        "source_family": config["source_family"],
        "contract_identity_sha256": config["contract_identity_sha256"],
        "upstream_revision": config["upstream"]["revision"],
        "source_shard_path": config["upstream"]["shard_path"],
        "source_shard_lfs_sha256": config["upstream"]["shard_lfs_sha256"],
        "full_shard_hash_verified": False,
        "examined_documents": examined,
        "accepted_documents": len(candidates),
        "accepted_normalized_utf8_bytes": accepted_bytes,
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "candidate_inventory_identity_sha256": _inventory_identity(inventory_rows),
        "rights_provenance_evidence_identity_sha256": _inventory_identity(evidence),
        "training_authorized_bytes": 0,
        "authorized_unique_loss_positions": 0,
        "canonical_capacity_credit_bytes": 0,
        "corpus_admitted": False,
        "evaluation_eligible": False,
        "privacy_gate": "NOT_RUN",
        "quality_gate": "NOT_RUN",
        "global_dedup_gate": "NOT_RUN",
        "reserved_evaluation_decontamination_gate": "NOT_RUN",
        "balance_family_caps_gate": "NOT_RUN",
        "cluster_safe_split_gate": "NOT_RUN",
        "packing_two_clean_builds_gate": "NOT_RUN",
        "positive_unique_loss_ledger_gate": "NOT_RUN",
        "model_training_executed": False,
        "final_test_payload_accessed": False,
        "paid_compute_used": False,
    }
    return candidates, report


def _verify_full_shard_bytes(raw_gzip: bytes, config: Mapping[str, Any]) -> None:
    validate_config(config)
    _require(type(raw_gzip) is bytes and bool(raw_gzip), "full shard bytes are empty")
    upstream = config["upstream"]
    _require(
        len(raw_gzip) == upstream["shard_compressed_bytes"],
        "full shard size mismatch",
    )
    _require(
        sha256_bytes(raw_gzip) == upstream["shard_lfs_sha256"],
        "full shard SHA-256 mismatch",
    )


def verify_full_shard(path: Path, config: Mapping[str, Any]) -> bytes:
    """Read a shard once and return the exact verified immutable byte snapshot."""
    raw_gzip = path.read_bytes()
    _verify_full_shard_bytes(raw_gzip, config)
    return raw_gzip


def materialize_verified_shard(
    config: Mapping[str, Any],
    raw_gzip: bytes,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Verify and materialize the same immutable full-shard byte snapshot."""
    _verify_full_shard_bytes(raw_gzip, config)
    selection = config["selection_policy"]
    records = iter_gzip_jsonl_bytes(
        raw_gzip,
        max_jsonl_line_bytes=selection["max_jsonl_line_bytes"],
        skip_oversize_lines=True,
    )
    candidates, report = materialize(config, records)
    report["full_shard_hash_verified"] = True
    return candidates, report


def _drain_oversized_line(handle: gzip.GzipFile, max_jsonl_line_bytes: int) -> None:
    while True:
        chunk = handle.readline(max_jsonl_line_bytes + 1)
        if not chunk or chunk.endswith(b"\n"):
            return


def iter_gzip_jsonl_bytes(
    raw_gzip: bytes,
    *,
    max_jsonl_line_bytes: int,
    skip_oversize_lines: bool = False,
) -> Iterable[dict[str, Any]]:
    _require(isinstance(raw_gzip, bytes) and bool(raw_gzip), "gzip input is empty")
    _require(
        isinstance(max_jsonl_line_bytes, int)
        and not isinstance(max_jsonl_line_bytes, bool)
        and max_jsonl_line_bytes > 0,
        "invalid JSONL line bound",
    )
    _require(type(skip_oversize_lines) is bool, "skip_oversize_lines must be an exact bool")
    with gzip.GzipFile(fileobj=io.BytesIO(raw_gzip), mode="rb") as handle:
        while True:
            raw_line = handle.readline(max_jsonl_line_bytes + 1)
            if not raw_line:
                return
            if len(raw_line) > max_jsonl_line_bytes:
                if not skip_oversize_lines:
                    raise LocIntakeError("JSONL line exceeds safety bound")
                if not raw_line.endswith(b"\n"):
                    _drain_oversized_line(handle, max_jsonl_line_bytes)
                continue
            _require(raw_line.endswith(b"\n"), "truncated JSONL line")
            try:
                line = raw_line.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise LocIntakeError("JSONL is not strict UTF-8") from exc
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise LocIntakeError("malformed JSONL row") from exc
            _require(isinstance(value, dict), "JSONL row must be an object")
            yield value


def write_materialization(
    output_dir: Path,
    candidates: Sequence[Mapping[str, Any]],
    report: Mapping[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [canonical_json(dict(row)) for row in candidates]
    (output_dir / "loc-candidate.jsonl").write_text(
        "\n".join(rows) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (output_dir / "loc-report.json").write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
