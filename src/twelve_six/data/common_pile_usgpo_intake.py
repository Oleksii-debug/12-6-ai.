from __future__ import annotations

import gzip
import hashlib
import io
import json
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

_CONTRACT_ID = "b514d417b722481e0f77dde485d1b38047577519600fae94a6e19f704cc70a7c"
_REVISION = "b1685d7dc7e3e71a62ec6777ed5e637b184f5f6c"
_SHARD_SHA256 = "ae0b8573ce72853590ca7a84a7b2dcf74bd984031694012d2cb6a89309033563"
_COLLECTOR_SHA1 = "073095cd64a9d9d1ddffe8e853720b0eb410988e"
_REGISTRY_SHA1 = "7b4d6828288672bf25c551e85a5d7f7399e8ef0f"
_ROW_KEYS = {"id", "created", "text", "source", "added", "metadata"}
_METADATA_KEYS = {"title", "author", "publisher", "category", "license", "url"}
_PACKAGE_ID_RE = re.compile(r"^[A-Za-z0-9._-]{2,200}$")
_EMAIL_RE = re.compile(r"(?i)(?<![\w.+-])[\w.+-]+@[a-z0-9.-]+\.[a-z]{2,}(?![\w.-])")
_PHONE_RE = re.compile(
    r"(?x)(?<!\d)(?:\+?1[\s.()-]*)?(?:\(?\d{3}\)?[\s.-]*)\d{3}[\s.-]*\d{4}(?!\d)"
)
_SECRET_RE = re.compile(
    r"(?i)(?:-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|"
    r"(?:api[_ -]?key|access[_ -]?token|password|client[_ -]?secret)\s*[:=]\s*\S+)"
)
_ALLOWED_CONTROLS = {"\n", "\r", "\t"}
_GOVINFO_HOSTS = {"govinfo.gov", "www.govinfo.gov"}


class UsgpoIntakeError(ValueError):
    """Fail-closed violation in the bounded USGPO candidate intake."""


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def self_identity(value: Mapping[str, Any], field: str) -> str:
    clone = deepcopy(dict(value))
    clone.pop(field, None)
    return sha256_bytes((canonical_json(clone) + "\n").encode("utf-8"))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise UsgpoIntakeError(message)


def validate_config(config: Mapping[str, Any]) -> None:
    # Specific authority checks precede the sealed identity so failures identify the drift.
    _require(config.get("training_authorized_bytes") == 0, "training_authorized_bytes drift")
    _require(
        config.get("authorized_unique_loss_positions") == 0,
        "authorized_unique_loss_positions drift",
    )
    _require(
        config.get("canonical_capacity_credit_bytes") == 0,
        "canonical_capacity_credit_bytes drift",
    )
    for field in (
        "corpus_admitted",
        "tokenizer_fit_permitted",
        "model_training_permitted",
        "evaluation_eligible",
        "paid_compute_authorized",
        "final_test_payload_accessed",
    ):
        _require(config.get(field) is False, f"{field} drift")

    rights = config.get("rights_policy")
    _require(isinstance(rights, Mapping), "rights policy missing")
    _require(rights.get("legal_conclusion_claimed") is False, "legal_conclusion_claimed drift")
    _require(rights.get("project_review_status") == "REVIEW_REQUIRED", "review status drift")
    _require(rights.get("source_item_verification_required") is True, "item review drift")
    _require(rights.get("expected_license") == "Public Domain", "expected license drift")
    _require(rights.get("expected_publisher") == "gpo", "expected publisher drift")

    audit = config.get("common_pile_audit")
    _require(isinstance(audit, Mapping), "audit missing")
    _require(audit.get("collector_blob_sha1") == _COLLECTOR_SHA1, "collector_blob_sha1 drift")
    _require(audit.get("registry_blob_sha1") == _REGISTRY_SHA1, "registry_blob_sha1 drift")
    _require(audit.get("source_key") == "usgpo", "source key drift")

    upstream = config.get("upstream")
    _require(isinstance(upstream, Mapping), "upstream missing")
    _require(upstream.get("revision") == _REVISION, "revision drift")
    _require(upstream.get("shard_lfs_sha256") == _SHARD_SHA256, "shard sha256 drift")
    _require(upstream.get("shard_compressed_bytes") == 208_278_377, "shard size drift")
    _require(upstream.get("source_name") == "usgpo", "source name drift")

    required = config.get("required_downstream_gates")
    _require(isinstance(required, list), "required gates missing")
    _require("item_level_rights_review" in required, "required gates drift")
    _require("positive_unique_loss_ledger" in required, "required gates drift")

    policy = config.get("selection_policy")
    _require(isinstance(policy, Mapping), "selection policy missing")
    _require(
        policy.get("max_total_normalized_utf8_bytes") == 5_000_000,
        "max_total_normalized_utf8_bytes drift",
    )
    _require(policy.get("ordering") == "SOURCE_ORDER", "ordering drift")

    identity = config.get("contract_identity_sha256")
    _require(identity == _CONTRACT_ID, "contract identity drift")
    _require(
        self_identity(config, "contract_identity_sha256") == _CONTRACT_ID,
        "contract identity mismatch",
    )


def verify_transport(config: Mapping[str, Any], path: Path) -> dict[str, object]:
    validate_config(config)
    _require(path.is_file(), f"source shard not found: {path}")
    actual_size = path.stat().st_size
    _require(
        actual_size == config["upstream"]["shard_compressed_bytes"],
        "source shard compressed size mismatch",
    )
    actual_sha = sha256_file(path)
    _require(actual_sha == config["upstream"]["shard_lfs_sha256"], "source shard sha256 mismatch")
    return {"compressed_bytes": actual_size, "sha256": actual_sha, "verified": True}


def _iter_jsonl_stream(
    stream: Any,
    *,
    max_jsonl_line_bytes: int,
    max_decompressed_jsonl_bytes: int | None,
) -> Iterator[dict[str, Any]]:
    total = 0
    line_number = 0
    while True:
        raw = stream.readline(max_jsonl_line_bytes + 1)
        if not raw:
            return
        line_number += 1
        total += len(raw)
        _require(
            len(raw) <= max_jsonl_line_bytes,
            f"JSONL line exceeds {max_jsonl_line_bytes} bytes at line {line_number}",
        )
        if max_decompressed_jsonl_bytes is not None:
            _require(
                total <= max_decompressed_jsonl_bytes,
                "decompressed JSONL scan budget exceeded",
            )
        try:
            decoded = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise UsgpoIntakeError(f"invalid UTF-8 at JSONL line {line_number}") from exc
        try:
            row = json.loads(decoded)
        except json.JSONDecodeError as exc:
            raise UsgpoIntakeError(f"malformed JSONL at line {line_number}") from exc
        _require(isinstance(row, dict), f"JSONL row {line_number} must be an object")
        yield row


def iter_gzip_jsonl_bytes(
    payload: bytes,
    *,
    max_jsonl_line_bytes: int,
    max_decompressed_jsonl_bytes: int | None = None,
) -> Iterator[dict[str, Any]]:
    with gzip.GzipFile(fileobj=io.BytesIO(payload), mode="rb") as stream:
        yield from _iter_jsonl_stream(
            stream,
            max_jsonl_line_bytes=max_jsonl_line_bytes,
            max_decompressed_jsonl_bytes=max_decompressed_jsonl_bytes,
        )


def iter_gzip_jsonl_path(path: Path, config: Mapping[str, Any]) -> Iterator[dict[str, Any]]:
    policy = config["selection_policy"]
    with gzip.open(path, "rb") as stream:
        yield from _iter_jsonl_stream(
            stream,
            max_jsonl_line_bytes=policy["max_jsonl_line_bytes"],
            max_decompressed_jsonl_bytes=policy["max_decompressed_jsonl_bytes"],
        )


def normalize_text(text: object) -> str:
    _require(isinstance(text, str), "text must be a string")
    normalized = unicodedata.normalize(
        "NFC",
        text.replace("\r\n", "\n").replace("\r", "\n"),
    ).strip()
    _require(bool(normalized), "text is empty")
    for char in normalized:
        if unicodedata.category(char) == "Cc" and char not in _ALLOWED_CONTROLS:
            raise UsgpoIntakeError("text contains forbidden control characters")
    return normalized


def _govinfo_url_binds_package(url: object, package_id: str) -> bool:
    if not isinstance(url, str):
        return False
    parts = urlsplit(url)
    if (
        parts.scheme != "https"
        or parts.hostname not in _GOVINFO_HOSTS
        or parts.username is not None
        or parts.password is not None
        or parts.query
        or parts.fragment
        or parts.port not in (None, 443)
    ):
        return False
    segments = [segment for segment in unquote(parts.path).split("/") if segment]
    return len(segments) >= 4 and segments[:2] == ["content", "pkg"] and segments[2] == package_id


def _record_contract(
    record: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[str, Mapping[str, Any]]:
    _require(set(record) == _ROW_KEYS, "USGPO row schema drift")
    _require(record.get("source") == "usgpo", "USGPO source drift")
    record_id = record.get("id")
    _require(
        isinstance(record_id, str) and bool(_PACKAGE_ID_RE.fullmatch(record_id)),
        "invalid USGPO package id",
    )
    metadata = record.get("metadata")
    _require(isinstance(metadata, Mapping), "USGPO metadata missing")
    _require(set(metadata) == _METADATA_KEYS, "USGPO metadata schema drift")
    _require(
        metadata.get("publisher") == config["rights_policy"]["expected_publisher"],
        "USGPO publisher drift",
    )
    _require(
        metadata.get("license") == config["rights_policy"]["expected_license"],
        "USGPO license drift",
    )
    _require(
        isinstance(metadata.get("title"), str) and bool(metadata["title"].strip()),
        "USGPO title missing",
    )
    _require(
        isinstance(record.get("added"), str) and bool(record["added"].strip()),
        "invalid USGPO added timestamp",
    )
    _require(
        _govinfo_url_binds_package(metadata.get("url"), record_id),
        "GovInfo text URL does not bind exact package id",
    )
    return record_id, metadata


def _authority_reason(metadata: Mapping[str, Any], config: Mapping[str, Any]) -> str | None:
    if metadata["category"] not in config["rights_policy"]["federal_collection_allowlist"]:
        return "unsupported_federal_scope_collection"
    author = metadata.get("author")
    if author is None:
        return None
    if isinstance(author, int) and not isinstance(author, bool) and 1 <= author <= 999:
        return None
    if isinstance(author, str) and author.strip().isdigit() and 1 <= int(author.strip()) <= 999:
        return None
    return "ambiguous_noncongress_author_metadata"


def _quality_reason(text: str, config: Mapping[str, Any]) -> str | None:
    raw = text.encode("utf-8")
    policy = config["selection_policy"]
    if len(raw) < policy["min_single_normalized_utf8_bytes"]:
        return "too_short"
    if len(raw) > policy["max_single_normalized_utf8_bytes"]:
        return "too_large"
    if _EMAIL_RE.search(text):
        return "email_like_contact"
    if _PHONE_RE.search(text):
        return "phone_like_contact"
    if _SECRET_RE.search(text):
        return "secret_like_text"
    alpha = [char for char in text if char.isalpha()]
    if len(alpha) < policy["min_total_alpha_characters"]:
        return "low_alpha_content"
    ascii_alpha = sum("a" <= char.lower() <= "z" for char in alpha)
    if ascii_alpha / len(alpha) < policy["min_ascii_alpha_ratio"]:
        return "non_english_like_text"
    return None


def materialize(
    config: Mapping[str, Any],
    records: Iterable[Mapping[str, Any]],
    *,
    full_shard_hash_verified: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    validate_config(config)
    policy = config["selection_policy"]
    candidates: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    rejection_counts: Counter[str] = Counter()
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    accepted_bytes = 0
    examined = 0

    for record in records:
        if (
            examined >= policy["max_examined_documents"]
            or len(candidates) >= policy["max_documents"]
            or accepted_bytes >= policy["max_total_normalized_utf8_bytes"]
        ):
            break
        _require(isinstance(record, Mapping), "JSONL row must be an object")
        examined += 1
        record_id, metadata = _record_contract(record, config)
        _require(record_id not in seen_ids, f"duplicate USGPO package id: {record_id}")
        seen_ids.add(record_id)
        text = normalize_text(record.get("text"))
        text_bytes = text.encode("utf-8")
        text_hash = sha256_bytes(text_bytes)
        reason = _authority_reason(metadata, config) or _quality_reason(text, config)
        if reason is None and text_hash in seen_hashes:
            reason = "exact_normalized_duplicate"
        if (
            reason is None
            and accepted_bytes + len(text_bytes)
            > policy["max_total_normalized_utf8_bytes"]
        ):
            reason = "family_byte_cap"

        evidence.append(
            {
                "collection_code": metadata["category"],
                "decision": "CANDIDATE" if reason is None else "REJECT",
                "govinfo_url": metadata["url"],
                "normalized_utf8_bytes": len(text_bytes),
                "package_id": record_id,
                "reason": "item_rights_review_required" if reason is None else reason,
                "text_sha256": text_hash,
            }
        )
        if reason is not None:
            rejection_counts[reason] += 1
            continue

        seen_hashes.add(text_hash)
        candidates.append(
            {
                "collection_code": metadata["category"],
                "evaluation_eligible": False,
                "govinfo_url": metadata["url"],
                "item_rights_review_required": True,
                "record_id": f"usgpo:{record_id}",
                "source_family": config["source_family"],
                "source_record_id": record_id,
                "source_revision": _REVISION,
                "source_shard_lfs_sha256": _SHARD_SHA256,
                "source_shard_path": config["upstream"]["shard_path"],
                "text": text,
                "text_sha256": text_hash,
                "training_eligible": False,
                "utf8_bytes": len(text_bytes),
            }
        )
        accepted_bytes += len(text_bytes)

    _require(bool(candidates), "bounded USGPO materialization retained zero candidates")
    inventory = [
        {
            key: row[key]
            for key in (
                "collection_code",
                "govinfo_url",
                "normalized_utf8_bytes",
                "package_id",
                "text_sha256",
            )
        }
        for row in evidence
        if row["decision"] == "CANDIDATE"
    ]
    report: dict[str, Any] = {
        "accepted_documents": len(candidates),
        "accepted_normalized_utf8_bytes": accepted_bytes,
        "authorized_unique_loss_positions": 0,
        "canonical_capacity_credit_bytes": 0,
        "candidate_inventory_identity_sha256": sha256_bytes(
            (canonical_json(inventory) + "\n").encode("utf-8")
        ),
        "contract_identity_sha256": _CONTRACT_ID,
        "corpus_admitted": False,
        "evaluation_eligible": False,
        "examined_documents": examined,
        "final_test_payload_accessed": False,
        "full_shard_hash_verified": bool(full_shard_hash_verified),
        "item_level_rights_review": "REQUIRED",
        "model_training_permitted": False,
        "paid_compute_authorized": False,
        "privacy_gate": "SOURCE_LOCAL_PREFILTER_ONLY",
        "project_review_status": "REVIEW_REQUIRED",
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "required_downstream_gates": list(config["required_downstream_gates"]),
        "schema_version": "12-6.d03-common-pile-usgpo-materialization.v1",
        "source_family": config["source_family"],
        "status": "CANDIDATE_MATERIALIZED_ZERO_CREDIT",
        "tokenizer_fit_permitted": False,
        "training_authorized_bytes": 0,
    }
    report["report_identity_sha256"] = self_identity(report, "report_identity_sha256")
    return candidates, evidence, report


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(canonical_json(row) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(canonical_json(payload) + "\n", encoding="utf-8", newline="\n")


def materialize_path(
    config: Mapping[str, Any],
    source_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    transport = verify_transport(config, source_path)
    candidates, evidence, report = materialize(
        config,
        iter_gzip_jsonl_path(source_path, config),
        full_shard_hash_verified=True,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = output_dir / "usgpo_candidates.jsonl"
    evidence_path = output_dir / "usgpo_item_review_inventory.jsonl"
    report_path = output_dir / "usgpo_materialization_report.json"
    manifest_path = output_dir / "usgpo_materialization_manifest.json"
    _write_jsonl(candidate_path, candidates)
    _write_jsonl(evidence_path, evidence)
    _write_json(report_path, report)
    manifest: dict[str, Any] = {
        "artifacts": {
            "candidate_jsonl": {
                "bytes": candidate_path.stat().st_size,
                "sha256": sha256_file(candidate_path),
            },
            "item_review_inventory_jsonl": {
                "bytes": evidence_path.stat().st_size,
                "sha256": sha256_file(evidence_path),
            },
            "report_json": {
                "bytes": report_path.stat().st_size,
                "sha256": sha256_file(report_path),
            },
        },
        "authorized_unique_loss_positions": 0,
        "canonical_capacity_credit_bytes": 0,
        "contract_identity_sha256": _CONTRACT_ID,
        "corpus_admitted": False,
        "evaluation_eligible": False,
        "final_test_payload_accessed": False,
        "item_level_rights_review": "REQUIRED",
        "model_training_permitted": False,
        "paid_compute_authorized": False,
        "schema_version": "12-6.d03-common-pile-usgpo-manifest.v1",
        "source_transport": transport,
        "status": "CANDIDATE_MATERIALIZED_ZERO_CREDIT",
        "tokenizer_fit_permitted": False,
        "training_authorized_bytes": 0,
    }
    manifest["manifest_identity_sha256"] = self_identity(manifest, "manifest_identity_sha256")
    _write_json(manifest_path, manifest)
    return manifest
