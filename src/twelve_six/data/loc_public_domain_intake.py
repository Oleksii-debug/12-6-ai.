from __future__ import annotations

import gzip
import hashlib
import json
import unicodedata
from collections.abc import Iterable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

_SCHEMA = "12-6.d03-loc-public-domain-intake.v1"
_REPORT_SCHEMA = "12-6.d03-loc-public-domain-materialization.v1"
_REQUIRED_KEYS = frozenset({"id", "text", "source", "added", "metadata"})
_HEX40 = frozenset("0123456789abcdef")
_HEX64 = _HEX40


class LocIntakeError(ValueError):
    """Fail-closed Library of Congress intake contract violation."""


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def self_identity(value: Mapping[str, Any], field: str) -> str:
    clone = deepcopy(dict(value))
    clone.pop(field, None)
    return sha256_bytes((canonical_json(clone) + "\n").encode("utf-8"))


def _require_hex(value: object, length: int, field: str) -> str:
    if not isinstance(value, str) or len(value) != length:
        raise LocIntakeError(f"{field} must be lowercase {length}-hex")
    alphabet = _HEX40 if length == 40 else _HEX64
    if any(ch not in alphabet for ch in value):
        raise LocIntakeError(f"{field} must be lowercase {length}-hex")
    return value


def validate_config(config: Mapping[str, Any]) -> None:
    expected = {
        "schema_version": _SCHEMA,
        "source_family": "en.loc.selected-digitized-books.public-domain",
        "execution_profile": "LOCAL_FREE",
        "training_authorized_bytes": 0,
        "authorized_unique_loss_positions": 0,
        "canonical_capacity_credit_bytes": 0,
        "corpus_admitted": False,
        "tokenizer_fit_permitted": False,
        "model_training_permitted": False,
        "paid_compute_authorized": False,
        "final_test_payload_accessed": False,
    }
    for field, wanted in expected.items():
        if config.get(field) != wanted:
            raise LocIntakeError(f"{field} drift")

    audit = config.get("common_pile_audit")
    if not isinstance(audit, Mapping):
        raise LocIntakeError("common_pile_audit missing")
    if audit.get("registry_path") != "configs/data/common_pile_source_rights_v1.json":
        raise LocIntakeError("common_pile_audit.registry_path drift")
    if audit.get("registry_blob_sha1") != "7b4d6828288672bf25c551e85a5d7f7399e8ef0f":
        raise LocIntakeError("common_pile_audit.registry_blob_sha1 drift")
    if audit.get("source_key") != "library_of_congress":
        raise LocIntakeError("common_pile_audit.source_key drift")
    if audit.get("audited_code_revision") != "9457f04a14cb2355ab00023420369d46ffd4a395":
        raise LocIntakeError("common_pile audited revision drift")
    if audit.get("collector_books_blob_sha1") != "0f8633d9a0c12bd04140420d74a95d82a065119f":
        raise LocIntakeError("Common Pile books collector blob drift")
    if audit.get("collector_metadata_blob_sha1") != "a219f88cbe8f6cc33bd72eed6c9d75958a8efb08":
        raise LocIntakeError("Common Pile metadata collector blob drift")

    upstream = config.get("upstream")
    if not isinstance(upstream, Mapping):
        raise LocIntakeError("upstream missing")
    if upstream.get("dataset") != "common-pile/library_of_congress":
        raise LocIntakeError("upstream.dataset drift")
    if upstream.get("revision") != "efa378ea3f6feab6f64f407e7555457f0034a985":
        raise LocIntakeError("upstream.revision drift")
    if upstream.get("shard_path") != "data/00000_loc_books.jsonl.gz":
        raise LocIntakeError("upstream.shard_path drift")
    if upstream.get("shard_sha256") != (
        "a6a6023d9cae067b5531ba84537877c180ab3ff1579f99aeac446d65d465ae0f"
    ):
        raise LocIntakeError("upstream.shard_sha256 drift")
    if upstream.get("xet_hash") != (
        "c93d3b06db47c6bd597d88097bfecc84c671967e20cedcb2c685742c4a3a6b48"
    ):
        raise LocIntakeError("upstream.xet_hash drift")
    if upstream.get("shard_bytes") != 358_594_502:
        raise LocIntakeError("upstream.shard_bytes drift")
    expected_url = (
        "https://huggingface.co/datasets/common-pile/library_of_congress/resolve/"
        f"{upstream['revision']}/{upstream['shard_path']}?download=true"
    )
    if upstream.get("download_url") != expected_url:
        raise LocIntakeError("upstream.download_url drift")

    policy = config.get("selection_policy")
    if not isinstance(policy, Mapping):
        raise LocIntakeError("selection_policy missing")
    exact_policy = {
        "ordering": "UPSTREAM_SHARD_ORDER",
        "required_source": "loc_books",
        "required_metadata_license": "Public Domain",
        "max_examined_documents": 512,
        "max_accepted_documents": 128,
        "max_total_normalized_utf8_bytes": 32_000_000,
        "max_document_normalized_utf8_bytes": 1_000_000,
        "max_json_line_bytes": 20_000_000,
        "min_document_normalized_utf8_bytes": 1_024,
        "normalization": "STRICT_UTF8_NFC_OUTER_TRIM",
        "exact_normalized_dedup": True,
    }
    if dict(policy) != exact_policy:
        raise LocIntakeError("selection_policy drift")

    required = config.get("required_downstream_gates")
    if required != [
        "source_rights_revalidation",
        "privacy",
        "quality",
        "global_exact_near_fragment_lineage_dedup",
        "reserved_evaluation_decontamination",
        "balance_family_caps",
        "cluster_safe_split",
        "deterministic_pack_two_clean_builds",
        "positive_unique_loss_ledger",
    ]:
        raise LocIntakeError("required downstream gates drift")

    identity = _require_hex(
        config.get("contract_identity_sha256"), 64, "contract_identity_sha256"
    )
    if self_identity(config, "contract_identity_sha256") != identity:
        raise LocIntakeError("contract identity mismatch")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> tuple[int, str]:
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            total += len(chunk)
            digest.update(chunk)
    return total, digest.hexdigest()


def verify_shard(path: Path, config: Mapping[str, Any]) -> None:
    validate_config(config)
    observed_bytes, observed_sha256 = sha256_file(path)
    upstream = config["upstream"]
    if observed_bytes != upstream["shard_bytes"]:
        raise LocIntakeError("immutable shard byte-size mismatch")
    if observed_sha256 != upstream["shard_sha256"]:
        raise LocIntakeError("immutable shard SHA-256 mismatch")


def normalize_text(value: object) -> str:
    if not isinstance(value, str):
        raise LocIntakeError("record text must be a string")
    if "\x00" in value:
        raise LocIntakeError("record text contains NUL")
    return unicodedata.normalize("NFC", value).strip()


def _inventory_identity(rows: list[dict[str, Any]]) -> str:
    return sha256_bytes((canonical_json(rows) + "\n").encode("utf-8"))


def materialize_records(
    config: Mapping[str, Any],
    records: Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    validate_config(config)
    policy = config["selection_policy"]
    candidates: list[dict[str, Any]] = []
    inventory: list[dict[str, Any]] = []
    seen_payloads: set[str] = set()
    rejected = {
        "field_schema": 0,
        "source": 0,
        "license": 0,
        "empty_or_short": 0,
        "document_too_large": 0,
        "duplicate": 0,
        "budget": 0,
    }
    examined = 0
    accepted_bytes = 0

    for record in records:
        if examined >= policy["max_examined_documents"]:
            break
        examined += 1
        if not isinstance(record, Mapping) or set(record) != _REQUIRED_KEYS:
            rejected["field_schema"] += 1
            continue
        if record.get("source") != policy["required_source"]:
            rejected["source"] += 1
            continue
        metadata = record.get("metadata")
        if (
            not isinstance(metadata, Mapping)
            or metadata.get("license") != policy["required_metadata_license"]
        ):
            rejected["license"] += 1
            continue
        record_id = record.get("id")
        if not isinstance(record_id, str) or not record_id.strip():
            rejected["field_schema"] += 1
            continue

        text = normalize_text(record.get("text"))
        payload = text.encode("utf-8")
        payload_bytes = len(payload)
        if payload_bytes < policy["min_document_normalized_utf8_bytes"]:
            rejected["empty_or_short"] += 1
            continue
        if payload_bytes > policy["max_document_normalized_utf8_bytes"]:
            rejected["document_too_large"] += 1
            continue
        payload_sha = sha256_bytes(payload)
        if payload_sha in seen_payloads:
            rejected["duplicate"] += 1
            continue
        if accepted_bytes + payload_bytes > policy["max_total_normalized_utf8_bytes"]:
            rejected["budget"] += 1
            continue

        clean_id = record_id.strip()
        row = {
            "record_id": f"loc:{clean_id}",
            "source_family": config["source_family"],
            "source_dataset": config["upstream"]["dataset"],
            "source_revision": config["upstream"]["revision"],
            "source_shard_path": config["upstream"]["shard_path"],
            "source_record_id": clean_id,
            "text": text,
            "text_sha256": payload_sha,
            "utf8_bytes": payload_bytes,
            "rights_signal": "metadata.license=Public Domain",
            "training_eligible": False,
            "evaluation_eligible": False,
        }
        candidates.append(row)
        inventory.append(
            {
                "record_id": row["record_id"],
                "source_record_id": clean_id,
                "text_sha256": payload_sha,
                "utf8_bytes": payload_bytes,
            }
        )
        seen_payloads.add(payload_sha)
        accepted_bytes += payload_bytes
        if len(candidates) >= policy["max_accepted_documents"]:
            break

    if not candidates:
        raise LocIntakeError("bounded materialization retained zero candidates")

    report = {
        "schema_version": _REPORT_SCHEMA,
        "status": "CANDIDATE_MATERIALIZED_ZERO_CREDIT",
        "source_family": config["source_family"],
        "upstream_revision": config["upstream"]["revision"],
        "upstream_shard_path": config["upstream"]["shard_path"],
        "upstream_shard_sha256": config["upstream"]["shard_sha256"],
        "contract_identity_sha256": config["contract_identity_sha256"],
        "examined_documents": examined,
        "accepted_documents": len(candidates),
        "accepted_normalized_utf8_bytes": accepted_bytes,
        "candidate_inventory_identity_sha256": _inventory_identity(inventory),
        "rejected_counts": rejected,
        "rights_boundary": (
            "ROW_METADATA_PUBLIC_DOMAIN_SIGNAL_CANDIDATE_ONLY_REVALIDATE_DOWNSTREAM"
        ),
        "training_authorized_bytes": 0,
        "authorized_unique_loss_positions": 0,
        "canonical_capacity_credit_bytes": 0,
        "corpus_admitted": False,
        "tokenizer_fit_permitted": False,
        "model_training_permitted": False,
        "paid_compute_authorized": False,
        "final_test_payload_accessed": False,
    }
    report["report_identity_sha256"] = self_identity(report, "report_identity_sha256")
    return candidates, report


def iter_gzip_jsonl(path: Path, *, max_json_line_bytes: int) -> Iterable[Mapping[str, Any]]:
    with gzip.open(path, "rb") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if len(raw) > max_json_line_bytes:
                raise LocIntakeError(f"JSONL line {line_number} exceeds safety limit")
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise LocIntakeError(f"JSONL line {line_number} is not strict UTF-8") from exc
            try:
                value = json.loads(text)
            except json.JSONDecodeError as exc:
                raise LocIntakeError(f"JSONL line {line_number} is invalid JSON") from exc
            if not isinstance(value, Mapping):
                raise LocIntakeError(f"JSONL line {line_number} is not an object")
            yield value


def materialize_shard(
    config: Mapping[str, Any],
    path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    verify_shard(path, config)
    records = iter_gzip_jsonl(
        path,
        max_json_line_bytes=config["selection_policy"]["max_json_line_bytes"],
    )
    return materialize_records(config, records)


def verify_report(config: Mapping[str, Any], report: Mapping[str, Any]) -> None:
    validate_config(config)
    if report.get("schema_version") != _REPORT_SCHEMA:
        raise LocIntakeError("unsupported report schema")
    expected = {
        "status": "CANDIDATE_MATERIALIZED_ZERO_CREDIT",
        "source_family": config["source_family"],
        "upstream_revision": config["upstream"]["revision"],
        "upstream_shard_path": config["upstream"]["shard_path"],
        "upstream_shard_sha256": config["upstream"]["shard_sha256"],
        "contract_identity_sha256": config["contract_identity_sha256"],
        "training_authorized_bytes": 0,
        "authorized_unique_loss_positions": 0,
        "canonical_capacity_credit_bytes": 0,
        "corpus_admitted": False,
        "tokenizer_fit_permitted": False,
        "model_training_permitted": False,
        "paid_compute_authorized": False,
        "final_test_payload_accessed": False,
    }
    for field, wanted in expected.items():
        if report.get(field) != wanted:
            raise LocIntakeError(f"report {field} drift")
    identity = _require_hex(
        report.get("report_identity_sha256"), 64, "report_identity_sha256"
    )
    if self_identity(report, "report_identity_sha256") != identity:
        raise LocIntakeError("report identity mismatch")
