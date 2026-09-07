from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

SCHEMA_VERSION = "12-6.d03-loc-books-intake.v1"
SOURCE_FAMILY = "en.us.loc.selected-digitized-books.public-domain"
DATASET = "common-pile/library_of_congress"
DATASET_REVISION = "d31bdba02cdad5104ccec2c02ae799c0bcb5a9a7"
SHARD_PATH = "data/00000_loc_books.jsonl.gz"
SHARD_SHA256 = "a6a6023d9cae067b5531ba84537877c180ab3ff1579f99aeac446d65d465ae0f"
SHARD_COMPRESSED_BYTES = 358_594_502
COMMON_PILE_CODE_REVISION = "9457f04a14cb2355ab00023420369d46ffd4a395"
COMMON_PILE_COLLECTOR_BLOB = "0f8633d9a0c12bd04140420d74a95d82a065119f"
COMMON_PILE_REGISTRY_BLOB = "7b4d6828288672bf25c551e85a5d7f7399e8ef0f"

_HEX40_RE = re.compile(r"^[0-9a-f]{40}$")
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_EMAIL_RE = re.compile(
    r"(?<![A-Za-z0-9._%+\-])"
    r"[A-Za-z0-9._%+\-]{1,64}@[A-Za-z0-9.\-]{1,253}\.[A-Za-z]{2,63}"
    r"(?![A-Za-z0-9._%+\-])"
)
_SECRET_MARKERS = (
    "-----begin private key-----",
    "-----begin rsa private key-----",
    "-----begin openssh private key-----",
    "aws_secret_access_key",
    "github_pat_",
    "sk-proj-",
)


class LocBooksIntakeError(ValueError):
    """Fail-closed Library of Congress intake contract violation."""


@dataclass(frozen=True)
class RecordDecision:
    accepted: bool
    reason: str
    record_id: str | None = None
    normalized_text: str | None = None
    normalized_bytes: int = 0
    text_sha256: str | None = None


def canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def self_identity(value: Mapping[str, Any], field: str) -> str:
    clone = deepcopy(dict(value))
    clone.pop(field, None)
    payload = (canonical_json(clone) + "\n").encode("utf-8")
    return sha256_bytes(payload)


def _require_hex(value: object, length: int, field: str) -> str:
    if not isinstance(value, str):
        raise LocBooksIntakeError(f"{field} must be a string")
    pattern = _HEX40_RE if length == 40 else _HEX64_RE
    if not pattern.fullmatch(value):
        raise LocBooksIntakeError(f"{field} must be lowercase {length}-hex")
    return value


def validate_config(config: Mapping[str, Any]) -> None:
    expected = {
        "schema_version": SCHEMA_VERSION,
        "source_family": SOURCE_FAMILY,
        "execution_profile": "LOCAL_FREE",
        "training_authorized_bytes": 0,
        "authorized_unique_loss_positions": 0,
        "canonical_capacity_credit_bytes": 0,
        "corpus_admitted": False,
        "tokenizer_fit_permitted": False,
        "model_training_permitted": False,
        "evaluation_use_permitted": False,
        "final_test_access_permitted": False,
        "paid_compute_authorized": False,
    }
    for field, value in expected.items():
        if config.get(field) != value:
            raise LocBooksIntakeError(f"{field} drift")

    audit = config.get("common_pile_audit")
    if not isinstance(audit, Mapping):
        raise LocBooksIntakeError("common_pile_audit missing")
    audit_expected = {
        "registry_path": "configs/data/common_pile_source_rights_v1.json",
        "registry_blob_sha1": COMMON_PILE_REGISTRY_BLOB,
        "source_key": "library_of_congress",
        "audited_code_revision": COMMON_PILE_CODE_REVISION,
        "collector_path": "sources/loc_books/books.py",
        "collector_blob_sha1": COMMON_PILE_COLLECTOR_BLOB,
    }
    for field, value in audit_expected.items():
        if audit.get(field) != value:
            raise LocBooksIntakeError(f"common_pile_audit.{field} drift")
    _require_hex(audit["registry_blob_sha1"], 40, "registry blob")
    _require_hex(audit["audited_code_revision"], 40, "audited code revision")
    _require_hex(audit["collector_blob_sha1"], 40, "collector blob")

    upstream = config.get("upstream")
    if not isinstance(upstream, Mapping):
        raise LocBooksIntakeError("upstream missing")
    upstream_expected = {
        "dataset": DATASET,
        "revision": DATASET_REVISION,
        "shard_path": SHARD_PATH,
        "shard_sha256": SHARD_SHA256,
        "compressed_bytes": SHARD_COMPRESSED_BYTES,
        "resolve_url": (
            "https://huggingface.co/datasets/common-pile/library_of_congress/"
            f"resolve/{DATASET_REVISION}/{SHARD_PATH}?download=true"
        ),
    }
    for field, value in upstream_expected.items():
        if upstream.get(field) != value:
            raise LocBooksIntakeError(f"upstream.{field} drift")
    _require_hex(upstream["revision"], 40, "upstream revision")
    _require_hex(upstream["shard_sha256"], 64, "shard sha256")

    record_policy = config.get("record_policy")
    if not isinstance(record_policy, Mapping):
        raise LocBooksIntakeError("record_policy missing")
    record_expected = {
        "required_source": "loc_books",
        "required_license": "Public Domain",
        "required_language": "english",
        "min_publication_year": 1500,
        "item_url_prefix": "https://www.loc.gov/item/",
        "text_file_url_https_loc_gov_only": True,
    }
    for field, value in record_expected.items():
        if record_policy.get(field) != value:
            raise LocBooksIntakeError(f"record_policy.{field} drift")

    policy = config.get("selection_policy")
    if not isinstance(policy, Mapping):
        raise LocBooksIntakeError("selection_policy missing")
    if policy.get("ordering") != "IMMUTABLE_SHARD_ORDER":
        raise LocBooksIntakeError("selection ordering drift")
    integer_fields = (
        "max_documents",
        "max_examined_documents",
        "max_total_candidate_bytes",
        "min_document_bytes",
        "max_document_bytes",
        "max_jsonl_line_bytes",
        "min_word_count",
    )
    for field in integer_fields:
        value = policy.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise LocBooksIntakeError(f"invalid selection_policy.{field}")
    if policy["max_examined_documents"] < policy["max_documents"]:
        raise LocBooksIntakeError("max_examined_documents below max_documents")
    if policy["max_total_candidate_bytes"] >= 5_000_000:
        raise LocBooksIntakeError("family planning ceiling must remain below 5,000,000 bytes")
    if policy["max_document_bytes"] > policy["max_total_candidate_bytes"]:
        raise LocBooksIntakeError("max_document_bytes exceeds total candidate ceiling")
    if policy["min_document_bytes"] >= policy["max_document_bytes"]:
        raise LocBooksIntakeError("invalid document byte bounds")
    ratio = policy.get("min_letter_ratio")
    if not isinstance(ratio, (int, float)) or isinstance(ratio, bool):
        raise LocBooksIntakeError("min_letter_ratio must be numeric")
    if not 0.0 < float(ratio) <= 1.0:
        raise LocBooksIntakeError("min_letter_ratio outside (0, 1]")

    normalization = config.get("normalization_policy")
    if normalization != {
        "unicode": "NFC",
        "newlines": "LF",
        "strip_trailing_horizontal_whitespace": True,
        "max_consecutive_blank_lines": 2,
    }:
        raise LocBooksIntakeError("normalization policy drift")

    privacy = config.get("privacy_policy")
    if privacy != {
        "reject_email_addresses": True,
        "reject_secret_markers": True,
    }:
        raise LocBooksIntakeError("privacy policy drift")

    required = config.get("required_downstream_gates")
    if required != [
        "privacy",
        "quality",
        "global_exact_near_fragment_lineage_dedup",
        "reserved_evaluation_decontamination",
        "balance_family_caps",
        "cluster_safe_split",
        "deterministic_pack_two_clean_builds",
        "positive_unique_loss_ledger",
    ]:
        raise LocBooksIntakeError("required downstream gates drift")

    identity = _require_hex(
        config.get("contract_identity_sha256"),
        64,
        "contract_identity_sha256",
    )
    if self_identity(config, "contract_identity_sha256") != identity:
        raise LocBooksIntakeError("contract identity mismatch")


def load_config(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LocBooksIntakeError(f"cannot load config: {exc}") from exc
    if not isinstance(value, dict):
        raise LocBooksIntakeError("config root must be an object")
    validate_config(value)
    return value


def normalize_text(text: str) -> str:
    value = unicodedata.normalize("NFC", text)
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip(" \t") for line in value.split("\n")]
    output: list[str] = []
    blank_run = 0
    for line in lines:
        if line:
            blank_run = 0
            output.append(line)
            continue
        blank_run += 1
        if blank_run <= 2:
            output.append("")
    return "\n".join(output).strip() + "\n"


def _has_disallowed_control(text: str) -> bool:
    for char in text:
        if char in "\n\t":
            continue
        category = unicodedata.category(char)
        if category in {"Cc", "Cs"}:
            return True
    return False


def _has_obvious_contact_or_secret(text: str) -> bool:
    if _EMAIL_RE.search(text):
        return True
    lower = text.casefold()
    return any(marker in lower for marker in _SECRET_MARKERS)


def _letter_ratio(text: str) -> float:
    non_space = sum(1 for char in text if not char.isspace())
    if non_space == 0:
        return 0.0
    letters = sum(1 for char in text if char.isalpha())
    return letters / non_space


def _valid_https_loc_url(value: object) -> bool:
    if not isinstance(value, str):
        return False
    parsed = urlparse(value)
    if parsed.scheme != "https" or not parsed.hostname:
        return False
    hostname = parsed.hostname.casefold()
    return hostname == "loc.gov" or hostname.endswith(".loc.gov")


def decide_record(record: Mapping[str, Any], config: Mapping[str, Any]) -> RecordDecision:
    policy = config["selection_policy"]
    rights = config["record_policy"]

    record_id = record.get("id")
    if not isinstance(record_id, str) or not record_id.strip():
        return RecordDecision(False, "invalid_id")
    record_id = record_id.strip()
    if len(record_id) > 128:
        return RecordDecision(False, "invalid_id")

    if record.get("source") != rights["required_source"]:
        return RecordDecision(False, "wrong_source")

    metadata = record.get("metadata")
    if not isinstance(metadata, Mapping):
        return RecordDecision(False, "missing_metadata")
    if metadata.get("license") != rights["required_license"]:
        return RecordDecision(False, "wrong_license")
    language = metadata.get("language")
    if not isinstance(language, str) or language.casefold() != rights["required_language"]:
        return RecordDecision(False, "wrong_language")
    year = metadata.get("year")
    if not isinstance(year, int) or isinstance(year, bool):
        return RecordDecision(False, "invalid_year")
    if year < rights["min_publication_year"]:
        return RecordDecision(False, "year_below_floor")
    item_url = metadata.get("item_url")
    if not isinstance(item_url, str) or not item_url.startswith(rights["item_url_prefix"]):
        return RecordDecision(False, "invalid_item_url")
    if not _valid_https_loc_url(metadata.get("text_file_url")):
        return RecordDecision(False, "invalid_text_file_url")

    text = record.get("text")
    if not isinstance(text, str):
        return RecordDecision(False, "invalid_text")
    normalized = normalize_text(text)
    payload = normalized.encode("utf-8")
    byte_count = len(payload)
    if byte_count < policy["min_document_bytes"]:
        return RecordDecision(False, "too_short")
    if byte_count > policy["max_document_bytes"]:
        return RecordDecision(False, "too_large")
    if _has_disallowed_control(normalized):
        return RecordDecision(False, "control_character")
    if "\ufffd" in normalized:
        return RecordDecision(False, "unicode_replacement_character")
    if _has_obvious_contact_or_secret(normalized):
        return RecordDecision(False, "contact_or_secret_marker")
    if len(normalized.split()) < policy["min_word_count"]:
        return RecordDecision(False, "too_few_words")
    if _letter_ratio(normalized) < float(policy["min_letter_ratio"]):
        return RecordDecision(False, "low_letter_ratio")

    return RecordDecision(
        True,
        "accepted",
        record_id=record_id,
        normalized_text=normalized,
        normalized_bytes=byte_count,
        text_sha256=sha256_bytes(payload),
    )


def iter_gzip_jsonl(path: Path, *, max_line_bytes: int) -> Iterator[dict[str, Any]]:
    try:
        raw = path.open("rb")
    except OSError as exc:
        raise LocBooksIntakeError(f"cannot open shard: {exc}") from exc
    with raw:
        try:
            with gzip.GzipFile(fileobj=raw, mode="rb") as stream:
                line_number = 0
                while True:
                    line = stream.readline(max_line_bytes + 1)
                    if not line:
                        break
                    line_number += 1
                    if len(line) > max_line_bytes:
                        raise LocBooksIntakeError(
                            f"jsonl line {line_number} exceeds max_jsonl_line_bytes"
                        )
                    try:
                        decoded = line.decode("utf-8")
                    except UnicodeDecodeError as exc:
                        raise LocBooksIntakeError(
                            f"jsonl line {line_number} is not UTF-8"
                        ) from exc
                    try:
                        record = json.loads(decoded)
                    except json.JSONDecodeError as exc:
                        raise LocBooksIntakeError(
                            f"jsonl line {line_number} is malformed JSON"
                        ) from exc
                    if not isinstance(record, dict):
                        raise LocBooksIntakeError(
                            f"jsonl line {line_number} root is not an object"
                        )
                    yield record
        except (gzip.BadGzipFile, EOFError, OSError) as exc:
            raise LocBooksIntakeError(f"invalid gzip stream: {exc}") from exc


def verify_bound_shard(path: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    validate_config(config)
    upstream = config["upstream"]
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise LocBooksIntakeError(f"cannot stat shard: {exc}") from exc
    if size != upstream["compressed_bytes"]:
        raise LocBooksIntakeError(
            f"compressed shard size mismatch: expected {upstream['compressed_bytes']}, got {size}"
        )
    digest = sha256_file(path)
    if digest != upstream["shard_sha256"]:
        raise LocBooksIntakeError("compressed shard sha256 mismatch")
    return {
        "dataset": upstream["dataset"],
        "revision": upstream["revision"],
        "shard_path": upstream["shard_path"],
        "compressed_bytes": size,
        "shard_sha256": digest,
        "full_shard_hash_verified": True,
    }


def _candidate_line(
    decision: RecordDecision,
    *,
    source_family: str,
) -> bytes:
    if not decision.accepted or decision.record_id is None or decision.normalized_text is None:
        raise LocBooksIntakeError("cannot serialize rejected record")
    item = {
        "evaluation_eligible": False,
        "id": decision.record_id,
        "source": "loc_books",
        "source_family": source_family,
        "text": decision.normalized_text,
        "text_sha256": decision.text_sha256,
        "training_eligible": False,
    }
    return (canonical_json(item) + "\n").encode("utf-8")


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    try:
        with temp.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temp.replace(path)
    finally:
        if temp.exists():
            temp.unlink()


def materialize_records(
    records: Iterable[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    source_snapshot: Mapping[str, Any],
    candidate_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    validate_config(config)
    policy = config["selection_policy"]
    seen_ids: dict[str, str] = {}
    seen_texts: set[str] = set()
    rejection_counts: Counter[str] = Counter()
    admitted: list[dict[str, Any]] = []
    candidate_lines: list[bytes] = []
    candidate_text_bytes = 0
    examined = 0
    stop_reason = "input_exhausted"

    for record in records:
        if examined >= policy["max_examined_documents"]:
            stop_reason = "examined_document_limit"
            break
        examined += 1
        decision = decide_record(record, config)
        if not decision.accepted:
            rejection_counts[decision.reason] += 1
            continue

        assert decision.record_id is not None
        assert decision.text_sha256 is not None
        prior = seen_ids.get(decision.record_id)
        if prior is not None:
            if prior != decision.text_sha256:
                raise LocBooksIntakeError(
                    f"record id collision with divergent payload: {decision.record_id}"
                )
            rejection_counts["duplicate_id"] += 1
            continue
        seen_ids[decision.record_id] = decision.text_sha256

        if decision.text_sha256 in seen_texts:
            rejection_counts["duplicate_normalized_text"] += 1
            continue

        next_total = candidate_text_bytes + decision.normalized_bytes
        if next_total > policy["max_total_candidate_bytes"]:
            stop_reason = "candidate_byte_cap"
            break

        candidate_lines.append(_candidate_line(decision, source_family=config["source_family"]))
        candidate_text_bytes = next_total
        seen_texts.add(decision.text_sha256)
        admitted.append(
            {
                "id_sha256": sha256_bytes(decision.record_id.encode("utf-8")),
                "normalized_bytes": decision.normalized_bytes,
                "text_sha256": decision.text_sha256,
            }
        )
        if len(admitted) >= policy["max_documents"]:
            stop_reason = "document_limit"
            break

    candidate_payload = b"".join(candidate_lines)
    candidate_sha256 = sha256_bytes(candidate_payload)
    report: dict[str, Any] = {
        "schema_version": "12-6.d03-loc-books-intake-report.v1",
        "status": "SOURCE_CANDIDATE_ZERO_CREDIT",
        "contract_identity_sha256": config["contract_identity_sha256"],
        "source_family": config["source_family"],
        "source_snapshot": dict(source_snapshot),
        "selection_ordering": policy["ordering"],
        "examined_documents": examined,
        "accepted_documents": len(admitted),
        "candidate_text_bytes": candidate_text_bytes,
        "candidate_file_bytes": len(candidate_payload),
        "candidate_sha256": candidate_sha256,
        "records": admitted,
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "stop_reason": stop_reason,
        "training_authorized_bytes": 0,
        "authorized_unique_loss_positions": 0,
        "canonical_capacity_credit_bytes": 0,
        "training_eligible": False,
        "evaluation_eligible": False,
        "tokenizer_fit_permitted": False,
        "model_training_permitted": False,
        "paid_compute_authorized": False,
        "required_downstream_gates": list(config["required_downstream_gates"]),
    }
    report["report_identity_sha256"] = self_identity(report, "report_identity_sha256")
    report_payload = (canonical_json(report) + "\n").encode("utf-8")
    _write_atomic(candidate_path, candidate_payload)
    _write_atomic(report_path, report_payload)
    return report


def materialize_bound_shard(
    config: Mapping[str, Any],
    shard_path: Path,
    *,
    candidate_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    snapshot = verify_bound_shard(shard_path, config)
    max_line_bytes = int(config["selection_policy"]["max_jsonl_line_bytes"])
    records = iter_gzip_jsonl(shard_path, max_line_bytes=max_line_bytes)
    return materialize_records(
        records,
        config,
        source_snapshot=snapshot,
        candidate_path=candidate_path,
        report_path=report_path,
    )


def verify_two_builds(
    *,
    candidate_a: Path,
    report_a: Path,
    candidate_b: Path,
    report_b: Path,
) -> dict[str, str]:
    candidate_a_bytes = candidate_a.read_bytes()
    candidate_b_bytes = candidate_b.read_bytes()
    report_a_bytes = report_a.read_bytes()
    report_b_bytes = report_b.read_bytes()
    if candidate_a_bytes != candidate_b_bytes:
        raise LocBooksIntakeError("two-build candidate mismatch")
    if report_a_bytes != report_b_bytes:
        raise LocBooksIntakeError("two-build report mismatch")
    return {
        "candidate_sha256": sha256_bytes(candidate_a_bytes),
        "report_sha256": sha256_bytes(report_a_bytes),
        "status": "BYTE_IDENTICAL_TWO_BUILD_PASS",
    }
