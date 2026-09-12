#!/usr/bin/env python3
"""Deterministically quality/privacy-filter the exact current Rada normalization.

This is a candidate-only gate. It consumes the exact PR #864 normalization
authority, verifies the parent manifest/JSONL and its frozen identities, chunks
long legal text deterministically, and applies the incumbent bounded quality and
privacy predicates. It never grants corpus capacity, tokenizer fit, evaluation,
or model-training authority.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import unicodedata
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

CONFIG_SCHEMA = "12-6.d03-rada-bulk-quality-privacy.v1"
PARENT_MANIFEST_SCHEMA = "12-6.d03-rada-bulk-normalization-manifest.v1"
REPORT_SCHEMA = "12-6.d03-rada-bulk-quality-privacy-report.v1"
WORKER_ID = "D03-RADA-BULK-QUALITY-PRIVACY-20260911"
PARENT_WORKER_ID = "D03-RADA-BULK-NORMALIZATION-20260907"
SOURCE_FAMILY = "ua.rada.open-data.laws-texts"
PARENT_SAFE_RESULT = "NORMALIZED_RECORD_MATERIALIZATION_ONLY_DOWNSTREAM_GATES_REQUIRED"
PARENT_NORMALIZER_NAME = "RADA_VISIBLE_TEXT_HTML_UTF8_CP1251_NFKC_V1"
SAFE_RESULT = "QUALITY_PRIVACY_FILTERED_CANDIDATE_ONLY_DOWNSTREAM_GATES_REQUIRED"
DEFAULT_CONFIG = Path("configs/data/d03_rada_bulk_quality_privacy_v1.json")
ALLOWED_SOURCE_ENCODINGS = {"utf-8", "windows-1251"}

EXPECTED_PARENT_BINDING: dict[str, Any] = {
    "pr": 864,
    "head_sha": "656dd4abe7bdf9c379a86ac9a19f046d5b0d8538",
    "branch": "d03/rada-bulk-normalization-current-main-20260907",
    "execution_head_sha": "f62670084f80041757e162743356ac16e0fd81a7",
    "execution_run_id": 34565921713,
    "execution_evidence_identity_sha256": (
        "e1633070beaff596ae11f4974bd9662785723618f502f382850f7e3bea27ab6a"
    ),
    "manifest_schema": PARENT_MANIFEST_SCHEMA,
    "manifest_worker_id": PARENT_WORKER_ID,
    "source_family": SOURCE_FAMILY,
    "safe_result": PARENT_SAFE_RESULT,
    "manifest_identity_sha256": (
        "ee1c59dbc481b83bffb1380f751880e5fa6e64d654b5abad266c3e2c4d18293b"
    ),
    "manifest_transport_sha256": (
        "8923d26024ba396db14e6afeb367a5d3c54de573019252083aec88f2082bb119"
    ),
    "jsonl_sha256": (
        "b46baa0f1c5087f4a9772ee273da459e87c0e82111dd5e51b22fc0b85cde840e"
    ),
    "record_count": 3052,
    "nonempty_record_count": 3052,
    "normalized_bytes_observed_not_credited": 211176449,
    "normalized_record_inventory_sha256": (
        "bd790b51809018950fcf8d9919f62920ab79aee64f8d73e131cbe16772554495"
    ),
    "source_encoding_counts": {"utf-8": 884, "windows-1251": 2168},
    "pinned_probe_report_sha256": (
        "9d94674323414d30f18517a71edbfee839a27a09a076eb94fe94d74cbaab53c8"
    ),
    "archive_sha256": (
        "0b9e8ed8fe8aa663a68d2bc4eba858a754c626391dd7b5c461d50c3b6260df63"
    ),
    "entry_identity_sha256": (
        "3939c8a407a910222173bba5a2df1b9c7d2edce1503c13b31a9c293d61c3a603"
    ),
}

EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d[\d ()-]{7,}\d)(?!\w)")
SHA256_RE = re.compile(r"[0-9a-f]{64}")
SHA1_RE = re.compile(r"[0-9a-f]{40}")

PARENT_RECORD_FIELDS = {
    "record_id",
    "source_path",
    "source_encoding",
    "raw_bytes",
    "raw_sha256",
    "normalized_bytes",
    "normalized_sha256",
    "text",
}
ACCEPTED_RECORD_FIELDS = [
    "record_id",
    "parent_record_id",
    "source_path",
    "source_encoding",
    "chunk_index",
    "normalized_bytes",
    "normalized_sha256",
    "text",
]


class QualityPrivacyError(RuntimeError):
    """Fail-closed quality/privacy materialization error."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise QualityPrivacyError(message)


def _require_exact_int(value: Any, expected: int, message: str) -> None:
    _require(type(value) is int and value == expected, message)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _load_json(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise QualityPrivacyError(f"cannot load JSON: {path}") from exc
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value, raw


def _validate_config(config: Mapping[str, Any]) -> None:
    _require(config.get("schema_version") == CONFIG_SCHEMA, "config schema drift")
    _require(config.get("worker_id") == WORKER_ID, "worker identity drift")
    _require(config.get("local_free_only") is True, "LOCAL_FREE boundary weakened")

    parent = config.get("parent_normalization")
    _require(isinstance(parent, Mapping), "parent_normalization missing")
    _require(dict(parent) == EXPECTED_PARENT_BINDING, "parent authority binding drift")

    chunking = config.get("chunking")
    _require(isinstance(chunking, Mapping), "chunking policy missing")
    _require(
        dict(chunking)
        == {
            "name": "DATA228_DATA181_GENERIC_NATURAL_TEXT_V1",
            "max_chars": 1200,
            "min_chars": 80,
        },
        "chunking policy drift",
    )

    quality = config.get("quality_privacy")
    _require(isinstance(quality, Mapping), "quality/privacy policy missing")
    _require(quality.get("name") == "DATA228_D03_PREVIEW_V1", "quality policy drift")
    _require(quality.get("min_chars") == 60, "quality min_chars drift")
    _require(quality.get("max_chars") == 1600, "quality max_chars drift")
    _require(quality.get("min_alpha_ratio") == 0.35, "alpha-ratio threshold drift")
    for key in (
        "reject_control_characters",
        "reject_email",
        "reject_phone",
        "reject_empty",
    ):
        _require(quality.get(key) is True, f"quality/privacy predicate weakened: {key}")

    output = config.get("output_contract")
    _require(isinstance(output, Mapping), "output_contract missing")
    _require(
        output.get("accepted_jsonl_fields") == ACCEPTED_RECORD_FIELDS,
        "accepted fields drift",
    )
    _require(
        output.get("preserve_source_encoding_provenance") is True,
        "source-encoding provenance disabled",
    )
    _require(output.get("rejected_text_emitted") is False, "rejected text enabled")
    _require(output.get("rejected_hashes_emitted") is False, "rejected hashes enabled")
    for key in (
        "deterministic_json_serialization",
        "self_hashed_report",
        "two_clean_builds_required",
    ):
        _require(output.get(key) is True, f"output invariant weakened: {key}")

    required_downstream = {
        "GLOBAL_CROSS_SOURCE_EXACT_NEAR_DEDUP",
        "EVALUATION_DECONTAMINATION",
        "BALANCE_DIVERSITY_AND_FAMILY_CAP_RETEST",
        "DETERMINISTIC_SPLIT_SHARD_PACK",
        "TWO_CLEAN_BYTE_IDENTICAL_BUILDS",
        "UNIQUE_CAUSAL_LOSS_LEDGER",
        "TOKENIZER_FIT_AUTHORIZATION",
        "D05_CHECKPOINT_REQUALIFICATION",
        "LEARNED_20M_COMPUTE_AUTHORIZATION",
    }
    downstream = config.get("downstream_required")
    _require(isinstance(downstream, list), "downstream_required missing")
    _require(set(downstream) == required_downstream, "downstream gate set drift")
    _require(len(downstream) == len(required_downstream), "duplicate downstream gates")

    boundary = config.get("claim_boundary")
    _require(isinstance(boundary, Mapping), "claim_boundary missing")
    for key in (
        "bulk_source_admitted",
        "canonical_capacity_credited",
        "tokenizer_fit_authorized",
        "model_training_executed",
        "paid_compute_used",
        "evaluation_authorized",
        "final_test_accessed",
        "research_corpus_v1_released",
        "learned_20m_claimed",
    ):
        _require(boundary.get(key) is False, f"truth boundary weakened: {key}")
    _require_exact_int(boundary.get("training_authorized_bytes"), 0, "training bytes nonzero")
    _require_exact_int(boundary.get("optimizer_updates"), 0, "optimizer updates nonzero")
    _require(boundary.get("safe_result") == SAFE_RESULT, "safe result drift")


def _verify_parent_manifest(manifest: Mapping[str, Any]) -> None:
    _require(
        manifest.get("schema_version") == PARENT_MANIFEST_SCHEMA,
        "parent manifest schema drift",
    )
    _require(manifest.get("worker_id") == PARENT_WORKER_ID, "parent worker drift")
    _require(manifest.get("local_free_only") is True, "parent LOCAL_FREE weakened")
    _require(manifest.get("safe_result") == PARENT_SAFE_RESULT, "parent result drift")
    _require_exact_int(manifest.get("training_authorized_bytes"), 0, "parent grants training")
    _require_exact_int(manifest.get("normalized_capacity_credited"), 0, "parent grants capacity")
    _require(manifest.get("tokenizer_fit_authorized") is False, "parent tokenizer open")
    _require(manifest.get("model_training_executed") is False, "parent training claim")
    _require(manifest.get("paid_compute_used") is False, "parent paid compute claim")
    _require(manifest.get("research_corpus_v1_released") is False, "parent release claim")

    gates = manifest.get("gates")
    _require(isinstance(gates, Mapping), "parent gates missing")
    _require(gates.get("exact_probe_inventory") == "PASS", "parent probe not PASS")
    _require(gates.get("canonical_normalization") == "PASS", "parent norm not PASS")
    for key in (
        "quality",
        "privacy",
        "global_cross_source_dedup",
        "evaluation_decontamination",
        "balance_diversity",
        "corpus_materialization",
        "unique_loss_ledger",
    ):
        _require(gates.get(key) == "NOT_RUN", f"parent downstream gate drift: {key}")

    normalization = manifest.get("normalization")
    _require(isinstance(normalization, Mapping), "parent normalization missing")
    _require(
        normalization.get("name") == PARENT_NORMALIZER_NAME,
        "parent normalizer drift",
    )
    counts = normalization.get("source_encoding_counts")
    _require(isinstance(counts, Mapping), "parent encoding counts missing")
    for key, value in counts.items():
        _require(key in ALLOWED_SOURCE_ENCODINGS, "parent source encoding drift")
        _require(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0,
            "parent source-encoding count invalid",
        )
    _require(
        sum(int(value) for value in counts.values()) == normalization.get("record_count"),
        "parent source-encoding count total drift",
    )

    identity = manifest.get("manifest_identity_sha256")
    _require(
        isinstance(identity, str) and SHA256_RE.fullmatch(identity) is not None,
        "parent manifest identity invalid",
    )
    unsigned = copy.deepcopy(dict(manifest))
    unsigned.pop("manifest_identity_sha256", None)
    _require(_sha256(_canonical_bytes(unsigned)) == identity, "parent manifest self-hash mismatch")

    parent_probe = manifest.get("parent_probe")
    _require(isinstance(parent_probe, Mapping), "parent probe binding missing")
    probe_head = parent_probe.get("head_sha")
    _require(
        isinstance(probe_head, str) and SHA1_RE.fullmatch(probe_head) is not None,
        "probe head invalid",
    )
    for key in (
        "probe_config_identity_sha256",
        "probe_report_sha256",
        "archive_sha256",
        "entry_identity_sha256",
    ):
        value = parent_probe.get(key)
        _require(
            isinstance(value, str) and SHA256_RE.fullmatch(value) is not None,
            f"parent probe identity invalid: {key}",
        )


def _verify_exact_parent_authority(
    parent_jsonl: bytes,
    manifest: Mapping[str, Any],
    *,
    parent_manifest_sha256: str,
) -> None:
    expected = EXPECTED_PARENT_BINDING
    _require(
        parent_manifest_sha256 == expected["manifest_transport_sha256"],
        "exact parent manifest transport SHA-256 mismatch",
    )
    _require(
        manifest.get("manifest_identity_sha256") == expected["manifest_identity_sha256"],
        "exact parent manifest identity mismatch",
    )
    normalization = manifest["normalization"]
    _require(
        _sha256(parent_jsonl) == expected["jsonl_sha256"],
        "exact parent JSONL SHA-256 mismatch",
    )
    _require(normalization.get("jsonl_sha256") == expected["jsonl_sha256"], "parent JSONL binding drift")
    _require(normalization.get("record_count") == expected["record_count"], "parent record count drift")
    _require(
        normalization.get("nonempty_record_count") == expected["nonempty_record_count"],
        "parent nonempty record count drift",
    )
    _require(
        normalization.get("normalized_bytes_observed_not_credited")
        == expected["normalized_bytes_observed_not_credited"],
        "parent normalized byte total drift",
    )
    _require(
        normalization.get("normalized_record_inventory_sha256")
        == expected["normalized_record_inventory_sha256"],
        "parent normalized inventory drift",
    )
    _require(
        dict(normalization.get("source_encoding_counts", {}))
        == expected["source_encoding_counts"],
        "parent source-encoding distribution drift",
    )
    probe = manifest["parent_probe"]
    _require(
        probe.get("probe_report_sha256") == expected["pinned_probe_report_sha256"],
        "parent pinned probe identity drift",
    )
    _require(probe.get("archive_sha256") == expected["archive_sha256"], "parent archive identity drift")
    _require(
        probe.get("entry_identity_sha256") == expected["entry_identity_sha256"],
        "parent entry inventory identity drift",
    )


def _parse_parent_records(
    parent_jsonl: bytes,
    manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    normalization = manifest["normalization"]
    expected_jsonl_sha = normalization.get("jsonl_sha256")
    _require(
        isinstance(expected_jsonl_sha, str)
        and SHA256_RE.fullmatch(expected_jsonl_sha) is not None,
        "parent JSONL identity invalid",
    )
    _require(_sha256(parent_jsonl) == expected_jsonl_sha, "parent JSONL SHA-256 mismatch")

    manifest_records = manifest.get("records")
    _require(isinstance(manifest_records, list), "parent record manifest missing")
    expected_by_id: dict[str, Mapping[str, Any]] = {}
    for metadata in manifest_records:
        _require(isinstance(metadata, Mapping), "parent metadata must be object")
        record_id = metadata.get("record_id")
        _require(isinstance(record_id, str) and record_id, "parent record_id invalid")
        _require(record_id not in expected_by_id, f"duplicate manifest record_id: {record_id}")
        expected_by_id[record_id] = metadata

    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    observed_encodings: Counter[str] = Counter()
    try:
        decoded = parent_jsonl.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise QualityPrivacyError("parent JSONL is not strict UTF-8") from exc

    for line_number, line in enumerate(decoded.splitlines(), start=1):
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise QualityPrivacyError(f"parent JSONL line {line_number} is invalid") from exc
        _require(isinstance(row, dict), f"parent line {line_number} must be object")
        _require(set(row) == PARENT_RECORD_FIELDS, f"parent record field drift at line {line_number}")
        record_id = row.get("record_id")
        _require(isinstance(record_id, str) and record_id, "parent record_id missing")
        _require(record_id not in seen, f"duplicate parent record_id: {record_id}")
        seen.add(record_id)
        source_encoding = row.get("source_encoding")
        _require(source_encoding in ALLOWED_SOURCE_ENCODINGS, f"parent source encoding invalid: {record_id}")
        observed_encodings[str(source_encoding)] += 1
        metadata = expected_by_id.get(record_id)
        _require(metadata is not None, f"parent record absent from manifest: {record_id}")
        expected_metadata = {key: value for key, value in row.items() if key != "text"}
        _require(dict(metadata) == expected_metadata, f"parent metadata mismatch: {record_id}")
        text = row.get("text")
        _require(isinstance(text, str), f"parent text is not a string: {record_id}")
        encoded = text.encode("utf-8")
        _require(row.get("normalized_bytes") == len(encoded), f"parent normalized byte drift: {record_id}")
        _require(row.get("normalized_sha256") == _sha256(encoded), f"parent normalized hash drift: {record_id}")
        raw_sha = row.get("raw_sha256")
        _require(
            isinstance(raw_sha, str) and SHA256_RE.fullmatch(raw_sha) is not None,
            f"parent raw hash invalid: {record_id}",
        )
        records.append(row)

    _require(seen == set(expected_by_id), "parent JSONL/manifest record coverage mismatch")
    _require(normalization.get("record_count") == len(records), "parent record count drift")
    _require(
        dict(sorted(observed_encodings.items())) == dict(normalization.get("source_encoding_counts", {})),
        "parent source-encoding counts do not match records",
    )
    _require(
        normalization.get("normalized_bytes_observed_not_credited")
        == sum(int(record["normalized_bytes"]) for record in records),
        "parent normalized byte total drift",
    )
    return records


def _chunk_text(text: str, *, max_chars: int, min_chars: int) -> tuple[str, ...]:
    _require(max_chars >= min_chars >= 20, "invalid chunk limits")
    paragraphs = [part.strip() for part in text.split("\n") if part.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    def flush() -> None:
        nonlocal current, current_len
        if current:
            value = "\n".join(current).strip()
            if len(value) >= min_chars:
                chunks.append(value)
            current = []
            current_len = 0

    for paragraph in paragraphs:
        if len(paragraph) <= max_chars:
            pieces = [paragraph]
        else:
            pieces: list[str] = []
            words = paragraph.split()
            part: list[str] = []
            part_len = 0
            for word in words:
                needed = len(word) if not part else len(word) + 1
                if part and part_len + needed > max_chars:
                    pieces.append(" ".join(part))
                    part = [word]
                    part_len = len(word)
                else:
                    part.append(word)
                    part_len += needed
            if part:
                pieces.append(" ".join(part))

        for piece in pieces:
            needed = len(piece) if not current else len(piece) + 1
            if current and current_len + needed > max_chars:
                flush()
            current.append(piece)
            current_len += needed
    flush()
    return tuple(chunks)


def _quality_reason(
    text: str,
    *,
    min_chars: int,
    max_chars: int,
    min_alpha_ratio: float,
) -> str | None:
    if len(text) < min_chars:
        return "too_short"
    if len(text) > max_chars:
        return "too_long"
    if any(unicodedata.category(char) == "Cc" and char not in "\n\t" for char in text):
        return "control_character"
    if EMAIL_RE.search(text):
        return "pii_email"
    if PHONE_RE.search(text):
        return "pii_phone"
    visible = [char for char in text if not char.isspace()]
    if not visible:
        return "empty"
    alpha_ratio = sum(char.isalpha() for char in visible) / len(visible)
    if alpha_ratio < min_alpha_ratio:
        return "low_alpha_ratio"
    return None


def _materialize_quality_privacy_candidate(
    parent_jsonl: bytes,
    parent_manifest: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    parent_manifest_sha256: str,
    enforce_exact_parent: bool,
) -> tuple[bytes, dict[str, Any]]:
    _validate_config(config)
    _verify_parent_manifest(parent_manifest)
    _require(
        isinstance(parent_manifest_sha256, str)
        and SHA256_RE.fullmatch(parent_manifest_sha256) is not None,
        "parent manifest transport SHA-256 invalid",
    )
    if enforce_exact_parent:
        _verify_exact_parent_authority(
            parent_jsonl,
            parent_manifest,
            parent_manifest_sha256=parent_manifest_sha256,
        )
    records = _parse_parent_records(parent_jsonl, parent_manifest)

    chunking = config["chunking"]
    quality = config["quality_privacy"]
    accepted: list[dict[str, Any]] = []
    rejected_reasons: Counter[str] = Counter()
    zero_chunk_parent_count = 0
    total_chunks = 0

    for parent in records:
        chunks = _chunk_text(
            parent["text"],
            max_chars=int(chunking["max_chars"]),
            min_chars=int(chunking["min_chars"]),
        )
        if not chunks:
            zero_chunk_parent_count += 1
        for chunk_index, chunk in enumerate(chunks):
            total_chunks += 1
            reason = _quality_reason(
                chunk,
                min_chars=int(quality["min_chars"]),
                max_chars=int(quality["max_chars"]),
                min_alpha_ratio=float(quality["min_alpha_ratio"]),
            )
            if reason is not None:
                rejected_reasons[reason] += 1
                continue
            encoded = chunk.encode("utf-8")
            accepted.append(
                {
                    "record_id": f"{parent['record_id']}.q{chunk_index:05d}",
                    "parent_record_id": parent["record_id"],
                    "source_path": parent["source_path"],
                    "source_encoding": parent["source_encoding"],
                    "chunk_index": chunk_index,
                    "normalized_bytes": len(encoded),
                    "normalized_sha256": _sha256(encoded),
                    "text": chunk,
                }
            )

    accepted.sort(key=lambda row: (row["parent_record_id"], int(row["chunk_index"])))
    accepted_ids = [str(row["record_id"]) for row in accepted]
    _require(len(accepted_ids) == len(set(accepted_ids)), "accepted record_id collision")
    accepted_jsonl = b"".join(_canonical_bytes(row) + b"\n" for row in accepted)
    accepted_metadata = [
        {key: value for key, value in row.items() if key != "text"} for row in accepted
    ]
    inventory_hasher = hashlib.sha256()
    for row in accepted_metadata:
        inventory_hasher.update(_canonical_bytes(row))
        inventory_hasher.update(b"\n")

    accepted_hashes = [str(row["normalized_sha256"]) for row in accepted]
    exact_duplicate_observations = len(accepted_hashes) - len(set(accepted_hashes))
    accepted_bytes = sum(int(row["normalized_bytes"]) for row in accepted)
    accepted_encoding_counts = Counter(str(row["source_encoding"]) for row in accepted)
    rejected_count = sum(rejected_reasons.values())
    _require(total_chunks == len(accepted) + rejected_count, "chunk accounting mismatch")

    parent_binding = config["parent_normalization"]
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA,
        "worker_id": WORKER_ID,
        "local_free_only": True,
        "parent_normalization": {
            "pr": parent_binding["pr"],
            "head_sha": parent_binding["head_sha"],
            "branch": parent_binding["branch"],
            "execution_head_sha": parent_binding["execution_head_sha"],
            "execution_run_id": parent_binding["execution_run_id"],
            "execution_evidence_identity_sha256": parent_binding[
                "execution_evidence_identity_sha256"
            ],
            "manifest_identity_sha256": parent_manifest["manifest_identity_sha256"],
            "manifest_transport_sha256": parent_manifest_sha256,
            "jsonl_sha256": parent_manifest["normalization"]["jsonl_sha256"],
            "source_family": SOURCE_FAMILY,
            "source_encoding_counts": dict(
                parent_manifest["normalization"]["source_encoding_counts"]
            ),
            "exact_parent_authority_enforced": enforce_exact_parent,
        },
        "policy": {
            "chunking": dict(chunking),
            "quality_privacy": dict(quality),
            "bounded_predicates_not_universal_privacy_claim": True,
            "source_encoding_provenance_preserved": True,
        },
        "input": {
            "parent_record_count": len(records),
            "parent_normalized_bytes": sum(int(row["normalized_bytes"]) for row in records),
            "zero_chunk_parent_count": zero_chunk_parent_count,
        },
        "filter_result": {
            "total_chunks": total_chunks,
            "accepted_chunk_count": len(accepted),
            "rejected_chunk_count": rejected_count,
            "rejection_reasons": dict(sorted(rejected_reasons.items())),
            "accepted_bytes_observed_not_credited": accepted_bytes,
            "accepted_source_encoding_counts": dict(sorted(accepted_encoding_counts.items())),
            "accepted_jsonl_sha256": _sha256(accepted_jsonl),
            "accepted_inventory_sha256": inventory_hasher.hexdigest(),
            "exact_duplicate_accepted_hashes_observed_not_removed": exact_duplicate_observations,
            "rejected_text_emitted": False,
            "rejected_hashes_emitted": False,
        },
        "gates": {
            "parent_manifest_integrity": "PASS",
            "parent_jsonl_integrity": "PASS",
            "exact_parent_authority": "PASS" if enforce_exact_parent else "TEST_FIXTURE_ONLY",
            "deterministic_chunking": "PASS",
            "bounded_quality_privacy_filter_execution": "PASS",
            "global_cross_source_dedup": "NOT_RUN",
            "evaluation_decontamination": "NOT_RUN",
            "balance_diversity": "NOT_RUN",
            "corpus_materialization": "NOT_RUN",
            "unique_loss_ledger": "NOT_RUN",
            "d05_checkpoint_requalification": "NOT_RUN",
        },
        "accepted_records": accepted_metadata,
        "canonical_capacity_credited": 0,
        "training_authorized_bytes": 0,
        "tokenizer_fit_authorized": False,
        "model_training_executed": False,
        "optimizer_updates": 0,
        "paid_compute_used": False,
        "evaluation_authorized": False,
        "final_test_accessed": False,
        "research_corpus_v1_released": False,
        "learned_20m_claimed": False,
        "safe_result": SAFE_RESULT,
    }
    report["report_identity_sha256"] = _sha256(_canonical_bytes(report))
    return accepted_jsonl, report


def materialize_quality_privacy_candidate(
    parent_jsonl: bytes,
    parent_manifest: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    parent_manifest_sha256: str,
) -> tuple[bytes, dict[str, Any]]:
    """Production API: exact current parent authority is always enforced."""
    return _materialize_quality_privacy_candidate(
        parent_jsonl,
        parent_manifest,
        config,
        parent_manifest_sha256=parent_manifest_sha256,
        enforce_exact_parent=True,
    )


def _materialize_quality_privacy_candidate_for_test(
    parent_jsonl: bytes,
    parent_manifest: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    parent_manifest_sha256: str,
) -> tuple[bytes, dict[str, Any]]:
    """Synthetic unit-fixture seam; never used by CLI or production execution."""
    return _materialize_quality_privacy_candidate(
        parent_jsonl,
        parent_manifest,
        config,
        parent_manifest_sha256=parent_manifest_sha256,
        enforce_exact_parent=False,
    )


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-jsonl", type=Path, required=True)
    parser.add_argument("--parent-manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config, _ = _load_json(args.config)
    manifest, manifest_bytes = _load_json(args.parent_manifest)
    try:
        parent_jsonl = args.parent_jsonl.read_bytes()
    except OSError as exc:
        raise QualityPrivacyError(f"cannot read parent JSONL: {args.parent_jsonl}") from exc

    accepted_jsonl, report = materialize_quality_privacy_candidate(
        parent_jsonl,
        manifest,
        config,
        parent_manifest_sha256=_sha256(manifest_bytes),
    )
    report_bytes = json.dumps(
        report,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    ).encode("utf-8") + b"\n"
    _atomic_write(args.output_jsonl, accepted_jsonl)
    _atomic_write(args.output_report, report_bytes)
    print(
        json.dumps(
            {
                "status": "PASS_QUALITY_PRIVACY_FILTERED_CANDIDATE_ONLY",
                "accepted_chunk_count": report["filter_result"]["accepted_chunk_count"],
                "rejected_chunk_count": report["filter_result"]["rejected_chunk_count"],
                "accepted_bytes_observed_not_credited": report["filter_result"][
                    "accepted_bytes_observed_not_credited"
                ],
                "accepted_source_encoding_counts": report["filter_result"][
                    "accepted_source_encoding_counts"
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
