#!/usr/bin/env python3
"""Filter normalized Rada bulk records through the incumbent quality/privacy seam.

This successor consumes the exact deterministic normalization output from PR #641,
verifies its manifest and JSONL byte-for-byte, chunks long legal documents with the
existing DATA-228/DATA-181 natural-text mechanics, and applies the bounded
DATA-228/D03 quality/privacy predicates per chunk.

It deliberately grants zero canonical capacity and zero training exposure. Global
cross-source deduplication, evaluation decontamination, balance/family caps,
packing, unique-loss accounting, tokenizer authorization, D05 requalification and
material-compute authorization remain downstream gates.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

CONFIG_SCHEMA = "12-6.d03-rada-bulk-quality-privacy.v1"
PARENT_MANIFEST_SCHEMA = "12-6.d03-rada-bulk-normalization-manifest.v1"
REPORT_SCHEMA = "12-6.d03-rada-bulk-quality-privacy-report.v1"
WORKER_ID = "D03-RADA-BULK-QUALITY-PRIVACY-20260826"
PARENT_WORKER_ID = "D03-RADA-BULK-NORMALIZATION-20260826"
PARENT_PR = 641
PARENT_HEAD = "ae79b078f849513dc202bcb723a4145455309e35"
PARENT_BRANCH = "gpt56/d03-rada-bulk-normalization-20260826"
SOURCE_FAMILY = "ua.rada.open-data.laws-texts"
PARENT_SAFE_RESULT = "NORMALIZED_RECORD_MATERIALIZATION_ONLY_DOWNSTREAM_GATES_REQUIRED"
PARENT_NORMALIZER_NAME = "RADA_VISIBLE_TEXT_HTML_UTF8_CP1251_NFKC_V1"
SAFE_RESULT = "QUALITY_PRIVACY_FILTERED_CANDIDATE_ONLY_DOWNSTREAM_GATES_REQUIRED"
DEFAULT_CONFIG = Path("configs/data/d03_rada_bulk_quality_privacy_v1.json")
ALLOWED_SOURCE_ENCODINGS = {"utf-8", "windows-1251"}

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
    expected_parent = {
        "pr": PARENT_PR,
        "head_sha": PARENT_HEAD,
        "branch": PARENT_BRANCH,
        "manifest_schema": PARENT_MANIFEST_SCHEMA,
        "manifest_worker_id": PARENT_WORKER_ID,
        "source_family": SOURCE_FAMILY,
        "safe_result": PARENT_SAFE_RESULT,
    }
    for key, expected in expected_parent.items():
        _require(
            parent.get(key) == expected,
            f"parent normalization binding drift: {key}",
        )

    chunking = config.get("chunking")
    _require(isinstance(chunking, Mapping), "chunking policy missing")
    _require(
        chunking.get("name") == "DATA228_DATA181_GENERIC_NATURAL_TEXT_V1",
        "chunking algorithm drift",
    )
    _require(chunking.get("max_chars") == 1200, "chunk max_chars drift")
    _require(chunking.get("min_chars") == 80, "chunk min_chars drift")

    quality = config.get("quality_privacy")
    _require(isinstance(quality, Mapping), "quality/privacy policy missing")
    _require(
        quality.get("name") == "DATA228_D03_PREVIEW_V1",
        "quality algorithm drift",
    )
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
    _require(
        output.get("rejected_text_emitted") is False,
        "rejected text emission enabled",
    )
    _require(
        output.get("rejected_hashes_emitted") is False,
        "rejected hash emission enabled",
    )
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
    _require(
        len(downstream) == len(required_downstream),
        "duplicate downstream gates",
    )

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
    _require(
        boundary.get("training_authorized_bytes") == 0,
        "training bytes must remain zero",
    )
    _require(
        boundary.get("optimizer_updates") == 0,
        "optimizer updates must remain zero",
    )
    _require(boundary.get("safe_result") == SAFE_RESULT, "safe result drift")


def _verify_parent_manifest(manifest: Mapping[str, Any]) -> None:
    _require(
        manifest.get("schema_version") == PARENT_MANIFEST_SCHEMA,
        "parent manifest schema drift",
    )
    _require(
        manifest.get("worker_id") == PARENT_WORKER_ID,
        "parent manifest worker drift",
    )
    _require(
        manifest.get("local_free_only") is True,
        "parent LOCAL_FREE boundary weakened",
    )
    _require(
        manifest.get("safe_result") == PARENT_SAFE_RESULT,
        "parent safe result drift",
    )
    _require(
        manifest.get("training_authorized_bytes") == 0,
        "parent grants training bytes",
    )
    _require(
        manifest.get("normalized_capacity_credited") == 0,
        "parent grants canonical capacity",
    )
    _require(
        manifest.get("tokenizer_fit_authorized") is False,
        "parent authorizes tokenizer fit",
    )
    _require(
        manifest.get("model_training_executed") is False,
        "parent claims model training",
    )
    _require(
        manifest.get("paid_compute_used") is False,
        "parent claims paid compute",
    )
    _require(
        manifest.get("research_corpus_v1_released") is False,
        "parent claims corpus release",
    )

    gates = manifest.get("gates")
    _require(isinstance(gates, Mapping), "parent gates missing")
    _require(
        gates.get("exact_probe_inventory") == "PASS",
        "parent probe inventory not PASS",
    )
    _require(
        gates.get("canonical_normalization") == "PASS",
        "parent normalization not PASS",
    )
    _require(gates.get("quality") == "NOT_RUN", "parent quality gate already changed")
    _require(gates.get("privacy") == "NOT_RUN", "parent privacy gate already changed")
    _require(
        gates.get("global_cross_source_dedup") == "NOT_RUN",
        "parent dedup state drift",
    )
    _require(
        gates.get("evaluation_decontamination") == "NOT_RUN",
        "parent decontamination state drift",
    )

    normalization = manifest.get("normalization")
    _require(isinstance(normalization, Mapping), "parent normalization section missing")
    _require(
        normalization.get("name") == PARENT_NORMALIZER_NAME,
        "parent normalizer identity drift",
    )
    source_encoding_counts = normalization.get("source_encoding_counts")
    _require(
        isinstance(source_encoding_counts, Mapping),
        "parent source-encoding counts missing",
    )
    for key, value in source_encoding_counts.items():
        _require(key in ALLOWED_SOURCE_ENCODINGS, "parent source encoding drift")
        _require(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0,
            "parent source-encoding count invalid",
        )
    _require(
        sum(int(value) for value in source_encoding_counts.values())
        == normalization.get("record_count"),
        "parent source-encoding count total drift",
    )

    identity = manifest.get("manifest_identity_sha256")
    _require(
        isinstance(identity, str) and SHA256_RE.fullmatch(identity) is not None,
        "parent manifest identity invalid",
    )
    unsigned = copy.deepcopy(dict(manifest))
    unsigned.pop("manifest_identity_sha256", None)
    _require(
        _sha256(_canonical_bytes(unsigned)) == identity,
        "parent manifest self-hash mismatch",
    )

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


def _parse_parent_records(
    parent_jsonl: bytes,
    manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    normalization = manifest.get("normalization")
    _require(isinstance(normalization, Mapping), "parent normalization section missing")
    expected_jsonl_sha = normalization.get("jsonl_sha256")
    _require(
        isinstance(expected_jsonl_sha, str)
        and SHA256_RE.fullmatch(expected_jsonl_sha) is not None,
        "parent JSONL identity invalid",
    )
    _require(
        _sha256(parent_jsonl) == expected_jsonl_sha,
        "parent JSONL SHA-256 mismatch",
    )

    manifest_records = manifest.get("records")
    _require(isinstance(manifest_records, list), "parent record manifest missing")
    expected_by_id: dict[str, Mapping[str, Any]] = {}
    for metadata in manifest_records:
        _require(
            isinstance(metadata, Mapping),
            "parent record metadata must be an object",
        )
        record_id = metadata.get("record_id")
        _require(isinstance(record_id, str) and record_id, "parent record_id invalid")
        _require(
            record_id not in expected_by_id,
            f"duplicate manifest record_id: {record_id}",
        )
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
            raise QualityPrivacyError(
                f"parent JSONL line {line_number} is invalid"
            ) from exc
        _require(
            isinstance(row, dict),
            f"parent JSONL line {line_number} must be an object",
        )
        _require(
            set(row) == PARENT_RECORD_FIELDS,
            f"parent record field drift at line {line_number}",
        )
        record_id = row.get("record_id")
        _require(isinstance(record_id, str) and record_id, "parent record_id missing")
        _require(record_id not in seen, f"duplicate parent record_id: {record_id}")
        seen.add(record_id)
        source_encoding = row.get("source_encoding")
        _require(
            source_encoding in ALLOWED_SOURCE_ENCODINGS,
            f"parent source encoding invalid: {record_id}",
        )
        observed_encodings[str(source_encoding)] += 1
        metadata = expected_by_id.get(record_id)
        _require(
            metadata is not None,
            f"parent record absent from manifest: {record_id}",
        )
        expected_metadata = {
            key: value for key, value in row.items() if key != "text"
        }
        _require(
            dict(metadata) == expected_metadata,
            f"parent metadata mismatch: {record_id}",
        )
        text = row.get("text")
        _require(isinstance(text, str), f"parent text is not a string: {record_id}")
        encoded = text.encode("utf-8")
        _require(
            row.get("normalized_bytes") == len(encoded),
            f"parent normalized byte drift: {record_id}",
        )
        _require(
            row.get("normalized_sha256") == _sha256(encoded),
            f"parent normalized hash drift: {record_id}",
        )
        raw_sha = row.get("raw_sha256")
        _require(
            isinstance(raw_sha, str) and SHA256_RE.fullmatch(raw_sha) is not None,
            f"parent raw hash invalid: {record_id}",
        )
        records.append(row)

    _require(
        seen == set(expected_by_id),
        "parent JSONL/manifest record coverage mismatch",
    )
    _require(
        normalization.get("record_count") == len(records),
        "parent record count drift",
    )
    _require(
        dict(sorted(observed_encodings.items()))
        == dict(normalization.get("source_encoding_counts", {})),
        "parent source-encoding counts do not match records",
    )
    _require(
        normalization.get("normalized_bytes_observed_not_credited")
        == sum(int(record["normalized_bytes"]) for record in records),
        "parent normalized byte total drift",
    )
    return records


def _chunk_text(text: str, *, max_chars: int, min_chars: int) -> tuple[str, ...]:
    """DATA-228/DATA-181 generic natural-text chunking semantics."""
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
    """Bounded DATA-228/D03 quality/privacy predicate."""
    if len(text) < min_chars:
        return "too_short"
    if len(text) > max_chars:
        return "too_long"
    if any(
        unicodedata.category(char) == "Cc" and char not in "\n\t"
        for char in text
    ):
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


def materialize_quality_privacy_candidate(
    parent_jsonl: bytes,
    parent_manifest: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    parent_manifest_sha256: str,
) -> tuple[bytes, dict[str, Any]]:
    """Return deterministic accepted-chunk JSONL plus a text-free audit report."""
    _validate_config(config)
    _verify_parent_manifest(parent_manifest)
    _require(
        isinstance(parent_manifest_sha256, str)
        and SHA256_RE.fullmatch(parent_manifest_sha256) is not None,
        "parent manifest transport SHA-256 invalid",
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

    accepted.sort(
        key=lambda row: (row["parent_record_id"], int(row["chunk_index"]))
    )
    accepted_ids = [str(row["record_id"]) for row in accepted]
    _require(
        len(accepted_ids) == len(set(accepted_ids)),
        "accepted record_id collision",
    )
    accepted_jsonl = b"".join(_canonical_bytes(row) + b"\n" for row in accepted)
    accepted_metadata = [
        {key: value for key, value in row.items() if key != "text"}
        for row in accepted
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
    _require(
        total_chunks == len(accepted) + rejected_count,
        "chunk accounting mismatch",
    )

    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA,
        "worker_id": WORKER_ID,
        "local_free_only": True,
        "parent_normalization": {
            "pr": PARENT_PR,
            "head_sha": PARENT_HEAD,
            "branch": PARENT_BRANCH,
            "manifest_identity_sha256": parent_manifest[
                "manifest_identity_sha256"
            ],
            "manifest_transport_sha256": parent_manifest_sha256,
            "jsonl_sha256": parent_manifest["normalization"]["jsonl_sha256"],
            "source_family": SOURCE_FAMILY,
            "source_encoding_counts": dict(
                parent_manifest["normalization"]["source_encoding_counts"]
            ),
        },
        "policy": {
            "chunking": dict(chunking),
            "quality_privacy": dict(quality),
            "bounded_predicates_not_universal_privacy_claim": True,
            "source_encoding_provenance_preserved": True,
        },
        "input": {
            "parent_record_count": len(records),
            "parent_normalized_bytes": sum(
                int(row["normalized_bytes"]) for row in records
            ),
            "zero_chunk_parent_count": zero_chunk_parent_count,
        },
        "filter_result": {
            "total_chunks": total_chunks,
            "accepted_chunk_count": len(accepted),
            "rejected_chunk_count": rejected_count,
            "rejection_reasons": dict(sorted(rejected_reasons.items())),
            "accepted_bytes_observed_not_credited": accepted_bytes,
            "accepted_source_encoding_counts": dict(
                sorted(accepted_encoding_counts.items())
            ),
            "accepted_jsonl_sha256": _sha256(accepted_jsonl),
            "accepted_inventory_sha256": inventory_hasher.hexdigest(),
            "exact_duplicate_accepted_hashes_observed_not_removed": (
                exact_duplicate_observations
            ),
            "rejected_text_emitted": False,
            "rejected_hashes_emitted": False,
        },
        "gates": {
            "parent_manifest_integrity": "PASS",
            "parent_jsonl_integrity": "PASS",
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
        raise QualityPrivacyError(
            f"cannot read parent JSONL: {args.parent_jsonl}"
        ) from exc

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
                "accepted_chunk_count": report["filter_result"][
                    "accepted_chunk_count"
                ],
                "rejected_chunk_count": report["filter_result"][
                    "rejected_chunk_count"
                ],
                "accepted_bytes_observed_not_credited": report[
                    "filter_result"
                ]["accepted_bytes_observed_not_credited"],
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