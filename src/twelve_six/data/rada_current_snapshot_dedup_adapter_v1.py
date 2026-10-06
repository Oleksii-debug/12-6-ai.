"""Project the exact current Rada replay candidate into global-dedup inputs.

The adapter preserves every accepted chunk and remains zero-credit. It is a
consumer seam for global cross-source dedup only; it does not establish
production Q/P, training rights, tokenizer fit, or corpus admission.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from twelve_six.data.rada_current_snapshot_qp_authority_v1 import (
    CANONICAL_AUTHORITY_FILE_SHA256,
    AUTHORITY_IDENTITY_SHA256,
    RadaAcceptedPayloadBinding,
    accepted_payload_binding,
    load_current_rada_replay_authority,
    verify_current_product_mechanics,
)


SOURCE_FAMILY = "ua.rada.open-data.laws-texts"
ARCHIVE_URL = "https://data.rada.gov.ua/ogd/zak/perv/text/texts.zip"
RECEIPT_SCHEMA = "12-6.d03-rada-current-snapshot-dedup-projection.v1"
ROW_KEYS = frozenset(
    {
        "record_id",
        "parent_record_id",
        "source_path",
        "source_encoding",
        "chunk_index",
        "normalized_bytes",
        "normalized_sha256",
        "text",
    }
)
METADATA_KEYS = ROW_KEYS - {"text"}
ALLOWED_ENCODINGS = frozenset({"utf-8", "windows-1251"})
PARENT_ID_RE = re.compile(r"d[0-9]+\.htm")
SOURCE_PATH_RE = re.compile(r"zak/perv/text/d[0-9]+\.htm")
SHA256_RE = re.compile(r"[0-9a-f]{64}")


class RadaCurrentSnapshotDedupAdapterError(RuntimeError):
    """Raised when current Rada candidate projection fails closed."""


@dataclass(frozen=True)
class RadaCurrentSnapshotProjection:
    receipt: dict[str, Any]
    sources: tuple[dict[str, Any], ...] | None
    payloads: dict[str, bytes] | None


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RadaCurrentSnapshotDedupAdapterError(message)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RadaCurrentSnapshotDedupAdapterError(
                f"duplicate candidate JSON key: {key}"
            )
        result[key] = value
    return result


def _bounded_int(token: str) -> int:
    digits = token[1:] if token.startswith("-") else token
    if not digits or len(digits) > 20:
        raise RadaCurrentSnapshotDedupAdapterError(
            "candidate integer token exceeds bound"
        )
    return int(token)


def _reject_float(token: str) -> float:
    del token
    raise RadaCurrentSnapshotDedupAdapterError(
        "candidate JSON floats are not permitted"
    )


def _reject_constant(token: str) -> float:
    del token
    raise RadaCurrentSnapshotDedupAdapterError(
        "candidate non-finite JSON is not permitted"
    )


def _strict_row(line: bytes, *, line_number: int) -> dict[str, Any]:
    _require(line.endswith(b"\n"), f"candidate row {line_number} lacks LF terminator")
    _require(
        not line.endswith(b"\r\n"),
        f"candidate row {line_number} uses CRLF instead of canonical LF",
    )
    try:
        row = json.loads(
            line[:-1].decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_int=_bounded_int,
            parse_float=_reject_float,
            parse_constant=_reject_constant,
        )
    except RadaCurrentSnapshotDedupAdapterError:
        raise
    except RecursionError as exc:
        raise RadaCurrentSnapshotDedupAdapterError(
            f"candidate row {line_number} nesting limit exceeded"
        ) from exc
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise RadaCurrentSnapshotDedupAdapterError(
            f"candidate row {line_number} is not strict UTF-8 JSON"
        ) from exc
    _require(
        type(row) is dict and set(row) == ROW_KEYS,
        f"candidate row {line_number} schema drift",
    )
    return row


def _validate_row(
    row: Mapping[str, Any],
    *,
    line_number: int,
) -> tuple[dict[str, Any], str, bytes, tuple[str, int]]:
    _require(type(row) is dict, f"candidate row {line_number} is not exact object")
    _require(set(row) == ROW_KEYS, f"candidate row {line_number} schema drift")

    record_id = row["record_id"]
    parent_id = row["parent_record_id"]
    source_path = row["source_path"]
    encoding = row["source_encoding"]
    chunk_index = row["chunk_index"]
    normalized_bytes = row["normalized_bytes"]
    normalized_sha = row["normalized_sha256"]
    text = row["text"]

    _require(
        type(parent_id) is str and PARENT_ID_RE.fullmatch(parent_id) is not None,
        f"candidate parent_record_id invalid at row {line_number}",
    )
    parent_basename = parent_id[len(PARENT_ID_PREFIX) :] + ".htm"
    _require(
        type(source_path) is str
        and SOURCE_PATH_RE.fullmatch(source_path) is not None
        and source_path == parent_basename,
        f"candidate source_path invalid at row {line_number}",
    )
    _require(
        type(encoding) is str and encoding in ALLOWED_ENCODINGS,
        f"candidate source_encoding invalid at row {line_number}",
    )
    _require(
        type(chunk_index) is int and 0 <= chunk_index <= 99999,
        f"candidate chunk_index invalid at row {line_number}",
    )
    expected_record_id = f"{parent_id}.q{chunk_index:05d}"
    _require(
        type(record_id) is str and record_id == expected_record_id,
        f"candidate record_id binding drift at row {line_number}",
    )
    _require(
        type(text) is str and bool(text),
        f"candidate text missing at row {line_number}",
    )
    payload = text.encode("utf-8")
    _require(
        type(normalized_bytes) is int
        and normalized_bytes > 0
        and normalized_bytes == len(payload),
        f"candidate normalized byte drift at row {line_number}",
    )
    _require(
        type(normalized_sha) is str
        and SHA256_RE.fullmatch(normalized_sha) is not None
        and _sha256(payload) == normalized_sha,
        f"candidate normalized SHA drift at row {line_number}",
    )

    source_id = f"rada-laws-qp:{record_id}"
    matcher_row = {
        "source_id": source_id,
        "source_family": SOURCE_FAMILY,
        "stable_origin_id": f"rada-laws-document:{parent_id}",
        "stable_object_id": f"sha256:{normalized_sha}",
        "modality": "uk",
        "evidence_status": "CURRENT_SNAPSHOT_REPLAY_ZERO_CREDIT",
        "authority_ref": f"current-rada-replay:{AUTHORITY_IDENTITY_SHA256}",
        "declared_capacity_bytes": len(payload),
        "expected_raw_bytes": len(payload),
        "expected_raw_sha256": normalized_sha,
        "acquisition_url": ARCHIVE_URL,
        "origin_key": f"rada-laws-unit:{record_id}",
    }
    return matcher_row, source_id, payload, (parent_id, chunk_index)


def _read_candidate(
    candidate_jsonl: Path,
    binding: RadaAcceptedPayloadBinding,
    *,
    retain_payloads: bool,
) -> RadaCurrentSnapshotProjection:
    _require(type(retain_payloads) is bool, "retain_payloads must be exact bool")
    try:
        metadata = os.lstat(candidate_jsonl)
    except OSError as exc:
        raise RadaCurrentSnapshotDedupAdapterError(
            f"cannot stat candidate JSONL: {candidate_jsonl}"
        ) from exc
    _require(
        stat.S_ISREG(metadata.st_mode),
        f"candidate JSONL is not a regular file: {candidate_jsonl}",
    )

    try:
        handle = candidate_jsonl.open("rb")
    except OSError as exc:
        raise RadaCurrentSnapshotDedupAdapterError(
            f"cannot open candidate JSONL: {candidate_jsonl}"
        ) from exc

    sources: list[dict[str, Any]] | None = [] if retain_payloads else None
    payloads: dict[str, bytes] | None = {} if retain_payloads else None
    transport_hasher = hashlib.sha256()
    inventory_hasher = hashlib.sha256()
    projection_hasher = hashlib.sha256()
    seen_source_ids: set[str] = set()
    seen_payload_hashes: set[str] = set()
    encoding_counts: Counter[str] = Counter()
    duplicate_hashes = 0
    transport_bytes = 0
    payload_bytes = 0
    row_count = 0
    previous_order: tuple[str, int] | None = None

    with handle:
        descriptor = os.fstat(handle.fileno())
        _require(
            stat.S_ISREG(descriptor.st_mode),
            "candidate descriptor is not a regular file",
        )
        _require(
            metadata.st_dev == descriptor.st_dev
            and metadata.st_ino == descriptor.st_ino,
            "candidate path changed before descriptor lock",
        )
        for line_number, line in enumerate(handle, 1):
            transport_hasher.update(line)
            transport_bytes += len(line)
            row = _strict_row(line, line_number=line_number)
            matcher_row, source_id, payload, order_key = _validate_row(
                row,
                line_number=line_number,
            )
            if previous_order is not None:
                _require(
                    previous_order < order_key,
                    f"candidate canonical order drift at row {line_number}",
                )
            previous_order = order_key
            _require(
                source_id not in seen_source_ids,
                f"candidate duplicate source_id at row {line_number}",
            )
            seen_source_ids.add(source_id)

            metadata_row = {key: row[key] for key in METADATA_KEYS}
            inventory_hasher.update(_canonical(metadata_row) + b"\n")
            projection_hasher.update(_canonical(matcher_row) + b"\n")
            normalized_sha = row["normalized_sha256"]
            if normalized_sha in seen_payload_hashes:
                duplicate_hashes += 1
            else:
                seen_payload_hashes.add(normalized_sha)
            encoding_counts[str(row["source_encoding"])] += 1
            payload_bytes += len(payload)
            row_count += 1
            if sources is not None and payloads is not None:
                sources.append(matcher_row)
                payloads[source_id] = payload

        final_descriptor = os.fstat(handle.fileno())
        _require(
            descriptor.st_dev == final_descriptor.st_dev
            and descriptor.st_ino == final_descriptor.st_ino,
            "candidate descriptor identity changed during read",
        )

    _require(
        transport_bytes == binding.accepted_jsonl_file_bytes,
        "candidate transport byte total drift",
    )
    _require(
        transport_hasher.hexdigest() == binding.accepted_jsonl_sha256,
        "candidate transport SHA-256 drift",
    )
    _require(
        row_count == binding.accepted_chunk_count,
        "candidate accepted chunk count drift",
    )
    _require(
        payload_bytes == binding.accepted_payload_utf8_bytes,
        "candidate accepted payload byte total drift",
    )
    _require(
        inventory_hasher.hexdigest() == binding.accepted_inventory_sha256,
        "candidate accepted inventory SHA-256 drift",
    )
    _require(
        dict(sorted(encoding_counts.items()))
        == dict(binding.accepted_source_encoding_counts),
        "candidate source-encoding distribution drift",
    )
    _require(
        duplicate_hashes
        == binding.exact_duplicate_payload_hashes_observed_not_removed,
        "candidate exact-duplicate observation drift",
    )

    receipt_core: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "source_family": SOURCE_FAMILY,
        "replay_authority_identity_sha256": AUTHORITY_IDENTITY_SHA256,
        "candidate_jsonl_sha256": binding.accepted_jsonl_sha256,
        "candidate_jsonl_file_bytes": binding.accepted_jsonl_file_bytes,
        "source_object_count": row_count,
        "source_payload_utf8_bytes": payload_bytes,
        "matcher_source_inventory_sha256": projection_hasher.hexdigest(),
        "accepted_inventory_sha256": binding.accepted_inventory_sha256,
        "exact_duplicate_payload_hashes_preserved": duplicate_hashes,
        "source_encoding_counts": dict(sorted(encoding_counts.items())),
        "requires_replace_existing_source_family": True,
        "must_not_append_to_existing_source_family": True,
        "raw_text_emitted": False,
        "consumer_gate": "CURRENT_GLOBAL_CROSS_SOURCE_DEDUP_ONLY",
        "rights_scope": binding.rights_scope,
        "bulk_corpus_admission_granted": False,
        "training_authority_granted": False,
        "rights_recheck_for_training_required": True,
        "canonical_capacity_credited": 0,
        "training_authorized_bytes": 0,
        "authorized_optimized_target_exposure": 0,
        "tokenizer_fit_authorized": False,
        "training_executed": False,
        "learned_weights_created": False,
        "final_test_outcomes_read": False,
        "paid_compute_used": False,
    }
    receipt = {
        **receipt_core,
        "receipt_identity_sha256": _sha256(_canonical(receipt_core)),
    }
    return RadaCurrentSnapshotProjection(
        receipt=receipt,
        sources=tuple(sources) if sources is not None else None,
        payloads=payloads,
    )


def validate_and_project_current_rada_candidate(
    candidate_jsonl: Path,
    replay_authority_path: Path,
    *,
    repository_root: Path,
    expected_authority_raw_sha256: str = CANONICAL_AUTHORITY_FILE_SHA256,
    retain_payloads: bool = True,
) -> RadaCurrentSnapshotProjection:
    """Validate exact current replay bytes and project them one-to-one for dedup."""

    replay_authority = load_current_rada_replay_authority(
        replay_authority_path,
        expected_raw_sha256=expected_authority_raw_sha256,
    )
    verify_current_product_mechanics(
        replay_authority,
        repository_root=repository_root,
    )
    binding = accepted_payload_binding(replay_authority)
    _require(
        binding.source_family == SOURCE_FAMILY,
        "current Rada source family drift",
    )
    _require(
        binding.bulk_corpus_admission_granted is False
        and binding.training_authority_granted is False
        and binding.rights_recheck_for_training_required is True,
        "current Rada rights boundary widened",
    )
    return _read_candidate(
        candidate_jsonl,
        binding,
        retain_payloads=retain_payloads,
    )
