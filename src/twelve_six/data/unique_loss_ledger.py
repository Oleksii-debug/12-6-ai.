"""Deterministic document-isolated unique causal-loss exposure accounting."""
from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from twelve_six.data.real_snapshot_registry import (
    ALLOWED,
    validate_real_snapshot_registry,
)
from twelve_six.tokenization import (
    BYTE_TOKENIZER_HASH,
    BYTE_TOKENIZER_VERSION,
    BYTE_VOCAB_HASH,
)

LEDGER_SCHEMA = "12-6.unique-loss-position-ledger.v1"
RESERVATION_SCHEMA = "12-6.data294-reserved-eval-ranges.v1"
EXPOSURE_STATE_SCHEMA = "12-6.unique-loss-exposure-state.v1"
POSITION_POLICY = "document-isolated-causal-byte-target-v1"


class ExposureAccountingError(ValueError):
    """Raised when source, reservation, position, or exposure accounting drifts."""


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _identity(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def ledger_identity(ledger: Mapping[str, Any]) -> str:
    core = copy.deepcopy(dict(ledger))
    core.pop("ledger_identity_sha256", None)
    return _identity(core)


def exposure_state_identity(state: Mapping[str, Any]) -> str:
    core = copy.deepcopy(dict(state))
    core.pop("state_identity_sha256", None)
    return _identity(core)


def _require_int(value: Any, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ExposureAccountingError(f"{field} must be an integer >= {minimum}")
    return value


def _normalized_ranges(
    raw_ranges: Any, *, size: int, source_id: str
) -> tuple[tuple[int, int], ...]:
    if not isinstance(raw_ranges, list):
        raise ExposureAccountingError(f"{source_id}: reserved_eval_byte_ranges must be a list")
    ranges: list[tuple[int, int]] = []
    for index, value in enumerate(raw_ranges):
        if not isinstance(value, list) or len(value) != 2:
            raise ExposureAccountingError(f"{source_id}: range {index} must be [start, end]")
        start = _require_int(value[0], f"{source_id} range start")
        end = _require_int(value[1], f"{source_id} range end")
        if start >= end or end > size:
            raise ExposureAccountingError(f"{source_id}: invalid reserved range [{start}, {end})")
        ranges.append((start, end))
    ranges.sort()
    previous_end = -1
    for start, end in ranges:
        if start < previous_end:
            raise ExposureAccountingError(f"{source_id}: reserved eval ranges overlap")
        previous_end = end
    return tuple(ranges)


def _eligible_segments(size: int, ranges: Sequence[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    segments: list[tuple[int, int]] = []
    cursor = 0
    for start, end in ranges:
        if cursor < start:
            segments.append((cursor, start))
        cursor = end
    if cursor < size:
        segments.append((cursor, size))
    return tuple(segments)


def _aggregate(rows: Sequence[Mapping[str, Any]], field: str) -> list[dict[str, Any]]:
    totals: dict[str, dict[str, int]] = {}
    for row in rows:
        key = str(row[field])
        bucket = totals.setdefault(
            key,
            {
                "normalized_bytes": 0,
                "reserved_eval_bytes": 0,
                "eligible_bytes": 0,
                "unique_optimized_targets": 0,
                "excluded_causal_targets_due_to_reservation": 0,
                "document_count": 0,
            },
        )
        for metric in (
            "normalized_bytes",
            "reserved_eval_bytes",
            "eligible_bytes",
            "unique_optimized_targets",
            "excluded_causal_targets_due_to_reservation",
        ):
            bucket[metric] += int(row[metric])
        bucket["document_count"] += 1
    return [{"key": key, **totals[key]} for key in sorted(totals)]


def build_unique_loss_ledger(
    registry: Mapping[str, Any],
    reservations: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a one-pass loss-position ledger for the frozen raw-byte tokenizer.

    Reserved evaluation ranges are half-open offsets into the exact normalized UTF-8
    payload. Each eligible contiguous byte segment is tokenized independently. The
    raw-byte tokenizer maps one byte to one token, so a segment of ``n`` bytes owns
    exactly ``max(n - 1, 0)`` causal targets. No transition crosses a document or a
    reserved-evaluation gap.
    """
    validate_real_snapshot_registry(registry)
    if reservations.get("schema_version") != RESERVATION_SCHEMA:
        raise ExposureAccountingError("unsupported reservation manifest schema")
    if reservations.get("local_free_only") is not True:
        raise ExposureAccountingError("DATA-294 reservation manifest must be LOCAL_FREE")
    registry_id = str(registry.get("registry_identity_sha256"))
    if reservations.get("real_snapshot_registry_identity_sha256") != registry_id:
        raise ExposureAccountingError("reservation manifest registry identity drift")

    tokenizer = reservations.get("tokenizer")
    expected_tokenizer = {
        "version": BYTE_TOKENIZER_VERSION,
        "config_sha256": BYTE_TOKENIZER_HASH,
        "vocab_sha256": BYTE_VOCAB_HASH,
        "mapping": "normalized-utf8-byte-identity",
        "add_bos": False,
        "add_eos": False,
        "cross_document": False,
    }
    if tokenizer != expected_tokenizer:
        raise ExposureAccountingError("tokenizer identity/policy drift; rebuild ledger explicitly")

    raw_entries = reservations.get("sources")
    if not isinstance(raw_entries, list):
        raise ExposureAccountingError("reservation sources must be a list")
    reservation_by_source: dict[str, Mapping[str, Any]] = {}
    for entry in raw_entries:
        if not isinstance(entry, Mapping):
            raise ExposureAccountingError("reservation source entry must be an object")
        source_id = str(entry.get("registry_source_id", ""))
        if not source_id or source_id in reservation_by_source:
            raise ExposureAccountingError("reservation source IDs must be unique and non-empty")
        reservation_by_source[source_id] = entry

    eligible_sources = [
        source
        for source in registry["sources"]
        if source["rights"]["model_training"]["status"] == ALLOWED
    ]
    eligible_ids = {str(source["registry_source_id"]) for source in eligible_sources}
    if set(reservation_by_source) != eligible_ids:
        missing = sorted(eligible_ids - set(reservation_by_source))
        extra = sorted(set(reservation_by_source) - eligible_ids)
        raise ExposureAccountingError(
            "reservation inventory must exactly cover training sources; "
            f"missing={missing}, extra={extra}"
        )

    documents: list[dict[str, Any]] = []
    for source in sorted(eligible_sources, key=lambda item: str(item["registry_source_id"])):
        source_id = str(source["registry_source_id"])
        normalization = source.get("normalization")
        family = source.get("source_family")
        if not isinstance(normalization, Mapping) or not isinstance(family, Mapping):
            raise ExposureAccountingError(f"{source_id}: normalization/source family missing")
        size = _require_int(
            normalization.get("extracted_normalized_utf8_bytes"),
            f"{source_id} normalized bytes",
        )
        normalized_sha = str(normalization.get("extracted_normalized_sha256", ""))
        if len(normalized_sha) != 64:
            raise ExposureAccountingError(f"{source_id}: normalized SHA-256 missing")
        entry = reservation_by_source[source_id]
        if entry.get("normalized_payload_sha256") != normalized_sha:
            raise ExposureAccountingError(f"{source_id}: reservation payload identity drift")
        ranges = _normalized_ranges(
            entry.get("reserved_eval_byte_ranges"),
            size=size,
            source_id=source_id,
        )
        segments = _eligible_segments(size, ranges)
        segment_rows: list[dict[str, Any]] = []
        unique_targets = 0
        eligible_bytes = 0
        for byte_start, byte_end in segments:
            token_count = byte_end - byte_start
            target_count = max(token_count - 1, 0)
            segment_core = {
                "registry_source_id": source_id,
                "normalized_payload_sha256": normalized_sha,
                "byte_start": byte_start,
                "byte_end": byte_end,
                "tokenizer_config_sha256": BYTE_TOKENIZER_HASH,
                "position_policy": POSITION_POLICY,
            }
            segment_id = _identity(segment_core)
            segment_rows.append(
                {
                    **segment_core,
                    "segment_identity_sha256": segment_id,
                    "token_count": token_count,
                    "loss_position_start": 1,
                    "loss_position_end_exclusive": token_count,
                    "unique_optimized_targets": target_count,
                }
            )
            eligible_bytes += token_count
            unique_targets += target_count

        reserved_bytes = sum(end - start for start, end in ranges)
        full_document_target_ceiling = max(size - 1, 0)
        documents.append(
            {
                "registry_source_id": source_id,
                "record_id": source["raw_identity"].get("record_id"),
                "language": source.get("language"),
                "modality": source.get("modality"),
                "family_id": family.get("family_id"),
                "family_identity_sha256": family.get("family_identity_sha256"),
                "normalized_payload_sha256": normalized_sha,
                "normalized_bytes": size,
                "reserved_eval_byte_ranges": [list(value) for value in ranges],
                "reserved_eval_bytes": reserved_bytes,
                "eligible_bytes": eligible_bytes,
                "full_document_target_ceiling": full_document_target_ceiling,
                "unique_optimized_targets": unique_targets,
                "excluded_causal_targets_due_to_reservation": (
                    full_document_target_ceiling - unique_targets
                ),
                "segments": segment_rows,
            }
        )

    total_bytes = sum(int(row["normalized_bytes"]) for row in documents)
    reserved_bytes = sum(int(row["reserved_eval_bytes"]) for row in documents)
    eligible_bytes = sum(int(row["eligible_bytes"]) for row in documents)
    one_pass = sum(int(row["unique_optimized_targets"]) for row in documents)
    excluded = sum(
        int(row["excluded_causal_targets_due_to_reservation"]) for row in documents
    )
    ledger: dict[str, Any] = {
        "schema_version": LEDGER_SCHEMA,
        "local_free_only": True,
        "real_snapshot_registry_identity_sha256": registry_id,
        "reservation_manifest_identity_sha256": _identity(reservations),
        "tokenizer": copy.deepcopy(expected_tokenizer),
        "position_policy": POSITION_POLICY,
        "position_key_contract": (
            "(segment_identity_sha256, target_token_index); "
            "target indices are 1 <= index < token_count"
        ),
        "document_boundary_policy": "isolate",
        "reserved_eval_boundary_policy": "split-before-tokenization-no-cross-gap-target",
        "padding_optimized_targets": 0,
        "cross_document_optimized_targets": 0,
        "source_document_count": len(documents),
        "normalized_bytes": total_bytes,
        "reserved_eval_bytes": reserved_bytes,
        "eligible_bytes": eligible_bytes,
        "excluded_causal_targets_due_to_reservation": excluded,
        "one_pass_max_unique_optimized_targets": one_pass,
        "by_language": _aggregate(documents, "language"),
        "by_modality": _aggregate(documents, "modality"),
        "by_family": _aggregate(documents, "family_id"),
        "documents": documents,
        "claim_boundary": {
            "exact_for_tokenizer": BYTE_TOKENIZER_VERSION,
            "non_byte_tokenizer_requires_new_ledger": True,
            "decontamination_clean_claim": False,
            "training_authority_claim": "registry model_training=ALLOWED only",
        },
    }
    ledger["ledger_identity_sha256"] = ledger_identity(ledger)
    validate_unique_loss_ledger(ledger)
    return ledger


def validate_unique_loss_ledger(ledger: Mapping[str, Any]) -> None:
    if ledger.get("schema_version") != LEDGER_SCHEMA:
        raise ExposureAccountingError("unsupported ledger schema")
    if ledger.get("local_free_only") is not True:
        raise ExposureAccountingError("ledger must remain LOCAL_FREE")
    if ledger.get("ledger_identity_sha256") != ledger_identity(ledger):
        raise ExposureAccountingError("ledger self-identity mismatch")
    if ledger.get("padding_optimized_targets") != 0:
        raise ExposureAccountingError("padding can never count as optimized exposure")
    if ledger.get("cross_document_optimized_targets") != 0:
        raise ExposureAccountingError("cross-document targets forbidden by DATA-294")
    documents = ledger.get("documents")
    if not isinstance(documents, list):
        raise ExposureAccountingError("ledger documents malformed")
    seen_segments: set[str] = set()
    total_targets = 0
    for document in documents:
        segments = document.get("segments")
        if not isinstance(segments, list):
            raise ExposureAccountingError("ledger segments malformed")
        document_targets = 0
        for segment in segments:
            token_count = _require_int(segment.get("token_count"), "segment token count")
            start = _require_int(segment.get("loss_position_start"), "loss position start")
            end = _require_int(
                segment.get("loss_position_end_exclusive"),
                "loss position end",
            )
            expected = max(token_count - 1, 0)
            if start != 1 or end != token_count:
                raise ExposureAccountingError("segment loss interval does not cover causal targets")
            if segment.get("unique_optimized_targets") != expected:
                raise ExposureAccountingError("segment target count mismatch")
            segment_id = str(segment.get("segment_identity_sha256", ""))
            core = {
                key: segment[key]
                for key in (
                    "registry_source_id",
                    "normalized_payload_sha256",
                    "byte_start",
                    "byte_end",
                    "tokenizer_config_sha256",
                    "position_policy",
                )
            }
            if segment_id != _identity(core) or segment_id in seen_segments:
                raise ExposureAccountingError("segment identity mismatch/duplicate")
            seen_segments.add(segment_id)
            document_targets += expected
        if document.get("unique_optimized_targets") != document_targets:
            raise ExposureAccountingError("document target total mismatch")
        total_targets += document_targets
    if ledger.get("one_pass_max_unique_optimized_targets") != total_targets:
        raise ExposureAccountingError("ledger one-pass target total mismatch")


class ExposureBudgetGuard:
    """Reject duplicate causal positions and exposure beyond a one-pass budget."""

    def __init__(
        self,
        ledger: Mapping[str, Any],
        authorized_budget: int,
        *,
        state: Mapping[str, Any] | None = None,
    ) -> None:
        validate_unique_loss_ledger(ledger)
        self.ledger_identity_sha256 = str(ledger["ledger_identity_sha256"])
        self.authorized_budget = _require_int(authorized_budget, "authorized budget")
        maximum = int(ledger["one_pass_max_unique_optimized_targets"])
        if self.authorized_budget > maximum:
            raise ExposureAccountingError(
                "authorized exposure budget exceeds ledger one-pass unique maximum"
            )
        self._limits: dict[str, int] = {}
        for document in ledger["documents"]:
            for segment in document["segments"]:
                self._limits[str(segment["segment_identity_sha256"])] = int(
                    segment["token_count"]
                )
        self._claims: dict[str, list[tuple[int, int]]] = {}
        self.consumed_targets = 0
        if state is not None:
            self.load_state_dict(state)

    def _claim_one(self, segment_id: str, start: int, end: int) -> int:
        if segment_id not in self._limits:
            raise ExposureAccountingError("claim references a segment outside the ledger")
        start = _require_int(start, "claim start", minimum=1)
        end = _require_int(end, "claim end", minimum=1)
        token_count = self._limits[segment_id]
        if start >= end or end > token_count:
            raise ExposureAccountingError(
                f"invalid claim interval [{start}, {end}) for token_count={token_count}"
            )
        spans = self._claims.setdefault(segment_id, [])
        for old_start, old_end in spans:
            if start < old_end and old_start < end:
                raise ExposureAccountingError("replayed causal loss position detected")
        count = end - start
        if self.consumed_targets + count > self.authorized_budget:
            raise ExposureAccountingError("authorized unique exposure budget exceeded")
        spans.append((start, end))
        spans.sort()
        compacted: list[tuple[int, int]] = []
        for span_start, span_end in spans:
            if compacted and compacted[-1][1] == span_start:
                compacted[-1] = (compacted[-1][0], span_end)
            else:
                compacted.append((span_start, span_end))
        self._claims[segment_id] = compacted
        self.consumed_targets += count
        return count

    def authorize_batch(
        self,
        claims: Iterable[Mapping[str, Any]],
        *,
        actual_optimized_targets: int,
    ) -> int:
        """Atomically reserve exact positions before a Trainer microbatch executes."""
        expected = _require_int(actual_optimized_targets, "actual optimized targets", minimum=1)
        old_claims = copy.deepcopy(self._claims)
        old_consumed = self.consumed_targets
        claimed = 0
        try:
            for claim in claims:
                if not isinstance(claim, Mapping):
                    raise ExposureAccountingError("exposure claim must be an object")
                claimed += self._claim_one(
                    str(claim.get("segment_identity_sha256", "")),
                    claim.get("loss_position_start"),
                    claim.get("loss_position_end_exclusive"),
                )
            if claimed != expected:
                raise ExposureAccountingError(
                    f"claim count {claimed} != Trainer optimized target count {expected}"
                )
        except Exception:
            self._claims = old_claims
            self.consumed_targets = old_consumed
            raise
        return claimed

    def state_dict(self) -> dict[str, Any]:
        claims = [
            {
                "segment_identity_sha256": segment_id,
                "loss_position_start": start,
                "loss_position_end_exclusive": end,
            }
            for segment_id in sorted(self._claims)
            for start, end in self._claims[segment_id]
        ]
        state: dict[str, Any] = {
            "schema_version": EXPOSURE_STATE_SCHEMA,
            "ledger_identity_sha256": self.ledger_identity_sha256,
            "authorized_budget": self.authorized_budget,
            "consumed_targets": self.consumed_targets,
            "remaining_budget": self.authorized_budget - self.consumed_targets,
            "claims": claims,
        }
        state["state_identity_sha256"] = exposure_state_identity(state)
        return state

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("schema_version") != EXPOSURE_STATE_SCHEMA:
            raise ExposureAccountingError("unsupported exposure-state schema")
        if state.get("state_identity_sha256") != exposure_state_identity(state):
            raise ExposureAccountingError("exposure-state self-identity mismatch")
        if state.get("ledger_identity_sha256") != self.ledger_identity_sha256:
            raise ExposureAccountingError("exposure state belongs to a different ledger")
        if state.get("authorized_budget") != self.authorized_budget:
            raise ExposureAccountingError("exposure budget changed across resume")
        claims = state.get("claims")
        if not isinstance(claims, list):
            raise ExposureAccountingError("exposure claims malformed")
        self._claims = {}
        self.consumed_targets = 0
        for claim in claims:
            self._claim_one(
                str(claim.get("segment_identity_sha256", "")),
                claim.get("loss_position_start"),
                claim.get("loss_position_end_exclusive"),
            )
        if state.get("consumed_targets") != self.consumed_targets:
            raise ExposureAccountingError("exposure-state consumed target count mismatch")
        if state.get("remaining_budget") != self.authorized_budget - self.consumed_targets:
            raise ExposureAccountingError("exposure-state remaining budget mismatch")
