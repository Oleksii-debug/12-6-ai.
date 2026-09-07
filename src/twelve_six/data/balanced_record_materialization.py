"""Deterministically materialize whole records under the frozen balance allocation.

NEXT100-106 computes family-level byte allocations but intentionally does not create
an exact corpus identity.  This module closes only that handoff: it binds the exact
balance result to an exact post-policy record graph and selects whole records under
each family allocation.  Records are never truncated, replayed, or duplicated.

If whole-record granularity cannot fill an allocation exactly, the result is a
nonterminal candidate with an explicit byte gap.  This is preferable to silently
turning byte-level planning capacity into a fabricated corpus identity.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

BALANCE_RESULT_SCHEMA = "12-6.next100-106-balance-gate-result.v1"
MATERIALIZATION_SCHEMA = "12-6.balanced-record-materialization.v1"
SELECTION_POLICY = "balance-result-seeded-whole-record-greedy-v1"
STRATA = ("ua", "en", "code")
_REQUIRED_STAGE_BINDINGS = ("dedup", "decontamination", "quality", "privacy", "balance")
_HEX = frozenset("0123456789abcdef")


class BalancedMaterializationError(ValueError):
    """Raised when balance-to-record materialization would be unsafe."""


def _canonical_json_bytes(value: Any, *, trailing_lf: bool) -> bytes:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if trailing_lf:
        text += "\n"
    return text.encode("utf-8")


def _sha256_obj(value: Any, *, trailing_lf: bool = True) -> str:
    return hashlib.sha256(_canonical_json_bytes(value, trailing_lf=trailing_lf)).hexdigest()


def _require_sha256(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in _HEX for character in value)
    ):
        raise BalancedMaterializationError(f"{field} must be exact lowercase SHA-256")
    return value


def _require_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BalancedMaterializationError(f"{field} must be non-empty text")
    return value


def _require_positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise BalancedMaterializationError(f"{field} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class BalancedRecord:
    """One exact post-decontamination, post-quality/privacy record."""

    record_id: str
    text: str
    source_id: str
    family_id: str
    stratum: str
    language: str
    modality: str
    normalized_payload_sha256: str
    source_bytes: int
    dedup_cluster_id: str
    training_eligible: bool = True
    evaluation_reserved: bool = False
    quality_pass: bool = True
    privacy_pass: bool = True

    def __post_init__(self) -> None:
        for field in (
            "record_id",
            "source_id",
            "family_id",
            "language",
            "dedup_cluster_id",
        ):
            _require_text(getattr(self, field), field)
        if not isinstance(self.text, str) or not self.text:
            raise BalancedMaterializationError("text must be non-empty")
        if self.stratum not in STRATA:
            raise BalancedMaterializationError(f"unsupported stratum: {self.stratum!r}")
        if self.modality not in {"text", "code"}:
            raise BalancedMaterializationError(f"unsupported modality: {self.modality!r}")
        if self.stratum == "code" and self.modality != "code":
            raise BalancedMaterializationError("code stratum requires code modality")
        if self.stratum != "code" and self.modality != "text":
            raise BalancedMaterializationError("ua/en strata require text modality")
        _require_sha256(self.normalized_payload_sha256, "normalized_payload_sha256")
        payload = self.text.encode("utf-8")
        if hashlib.sha256(payload).hexdigest() != self.normalized_payload_sha256:
            raise BalancedMaterializationError(
                f"{self.record_id}: normalized payload hash does not match text"
            )
        if (
            isinstance(self.source_bytes, bool)
            or not isinstance(self.source_bytes, int)
            or self.source_bytes <= 0
            or self.source_bytes != len(payload)
        ):
            raise BalancedMaterializationError(
                f"{self.record_id}: source_bytes must equal positive normalized UTF-8 bytes"
            )
        if self.training_eligible is not True:
            raise BalancedMaterializationError(f"{self.record_id}: record is not training eligible")
        if self.evaluation_reserved is not False:
            raise BalancedMaterializationError(f"{self.record_id}: evaluation-reserved record")
        if self.quality_pass is not True:
            raise BalancedMaterializationError(f"{self.record_id}: quality gate is not PASS")
        if self.privacy_pass is not True:
            raise BalancedMaterializationError(f"{self.record_id}: privacy gate is not PASS")

    def identity_mapping(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("text")
        return value


def record_graph_identity(records: Sequence[BalancedRecord]) -> str:
    """Bind the complete post-policy record graph without serializing raw text."""
    rows = sorted((record.identity_mapping() for record in records), key=lambda row: row["record_id"])
    return _sha256_obj({"records": rows})


def _normalize_stage_bindings(value: Mapping[str, str]) -> dict[str, str]:
    if set(value) != set(_REQUIRED_STAGE_BINDINGS):
        raise BalancedMaterializationError(
            "stage_bindings must contain exactly dedup, decontamination, quality, privacy, balance"
        )
    return {
        name: _require_sha256(value[name], f"stage_bindings.{name}")
        for name in _REQUIRED_STAGE_BINDINGS
    }


def _validate_balance_result(
    result: Mapping[str, Any], expected_identity_sha256: str
) -> tuple[str, list[dict[str, Any]]]:
    if result.get("schema_version") != BALANCE_RESULT_SCHEMA:
        raise BalancedMaterializationError("unsupported balance result schema")
    expected = _require_sha256(expected_identity_sha256, "expected_balance_result_identity_sha256")
    body = dict(result)
    observed = _require_sha256(body.pop("result_identity_sha256", None), "result_identity_sha256")
    recomputed = _sha256_obj(body, trailing_lf=False)
    if observed != recomputed:
        raise BalancedMaterializationError("balance result self-identity mismatch")
    if observed != expected:
        raise BalancedMaterializationError("balance result does not match expected authority")
    if result.get("target_total_source_bytes") != 20_000_000:
        raise BalancedMaterializationError("20M source-byte target drift")

    claim = result.get("claim_boundary")
    if not isinstance(claim, Mapping):
        raise BalancedMaterializationError("balance claim_boundary must be an object")
    if claim.get("authorized_training_exposure_loss_positions") != 0:
        raise BalancedMaterializationError("balance result already claims training exposure")
    for field in ("tokenizer_fit_authorized", "model_training_authorized", "paid_compute_authorized"):
        if claim.get(field) is not False:
            raise BalancedMaterializationError(f"balance claim boundary drift: {field}")
    if claim.get("source_bytes_are_loss_positions") is not False:
        raise BalancedMaterializationError("source bytes must not be relabelled as loss positions")

    allocations = result.get("deterministic_maximum_allocation")
    if not isinstance(allocations, list):
        raise BalancedMaterializationError("deterministic_maximum_allocation must be a list")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(allocations):
        if not isinstance(raw, Mapping):
            raise BalancedMaterializationError(f"allocation[{index}] must be an object")
        family_id = _require_text(raw.get("family_id"), f"allocation[{index}].family_id")
        if family_id in seen:
            raise BalancedMaterializationError(f"duplicate family allocation: {family_id}")
        seen.add(family_id)
        stratum = raw.get("stratum")
        if stratum not in STRATA:
            raise BalancedMaterializationError(f"invalid allocation stratum: {stratum!r}")
        allocated = _require_positive_int(raw.get("allocated_bytes"), "allocated_bytes")
        available = _require_positive_int(
            raw.get("available_unique_bytes"), "available_unique_bytes"
        )
        cap = _require_positive_int(
            raw.get("effective_family_cap_bytes"), "effective_family_cap_bytes"
        )
        if allocated > available or allocated > cap:
            raise BalancedMaterializationError("family allocation exceeds capacity/cap")
        normalized.append(
            {
                "family_id": family_id,
                "stratum": stratum,
                "allocated_bytes": allocated,
                "available_unique_bytes": available,
                "effective_family_cap_bytes": cap,
            }
        )

    maximum = result.get("maximum_feasible_total_source_bytes")
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 0:
        raise BalancedMaterializationError("invalid maximum_feasible_total_source_bytes")
    if sum(row["allocated_bytes"] for row in normalized) != maximum:
        raise BalancedMaterializationError("allocation total does not match maximum feasible total")
    return observed, sorted(normalized, key=lambda row: row["family_id"])


def _selection_rank(balance_identity: str, record: BalancedRecord) -> str:
    payload = f"{balance_identity}\0{record.family_id}\0{record.record_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def materialize_balanced_record_set(
    records: Sequence[BalancedRecord],
    balance_result: Mapping[str, Any],
    *,
    expected_balance_result_identity_sha256: str,
    stage_bindings: Mapping[str, str],
) -> tuple[dict[str, Any], tuple[BalancedRecord, ...]]:
    """Select whole records under the exact frozen family allocation.

    Selection is deterministic and model-result independent.  For each allocated
    family, records are ordered by a hash seeded by the exact balance-result identity;
    a record is accepted only when its complete normalized bytes fit the remaining
    family allocation.  No truncation or replay is permitted.
    """
    if not records:
        raise BalancedMaterializationError("records must not be empty")
    balance_identity, allocations = _validate_balance_result(
        balance_result, expected_balance_result_identity_sha256
    )
    bindings = _normalize_stage_bindings(stage_bindings)
    if bindings["balance"] != balance_identity:
        raise BalancedMaterializationError("stage_bindings.balance does not match result identity")

    ordered = sorted(records, key=lambda record: record.record_id)
    ids = [record.record_id for record in ordered]
    if len(set(ids)) != len(ids):
        raise BalancedMaterializationError("record_id values must be unique")
    graph_identity = record_graph_identity(ordered)

    by_family: dict[str, list[BalancedRecord]] = defaultdict(list)
    for record in ordered:
        by_family[record.family_id].append(record)

    selected: list[BalancedRecord] = []
    family_rows: list[dict[str, Any]] = []
    selected_clusters: set[str] = set()
    allocation_families = {row["family_id"] for row in allocations}

    for allocation in allocations:
        family_id = allocation["family_id"]
        family_records = by_family.get(family_id, [])
        if not family_records:
            raise BalancedMaterializationError(f"allocated family has no records: {family_id}")
        if any(record.stratum != allocation["stratum"] for record in family_records):
            raise BalancedMaterializationError(f"family stratum drift: {family_id}")
        available = sum(record.source_bytes for record in family_records)
        if available != allocation["available_unique_bytes"]:
            raise BalancedMaterializationError(
                f"{family_id}: exact record bytes do not match balance available_unique_bytes"
            )

        remaining = allocation["allocated_bytes"]
        chosen: list[BalancedRecord] = []
        ranked = sorted(
            family_records,
            key=lambda record: (_selection_rank(balance_identity, record), record.record_id),
        )
        for record in ranked:
            if record.source_bytes <= remaining:
                if record.dedup_cluster_id in selected_clusters:
                    raise BalancedMaterializationError(
                        f"dedup cluster replay across selected records: {record.dedup_cluster_id}"
                    )
                chosen.append(record)
                selected_clusters.add(record.dedup_cluster_id)
                remaining -= record.source_bytes
            if remaining == 0:
                break

        selected.extend(chosen)
        selected_bytes = allocation["allocated_bytes"] - remaining
        family_rows.append(
            {
                **allocation,
                "selected_record_ids": sorted(record.record_id for record in chosen),
                "selected_bytes": selected_bytes,
                "whole_record_granularity_gap_bytes": remaining,
            }
        )

    ignored_families = sorted(set(by_family) - allocation_families)
    selected = sorted(selected, key=lambda record: record.record_id)
    selected_ids = {record.record_id for record in selected}
    selected_rows = [record.identity_mapping() for record in selected]
    selected_total = sum(record.source_bytes for record in selected)
    by_stratum = {
        stratum: sum(record.source_bytes for record in selected if record.stratum == stratum)
        for stratum in STRATA
    }
    total_gap = sum(row["whole_record_granularity_gap_bytes"] for row in family_rows)
    allocation_exact = total_gap == 0
    balance_status = balance_result.get("status")
    target_feasible = balance_status == "TARGET_20M_SOURCE_MIX_FEASIBLE"
    terminal_for_learned20 = allocation_exact and target_feasible and selected_total == 20_000_000

    core: dict[str, Any] = {
        "schema_version": MATERIALIZATION_SCHEMA,
        "selection_policy": SELECTION_POLICY,
        "balance_result_identity_sha256": balance_identity,
        "post_policy_record_graph_identity_sha256": graph_identity,
        "stage_bindings": bindings,
        "input_record_count": len(ordered),
        "selected_record_count": len(selected),
        "selected_source_bytes": selected_total,
        "selected_source_bytes_by_stratum": by_stratum,
        "selected_records": selected_rows,
        "family_materialization": family_rows,
        "unallocated_input_family_ids": ignored_families,
        "whole_record_granularity_gap_bytes": total_gap,
        "allocation_exact_at_record_boundaries": allocation_exact,
        "balance_status": balance_status,
        "terminal_for_learned20": terminal_for_learned20,
        "claim_boundary": {
            "training_authorized_loss_positions": 0,
            "source_bytes_are_loss_positions": False,
            "tokenizer_fit_authorized": False,
            "model_training_authorized": False,
            "paid_compute_authorized": False,
        },
    }
    core["materialization_identity_sha256"] = _sha256_obj(core)
    core["terminal_corpus_identity_sha256"] = (
        _sha256_obj(
            {
                "materialization_identity_sha256": core["materialization_identity_sha256"],
                "selected_records": selected_rows,
            }
        )
        if terminal_for_learned20
        else None
    )

    if selected_ids != {row["record_id"] for row in selected_rows}:
        raise AssertionError("selected record projection mismatch")
    return core, tuple(selected)
