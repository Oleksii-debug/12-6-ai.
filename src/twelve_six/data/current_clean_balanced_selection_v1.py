"""Materialize the exact current-clean record selection required by canonical split.

This module is a record-granularity realization seam only. It does not change the
canonical NEXT100-106 balance policy or the canonical split algorithm. A terminal
family-byte allocation is accepted only when whole authenticated survivor records can
realize every allocated family byte exactly. Otherwise the seam fails closed.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from twelve_six.data.balanced_split_application_v1 import (
    BalancedSplitApplicationError,
    verify_balanced_selection,
)
from twelve_six.data.postdecontam_balance_projection_v1 import ProjectionError
from twelve_six.data.postmaterialization_balance_projection_v1 import (
    CURRENT_CLEAN_FAMILY_VECTOR_SCHEMA,
    _canonical_bytes,
    _current_clean_receipt_self_hash,
    _family_capacity_projection,
    _rebuild_current_clean_survivor_inventory,
    _require_nonnegative_int,
    _require_sha256,
    _self_hash,
    _sha256_bytes,
    _verify_current_clean_receipt,
    build_balance_result_binding,
    load_strict_json_object,
    require_balanced_selection_ready,
    verify_postmaterialization_family_vector,
)
from twelve_six.data.trusted_family_authority_v1 import TRUSTED_FAMILY_SEMANTICS

SELECTION_SCHEMA = "12-6.d03-balanced-selection-authority.v1"
LEGACY_SELECTION_REALIZATION_POLICY = "record-id-ascending-exact-family-byte-subset-v1"
SELECTION_REALIZATION_POLICY = "record-id-ascending-policy-cap-aware-whole-record-subset-v2"
DEFAULT_MAX_EXACT_SUBSET_STATES = 250_000
DEFAULT_MAX_EXACT_SUBSET_EXPANSIONS = 5_000_000
DEFAULT_EXACT_SUBSET_MAX_TARGET_BYTES = 5_000_000
DEFAULT_EXACT_SUBSET_BLOCK_RECORDS = 2_048
DEFAULT_EXACT_SUBSET_MAX_CHECKPOINT_BYTES = 128 * 1024 * 1024
_BIT_REVERSE_TABLE = bytes(int(f"{value:08b}"[::-1], 2) for value in range(256))
_ALLOWED_ALLOCATION_FIELDS = {
    "family_id",
    "stratum",
    "allocated_bytes",
    "available_unique_bytes",
    "effective_family_cap_bytes",
}
_STRATUM_TO_BALANCE = {"uk": "ua", "en": "en", "code": "code"}
_SELECTION_FIELDS = {
    "schema",
    "terminal",
    "status",
    "balanced_selection_identity_sha256",
    "retained_inventory_identity_sha256",
    "decontamination_authority_sha256",
    "dedup_authority_sha256",
    "balance_policy_identity_sha256",
    "balance_result_identity_sha256",
    "records",
    "totals",
    "claim_boundary",
}
_SELECTION_ROW_FIELDS = {
    "record_id",
    "source_id",
    "family",
    "stratum",
    "modality",
    "payload_sha256",
    "payload_bytes",
    "near_duplicate_cluster_id",
    "purpose",
    "training_eligible",
    "evaluation_eligible",
    "evaluation_reserved",
}
_CLAIM_BOUNDARY = {
    "training_eligible": False,
    "evaluation_eligible": False,
    "tokenizer_fit_authorized": False,
    "model_training_authorized": False,
    "paid_compute_authorized": False,
    "final_test_outcomes_read": False,
    "authorized_optimized_target_exposure": 0,
}


def _require_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProjectionError(f"{field} must be non-empty text")
    return value


def _parse_survivor_records(raw: bytes) -> list[dict[str, Any]]:
    if not isinstance(raw, bytes) or not raw or not raw.endswith(b"\n"):
        raise ProjectionError("current-clean survivor JSONL must be non-empty and LF-terminated")

    expected = {
        "record_id",
        "source_id",
        "family",
        "modality",
        "normalized_payload",
    }
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, physical in enumerate(raw.splitlines(keepends=True)):
        if not physical.endswith(b"\n"):
            raise ProjectionError(f"current-clean survivor record[{index}] lacks terminal LF")
        line = physical[:-1]
        row = load_strict_json_object(
            line,
            label=f"current-clean survivor record[{index}]",
        )
        if set(row) != expected:
            raise ProjectionError(f"current-clean survivor record[{index}] schema drift")
        if _canonical_bytes(row) != line:
            raise ProjectionError(f"current-clean survivor record[{index}] is not canonical JSON")
        for field in expected:
            _require_text(row.get(field), f"current-clean survivor record[{index}].{field}")
        record_id = row["record_id"]
        if record_id in seen:
            raise ProjectionError(f"duplicate current-clean survivor record_id: {record_id}")
        seen.add(record_id)

        family = row["family"]
        authority = TRUSTED_FAMILY_SEMANTICS.get(family)
        if authority is None:
            raise ProjectionError(f"survivor family absent from trusted authority: {family}")
        stratum = authority.get("stratum")
        if stratum not in _STRATUM_TO_BALANCE:
            raise ProjectionError(f"unsupported survivor family stratum: {family}")
        modality = row["modality"]
        if stratum == "code" and modality != "code":
            raise ProjectionError(f"code family has non-code survivor modality: {family}")
        if stratum == "en" and modality not in {"en", "text"}:
            raise ProjectionError(f"English family survivor modality drift: {family}")
        if stratum == "uk" and modality not in {"uk", "ua", "text"}:
            raise ProjectionError(f"Ukrainian family survivor modality drift: {family}")

        payload = row["normalized_payload"].encode("utf-8")
        rows.append(
            {
                **row,
                "stratum": stratum,
                "payload_sha256": hashlib.sha256(payload).hexdigest(),
                "payload_bytes": len(payload),
            }
        )
    rows.sort(key=lambda item: item["record_id"])
    return rows


def _exact_record_subset(
    rows: Sequence[Mapping[str, Any]],
    *,
    target_bytes: int,
    family: str,
    max_states: int = DEFAULT_MAX_EXACT_SUBSET_STATES,
    max_expansions: int = DEFAULT_MAX_EXACT_SUBSET_EXPANSIONS,
) -> list[dict[str, Any]]:
    """Return one deterministic exact whole-record realization of a byte allocation."""

    target = _require_nonnegative_int(target_bytes, f"allocation[{family}].allocated_bytes")
    state_budget = _require_nonnegative_int(max_states, "max_states")
    expansion_budget = _require_nonnegative_int(max_expansions, "max_expansions")
    if state_budget <= 0 or expansion_budget <= 0:
        raise ProjectionError("exact-subset budgets must be positive")
    if target <= 0:
        raise ProjectionError(f"allocation[{family}] must be positive")

    ordered = [dict(row) for row in sorted(rows, key=lambda item: str(item["record_id"]))]
    total = sum(
        _require_nonnegative_int(row.get("payload_bytes"), "payload_bytes")
        for row in ordered
    )
    if target > total:
        raise ProjectionError(f"allocation[{family}] exceeds authenticated family capacity")
    if target == total:
        return ordered

    # Sparse exact subset DP. Insertion order is deterministic because record order is
    # deterministic. We stop as soon as the exact allocation first becomes reachable.
    # No partial record, padding, replay or byte truncation is permitted.
    parent: dict[int, tuple[int, int] | None] = {0: None}
    reached_at = -1
    expansions = 0
    for index, row in enumerate(ordered):
        weight = _require_nonnegative_int(row.get("payload_bytes"), "payload_bytes")
        if weight <= 0:
            raise ProjectionError("selected survivor payload_bytes must be positive")
        existing = tuple(parent)
        for current in existing:
            expansions += 1
            if expansions > expansion_budget:
                raise ProjectionError(
                    f"allocation[{family}] exact-subset work budget exceeded"
                )
            candidate = current + weight
            if candidate > target or candidate in parent:
                continue
            if len(parent) >= state_budget:
                raise ProjectionError(
                    f"allocation[{family}] exact-subset state budget exceeded"
                )
            parent[candidate] = (current, index)
            if candidate == target:
                reached_at = index
                break
        if reached_at >= 0:
            break

    if target not in parent:
        raise ProjectionError(
            f"allocation[{family}] has no exact whole-record realization under "
            f"{LEGACY_SELECTION_REALIZATION_POLICY}"
        )

    selected_indexes: list[int] = []
    cursor = target
    while cursor:
        link = parent.get(cursor)
        if link is None:
            raise ProjectionError(f"allocation[{family}] exact-subset reconstruction failed")
        previous, index = link
        selected_indexes.append(index)
        cursor = previous
    selected_indexes.reverse()
    return [ordered[index] for index in selected_indexes]


class _ExactSubsetSearchError(ProjectionError):
    """Raised when a bounded exact-subset search has no safe result."""


def _reverse_low_bits(bits: int, width: int) -> int:
    """Reverse exactly width low-order bits using byte-table operations."""

    if width <= 0:
        return 0
    byte_count = (width + 7) // 8
    raw = bits.to_bytes(byte_count, "little")
    reversed_bytes = raw.translate(_BIT_REVERSE_TABLE)[::-1]
    reversed_value = int.from_bytes(reversed_bytes, "little")
    return reversed_value >> (byte_count * 8 - width)


def _reachable_subset_bits(weights: Sequence[int], target: int) -> int:
    """Return the exact subset-sum reachability bitset through target."""

    mask = (1 << (target + 1)) - 1
    reachable = 1
    for weight in weights:
        if weight <= target:
            reachable |= (reachable << weight) & mask
    return reachable


def _reconstruct_block_subset(weights: Sequence[int], target: int) -> list[int]:
    """Reconstruct one deterministic exact subset inside one bounded block."""

    if target == 0:
        return []
    total = sum(weights)
    if target > total:
        raise _ExactSubsetSearchError("block target exceeds block capacity")
    if target == total:
        return list(range(len(weights)))
    if len(weights) == 1:
        if weights[0] == target:
            return [0]
        raise _ExactSubsetSearchError("block target has no exact subset")

    midpoint = len(weights) // 2
    left = weights[:midpoint]
    right = weights[midpoint:]
    left_bits = _reachable_subset_bits(left, target)
    right_bits = _reachable_subset_bits(right, target)
    candidates = left_bits & _reverse_low_bits(right_bits, target + 1)
    if not candidates:
        raise _ExactSubsetSearchError("block reconstruction found no exact split")

    # Prefer the largest contribution from the earlier half. Combined with the
    # outer earliest-prefix rule, this makes reconstruction deterministic.
    left_target = candidates.bit_length() - 1
    left_indexes = _reconstruct_block_subset(left, left_target)
    right_indexes = _reconstruct_block_subset(right, target - left_target)
    return left_indexes + [midpoint + index for index in right_indexes]


def _bounded_exact_record_subset(
    rows: Sequence[Mapping[str, Any]],
    *,
    target_bytes: int,
    label: str,
    max_target_bytes: int = DEFAULT_EXACT_SUBSET_MAX_TARGET_BYTES,
    block_records: int = DEFAULT_EXACT_SUBSET_BLOCK_RECORDS,
    max_checkpoint_bytes: int = DEFAULT_EXACT_SUBSET_MAX_CHECKPOINT_BYTES,
) -> list[dict[str, Any]]:
    """Return a deterministic exact whole-record subset under explicit bounds."""

    target = _require_nonnegative_int(target_bytes, f"{label}.target_bytes")
    target_limit = _require_nonnegative_int(max_target_bytes, "max_target_bytes")
    block_size = _require_nonnegative_int(block_records, "block_records")
    checkpoint_limit = _require_nonnegative_int(
        max_checkpoint_bytes,
        "max_checkpoint_bytes",
    )
    if target <= 0:
        raise ProjectionError(f"{label} target must be positive")
    if target_limit <= 0 or block_size <= 0 or checkpoint_limit <= 0:
        raise ProjectionError("bounded exact-subset limits must be positive")

    ordered = [dict(row) for row in sorted(rows, key=lambda item: str(item["record_id"]))]
    weights: list[int] = []
    for row in ordered:
        weight = _require_nonnegative_int(row.get("payload_bytes"), "payload_bytes")
        if weight <= 0:
            raise ProjectionError("selected survivor payload_bytes must be positive")
        weights.append(weight)

    total = sum(weights)
    if target > total:
        raise ProjectionError(f"{label} exceeds authenticated record capacity")
    if target == total:
        return ordered

    complement = total - target < target
    solve_target = total - target if complement else target
    if solve_target > target_limit:
        raise _ExactSubsetSearchError(
            f"{label} bounded exact-subset target exceeds {target_limit} bytes"
        )

    checkpoint_count = (len(ordered) + block_size - 1) // block_size + 1
    checkpoint_bytes = (solve_target + 8) // 8
    estimated_checkpoint_bytes = checkpoint_count * checkpoint_bytes
    if estimated_checkpoint_bytes > checkpoint_limit:
        raise _ExactSubsetSearchError(
            f"{label} bounded exact-subset checkpoint budget exceeded"
        )

    mask = (1 << (solve_target + 1)) - 1
    reachable = 1
    checkpoints = [reachable]
    block_ranges: list[tuple[int, int]] = []
    for start in range(0, len(ordered), block_size):
        stop = min(start + block_size, len(ordered))
        processed_stop = start
        for index in range(start, stop):
            weight = weights[index]
            if weight <= solve_target:
                reachable |= (reachable << weight) & mask
            processed_stop = index + 1
            if (reachable >> solve_target) & 1:
                break
        block_ranges.append((start, processed_stop))
        checkpoints.append(reachable)
        if (reachable >> solve_target) & 1:
            break

    if not ((reachable >> solve_target) & 1):
        raise _ExactSubsetSearchError(f"{label} has no exact whole-record realization")

    cursor = solve_target
    chosen_indexes: list[int] = []
    for block_index in range(len(block_ranges) - 1, -1, -1):
        if cursor == 0:
            break
        prefix = checkpoints[block_index]
        if (prefix >> cursor) & 1:
            continue

        start, stop = block_ranges[block_index]
        chunk_weights = weights[start:stop]
        chunk_bits = _reachable_subset_bits(chunk_weights, cursor)
        prefix_mask = (1 << (cursor + 1)) - 1
        aligned_prefix = _reverse_low_bits(prefix & prefix_mask, cursor + 1)
        candidates = chunk_bits & aligned_prefix
        if not candidates:
            raise ProjectionError(f"{label} exact-subset checkpoint reconstruction failed")
        contribution_bit = candidates & -candidates
        contribution = contribution_bit.bit_length() - 1
        local_indexes = _reconstruct_block_subset(chunk_weights, contribution)
        chosen_indexes.extend(start + index for index in local_indexes)
        cursor -= contribution

    if cursor != 0:
        raise ProjectionError(f"{label} exact-subset reconstruction did not reach zero")

    chosen = set(chosen_indexes)
    if complement:
        result = [row for index, row in enumerate(ordered) if index not in chosen]
    else:
        result = [ordered[index] for index in sorted(chosen)]
    if sum(int(row["payload_bytes"]) for row in result) != target:
        raise ProjectionError(f"{label} exact-subset reconstruction byte mismatch")
    return result


def _allocation_caps_by_stratum(
    allocations: Mapping[str, Mapping[str, Any]],
) -> dict[str, int]:
    caps: defaultdict[str, set[int]] = defaultdict(set)
    for family, allocation in allocations.items():
        stratum = _require_text(allocation.get("stratum"), f"allocation[{family}].stratum")
        cap = _require_nonnegative_int(
            allocation.get("effective_family_cap_bytes"),
            f"allocation[{family}].effective_family_cap_bytes",
        )
        caps[stratum].add(cap)

    expected = {"ua", "en", "code"}
    if set(caps) != expected:
        raise ProjectionError("balance allocation does not cover every required stratum")
    result: dict[str, int] = {}
    for stratum in sorted(expected):
        if len(caps[stratum]) != 1:
            raise ProjectionError(f"balance allocation family cap drift: {stratum}")
        result[stratum] = next(iter(caps[stratum]))
    return result


def _select_stratum_records(
    rows: Sequence[Mapping[str, Any]],
    *,
    stratum: str,
    allocations: Mapping[str, Mapping[str, Any]],
    target_bytes: int,
    family_cap_bytes: int,
) -> list[dict[str, Any]]:
    """Realize one exact stratum without weakening the authenticated family cap."""

    target = _require_nonnegative_int(target_bytes, f"stratum[{stratum}].target_bytes")
    cap = _require_nonnegative_int(
        family_cap_bytes,
        f"stratum[{stratum}].family_cap_bytes",
    )
    if target <= 0 or cap <= 0:
        raise ProjectionError(f"stratum[{stratum}] target/cap must be positive")

    stratum_allocations = {
        family: allocation
        for family, allocation in allocations.items()
        if allocation["stratum"] == stratum
    }
    if not stratum_allocations:
        raise ProjectionError(f"stratum[{stratum}] has no authenticated allocation")

    by_family: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in rows:
        row = dict(raw)
        family = _require_text(row.get("family"), "family")
        row_stratum = _STRATUM_TO_BALANCE.get(str(row.get("stratum")))
        if row_stratum != stratum:
            raise ProjectionError(f"stratum[{stratum}] received foreign survivor row")
        by_family[family].append(row)

    try:
        canonical: list[dict[str, Any]] = []
        for family in sorted(stratum_allocations):
            candidates = by_family.get(family, [])
            if not candidates:
                raise ProjectionError(
                    f"allocated family has no authenticated survivor rows: {family}"
                )
            canonical.extend(
                _bounded_exact_record_subset(
                    candidates,
                    target_bytes=int(stratum_allocations[family]["allocated_bytes"]),
                    label=f"allocation[{family}]",
                )
            )
        if sum(int(row["payload_bytes"]) for row in canonical) != target:
            raise ProjectionError(f"stratum[{stratum}] canonical allocation total drift")
        return canonical
    except _ExactSubsetSearchError as canonical_error:
        # The continuous family-byte witness can be impossible at whole-record
        # granularity. Decouple from that witness only when every physical family
        # in this stratum is already wholly below the authenticated family cap;
        # then any record subset is cap-safe by construction.
        full_family_bytes = {
            family: sum(int(row["payload_bytes"]) for row in family_rows)
            for family, family_rows in by_family.items()
        }
        over_cap = sorted(
            family for family, total in full_family_bytes.items() if total > cap
        )
        if over_cap:
            raise ProjectionError(
                f"stratum[{stratum}] cannot safely decouple record realization "
                f"from family allocation; full family exceeds cap: {over_cap[0]}"
            ) from canonical_error

        selected = _bounded_exact_record_subset(
            rows,
            target_bytes=target,
            label=f"stratum[{stratum}]",
        )
        selected_family_bytes: defaultdict[str, int] = defaultdict(int)
        for row in selected:
            selected_family_bytes[str(row["family"])] += int(row["payload_bytes"])
        if any(total > cap for total in selected_family_bytes.values()):
            raise ProjectionError(f"stratum[{stratum}] record selection exceeds family cap")
        return selected


def _validated_allocations(
    balance_result: Mapping[str, Any],
    family_vector: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    raw = balance_result.get("deterministic_maximum_allocation")
    if not isinstance(raw, list) or not raw:
        raise ProjectionError("terminal balance result has no deterministic allocation")

    capacities = {
        row["family"]: {
            "stratum": _STRATUM_TO_BALANCE[str(row["stratum"])],
            "capacity": _require_nonnegative_int(row["capacity_bytes"], "capacity_bytes"),
        }
        for row in family_vector["families"]
    }
    allocations: dict[str, dict[str, Any]] = {}
    by_stratum: defaultdict[str, int] = defaultdict(int)
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping) or set(item) != _ALLOWED_ALLOCATION_FIELDS:
            raise ProjectionError(f"balance allocation[{index}] fields are not closed-world")
        family = _require_text(item.get("family_id"), f"allocation[{index}].family_id")
        if family in allocations:
            raise ProjectionError(f"duplicate balance allocation family: {family}")
        if family not in capacities:
            raise ProjectionError(f"balance allocation references unknown family: {family}")
        stratum = _require_text(item.get("stratum"), f"allocation[{index}].stratum")
        if stratum != capacities[family]["stratum"]:
            raise ProjectionError(f"balance allocation family/stratum drift: {family}")
        allocated = _require_nonnegative_int(
            item.get("allocated_bytes"),
            f"allocation[{index}].allocated_bytes",
        )
        available = _require_nonnegative_int(
            item.get("available_unique_bytes"),
            f"allocation[{index}].available_unique_bytes",
        )
        cap = _require_nonnegative_int(
            item.get("effective_family_cap_bytes"),
            f"allocation[{index}].effective_family_cap_bytes",
        )
        if allocated <= 0 or available != capacities[family]["capacity"]:
            raise ProjectionError(f"balance allocation family capacity drift: {family}")
        if allocated > available or allocated > cap:
            raise ProjectionError(f"balance allocation exceeds family authority: {family}")
        allocations[family] = dict(item)
        by_stratum[stratum] += allocated

    maximum = _require_nonnegative_int(
        balance_result.get("maximum_feasible_total_source_bytes"),
        "maximum_feasible_total_source_bytes",
    )
    if sum(int(item["allocated_bytes"]) for item in allocations.values()) != maximum:
        raise ProjectionError("balance allocation total does not match maximum feasible total")
    expected_strata = balance_result.get("maximum_feasible_stratum_bytes")
    if not isinstance(expected_strata, Mapping):
        raise ProjectionError("balance result maximum stratum bytes are missing")
    for stratum in ("ua", "en", "code"):
        expected = _require_nonnegative_int(
            expected_strata.get(stratum),
            f"maximum_feasible_stratum_bytes.{stratum}",
        )
        if by_stratum[stratum] != expected:
            raise ProjectionError(f"balance allocation stratum total drift: {stratum}")
    return allocations


def build_current_clean_balanced_selection(
    *,
    family_vector: Mapping[str, Any],
    next100_input: Mapping[str, Any],
    balance_result: Mapping[str, Any],
    balance_binding: Mapping[str, Any],
    composition_receipt_raw: bytes,
    survivor_records_raw: bytes,
    expected_family_vector_identity_sha256: str,
    expected_balance_binding_identity_sha256: str,
    expected_policy_identity_sha256: str,
    expected_result_identity_sha256: str,
) -> dict[str, Any]:
    """Materialize the canonical split selection from exact current-clean records."""

    if family_vector.get("schema") != CURRENT_CLEAN_FAMILY_VECTOR_SCHEMA:
        raise ProjectionError("balanced selection requires current-clean family vector")
    family_identity = verify_postmaterialization_family_vector(
        family_vector,
        expected_identity_sha256=expected_family_vector_identity_sha256,
    )
    rebuilt_binding = build_balance_result_binding(
        family_vector=family_vector,
        expected_family_vector_identity_sha256=family_identity,
        next100_input=next100_input,
        balance_result=balance_result,
        expected_policy_identity_sha256=expected_policy_identity_sha256,
        expected_result_identity_sha256=expected_result_identity_sha256,
    )
    # Python mapping equality treats integer/float (and boolean/integer) aliases
    # as equal. Compare canonical JSON bytes so a caller-resealed binding cannot
    # substitute differently typed evidence for the deterministic rebuild.
    try:
        matches_rebuild = (
            _canonical_bytes(dict(balance_binding))
            == _canonical_bytes(rebuilt_binding)
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProjectionError("balance binding is not canonical JSON") from exc
    if not matches_rebuild:
        raise ProjectionError("balance binding differs from deterministic authenticated rebuild")
    require_balanced_selection_ready(
        balance_binding,
        expected_binding_identity_sha256=expected_balance_binding_identity_sha256,
    )

    if _sha256_bytes(composition_receipt_raw) != family_vector.get(
        "composition_receipt_json_sha256"
    ):
        raise ProjectionError("composition receipt raw bytes differ from family-vector authority")
    receipt = load_strict_json_object(
        composition_receipt_raw,
        label="current-clean composition receipt",
    )
    if receipt.get("receipt_identity_sha256") != family_vector.get(
        "current_clean_receipt_identity_sha256"
    ):
        raise ProjectionError("composition receipt identity differs from family-vector authority")
    if _current_clean_receipt_self_hash(receipt) != receipt.get(
        "receipt_identity_sha256"
    ):
        raise ProjectionError("composition receipt self-hash mismatch")

    if _sha256_bytes(survivor_records_raw) != family_vector.get(
        "survivor_records_jsonl_sha256"
    ):
        raise ProjectionError("survivor JSONL bytes differ from family-vector authority")
    survivor_rows = _parse_survivor_records(survivor_records_raw)
    rebuilt_inventory = _rebuild_current_clean_survivor_inventory(survivor_records_raw)
    inventory_checks = {
        "record_count": family_vector.get("record_count"),
        "total_payload_bytes": family_vector.get("total_payload_bytes"),
        "record_inventory_digest_sha256": family_vector.get(
            "record_inventory_digest_sha256"
        ),
        "payload_inventory_digest_sha256": family_vector.get(
            "payload_inventory_digest_sha256"
        ),
    }
    for field, expected in inventory_checks.items():
        if type(rebuilt_inventory.get(field)) is not type(expected) or rebuilt_inventory.get(
            field
        ) != expected:
            raise ProjectionError(f"survivor inventory/family-vector mismatch: {field}")

    # Bind producer-declared source cardinality to the same authenticated
    # JSONL whose inventory digests and payload bytes were just reconstructed.
    physical_sources = len(
        {row["source_id"] for row in rebuilt_inventory["records"]}
    )
    if family_vector.get("source_object_count") != physical_sources:
        raise ProjectionError("current-clean source-object count differs from survivor JSONL")

    # The vector can be self-resealed and its digest supplied by a caller. Its
    # membership and per-family counts must still come from the same raw rows
    # that authenticated the inventory and producer receipt above.
    physical_projection = _family_capacity_projection(rebuilt_inventory["records"])
    for field in (
        "record_membership_sha256",
        "families",
        "stratum_capacity_bytes",
        "stratum_family_counts",
        "trusted_family_authority_root_sha256",
    ):
        observed = family_vector.get(field)
        actual = physical_projection[field]
        if type(observed) is not type(actual) or _canonical_bytes(observed) != _canonical_bytes(actual):
            raise ProjectionError(f"survivor inventory/family-vector physical projection mismatch: {field}")
    _verify_current_clean_receipt(
        receipt,
        expected_receipt_identity_sha256=family_vector[
            "current_clean_receipt_identity_sha256"
        ],
        expected_survivor_jsonl_sha256=family_vector[
            "survivor_records_jsonl_sha256"
        ],
        expected_record_inventory_digest_sha256=family_vector[
            "record_inventory_digest_sha256"
        ],
        expected_payload_inventory_digest_sha256=family_vector[
            "payload_inventory_digest_sha256"
        ],
        expected_record_count=rebuilt_inventory["record_count"],
        expected_total_payload_bytes=rebuilt_inventory["total_payload_bytes"],
        expected_source_object_count=physical_sources,
    )

    allocations = _validated_allocations(balance_result, family_vector)
    allocation_caps = _allocation_caps_by_stratum(allocations)
    rows_by_stratum: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in survivor_rows:
        balance_stratum = _STRATUM_TO_BALANCE[str(row["stratum"])]
        rows_by_stratum[balance_stratum].append(row)

    expected_strata = balance_result.get("maximum_feasible_stratum_bytes")
    if not isinstance(expected_strata, Mapping):
        raise ProjectionError("balance result maximum stratum bytes are missing")

    selected: list[dict[str, Any]] = []
    for stratum in ("ua", "en", "code"):
        target = _require_nonnegative_int(
            expected_strata.get(stratum),
            f"maximum_feasible_stratum_bytes.{stratum}",
        )
        selected.extend(
            _select_stratum_records(
                rows_by_stratum[stratum],
                stratum=stratum,
                allocations=allocations,
                target_bytes=target,
                family_cap_bytes=allocation_caps[stratum],
            )
        )

    selected.sort(key=lambda row: row["record_id"])
    if len({row["record_id"] for row in selected}) != len(selected):
        raise ProjectionError("balanced selection reuses a survivor record")

    family_bytes: defaultdict[str, int] = defaultdict(int)
    stratum_bytes: defaultdict[str, int] = defaultdict(int)
    balance_stratum_bytes: defaultdict[str, int] = defaultdict(int)
    output_rows: list[dict[str, Any]] = []
    for row in selected:
        output = {
            "record_id": row["record_id"],
            "source_id": row["source_id"],
            "family": row["family"],
            "stratum": row["stratum"],
            "modality": row["modality"],
            "payload_sha256": row["payload_sha256"],
            "payload_bytes": row["payload_bytes"],
            # Terminal global-dedup already selected one survivor source per duplicate
            # component. Grouping every record from the same survivor source together
            # is conservative and cannot split one source across validation/train.
            "near_duplicate_cluster_id": row["source_id"],
            "purpose": "pretraining_eligible",
            "training_eligible": False,
            "evaluation_eligible": False,
            "evaluation_reserved": False,
        }
        output_rows.append(output)
        family_bytes[row["family"]] += int(row["payload_bytes"])
        stratum_bytes[row["stratum"]] += int(row["payload_bytes"])
        balance_stratum_bytes[_STRATUM_TO_BALANCE[str(row["stratum"])]] += int(
            row["payload_bytes"]
        )

    selected_families_by_stratum: defaultdict[str, set[str]] = defaultdict(set)
    for family, selected_bytes in family_bytes.items():
        authority = TRUSTED_FAMILY_SEMANTICS.get(family)
        if authority is None:
            raise ProjectionError(f"selected family absent from trusted authority: {family}")
        balance_stratum = _STRATUM_TO_BALANCE[str(authority["stratum"])]
        if selected_bytes > allocation_caps[balance_stratum]:
            raise ProjectionError(f"balanced selection family cap exceeded: {family}")
        selected_families_by_stratum[balance_stratum].add(family)

    family_minimum = balance_result.get("family_minimum")
    if not isinstance(family_minimum, Mapping):
        raise ProjectionError("balance result family minimum is missing")
    minimum_families = _require_nonnegative_int(
        family_minimum.get("required_per_stratum"),
        "family_minimum.required_per_stratum",
    )
    for stratum in ("ua", "en", "code"):
        expected_bytes = _require_nonnegative_int(
            expected_strata.get(stratum),
            f"maximum_feasible_stratum_bytes.{stratum}",
        )
        if balance_stratum_bytes[stratum] != expected_bytes:
            raise ProjectionError(f"balanced selection stratum allocation mismatch: {stratum}")
        if len(selected_families_by_stratum[stratum]) < minimum_families:
            raise ProjectionError(f"balanced selection family minimum mismatch: {stratum}")

    source_bytes = sum(row["payload_bytes"] for row in output_rows)
    if source_bytes != balance_result.get("maximum_feasible_total_source_bytes"):
        raise ProjectionError("balanced selection bytes do not equal terminal balance total")

    dedup = next100_input.get("dedup_authority")
    if not isinstance(dedup, Mapping):
        raise ProjectionError("NEXT100 input lacks dedup authority")
    dedup_identity = _require_sha256(
        dedup.get("evidence_identity_sha256"),
        "dedup evidence_identity_sha256",
    )
    decontamination_identity = _require_sha256(
        receipt.get("decontamination_execution_identity_sha256"),
        "decontamination_execution_identity_sha256",
    )
    policy_identity = _require_sha256(
        expected_policy_identity_sha256,
        "expected_policy_identity_sha256",
    )
    result_identity = _require_sha256(
        expected_result_identity_sha256,
        "expected_result_identity_sha256",
    )

    document: dict[str, Any] = {
        "schema": SELECTION_SCHEMA,
        "terminal": True,
        "status": "PASS",
        "balanced_selection_identity_sha256": "0" * 64,
        "retained_inventory_identity_sha256": family_vector[
            "record_inventory_digest_sha256"
        ],
        "decontamination_authority_sha256": decontamination_identity,
        "dedup_authority_sha256": dedup_identity,
        "balance_policy_identity_sha256": policy_identity,
        "balance_result_identity_sha256": result_identity,
        "records": output_rows,
        "totals": {
            "record_count": len(output_rows),
            "source_bytes": source_bytes,
            "family_source_bytes": dict(sorted(family_bytes.items())),
            "stratum_source_bytes": dict(sorted(stratum_bytes.items())),
        },
        "claim_boundary": dict(_CLAIM_BOUNDARY),
    }
    document["balanced_selection_identity_sha256"] = _self_hash(
        document,
        "balanced_selection_identity_sha256",
    )
    try:
        verify_balanced_selection(
            document,
            expected_selection_identity_sha256=document[
                "balanced_selection_identity_sha256"
            ],
            expected_retained_inventory_identity_sha256=family_vector[
                "record_inventory_digest_sha256"
            ],
            expected_decontamination_authority_sha256=decontamination_identity,
            expected_dedup_authority_sha256=dedup_identity,
            expected_balance_policy_identity_sha256=policy_identity,
            expected_balance_result_identity_sha256=result_identity,
        )
    except BalancedSplitApplicationError as exc:
        raise ProjectionError(
            "built balanced selection violates canonical PR1091 contract"
        ) from exc
    return document


def project_selected_current_clean_raw_records(
    selection: Mapping[str, Any],
    survivor_records_raw: bytes,
    *,
    expected_selection_identity_sha256: str,
    expected_retained_inventory_identity_sha256: str,
    expected_decontamination_authority_sha256: str,
    expected_dedup_authority_sha256: str,
    expected_balance_policy_identity_sha256: str,
    expected_balance_result_identity_sha256: str,
) -> list[dict[str, Any]]:
    """Project selected current-clean payloads after canonical authority verification."""

    try:
        selected, _totals = verify_balanced_selection(
            selection,
            expected_selection_identity_sha256=expected_selection_identity_sha256,
            expected_retained_inventory_identity_sha256=(
                expected_retained_inventory_identity_sha256
            ),
            expected_decontamination_authority_sha256=(
                expected_decontamination_authority_sha256
            ),
            expected_dedup_authority_sha256=expected_dedup_authority_sha256,
            expected_balance_policy_identity_sha256=(
                expected_balance_policy_identity_sha256
            ),
            expected_balance_result_identity_sha256=(
                expected_balance_result_identity_sha256
            ),
        )
    except BalancedSplitApplicationError as exc:
        raise ProjectionError(
            "canonical balanced selection verification failed"
        ) from exc

    # A selection authenticates the entire retained inventory, not only the
    # subset emitted below. Reject added/changed unselected survivor records
    # instead of silently accepting a different raw JSONL under its identity.
    physical_inventory = _rebuild_current_clean_survivor_inventory(survivor_records_raw)
    if (
        physical_inventory["record_inventory_digest_sha256"]
        != expected_retained_inventory_identity_sha256
    ):
        raise ProjectionError("balanced selection raw survivor inventory differs from retained authority")
    physical = {row["record_id"]: row for row in _parse_survivor_records(survivor_records_raw)}
    if not set(selected) <= set(physical):
        raise ProjectionError("balanced selection references record absent from survivor JSONL")

    projected: list[dict[str, Any]] = []
    for record_id in sorted(selected):
        authority = selected[record_id]
        raw = physical[record_id]
        exact = {
            "source_id": raw["source_id"],
            "family": raw["family"],
            "stratum": raw["stratum"],
            "modality": raw["modality"],
            "payload_sha256": raw["payload_sha256"],
            "payload_bytes": raw["payload_bytes"],
            "near_duplicate_cluster_id": raw["source_id"],
            "purpose": "pretraining_eligible",
            "training_eligible": False,
            "evaluation_eligible": False,
            "evaluation_reserved": False,
        }
        for field, expected in exact.items():
            if type(authority.get(field)) is not type(expected) or authority.get(field) != expected:
                raise ProjectionError(f"balanced selection/raw survivor drift: {record_id}:{field}")
        projected.append(
            {
                "record_id": record_id,
                "source_id": raw["source_id"],
                "family": raw["family"],
                "stratum": raw["stratum"],
                "modality": raw["modality"],
                "near_duplicate_cluster_id": raw["source_id"],
                "normalized_payload": raw["normalized_payload"],
                "purpose": "pretraining_eligible",
                "training_eligible": False,
                "evaluation_eligible": False,
                "evaluation_reserved": False,
            }
        )
    return projected
