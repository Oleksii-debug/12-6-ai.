"""Materialize the exact current-clean record selection required by canonical split.

This module is a record-granularity realization seam only. It does not change the
canonical NEXT100-106 balance policy or the canonical split algorithm. It first realizes
the deterministic family-byte witness exactly. If that source-byte witness is not
whole-record representable, it may re-realize the same exact stratum target only when
every authenticated family in that stratum is already wholly below the canonical
per-family cap. Otherwise the seam fails closed.
"""

from __future__ import annotations

import hashlib
import math
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
SELECTION_REALIZATION_POLICY = (
    "record-id-ascending-bitset-exact-with-safe-stratum-repair-v2"
)
DEFAULT_MAX_EXACT_SUBSET_RECORDS = 250_000
DEFAULT_MAX_EXACT_SUBSET_TARGET_BYTES = 20_000_000
DEFAULT_MAX_EXACT_SUBSET_RECONSTRUCTION_BYTES = 512 * 1024 * 1024
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


class _NoExactRecordSubset(ProjectionError):
    """Raised only when an authenticated whole-record target is unreachable."""


def _exact_record_subset(
    rows: Sequence[Mapping[str, Any]],
    *,
    target_bytes: int,
    family: str,
    max_records: int = DEFAULT_MAX_EXACT_SUBSET_RECORDS,
    max_target_bytes: int = DEFAULT_MAX_EXACT_SUBSET_TARGET_BYTES,
    max_reconstruction_bytes: int = DEFAULT_MAX_EXACT_SUBSET_RECONSTRUCTION_BYTES,
) -> list[dict[str, Any]]:
    """Return one deterministic exact whole-record subset with bounded memory.

    Reachability uses one integer bitset. Reconstruction replays bounded square-root
    blocks, retaining only sparse prefix checkpoints plus one local block at a time.
    Later records are skipped whenever an earlier-record witness already exists, so
    the selected witness is deterministic under record-id ordering.
    """

    target = _require_nonnegative_int(
        target_bytes,
        f"allocation[{family}].allocated_bytes",
    )
    record_bound = _require_nonnegative_int(max_records, "max_records")
    target_bound = _require_nonnegative_int(max_target_bytes, "max_target_bytes")
    reconstruction_bound = _require_nonnegative_int(
        max_reconstruction_bytes,
        "max_reconstruction_bytes",
    )
    if record_bound <= 0 or target_bound <= 0 or reconstruction_bound <= 0:
        raise ProjectionError("exact-subset bounds must be positive")
    if target <= 0:
        raise ProjectionError(f"allocation[{family}] must be positive")
    if target > target_bound:
        raise ProjectionError(f"allocation[{family}] exceeds exact-subset target bound")

    ordered = [
        dict(row)
        for row in sorted(rows, key=lambda item: str(item["record_id"]))
    ]
    if len(ordered) > record_bound:
        raise ProjectionError(f"allocation[{family}] exceeds exact-subset record bound")

    weights: list[int] = []
    total = 0
    for row in ordered:
        weight = _require_nonnegative_int(row.get("payload_bytes"), "payload_bytes")
        if weight <= 0:
            raise ProjectionError("selected survivor payload_bytes must be positive")
        weights.append(weight)
        total += weight
    if target > total:
        raise _NoExactRecordSubset(
            f"allocation[{family}] exceeds authenticated family capacity"
        )
    if target == total:
        return ordered

    mask = (1 << (target + 1)) - 1
    reachable = 1
    reached_count = 0
    for index, weight in enumerate(weights):
        if weight <= target:
            reachable |= (reachable << weight) & mask
        if (reachable >> target) & 1:
            reached_count = index + 1
            break
    if reached_count == 0:
        raise _NoExactRecordSubset(
            f"allocation[{family}] has no exact whole-record realization under "
            f"{SELECTION_REALIZATION_POLICY}"
        )

    block_size = math.isqrt(reached_count) + 1
    bitset_bytes = (target + 8) // 8
    checkpoint_count = ((reached_count - 1) // block_size) + 1
    estimated_bytes = (
        2 * (checkpoint_count + block_size + 8) * bitset_bytes
    )
    if estimated_bytes > reconstruction_bound:
        raise ProjectionError(
            f"allocation[{family}] exact-subset reconstruction memory bound exceeded"
        )

    checkpoints: dict[int, int] = {0: 1}
    reachable = 1
    for index in range(reached_count):
        weight = weights[index]
        if weight <= target:
            reachable |= (reachable << weight) & mask
        boundary = index + 1
        if boundary % block_size == 0:
            checkpoints[boundary] = reachable

    selected_indexes: list[int] = []
    cursor = target
    end_index = reached_count
    while end_index:
        start_index = ((end_index - 1) // block_size) * block_size
        prefix = checkpoints.get(start_index)
        if prefix is None:
            raise ProjectionError(
                f"allocation[{family}] exact-subset checkpoint reconstruction failed"
            )
        local: list[int] = [prefix]
        current = prefix
        for index in range(start_index, end_index):
            weight = weights[index]
            if weight <= target:
                current |= (current << weight) & mask
            local.append(current)
        if not ((local[-1] >> cursor) & 1):
            raise ProjectionError(
                f"allocation[{family}] exact-subset reconstruction lost reachability"
            )

        for index in range(end_index - 1, start_index - 1, -1):
            previous = local[index - start_index]
            if (previous >> cursor) & 1:
                continue
            weight = weights[index]
            if cursor < weight or not ((previous >> (cursor - weight)) & 1):
                raise ProjectionError(
                    f"allocation[{family}] exact-subset reconstruction failed"
                )
            selected_indexes.append(index)
            cursor -= weight
        end_index = start_index

    if cursor != 0:
        raise ProjectionError(
            f"allocation[{family}] exact-subset reconstruction did not reach zero"
        )
    selected_indexes.reverse()
    return [ordered[index] for index in selected_indexes]

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
    by_family: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    by_balance_stratum: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    family_balance_stratum: dict[str, str] = {}
    for row in survivor_rows:
        family = row["family"]
        balance_stratum = _STRATUM_TO_BALANCE[row["stratum"]]
        by_family[family].append(row)
        by_balance_stratum[balance_stratum].append(row)
        previous = family_balance_stratum.setdefault(family, balance_stratum)
        if previous != balance_stratum:
            raise ProjectionError(f"survivor family spans balance strata: {family}")

    stratum_caps: dict[str, int] = {}
    for allocation in allocations.values():
        stratum = str(allocation["stratum"])
        cap = _require_nonnegative_int(
            allocation["effective_family_cap_bytes"],
            f"allocation[{allocation['family_id']}].effective_family_cap_bytes",
        )
        previous = stratum_caps.setdefault(stratum, cap)
        if previous != cap:
            raise ProjectionError(f"balance allocation cap drift within stratum: {stratum}")

    minimum = balance_result.get("family_minimum")
    if not isinstance(minimum, Mapping):
        raise ProjectionError("balance result family minimum authority missing")
    minimum_selected_families = _require_nonnegative_int(
        minimum.get("required_per_stratum"),
        "family_minimum.required_per_stratum",
    )
    if minimum_selected_families <= 0:
        raise ProjectionError("family minimum must be positive")

    selected: list[dict[str, Any]] = []
    repaired_strata: set[str] = set()
    for stratum in ("ua", "en", "code"):
        stratum_allocations = {
            family: allocation
            for family, allocation in allocations.items()
            if allocation["stratum"] == stratum
        }
        if not stratum_allocations:
            raise ProjectionError(f"terminal balance has no allocation for stratum: {stratum}")

        canonical_selected: list[dict[str, Any]] = []
        impossible_family: str | None = None
        for family in sorted(stratum_allocations):
            candidates = by_family.get(family, [])
            if not candidates:
                raise ProjectionError(
                    f"allocated family has no authenticated survivor rows: {family}"
                )
            try:
                chosen = _exact_record_subset(
                    candidates,
                    target_bytes=int(stratum_allocations[family]["allocated_bytes"]),
                    family=family,
                )
            except _NoExactRecordSubset:
                impossible_family = family
                break
            canonical_selected.extend(chosen)

        if impossible_family is None:
            selected.extend(canonical_selected)
            continue

        # The canonical byte allocator is source-byte exact, not record-granular.
        # A stratum-level repair is allowed only when every full authenticated family
        # is already no larger than the independently fixed per-family cap. Under
        # that condition any whole-record subset is cap-safe by construction; there
        # is no hidden within-family truncation problem to solve.
        cap = stratum_caps.get(stratum)
        if cap is None:
            raise ProjectionError(f"missing canonical family cap for stratum: {stratum}")
        stratum_families = sorted(
            {row["family"] for row in by_balance_stratum[stratum]}
        )
        unsafe = [
            family
            for family in stratum_families
            if sum(int(row["payload_bytes"]) for row in by_family[family]) > cap
        ]
        if unsafe:
            raise ProjectionError(
                f"unsafe whole-record stratum repair for {stratum}: "
                f"authenticated family exceeds canonical cap: {unsafe[0]}"
            )

        expected_strata = balance_result.get("maximum_feasible_stratum_bytes")
        if not isinstance(expected_strata, Mapping):
            raise ProjectionError("balance result maximum stratum bytes are missing")
        stratum_target = _require_nonnegative_int(
            expected_strata.get(stratum),
            f"maximum_feasible_stratum_bytes.{stratum}",
        )
        try:
            repaired = _exact_record_subset(
                by_balance_stratum[stratum],
                target_bytes=stratum_target,
                family=f"stratum:{stratum}",
            )
        except _NoExactRecordSubset as exc:
            raise ProjectionError(
                f"stratum[{stratum}] has no policy-safe whole-record realization"
            ) from exc

        selected_families = {row["family"] for row in repaired}
        if len(selected_families) < minimum_selected_families:
            raise ProjectionError(
                f"stratum[{stratum}] whole-record repair violates family minimum"
            )
        selected.extend(repaired)
        repaired_strata.add(stratum)

    selected.sort(key=lambda row: row["record_id"])
    if len({row["record_id"] for row in selected}) != len(selected):
        raise ProjectionError("balanced selection reuses a survivor record")

    family_bytes: defaultdict[str, int] = defaultdict(int)
    stratum_bytes: defaultdict[str, int] = defaultdict(int)
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

    for family, allocation in allocations.items():
        if allocation["stratum"] in repaired_strata:
            continue
        if family_bytes[family] != allocation["allocated_bytes"]:
            raise ProjectionError(f"balanced selection family allocation mismatch: {family}")

    for stratum in repaired_strata:
        cap = stratum_caps[stratum]
        for family, selected_bytes in family_bytes.items():
            if family_balance_stratum.get(family) == stratum and selected_bytes > cap:
                raise ProjectionError(
                    f"balanced selection repaired family exceeds cap: {family}"
                )

    balance_stratum_bytes: defaultdict[str, int] = defaultdict(int)
    for row in output_rows:
        balance_stratum_bytes[_STRATUM_TO_BALANCE[row["stratum"]]] += int(
            row["payload_bytes"]
        )
    expected_strata = balance_result.get("maximum_feasible_stratum_bytes")
    if not isinstance(expected_strata, Mapping):
        raise ProjectionError("balance result maximum stratum bytes are missing")
    for stratum in ("ua", "en", "code"):
        expected = _require_nonnegative_int(
            expected_strata.get(stratum),
            f"maximum_feasible_stratum_bytes.{stratum}",
        )
        if balance_stratum_bytes[stratum] != expected:
            raise ProjectionError(
                f"balanced selection stratum total mismatch: {stratum}"
            )

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
