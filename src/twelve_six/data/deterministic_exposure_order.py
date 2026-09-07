from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from twelve_six.data.identity_safe_exposure_guard import (
    IdentitySafeExposureReplayGuard,
    require_expected_ledger_identity,
)
from twelve_six.data.unique_loss_ledger_v2 import LedgerError

_PLAN_SCHEMA = "12-6.deterministic-exposure-order.v1"
_PLAN_PREFLIGHT_SCHEMA = "12-6.exposure-plan-preflight.v1"
_BATCH_KEYS = frozenset(
    {
        "global_batch_index",
        "shard_index",
        "worker_id",
        "claims",
        "actual_nonignored_targets",
    }
)
_CLAIM_KEYS = frozenset(
    {"segment_identity_sha256", "offset_start", "offset_end"}
)
_PLAN_KEYS = frozenset(
    {
        "schema_version",
        "num_workers",
        "batches_per_shard",
        "shard_count",
        "batches",
        "plan_identity_sha256",
    }
)


def _canonical_sha256(value: Any) -> str:
    payload = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise LedgerError(f"{label} must be a non-negative integer")
    return value


def _positive_int(value: Any, label: str) -> int:
    result = _nonnegative_int(value, label)
    if result == 0:
        raise LedgerError(f"{label} must be positive")
    return result


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise LedgerError(f"{label} must be a 64-hex SHA-256 string")
    try:
        int(value, 16)
    except ValueError as exc:
        raise LedgerError(f"{label} must be a 64-hex SHA-256 string") from exc
    return value.lower()


def _validated_plan_batches(
    plan: Mapping[str, Any],
) -> tuple[Sequence[Mapping[str, Any]], str]:
    if not isinstance(plan, Mapping) or set(plan) != _PLAN_KEYS:
        raise LedgerError("exposure plan fields do not match schema")
    if plan["schema_version"] != _PLAN_SCHEMA:
        raise LedgerError("unsupported exposure plan schema")
    expected_plan_identity = _require_sha256(
        plan["plan_identity_sha256"], "plan_identity_sha256"
    )
    unhashed = dict(plan)
    unhashed.pop("plan_identity_sha256")
    if _canonical_sha256(unhashed) != expected_plan_identity:
        raise LedgerError("exposure plan self-hash mismatch")
    batches = plan["batches"]
    if not isinstance(batches, Sequence) or isinstance(batches, (str, bytes)):
        raise LedgerError("exposure plan batches must be a sequence")
    return batches, expected_plan_identity


def build_deterministic_exposure_plan(
    batches: Sequence[Mapping[str, Any]],
    *,
    num_workers: int,
    batches_per_shard: int,
    shard_count: int,
) -> dict[str, Any]:
    """Build a fail-closed deterministic worker/shard schedule.

    Worker assignment is round-robin by global batch index. Shard assignment is
    contiguous by ``batches_per_shard``. This makes the intended order explicit
    and hashable instead of inheriting process scheduling or iterator timing.
    """
    workers = _positive_int(num_workers, "num_workers")
    per_shard = _positive_int(batches_per_shard, "batches_per_shard")
    shards = _positive_int(shard_count, "shard_count")
    normalized: list[dict[str, Any]] = []
    for expected_index, batch in enumerate(batches):
        if not isinstance(batch, Mapping) or set(batch) != _BATCH_KEYS:
            raise LedgerError(f"batches[{expected_index}] fields do not match schema")
        index = _nonnegative_int(batch["global_batch_index"], "global_batch_index")
        shard = _nonnegative_int(batch["shard_index"], "shard_index")
        worker = _nonnegative_int(batch["worker_id"], "worker_id")
        targets = _nonnegative_int(
            batch["actual_nonignored_targets"], "actual_nonignored_targets"
        )
        claims = batch["claims"]
        if not isinstance(claims, Sequence) or isinstance(claims, (str, bytes)):
            raise LedgerError("claims must be a sequence")
        if index != expected_index:
            raise LedgerError("global batch indexes must be contiguous from zero")
        if worker != index % workers:
            raise LedgerError("worker assignment is not deterministic round-robin")
        expected_shard = index // per_shard
        if shard != expected_shard or shard >= shards:
            raise LedgerError("shard assignment does not match deterministic plan")
        normalized.append(
            {
                "global_batch_index": index,
                "shard_index": shard,
                "worker_id": worker,
                "claims": [dict(claim) for claim in claims],
                "actual_nonignored_targets": targets,
            }
        )

    plan: dict[str, Any] = {
        "schema_version": _PLAN_SCHEMA,
        "num_workers": workers,
        "batches_per_shard": per_shard,
        "shard_count": shards,
        "batches": normalized,
    }
    plan["plan_identity_sha256"] = _canonical_sha256(plan)
    return plan


def validate_exposure_plan_preflight(
    plan: Mapping[str, Any],
    ledger: Mapping[str, Any],
    *,
    expected_ledger_identity_sha256: str,
    expected_unique_budget: int,
) -> dict[str, Any]:
    """Prove a deterministic plan cannot manufacture unique training capacity.

    The preflight validates the exact externally bound ledger, rechecks deterministic
    worker/shard placement, rejects unknown/out-of-range/replayed claims before the
    first optimizer update, requires every batch's observed target cardinality to
    equal its claim cardinality, and requires the whole plan to sum to the exact
    unique exposure budget. Padding-only zero-target batches are rejected.
    """
    require_expected_ledger_identity(
        ledger,
        expected_ledger_identity_sha256=expected_ledger_identity_sha256,
    )
    budget = _nonnegative_int(expected_unique_budget, "expected_unique_budget")
    one_pass_maximum = _nonnegative_int(
        ledger.get("one_pass_unique_nonignored_causal_loss_positions"),
        "one_pass_unique_nonignored_causal_loss_positions",
    )
    if budget > one_pass_maximum:
        raise LedgerError("expected unique exposure budget exceeds ledger maximum")

    batches, plan_identity = _validated_plan_batches(plan)
    workers = _positive_int(plan["num_workers"], "num_workers")
    per_shard = _positive_int(plan["batches_per_shard"], "batches_per_shard")
    shards = _positive_int(plan["shard_count"], "shard_count")

    raw_segments = ledger.get("segments")
    if not isinstance(raw_segments, Sequence) or isinstance(raw_segments, (str, bytes)):
        raise LedgerError("ledger segments must be a sequence")
    segment_lengths: dict[str, int] = {}
    claimed_intervals: dict[str, list[tuple[int, int]]] = {}
    for segment_index, segment in enumerate(raw_segments):
        if not isinstance(segment, Mapping):
            raise LedgerError(f"ledger segments[{segment_index}] must be an object")
        segment_id = _require_sha256(
            segment.get("segment_identity_sha256"),
            f"ledger segments[{segment_index}].segment_identity_sha256",
        )
        if segment_id in segment_lengths:
            raise LedgerError("duplicate ledger segment identity")
        segment_lengths[segment_id] = _nonnegative_int(
            segment.get("loss_position_count"),
            f"ledger segments[{segment_index}].loss_position_count",
        )
        claimed_intervals[segment_id] = []

    total_claimed = 0
    for expected_index, batch in enumerate(batches):
        if not isinstance(batch, Mapping) or set(batch) != _BATCH_KEYS:
            raise LedgerError(f"batches[{expected_index}] fields do not match schema")
        index = _nonnegative_int(batch["global_batch_index"], "global_batch_index")
        shard = _nonnegative_int(batch["shard_index"], "shard_index")
        worker = _nonnegative_int(batch["worker_id"], "worker_id")
        if index != expected_index:
            raise LedgerError("global batch indexes must be contiguous from zero")
        if worker != index % workers:
            raise LedgerError("worker assignment is not deterministic round-robin")
        expected_shard = index // per_shard
        if shard != expected_shard or shard >= shards:
            raise LedgerError("shard assignment does not match deterministic plan")

        actual_targets = _nonnegative_int(
            batch["actual_nonignored_targets"], "actual_nonignored_targets"
        )
        if actual_targets == 0:
            raise LedgerError(
                "exposure plan batches must contain positive nonignored targets"
            )
        claims = batch["claims"]
        if not isinstance(claims, Sequence) or isinstance(claims, (str, bytes)):
            raise LedgerError("claims must be a sequence")
        batch_claimed = 0
        for claim_index, claim in enumerate(claims):
            if not isinstance(claim, Mapping) or set(claim) != _CLAIM_KEYS:
                raise LedgerError(
                    f"batches[{expected_index}].claims[{claim_index}] fields do not "
                    "match schema"
                )
            segment_id = _require_sha256(
                claim["segment_identity_sha256"],
                f"batches[{expected_index}].claims[{claim_index}]"
                ".segment_identity_sha256",
            )
            if segment_id not in segment_lengths:
                raise LedgerError("claim references unknown ledger segment")
            start = _nonnegative_int(claim["offset_start"], "offset_start")
            end = _nonnegative_int(claim["offset_end"], "offset_end")
            if start >= end or end > segment_lengths[segment_id]:
                raise LedgerError("claim interval is outside ledger segment")
            for old_start, old_end in claimed_intervals[segment_id]:
                if start < old_end and old_start < end:
                    raise LedgerError("exposure plan replays/overlaps unique loss positions")
            claimed_intervals[segment_id].append((start, end))
            batch_claimed += end - start
        if batch_claimed != actual_targets:
            raise LedgerError(
                "batch claim cardinality does not match actual nonignored targets"
            )
        total_claimed += batch_claimed

    if total_claimed != budget:
        raise LedgerError("exposure plan unique target count does not match expected budget")

    complete_one_pass = budget == one_pass_maximum
    if complete_one_pass:
        for segment_id, segment_length in segment_lengths.items():
            intervals = sorted(claimed_intervals[segment_id])
            cursor = 0
            for start, end in intervals:
                if start != cursor:
                    raise LedgerError("full one-pass exposure plan has a segment gap")
                cursor = end
            if cursor != segment_length:
                raise LedgerError("full one-pass exposure plan has a segment gap")

    proof: dict[str, Any] = {
        "schema_version": _PLAN_PREFLIGHT_SCHEMA,
        "ledger_identity_sha256": _require_sha256(
            ledger.get("ledger_identity_sha256"), "ledger_identity_sha256"
        ),
        "plan_identity_sha256": plan_identity,
        "expected_unique_budget": budget,
        "plan_unique_nonignored_targets": total_claimed,
        "ledger_one_pass_maximum": one_pass_maximum,
        "complete_one_pass": complete_one_pass,
        "replay_or_overlap_detected": False,
        "padding_capacity_credit": 0,
        "training_authorized_by_this_preflight": False,
    }
    proof["preflight_identity_sha256"] = _canonical_sha256(proof)
    return proof


def ordered_next_exposure_identity(
    guard: IdentitySafeExposureReplayGuard,
    plan: Mapping[str, Any],
    *,
    batch_index: int,
) -> str:
    """Bind the guard's next exposure to exact deterministic scheduling context."""
    index = _nonnegative_int(batch_index, "batch_index")
    batches, expected_plan_identity = _validated_plan_batches(plan)
    if index >= len(batches):
        raise LedgerError("batch_index is outside exposure plan")
    if index != guard.claim_sequence:
        raise LedgerError("batch_index does not match next exposure sequence")
    batch = batches[index]
    base_identity = guard.next_exposure_identity(
        batch["claims"], actual_nonignored_targets=batch["actual_nonignored_targets"]
    )
    return _canonical_sha256(
        {
            "schema_version": "12-6.ordered-next-exposure.v1",
            "plan_identity_sha256": expected_plan_identity,
            "base_next_exposure_identity_sha256": base_identity,
            "global_batch_index": batch["global_batch_index"],
            "shard_index": batch["shard_index"],
            "worker_id": batch["worker_id"],
        }
    )


def authorize_ordered_batch(
    guard: IdentitySafeExposureReplayGuard,
    plan: Mapping[str, Any],
    *,
    batch_index: int,
    expected_ordered_next_exposure_identity_sha256: str,
) -> str:
    """Authorize one deterministic scheduled batch without mutation on mismatch."""
    observed = ordered_next_exposure_identity(guard, plan, batch_index=batch_index)
    if observed != expected_ordered_next_exposure_identity_sha256:
        raise LedgerError("ordered next exposure identity does not match expected handoff")
    batches, _ = _validated_plan_batches(plan)
    batch = batches[batch_index]
    base_identity = guard.next_exposure_identity(
        batch["claims"], actual_nonignored_targets=batch["actual_nonignored_targets"]
    )
    guard.authorize_batch_with_identity(
        batch["claims"],
        actual_nonignored_targets=batch["actual_nonignored_targets"],
        expected_next_exposure_identity_sha256=base_identity,
    )
    return observed
