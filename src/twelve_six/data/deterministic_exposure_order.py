from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from twelve_six.data.identity_safe_exposure_guard import IdentitySafeExposureReplayGuard
from twelve_six.data.unique_loss_ledger_v2 import LedgerError

_PLAN_SCHEMA = "12-6.deterministic-exposure-order.v1"
_BATCH_KEYS = frozenset(
    {
        "global_batch_index",
        "shard_index",
        "worker_id",
        "claims",
        "actual_nonignored_targets",
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


def ordered_next_exposure_identity(
    guard: IdentitySafeExposureReplayGuard,
    plan: Mapping[str, Any],
    *,
    batch_index: int,
) -> str:
    """Bind the guard's next exposure to exact deterministic scheduling context."""
    index = _nonnegative_int(batch_index, "batch_index")
    if set(plan) != {
        "schema_version",
        "num_workers",
        "batches_per_shard",
        "shard_count",
        "batches",
        "plan_identity_sha256",
    }:
        raise LedgerError("exposure plan fields do not match schema")
    if plan["schema_version"] != _PLAN_SCHEMA:
        raise LedgerError("unsupported exposure plan schema")
    expected_plan_identity = plan["plan_identity_sha256"]
    unhashed = dict(plan)
    unhashed.pop("plan_identity_sha256")
    if not isinstance(expected_plan_identity, str) or _canonical_sha256(unhashed) != expected_plan_identity:
        raise LedgerError("exposure plan self-hash mismatch")
    batches = plan["batches"]
    if not isinstance(batches, Sequence) or index >= len(batches):
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
    batch = plan[batch_index]
    base_identity = guard.next_exposure_identity(
        batch["claims"], actual_nonignored_targets=batch["actual_nonignored_targets"]
    )
    guard.authorize_batch_with_identity(
        batch["claims"],
        actual_nonignored_targets=batch["actual_nonignored_targets"],
        expected_next_exposure_identity_sha256=base_identity,
    )
    return observed
