from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

from twelve_six.data.deterministic_exposure_order import (
    _validated_plan_batches,
    ordered_next_exposure_identity,
)
from twelve_six.data.identity_safe_exposure_guard import IdentitySafeExposureReplayGuard
from twelve_six.data.unique_loss_ledger_v2 import LedgerError

ORDERED_RESUME_SCHEMA = "12-6.ordered-exposure-resume-state.v1"
_ORDERED_RESUME_KEYS = frozenset(
    {
        "schema_version",
        "plan_identity_sha256",
        "next_batch_index",
        "next_ordered_exposure_identity_sha256",
        "complete",
        "guard_state",
        "ordered_resume_identity_sha256",
    }
)


def _canonical_sha256(value: Any) -> str:
    payload = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise LedgerError(f"{label} must be a 64-hex SHA-256 string")
    try:
        int(value, 16)
    except ValueError as exc:
        raise LedgerError(f"{label} must be a 64-hex SHA-256 string") from exc
    return value.lower()


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise LedgerError(f"{label} must be a non-negative integer")
    return value


def _expected_prefix_claims(
    batches: Sequence[Mapping[str, Any]], next_batch_index: int
) -> tuple[dict[str, list[list[int]]], int]:
    intervals: dict[str, list[tuple[int, int]]] = {}
    total = 0
    for batch in batches[:next_batch_index]:
        for claim in batch["claims"]:
            segment_id = claim["segment_identity_sha256"]
            start = claim["offset_start"]
            end = claim["offset_end"]
            existing = intervals.setdefault(segment_id, [])
            for old_start, old_end in existing:
                if start < old_end and old_start < end:
                    raise LedgerError("exposure plan prefix replays/overlaps loss positions")
            existing.append((start, end))
            total += end - start
    normalized = {
        segment_id: [[start, end] for start, end in sorted(segment_intervals)]
        for segment_id, segment_intervals in sorted(intervals.items())
    }
    return normalized, total


def _validate_guard_state_prefix(
    guard_state: Mapping[str, Any],
    batches: Sequence[Mapping[str, Any]],
    *,
    next_batch_index: int,
) -> None:
    if not isinstance(guard_state, Mapping):
        raise LedgerError("guard_state must be an object")
    saved_sequence = _nonnegative_int(
        guard_state.get("claim_sequence"), "guard_state.claim_sequence"
    )
    if saved_sequence != next_batch_index:
        raise LedgerError("resume claim sequence does not match deterministic plan prefix")
    expected_claims, expected_count = _expected_prefix_claims(
        batches, next_batch_index
    )
    if guard_state.get("claims") != expected_claims:
        raise LedgerError("resume claims do not match deterministic plan prefix")
    consumed = _nonnegative_int(
        guard_state.get("consumed_loss_positions"),
        "guard_state.consumed_loss_positions",
    )
    if consumed != expected_count:
        raise LedgerError("resume consumed count does not match deterministic plan prefix")


def ordered_resume_state_dict(
    guard: IdentitySafeExposureReplayGuard,
    plan: Mapping[str, Any],
    *,
    expected_plan_identity_sha256: str,
) -> dict[str, Any]:
    """Build a durable checkpoint state bound to the exact deterministic plan prefix."""
    batches, plan_identity = _validated_plan_batches(
        plan,
        expected_plan_identity_sha256=expected_plan_identity_sha256,
    )
    next_batch_index = _nonnegative_int(guard.claim_sequence, "guard.claim_sequence")
    if next_batch_index > len(batches):
        raise LedgerError("guard claim sequence exceeds exposure plan length")

    guard_state = guard.state_dict()
    _validate_guard_state_prefix(
        guard_state,
        batches,
        next_batch_index=next_batch_index,
    )
    complete = next_batch_index == len(batches)
    next_identity = None
    if not complete:
        next_identity = ordered_next_exposure_identity(
            guard,
            plan,
            batch_index=next_batch_index,
            expected_plan_identity_sha256=plan_identity,
        )

    state: dict[str, Any] = {
        "schema_version": ORDERED_RESUME_SCHEMA,
        "plan_identity_sha256": plan_identity,
        "next_batch_index": next_batch_index,
        "next_ordered_exposure_identity_sha256": next_identity,
        "complete": complete,
        "guard_state": guard_state,
    }
    state["ordered_resume_identity_sha256"] = _canonical_sha256(state)
    return state


def load_ordered_resume_state(
    guard: IdentitySafeExposureReplayGuard,
    plan: Mapping[str, Any],
    state: Mapping[str, Any],
    *,
    expected_plan_identity_sha256: str,
    expected_trainer_state_binding: Mapping[str, Any],
) -> str:
    """Restore only an exact plan-prefix state and preserve the guard on rejection."""
    if not isinstance(state, Mapping) or set(state) != _ORDERED_RESUME_KEYS:
        raise LedgerError("ordered resume state fields do not match schema")
    if state.get("schema_version") != ORDERED_RESUME_SCHEMA:
        raise LedgerError("ordered resume state schema mismatch")

    state_copy = deepcopy(dict(state))
    observed_state_identity = _require_sha256(
        state_copy.pop("ordered_resume_identity_sha256", None),
        "ordered_resume_identity_sha256",
    )
    if _canonical_sha256(state_copy) != observed_state_identity:
        raise LedgerError("ordered resume state self-hash mismatch")

    batches, plan_identity = _validated_plan_batches(
        plan,
        expected_plan_identity_sha256=expected_plan_identity_sha256,
    )
    saved_plan_identity = _require_sha256(
        state.get("plan_identity_sha256"), "plan_identity_sha256"
    )
    if saved_plan_identity != plan_identity:
        raise LedgerError("resume plan identity does not match expected handoff")

    next_batch_index = _nonnegative_int(
        state.get("next_batch_index"), "next_batch_index"
    )
    if next_batch_index > len(batches):
        raise LedgerError("resume next batch index exceeds exposure plan length")
    complete = state.get("complete")
    if not isinstance(complete, bool):
        raise LedgerError("resume complete flag must be boolean")
    if complete != (next_batch_index == len(batches)):
        raise LedgerError("resume complete flag does not match exposure plan position")

    guard_state = state.get("guard_state")
    _validate_guard_state_prefix(
        guard_state,
        batches,
        next_batch_index=next_batch_index,
    )
    expected_guard_state_identity = _require_sha256(
        guard_state.get("state_identity_sha256"),
        "guard_state.state_identity_sha256",
    )

    saved_next_identity = state.get("next_ordered_exposure_identity_sha256")
    if complete:
        if saved_next_identity is not None:
            raise LedgerError("completed resume state must not carry a next exposure identity")
    else:
        _require_sha256(
            saved_next_identity,
            "next_ordered_exposure_identity_sha256",
        )

    before = guard.state_dict()
    try:
        guard.load_state_dict(
            guard_state,
            expected_state_identity_sha256=expected_guard_state_identity,
            expected_trainer_state_binding=expected_trainer_state_binding,
        )
        if guard.claim_sequence != next_batch_index:
            raise LedgerError("loaded claim sequence does not match resume position")
        if not complete:
            observed_next_identity = ordered_next_exposure_identity(
                guard,
                plan,
                batch_index=next_batch_index,
                expected_plan_identity_sha256=plan_identity,
            )
            if observed_next_identity != saved_next_identity:
                raise LedgerError("resume next exposure identity does not match saved handoff")
    except LedgerError:
        guard.load_state_dict(
            before,
            expected_state_identity_sha256=before["state_identity_sha256"],
            expected_trainer_state_binding=before["trainer_state_binding"],
        )
        raise

    return observed_state_identity
