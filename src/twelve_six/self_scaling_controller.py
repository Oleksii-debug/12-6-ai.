"""Plan 7 S3: bounded, deterministic, advisory self-scaling decisions.

This does not start training, access external tools/data, or authorize compute.
Measured evidence is caller-supplied and valid only within its pinned fixture;
real-world promotion requires a separate trusted evidence and launch authority.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from typing import Any

SCHEMA_VERSION = 1
ACTIONS = (
    "SAME_SIZE_LEARNING",
    "MEMORY_RAG",
    "TOOL_USE",
    "ARCHITECTURE_CHANGE",
    "LARGER_MODEL",
)
_SHA = re.compile(r"[0-9a-f]{64}\Z")


class ScalingDecisionRejected(ValueError):
    """Invalid control packet; no action may be selected."""


def _finite(name: str, value: float, *, positive: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ScalingDecisionRejected(f"{name}: expected finite number")
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ScalingDecisionRejected(f"{name}: invalid finite bound")


def _integer(name: str, value: int, *, allow_zero: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < int(not allow_zero):
        raise ScalingDecisionRejected(f"{name}: invalid integer bound")


def _hash(name: str, value: str) -> None:
    if not isinstance(value, str) or _SHA.fullmatch(value) is None:
        raise ScalingDecisionRejected(f"{name}: expected lowercase SHA-256")


def _digest(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ScalingProposal:
    """One independently measured/proposed path; never a resource grant."""

    action: str
    evidence_sha256: str
    observations: int
    measured_gain: float
    expected_gain: float
    measured_cost_units: float
    projected_cost_units: float
    projected_peak_memory_bytes: int
    projected_flops: int
    workers_required: int
    healthy_workers: int
    resource_class: str = "LOCAL_FREE"
    paid_compute_requested: bool = False

    def __post_init__(self) -> None:
        if self.action not in ACTIONS:
            raise ScalingDecisionRejected("unknown scaling action")
        _hash("evidence_sha256", self.evidence_sha256)
        _integer("observations", self.observations, allow_zero=True)
        for field in (
            "measured_gain", "expected_gain", "measured_cost_units", "projected_cost_units"
        ):
            _finite(field, getattr(self, field))
        for field in (
            "projected_peak_memory_bytes", "projected_flops", "healthy_workers"
        ):
            _integer(field, getattr(self, field), allow_zero=True)
        _integer("workers_required", self.workers_required)
        if not isinstance(self.resource_class, str) or not self.resource_class:
            raise ScalingDecisionRejected("missing resource class")
        if not isinstance(self.paid_compute_requested, bool):
            raise ScalingDecisionRejected("invalid paid-compute marker")


@dataclass(frozen=True, slots=True)
class ScalingEnvelope:
    """Local-only maximum cost/resources and minimum decision quality."""

    version: int
    run_id: str
    epoch: int
    modelspec_sha256: str
    protocol_sha256: str
    remaining_cost_units: float
    min_expected_gain: float
    min_value_per_cost: float
    min_observations: int
    max_peak_memory_bytes: int
    max_flops: int
    max_workers: int
    halted: bool = False

    def __post_init__(self) -> None:
        if self.version != SCHEMA_VERSION:
            raise ScalingDecisionRejected("unsupported controller contract version")
        if not isinstance(self.run_id, str) or not 1 <= len(self.run_id) <= 128:
            raise ScalingDecisionRejected("invalid run identity")
        _integer("epoch", self.epoch, allow_zero=True)
        for field in ("modelspec_sha256", "protocol_sha256"):
            _hash(field, getattr(self, field))
        for field in ("remaining_cost_units", "min_expected_gain", "min_value_per_cost"):
            _finite(field, getattr(self, field))
        for field in (
            "min_observations", "max_peak_memory_bytes", "max_flops", "max_workers"
        ):
            _integer(field, getattr(self, field))
        if not isinstance(self.halted, bool):
            raise ScalingDecisionRejected("invalid stop marker")


@dataclass(frozen=True, slots=True)
class ScalingDecision:
    schema_version: int
    run_id: str
    epoch: int
    action: str
    reason: str
    proposal_evidence_sha256: str | None
    measured_gain: float
    expected_gain: float
    projected_cost_units: float
    evidence_packet_sha256: str
    rejected: tuple[str, ...]
    compute_authorized: bool
    training_authorized: bool
    model_promotion_authorized: bool
    receipt_sha256: str


def choose_scaling_action(
    envelope: ScalingEnvelope, proposals: tuple[ScalingProposal, ...]
) -> ScalingDecision:
    """Select one admissible *advisory* action or STOP; no side effects.

    Compare the same measured gain units across proposals, with conservative
    min(measured,expected) / (1 + max(measured, projected cost)). All candidate identities,
    packet resources, denials and stop state bind the deterministic receipt.
    """
    if not isinstance(envelope, ScalingEnvelope):
        raise ScalingDecisionRejected("expected typed ScalingEnvelope")
    if not isinstance(proposals, tuple) or len(proposals) > len(ACTIONS):
        raise ScalingDecisionRejected("expected bounded proposal tuple")
    if any(not isinstance(p, ScalingProposal) for p in proposals):
        raise ScalingDecisionRejected("untyped scaling proposal")
    if len({p.action for p in proposals}) != len(proposals):
        raise ScalingDecisionRejected("duplicate scaling action")

    identity = _digest({
        "envelope": asdict(envelope),
        "proposals": sorted((asdict(p) for p in proposals), key=lambda p: p["action"]),
    })
    rejected: list[str] = []
    selected: ScalingProposal | None = None
    selected_score = -1.0

    if not envelope.halted and envelope.remaining_cost_units > 0:
        for p in sorted(proposals, key=lambda p: ACTIONS.index(p.action)):
            reason = None
            if p.resource_class != "LOCAL_FREE" or p.paid_compute_requested:
                reason = "paid_or_nonlocal_compute_denied"
            elif p.observations < envelope.min_observations:
                reason = "insufficient_measurements"
            elif (
                p.healthy_workers < p.workers_required
                or p.workers_required > envelope.max_workers
            ):
                reason = "worker_capacity_or_failure"
            elif (
                p.projected_peak_memory_bytes > envelope.max_peak_memory_bytes
                or p.projected_flops > envelope.max_flops
                or p.projected_cost_units > envelope.remaining_cost_units
            ):
                reason = "resource_or_cost_budget_exceeded"
            elif p.measured_gain <= 0 or p.expected_gain < envelope.min_expected_gain:
                reason = "insufficient_measured_or_expected_gain"
            else:
                score = min(p.measured_gain, p.expected_gain) / (
                    1.0 + max(p.measured_cost_units, p.projected_cost_units)
                )
                if score < envelope.min_value_per_cost:
                    reason = "value_per_cost_below_threshold"
                elif score > selected_score:
                    selected_score = score
                    selected = p
            if reason:
                rejected.append(f"{p.action}:{reason}")

    if selected is None:
        why = "halted" if envelope.halted else (
            "budget_exhausted" if envelope.remaining_cost_units == 0 else "no_admissible_option"
        )
        action = "STOP"
    else:
        why = "evidence_driven_local_free_advisory_only"
        action = selected.action
    fields = {
        "schema_version": SCHEMA_VERSION,
        "run_id": envelope.run_id,
        "epoch": envelope.epoch,
        "action": action,
        "reason": why,
        "proposal_evidence_sha256": selected.evidence_sha256 if selected else None,
        "measured_gain": selected.measured_gain if selected else 0.0,
        "expected_gain": selected.expected_gain if selected else 0.0,
        "projected_cost_units": selected.projected_cost_units if selected else 0.0,
        "evidence_packet_sha256": identity,
        "rejected": tuple(sorted(rejected)),
        "compute_authorized": False,
        "training_authorized": False,
        "model_promotion_authorized": False,
    }
    return ScalingDecision(**fields, receipt_sha256=_digest(fields))


def verify_scaling_decision(
    envelope: ScalingEnvelope,
    proposals: tuple[ScalingProposal, ...],
    record: ScalingDecision,
) -> bool:
    """Fail closed on corrupted/stale receipts after restart or worker transfer."""
    if not isinstance(record, ScalingDecision):
        return False
    return choose_scaling_action(envelope, proposals) == record
