from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime

WORK_KINDS = (
    "PRODUCT_VERTICAL",
    "INDEPENDENT_VERIFY",
    "REDTEAM_AUDIT",
    "INTEGRATION_CONVERGENCE",
    "PERFORMANCE_RUNTIME",
    "DATA_SOURCE_OR_PIPELINE",
    "OPEN_SOURCE_REUSE_RESEARCH",
    "REPRODUCIBILITY_RELEASE",
)

ACTIVE_CLAIM_STATUSES = {"ACTIVE", "BLOCKED"}


@dataclass(frozen=True)
class RoutingSlot:
    lane_issue: int
    work_kind: str


@dataclass(frozen=True)
class Claim:
    issue_number: int
    lane_key: str
    created_at: str
    status: str = "ACTIVE"


def routing_slot(issue_number: int) -> RoutingSlot:
    if isinstance(issue_number, bool) or not isinstance(issue_number, int) or issue_number <= 0:
        raise ValueError("issue_number must be a positive integer")
    lane_issue = 2 + (issue_number % 15)
    work_kind = WORK_KINDS[(issue_number // 15) % len(WORK_KINDS)]
    return RoutingSlot(lane_issue=lane_issue, work_kind=work_kind)


def _normalize_key_part(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("lane-key parts must be non-empty strings")
    tokens = re.findall(r"[A-Z0-9]+", value.upper())
    if not tokens:
        raise ValueError("lane-key part has no ASCII alphanumeric identity")
    return "-".join(tokens)


def canonical_lane_key(lane: str, object_name: str, work_kind: str, qualifier: str) -> str:
    parts = (lane, object_name, work_kind, qualifier)
    key = "|".join(_normalize_key_part(part) for part in parts)
    if len(key) > 180:
        raise ValueError("canonical lane key is too long")
    return key


def _parse_created_at(value: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise ValueError("created_at must be a non-empty ISO-8601 string")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        raise ValueError("created_at must include a timezone")
    return parsed


def winning_claim(claims: Iterable[Claim], lane_key: str) -> Claim | None:
    exact = [
        claim
        for claim in claims
        if claim.lane_key == lane_key and claim.status in ACTIVE_CLAIM_STATUSES
    ]
    if not exact:
        return None
    return min(exact, key=lambda claim: (_parse_created_at(claim.created_at), claim.issue_number))


def package_is_large(dimensions: Iterable[str]) -> bool:
    present = set(dimensions)
    primary = "implementation_or_primary_research" in present
    validation = bool(
        {"focused_tests", "adversarial_or_negative_tests"}.intersection(present)
    )
    evidence = bool(
        {
            "machine_readable_evidence_or_validator",
            "live_authority_binding",
            "end_to_end_or_integration_proof",
            "measured_benchmark_or_reproducibility_proof",
        }.intersection(present)
    )
    return len(present) >= 4 and primary and validation and evidence


def ci_pressure(queued: int, in_progress: int) -> str:
    for value in (queued, in_progress):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("CI counts must be non-negative integers")
    total = queued + in_progress
    if total <= 25:
        return "GREEN"
    if total <= 100:
        return "AMBER"
    return "RED"


def distribution_counts(issue_numbers: Iterable[int]) -> Mapping[tuple[int, str], int]:
    counter: Counter[tuple[int, str]] = Counter()
    for issue_number in issue_numbers:
        slot = routing_slot(issue_number)
        counter[(slot.lane_issue, slot.work_kind)] += 1
    return dict(counter)
