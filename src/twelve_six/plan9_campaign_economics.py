"""Plan-9 bounded campaign economics decisions; never an optimizer or spend authority.

All costs use integer micro-USD. Missing observations are None, never fabricated zeros.
A self-hashed receipt is only an accounting proposal, NOT trusted external budget,
independent evaluation, training authority, or permission to launch.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from twelve_six.plan9_optional_evolution import verify_not_activated

SCHEMA = 1
_COST_CATEGORIES = frozenset({"compute", "storage", "api"})
_INPUT_FIELDS = frozenset({
    "schema_version", "campaign_id", "source_section", "cost_estimate_microusd",
    "cost_observed_microusd", "estimated_seconds", "observed_seconds",
    "expected_gain_ppm", "observed_gain_ppm", "capacity_impact_bytes",
    "budget_ceiling_microusd", "owner_approval_ref",
})
_RECEIPT_FIELDS = frozenset({
    "schema_version", "plan", "section", "campaign_id", "source_receipt_sha256",
    "proposal", "cost_estimate_total_microusd", "cost_observed_total_microusd",
    "time_per_expected_gain_unit", "decision", "reason_codes",
    "external_budget_verified", "training_authorized", "compute_authorized",
    "launch_permitted", "champion_promoted", "receipt_sha256",
})
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}\Z")


class CampaignEconomicsDenied(ValueError):
    """Invalid or untrusted budget/evidence packet is rejected without action."""


def _digest(packet: Mapping[str, Any]) -> str:
    try:
        raw = json.dumps(packet, sort_keys=True, separators=(",", ":"),
                         allow_nan=False, ensure_ascii=True)
    except (TypeError, ValueError) as exc:
        raise CampaignEconomicsDenied("noncanonical accounting packet") from exc
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _amount(value: Any, label: str, *, nullable: bool = True) -> int | None:
    if value is None and nullable:
        return None
    if type(value) is not int or value < 0:
        raise CampaignEconomicsDenied(f"{label}: expected nonnegative integer micro-unit")
    return value


def _gain(value: Any, label: str) -> int | None:
    if value is not None and type(value) is not int:
        raise CampaignEconomicsDenied(f"{label}: expected exact integer ppm or unknown")
    return value


def _costs(value: Any, label: str) -> dict[str, int | None]:
    if type(value) is not dict or frozenset(value) != _COST_CATEGORIES:
        raise CampaignEconomicsDenied(f"{label}: compute/storage/api must all be recorded")
    return {key: _amount(value[key], f"{label}.{key}") for key in sorted(_COST_CATEGORIES)}


def _sum_if_complete(costs: Mapping[str, int | None]) -> int | None:
    if any(cost is None for cost in costs.values()):
        return None
    return sum(costs.values())  # type: ignore[arg-type]


def assess_campaign_economics(
    proposal: Mapping[str, Any], source_decision: Mapping[str, Any]
) -> dict[str, Any]:
    """Return a sealed STOP/NO_GO accounting ledger without launch authority.

    REVIEW_ONLY is never an approval to train or spend. A future execution owner
    must independently verify budgets, corpus, model, compute, and run authority.
    """
    if type(proposal) is not dict or frozenset(proposal) != _INPUT_FIELDS:
        raise CampaignEconomicsDenied("proposal has missing or extra fields")
    if proposal.get("schema_version") != SCHEMA or type(proposal["schema_version"]) is not int:
        raise CampaignEconomicsDenied("unsupported proposal version")
    cid = proposal.get("campaign_id")
    if type(cid) is not str or _ID.fullmatch(cid) is None:
        raise CampaignEconomicsDenied("invalid campaign ID")
    section = proposal["source_section"]
    if type(section) is not int or section not in (17, 18, 19):
        raise CampaignEconomicsDenied("expected exact Plan9 conditional producer")
    try:
        verify_not_activated(source_decision)
    except ValueError as exc:
        raise CampaignEconomicsDenied("source decision is invalid") from exc
    if source_decision.get("section") != section:
        raise CampaignEconomicsDenied("cross-section source mismatch")
    estimated = _costs(proposal["cost_estimate_microusd"], "estimate")
    observed = _costs(proposal["cost_observed_microusd"], "observed")
    duration = _amount(proposal["estimated_seconds"], "estimated_seconds")
    actual_duration = _amount(proposal["observed_seconds"], "observed_seconds")
    expected = _gain(proposal["expected_gain_ppm"], "expected_gain_ppm")
    actual = _gain(proposal["observed_gain_ppm"], "observed_gain_ppm")
    capacity = proposal["capacity_impact_bytes"]
    if capacity is not None and type(capacity) is not int:
        raise CampaignEconomicsDenied("capacity delta must be integer bytes or unknown")
    ceiling = _amount(proposal["budget_ceiling_microusd"], "budget_ceiling")
    approval = proposal["owner_approval_ref"]
    if approval is not None and (type(approval) is not str or _SHA256.fullmatch(approval) is None):
        raise CampaignEconomicsDenied("owner approval is only an opaque SHA256 reference")
    projected = _sum_if_complete(estimated)
    incurred = _sum_if_complete(observed)
    time_metric = None
    if duration is not None and expected is not None and expected > 0:
        # Smaller metric means less wall time per unit of expected quality gain.
        time_metric = (duration * 1_000_000 + expected - 1) // expected

    reasons: list[str] = []
    if expected is not None and expected <= 0:
        reasons.append("STOP_NO_EXPECTED_QUALITY_GAIN")
    if actual is not None and actual <= 0:
        reasons.append("STOP_NO_OBSERVED_QUALITY_GAIN")
    if projected is not None and ceiling is not None and projected > ceiling:
        reasons.append("STOP_FORECAST_EXCEEDS_BUDGET")
    if incurred is not None and ceiling is not None and incurred > ceiling:
        reasons.append("STOP_OBSERVED_EXCEEDS_BUDGET")
    if projected is not None and incurred is not None and incurred > projected:
        reasons.append("STOP_COST_OVERRUN")
    if duration is not None and actual_duration is not None and actual_duration > duration:
        reasons.append("STOP_TIME_OVERRUN")
    if not reasons:
        if projected is None or duration is None or capacity is None or expected is None:
            reasons.append("NO_GO_ECONOMIC_FORECAST_INCOMPLETE")
        if ceiling is None or approval is None:
            reasons.append("NO_GO_OWNER_BUDGET_NOT_VERIFIED")
        if source_decision["outcome"] != "NOT_ACTIVATED":
            raise CampaignEconomicsDenied("unsupported upstream outcome")
        reasons.append("NO_GO_UPSTREAM_NOT_ACTIVATED")
    decision = "STOP" if any(x.startswith("STOP_") for x in reasons) else "NO_GO"
    # A signed/hash-shaped approval reference is *not* independently trusted.
    # This module can NEVER produce launch permission or override Plan9 upstream.
    frozen_input = {
        **dict(proposal), "cost_estimate_microusd": estimated,
        "cost_observed_microusd": observed,
    }
    packet: dict[str, Any] = {
        "schema_version": 1, "plan": 9, "section": 20, "campaign_id": cid,
        "source_receipt_sha256": source_decision["receipt_sha256"],
        "proposal": frozen_input, "cost_estimate_total_microusd": projected,
        "cost_observed_total_microusd": incurred,
        "time_per_expected_gain_unit": time_metric,
        "decision": decision, "reason_codes": reasons,
        "external_budget_verified": False, "training_authorized": False,
        "compute_authorized": False, "launch_permitted": False,
        "champion_promoted": False,
    }
    packet["receipt_sha256"] = _digest(packet)
    return packet


def verify_campaign_economics_receipt(
    receipt: Mapping[str, Any], source_decision: Mapping[str, Any],
    *, expected_receipt_sha256: str,
) -> None:
    """Verify exact pinned receipt with a real Plan9 source, including restart."""
    if type(receipt) is not dict or frozenset(receipt) != _RECEIPT_FIELDS:
        raise CampaignEconomicsDenied("wrong receipt envelope")
    if (type(expected_receipt_sha256) is not str
            or _SHA256.fullmatch(expected_receipt_sha256) is None):
        raise CampaignEconomicsDenied("independent expected SHA256 required")
    if receipt.get("receipt_sha256") != expected_receipt_sha256:
        raise CampaignEconomicsDenied("receipt differs from independently pinned identity")
    expected = assess_campaign_economics(receipt.get("proposal"), source_decision)
    if receipt != expected:
        raise CampaignEconomicsDenied("forged, obsolete or noncanonical decision")


def fastest_review_candidate(receipts: Sequence[Mapping[str, Any]]) -> str | None:
    """Advisory ranking only; excludes stopped/blocked and never grants authority.

    For now all upstream scale receipts are NOT_ACTIVATED, so returns None.
    This intentionally cannot prioritize parameter count or authorize a launch.
    """
    for receipt in receipts:
        if receipt.get("decision") not in ("NO_GO", "STOP"):
            raise CampaignEconomicsDenied("unverified review candidate cannot be ranked")
    return None
