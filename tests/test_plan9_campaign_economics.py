"""Plan9 S20 economical NO_GO/STOP receipts: no budget bypass or fake GO."""
from __future__ import annotations

import copy
import json

import pytest

from twelve_six.plan9_campaign_economics import (
    CampaignEconomicsDenied,
    assess_campaign_economics,
    fastest_review_candidate,
    verify_campaign_economics_receipt,
)
from twelve_six.plan9_optional_evolution import not_activated

UNKNOWN = {"compute": None, "storage": None, "api": None}


def proposal(section=19, **updates):
    data = {
        "schema_version": 1, "campaign_id": f"PLAN9-SECTION-{section}",
        "source_section": section, "cost_estimate_microusd": dict(UNKNOWN),
        "cost_observed_microusd": dict(UNKNOWN),
        "estimated_seconds": None, "observed_seconds": None,
        "expected_gain_ppm": None, "observed_gain_ppm": None,
        "capacity_impact_bytes": None, "budget_ceiling_microusd": None,
        "owner_approval_ref": None,
    }
    data.update(updates)
    return data


def estimate(**updates):
    values = {"compute": 500_000, "storage": 20_000, "api": 10_000}
    values.update(updates)
    return values


@pytest.mark.parametrize("section", [17, 18, 19])
def test_current_upstream_truth_is_no_go(section, tmp_path):
    source = not_activated(section)
    receipt = assess_campaign_economics(proposal(section), source)
    assert receipt["decision"] == "NO_GO"
    assert "NO_GO_UPSTREAM_NOT_ACTIVATED" in receipt["reason_codes"]
    assert receipt["cost_estimate_total_microusd"] is None
    assert receipt["cost_observed_total_microusd"] is None
    assert receipt["time_per_expected_gain_unit"] is None
    assert receipt["source_receipt_sha256"] == source["receipt_sha256"]
    assert all(receipt[key] is False for key in (
        "external_budget_verified", "training_authorized", "compute_authorized",
        "launch_permitted", "champion_promoted"))
    expected_sha = receipt["receipt_sha256"]
    verify_campaign_economics_receipt(receipt, source, expected_receipt_sha256=expected_sha)
    path = tmp_path / "economic_receipt.json"
    path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    from_disk = json.loads(path.read_text(encoding="utf-8"))
    verify_campaign_economics_receipt(from_disk, source, expected_receipt_sha256=expected_sha)
    assert receipt == assess_campaign_economics(proposal(section), source)


def test_fully_populated_budget_and_positive_expected_gain_still_not_authorization():
    request = proposal(
        cost_estimate_microusd=estimate(),
        cost_observed_microusd=estimate(compute=200_000),
        estimated_seconds=3600, observed_seconds=1800,
        expected_gain_ppm=1000, observed_gain_ppm=600,
        capacity_impact_bytes=123456,
        budget_ceiling_microusd=1_000_000,
        owner_approval_ref="a" * 64,
    )
    receipt = assess_campaign_economics(request, not_activated(19))
    assert receipt["cost_estimate_total_microusd"] == 530_000
    assert receipt["cost_observed_total_microusd"] == 230_000
    assert receipt["time_per_expected_gain_unit"] == 3_600_000
    assert receipt["decision"] == "NO_GO"
    assert receipt["external_budget_verified"] is False
    assert receipt["launch_permitted"] is False
    assert fastest_review_candidate([receipt]) is None


@pytest.mark.parametrize("extra,reason", [
    ({"expected_gain_ppm": 0}, "STOP_NO_EXPECTED_QUALITY_GAIN"),
    ({"expected_gain_ppm": -3}, "STOP_NO_EXPECTED_QUALITY_GAIN"),
    ({"observed_gain_ppm": 0}, "STOP_NO_OBSERVED_QUALITY_GAIN"),
    ({"observed_gain_ppm": -9}, "STOP_NO_OBSERVED_QUALITY_GAIN"),
    ({"cost_estimate_microusd": estimate(),
      "budget_ceiling_microusd": 20}, "STOP_FORECAST_EXCEEDS_BUDGET"),
    ({"cost_observed_microusd": estimate(),
      "budget_ceiling_microusd": 20}, "STOP_OBSERVED_EXCEEDS_BUDGET"),
    ({"cost_estimate_microusd": estimate(compute=100),
      "cost_observed_microusd": estimate(compute=150)}, "STOP_COST_OVERRUN"),
    ({"estimated_seconds": 100, "observed_seconds": 101}, "STOP_TIME_OVERRUN"),
])
def test_economic_stop_reasons(extra, reason):
    receipt = assess_campaign_economics(proposal(**extra), not_activated(19))
    assert receipt["decision"] == "STOP"
    assert reason in receipt["reason_codes"]
    assert receipt["launch_permitted"] is False


@pytest.mark.parametrize("changes", [
    {"schema_version": 2}, {"schema_version": True},
    {"source_section": 16}, {"source_section": True},
    {"campaign_id": "../../etc/passwd"}, {"campaign_id": ""},
    {"cost_estimate_microusd": {"compute": 1}},
    {"cost_observed_microusd": {"compute": 1, "storage": 2, "api": 0, "foo": 4}},
    {"cost_estimate_microusd": estimate(compute=-1)},
    {"cost_estimate_microusd": estimate(compute=True)},
    {"cost_estimate_microusd": estimate(compute=0.5)},
    {"actual_costs": {"api": 0}},
    {"estimated_seconds": True}, {"estimated_seconds": -1},
    {"observed_seconds": 1.5}, {"observed_gain_ppm": float("nan")},
    {"expected_gain_ppm": True}, {"expected_gain_ppm": 1.2},
    {"capacity_impact_bytes": False},
    {"budget_ceiling_microusd": True}, {"budget_ceiling_microusd": -1},
    {"owner_approval_ref": "no-approval"},
    {"training_authorized": True}, {"compute_authorized": True},
    {"paid_compute": True}, {"launch_permitted": True},
])
def test_bad_inputs_do_not_create_receipts(changes):
    data = proposal()
    data.update(changes)
    with pytest.raises(CampaignEconomicsDenied):
        assess_campaign_economics(data, not_activated(19))


@pytest.mark.parametrize("changes", [
    {"decision": "GO"}, {"launch_permitted": True},
    {"compute_authorized": True}, {"training_authorized": True},
    {"champion_promoted": True}, {"external_budget_verified": True},
    {"cost_estimate_total_microusd": 0}, {"cost_observed_total_microusd": 0},
    {"time_per_expected_gain_unit": 0},
    {"source_receipt_sha256": "0" * 64},
    {"receipt_sha256": "0" * 64}, {"plan": 7}, {"section": 19},
    {"reason_codes": []}, {"unexpected": "forge"},
])
def test_tampering_fails_independently_pinned_readback(changes):
    source = not_activated(19)
    receipt = assess_campaign_economics(proposal(), source)
    expected = receipt["receipt_sha256"]
    tampered = copy.deepcopy(receipt)
    tampered.update(changes)
    with pytest.raises(CampaignEconomicsDenied):
        verify_campaign_economics_receipt(
            tampered, source, expected_receipt_sha256=expected)


def test_resealed_forgery_cannot_override_independent_sha():
    source = not_activated(19)
    receipt = assess_campaign_economics(proposal(), source)
    tampered = copy.deepcopy(receipt)
    tampered["launch_permitted"] = True
    # A new self-hash cannot replace the independent verifier's expected hash.
    with pytest.raises(CampaignEconomicsDenied):
        verify_campaign_economics_receipt(
            tampered, source, expected_receipt_sha256=receipt["receipt_sha256"])


def test_wrong_upstream_section_and_forged_source_fail_closed():
    with pytest.raises(CampaignEconomicsDenied):
        assess_campaign_economics(proposal(18), not_activated(19))
    source = not_activated(19)
    source["compute_authorized"] = True
    with pytest.raises(CampaignEconomicsDenied):
        assess_campaign_economics(proposal(), source)


def test_unknown_fake_go_receipt_never_candidate():
    with pytest.raises(CampaignEconomicsDenied):
        fastest_review_candidate([{"decision": "GO"}])


def test_repository_sealed_receipt_matches_current_source():
    from pathlib import Path
    path = (Path(__file__).resolve().parents[1] / "configs" / "research"
            / "plan9_section20_economics_no_go_v1.json")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    source = not_activated(19)
    expected = assess_campaign_economics(proposal(), source)
    assert receipt == expected
    verify_campaign_economics_receipt(
        receipt, source, expected_receipt_sha256=receipt["receipt_sha256"])
