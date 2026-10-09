"""Plan9 S19 exact Plan7 MoE producer, honest NO_GO and adverse persistence."""
import copy
import json

import pytest

from twelve_six.plan9_optional_evolution import (
    EvolutionDecisionDenied,
    not_activated,
    verify_not_activated,
)


SOURCE = "src/twelve_six/extreme_moe_paths.py"
BLOB = "e964c124892bfb004b1266c4fdad2b54dbc3a355"


def test_exact_provider_and_absent_economic_evidence(tmp_path):
    receipt = not_activated(19)
    assert receipt["tiers"] == ["300B", "1T", "MoE"]
    assert (receipt["producer_path"], receipt["producer_git_blob_sha1"]) == (SOURCE, BLOB)
    assert receipt["reason_codes"] == [
        "NO_EXPLICIT_OWNER_SCALE_ACTIVATION",
        "UPSTREAM_PRODUCTION_BASE_MISSING",
        "NO_MEASURED_DENSE_MOE_EFFICIENCY",
    ]
    assert receipt["outcome"] == "NOT_ACTIVATED"
    for key in ("activated", "trained", "production_candidate", "champion_promoted",
                "training_authorized", "compute_authorized", "release_blocking"):
        assert receipt[key] is False
    verify_not_activated(receipt)
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    verify_not_activated(json.loads(path.read_text(encoding="utf-8")))
    assert receipt == not_activated(19)


@pytest.mark.parametrize("activation", [True, 1, "yes", None])
def test_owner_activation_fail_closed(activation):
    with pytest.raises(EvolutionDecisionDenied):
        not_activated(19, owner_activation=activation)


@pytest.mark.parametrize("overrides", [
    {"activated": True}, {"trained": True}, {"compute_authorized": True},
    {"training_authorized": True}, {"production_candidate": True},
    {"champion_promoted": True}, {"release_blocking": True},
    {"tiers": ["MoE", "1T", "300B"]},
    {"producer_path": "src/twelve_six/large_scale_paths.py"},
    {"producer_git_blob_sha1": "0" * 40},
    {"reason_codes": ["NO_EXPLICIT_OWNER_SCALE_ACTIVATION"]},
    {"receipt_sha256": "0" * 64}, {"schema_version": 2},
    {"plan": 7}, {"section": 17}, {"new_field": "bogus"},
])
def test_adversarial_forgery(overrides):
    packet = copy.deepcopy(not_activated(19))
    packet.update(overrides)
    with pytest.raises(EvolutionDecisionDenied):
        verify_not_activated(packet)


def test_repo_receipt_matches_contract():
    from pathlib import Path
    path = (Path(__file__).resolve().parents[1] / "configs" / "research"
            / "plan9_section19_not_activated_v1.json")
    packet = json.loads(path.read_text(encoding="utf-8"))
    verify_not_activated(packet)
    assert packet == not_activated(19)


def test_cross_section_reuse_is_not_promotion():
    for section in (17, 18, 19):
        verify_not_activated(not_activated(section))
    assert not_activated(19)["receipt_sha256"] != not_activated(18)["receipt_sha256"]
