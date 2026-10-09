"""Plan 9 S17–S18 fail-closed decisions, persistence, restart, forgeries."""
import copy
import json

import pytest

from twelve_six.plan9_optional_evolution import (
    EvolutionDecisionDenied,
    not_activated,
    verify_not_activated,
)


@pytest.mark.parametrize("section,tiers", [
    (17, ["3B", "7B", "13B"]),
    (18, ["30B", "70B", "100B"]),
])
def test_terminal_nonactivation(section, tiers, tmp_path):
    receipt = not_activated(section)
    assert receipt["outcome"] == "NOT_ACTIVATED"
    assert receipt["tiers"] == tiers
    assert not any(receipt[k] for k in (
        "activated", "trained", "production_candidate", "champion_promoted",
        "training_authorized", "compute_authorized", "release_blocking"))
    verify_not_activated(receipt)
    assert receipt == not_activated(section)
    root = tmp_path / "decisions.json"
    root.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    verify_not_activated(json.loads(root.read_text(encoding="utf-8")))


@pytest.mark.parametrize("section", [0, 16, 20, True, "17", None])
def test_wrong_section_denied(section):
    with pytest.raises(EvolutionDecisionDenied):
        not_activated(section)


@pytest.mark.parametrize("activation", [True, 1, "yes", None])
def test_activation_cannot_bypass(activation):
    with pytest.raises(EvolutionDecisionDenied):
        not_activated(17, owner_activation=activation)


@pytest.mark.parametrize("change", [
    {"activated": True}, {"training_authorized": 1}, {"compute_authorized": True},
    {"production_candidate": True}, {"champion_promoted": True},
    {"tiers": ["13B", "7B", "3B"]}, {"release_blocking": True},
    {"outcome": "DONE_TRAINED"}, {"producer_git_blob_sha1": "0" * 40},
    {"activation_prerequisites": []}, {"reason_codes": []},
    {"receipt_sha256": "0" * 64}, {"new": "surprise"},
])
@pytest.mark.parametrize("section", [17, 18])
def test_forgery_fails_closed(change, section):
    packet = copy.deepcopy(not_activated(section))
    packet.update(change)
    with pytest.raises(EvolutionDecisionDenied):
        verify_not_activated(packet)


def test_renumbering_without_producer_binding_denied():
    packet = not_activated(17)
    packet["section"] = 18
    with pytest.raises(EvolutionDecisionDenied):
        verify_not_activated(packet)


@pytest.mark.parametrize("section", [17, 18])
def test_repository_receipt_matches_executable_contract(section):
    from pathlib import Path

    path = (Path(__file__).resolve().parents[1] / "configs" / "research"
            / f"plan9_section{section}_not_activated_v1.json")
    packet = json.loads(path.read_text(encoding="utf-8"))
    verify_not_activated(packet)
    assert packet == not_activated(section)


def test_forged_s18_as_s17_denied():
    packet = copy.deepcopy(not_activated(18))
    packet["section"] = 17
    with pytest.raises(EvolutionDecisionDenied):
        verify_not_activated(packet)


def test_s18_activation_denied_for_any_truthy_owner_request():
    for activation in (True, 1, "yes", None):
        with pytest.raises(EvolutionDecisionDenied):
            not_activated(18, owner_activation=activation)
