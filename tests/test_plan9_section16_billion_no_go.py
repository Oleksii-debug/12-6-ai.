"""Plan9 S16: strict no-go/restart/forgery regression against production blockers."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from twelve_six.plan9_billion_no_go import (
    BillionDecisionDenied,
    decide_1b_no_go,
    verify_1b_no_go,
)

ROOT = Path(__file__).resolve().parents[1]


def inputs():
    r = json.loads((ROOT / "configs/research/r01_learned20m_launch_readiness_v1.json")
                   .read_text(encoding="utf-8"))
    e = json.loads((ROOT / "configs/research/plan9_section20_economics_no_go_v1.json")
                   .read_text(encoding="utf-8"))
    return r, e


def test_no_go_receipt_and_repeated_cold_restart(tmp_path):
    readiness, economics = inputs()
    receipt = decide_1b_no_go(readiness, economics)
    assert receipt["decision"] == "NO_GO"
    assert receipt["section"] == 16 and receipt["plan"] == 9
    assert receipt["systems_admission"] == "NOT_EXECUTED_PRODUCTION"
    assert receipt["one_b_cost_observation"] is None
    assert receipt["one_b_capacity_observation"] is None
    assert receipt["learned_parent_200m"] is None
    for key in ("candidate", "real_backend", "champion_promoted",
                "training_authorized", "compute_authorized", "launch_authorized"):
        assert receipt[key] is False
    pinned = receipt["receipt_sha256"]
    for attempt in range(2):
        p = tmp_path / f"restarted-{attempt}.json"
        p.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
        reread = json.loads(p.read_text(encoding="utf-8"))
        verify_1b_no_go(reread, readiness, economics, expected_receipt_sha256=pinned)
        assert reread == decide_1b_no_go(readiness, economics)


@pytest.mark.parametrize("path,value", [
    (("evidence", "corpus", "manifest_sha256"), "a" * 64),
    (("evidence", "corpus", "packing_sha256"), "a" * 64),
    (("evidence", "tokenizer", "identity_sha256"), "a" * 64),
    (("evidence", "loss_ledger", "unique_causal_loss_positions"), 10),
    (("evidence", "loss_ledger", "unique_causal_loss_positions"), False),
    (("evidence", "training_recipe", "status"), "ACCEPTED"),
    (("evidence", "compute_authorization", "status"), "AUTHORIZED"),
    (("evidence", "training_authorization", "status"), "AUTHORIZED"),
    (("evidence", "bounded_pilot", "status"), "PASS"),
    (("evidence", "cost_envelope", "status"), "ESTIMATED"),
])
def test_changed_or_spoofed_readiness_denied(path, value):
    readiness, economics = inputs()
    obj = readiness
    for key in path[:-1]:
        obj = obj[key]
    obj[path[-1]] = value
    with pytest.raises(BillionDecisionDenied):
        decide_1b_no_go(readiness, economics)


@pytest.mark.parametrize("field,value", [
    ("decision", "GO"), ("launch_authorized", True),
    ("compute_authorized", True), ("training_authorized", True),
    ("champion_promoted", True), ("candidate", True),
    ("real_backend", True), ("plan7_systems_source_git_blob_sha1", "0" * 40),
    ("one_b_cost_observation", 0), ("readiness_sha256", "a" * 64),
    ("receipt_sha256", "a" * 64), ("section", 17),
    ("extra", "bypass"),
])
def test_tampered_packet_denied_even_if_resealed(field, value):
    readiness, economics = inputs()
    packet = decide_1b_no_go(readiness, economics)
    pinned = packet["receipt_sha256"]
    modified = copy.deepcopy(packet)
    modified[field] = value
    with pytest.raises(BillionDecisionDenied):
        verify_1b_no_go(modified, readiness, economics, expected_receipt_sha256=pinned)


def test_forged_economics_owner_or_activation_refused():
    readiness, economics = inputs()
    for field, value in (("decision", "GO"), ("launch_permitted", True),
                         ("external_budget_verified", True)):
        forged = copy.deepcopy(economics)
        forged[field] = value
        with pytest.raises(BillionDecisionDenied):
            decide_1b_no_go(readiness, forged)


def test_checked_in_no_go_receipt_is_canonical():
    readiness, economics = inputs()
    stored = json.loads((ROOT / "configs/research/plan9_section16_billion_no_go_v1.json")
                        .read_text(encoding="utf-8"))
    expected = decide_1b_no_go(readiness, economics)
    assert stored == expected
    verify_1b_no_go(stored, readiness, economics,
                    expected_receipt_sha256=stored["receipt_sha256"])
