from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from twelve_six.hplt3_source_policy import (
    BLOCKED_VERDICT,
    HPLT3ContractError,
    contract_sha256,
    validate_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "configs/data/d03_hplt3_ukr_cyrl_source_audit_v1.json"


def load_contract() -> dict:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def resign(contract: dict) -> dict:
    contract["contract_sha256"] = contract_sha256(contract)
    return contract


def test_canonical_contract_is_valid_and_zero_credit() -> None:
    result = validate_contract(load_contract())
    assert result["verdict"] == BLOCKED_VERDICT
    assert result["training_authorized_bytes"] == 0
    assert result["unique_causal_loss_positions"] == 0


def test_contract_identity_is_deterministic() -> None:
    contract = load_contract()
    assert contract_sha256(contract) == contract["contract_sha256"]
    reordered = dict(reversed(list(contract.items())))
    assert contract_sha256(reordered) == contract["contract_sha256"]


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (
            lambda c: c["rights"].update({"package_license_implies_training_authority": True}),
            "package license may not imply training authority",
        ),
        (
            lambda c: c["rights"].update({"canonical_training_authorized": True}),
            "training authorization overclaim",
        ),
        (
            lambda c: c["credit"].update({"training_authorized_bytes": 1}),
            "training_authorized_bytes must remain 0",
        ),
        (
            lambda c: c["credit"].update({"unique_causal_loss_positions": 1}),
            "unique_causal_loss_positions must remain 0",
        ),
        (
            lambda c: c["project_gates"].update({"privacy_review": "PASS"}),
            "privacy_review must remain BLOCKED",
        ),
        (
            lambda c: c["acquisition"].update({"immutable_acquisition_identity": True}),
            "immutable acquisition identity overclaim",
        ),
        (
            lambda c: c["upstream"].update({"dataset_card_commit": "main"}),
            "immutable dataset-card commit drift",
        ),
    ],
)
def test_adversarial_overclaims_fail_closed(mutator, match: str) -> None:
    contract = copy.deepcopy(load_contract())
    mutator(contract)
    resign(contract)
    with pytest.raises(HPLT3ContractError, match=match):
        validate_contract(contract)


def test_tampering_without_resigning_fails_identity() -> None:
    contract = load_contract()
    contract["upstream"]["published_ukrainian_statistics"]["sorted_size_human"] = "999 TB"
    with pytest.raises(HPLT3ContractError, match="contract SHA-256 mismatch"):
        validate_contract(contract)


def test_missing_project_gate_fails_closed() -> None:
    contract = copy.deepcopy(load_contract())
    del contract["project_gates"]["evaluation_decontamination"]
    resign(contract)
    with pytest.raises(HPLT3ContractError, match="project gate set drift"):
        validate_contract(contract)
