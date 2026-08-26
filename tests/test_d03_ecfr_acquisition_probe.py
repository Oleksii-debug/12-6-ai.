from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "validate_d03_ecfr_acquisition_probe.py"
CONFIG_PATH = ROOT / "configs" / "data" / "d03_ecfr_acquisition_probe_v1.json"

SPEC = importlib.util.spec_from_file_location("validate_d03_ecfr_acquisition_probe", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _load() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _rehash(config: dict) -> dict:
    config["contract_identity_sha256"] = MODULE._canonical_identity(config)
    return config


def test_canonical_probe_validates() -> None:
    result = MODULE.validate_probe(_load())
    assert result["valid"] is True
    assert result["canonical_capacity_credit_bytes"] == 0
    assert result["authorized_unique_loss_positions"] == 0
    assert result["training_authorized"] is False
    assert result["paid_compute_authorized"] is False


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("materialization", "executed"), True),
        (("materialization", "canonical_capacity_credit_bytes"), 1),
        (("materialization", "family_credit"), 1),
        (("source", "title_or_agency_multiplication_allowed"), True),
        (("source", "point_in_time_required"), False),
        (("rights", "public_availability_is_rights_authority"), True),
        (("rights", "blanket_us_government_work_assumption_allowed"), True),
        (("rights", "rights_review_required_before_training_eligibility"), False),
        (("claims", "training_eligible"), True),
        (("claims", "tokenizer_fit_authorized"), True),
        (("claims", "paid_compute_authorized"), True),
        (("claims", "authorized_unique_loss_positions"), 1),
        (("execution", "bulk_download_executed"), True),
        (("execution", "final_test_accessed"), True),
        (("execution", "dedicated_actions_workflow_added"), True),
    ],
)
def test_fail_closed_mutations(path: tuple[str, str], value: object) -> None:
    candidate = copy.deepcopy(_load())
    candidate[path[0]][path[1]] = value
    _rehash(candidate)
    with pytest.raises(MODULE.ProbeValidationError):
        MODULE.validate_probe(candidate)


def test_mutable_current_endpoint_cannot_become_authority() -> None:
    candidate = copy.deepcopy(_load())
    candidate["source"]["current_or_latest_endpoint_authority"] = "MATERIALIZATION_AUTHORITY"
    _rehash(candidate)
    with pytest.raises(MODULE.ProbeValidationError, match="mutable current/latest"):
        MODULE.validate_probe(candidate)


def test_rights_exclusion_cannot_be_removed() -> None:
    candidate = copy.deepcopy(_load())
    candidate["rights"]["must_exclude_or_separately_clear"].remove(
        "incorporated_by_reference_material"
    )
    _rehash(candidate)
    with pytest.raises(MODULE.ProbeValidationError, match="exclusions"):
        MODULE.validate_probe(candidate)


def test_successor_gate_cannot_be_skipped() -> None:
    candidate = copy.deepcopy(_load())
    candidate["required_successor_gates"].remove("GLOBAL_EXACT_NEAR_FRAGMENT_LINEAGE_DEDUP")
    _rehash(candidate)
    with pytest.raises(MODULE.ProbeValidationError, match="successor gate"):
        MODULE.validate_probe(candidate)


def test_gate_cannot_be_self_promoted() -> None:
    candidate = copy.deepcopy(_load())
    candidate["gate_state"]["global_dedup"] = "PASS"
    _rehash(candidate)
    with pytest.raises(MODULE.ProbeValidationError, match="NOT_RUN"):
        MODULE.validate_probe(candidate)


def test_identity_tamper_is_rejected() -> None:
    candidate = copy.deepcopy(_load())
    candidate["discovery_observation"]["non_reserved_titles_observed"] = 48
    with pytest.raises(MODULE.ProbeValidationError, match="identity mismatch"):
        MODULE.validate_probe(candidate)
