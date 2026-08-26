from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tools.validate_d03_ecfr_versioned_probe import (
    ProbeValidationError,
    build_successor_request,
    sha256_json,
    validate_probe_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/data/d03_ecfr_versioned_probe_v1.json"


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_frozen_probe_contract_validates_with_zero_authority() -> None:
    report = validate_probe_contract(_config())

    assert report["status"] == "PASS_ZERO_CREDIT_PROBE_CONTRACT"
    assert report["titles_metadata_as_of"] == "2026-08-06"
    assert report["reserved_titles"] == [35]
    assert report["family_credit"] == 0
    assert report["training_authorized_bytes"] == 0
    assert report["training_authorized_loss_positions"] == 0
    assert report["model_training_authorized"] is False
    assert report["paid_compute_authorized"] is False


def test_successor_request_is_point_in_time_and_deterministic() -> None:
    config = _config()

    first = build_successor_request(config, request_date="2026-08-05", title=12)
    second = build_successor_request(config, request_date="2026-08-05", title=12)

    assert first == second
    assert first["url"] == (
        "https://www.ecfr.gov/api/versioner/v1/full/2026-08-05/title-12.xml"
    )
    assert first["titles_metadata_as_of"] == "2026-08-06"
    assert first["reserved_title_check_passed"] is True
    assert first["two_byte_identical_acquisitions_required"] is True
    assert first["rights_and_provenance_status"] == "NOT_RUN"
    assert first["family_credit"] == 0
    assert first["training_authorized_bytes"] == 0

    identity = first["request_identity_sha256"]
    unhashed = dict(first)
    del unhashed["request_identity_sha256"]
    assert identity == sha256_json(unhashed)


def test_request_newer_than_available_titles_metadata_fails_closed() -> None:
    with pytest.raises(ProbeValidationError, match="newer than the frozen eCFR titles metadata"):
        build_successor_request(_config(), request_date="2026-08-07", title=1)


def test_reserved_title_fails_closed() -> None:
    with pytest.raises(ProbeValidationError, match="title 35 is reserved"):
        build_successor_request(_config(), request_date="2026-08-05", title=35)


@pytest.mark.parametrize("title", [0, 51, -1])
def test_out_of_range_title_fails_closed(title: int) -> None:
    with pytest.raises(ProbeValidationError, match="between 1 and 50"):
        build_successor_request(_config(), request_date="2026-08-05", title=title)


def test_metadata_date_drift_fails_closed() -> None:
    config = _config()
    config["source"]["titles_metadata_as_of"] = "2026-08-07"

    with pytest.raises(ProbeValidationError, match="titles_metadata_as_of"):
        validate_probe_contract(config)


def test_reserved_title_inventory_drift_fails_closed() -> None:
    config = _config()
    config["source"]["reserved_titles_at_observation"] = []

    with pytest.raises(ProbeValidationError, match="reserved_titles_at_observation"):
        validate_probe_contract(config)


def test_mutable_current_capacity_claim_fails_closed() -> None:
    config = _config()
    config["versioning"]["mutable_current_endpoint_allowed_for_capacity"] = True

    with pytest.raises(ProbeValidationError, match="mutable_current_endpoint"):
        validate_probe_contract(config)


def test_rights_overclaim_fails_closed() -> None:
    config = _config()
    config["rights"]["blanket_training_permission_claimed"] = True

    with pytest.raises(ProbeValidationError, match="blanket_training_permission"):
        validate_probe_contract(config)


def test_transferred_copyright_caveat_cannot_be_removed() -> None:
    config = _config()
    config["rights"]["government_can_hold_transferred_copyrights"] = False

    with pytest.raises(ProbeValidationError, match="transferred_copyrights"):
        validate_probe_contract(config)


def test_nonzero_probe_credit_fails_closed() -> None:
    config = _config()
    config["credit"]["candidate_raw_bytes"] = 1

    with pytest.raises(ProbeValidationError, match="candidate_raw_bytes"):
        validate_probe_contract(config)


def test_training_authority_cannot_be_enabled_at_probe_stage() -> None:
    config = _config()
    config["credit"]["model_training_authorized"] = True

    with pytest.raises(ProbeValidationError, match="model_training_authorized"):
        validate_probe_contract(config)


def test_xml_external_entity_safety_cannot_be_relaxed() -> None:
    config = _config()
    config["xml_safety"]["external_entities_allowed"] = True

    with pytest.raises(ProbeValidationError, match="external_entities_allowed"):
        validate_probe_contract(config)


def test_successor_chain_cannot_skip_global_dedup() -> None:
    config = copy.deepcopy(_config())
    config["required_successors"].remove(
        "GLOBAL_EXACT_NEAR_FRAGMENT_AND_LINEAGE_DEDUP"
    )

    with pytest.raises(ProbeValidationError, match="required_successors"):
        validate_probe_contract(config)


def test_historical_endpoint_drift_fails_closed() -> None:
    config = _config()
    config["source"]["historical_title_endpoint_template"] = (
        "https://www.ecfr.gov/current/title-{title}.xml"
    )

    with pytest.raises(ProbeValidationError, match="historical_title_endpoint_template"):
        validate_probe_contract(config)
