from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/validate_eval233_final_test_reservation_current_main.py"
CONFIG = ROOT / "configs/evaluation/eval233_final_test_reservation_current_main_v1.json"
spec = importlib.util.spec_from_file_location("reservation_validator", TOOL)
assert spec and spec.loader
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)


def _load() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def _reseal(data: dict) -> dict:
    body = copy.deepcopy(data)
    body.pop("authority_identity_sha256", None)
    data["authority_identity_sha256"] = hashlib.sha256(
        validator._canonical_bytes(body)
    ).hexdigest()
    return data


def test_canonical_authority_passes() -> None:
    data = validator.validate_path(CONFIG)
    assert data["reservation"]["documents"] == 16
    assert data["firewall"]["authorized_training_exposure"] == 0


def test_self_consistent_final_test_identity_substitution_fails() -> None:
    data = _load()
    data["reservation"]["final_test_identity_sha256"] = "f" * 64
    data["late_bound_decontamination"]["expected_final_test_identity_sha256"] = "f" * 64
    _reseal(data)
    with pytest.raises(validator.ReservationValidationError, match="final-test identity drift"):
        validator.validate_authority(data)


def test_historical_seed_substitution_fails_even_if_resealed() -> None:
    data = _load()
    data["current_main_port"]["recover174_seed_git_blob_sha1"] = "a" * 40
    _reseal(data)
    with pytest.raises(
        validator.ReservationValidationError,
        match="recover174_seed_git_blob_sha1 drift",
    ):
        validator.validate_authority(data)


def test_training_or_early_scoring_cannot_be_enabled() -> None:
    data = _load()
    data["firewall"]["training_allowed"] = True
    _reseal(data)
    with pytest.raises(
        validator.ReservationValidationError,
        match="firewall weakened: training_allowed",
    ):
        validator.validate_authority(data)


def test_payload_or_outcome_data_cannot_be_embedded() -> None:
    data = _load()
    data["reservation"]["text"] = "forbidden"
    _reseal(data)
    with pytest.raises(
        validator.ReservationValidationError,
        match="forbidden payload/outcome key",
    ):
        validator.validate_authority(data)

    data = _load()
    data["late_bound_decontamination"]["outcome"] = "forbidden"
    _reseal(data)
    with pytest.raises(
        validator.ReservationValidationError,
        match="forbidden payload/outcome key",
    ):
        validator.validate_authority(data)


def test_member_level_binding_must_remain_late_bound() -> None:
    data = _load()
    data["late_bound_decontamination"]["member_level_payload_binding_identity_sha256"] = (
        "b" * 64
    )
    _reseal(data)
    with pytest.raises(
        validator.ReservationValidationError,
        match="member-level binding must remain late-bound",
    ):
        validator.validate_authority(data)


def test_source_family_or_count_drift_fails() -> None:
    data = _load()
    data["reservation"]["modality_documents"]["ua"] = 7
    _reseal(data)
    with pytest.raises(validator.ReservationValidationError, match="modality counts drift"):
        validator.validate_authority(data)

    data = _load()
    data["reservation"]["source_families"][0]["admitted_source_snapshots_sha256"] = [
        "c" * 64
    ]
    _reseal(data)
    with pytest.raises(
        validator.ReservationValidationError,
        match="admitted_source_snapshots_sha256 drift",
    ):
        validator.validate_authority(data)
