from __future__ import annotations

import copy
import json
from pathlib import Path

from twelve_six.pandas_source_authority import (
    CONFIG_PATH,
    RECEIPT_FILE_SHA256,
    validate_pandas_source_authority,
    validate_pandas_source_authority_files,
)

ROOT = Path(__file__).resolve().parents[1]


def _load() -> tuple[dict, bytes]:
    config = json.loads((ROOT / CONFIG_PATH).read_text(encoding="utf-8"))
    receipt = (ROOT / config["historical_execution"]["receipt_path"]).read_bytes()
    return config, receipt


def test_current_main_pandas_source_authority_is_zero_credit_and_valid() -> None:
    config, receipt = _load()
    assert validate_pandas_source_authority(config, receipt) == []
    assert validate_pandas_source_authority_files(ROOT) == []
    assert config["current_composition"]["canonical_capacity_credit_bytes"] == 0
    assert config["truth_boundary"]["authorized_optimized_target_exposure"] == 0


def test_exact_historical_receipt_is_cryptographically_bound() -> None:
    config, receipt = _load()
    assert config["historical_execution"]["receipt_file_sha256"] == RECEIPT_FILE_SHA256
    tampered = receipt.replace(b'"verdict":"ADMIT"', b'"verdict":"REJECT"', 1)
    assert "historical_receipt_file_sha256_mismatch" in validate_pandas_source_authority(
        config, tampered
    )


def test_source_identity_drift_fails_closed() -> None:
    config, receipt = _load()
    mutated = copy.deepcopy(config)
    mutated["bounded_source"]["raw_sha256"] = "0" * 64
    assert "bounded_source_identity_mismatch" in validate_pandas_source_authority(mutated, receipt)


def test_historical_dedup_cannot_be_promoted_to_current_authority() -> None:
    config, receipt = _load()
    mutated = copy.deepcopy(config)
    mutated["historical_dedup"]["current_global_authority"] = True
    blockers = validate_pandas_source_authority(mutated, receipt)
    assert "historical_dedup_boundary_mismatch" in blockers


def test_current_global_dedup_cannot_be_fabricated_as_complete() -> None:
    config, receipt = _load()
    mutated = copy.deepcopy(config)
    mutated["current_composition"]["global_dedup"] = "PASS"
    assert "current_composition_global_dedup_mismatch" in validate_pandas_source_authority(
        mutated, receipt
    )


def test_capacity_credit_cannot_be_created_by_source_authority() -> None:
    config, receipt = _load()
    mutated = copy.deepcopy(config)
    mutated["current_composition"]["canonical_capacity_credit_bytes"] = 15837
    blockers = validate_pandas_source_authority(mutated, receipt)
    assert "current_composition_canonical_capacity_credit_bytes_must_be_exact_int_zero" in blockers


def test_evaluation_use_cannot_be_silently_admitted() -> None:
    config, receipt = _load()
    mutated = copy.deepcopy(config)
    mutated["license"]["evaluation"] = "ALLOWED"
    assert "license_identity_or_use_boundary_mismatch" in validate_pandas_source_authority(
        mutated, receipt
    )


def test_truth_boundary_rejects_bool_masquerading_as_zero() -> None:
    config, receipt = _load()
    mutated = copy.deepcopy(config)
    mutated["truth_boundary"]["authorized_optimized_target_exposure"] = False
    assert "authorized_optimized_target_exposure_must_be_exact_int_zero" in (
        validate_pandas_source_authority(mutated, receipt)
    )


def test_unknown_top_level_authority_is_rejected() -> None:
    config, receipt = _load()
    mutated = copy.deepcopy(config)
    mutated["legacy_registry_authority"] = True
    assert "config_top_level_keys_mismatch" in validate_pandas_source_authority(mutated, receipt)
