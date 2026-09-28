from __future__ import annotations

import copy
import json
from pathlib import Path

from twelve_six.postgresql_source_authority import (
    CONFIG_PATH,
    RECEIPT_AUTHORITY_ID,
    RECEIPT_GIT_BLOB_SHA1,
    validate_postgresql_source_authority,
    validate_postgresql_source_authority_files,
)

ROOT = Path(__file__).resolve().parents[1]


def _load() -> tuple[dict, bytes]:
    config = json.loads((ROOT / CONFIG_PATH).read_text(encoding="utf-8"))
    receipt = (ROOT / config["historical_execution"]["receipt_path"]).read_bytes()
    return config, receipt


def test_current_main_postgresql_source_authority_is_valid_and_zero_credit() -> None:
    config, receipt = _load()
    assert validate_postgresql_source_authority(config, receipt) == []
    assert validate_postgresql_source_authority_files(ROOT) == []
    assert config["bounded_source"]["normalized_bytes"] == 624335
    assert config["current_composition"]["canonical_capacity_credit_bytes"] == 0
    assert config["truth_boundary"]["authorized_optimized_target_exposure"] == 0


def test_exact_historical_receipt_is_git_and_self_hash_bound() -> None:
    config, receipt = _load()
    historical = config["historical_execution"]
    assert historical["receipt_git_blob_sha1"] == RECEIPT_GIT_BLOB_SHA1
    assert historical["receipt_authority_identity_sha256"] == RECEIPT_AUTHORITY_ID
    tampered = receipt.replace(b'"verdict":"ADMIT"', b'"verdict":"REJECT"', 1)
    blockers = validate_postgresql_source_authority(config, tampered)
    assert "historical_receipt_git_blob_sha1_mismatch" in blockers
    assert "historical_receipt_authority_identity_mismatch" in blockers


def test_source_upstream_identity_drift_fails_closed() -> None:
    config, receipt = _load()
    mutated = copy.deepcopy(config)
    mutated["bounded_source"]["commit"] = "0" * 40
    assert "bounded_source_identity_mismatch" in validate_postgresql_source_authority(
        mutated, receipt
    )


def test_source_count_or_byte_drift_fails_closed() -> None:
    config, receipt = _load()
    for key, value in (("selected_document_count", 11), ("normalized_bytes", 624336)):
        mutated = copy.deepcopy(config)
        mutated["bounded_source"][key] = value
        assert "bounded_source_identity_mismatch" in validate_postgresql_source_authority(
            mutated, receipt
        )


def test_license_or_evaluation_boundary_cannot_be_widened() -> None:
    config, receipt = _load()
    mutated = copy.deepcopy(config)
    mutated["license"]["evaluation"] = "ALLOWED"
    assert "license_identity_or_use_boundary_mismatch" in (
        validate_postgresql_source_authority(mutated, receipt)
    )


def test_historical_registry_dedup_cannot_be_promoted_to_current_authority() -> None:
    config, receipt = _load()
    mutated = copy.deepcopy(config)
    mutated["historical_dedup"]["current_global_authority"] = True
    assert "historical_dedup_boundary_mismatch" in (
        validate_postgresql_source_authority(mutated, receipt)
    )


def test_current_global_dedup_cannot_be_fabricated_as_complete() -> None:
    config, receipt = _load()
    mutated = copy.deepcopy(config)
    mutated["current_composition"]["global_dedup"] = "PASS"
    assert "current_composition_global_dedup_mismatch" in (
        validate_postgresql_source_authority(mutated, receipt)
    )


def test_source_authority_cannot_create_capacity_or_family_credit() -> None:
    config, receipt = _load()
    for key, value in (
        ("canonical_capacity_credit_bytes", 624335),
        ("canonical_family_credit", 1),
        ("canonical_files_credit", 10),
    ):
        mutated = copy.deepcopy(config)
        mutated["current_composition"][key] = value
        blockers = validate_postgresql_source_authority(mutated, receipt)
        assert f"current_composition_{key}_must_be_exact_int_zero" in blockers


def test_truth_boundary_rejects_bool_masquerading_as_zero() -> None:
    config, receipt = _load()
    mutated = copy.deepcopy(config)
    mutated["truth_boundary"]["authorized_optimized_target_exposure"] = False
    assert "authorized_optimized_target_exposure_must_be_exact_int_zero" in (
        validate_postgresql_source_authority(mutated, receipt)
    )


def test_truth_boundary_rejects_training_promotion() -> None:
    config, receipt = _load()
    mutated = copy.deepcopy(config)
    mutated["truth_boundary"]["training_executed"] = True
    assert "training_executed_must_be_false" in (
        validate_postgresql_source_authority(mutated, receipt)
    )


def test_unknown_top_level_authority_is_rejected() -> None:
    config, receipt = _load()
    mutated = copy.deepcopy(config)
    mutated["legacy_registry_authority"] = True
    assert "config_top_level_keys_mismatch" in (
        validate_postgresql_source_authority(mutated, receipt)
    )
