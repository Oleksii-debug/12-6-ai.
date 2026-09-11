from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
CONFIG = ROOT / "configs/data/next100_025_derzhgeocadastre_open_registry_v1.json"
SPEC = importlib.util.spec_from_file_location(
    "derzh_snapshot",
    TOOLS / "next100_025_data_gov_registry_snapshot.py",
)
assert SPEC and SPEC.loader
snapshot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(snapshot)


def _config() -> dict:
    return {
        "family": {"family_id": "ua.data-gov.derzhgeocadastre.dataset-register"},
        "dataset": {"dataset_id": "dataset-id"},
        "rights": {
            "dataset_license_label": "Creative Commons Attribution",
        },
    }


def _item() -> dict:
    return {
        "normalized_sha256": "1" * 64,
        "text": "назва: Реєстр наборів даних України\n",
    }


def test_locked_snapshot_is_identity_lock_not_training_authority() -> None:
    boundary = snapshot.current_main_claim_boundary("LOCKED")

    assert boundary["source_snapshot_identity_locked"] is True
    assert boundary["source_specific_training_rights_compatible"] is True
    assert boundary["candidate_snapshot_only"] is True
    assert boundary["canonical_corpus_admitted"] is False
    assert boundary["family_credit"] is False
    assert boundary["source_capacity_bytes_credited"] == 0
    assert boundary["training_authorized_bytes"] == 0
    assert boundary["authorized_optimized_target_exposure"] == 0
    assert boundary["global_dedup_complete"] is False
    assert boundary["reserved_evaluation_decontamination_complete"] is False
    assert boundary["cluster_safe_split_complete"] is False
    assert boundary["deterministic_packing_complete"] is False
    assert boundary["postpack_unique_loss_ledger_complete"] is False
    assert boundary["tokenizer_fit_authorized"] is False
    assert boundary["optimizer_updates"] == 0
    assert boundary["model_training_executed"] is False
    assert snapshot.snapshot_status("LOCKED") == "SOURCE_SNAPSHOT_LOCKED_ZERO_CREDIT"


def test_locked_candidate_row_cannot_become_training_or_evaluation_eligible() -> None:
    row = snapshot.build_candidate_row(
        cfg=_config(),
        resource={"id": "resource-id"},
        resource_url="https://data.gov.ua/resource.json",
        raw_hash="2" * 64,
        item=_item(),
    )

    assert row["artifact_role"] == "SOURCE_CANDIDATE_ONLY"
    assert row["source_training_rights_compatible"] is True
    assert row["training_eligible"] is False
    assert row["evaluation_eligible"] is False


def test_probe_and_locked_modes_have_same_zero_credit_training_boundary() -> None:
    probe = snapshot.current_main_claim_boundary("PROBE")
    locked = snapshot.current_main_claim_boundary("LOCKED")

    for key in (
        "source_capacity_bytes_credited",
        "training_authorized_bytes",
        "authorized_optimized_target_exposure",
        "evaluation_authorized_bytes",
        "optimizer_updates",
    ):
        assert probe[key] == locked[key] == 0

    for key in (
        "canonical_corpus_admitted",
        "family_credit",
        "global_dedup_complete",
        "reserved_evaluation_decontamination_complete",
        "post_composition_quality_privacy_complete",
        "balance_family_caps_complete",
        "cluster_safe_split_complete",
        "deterministic_packing_complete",
        "postpack_unique_loss_ledger_complete",
        "tokenizer_fit_authorized",
        "model_training_executed",
        "paid_compute_used",
    ):
        assert probe[key] is locked[key] is False


def test_unknown_mode_fails_closed() -> None:
    with pytest.raises(RuntimeError, match="unsupported snapshot mode"):
        snapshot.current_main_claim_boundary("TRAIN")


def test_direct_mode_based_training_eligibility_regression_is_absent() -> None:
    source = (TOOLS / "next100_025_data_gov_registry_snapshot.py").read_text(
        encoding="utf-8"
    )
    assert '"training_eligible": cfg["mode"] == "LOCKED"' not in source
    assert '"training_eligible": False' in source


def test_archived_json_resource_cannot_be_locked_as_current_snapshot() -> None:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    package = {
        "resources": [
            {
                "id": "current-csv",
                "name": "register",
                "format": ".csv",
                "url": "https://data.gov.ua/dataset/x/resource/current-csv/download/register.csv",
                "last_modified": "2026-09-03T11:38:00",
            },
            {
                "id": "archived-json",
                "name": "Архівний - Реєстр наборів даних, які перебувають у володінні розпорядника інформації",
                "format": "JSON",
                "url": "https://data.gov.ua/dataset/x/resource/archived-json/download/register.json",
                "last_modified": "2025-01-01T00:00:00",
            },
        ]
    }

    with pytest.raises(RuntimeError, match="no admissible JSON resource candidate"):
        snapshot.pick_resource(package, cfg)
