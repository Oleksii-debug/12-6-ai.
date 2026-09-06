from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from twelve_six.data.bulk_acquisition_rebind_v2 import (
    BulkAcquisitionRebindError,
    _git_blob_sha1,
    load_and_validate,
    validate_rebind,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/data/research_corpus_v1_bulk_rebind_v2.json"
PARENT_PATH = ROOT / "configs/data/next100_063_terminal_source_registry_v4.json"


def _inputs() -> tuple[dict[str, object], dict[str, object], str]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    parent_payload = PARENT_PATH.read_bytes()
    parent = json.loads(parent_payload.decode("utf-8"))
    return config, parent, _git_blob_sha1(parent_payload)


def test_live_v4_rebind_contract_passes() -> None:
    report = load_and_validate(CONFIG_PATH, ROOT)
    assert report["status"] == "PASS_PLANNING_REBIND_ONLY"
    assert report["credited_pre_successor_global_dedup_bytes"] == {
        "uk": 100856,
        "en": 1838293,
        "code": 106031,
        "total": 2045180,
    }
    assert report["remaining_gap_bytes"]["total"] == 17954820
    assert report["buffered_gross_required_bytes"]["total"] == 29924701
    assert report["acquisition_priority"] == ["uk", "code", "en"]
    assert report["training_authorized"] is False


def test_parent_blob_rebinding_fails_closed() -> None:
    config, parent, blob = _inputs()
    assert blob == config["parent_authority"]["config_blob_sha1"]
    with pytest.raises(BulkAcquisitionRebindError, match="parent config blob drift"):
        validate_rebind(config, parent, "0" * 40)


def test_parent_capacity_movement_invalidates_planner() -> None:
    config, parent, blob = _inputs()
    parent = copy.deepcopy(parent)
    parent["pre_successor_global_dedup_inventory"]["by_stratum"]["uk"][
        "numeric_training_capacity_bytes"
    ] += 1
    parent["pre_successor_global_dedup_inventory"]["candidate_numeric_training_capacity_bytes"] += 1
    with pytest.raises(BulkAcquisitionRebindError, match="planner credited vector drifted"):
        validate_rebind(config, parent, blob)


def test_full_cpython_envelope_cannot_be_silently_credited() -> None:
    config, parent, blob = _inputs()
    config = copy.deepcopy(config)
    extra = 17901 - 15540
    config["credited_pre_successor_global_dedup_bytes"]["en"] += extra
    config["credited_pre_successor_global_dedup_bytes"]["total"] += extra
    with pytest.raises(BulkAcquisitionRebindError, match="planner credited vector drifted"):
        validate_rebind(config, parent, blob)


def test_stale_old_bulk_gap_is_rejected() -> None:
    config, parent, blob = _inputs()
    config = copy.deepcopy(config)
    config["remaining_gap_bytes"] = {
        "uk": 8899144,
        "en": 6849357,
        "code": 3930867,
        "total": 19679368,
    }
    with pytest.raises(BulkAcquisitionRebindError, match="remaining acquisition gap arithmetic drift"):
        validate_rebind(config, parent, blob)


def test_gutenberg_volume_cannot_waive_family_caps() -> None:
    config, parent, blob = _inputs()
    config = copy.deepcopy(config)
    config["family_cap_policy"][
        "gutenberg_requires_downselection_or_en_family_diversification"
    ] = False
    with pytest.raises(BulkAcquisitionRebindError, match="Gutenberg cap mitigation requirement weakened"):
        validate_rebind(config, parent, blob)


def test_training_authorization_promotion_fails_closed() -> None:
    config, parent, blob = _inputs()
    config = copy.deepcopy(config)
    config["claim_boundary"]["model_training_authorized"] = True
    with pytest.raises(BulkAcquisitionRebindError, match="claim boundary weakened"):
        validate_rebind(config, parent, blob)


def test_underbuffered_code_plan_is_rejected() -> None:
    config, parent, blob = _inputs()
    config = copy.deepcopy(config)
    config["planned_gross_bytes"]["code"] = 6000000
    config["planned_gross_bytes"]["total"] = (
        config["planned_gross_bytes"]["uk"]
        + config["planned_gross_bytes"]["en"]
        + config["planned_gross_bytes"]["code"]
    )
    config["planning_headroom_bytes"]["code"] = -489949
    config["planning_headroom_bytes"]["total"] = 75101
    with pytest.raises(BulkAcquisitionRebindError, match="code: planned gross below buffered requirement"):
        validate_rebind(config, parent, blob)
