from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from twelve_six.data.bulk_acquisition_rebind_v3 import (
    BulkAcquisitionRebindV3Error,
    _git_blob_sha1,
    load_and_validate,
    validate_rebind,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/data/research_corpus_v1_bulk_rebind_v3.json"
PARENT_PATH = ROOT / "configs/data/next100_063_terminal_source_registry_v5.json"
BASE_V4_PATH = ROOT / "configs/data/next100_063_terminal_source_registry_v4.json"


def _inputs() -> tuple[dict[str, object], dict[str, object], str, dict[str, object], str]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    parent_payload = PARENT_PATH.read_bytes()
    base_payload = BASE_V4_PATH.read_bytes()
    return config, json.loads(parent_payload.decode("utf-8")), _git_blob_sha1(parent_payload), json.loads(base_payload.decode("utf-8")), _git_blob_sha1(base_payload)


def test_live_v5_dedup_rebind_contract_passes() -> None:
    report = load_and_validate(CONFIG_PATH, ROOT)
    assert report["status"] == "PASS_V5_DEDUP_PLANNING_REBIND_ONLY"
    assert report["credited_post_successor_global_dedup_bytes"] == {"uk": 100856, "en": 1838293, "code": 276466, "total": 2215615}
    assert report["remaining_gap_bytes"]["total"] == 17784385
    assert report["buffered_gross_required_bytes"]["total"] == 29640642
    assert report["acquisition_priority"] == ["uk", "code", "en"]
    assert report["corpus_materialized"] is False
    assert report["authorized_unique_loss_positions"] == 0
    assert report["training_authorized"] is False


def test_parent_blob_rebinding_fails_closed() -> None:
    config, parent, _blob, base, base_blob = _inputs()
    with pytest.raises(BulkAcquisitionRebindV3Error, match="parent config blob drift"):
        validate_rebind(config, parent, "0" * 40, base, base_blob)


def test_v5_attrs_capacity_cannot_be_dropped() -> None:
    config, parent, blob, base, base_blob = _inputs()
    config = copy.deepcopy(config)
    config["credited_post_successor_global_dedup_bytes"]["code"] -= 170435
    config["credited_post_successor_global_dedup_bytes"]["total"] -= 170435
    with pytest.raises(BulkAcquisitionRebindV3Error, match="planner credited vector drifted"):
        validate_rebind(config, parent, blob, base, base_blob)


def test_dedup_artifact_identity_is_fail_closed() -> None:
    config, parent, blob, base, base_blob = _inputs()
    config = copy.deepcopy(config)
    config["observed_successor_global_dedup"]["artifact_digest"] = "sha256:" + "0" * 64
    with pytest.raises(BulkAcquisitionRebindV3Error, match="dedup artifact digest drift"):
        validate_rebind(config, parent, blob, base, base_blob)


def test_dedup_cannot_masquerade_as_materialized_corpus() -> None:
    config, parent, blob, base, base_blob = _inputs()
    config = copy.deepcopy(config)
    config["observed_successor_global_dedup"]["corpus_materialized"] = True
    with pytest.raises(BulkAcquisitionRebindV3Error, match="cannot masquerade"):
        validate_rebind(config, parent, blob, base, base_blob)


def test_claim_boundary_cannot_authorize_training() -> None:
    config, parent, blob, base, base_blob = _inputs()
    config = copy.deepcopy(config)
    config["claim_boundary"]["model_training_authorized"] = True
    with pytest.raises(BulkAcquisitionRebindV3Error, match="claim boundary weakened"):
        validate_rebind(config, parent, blob, base, base_blob)


def test_remaining_gap_is_derived_from_unique_capacity() -> None:
    config, parent, blob, base, base_blob = _inputs()
    config = copy.deepcopy(config)
    config["remaining_gap_bytes"]["code"] += 1
    config["remaining_gap_bytes"]["total"] += 1
    with pytest.raises(BulkAcquisitionRebindV3Error, match="remaining acquisition gap arithmetic drift"):
        validate_rebind(config, parent, blob, base, base_blob)


def test_underbuffered_code_plan_is_rejected() -> None:
    config, parent, blob, base, base_blob = _inputs()
    config = copy.deepcopy(config)
    config["planned_gross_bytes"]["code"] = 6000000
    config["planned_gross_bytes"]["total"] = 30000000
    with pytest.raises(BulkAcquisitionRebindV3Error, match="code: planned gross below buffered requirement"):
        validate_rebind(config, parent, blob, base, base_blob)


def test_base_v4_identity_is_fail_closed() -> None:
    config, parent, blob, base, _base_blob = _inputs()
    with pytest.raises(BulkAcquisitionRebindV3Error, match="base V4 checkout blob drift"):
        validate_rebind(config, parent, blob, base, "0" * 40)
