from __future__ import annotations

import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools" / "plan_research_corpus_pretraining.py"
CONFIG_PATH = ROOT / "configs" / "scaling" / "research_corpus_pretraining_gate_v1.json"

spec = importlib.util.spec_from_file_location("research_corpus_pretraining", TOOL_PATH)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_current_snapshot_fails_closed_with_exact_acquisition_gaps() -> None:
    report = module.build_report(load_config())

    assert report["training_authorized"] is False
    assert report["long_training_decision"] == "BLOCK"
    corpus = report["research_corpus_v1"]
    assert corpus["current_dedup_certified_unique_normalized_bytes"] == 243970
    assert corpus["total_byte_gap"] == 19756030
    assert corpus["strata"]["uk_text"]["byte_gap"] == 8909956
    assert corpus["strata"]["en_text"]["byte_gap"] == 6915207
    assert corpus["strata"]["code"]["byte_gap"] == 3930867
    assert corpus["family_gate_pass"] is False
    assert corpus["family_blockers"] == ["en_text"]
    assert corpus["current_feasible_fixed_mixture_bytes"] == 0


def test_research_campaign_is_not_ready_without_terminal_corpus_and_shards() -> None:
    report = module.build_report(load_config())
    research = report["training_modes"]["RESEARCH_CAMPAIGN"]

    assert research["ready"] is False
    assert research["authorized_positions_now"] == 0
    assert research["preregistered_positions"] == 20_000_000
    assert research["meaningful_floor_positions"] == 10_000_000


def test_chinchilla_reference_is_planning_only_and_calculated_exactly() -> None:
    report = module.build_report(load_config())
    quality = report["training_modes"]["QUALITY_PRETRAIN_REFERENCE"]

    assert quality["hard_gate"] is False
    targets = {row["parameter_count"]: row for row in quality["derived_targets"]}
    assert targets[20_613_440]["reference_training_tokens"] == 382_821_029
    assert targets[100_000_000]["reference_training_tokens"] == 1_857_142_858
    assert targets[1_000_000_000]["reference_training_tokens"] == 18_571_428_572


def test_source_bytes_never_self_promote_to_tokens_or_loss_positions() -> None:
    report = module.build_report(load_config())
    separation = report["unit_separation"]

    assert separation == {
        "source_bytes_are_tokens": False,
        "tokens_are_optimized_loss_positions": False,
        "parameter_count_is_data_budget": False,
        "epochs_can_create_unique_capacity": False,
    }


def test_en_family_gap_can_be_removed_without_fabricating_volume_readiness() -> None:
    config = load_config()
    config["research_corpus_v1_minimum_source_capacity"]["current_planning_snapshot"]["strata"][
        "en_text"
    ]["independent_families"] = 2
    report = module.build_report(config)

    corpus = report["research_corpus_v1"]
    assert corpus["family_gate_pass"] is True
    assert corpus["family_blockers"] == []
    assert corpus["current_feasible_fixed_mixture_bytes"] == 200097
    assert corpus["total_byte_gap"] == 19756030
    assert report["training_authorized"] is False


def test_paid_compute_cannot_be_enabled_by_config_drift() -> None:
    config = load_config()
    mutated = deepcopy(config)
    mutated["training_modes"]["RESEARCH_CAMPAIGN"]["material_paid_compute_allowed"] = True

    with pytest.raises(ValueError, match="paid compute"):
        module.validate_config(mutated)


def test_live_readiness_decision_cannot_silently_flip_to_start() -> None:
    config = load_config()
    config["project_state"]["live_readiness_decision"] = "START_LONG_TRAINING"

    with pytest.raises(ValueError, match="silently authorize"):
        module.validate_config(config)
