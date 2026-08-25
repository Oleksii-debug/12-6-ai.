from __future__ import annotations

import copy
from pathlib import Path

from twelve_six.mixture_experiment import (
    build_plan,
    load_config,
    schedule_preview,
    selection_decision,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/experiments/data34_mixture_268k_v1.json"


def _run(mixture_id: str, uk: float, en: float, code: float, test_macro: float = 99.0):
    by_modality = {
        "uk": {"bpb": uk},
        "en": {"bpb": en},
        "code": {"bpb": code},
    }
    return {
        "mixture_id": mixture_id,
        "selection_validation": {
            "by_modality": by_modality,
            "macro_bpb": (uk + en + code) / 3.0,
        },
        "final_test": {"macro_bpb": test_macro},
    }


def test_config_prespecifies_four_fixed_control_mixtures() -> None:
    config = load_config(CONFIG_PATH)
    assert config["controls"]["expected_trainable_parameters"] == 267_912
    assert config["controls"]["requested_loss_tokens"] == 131_072
    assert config["controls"]["tokenizer"] == "s0-byte-v1"
    assert len(config["mixtures"]) == 4
    assert {item["id"] for item in config["mixtures"]} == {
        "incumbent_45_35_20",
        "uk_heavy_60_25_15",
        "balanced_40_40_20",
        "code_heavy_40_30_30",
    }
    for item in config["mixtures"]:
        assert sum(item["weights"].values()) == 100


def test_all_mixtures_have_identical_optimized_token_budget() -> None:
    config = load_config(CONFIG_PATH)
    manifests = {
        "uk": "1" * 64,
        "en": "2" * 64,
        "code": "3" * 64,
    }
    previews = []
    for mixture in config["mixtures"]:
        plan = build_plan(mixture, manifests, config=config)
        first = schedule_preview(
            plan,
            requested_loss_tokens=config["controls"]["requested_loss_tokens"],
            batch_size=config["controls"]["batch_size"],
            sequence_length=config["controls"]["sequence_length"],
        )
        second = schedule_preview(
            plan,
            requested_loss_tokens=config["controls"]["requested_loss_tokens"],
            batch_size=config["controls"]["batch_size"],
            sequence_length=config["controls"]["sequence_length"],
        )
        assert first == second
        assert sum(first["loss_tokens_by_modality"].values()) == first["actual_loss_tokens"]
        previews.append(first)
    assert {preview["actual_loss_tokens"] for preview in previews} == {131_292}
    assert {preview["optimizer_steps"] for preview in previews} == {521}


def test_selection_uses_validation_only_and_respects_regression_guard() -> None:
    config = load_config(CONFIG_PATH)
    selection = config["selection"]
    runs = [
        _run("incumbent_45_35_20", 3.0, 3.0, 3.0, test_macro=1.0),
        _run("uk_heavy_60_25_15", 2.7, 3.3, 3.0, test_macro=0.1),
        _run("balanced_40_40_20", 2.92, 2.92, 2.92, test_macro=50.0),
        _run("code_heavy_40_30_30", 3.1, 3.1, 2.5, test_macro=0.01),
    ]
    decision = selection_decision(runs, selection)
    assert decision["winner_id"] == "balanced_40_40_20"
    assert decision["test_used_for_selection"] is False
    assert decision["comparisons"]["uk_heavy_60_25_15"][
        "severe_regression_modalities"
    ] == ["en"]

    mutated = copy.deepcopy(runs)
    for index, run in enumerate(mutated):
        run["final_test"]["macro_bpb"] = float(index) * 1000.0
    assert selection_decision(mutated, selection)["winner_id"] == decision["winner_id"]


def test_minimum_gain_keeps_incumbent_for_noise_sized_change() -> None:
    config = load_config(CONFIG_PATH)
    selection = config["selection"]
    runs = [
        _run("incumbent_45_35_20", 3.0, 3.0, 3.0),
        _run("uk_heavy_60_25_15", 2.999, 2.999, 2.999),
        _run("balanced_40_40_20", 3.01, 2.99, 3.0),
        _run("code_heavy_40_30_30", 3.0, 3.0, 3.01),
    ]
    decision = selection_decision(runs, selection)
    assert decision["best_guard_passing_id"] == "uk_heavy_60_25_15"
    assert decision["winner_id"] == "incumbent_45_35_20"
    assert decision["retained_incumbent_for_minimum_gain"] is True
