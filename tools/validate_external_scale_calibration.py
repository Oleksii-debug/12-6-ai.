#!/usr/bin/env python3
"""Validate the fail-closed external scale calibration contract."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

DEFAULT_PATH = Path("configs/research/external_scale_calibration_v1.json")
EXPECTED_SCHEMA = "12-6.external-scale-calibration.v1"


class CalibrationError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CalibrationError(message)


def validate(data: dict[str, Any]) -> None:
    _require(data.get("schema_version") == EXPECTED_SCHEMA, "unexpected schema")
    _require(data.get("execution_profile") == "LOCAL_FREE", "must remain LOCAL_FREE")
    _require(data.get("paid_compute_used") is False, "paid compute must remain false")
    _require(data.get("training_executed") is False, "this contract must not execute training")
    _require(
        data.get("decision") == "DATA_FIRST_DO_NOT_LAUNCH_LONG_20M_TRAINING",
        "fail-closed decision changed",
    )

    authority = data["current_authority"]
    _require(authority["primary_20m_parameters"] == 20_613_440, "MODEL-341 parameter drift")
    _require(
        authority["primary_20m_modelspec_sha256"]
        == "fbff24d561a2818453554d58ca23fc6ace3303b078f1935a8576c4565bd92441",
        "MODEL-341 ModelSpec drift",
    )
    _require(
        authority["authorized_current_external_unique_optimized_targets"] == 0,
        "current external-real capacity must stay fail-closed until refreshed authority exists",
    )
    _require(authority["long_training_allowed_now"] is False, "long training cannot be enabled here")

    bands = data["project_planning_bands"]
    meaningful = bands["meaningful_science_unique_targets_per_parameter"]
    full = bands["full_pretraining_training_tokens_per_parameter"]
    _require((meaningful["minimum"], meaningful["maximum"]) == (0.5, 2.0), "meaningful band drift")
    _require((full["minimum"], full["maximum"]) == (5.0, 20.0), "full-pretraining band drift")
    _require(meaningful["universal_optimum_claimed"] is False, "meaningful band cannot be universal")
    _require(full["universal_optimum_claimed"] is False, "full band cannot be universal")

    expected_params = [20_613_440, 100_000_000, 1_000_000_000]
    targets = data["scale_targets"]
    _require([row["parameters"] for row in targets] == expected_params, "scale ladder drift")
    for row in targets:
        p = row["parameters"]
        _require(row["meaningful_unique_targets_min"] == int(p * 0.5), f"{p}: meaningful min mismatch")
        _require(row["meaningful_unique_targets_max"] == int(p * 2.0), f"{p}: meaningful max mismatch")
        _require(row["full_pretraining_tokens_min"] == int(p * 5.0), f"{p}: full min mismatch")
        _require(row["full_pretraining_tokens_max"] == int(p * 20.0), f"{p}: full max mismatch")
        _require(
            row["twenty_tokens_per_parameter_reference"] == p * 20,
            f"{p}: 20-token reference mismatch",
        )

    refs = {item["id"]: item for item in data["external_calibration"]}
    _require(
        set(refs)
        == {
            "HOFFMANN_2022_CHINCHILLA",
            "MUENNIGHOFF_2025_DATA_CONSTRAINED",
            "META_LLAMA3_2024",
            "HF_SMOLLM_2024",
        },
        "external reference set drift",
    )
    for item in refs.values():
        _require(str(item["url"]).startswith("https://"), f"{item['id']}: URL must be HTTPS")
    _require(
        refs["MUENNIGHOFF_2025_DATA_CONSTRAINED"]["repeated_tokens_count_as_unique"] is False,
        "repeated tokens must never be relabelled as unique",
    )
    for key in ("HOFFMANN_2022_CHINCHILLA", "META_LLAMA3_2024", "HF_SMOLLM_2024"):
        _require(refs[key]["universal_optimum_claimed"] is False, f"{key}: universal optimum forbidden")

    repetition = data["repetition_policy"]
    _require(repetition["default_for_authority"] == "NO_REPLAY_UNIQUE_LEDGER_FIRST", "no-replay default drift")
    _require(repetition["research_reference_epoch_ceiling"] == 4, "data-constrained reference ceiling drift")
    _require(repetition["research_reference_is_authorization"] is False, "paper result is not authorization")
    _require(repetition["repeated_tokens_may_increase_unique_capacity"] is False, "replay cannot increase unique capacity")
    _require(repetition["padding_may_increase_data_capacity"] is False, "padding cannot increase capacity")

    required = set(data["promotion_requirements"])
    for phrase in (
        "terminal immutable training corpus identity",
        "terminal unique nonignored causal-loss ledger",
        "terminal tokenizer identity bound to the corpus",
        "checkpoint corruption matrix passes on direct production path",
        "material compute receives explicit authorization before long training",
    ):
        _require(phrase in required, f"missing promotion requirement: {phrase}")


def self_test(data: dict[str, Any]) -> None:
    validate(data)
    mutations: list[tuple[str, Any]] = [
        ("enable long training", lambda d: d["current_authority"].__setitem__("long_training_allowed_now", True)),
        ("invent unique capacity", lambda d: d["current_authority"].__setitem__("authorized_current_external_unique_optimized_targets", 1)),
        ("count replay as unique", lambda d: d["repetition_policy"].__setitem__("repeated_tokens_may_increase_unique_capacity", True)),
        ("change 100M token math", lambda d: d["scale_targets"][1].__setitem__("full_pretraining_tokens_max", 123)),
    ]
    for label, mutate in mutations:
        candidate = copy.deepcopy(data)
        mutate(candidate)
        try:
            validate(candidate)
        except CalibrationError:
            continue
        raise CalibrationError(f"self-test mutation was not rejected: {label}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    data = json.loads(args.config.read_text(encoding="utf-8"))
    validate(data)
    if args.self_test:
        self_test(data)
    print("external scale calibration: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
