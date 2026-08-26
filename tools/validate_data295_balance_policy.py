#!/usr/bin/env python3
"""Validate the DATA-295 preregistered corpus mixture without model outcomes."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


EXPECTED_SCHEMA = "12-6.data295-balance-policy-20m.v1"
STRATA = ("uk", "en", "code")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _budget_granularity(weights: dict[str, float]) -> int:
    return 200 if any(float(v) != int(v) for v in weights.values()) else 100


def _modality_ceiling(inventory: dict[str, int], weights: dict[str, float]) -> int:
    continuous = min(inventory[key] / (float(weights[key]) / 100.0) for key in STRATA)
    granularity = _budget_granularity(weights)
    return math.floor(continuous / granularity) * granularity


def validate(policy: dict) -> None:
    assert policy["schema_version"] == EXPECTED_SCHEMA
    assert policy["execution_profile"] == "LOCAL_FREE"
    assert policy["truth_boundary"]["model_result_tuning_permitted"] is False
    assert policy["truth_boundary"]["document_duplication_permitted"] is False
    assert policy["truth_boundary"]["sampling_replay_to_fill_quota_permitted"] is False
    assert policy["truth_boundary"]["byte_counts_are_proven_optimized_loss_targets"] is False

    inventory = policy["current_cross_authority_inventory"]
    strata = inventory["strata"]
    assert inventory["unit"] == "unique_training_eligible_source_bytes"
    assert sum(strata.values()) == inventory["total"] == 183061
    assert strata == {"uk": 88565, "en": 84793, "code": 9703}
    assert sum(item["unique_bytes"] for item in inventory["families"]) == inventory["total"]

    selected = []
    for candidate in policy["candidate_policies"]:
        weights = candidate["weights_percent"]
        assert math.isclose(sum(float(weights[key]) for key in STRATA), 100.0)
        expected_ceiling = _modality_ceiling(strata, weights)
        assert candidate["current_modality_only_max_no_replay_source_byte_budget"] == expected_ceiling
        allocation = candidate["current_modality_only_allocation"]
        assert sum(allocation.values()) == expected_ceiling
        for key in STRATA:
            expected = int(round(expected_ceiling * float(weights[key]) / 100.0))
            assert allocation[key] == expected
            assert allocation[key] <= strata[key]
        target = candidate["target_20m_allocation"]
        assert sum(target.values()) == 20_000_000
        for key in STRATA:
            assert target[key] == int(20_000_000 * float(weights[key]) / 100.0)
        if candidate["selected"]:
            selected.append(candidate["policy_id"])

    assert selected == ["continuity_45_35_20"]
    chosen = policy["selected_policy"]
    assert chosen["policy_id"] == selected[0]
    assert chosen["selection_basis"] == (
        "PREEXISTING_PREREGISTERED_TOP_LEVEL_MIXTURE_CONTINUITY_NOT_MODEL_OUTCOMES"
    )
    assert chosen["target_total_source_byte_tokens"] == 20_000_000
    assert chosen["weights_percent"] == {"uk": 45, "en": 35, "code": 20}
    assert chosen["target_unique_source_byte_tokens"] == {"uk": 9_000_000, "en": 7_000_000, "code": 4_000_000}
    assert chosen["current_unique_source_byte_shortfall"] == {
        key: chosen["target_unique_source_byte_tokens"][key] - strata[key] for key in STRATA
    }

    family_policy = policy["source_family_policy"]
    assert family_policy["minimum_independent_families_per_stratum"] == {"uk": 2, "en": 2, "code": 2}
    assert family_policy["maximum_family_share_of_total_percent"] == 25
    assert family_policy["maximum_family_share_within_its_stratum_percent"] == 60
    assert family_policy["family_caps_are_hard"] is True
    assert family_policy["silent_family_overflow_permitted"] is False

    family_counts = {key: 0 for key in STRATA}
    for item in inventory["families"]:
        family_counts[item["stratum"]] += 1
    minimums = family_policy["minimum_independent_families_per_stratum"]
    family_gate_passes = all(family_counts[key] >= minimums[key] for key in STRATA)
    assert family_counts == {"uk": 1, "en": 1, "code": 2}
    assert family_gate_passes is False

    activation = policy["activation_state"]
    assert activation["selected_policy_20m_ready"] is False
    assert activation["current_full_family_constrained_no_replay_budget"] == 0
    assert activation["modality_only_source_byte_ceiling_before_family_gate"] == 48500
    assert activation["optimized_loss_target_budget"] == (
        "UNPUBLISHED_REQUIRES_EXACT_POST_SPLIT_TOKEN_POSITION_LEDGER"
    )

    authorities = policy["authority_bindings"]
    assert authorities["data229_text_registry"]["dedicated_workflow_conclusion"] == "success"
    assert authorities["data227_code_admission"]["dedicated_workflow_conclusion"] == "success"
    assert authorities["data228_diversity_probe"]["consumed_as_training_authority"] is False
    assert authorities["data230_research_corpus"]["terminal_authority_found"] is False
    assert authorities["data278_english_large_expansion"]["consumed"] is False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "policy",
        nargs="?",
        default="configs/data/data295_balance_policy_20m_v1.json",
        type=Path,
    )
    args = parser.parse_args()
    policy = _load(args.policy)
    validate(policy)
    print("DATA295_BALANCE_POLICY_VALID")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
