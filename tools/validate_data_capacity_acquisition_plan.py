#!/usr/bin/env python3
"""Fail-closed validator for the 12-6 AI data-capacity acquisition plan.

This is a planning/control artifact. It never converts source bytes into proven
optimized loss positions and never authorizes long training or paid compute.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

EXPECTED_ID = "8328861c43c2e87b250715eb13dbe06318ee83e3520e8f020f3c9c4c52f99019"
EXPECTED_MODEL_PARAMS = 20_613_440
EXPECTED_MODELSPEC = "fbff24d561a2818453554d58ca23fc6ace3303b078f1935a8576c4565bd92441"
EXPECTED_DATA301 = "8820ba1b255f6bb95c7db0531fd846078a1aae01"
EXPECTED_EVAL303 = "5e5a1de3b594cee5612e63d3d4c2a70499740ac7"
EXPECTED_TERMINAL_SOURCE_HEADS = {
    "NEXT100-022-DATA-UA-WIKISOURCE": "84c51e42b6daa51796fd20d793b5ef1ff01cc9d2",
    "NEXT100-027-DATA-UA-PUBLIC-DOMAIN-LIT": "d75edd497c7fb1054e86d892c9462f059c1f4aa9",
    "NEXT100-037-DATA-EN-PYTHON-DOCS": "5a6a495a24bce449334cbc5126d0114f61a9f57c",
}
EXPECTED_RUNS = {
    "NEXT100-022-DATA-UA-WIKISOURCE": 32998002424,
    "NEXT100-027-DATA-UA-PUBLIC-DOMAIN-LIT": 32998503672,
    "NEXT100-037-DATA-EN-PYTHON-DOCS": 32998356906,
}
EXPECTED_CAPACITY = {"uk": 91_703, "en": 84_793, "code": 9_703}
TARGET_SHARE = {"uk": 0.45, "en": 0.35, "code": 0.20}


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"data-capacity plan validation failed: {message}")


def validate(path: Path) -> dict[str, object]:
    plan = json.loads(path.read_text(encoding="utf-8"))
    claimed = plan.pop("plan_identity_sha256", None)
    computed = hashlib.sha256(_canonical(plan)).hexdigest()
    _require(claimed == EXPECTED_ID, "unexpected plan identity")
    _require(computed == claimed, "self-hash drift")

    _require(plan["schema_version"] == "12-6.data-capacity-acquisition-plan.v1", "schema drift")
    _require(plan["worker_id"] == "AUTOPULSE-DATA-CAPACITY-ACQUISITION", "worker drift")
    _require(plan["execution_profile"] == "LOCAL_FREE", "execution profile drift")

    vector = plan["authority_vector"]
    data301 = vector["data301_terminal_build"]
    _require(data301["head_sha"] == EXPECTED_DATA301, "DATA-301 head drift")
    _require(data301["status"] == "TERMINAL_BLOCKED", "DATA-301 blocker hidden")
    _require(data301["corpus_identity"] is None, "fabricated DATA-301 corpus identity")
    _require(data301["authorized_balanced_no_replay_capacity"] == 0, "fabricated DATA-301 capacity")

    observed_sources = vector["terminal_additive_sources"]
    by_worker = {entry["worker"]: entry for entry in observed_sources}
    _require(set(by_worker) == set(EXPECTED_TERMINAL_SOURCE_HEADS), "terminal source vector drift")
    for worker, head in EXPECTED_TERMINAL_SOURCE_HEADS.items():
        source = by_worker[worker]
        _require(source["head_sha"] == head, f"{worker} head drift")
        _require(source["dedicated_workflow_run"] == EXPECTED_RUNS[worker], f"{worker} run drift")
        _require(
            source["dedicated_workflow_conclusion"] == "success",
            f"{worker} not exact-head green",
        )

    python_docs = by_worker["NEXT100-037-DATA-EN-PYTHON-DOCS"]
    _require(python_docs["accepted_chunk_count"] == 14, "CPython accepted chunk count drift")
    _require(
        python_docs["conservative_training_bytes"] == 0,
        "CPython unpublished retained bytes counted",
    )

    selection = vector["selection_validation"]
    _require(selection["head_sha"] == EXPECTED_EVAL303, "EVAL-303 head drift")
    _require(selection["nonempty"] is True, "selection validation marked empty")
    _require(sum(selection["records"].values()) == 10, "selection-validation record count drift")

    capacity = plan["observed_training_source_capacity"]
    _require(
        capacity["capacity_semantics"]
        == "CONSERVATIVE_SOURCE_BYTES_ONLY_NOT_OPTIMIZED_LOSS_POSITIONS",
        "capacity semantics weakened",
    )
    by_stratum = capacity["by_stratum"]
    for stratum, expected in EXPECTED_CAPACITY.items():
        _require(
            by_stratum[stratum]["conservative_training_bytes"] == expected,
            f"{stratum} capacity drift",
        )
        _require(
            by_stratum[stratum]["independent_families"] >= 2,
            f"{stratum} family diversity regressed",
        )
    _require(
        capacity["conservative_training_bytes_total"] == sum(EXPECTED_CAPACITY.values()),
        "total conservative capacity arithmetic drift",
    )
    all_families = [family for item in by_stratum.values() for family in item["families"]]
    _require(
        len(all_families) == len(set(all_families)) == 7,
        "family identity duplication/inflation",
    )
    _require(capacity["independent_family_count"] == 7, "family count drift")
    _require(capacity["corpus_identity"] is None, "source planning mislabelled as corpus identity")
    _require(
        capacity["authorized_unique_loss_positions"] == 0,
        "source bytes mislabelled as training authority",
    )

    policy = plan["mixture_policy"]
    _require(policy["target_shares"] == TARGET_SHARE, "mixture shares drift")
    _require(
        abs(sum(policy["target_shares"].values()) - 1.0) < 1e-12,
        "mixture shares do not sum to one",
    )
    _require(policy["max_family_share_total"] == 0.25, "global family cap drift")
    _require(policy["max_family_share_within_stratum"] == 0.60, "within-stratum family cap drift")
    _require(policy["minimum_independent_families_per_stratum"] == 2, "family minimum weakened")
    for key in ("replay_allowed", "sampling_with_replacement_allowed", "padding_counts_as_data"):
        _require(policy[key] is False, f"prohibited capacity inflation enabled: {key}")

    gates = plan["scale_gates"]
    primary = gates["primary_20m"]
    _require(primary["parameter_count"] == EXPECTED_MODEL_PARAMS, "primary parameter count drift")
    _require(primary["modelspec_sha256"] == EXPECTED_MODELSPEC, "primary ModelSpec drift")
    _require(
        primary["requested_unique_optimized_positions"] == 20_000_000,
        "20M campaign budget drift",
    )
    _require(
        primary["meaningful_floor_unique_optimized_positions"] == 10_000_000,
        "20M floor drift",
    )
    _require(
        gates["next_100m"]["meaningful_unique_position_range"]
        == [50_000_000, 200_000_000],
        "100M range drift",
    )
    _require(
        gates["later_1b"]["meaningful_unique_position_range"]
        == [625_000_000, 2_500_000_000],
        "1B range drift",
    )

    gap = plan["primary_20m_acquisition_gap"]
    floor = gap["planning_source_byte_floor"]
    _require(floor == 20_000_000, "planning floor drift")
    expected_targets = {k: int(floor * v) for k, v in TARGET_SHARE.items()}
    _require(gap["target_by_stratum"] == expected_targets, "stratum target arithmetic drift")
    expected_gaps = {k: expected_targets[k] - EXPECTED_CAPACITY[k] for k in EXPECTED_CAPACITY}
    _require(gap["conservative_gap_by_stratum"] == expected_gaps, "stratum gap arithmetic drift")
    _require(
        gap["conservative_gap_total"] == floor - sum(EXPECTED_CAPACITY.values()),
        "total gap arithmetic drift",
    )
    expected_family_caps = {
        k: int(
            min(
                floor * policy["max_family_share_total"],
                expected_targets[k] * policy["max_family_share_within_stratum"],
            )
        )
        for k in EXPECTED_CAPACITY
    }
    _require(
        gap["effective_max_bytes_per_family_at_floor"] == expected_family_caps,
        "family cap arithmetic drift",
    )

    boundary = plan["truth_boundary"]
    for key in (
        "research_corpus_v1_frozen",
        "long_training_authorized",
        "training_executed",
        "paid_compute_used",
        "learned_20m_checkpoint_exists",
        "learned_100m_checkpoint_exists",
        "source_bytes_equal_loss_positions_claimed",
    ):
        _require(boundary[key] is False, f"truth boundary weakened: {key}")

    return {
        "status": "PASS",
        "plan_identity_sha256": claimed,
        "conservative_training_bytes_total": capacity["conservative_training_bytes_total"],
        "conservative_20m_source_byte_gap": gap["conservative_gap_total"],
        "independent_family_count": capacity["independent_family_count"],
        "long_training_authorized": False,
        "execution_profile": "LOCAL_FREE",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        nargs="?",
        default="configs/data/data_capacity_acquisition_plan_v1.json",
    )
    args = parser.parse_args()
    print(json.dumps(validate(Path(args.path)), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
