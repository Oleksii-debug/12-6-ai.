#!/usr/bin/env python3
"""Plan Research Corpus V1 acquisition and separate 12-6 training-budget semantics.

This tool is deliberately fail-closed. It does not authorize training and it never
converts source bytes into tokenizer tokens or optimized causal loss positions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path
from typing import Any

EXPECTED_SCHEMA = "12-6.research-corpus-pretraining-gate.v1"
STRATA = ("uk_text", "en_text", "code")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def _ceil_fraction(value: Fraction) -> int:
    return (value.numerator + value.denominator - 1) // value.denominator


def validate_config(config: dict[str, Any]) -> None:
    _require(config.get("schema_version") == EXPECTED_SCHEMA, "unexpected schema_version")
    _require(
        config.get("authority_boundary")
        == "PLANNING_AND_FAIL_CLOSED_GATING_NOT_TRAINING_AUTHORIZATION",
        "authority boundary must remain fail-closed",
    )

    state = config["project_state"]
    _require(state["primary_20m_parameter_count"] > 0, "parameter count must be positive")
    _require(
        state["live_readiness_decision"]
        == "BLOCK_LONG_TRAINING_CONTINUE_LOCAL_FREE_ENGINEERING",
        "this snapshot must not silently authorize long training",
    )
    _require(state["terminal_corpus_identity"] is None, "unexpected terminal corpus identity")
    _require(state["terminal_shard_identity"] is None, "unexpected terminal shard identity")
    _require(
        state["authorized_balanced_no_replay_loss_positions"] == 0,
        "unexpected positive training authorization",
    )

    corpus = config["research_corpus_v1_minimum_source_capacity"]
    target = corpus["target_unique_normalized_bytes"]
    _require(isinstance(target, int) and not isinstance(target, bool) and target > 0, "bad target")
    mixture = corpus["mixture"]
    _require(set(mixture) == set(STRATA), "mixture strata mismatch")
    weights = [Fraction(str(mixture[name])) for name in STRATA]
    _require(sum(weights) == 1, "mixture weights must sum exactly to 1")
    for weight in weights:
        _require(weight > 0, "mixture weights must be positive")
    _require(corpus["replay_or_duplicate_capacity_forbidden"] is True, "replay must stay forbidden")

    minimum_families = corpus["minimum_independent_families_per_stratum"]
    _require(isinstance(minimum_families, int) and minimum_families >= 2, "bad family minimum")
    snapshot = corpus["current_planning_snapshot"]["strata"]
    _require(set(snapshot) == set(STRATA), "snapshot strata mismatch")
    for name in STRATA:
        row = snapshot[name]
        _require(row["unique_normalized_bytes"] >= 0, f"negative bytes for {name}")
        _require(row["independent_families"] >= 0, f"negative families for {name}")

    modes = config["training_modes"]
    _require(set(modes) == {"MECHANICS_PILOT", "RESEARCH_CAMPAIGN", "QUALITY_PRETRAIN_REFERENCE"}, "training mode drift")
    _require(modes["MECHANICS_PILOT"]["quality_claim_allowed"] is False, "mechanics cannot claim quality")
    _require(modes["RESEARCH_CAMPAIGN"]["terminal_corpus_required"] is True, "research requires corpus")
    _require(modes["QUALITY_PRETRAIN_REFERENCE"]["terminal_corpus_required"] is True, "quality reference requires corpus")
    for mode in modes.values():
        _require(mode["material_paid_compute_allowed"] is False, "paid compute must remain unauthorized")

    ref = modes["QUALITY_PRETRAIN_REFERENCE"]["reference"]
    _require(ref["parameters"] > 0 and ref["training_tokens"] > 0, "bad external reference")
    _require(ref["source_url"].startswith("https://"), "reference source must be HTTPS")
    observation = config["modern_overtraining_observation"]
    _require(observation["nominal_parameters"] > 0, "bad modern observation parameters")
    _require(observation["pretraining_tokens"] > 0, "bad modern observation tokens")


def build_report(config: dict[str, Any]) -> dict[str, Any]:
    validate_config(config)

    corpus = config["research_corpus_v1_minimum_source_capacity"]
    target_total = corpus["target_unique_normalized_bytes"]
    minimum_families = corpus["minimum_independent_families_per_stratum"]
    snapshot = corpus["current_planning_snapshot"]["strata"]

    acquisition: dict[str, Any] = {}
    family_blockers: list[str] = []
    max_balanced_candidates: list[int] = []

    for name in STRATA:
        weight = Fraction(str(corpus["mixture"][name]))
        target_fraction = target_total * weight
        _require(target_fraction.denominator == 1, f"non-integer byte target for {name}")
        target_bytes = target_fraction.numerator
        current_bytes = snapshot[name]["unique_normalized_bytes"]
        families = snapshot[name]["independent_families"]
        family_gap = max(minimum_families - families, 0)
        if family_gap:
            family_blockers.append(name)
        max_balanced_candidates.append(current_bytes * weight.denominator // weight.numerator)
        acquisition[name] = {
            "target_unique_normalized_bytes": target_bytes,
            "current_dedup_certified_unique_normalized_bytes": current_bytes,
            "byte_gap": max(target_bytes - current_bytes, 0),
            "independent_families": families,
            "minimum_independent_families": minimum_families,
            "family_gap": family_gap,
        }

    family_gate_pass = not family_blockers
    feasible_balanced_bytes = min(max_balanced_candidates) if family_gate_pass else 0

    modes = config["training_modes"]
    ref = modes["QUALITY_PRETRAIN_REFERENCE"]["reference"]
    ratio = Fraction(ref["training_tokens"], ref["parameters"])
    quality_reference = []
    for parameter_count in modes["QUALITY_PRETRAIN_REFERENCE"]["future_parameter_targets"]:
        token_target = _ceil_fraction(ratio * parameter_count)
        quality_reference.append(
            {
                "parameter_count": parameter_count,
                "reference_training_tokens": token_target,
                "reference_tokens_per_parameter": float(ratio),
            }
        )

    project_state = config["project_state"]
    research = modes["RESEARCH_CAMPAIGN"]
    research_ready = bool(
        project_state["terminal_corpus_identity"]
        and project_state["terminal_shard_identity"]
        and project_state["authorized_balanced_no_replay_loss_positions"]
        >= research["primary_20m_meaningful_floor_positions"]
    )

    report = {
        "schema_version": "12-6.research-corpus-pretraining-report.v1",
        "training_authorized": False,
        "long_training_decision": "BLOCK",
        "research_corpus_v1": {
            "target_unique_normalized_bytes": target_total,
            "current_dedup_certified_unique_normalized_bytes": sum(
                snapshot[name]["unique_normalized_bytes"] for name in STRATA
            ),
            "total_byte_gap": sum(row["byte_gap"] for row in acquisition.values()),
            "strata": acquisition,
            "family_gate_pass": family_gate_pass,
            "family_blockers": family_blockers,
            "current_feasible_fixed_mixture_bytes": feasible_balanced_bytes,
            "terminal_corpus_identity_present": project_state["terminal_corpus_identity"] is not None,
            "terminal_shard_identity_present": project_state["terminal_shard_identity"] is not None,
        },
        "training_modes": {
            "MECHANICS_PILOT": {
                "may_continue_local_free": True,
                "quality_claim_allowed": False,
            },
            "RESEARCH_CAMPAIGN": {
                "ready": research_ready,
                "preregistered_positions": research["primary_20m_preregistered_positions"],
                "meaningful_floor_positions": research["primary_20m_meaningful_floor_positions"],
                "planning_ceiling_positions": research["primary_20m_planning_ceiling_positions"],
                "authorized_positions_now": project_state[
                    "authorized_balanced_no_replay_loss_positions"
                ],
            },
            "QUALITY_PRETRAIN_REFERENCE": {
                "hard_gate": False,
                "external_reference_name": ref["name"],
                "derived_targets": quality_reference,
            },
        },
        "unit_separation": {
            "source_bytes_are_tokens": False,
            "tokens_are_optimized_loss_positions": False,
            "parameter_count_is_data_budget": False,
            "epochs_can_create_unique_capacity": False,
        },
        "next_critical_path": [
            "converge terminal training-admitted source authorities into one successor source registry",
            "close the remaining independent-family and source-volume gaps without replay",
            "freeze exact pre-decontamination record inventory and corpus candidate identity",
            "run quality privacy global dedup and evaluation decontamination on that exact inventory",
            "materialize cluster-safe split tokenized packing and exact post-pack unique-loss ledger",
            "prove two clean byte-identical builds before requesting material compute authorization",
        ],
    }
    report["report_sha256"] = hashlib.sha256(_canonical_json(report)).hexdigest()
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/scaling/research_corpus_pretraining_gate_v1.json"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    report = build_report(config)
    rendered = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
