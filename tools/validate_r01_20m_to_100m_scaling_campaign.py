#!/usr/bin/env python3
"""Validate the fail-closed R01 20M -> 100M scaling campaign contract."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

EXPECTED_AUTHORITY = {
    "main_sha_at_claim": "23b258d8599aa2c5381b735fdb58a6d0b4a8deb8",
    "model341_branch": "model341/20m-candidate-a-20260826",
    "model341_sha": "e4ff486fd90802fc123bebf60eed4e59196a98df",
    "modelspec_sha256": "fbff24d561a2818453554d58ca23fc6ace3303b078f1935a8576c4565bd92441",
    "parameter_count": 20613440,
    "canonical_base": "random_init",
}

EXPECTED_BASELINE = {
    "vocab_size": 256,
    "max_seq_len": 1024,
    "d_model": 320,
    "n_layers": 16,
    "n_heads": 10,
    "n_kv_heads": 2,
    "head_dim": 32,
    "d_ff": 1080,
    "activation": "swiglu",
    "norm": "rmsnorm_pre",
    "position": "rope",
    "tie_word_embeddings": True,
}

REQUIRED_PROMOTION_GATES = {
    "exact_corpus_identity",
    "reserved_evaluation_decontamination",
    "quality_privacy_dedup_split",
    "unique_post_pack_loss_ledger",
    "trained_tokenizer_identity_or_explicit_byte_baseline_decision",
    "checkpoint_integrity_terminal_retest",
    "selection_validation_terminal",
    "exact_code_model_tokenizer_data_config_hashes",
    "bounded_pilot_loss_and_numerics",
    "compute_authorization_for_material_cost",
    "independent_audit",
}

REQUIRED_LONG_TRAINING_GATES = {
    "requires_corpus_identity",
    "requires_tokenizer_fit_identity",
    "requires_unique_post_pack_loss_ledger",
    "requires_checkpoint_integrity_terminal_retest",
    "requires_selection_validation_terminal",
    "requires_compute_authorization_if_material_cost",
}

REQUIRED_METRICS = {
    "bits_per_byte",
    "heldout_loss",
    "loss_curve",
    "gradient_health",
    "tokens_per_second",
    "peak_memory_bytes",
    "checkpoint_resume_equivalence",
    "deterministic_rebuild_or_seed_evidence",
}

EXPECTED_SOURCE_URLS = {
    "https://arxiv.org/abs/2203.15556",
    "https://arxiv.org/abs/2402.14905",
    "https://arxiv.org/abs/2502.02737",
}


def _expect(errors: list[str], condition: bool, message: str) -> None:
    if not condition:
        errors.append(message)


def validate_campaign(data: dict[str, Any]) -> list[str]:
    errors: list[str] = []

    _expect(errors, data.get("schema_version") == 1, "schema_version must be 1")
    _expect(
        errors,
        data.get("campaign_id") == "R01-20M-TO-100M-SCALING-V1",
        "campaign_id mismatch",
    )
    _expect(
        errors,
        data.get("status") == "CANDIDATE_PLANNING_ONLY",
        "campaign must remain planning-only",
    )

    authority = data.get("authority")
    _expect(errors, isinstance(authority, dict), "authority must be an object")
    if isinstance(authority, dict):
        for key, value in EXPECTED_AUTHORITY.items():
            _expect(errors, authority.get(key) == value, f"authority.{key} mismatch")

    baseline = data.get("baseline_model")
    _expect(errors, isinstance(baseline, dict), "baseline_model must be an object")
    if isinstance(baseline, dict):
        for key, value in EXPECTED_BASELINE.items():
            _expect(errors, baseline.get(key) == value, f"baseline_model.{key} mismatch")

    boundaries = data.get("hard_boundaries")
    _expect(errors, isinstance(boundaries, dict), "hard_boundaries must be an object")
    if isinstance(boundaries, dict):
        _expect(errors, boundaries.get("base_lineage") == "PRETRAINING_ONLY", "Base lineage drift")
        for key in (
            "foreign_pretrained_weights_allowed",
            "alignment_or_personality_in_base_allowed",
            "paid_compute_authorized",
            "long_training_authorized",
            "stage_promotion_authorized",
            "corpus_mutation_in_this_package",
            "checkpoint_code_mutation_in_this_package",
        ):
            _expect(errors, boundaries.get(key) is False, f"hard_boundaries.{key} must be false")

    readiness = data.get("current_readiness")
    _expect(errors, isinstance(readiness, dict), "current_readiness must be an object")
    if isinstance(readiness, dict):
        _expect(errors, readiness.get("research_corpus_v1_identity") is None, "v1 snapshot must not fabricate corpus identity")
        _expect(errors, readiness.get("tokenizer_fit_identity") is None, "v1 snapshot must not fabricate tokenizer identity")
        _expect(
            errors,
            readiness.get("checkpoint_integrity_terminal_retest") is False,
            "v1 snapshot must not fabricate checkpoint-integrity terminality",
        )
        _expect(
            errors,
            readiness.get("selection_validation_terminal") is False,
            "v1 snapshot must not fabricate selection-validation terminality",
        )
        _expect(errors, readiness.get("compute_authorization") == "NOT_AUTHORIZED", "compute authorization drift")
        _expect(errors, readiness.get("long_training_decision") == "BLOCKED", "long training must remain blocked")

    principles = data.get("scientific_principles")
    _expect(errors, isinstance(principles, dict), "scientific_principles must be an object")
    if isinstance(principles, dict):
        for key in (
            "parameter_count_is_not_quality",
            "parameter_count_is_not_training_authorization",
            "data_quality_and_mixture_are_independent_axes",
            "tokenizer_identity_must_be_bound_before_learned_campaign",
        ):
            _expect(errors, principles.get(key) is True, f"scientific_principles.{key} must be true")
        _expect(errors, principles.get("cross_tokenizer_primary_metric") == "bits_per_byte", "cross-tokenizer primary metric must be BPB")

    matrix = data.get("experiment_matrix")
    _expect(errors, isinstance(matrix, list), "experiment_matrix must be an array")
    if isinstance(matrix, list):
        ids = [entry.get("id") for entry in matrix if isinstance(entry, dict)]
        _expect(errors, len(ids) == len(set(ids)), "experiment ids must be unique")
        _expect(errors, set(ids) == {"R01-E00", "R01-E10", "R01-E20", "R01-E30"}, "experiment matrix ids mismatch")
        for entry in matrix:
            if not isinstance(entry, dict):
                errors.append("experiment entry must be an object")
                continue
            entry_id = entry.get("id")
            if entry.get("long_training") is True:
                _expect(errors, entry.get("authorized_now") is False, f"{entry_id} long training cannot be authorized now")
                for gate in REQUIRED_LONG_TRAINING_GATES:
                    _expect(
                        errors,
                        entry.get(gate) is True,
                        f"{entry_id}.{gate} must remain true for long training",
                    )
            if entry_id == "R01-E00":
                _expect(errors, entry.get("parameters") == 20613440, "R01-E00 must bind exact MODEL-341 parameter count")
                _expect(errors, entry.get("authorized_now") is True, "R01-E00 local mechanics control should remain executable")
            if entry_id == "R01-E10":
                _expect(errors, entry.get("tokenizer_candidate_vocab_sizes") == [320, 384, 437, 512], "R01-E10 tokenizer grid drift")
                _expect(errors, entry.get("requires_corpus_identity") is True, "R01-E10 must remain corpus-bound")
            if entry_id in {"R01-E20", "R01-E30"}:
                _expect(errors, entry.get("planned_tokens_per_parameter") == [10, 20, 40], f"{entry_id} token sweep drift")
        e30 = next((entry for entry in matrix if isinstance(entry, dict) and entry.get("id") == "R01-E30"), {})
        _expect(errors, e30.get("parameter_targets") == [20000000, 50000000, 100000000], "R01-E30 target ladder drift")
        _expect(errors, e30.get("freeze_100m_modelspec_now") is False, "100M ModelSpec must not be frozen before measured evidence")
        _expect(errors, e30.get("requires_20m_learned_evidence") is True, "R01-E30 must require terminal learned-20M evidence")

    gates = data.get("promotion_gates")
    _expect(errors, isinstance(gates, list), "promotion_gates must be an array")
    if isinstance(gates, list):
        _expect(errors, REQUIRED_PROMOTION_GATES.issubset(set(gates)), "promotion gate set is incomplete")

    metrics = data.get("metric_contract")
    _expect(errors, isinstance(metrics, dict), "metric_contract must be an object")
    if isinstance(metrics, dict):
        required = metrics.get("required")
        _expect(errors, isinstance(required, list), "metric_contract.required must be an array")
        if isinstance(required, list):
            _expect(errors, REQUIRED_METRICS.issubset(set(required)), "required metric set is incomplete")
        _expect(
            errors,
            metrics.get("tokenizer_comparison_rule")
            == "do_not_compare_token_level_perplexity_across_different_tokenizer_identities_as_primary_quality_evidence",
            "tokenizer comparison firewall drift",
        )

    sources = data.get("research_sources")
    _expect(errors, isinstance(sources, list), "research_sources must be an array")
    if isinstance(sources, list):
        urls = {item.get("url") for item in sources if isinstance(item, dict)}
        _expect(errors, EXPECTED_SOURCE_URLS.issubset(urls), "required research source set is incomplete")

    return errors


def validate_path(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return ["campaign root must be an object"]
    return validate_campaign(data)


def main(argv: list[str]) -> int:
    path = Path(argv[1]) if len(argv) > 1 else Path(
        "configs/research/r01_20m_to_100m_scaling_campaign_v1.json"
    )
    errors = validate_path(path)
    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        return 1
    print("PASS: R01 20M -> 100M scaling campaign is internally consistent and fail-closed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
