"""Deterministic, text-free analysis for R02 tokenizer/FLOP calibration.

The module consumes only aggregate measurements. It never fits a tokenizer, reads
corpus payloads, trains a model, or authorizes compute. Cross-tokenizer quality is
normalized to UTF-8 bytes and cross-tokenizer cost is normalized to measured FLOPs.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

CONTRACT_SCHEMA = "12-6.r02-tokenizer-flop-calibration.v1"
MEASUREMENT_SCHEMA = "12-6.r02-tokenizer-flop-measurement.v1"
REPORT_SCHEMA = "12-6.r02-tokenizer-flop-calibration-report.v1"
CAMPAIGN_ID = "R02-TOKENIZER-FLOP-CALIBRATION-V1"


class CalibrationError(ValueError):
    """Raised when calibration evidence violates the fail-closed contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CalibrationError(message)


def _round(value: float) -> float:
    return round(float(value), 12)


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CalibrationError(f"{path}: JSON root must be an object")
    return value


def _candidate_map(contract: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    candidates = contract["calibration_scope"]["required_tokenizer_candidates"]
    result = {str(item["id"]): dict(item) for item in candidates}
    _require(len(result) == len(candidates), "duplicate tokenizer candidate id")
    return result


def validate_contract(contract: Mapping[str, Any]) -> None:
    _require(contract.get("schema_version") == CONTRACT_SCHEMA, "contract schema drift")
    _require(contract.get("campaign_id") == CAMPAIGN_ID, "campaign id drift")
    _require(
        contract.get("status") == "MEASUREMENT_CONTRACT_READY_DATA_BLOCKED",
        "contract status drift",
    )

    parent = contract["parent_r01"]
    _require(parent.get("main_sha") == "a73ab38026cb7849f478cc13ad58b93534a76e2f", "R01 main binding drift")
    _require(parent.get("model341_sha") == "e4ff486fd90802fc123bebf60eed4e59196a98df", "MODEL-341 binding drift")
    _require(parent.get("baseline_total_parameter_count") == 20613440, "baseline parameter count drift")
    _require(parent.get("baseline_vocab_size") == 256, "baseline vocab drift")
    _require(parent.get("d_model") == 320, "baseline d_model drift")
    _require(parent.get("tied_embedding_parameter_count") == 81920, "baseline embedding count drift")
    _require(parent.get("nonembedding_parameter_count") == 20531520, "non-embedding count drift")
    _require(
        parent["nonembedding_parameter_count"]
        + parent["baseline_vocab_size"] * parent["d_model"]
        == parent["baseline_total_parameter_count"],
        "baseline tied-embedding arithmetic drift",
    )

    boundaries = contract["hard_boundaries"]
    _require(boundaries.get("local_free_only") is True, "LOCAL_FREE boundary weakened")
    for key in (
        "corpus_payload_read_by_contract_validator",
        "tokenizer_fit_executed",
        "model_training_executed",
        "selection_validation_consumed",
        "final_test_consumed",
        "paid_compute_authorized",
        "long_training_authorized",
        "stage_promotion_authorized",
    ):
        _require(boundaries.get(key) is False, f"hard boundary weakened: {key}")
    _require(boundaries.get("optimizer_updates") == 0, "optimizer updates must remain zero")

    truth = contract["budget_truth"]
    _require(truth.get("engineering_pilot_positions") == 20000000, "pilot position count drift")
    _require(truth.get("engineering_pilot_is_science_complete_budget") is False, "pilot promoted to science budget")
    _require(
        truth.get("science_complete_20m_budget_status")
        == "UNDEFINED_PENDING_TOKENIZER_AND_FLOP_CALIBRATION",
        "science budget truth drift",
    )
    _require(truth.get("parameter_count_is_not_compute_budget") is True, "parameter/compute boundary weakened")
    _require(truth.get("token_count_is_not_cross_tokenizer_invariant") is True, "token invariant incorrectly claimed")
    _require(truth.get("primary_cross_tokenizer_quality_metric") == "bits_per_utf8_byte", "quality metric drift")
    _require(
        truth.get("primary_cross_tokenizer_cost_metric")
        == "measured_training_flops_per_utf8_byte",
        "cost metric drift",
    )

    scope = contract["calibration_scope"]
    _require(scope.get("required_strata") == ["uk", "en", "code"], "required strata drift")
    _require(scope.get("minimum_repeats_per_candidate") == 2, "repeat floor drift")
    for key in (
        "same_exact_calibration_bytes_required_across_tokenizers",
        "same_nonembedding_parameter_count_required",
        "same_transformer_body_identity_required",
        "total_parameter_count_may_vary_only_by_tied_embedding_vocab_term",
        "same_loss_mask_semantics_required",
        "same_calibration_slice_identity_required",
        "future_measurement_requires_terminal_research_corpus_v1_identity",
        "future_subword_measurement_requires_exact_tokenizer_identity",
    ):
        _require(scope.get(key) is True, f"calibration invariant weakened: {key}")

    expected = {
        "byte-v256": ("byte", 256, 20613440),
        "subword-v320": ("subword", 320, 20633920),
        "subword-v384": ("subword", 384, 20654400),
        "subword-v437": ("subword", 437, 20671360),
        "subword-v512": ("subword", 512, 20695360),
    }
    candidates = _candidate_map(contract)
    _require(set(candidates) == set(expected), "tokenizer candidate set drift")
    for candidate_id, (kind, vocab, total_params) in expected.items():
        candidate = candidates[candidate_id]
        _require(candidate.get("kind") == kind, f"{candidate_id}: kind drift")
        _require(candidate.get("vocab_size") == vocab, f"{candidate_id}: vocab drift")
        _require(candidate.get("expected_total_parameters") == total_params, f"{candidate_id}: parameter drift")
        computed = parent["nonembedding_parameter_count"] + vocab * parent["d_model"]
        _require(computed == total_params, f"{candidate_id}: tied-embedding arithmetic drift")

    schema = contract["measurement_schema"]
    _require(schema.get("schema_version") == MEASUREMENT_SCHEMA, "measurement schema drift")
    required_top = set(schema["required_top_level_fields"])
    for field in (
        "tokenizer_id",
        "tokenizer_identity",
        "research_corpus_identity",
        "calibration_slice_identity",
        "model_total_parameter_count",
        "model_nonembedding_parameter_count",
        "model_body_identity",
        "loss_mask_identity",
        "context_window_loss_positions",
        "repeat_id",
        "peak_memory_bytes",
        "strata",
    ):
        _require(field in required_top, f"measurement field omitted: {field}")

    rules = contract["decision_rules"]
    for key in (
        "do_not_rank_cross_tokenizer_candidates_by_token_perplexity",
        "require_bits_per_utf8_byte_for_quality_ranking",
        "require_measured_not_formula_only_flops_for_final_cost_ranking",
        "require_per_stratum_results_before_weighted_aggregate",
        "require_equal_flop_projection_before_science_budget_proposal",
        "require_heldout_learning_curves_after_tokenizer_flop_calibration",
        "calibration_pass_does_not_authorize_training",
        "calibration_pass_does_not_authorize_paid_compute",
    ):
        _require(rules.get(key) is True, f"decision rule weakened: {key}")


def validate_measurement(measurement: Mapping[str, Any], contract: Mapping[str, Any]) -> None:
    validate_contract(contract)
    _require(measurement.get("schema_version") == MEASUREMENT_SCHEMA, "measurement schema mismatch")
    schema = contract["measurement_schema"]
    for field in schema["required_top_level_fields"]:
        _require(field in measurement, f"measurement missing field: {field}")

    candidates = _candidate_map(contract)
    tokenizer_id = str(measurement["tokenizer_id"])
    _require(tokenizer_id in candidates, f"unregistered tokenizer candidate: {tokenizer_id}")
    candidate = candidates[tokenizer_id]
    _require(measurement["tokenizer_kind"] == candidate["kind"], f"{tokenizer_id}: tokenizer kind mismatch")
    _require(measurement["vocab_size"] == candidate["vocab_size"], f"{tokenizer_id}: vocab mismatch")
    _require(
        measurement["model_total_parameter_count"] == candidate["expected_total_parameters"],
        f"{tokenizer_id}: total parameter count mismatch",
    )
    parent = contract["parent_r01"]
    _require(
        measurement["model_nonembedding_parameter_count"] == parent["nonembedding_parameter_count"],
        f"{tokenizer_id}: non-embedding body parameter count mismatch",
    )

    for field in (
        "tokenizer_identity",
        "research_corpus_identity",
        "calibration_slice_identity",
        "model_body_identity",
        "loss_mask_identity",
        "repeat_id",
    ):
        value = measurement[field]
        _require(isinstance(value, str) and value.strip() != "", f"{tokenizer_id}: empty {field}")
    if candidate["kind"] == "subword":
        _require(
            str(measurement["tokenizer_identity"]).lower() not in {"none", "null", "pending", "unfitted"},
            f"{tokenizer_id}: subword tokenizer identity is not terminal",
        )

    _require(
        isinstance(measurement["context_window_loss_positions"], int)
        and measurement["context_window_loss_positions"] > 0,
        f"{tokenizer_id}: invalid context window",
    )
    _require(
        isinstance(measurement["peak_memory_bytes"], int) and measurement["peak_memory_bytes"] > 0,
        f"{tokenizer_id}: invalid peak memory",
    )

    strata = measurement["strata"]
    _require(isinstance(strata, Mapping), f"{tokenizer_id}: strata must be an object")
    required_strata = contract["calibration_scope"]["required_strata"]
    _require(set(strata) == set(required_strata), f"{tokenizer_id}: stratum set mismatch")
    for stratum in required_strata:
        row = strata[stratum]
        _require(isinstance(row, Mapping), f"{tokenizer_id}/{stratum}: row must be object")
        for field in schema["required_per_stratum_fields"]:
            _require(field in row, f"{tokenizer_id}/{stratum}: missing {field}")
        for field in ("utf8_bytes", "nonignored_loss_positions"):
            _require(isinstance(row[field], int) and row[field] > 0, f"{tokenizer_id}/{stratum}: invalid {field}")
        for field in ("total_nll_nats", "measured_training_flops", "wall_seconds"):
            _require(isinstance(row[field], (int, float)) and float(row[field]) > 0.0, f"{tokenizer_id}/{stratum}: invalid {field}")


def derive_stratum_metrics(row: Mapping[str, Any], context_window_loss_positions: int) -> dict[str, float | int]:
    utf8_bytes = int(row["utf8_bytes"])
    positions = int(row["nonignored_loss_positions"])
    nll_nats = float(row["total_nll_nats"])
    flops = float(row["measured_training_flops"])
    wall_seconds = float(row["wall_seconds"])
    bytes_per_position = utf8_bytes / positions
    return {
        "utf8_bytes": utf8_bytes,
        "nonignored_loss_positions": positions,
        "bits_per_utf8_byte": _round(nll_nats / math.log(2.0) / utf8_bytes),
        "loss_positions_per_utf8_byte": _round(positions / utf8_bytes),
        "utf8_bytes_per_loss_position": _round(bytes_per_position),
        "measured_training_flops_per_utf8_byte": _round(flops / utf8_bytes),
        "measured_training_flops_per_loss_position": _round(flops / positions),
        "semantic_context_span_utf8_bytes": _round(context_window_loss_positions * bytes_per_position),
        "utf8_bytes_per_second": _round(utf8_bytes / wall_seconds),
        "loss_positions_per_second": _round(positions / wall_seconds),
    }


def _aggregate_repeats(
    tokenizer_id: str,
    repeats: list[Mapping[str, Any]],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    required_strata = contract["calibration_scope"]["required_strata"]
    first = repeats[0]
    for repeat in repeats[1:]:
        for field in (
            "tokenizer_kind",
            "vocab_size",
            "tokenizer_identity",
            "research_corpus_identity",
            "calibration_slice_identity",
            "model_total_parameter_count",
            "model_nonembedding_parameter_count",
            "model_body_identity",
            "loss_mask_identity",
            "context_window_loss_positions",
        ):
            _require(repeat[field] == first[field], f"{tokenizer_id}: repeat drift in {field}")
        for stratum in required_strata:
            for field in ("utf8_bytes", "nonignored_loss_positions"):
                _require(
                    repeat["strata"][stratum][field] == first["strata"][stratum][field],
                    f"{tokenizer_id}/{stratum}: repeat drift in {field}",
                )

    context = int(first["context_window_loss_positions"])
    strata_result: dict[str, Any] = {}
    aggregate_bytes = 0
    aggregate_positions = 0
    aggregate_nll = 0.0
    aggregate_flops = 0.0
    aggregate_wall = 0.0
    for stratum in required_strata:
        base = first["strata"][stratum]
        avg_row = {
            "utf8_bytes": int(base["utf8_bytes"]),
            "nonignored_loss_positions": int(base["nonignored_loss_positions"]),
            "total_nll_nats": sum(float(r["strata"][stratum]["total_nll_nats"]) for r in repeats) / len(repeats),
            "measured_training_flops": sum(float(r["strata"][stratum]["measured_training_flops"]) for r in repeats) / len(repeats),
            "wall_seconds": sum(float(r["strata"][stratum]["wall_seconds"]) for r in repeats) / len(repeats),
        }
        strata_result[stratum] = derive_stratum_metrics(avg_row, context)
        aggregate_bytes += int(avg_row["utf8_bytes"])
        aggregate_positions += int(avg_row["nonignored_loss_positions"])
        aggregate_nll += float(avg_row["total_nll_nats"])
        aggregate_flops += float(avg_row["measured_training_flops"])
        aggregate_wall += float(avg_row["wall_seconds"])

    aggregate_row = {
        "utf8_bytes": aggregate_bytes,
        "nonignored_loss_positions": aggregate_positions,
        "total_nll_nats": aggregate_nll,
        "measured_training_flops": aggregate_flops,
        "wall_seconds": aggregate_wall,
    }
    return {
        "tokenizer_id": tokenizer_id,
        "tokenizer_kind": first["tokenizer_kind"],
        "vocab_size": first["vocab_size"],
        "tokenizer_identity": first["tokenizer_identity"],
        "model_total_parameter_count": first["model_total_parameter_count"],
        "model_nonembedding_parameter_count": first["model_nonembedding_parameter_count"],
        "repeat_count": len(repeats),
        "repeat_ids": sorted(str(r["repeat_id"]) for r in repeats),
        "peak_memory_bytes_max": max(int(r["peak_memory_bytes"]) for r in repeats),
        "strata": strata_result,
        "aggregate": derive_stratum_metrics(aggregate_row, context),
    }


def build_report(contract: Mapping[str, Any], measurements: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    validate_contract(contract)
    materialized = [dict(item) for item in measurements]
    for measurement in materialized:
        validate_measurement(measurement, contract)

    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    repeat_ids: set[tuple[str, str]] = set()
    for measurement in materialized:
        key = (str(measurement["tokenizer_id"]), str(measurement["repeat_id"]))
        _require(key not in repeat_ids, f"duplicate measurement repeat: {key[0]}/{key[1]}")
        repeat_ids.add(key)
        groups[key[0]].append(measurement)

    candidates = _candidate_map(contract)
    minimum_repeats = int(contract["calibration_scope"]["minimum_repeats_per_candidate"])
    missing = sorted(set(candidates) - set(groups))
    short = {
        tokenizer_id: len(groups[tokenizer_id])
        for tokenizer_id in sorted(groups)
        if len(groups[tokenizer_id]) < minimum_repeats
    }

    common_fields = (
        "research_corpus_identity",
        "calibration_slice_identity",
        "model_body_identity",
        "loss_mask_identity",
        "context_window_loss_positions",
        "model_nonembedding_parameter_count",
    )
    if materialized:
        first = materialized[0]
        for measurement in materialized[1:]:
            for field in common_fields:
                _require(measurement[field] == first[field], f"cross-tokenizer drift in {field}")

    aggregates: dict[str, dict[str, Any]] = {}
    for tokenizer_id in sorted(groups):
        aggregates[tokenizer_id] = _aggregate_repeats(tokenizer_id, groups[tokenizer_id], contract)

    if len(aggregates) > 1:
        required_strata = contract["calibration_scope"]["required_strata"]
        first_id = sorted(aggregates)[0]
        for tokenizer_id in sorted(aggregates):
            for stratum in required_strata:
                _require(
                    aggregates[tokenizer_id]["strata"][stratum]["utf8_bytes"]
                    == aggregates[first_id]["strata"][stratum]["utf8_bytes"],
                    f"cross-tokenizer calibration bytes differ in {stratum}",
                )

    complete = not missing and not short
    projections: dict[str, Any] = {}
    if "byte-v256" in aggregates:
        baseline = aggregates["byte-v256"]["aggregate"]
        pilot_positions = int(contract["budget_truth"]["engineering_pilot_positions"])
        reference_flops = pilot_positions * float(baseline["measured_training_flops_per_loss_position"])
        for tokenizer_id, aggregate in aggregates.items():
            metrics = aggregate["aggregate"]
            projections[tokenizer_id] = {
                "reference_byte_pilot_loss_positions": pilot_positions,
                "reference_equal_flop_budget": _round(reference_flops),
                "equal_flop_projected_loss_positions": _round(
                    reference_flops / float(metrics["measured_training_flops_per_loss_position"])
                ),
                "equal_flop_projected_utf8_bytes": _round(
                    reference_flops / float(metrics["measured_training_flops_per_utf8_byte"])
                ),
            }

    calibration_status = (
        "TOKENIZER_FLOP_CALIBRATION_READY_REQUIRES_LEARNING_CURVES"
        if complete
        else "INCOMPLETE_MEASUREMENTS"
    )
    science_budget_status = (
        "UNDEFINED_PENDING_HELDOUT_LEARNING_CURVES_AND_COMPUTE_AUTHORIZATION"
        if complete
        else contract["budget_truth"]["science_complete_20m_budget_status"]
    )

    core: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "status": calibration_status,
        "contract_identity_sha256": _sha256(contract),
        "measurement_count": len(materialized),
        "coverage": {
            "required_candidate_ids": sorted(candidates),
            "missing_candidate_ids": missing,
            "repeat_counts": {tokenizer_id: len(groups.get(tokenizer_id, [])) for tokenizer_id in sorted(candidates)},
            "below_minimum_repeat_counts": short,
            "minimum_repeats_per_candidate": minimum_repeats,
        },
        "common_evidence": (
            {
                field: materialized[0][field]
                for field in common_fields
            }
            if materialized
            else None
        ),
        "tokenizers": aggregates,
        "equal_flop_projection": projections,
        "science_complete_20m_budget_status": science_budget_status,
        "truth_boundary": {
            "corpus_payload_read": False,
            "tokenizer_fit_executed": False,
            "model_training_executed": False,
            "optimizer_updates": 0,
            "selection_validation_consumed": False,
            "final_test_consumed": False,
            "training_authorized": False,
            "paid_compute_authorized": False,
            "stage_promotion_authorized": False,
        },
        "next_required_evidence": [
            "terminal Research Corpus V1 identity and calibration-slice identity",
            "exact reproducible tokenizer identities for all subword candidates",
            "at least two measured repeats per tokenizer candidate",
            "per-stratum bits-per-byte and measured FLOPs on identical UTF-8 bytes",
            "bounded equal-FLOP held-out learning curves",
            "empirical loss-versus-FLOP fit before science-complete budget proposal",
        ],
        "raw_text_emitted": False,
    }
    return {**core, "report_sha256": _sha256(core)}


def verify_report(report: Mapping[str, Any]) -> None:
    _require(report.get("schema_version") == REPORT_SCHEMA, "report schema mismatch")
    expected = report.get("report_sha256")
    core = dict(report)
    core.pop("report_sha256", None)
    _require(expected == _sha256(core), "report self-hash mismatch")
    boundary = report["truth_boundary"]
    for key in (
        "corpus_payload_read",
        "tokenizer_fit_executed",
        "model_training_executed",
        "selection_validation_consumed",
        "final_test_consumed",
        "training_authorized",
        "paid_compute_authorized",
        "stage_promotion_authorized",
    ):
        _require(boundary.get(key) is False, f"report truth boundary weakened: {key}")
    _require(boundary.get("optimizer_updates") == 0, "report optimizer updates must remain zero")
    _require(report.get("raw_text_emitted") is False, "raw text emission boundary weakened")
    _require(
        report.get("science_complete_20m_budget_status")
        in {
            "UNDEFINED_PENDING_TOKENIZER_AND_FLOP_CALIBRATION",
            "UNDEFINED_PENDING_HELDOUT_LEARNING_CURVES_AND_COMPUTE_AUTHORIZATION",
        },
        "science budget was improperly promoted",
    )
