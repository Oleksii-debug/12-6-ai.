"""R02 measured-FLOP calibration over externally produced BPB evidence.

This module is analysis-only. It consumes aggregate measurements, does not read
corpus text, does not compute BPB itself, and never performs an optimizer step.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

CONTRACT_SCHEMA = "12-6.r02-measured-flop-equal-budget.v1"
REPORT_SCHEMA = "12-6.r02-measured-flop-equal-budget-report.v1"
CAMPAIGN_ID = "R02-MEASURED-FLOP-EQUAL-BUDGET-V1"


class FlopCalibrationError(ValueError):
    """Fail-closed R02 calibration error."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FlopCalibrationError(message)


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _rounded(value: float) -> float:
    return round(float(value), 12)


def load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FlopCalibrationError(f"{path}: JSON root must be an object")
    return value


def _candidates(contract: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows = contract["candidates"]
    result = {str(row["id"]): dict(row) for row in rows}
    _require(len(result) == len(rows), "duplicate candidate id")
    return result


def validate_contract(contract: Mapping[str, Any]) -> None:
    _require(contract.get("schema") == CONTRACT_SCHEMA, "contract schema drift")
    _require(contract.get("campaign_id") == CAMPAIGN_ID, "campaign id drift")
    _require(
        contract.get("status") == "MEASUREMENT_CONTRACT_READY_DATA_BLOCKED",
        "contract status drift",
    )

    parent = contract["parent"]
    _require(parent.get("pr") == 706, "parent PR drift")
    _require(
        parent.get("head_sha") == "17949f5a4d65d8275c99687df3e9b3b43a5d3c77",
        "parent exact-head drift",
    )
    _require(
        parent.get("unit_firewall_git_blob_sha1")
        == "8c9ea85e376999acb65ff6243059edc8f42fcd27",
        "unit-firewall blob drift",
    )
    _require(parent.get("required_cross_tokenizer_metric") == "BITS_PER_BYTE", "BPB authority drift")
    _require(
        parent.get("required_flop_calibration")
        == "flop_normalized_byte_vs_subword_ablation",
        "FLOP calibration requirement drift",
    )

    geometry = contract["model_geometry"]
    _require(
        geometry.get("model341_sha") == "e4ff486fd90802fc123bebf60eed4e59196a98df",
        "MODEL-341 identity drift",
    )
    _require(geometry.get("baseline_vocab_size") == 256, "baseline vocab drift")
    _require(geometry.get("d_model") == 320, "d_model drift")
    _require(geometry.get("baseline_total_parameters") == 20613440, "baseline total drift")
    _require(geometry.get("baseline_tied_embedding_parameters") == 81920, "embedding count drift")
    _require(geometry.get("nonembedding_parameters") == 20531520, "body count drift")
    _require(geometry.get("tied_embeddings_required") is True, "tied embedding invariant weakened")
    _require(
        geometry["nonembedding_parameters"]
        + geometry["baseline_vocab_size"] * geometry["d_model"]
        == geometry["baseline_total_parameters"],
        "baseline tied-embedding arithmetic drift",
    )

    expected = {
        "byte-v256": ("byte", 256, 20613440),
        "subword-v320": ("subword", 320, 20633920),
        "subword-v384": ("subword", 384, 20654400),
        "subword-v437": ("subword", 437, 20671360),
        "subword-v512": ("subword", 512, 20695360),
    }
    candidates = _candidates(contract)
    _require(set(candidates) == set(expected), "candidate set drift")
    for candidate_id, (kind, vocab, total) in expected.items():
        row = candidates[candidate_id]
        _require(row.get("kind") == kind, f"{candidate_id}: kind drift")
        _require(row.get("vocab_size") == vocab, f"{candidate_id}: vocab drift")
        _require(row.get("expected_total_parameters") == total, f"{candidate_id}: total drift")
        _require(
            geometry["nonembedding_parameters"] + vocab * geometry["d_model"] == total,
            f"{candidate_id}: tied-embedding arithmetic drift",
        )

    measurement = contract["measurement_contract"]
    _require(measurement.get("required_strata") == ["UA", "EN", "CODE"], "strata drift")
    _require(measurement.get("minimum_repeats_per_candidate") == 2, "repeat floor drift")
    for key in (
        "same_exact_calibration_slice_identity_across_candidates",
        "same_exact_utf8_bytes_per_stratum_across_candidates",
        "same_transformer_body_identity_across_candidates",
        "same_nonembedding_parameter_count_across_candidates",
        "same_loss_mask_identity_across_candidates",
        "same_context_window_loss_positions_across_candidates",
        "total_parameter_count_may_vary_only_by_tied_embedding_vocab_term",
        "bits_per_byte_must_come_from_external_d06_authority",
        "measured_flops_must_not_be_formula_only_for_final_ranking",
        "optimizer_step_forbidden_during_calibration",
    ):
        _require(measurement.get(key) is True, f"measurement invariant weakened: {key}")

    reference = contract["reference_compute_envelope"]
    _require(reference.get("byte_baseline_loss_positions") == 20000000, "reference pilot drift")
    _require(reference.get("is_science_complete_budget") is False, "pilot promoted to science budget")

    completion = contract["completion_rule"]
    _require(
        completion.get("complete_measurements_status")
        == "TOKENIZER_FLOP_CALIBRATION_READY_REQUIRES_HELDOUT_LEARNING_CURVES",
        "completion status drift",
    )
    _require(
        completion.get("science_complete_20m_budget_after_measurements")
        == "UNDEFINED_PENDING_HELDOUT_LEARNING_CURVES_AND_COMPUTE_AUTHORIZATION",
        "science-budget truth drift",
    )
    for key in ("training_authorized", "paid_compute_authorized", "promotion_to_100m_authorized"):
        _require(completion.get(key) is False, f"completion authority widened: {key}")

    boundary = contract["truth_boundary"]
    for key in (
        "corpus_payload_read_by_analyzer",
        "tokenizer_fit_executed_by_analyzer",
        "model_training_executed_by_analyzer",
        "selection_validation_consumed",
        "final_test_consumed",
        "training_authorized",
        "paid_compute_authorized",
        "promotion_to_100m_authorized",
    ):
        _require(boundary.get(key) is False, f"truth boundary widened: {key}")
    _require(boundary.get("optimizer_updates") == 0, "optimizer updates must remain zero")


def _positive_number(value: Any, label: str, *, allow_zero: bool = False) -> float:
    _require(not isinstance(value, bool) and isinstance(value, (int, float)), f"{label}: number required")
    result = float(value)
    _require(math.isfinite(result), f"{label}: finite number required")
    _require(result >= 0.0 if allow_zero else result > 0.0, f"{label}: invalid value")
    return result


def validate_measurement(measurement: Mapping[str, Any], contract: Mapping[str, Any]) -> None:
    validate_contract(contract)
    spec = contract["measurement_contract"]
    for field in spec["required_top_level_fields"]:
        _require(field in measurement, f"missing top-level field: {field}")

    candidates = _candidates(contract)
    candidate_id = str(measurement["tokenizer_id"])
    _require(candidate_id in candidates, f"unknown tokenizer candidate: {candidate_id}")
    candidate = candidates[candidate_id]
    geometry = contract["model_geometry"]

    for field in (
        "tokenizer_identity",
        "research_corpus_identity",
        "calibration_slice_identity",
        "model_body_identity",
        "loss_mask_identity",
        "bpb_metric_authority_identity",
        "flop_counter_identity",
        "repeat_id",
    ):
        value = measurement[field]
        _require(isinstance(value, str) and value.strip(), f"{candidate_id}: empty {field}")

    _require(
        measurement["model_total_parameters"] == candidate["expected_total_parameters"],
        f"{candidate_id}: total parameter mismatch",
    )
    _require(
        measurement["model_nonembedding_parameters"] == geometry["nonembedding_parameters"],
        f"{candidate_id}: nonembedding parameter mismatch",
    )
    _require(
        isinstance(measurement["context_window_loss_positions"], int)
        and not isinstance(measurement["context_window_loss_positions"], bool)
        and measurement["context_window_loss_positions"] > 0,
        f"{candidate_id}: invalid context window",
    )
    _require(
        isinstance(measurement["peak_memory_bytes"], int)
        and not isinstance(measurement["peak_memory_bytes"], bool)
        and measurement["peak_memory_bytes"] > 0,
        f"{candidate_id}: invalid peak memory",
    )
    _require(measurement["optimizer_steps"] == 0, f"{candidate_id}: optimizer step forbidden")

    strata = measurement["strata"]
    _require(isinstance(strata, Mapping), f"{candidate_id}: strata must be an object")
    required_strata = spec["required_strata"]
    _require(set(strata) == set(required_strata), f"{candidate_id}: stratum set mismatch")
    for stratum in required_strata:
        row = strata[stratum]
        _require(isinstance(row, Mapping), f"{candidate_id}/{stratum}: row must be object")
        for field in spec["required_per_stratum_fields"]:
            _require(field in row, f"{candidate_id}/{stratum}: missing {field}")
        for field in ("utf8_bytes", "nonignored_loss_positions"):
            value = row[field]
            _require(
                isinstance(value, int) and not isinstance(value, bool) and value > 0,
                f"{candidate_id}/{stratum}: invalid {field}",
            )
        _positive_number(row["bits_per_byte"], f"{candidate_id}/{stratum}/bits_per_byte", allow_zero=True)
        _positive_number(row["measured_training_flops"], f"{candidate_id}/{stratum}/measured_training_flops")
        _positive_number(row["wall_seconds"], f"{candidate_id}/{stratum}/wall_seconds")
        bpb_id = row["bpb_result_identity"]
        _require(isinstance(bpb_id, str) and bpb_id.strip(), f"{candidate_id}/{stratum}: empty BPB result identity")


def _metrics(
    *,
    utf8_bytes: int,
    loss_positions: int,
    bits_per_byte: float,
    measured_flops: float,
    wall_seconds: float,
    context_positions: int,
) -> dict[str, float | int]:
    bytes_per_position = utf8_bytes / loss_positions
    return {
        "utf8_bytes": utf8_bytes,
        "nonignored_loss_positions": loss_positions,
        "bits_per_byte": _rounded(bits_per_byte),
        "loss_positions_per_utf8_byte": _rounded(loss_positions / utf8_bytes),
        "utf8_bytes_per_loss_position": _rounded(bytes_per_position),
        "measured_training_flops_per_utf8_byte": _rounded(measured_flops / utf8_bytes),
        "measured_training_flops_per_loss_position": _rounded(measured_flops / loss_positions),
        "semantic_context_span_utf8_bytes": _rounded(context_positions * bytes_per_position),
        "utf8_bytes_per_second": _rounded(utf8_bytes / wall_seconds),
        "loss_positions_per_second": _rounded(loss_positions / wall_seconds),
    }


def _aggregate_candidate(
    candidate_id: str,
    repeats: list[Mapping[str, Any]],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    first = repeats[0]
    strata_names = contract["measurement_contract"]["required_strata"]
    for repeat in repeats[1:]:
        for field in (
            "tokenizer_identity",
            "research_corpus_identity",
            "calibration_slice_identity",
            "model_body_identity",
            "loss_mask_identity",
            "bpb_metric_authority_identity",
            "flop_counter_identity",
            "context_window_loss_positions",
            "model_total_parameters",
            "model_nonembedding_parameters",
        ):
            _require(repeat[field] == first[field], f"{candidate_id}: repeat drift in {field}")
        for stratum in strata_names:
            for field in ("utf8_bytes", "nonignored_loss_positions", "bpb_result_identity"):
                _require(
                    repeat["strata"][stratum][field] == first["strata"][stratum][field],
                    f"{candidate_id}/{stratum}: repeat drift in {field}",
                )

    context = int(first["context_window_loss_positions"])
    per_stratum: dict[str, Any] = {}
    total_bytes = 0
    total_positions = 0
    weighted_bpb_numerator = 0.0
    total_flops = 0.0
    total_wall = 0.0
    for stratum in strata_names:
        row = first["strata"][stratum]
        utf8_bytes = int(row["utf8_bytes"])
        positions = int(row["nonignored_loss_positions"])
        avg_bpb = sum(float(item["strata"][stratum]["bits_per_byte"]) for item in repeats) / len(repeats)
        avg_flops = sum(float(item["strata"][stratum]["measured_training_flops"]) for item in repeats) / len(repeats)
        avg_wall = sum(float(item["strata"][stratum]["wall_seconds"]) for item in repeats) / len(repeats)
        per_stratum[stratum] = {
            **_metrics(
                utf8_bytes=utf8_bytes,
                loss_positions=positions,
                bits_per_byte=avg_bpb,
                measured_flops=avg_flops,
                wall_seconds=avg_wall,
                context_positions=context,
            ),
            "bpb_result_identity": row["bpb_result_identity"],
        }
        total_bytes += utf8_bytes
        total_positions += positions
        weighted_bpb_numerator += avg_bpb * utf8_bytes
        total_flops += avg_flops
        total_wall += avg_wall

    aggregate_bpb = weighted_bpb_numerator / total_bytes
    return {
        "tokenizer_id": candidate_id,
        "tokenizer_identity": first["tokenizer_identity"],
        "model_total_parameters": first["model_total_parameters"],
        "model_nonembedding_parameters": first["model_nonembedding_parameters"],
        "repeat_count": len(repeats),
        "repeat_ids": sorted(str(item["repeat_id"]) for item in repeats),
        "peak_memory_bytes_max": max(int(item["peak_memory_bytes"]) for item in repeats),
        "strata": per_stratum,
        "aggregate": _metrics(
            utf8_bytes=total_bytes,
            loss_positions=total_positions,
            bits_per_byte=aggregate_bpb,
            measured_flops=total_flops,
            wall_seconds=total_wall,
            context_positions=context,
        ),
    }


def build_report(contract: Mapping[str, Any], measurements: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    validate_contract(contract)
    materialized = [dict(item) for item in measurements]
    for measurement in materialized:
        validate_measurement(measurement, contract)

    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    seen_repeats: set[tuple[str, str]] = set()
    for measurement in materialized:
        candidate_id = str(measurement["tokenizer_id"])
        repeat_id = str(measurement["repeat_id"])
        key = (candidate_id, repeat_id)
        _require(key not in seen_repeats, f"duplicate repeat: {candidate_id}/{repeat_id}")
        seen_repeats.add(key)
        groups[candidate_id].append(measurement)

    common_fields = (
        "research_corpus_identity",
        "calibration_slice_identity",
        "model_body_identity",
        "loss_mask_identity",
        "bpb_metric_authority_identity",
        "context_window_loss_positions",
        "model_nonembedding_parameters",
    )
    if materialized:
        anchor = materialized[0]
        for measurement in materialized[1:]:
            for field in common_fields:
                _require(measurement[field] == anchor[field], f"cross-tokenizer drift in {field}")

    aggregates = {
        candidate_id: _aggregate_candidate(candidate_id, groups[candidate_id], contract)
        for candidate_id in sorted(groups)
    }
    strata_names = contract["measurement_contract"]["required_strata"]
    if len(aggregates) > 1:
        anchor_id = sorted(aggregates)[0]
        for candidate_id in sorted(aggregates):
            for stratum in strata_names:
                _require(
                    aggregates[candidate_id]["strata"][stratum]["utf8_bytes"]
                    == aggregates[anchor_id]["strata"][stratum]["utf8_bytes"],
                    f"cross-tokenizer calibration-byte drift in {stratum}",
                )

    candidates = _candidates(contract)
    minimum = int(contract["measurement_contract"]["minimum_repeats_per_candidate"])
    missing = sorted(set(candidates) - set(groups))
    below_minimum = {
        candidate_id: len(groups.get(candidate_id, []))
        for candidate_id in sorted(candidates)
        if candidate_id in groups and len(groups[candidate_id]) < minimum
    }
    complete = not missing and not below_minimum

    projections: dict[str, Any] = {}
    if "byte-v256" in aggregates:
        baseline = aggregates["byte-v256"]["aggregate"]
        reference_positions = int(contract["reference_compute_envelope"]["byte_baseline_loss_positions"])
        reference_flops = reference_positions * float(
            baseline["measured_training_flops_per_loss_position"]
        )
        for candidate_id, row in aggregates.items():
            metric = row["aggregate"]
            projections[candidate_id] = {
                "reference_byte_pilot_loss_positions": reference_positions,
                "reference_equal_flop_envelope": _rounded(reference_flops),
                "equal_flop_projected_loss_positions": _rounded(
                    reference_flops / float(metric["measured_training_flops_per_loss_position"])
                ),
                "equal_flop_projected_utf8_bytes": _rounded(
                    reference_flops / float(metric["measured_training_flops_per_utf8_byte"])
                ),
            }

    completion = contract["completion_rule"]
    core: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "status": (
            completion["complete_measurements_status"] if complete else "INCOMPLETE_MEASUREMENTS"
        ),
        "contract_identity_sha256": _sha256(contract),
        "measurement_count": len(materialized),
        "coverage": {
            "required_candidates": sorted(candidates),
            "missing_candidates": missing,
            "repeat_counts": {
                candidate_id: len(groups.get(candidate_id, []))
                for candidate_id in sorted(candidates)
            },
            "below_minimum_repeat_counts": below_minimum,
            "minimum_repeats_per_candidate": minimum,
        },
        "common_evidence": (
            {field: materialized[0][field] for field in common_fields} if materialized else None
        ),
        "tokenizers": aggregates,
        "equal_flop_projection": projections,
        "science_complete_20m_budget": None,
        "science_complete_20m_budget_status": (
            completion["science_complete_20m_budget_after_measurements"]
            if complete
            else "UNDEFINED_PENDING_TOKENIZER_AND_FLOP_CALIBRATION"
        ),
        "truth_boundary": {
            "corpus_payload_read_by_analyzer": False,
            "bits_per_byte_computed_by_analyzer": False,
            "tokenizer_fit_executed_by_analyzer": False,
            "model_training_executed_by_analyzer": False,
            "optimizer_updates": 0,
            "selection_validation_consumed": False,
            "final_test_consumed": False,
            "training_authorized": False,
            "paid_compute_authorized": False,
            "promotion_to_100m_authorized": False,
        },
        "next_required_evidence": [
            "terminal Research Corpus V1 and exact calibration-slice identity",
            "terminal D06 BPB authority and per-candidate BPB result identities",
            "at least two measured-FLOP repeats for every candidate on identical UTF-8 bytes",
            "bounded equal-FLOP held-out learning curves",
            "empirical loss-versus-FLOP fit before science-complete budget proposal",
            "explicit material-compute authorization after hardware/runtime evidence",
        ],
        "raw_text_emitted": False,
    }
    return {**core, "report_sha256": _sha256(core)}


def verify_report(report: Mapping[str, Any]) -> None:
    _require(report.get("schema") == REPORT_SCHEMA, "report schema drift")
    expected = report.get("report_sha256")
    core = dict(report)
    core.pop("report_sha256", None)
    _require(expected == _sha256(core), "report self-hash mismatch")
    _require(report.get("science_complete_20m_budget") is None, "science budget invented")
    _require(
        report.get("science_complete_20m_budget_status")
        in {
            "UNDEFINED_PENDING_TOKENIZER_AND_FLOP_CALIBRATION",
            "UNDEFINED_PENDING_HELDOUT_LEARNING_CURVES_AND_COMPUTE_AUTHORIZATION",
        },
        "science budget improperly promoted",
    )
    boundary = report["truth_boundary"]
    for key in (
        "corpus_payload_read_by_analyzer",
        "bits_per_byte_computed_by_analyzer",
        "tokenizer_fit_executed_by_analyzer",
        "model_training_executed_by_analyzer",
        "selection_validation_consumed",
        "final_test_consumed",
        "training_authorized",
        "paid_compute_authorized",
        "promotion_to_100m_authorized",
    ):
        _require(boundary.get(key) is False, f"report boundary widened: {key}")
    _require(boundary.get("optimizer_updates") == 0, "report optimizer updates must be zero")
    _require(report.get("raw_text_emitted") is False, "raw text boundary widened")
