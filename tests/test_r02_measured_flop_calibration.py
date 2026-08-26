from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from twelve_six import measured_flop_calibration as r02

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "configs/research/r02_measured_flop_equal_budget_v1.json"


def _contract() -> dict:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def _measurement(
    candidate_id: str,
    repeat_id: str,
    *,
    corpus_identity: str = "corpus-terminal-sha256",
    byte_delta: int = 0,
) -> dict:
    contract = _contract()
    candidate = next(row for row in contract["candidates"] if row["id"] == candidate_id)
    position_ratio = {
        "byte-v256": 1.0,
        "subword-v320": 0.82,
        "subword-v384": 0.76,
        "subword-v437": 0.71,
        "subword-v512": 0.66,
    }[candidate_id]
    bpb = {
        "byte-v256": 1.60,
        "subword-v320": 1.53,
        "subword-v384": 1.49,
        "subword-v437": 1.47,
        "subword-v512": 1.46,
    }[candidate_id]
    repeat_scale = 1.0 if repeat_id == "r1" else 1.01
    strata = {}
    for stratum, base_bytes in {"UA": 1000, "EN": 1200, "CODE": 800}.items():
        utf8_bytes = base_bytes + byte_delta
        positions = max(1, int(round(utf8_bytes * position_ratio)))
        strata[stratum] = {
            "utf8_bytes": utf8_bytes,
            "nonignored_loss_positions": positions,
            "bits_per_byte": bpb * repeat_scale,
            "bpb_result_identity": f"bpb-{candidate_id}-{stratum.lower()}-terminal",
            "measured_training_flops": (
                positions * candidate["expected_total_parameters"] * 6.0 * repeat_scale
            ),
            "wall_seconds": 0.5 + positions / 10000.0,
        }
    return {
        "tokenizer_id": candidate_id,
        "tokenizer_identity": f"tok-{candidate_id}-terminal",
        "research_corpus_identity": corpus_identity,
        "calibration_slice_identity": "calibration-slice-terminal-sha256",
        "model_body_identity": "model341-nonembedding-body-v1",
        "loss_mask_identity": "causal-loss-mask-v1",
        "bpb_metric_authority_identity": "d06-bpb-terminal-authority",
        "flop_counter_identity": "measured-flop-counter-v1",
        "context_window_loss_positions": 1024,
        "model_total_parameters": candidate["expected_total_parameters"],
        "model_nonembedding_parameters": contract["model_geometry"]["nonembedding_parameters"],
        "peak_memory_bytes": 123456789 + candidate["vocab_size"],
        "repeat_id": repeat_id,
        "optimizer_steps": 0,
        "strata": strata,
    }


def _complete_measurements() -> list[dict]:
    values = []
    for candidate in _contract()["candidates"]:
        values.append(_measurement(candidate["id"], "r1"))
        values.append(_measurement(candidate["id"], "r2"))
    return values


def test_contract_binds_parent_and_vocab_dependent_parameter_term() -> None:
    contract = _contract()
    r02.validate_contract(contract)
    geometry = contract["model_geometry"]
    assert geometry["nonembedding_parameters"] == 20531520
    for candidate in contract["candidates"]:
        assert candidate["expected_total_parameters"] == (
            geometry["nonembedding_parameters"]
            + candidate["vocab_size"] * geometry["d_model"]
        )


def test_empty_report_never_invents_science_budget() -> None:
    report = r02.build_report(_contract(), [])
    r02.verify_report(report)
    assert report["status"] == "INCOMPLETE_MEASUREMENTS"
    assert report["science_complete_20m_budget"] is None
    assert (
        report["science_complete_20m_budget_status"]
        == "UNDEFINED_PENDING_TOKENIZER_AND_FLOP_CALIBRATION"
    )
    assert report["truth_boundary"]["training_authorized"] is False


def test_complete_measurements_create_equal_flop_projection_but_not_authority() -> None:
    report = r02.build_report(_contract(), _complete_measurements())
    r02.verify_report(report)
    assert (
        report["status"]
        == "TOKENIZER_FLOP_CALIBRATION_READY_REQUIRES_HELDOUT_LEARNING_CURVES"
    )
    assert report["coverage"]["missing_candidates"] == []
    assert report["coverage"]["below_minimum_repeat_counts"] == {}
    assert set(report["equal_flop_projection"]) == {
        "byte-v256",
        "subword-v320",
        "subword-v384",
        "subword-v437",
        "subword-v512",
    }
    assert report["science_complete_20m_budget"] is None
    assert (
        report["science_complete_20m_budget_status"]
        == "UNDEFINED_PENDING_HELDOUT_LEARNING_CURVES_AND_COMPUTE_AUTHORIZATION"
    )
    assert report["truth_boundary"]["optimizer_updates"] == 0
    assert report["truth_boundary"]["promotion_to_100m_authorized"] is False


def test_aggregate_bpb_is_byte_weighted_and_external() -> None:
    report = r02.build_report(
        _contract(),
        [_measurement("byte-v256", "r1"), _measurement("byte-v256", "r2")],
    )
    aggregate = report["tokenizers"]["byte-v256"]["aggregate"]
    assert aggregate["utf8_bytes"] == 3000
    assert aggregate["nonignored_loss_positions"] == 3000
    assert aggregate["bits_per_byte"] == pytest.approx(1.608, abs=1e-12)
    assert aggregate["semantic_context_span_utf8_bytes"] == pytest.approx(1024.0)
    assert report["truth_boundary"]["bits_per_byte_computed_by_analyzer"] is False


def test_cross_tokenizer_corpus_identity_drift_fails_closed() -> None:
    measurements = [
        _measurement("byte-v256", "r1"),
        _measurement("byte-v256", "r2"),
        _measurement("subword-v320", "r1", corpus_identity="other-corpus"),
    ]
    with pytest.raises(r02.FlopCalibrationError, match="research_corpus_identity"):
        r02.build_report(_contract(), measurements)


def test_cross_tokenizer_exact_byte_drift_fails_closed() -> None:
    measurements = [
        _measurement("byte-v256", "r1"),
        _measurement("byte-v256", "r2"),
        _measurement("subword-v320", "r1", byte_delta=1),
        _measurement("subword-v320", "r2", byte_delta=1),
    ]
    with pytest.raises(r02.FlopCalibrationError, match="calibration-byte drift"):
        r02.build_report(_contract(), measurements)


def test_optimizer_step_is_forbidden() -> None:
    measurement = _measurement("subword-v384", "r1")
    measurement["optimizer_steps"] = 1
    with pytest.raises(r02.FlopCalibrationError, match="optimizer step forbidden"):
        r02.validate_measurement(measurement, _contract())


def test_wrong_vocab_dependent_total_fails_closed() -> None:
    measurement = _measurement("subword-v512", "r1")
    measurement["model_total_parameters"] += 1
    with pytest.raises(r02.FlopCalibrationError, match="total parameter mismatch"):
        r02.validate_measurement(measurement, _contract())


def test_duplicate_repeat_fails_closed() -> None:
    measurement = _measurement("byte-v256", "r1")
    with pytest.raises(r02.FlopCalibrationError, match="duplicate repeat"):
        r02.build_report(_contract(), [measurement, copy.deepcopy(measurement)])


def test_nan_or_formula_placeholder_flops_fail_closed() -> None:
    measurement = _measurement("byte-v256", "r1")
    measurement["strata"]["UA"]["measured_training_flops"] = float("nan")
    with pytest.raises(r02.FlopCalibrationError, match="finite number"):
        r02.validate_measurement(measurement, _contract())
