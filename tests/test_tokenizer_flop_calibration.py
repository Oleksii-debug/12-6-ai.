from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest

from twelve_six import tokenizer_flop_calibration as r02

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "configs/research/r02_tokenizer_flop_calibration_v1.json"


def _contract() -> dict:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def _measurement(
    tokenizer_id: str,
    *,
    repeat_id: str,
    corpus: str = "corpus-sha256-terminal-v1",
    calibration_slice: str = "calibration-slice-sha256-v1",
    byte_delta: int = 0,
) -> dict:
    contract = _contract()
    candidate = next(
        item
        for item in contract["calibration_scope"]["required_tokenizer_candidates"]
        if item["id"] == tokenizer_id
    )
    position_ratio = {
        "byte-v256": 1.0,
        "subword-v320": 0.82,
        "subword-v384": 0.76,
        "subword-v437": 0.71,
        "subword-v512": 0.66,
    }[tokenizer_id]
    bpb = {
        "byte-v256": 1.60,
        "subword-v320": 1.53,
        "subword-v384": 1.49,
        "subword-v437": 1.47,
        "subword-v512": 1.46,
    }[tokenizer_id]
    repeat_factor = 1.0 if repeat_id == "r1" else 1.01

    strata = {}
    for name, utf8_bytes in {"uk": 1000, "en": 1200, "code": 800}.items():
        utf8_bytes += byte_delta
        positions = max(1, int(round(utf8_bytes * position_ratio)))
        total_nll_nats = bpb * utf8_bytes * math.log(2.0) * repeat_factor
        measured_flops = positions * candidate["expected_total_parameters"] * 6.0 * repeat_factor
        strata[name] = {
            "utf8_bytes": utf8_bytes,
            "nonignored_loss_positions": positions,
            "total_nll_nats": total_nll_nats,
            "measured_training_flops": measured_flops,
            "wall_seconds": 0.5 + positions / 10000.0,
        }

    return {
        "schema_version": r02.MEASUREMENT_SCHEMA,
        "tokenizer_id": tokenizer_id,
        "tokenizer_kind": candidate["kind"],
        "vocab_size": candidate["vocab_size"],
        "tokenizer_identity": (
            "byte-identity-256-v1"
            if candidate["kind"] == "byte"
            else f"sha256-{tokenizer_id}-terminal-fit"
        ),
        "research_corpus_identity": corpus,
        "calibration_slice_identity": calibration_slice,
        "model_total_parameter_count": candidate["expected_total_parameters"],
        "model_nonembedding_parameter_count": contract["parent_r01"]["nonembedding_parameter_count"],
        "model_body_identity": "model341-transformer-body-v1",
        "loss_mask_identity": "causal-loss-mask-v1",
        "context_window_loss_positions": 1024,
        "repeat_id": repeat_id,
        "peak_memory_bytes": 123456789 + candidate["vocab_size"],
        "strata": strata,
    }


def _complete_measurements() -> list[dict]:
    contract = _contract()
    values = []
    for candidate in contract["calibration_scope"]["required_tokenizer_candidates"]:
        values.append(_measurement(candidate["id"], repeat_id="r1"))
        values.append(_measurement(candidate["id"], repeat_id="r2"))
    return values


def test_contract_binds_tied_embedding_parameter_arithmetic() -> None:
    contract = _contract()
    r02.validate_contract(contract)
    parent = contract["parent_r01"]
    assert parent["nonembedding_parameter_count"] == 20531520
    for candidate in contract["calibration_scope"]["required_tokenizer_candidates"]:
        assert candidate["expected_total_parameters"] == (
            parent["nonembedding_parameter_count"]
            + candidate["vocab_size"] * parent["d_model"]
        )


def test_partial_report_stays_science_budget_undefined() -> None:
    contract = _contract()
    report = r02.build_report(contract, [])
    r02.verify_report(report)
    assert report["status"] == "INCOMPLETE_MEASUREMENTS"
    assert report["coverage"]["missing_candidate_ids"] == [
        "byte-v256",
        "subword-v320",
        "subword-v384",
        "subword-v437",
        "subword-v512",
    ]
    assert (
        report["science_complete_20m_budget_status"]
        == "UNDEFINED_PENDING_TOKENIZER_AND_FLOP_CALIBRATION"
    )
    assert report["truth_boundary"]["training_authorized"] is False
    assert report["truth_boundary"]["paid_compute_authorized"] is False


def test_complete_measurements_produce_equal_flop_projection_not_training_authority() -> None:
    contract = _contract()
    report = r02.build_report(contract, _complete_measurements())
    r02.verify_report(report)
    assert report["status"] == "TOKENIZER_FLOP_CALIBRATION_READY_REQUIRES_LEARNING_CURVES"
    assert report["coverage"]["missing_candidate_ids"] == []
    assert report["coverage"]["below_minimum_repeat_counts"] == {}
    assert set(report["equal_flop_projection"]) == {
        "byte-v256",
        "subword-v320",
        "subword-v384",
        "subword-v437",
        "subword-v512",
    }
    assert (
        report["science_complete_20m_budget_status"]
        == "UNDEFINED_PENDING_HELDOUT_LEARNING_CURVES_AND_COMPUTE_AUTHORIZATION"
    )
    assert report["truth_boundary"]["model_training_executed"] is False
    assert report["truth_boundary"]["training_authorized"] is False


def test_bits_per_byte_is_derived_from_nll_and_original_utf8_bytes() -> None:
    contract = _contract()
    measurement = _measurement("byte-v256", repeat_id="r1")
    report = r02.build_report(
        contract,
        [measurement, _measurement("byte-v256", repeat_id="r2")],
    )
    uk = report["tokenizers"]["byte-v256"]["strata"]["uk"]
    assert uk["utf8_bytes"] == 1000
    assert uk["nonignored_loss_positions"] == 1000
    assert uk["bits_per_utf8_byte"] == pytest.approx(1.608, abs=1e-12)
    assert uk["semantic_context_span_utf8_bytes"] == pytest.approx(1024.0)


def test_cross_tokenizer_corpus_identity_drift_fails_closed() -> None:
    contract = _contract()
    measurements = [
        _measurement("byte-v256", repeat_id="r1"),
        _measurement("byte-v256", repeat_id="r2"),
        _measurement("subword-v320", repeat_id="r1", corpus="other-corpus"),
    ]
    with pytest.raises(r02.CalibrationError, match="research_corpus_identity"):
        r02.build_report(contract, measurements)


def test_cross_tokenizer_exact_byte_drift_fails_closed() -> None:
    contract = _contract()
    measurements = [
        _measurement("byte-v256", repeat_id="r1"),
        _measurement("byte-v256", repeat_id="r2"),
        _measurement("subword-v320", repeat_id="r1", byte_delta=1),
        _measurement("subword-v320", repeat_id="r2", byte_delta=1),
    ]
    with pytest.raises(r02.CalibrationError, match="calibration bytes differ"):
        r02.build_report(contract, measurements)


def test_vocab_parameter_mismatch_fails_closed() -> None:
    contract = _contract()
    measurement = _measurement("subword-v512", repeat_id="r1")
    measurement["model_total_parameter_count"] += 1
    with pytest.raises(r02.CalibrationError, match="total parameter count mismatch"):
        r02.validate_measurement(measurement, contract)


def test_duplicate_repeat_id_fails_closed() -> None:
    contract = _contract()
    measurement = _measurement("byte-v256", repeat_id="r1")
    with pytest.raises(r02.CalibrationError, match="duplicate measurement repeat"):
        r02.build_report(contract, [measurement, copy.deepcopy(measurement)])
