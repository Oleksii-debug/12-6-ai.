import json
from pathlib import Path

import pytest

from twelve_six.training.embedding_tax import (
    EmbeddingTaxError,
    RETAINED_BPE,
    TARGETS,
    allocation_row,
    geometry_profiles,
    tokenizer_candidates_from_report,
)


def _report() -> dict:
    return {
        "requested_vocab_size": 512,
        "algorithms": {
            "bpe": {
                "artifact": {
                    "training_manifest_sha256": RETAINED_BPE["training_manifest_sha256"],
                    "tokenizer_json_sha256": RETAINED_BPE["tokenizer_json_sha256"],
                    "vocab_sha256": RETAINED_BPE["vocab_sha256"],
                    "config_sha256": RETAINED_BPE["config_sha256"],
                    "vocab_size": 472,
                },
                "held_out": {
                    "byte_baseline_tokens": 520,
                    "tokens": 286,
                    "token_reduction_vs_bytes": 0.45,
                    "strict_round_trip_all": True,
                    "unknown_tokens": 0,
                },
                "repeatability_status": "PASS",
            },
            "unigram": {
                "artifact": {"vocab_size": 497},
                "held_out": {
                    "tokens": 284,
                    "token_reduction_vs_bytes": 0.45384615384615384,
                    "strict_round_trip_all": True,
                    "unknown_tokens": 0,
                },
                "repeatability_status": "FAIL",
            },
        },
    }


def test_live_report_uses_actual_not_requested_vocab() -> None:
    candidates = tokenizer_candidates_from_report(_report())
    assert [(row["requested_vocab_size"], row["actual_vocab_size"]) for row in candidates] == [
        (256, 256),
        (512, 472),
        (512, 497),
    ]
    assert candidates[1]["repeatability_status"] == "PASS"
    assert candidates[2]["repeatability_status"] == "FAIL"


def test_bpe_artifact_drift_fails_closed() -> None:
    report = _report()
    report["algorithms"]["bpe"]["artifact"]["vocab_size"] = 512
    with pytest.raises(EmbeddingTaxError, match="actual vocabulary drift"):
        tokenizer_candidates_from_report(report)


def test_cross_scale_rows_include_tied_untied_and_block_tax() -> None:
    profiles = geometry_profiles(Path("."))
    bpe = tokenizer_candidates_from_report(_report())[1]
    for target in TARGETS:
        row = allocation_row(profiles[target], target=target, tokenizer=bpe)
        assert row["actual_vocab_size"] == 472
        assert row["tied"]["embedding_parameters"] == 472 * row["tied"]["d_model"]
        assert 0.0 < row["tied"]["embedding_fraction"] < 1.0
        assert 0.0 < row["tied"]["transformer_block_fraction"] < 1.0
        assert row["hypothetical_untied_same_d_ff"]["extra_output_matrix_parameters"] == 472 * row["tied"]["d_model"]
        assert row["untied_rebalanced_near_target"]["d_ff"] <= row["tied"]["d_ff"]
        assert abs(row["tied"]["target_delta"]) < 10000
