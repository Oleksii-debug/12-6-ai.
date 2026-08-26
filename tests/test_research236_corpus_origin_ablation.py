from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from research236_analyze import analyze  # noqa: E402
from research236_prerequisite_gate import (  # noqa: E402
    DATA25_ID,
    EVAL_ID,
    MATCHED_OPTIMIZED_TOKENS,
    MODEL_SPECS,
    PAIRED_SEEDS,
    build_gate_report,
)


def _data230() -> dict:
    return {
        "worker_id": "DATA-230-CORPUS-V03-EXTERNAL-REAL",
        "status": "PASS",
        "origin_classes": ["EXTERNAL_REAL", "PROJECT_AUTHORED"],
        "deterministic_two_builds_identical": True,
        "train_loss_token_supply": 1_000_000,
        "artificial_repetition": False,
    }


def _eval233() -> dict:
    return {
        "worker_id": "EVAL-233-REAL-HOLDOUT-V2",
        "status": "PASS",
        "purposes": ["selection-validation", "final-test"],
        "final_test_exposed_to_selection": False,
    }


def test_gate_blocks_when_external_authorities_absent() -> None:
    report = build_gate_report(data230=None, eval233=None, source_sha="a" * 40)
    assert report["status"].startswith("BLOCKED")
    assert any("DATA-230" in blocker for blocker in report["blockers"])
    assert any("EVAL-233" in blocker for blocker in report["blockers"])
    assert report["claim_boundary"]["numerical_ablation_claim_permitted"] is False


def test_gate_accepts_terminal_deterministic_authorities() -> None:
    report = build_gate_report(data230=_data230(), eval233=_eval233(), source_sha="b" * 40)
    assert report["status"] == "READY_TO_EXECUTE"
    assert report["blockers"] == []
    assert report["frozen"]["matched_actual_optimized_tokens"] == MATCHED_OPTIMIZED_TOKENS
    assert report["frozen"]["source_loss_token_repetition_allowed"] is False


def _cell(scale: str, corpus: str, seed: int, base: float) -> dict:
    spec = MODEL_SPECS[scale]
    metrics = {
        "train_bpb": base - 0.4,
        "data25_selection_bpb": base,
        "external_selection_bpb": base + 0.1,
        "common_real_holdout_bpb": base + 0.2,
        "ua_bpb": base + 0.3,
        "en_bpb": base + 0.1,
        "code_bpb": base + 0.4,
        "data25_train_probe_bpb": base - 0.2,
        "external_train_probe_bpb": base - 0.1,
    }
    return {
        "scale": scale,
        "corpus": corpus,
        "seed": seed,
        "tokenizer": "s0-byte-v1",
        "evaluation_identity": EVAL_ID,
        "parameters": spec["parameters"],
        "model_spec_sha256": spec["model_spec_sha256"],
        "actual_optimized_loss_tokens": MATCHED_OPTIMIZED_TOKENS,
        "source_loss_tokens_consumed": MATCHED_OPTIMIZED_TOKENS,
        "unique_source_loss_token_positions": MATCHED_OPTIMIZED_TOKENS,
        "repeated_source_loss_token_positions": 0,
        "padded_tensor_positions_counted": 0,
        "fresh_random_initialization": True,
        "training_corpus_identity": DATA25_ID if corpus == "data25" else "external-real-id",
        "metrics": metrics,
        "source_family_bpb": {"family_a": base + 0.25, "family_b": base + 0.35},
    }


def _payload() -> dict:
    data25 = [_cell("500k", "data25", seed, 2.0 + idx * 0.01) for idx, seed in enumerate(PAIRED_SEEDS)]
    ext = [_cell("500k", "external_real", seed, 1.9 + idx * 0.01) for idx, seed in enumerate(PAIRED_SEEDS)]
    return {
        "worker_id": "RESEARCH-236-CORPUS-ORIGIN-ABLATION",
        "data230_terminal_identity": "data230-terminal",
        "common_real_holdout_identity": "eval233-common",
        "scales": {"500k": {"data25": data25, "external_real": ext}},
    }


def test_analyzer_reports_paired_direction_without_superiority_claim() -> None:
    result = analyze(_payload())
    summary = result["scales"]["500k"]["paired_same_bytes_metric_summaries"]["common_real_holdout_bpb"]
    assert summary["direction"] == "EXTERNAL_REAL_LOWER_BPB_ALL_PAIRED_SEEDS"
    assert summary["mean_delta"] == pytest.approx(-0.1)
    sensitivity = result["scales"]["500k"]["source_family_sensitivity"]
    assert sensitivity["by_training_corpus"]["data25"]["sensitivity_spread_bpb"] == pytest.approx(0.1)
    assert result["conclusion"]["external_real_automatically_better"] is False


def test_analyzer_rejects_nominal_or_padded_exposure_mismatch() -> None:
    payload = _payload()
    broken = copy.deepcopy(payload)
    broken["scales"]["500k"]["external_real"][0]["actual_optimized_loss_tokens"] += 1
    with pytest.raises(ValueError, match="optimized-token mismatch"):
        analyze(broken)


def test_analyzer_rejects_source_token_repetition() -> None:
    payload = _payload()
    broken = copy.deepcopy(payload)
    broken["scales"]["500k"]["external_real"][0]["repeated_source_loss_token_positions"] = 1
    with pytest.raises(ValueError, match="repetition"):
        analyze(broken)


def test_analyzer_rejects_noncommon_source_family_keys() -> None:
    payload = _payload()
    broken = copy.deepcopy(payload)
    broken["scales"]["500k"]["external_real"][0]["source_family_bpb"] = {"different": 1.0}
    with pytest.raises(ValueError, match="source-family heldout key mismatch"):
        analyze(broken)
