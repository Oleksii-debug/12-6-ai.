from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from twelve_six.fixed_token_efficiency import (
    DEFAULT_TOKEN_BUDGETS,
    MODEL_SCHEMA,
    _EXPECTED_COUNTS,
    _assert_token_transition,
    _canonical_hash,
    _make_pair_batch,
    _trainer_config,
    controlled_specs,
    validate_evidence,
)
from twelve_six.model import InitSpec, TwelveSixDecoder
from twelve_six.training import Trainer


def test_reuses_exact_research41_control_family() -> None:
    specs = controlled_specs()
    assert tuple(spec.parameter_count() for spec in specs) == _EXPECTED_COUNTS
    assert {spec.vocab_size for spec in specs} == {256}
    assert {spec.max_seq_len for spec in specs} == {256}


def test_aligned_pair_batch_counts_only_requested_valid_causal_pairs() -> None:
    stream = bytes(range(32))
    batch = _make_pair_batch(
        stream,
        causal_offset=3,
        batch_size=2,
        sequence_length=4,
        valid_pairs=5,
    )
    assert int(batch["loss_mask"].sum().item()) == 5
    flat_inputs = batch["input_ids"].reshape(-1).tolist()
    flat_targets = batch["target_ids"].reshape(-1).tolist()
    assert flat_inputs[:5] == [3, 4, 5, 6, 7]
    assert flat_targets[:5] == [4, 5, 6, 7, 8]
    assert batch["loss_mask"].reshape(-1).tolist() == [
        True,
        True,
        True,
        True,
        True,
        False,
        False,
        False,
    ]


def test_real_trainer_ledger_uses_loss_mask_valid_pair_count_exactly() -> None:
    torch.manual_seed(7)
    spec = controlled_specs()[0]
    config = _trainer_config(
        final_tokens=17,
        batch_size=2,
        sequence_length=4,
        seed=7,
    )
    model = TwelveSixDecoder(spec, InitSpec())
    trainer = Trainer(model, config, device="cpu")
    batch = _make_pair_batch(
        bytes(range(64)),
        causal_offset=0,
        batch_size=2,
        sequence_length=4,
        valid_pairs=5,
    )
    metrics = trainer.train_microbatch(batch)
    assert metrics.tokens == 5
    assert trainer.tokens_seen == 5
    assert trainer.optimizer_step == 1


def test_token_transition_fails_closed_on_any_drift() -> None:
    _assert_token_transition(before=11, metrics_tokens=7, after=18, requested=7)
    with pytest.raises(RuntimeError, match="valid-causal-token count drift"):
        _assert_token_transition(before=11, metrics_tokens=6, after=18, requested=7)
    with pytest.raises(RuntimeError, match="optimized-token ledger drift"):
        _assert_token_transition(before=11, metrics_tokens=7, after=19, requested=7)


def test_model_evidence_validation_rejects_requested_vs_optimized_drift(
    tmp_path: Path,
) -> None:
    checkpoints = [
        {
            "requested_token_budget": budget,
            "optimized_tokens": budget,
            "evaluation_optimized_tokens": 0,
        }
        for budget in DEFAULT_TOKEN_BUDGETS
    ]
    payload = {
        "schema": MODEL_SCHEMA,
        "source_sha": "a" * 40,
        "checkpoints": checkpoints,
        "resume": {"fresh_process": True},
    }
    payload["report_sha256"] = _canonical_hash(payload)
    path = tmp_path / "model.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    validate_evidence(path, expected_source_sha="a" * 40)

    payload["checkpoints"][1]["optimized_tokens"] += 1
    payload["report_sha256"] = _canonical_hash(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="exact common token budgets"):
        validate_evidence(path, expected_source_sha="a" * 40)


def test_model_evidence_validation_rejects_evaluation_token_leak(tmp_path: Path) -> None:
    checkpoints = [
        {
            "requested_token_budget": budget,
            "optimized_tokens": budget,
            "evaluation_optimized_tokens": 0,
        }
        for budget in DEFAULT_TOKEN_BUDGETS
    ]
    checkpoints[-1]["evaluation_optimized_tokens"] = 1
    payload = {
        "schema": MODEL_SCHEMA,
        "source_sha": "b" * 40,
        "checkpoints": checkpoints,
        "resume": {"fresh_process": True},
    }
    payload["report_sha256"] = _canonical_hash(payload)
    path = tmp_path / "model.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="evaluation tokens"):
        validate_evidence(path, expected_source_sha="b" * 40)
