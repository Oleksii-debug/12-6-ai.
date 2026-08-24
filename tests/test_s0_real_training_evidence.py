from __future__ import annotations

import math
from pathlib import Path

from twelve_six.training.s0_evidence import run_s0_training_evidence

ROOT = Path(__file__).resolve().parents[1]
BASE_INTEGRATION_SHA = "c1e37854829faa96291ee76088f703f5096ea10b"


def test_real_s0_training_uses_train_only_and_emits_finite_learning_evidence() -> None:
    evidence = run_s0_training_evidence(
        ROOT,
        source_sha=BASE_INTEGRATION_SHA,
        max_steps=3,
        batch_size=3,
    )

    assert evidence["identity"]["parameter_count"] == 10_140
    assert evidence["seed_ordering"]["seed_applied_before_model_construction"] is True
    assert evidence["split_isolation"]["optimized_split"] == "train"
    assert evidence["split_isolation"]["record_id_overlap"] == []
    assert evidence["split_isolation"]["validation_optimized_tokens"] == 0
    assert evidence["split_isolation"]["validation_scoreable_tokens"] == 404
    assert evidence["split_isolation"]["train_tokens_per_full_epoch"] == 1_910

    training = evidence["training"]
    assert training["optimizer_steps"] == 3
    assert training["optimized_tokens"] == training["trainer_tokens_seen"] > 0
    assert training["final_train_loss"] < training["initial_train_loss"]
    assert math.isfinite(training["initial_validation_loss"])
    assert math.isfinite(training["final_validation_loss"])
    assert math.isfinite(training["gradient_norm_min"])
    assert math.isfinite(training["gradient_norm_max"])
    assert training["weight_delta"]["changed_parameter_elements"] > 0
    assert training["weight_delta"]["trainable_parameter_elements"] == 10_140

    assert evidence["failure_semantics"] == {
        "nan_fail_closed_and_fresh_recovery": True,
        "inf_fail_closed_and_fresh_recovery": True,
    }
    assert evidence["claims"]["candidate_or_stable_promotion"] is False
    assert evidence["claims"]["foreign_pretrained_weights_used"] is False
