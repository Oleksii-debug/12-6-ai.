from __future__ import annotations

from twelve_six import recover170_train41 as r170


def _point(tokens: int, heldout_bpb: float, train_bpb: float) -> dict[str, object]:
    return {
        "actual_optimized_tokens": tokens,
        "optimizer_step": tokens // 100,
        "training_bpb_since_previous_eval": train_bpb,
        "heldout": {"bits_per_byte": heldout_bpb},
    }


def test_recovery_preserves_train41_model_and_budget() -> None:
    spec, init = r170._model_truth()
    assert spec.parameter_count() == 95_568
    assert spec.identity_sha256() == r170.EXPECTED_MODEL_SPEC_SHA256
    assert init.identity_sha256() == r170.EXPECTED_INIT_SPEC_SHA256
    assert r170.BATCH_SIZE == 4
    assert r170.SEQUENCE_LENGTH == 64
    assert r170.FINAL_TOKENS == 2_097_152
    assert r170.RESUME_TOKENS == 1_048_576


def test_json_normalization_removes_tuple_list_process_drift() -> None:
    left = {"optimizer": {"betas": (0.9, 0.95)}}
    right = {"optimizer": {"betas": [0.9, 0.95]}}
    assert r170._json_normalize(left) == r170._json_normalize(right)
    assert r170._self_hashed(left) == r170._self_hashed(right)


def test_overfit_proxy_requires_training_improvement_and_heldout_rise() -> None:
    points = [
        _point(1_000, 4.0, 4.1),
        _point(2_000, 3.8, 3.9),
        _point(4_000, 3.82, 3.7),
    ]
    analysis = r170._overfit_analysis(points)
    assert analysis["status"] == "OVERFIT_PROXY_ONSET_DETECTED"
    assert analysis["onset"]["actual_optimized_tokens"] == 4_000


def test_overfit_proxy_does_not_call_flat_curve_overfit() -> None:
    points = [
        _point(1_000, 4.0, 4.1),
        _point(2_000, 3.9, 3.9),
        _point(4_000, 3.89, 3.8),
    ]
    analysis = r170._overfit_analysis(points)
    assert analysis["status"] == "NO_PROXY_ONSET_DETECTED"
