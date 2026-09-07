import json
from pathlib import Path

import pytest

from twelve_six.evaluation import (
    BenchmarkRegistry,
    BenchmarkSpec,
    GateStatus,
    S0GatePolicy,
    dump_stage_gate_result,
    evaluate_s0,
    perplexity_from_nll,
    stable_text_sha256,
)


def complete_evidence() -> dict:
    return {
        "candidate": {
            "sha": "a" * 40,
            "id": "s0-candidate-001",
            "random_init": True,
            "model_constructed": True,
            "parameter_count": 10_240,
        },
        "eval_config": {"id": "s0-eval-v1"},
        "dataset": {
            "identity": "tiny-corpus@sha256:abc",
            "heldout_used_for_training": False,
            "train_validation_overlap": 0,
            "validation_examples": 16,
            "distinct_train_batches": 4,
        },
        "metrics": {
            "train_loss_before": 5.0,
            "train_loss_after": 2.5,
            "validation_loss_before": 5.1,
            "validation_loss_after": 3.2,
            "random_validation_loss": 5.1,
            "trained_validation_loss": 3.2,
        },
        "generation_probes": [
            {
                "id": "fixed-prompt-1",
                "token_count": 8,
                "output_sha256": stable_text_sha256("abc"),
                "seed": 123,
                "sampler": "greedy",
            }
        ],
        "checkpoint": {
            "save_load_verified": True,
            "resume_verified": True,
        },
        "contamination": {
            "checked": True,
            "benchmark_overlap_count": 0,
            "heldout_overlap_count": 0,
        },
        "regressions": {"executed": True, "failures": 0},
    }


def test_complete_s0_evidence_passes_all_required_gates():
    result = evaluate_s0(complete_evidence())

    assert result["summary"]["overall_status"] == GateStatus.PASS.value
    assert result["summary"]["promotion_eligible"] is True
    assert result["summary"]["counts"]["FAIL"] == 0
    assert result["summary"]["counts"]["NOT_TESTED"] == 0
    assert result["derived_metrics"]["validation_loss_after_perplexity"] > 1


def test_missing_evidence_is_not_tested_and_blocks_promotion():
    result = evaluate_s0({"candidate": {"model_constructed": True}})

    assert result["summary"]["promotion_eligible"] is False
    assert result["summary"]["overall_status"] == GateStatus.NOT_TESTED.value
    assert result["summary"]["counts"]["NOT_TESTED"] > 0


def test_present_bad_baseline_is_fail_not_not_tested():
    evidence = complete_evidence()
    evidence["metrics"]["trained_validation_loss"] = evidence["metrics"]["random_validation_loss"]

    result = evaluate_s0(evidence)
    gate = next(item for item in result["gates"] if item["gate_id"] == "s0.random_vs_trained")

    assert gate["status"] == "FAIL"
    assert result["summary"]["overall_status"] == "FAIL"
    assert result["summary"]["promotion_eligible"] is False


def test_heldout_training_use_fails():
    evidence = complete_evidence()
    evidence["dataset"]["heldout_used_for_training"] = True

    result = evaluate_s0(evidence)
    gate = next(item for item in result["gates"] if item["gate_id"] == "s0.heldout_integrity")

    assert gate["status"] == "FAIL"


def test_single_fixed_batch_fails_memorization_control():
    evidence = complete_evidence()
    evidence["dataset"]["distinct_train_batches"] = 1

    result = evaluate_s0(evidence)
    gate = next(item for item in result["gates"] if item["gate_id"] == "s0.not_single_fixed_batch")

    assert gate["status"] == "FAIL"


def test_parameter_range_is_explicit_and_configurable():
    evidence = complete_evidence()
    evidence["candidate"]["parameter_count"] = 12_500

    default_result = evaluate_s0(evidence)
    default_gate = next(
        item for item in default_result["gates"] if item["gate_id"] == "s0.parameter_range"
    )
    assert default_gate["status"] == "FAIL"

    policy = S0GatePolicy(max_parameters=13_000)
    relaxed_result = evaluate_s0(evidence, policy)
    relaxed_gate = next(
        item for item in relaxed_result["gates"] if item["gate_id"] == "s0.parameter_range"
    )
    assert relaxed_gate["status"] == "PASS"


def test_perplexity_only_accepts_finite_nonnegative_nll():
    assert perplexity_from_nll(0.0) == 1.0
    assert perplexity_from_nll(1.0) == pytest.approx(2.718281828459045)

    with pytest.raises(ValueError):
        perplexity_from_nll(-0.1)
    with pytest.raises(ValueError):
        perplexity_from_nll(float("nan"))
    with pytest.raises(TypeError):
        perplexity_from_nll(True)


def test_benchmark_registry_rejects_heldout_training_use():
    with pytest.raises(ValueError, match="held-out benchmark"):
        BenchmarkSpec(
            benchmark_id="bench",
            version="1",
            source_id="source-1",
            held_out=True,
            allowed_uses=("evaluation", "pretrain"),
        )


def test_benchmark_registry_detects_training_source_collision_and_hashes_manifest():
    registry = BenchmarkRegistry(
        [
            BenchmarkSpec(
                benchmark_id="bench",
                version="1",
                source_id="source-1",
                allowed_uses=("evaluation",),
            )
        ]
    )

    collisions = registry.training_collisions(["source-1", "training-only"])
    manifest = registry.manifest()

    assert collisions == [{"benchmark_key": "bench@1", "source_id": "source-1"}]
    assert len(manifest["manifest_sha256"]) == 64
    assert manifest["benchmarks"][0]["benchmark_id"] == "bench"


def test_stage_gate_json_is_deterministic(tmp_path: Path):
    result = evaluate_s0(complete_evidence())
    first = tmp_path / "one.json"
    second = tmp_path / "two.json"

    dump_stage_gate_result(result, first)
    dump_stage_gate_result(result, second)

    assert first.read_bytes() == second.read_bytes()
    loaded = json.loads(first.read_text(encoding="utf-8"))
    assert loaded["schema_version"] == "12-6.stage-gate-result.v1"


def test_unknown_policy_fields_fail_closed():
    with pytest.raises(ValueError, match="unknown policy fields"):
        S0GatePolicy.from_mapping({"stage": "S0", "surprise": 1})
