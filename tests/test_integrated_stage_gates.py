from twelve_six.evaluation import stable_text_sha256
from twelve_six.stage_gates import evaluate_s0_integrated


def complete_integrated_evidence() -> dict:
    return {
        "candidate": {
            "sha": "b" * 40,
            "id": "integrated-synthetic-fixture",
            "random_init": True,
            "model_constructed": True,
            "parameter_count": 10_200,
            "model_vocab_size": 259,
        },
        "tokenizer": {
            "identity": "s0-byte-v1@synthetic",
            "vocab_size": 259,
            "max_token_id": 258,
        },
        "eval_config": {"id": "s0-integrated-eval-v1"},
        "dataset": {
            "identity": "synthetic-integrated-fixture",
            "heldout_used_for_training": False,
            "train_validation_overlap": 0,
            "validation_examples": 2,
            "distinct_train_batches": 3,
        },
        "metrics": {
            "train_loss_before": 5.0,
            "train_loss_after": 3.0,
            "validation_loss_before": 5.1,
            "validation_loss_after": 3.8,
            "random_validation_loss": 5.1,
            "trained_validation_loss": 3.8,
        },
        "generation_probes": [
            {
                "id": "probe",
                "token_count": 4,
                "output_sha256": stable_text_sha256("test"),
                "seed": 7,
                "sampler": "greedy",
            }
        ],
        "checkpoint": {"save_load_verified": True, "resume_verified": True},
        "contamination": {
            "checked": True,
            "benchmark_overlap_count": 0,
            "heldout_overlap_count": 0,
        },
        "regressions": {"executed": True, "failures": 0},
    }


def gate(result: dict, gate_id: str) -> dict:
    return next(item for item in result["gates"] if item["gate_id"] == gate_id)


def test_integrated_fixture_passes_fifteen_gates():
    result = evaluate_s0_integrated(complete_integrated_evidence())

    assert result["schema_version"] == "12-6.integrated-stage-gate-result.v1"
    assert result["summary"]["promotion_eligible"] is True
    assert result["summary"]["counts"] == {"PASS": 15, "FAIL": 0, "NOT_TESTED": 0}
    assert gate(result, "s0.tokenizer_model_vocab")["status"] == "PASS"


def test_d01_d04_current_vocab_mismatch_fails():
    evidence = complete_integrated_evidence()
    evidence["candidate"]["model_vocab_size"] = 256
    evidence["candidate"]["parameter_count"] = 10_140

    result = evaluate_s0_integrated(evidence)
    compatibility = gate(result, "s0.tokenizer_model_vocab")

    assert compatibility["status"] == "FAIL"
    assert compatibility["evidence"]["model_vocab_size"] == 256
    assert compatibility["evidence"]["tokenizer_vocab_size"] == 259
    assert compatibility["evidence"]["max_token_id"] == 258
    assert result["summary"]["promotion_eligible"] is False
    assert result["summary"]["overall_status"] == "FAIL"


def test_max_token_id_must_fit_even_when_vocab_sizes_match():
    evidence = complete_integrated_evidence()
    evidence["tokenizer"]["max_token_id"] = 259

    result = evaluate_s0_integrated(evidence)

    assert gate(result, "s0.tokenizer_model_vocab")["status"] == "FAIL"


def test_missing_tokenizer_evidence_is_not_tested():
    evidence = complete_integrated_evidence()
    evidence.pop("tokenizer")

    result = evaluate_s0_integrated(evidence)

    assert gate(result, "s0.tokenizer_model_vocab")["status"] == "NOT_TESTED"
    assert result["summary"]["promotion_eligible"] is False
