from twelve_six.evaluation import stable_text_sha256
from twelve_six.stage_gates import evaluate_s0_integrated


def complete_integrated_evidence() -> dict:
    return {
        "candidate": {
            "sha": "b" * 40,
            "id": "integrated-synthetic-fixture",
            "random_init": True,
            "model_constructed": True,
            "parameter_count": 10_140,
            "model_vocab_size": 256,
        },
        "tokenizer": {
            "identity": "s0-byte-v1@synthetic",
            "vocab_size": 256,
            "max_token_id": 255,
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


def add_bound_promotion_authority(evidence: dict) -> None:
    candidate_sha = evidence["candidate"]["sha"]
    evidence["candidate"]["integrated"] = True
    evidence["promotion"] = {
        "candidate_manifest_validated": True,
        "candidate_manifest_sha256": "c" * 64,
        "candidate_ci": {"success": True, "run_id": 123456789},
        "audit_a": {
            "verdict": "PASS",
            "candidate_sha": candidate_sha,
            "evidence_ref": "issue-13#candidate-pass",
        },
        "audit_b": {
            "verdict": "PASS",
            "candidate_sha": candidate_sha,
            "evidence_ref": "issue-14#candidate-pass",
        },
    }


def gate(result: dict, gate_id: str) -> dict:
    return next(item for item in result["gates"] if item["gate_id"] == gate_id)


def test_integrated_fixture_passes_fifteen_eval_gates_but_is_not_promotable():
    result = evaluate_s0_integrated(complete_integrated_evidence())

    assert result["schema_version"] == "12-6.integrated-stage-gate-result.v2"
    assert result["summary"]["evaluation_complete"] is True
    assert result["summary"]["promotion_eligible"] is False
    assert result["summary"]["promotion_authority_status"] == "NOT_TESTED"
    assert result["summary"]["counts"] == {"PASS": 15, "FAIL": 0, "NOT_TESTED": 0}
    assert gate(result, "s0.tokenizer_model_vocab")["status"] == "PASS"


def test_bound_candidate_ci_manifest_and_dual_audits_allow_promotion_eligibility():
    evidence = complete_integrated_evidence()
    add_bound_promotion_authority(evidence)

    result = evaluate_s0_integrated(evidence)

    assert result["summary"]["evaluation_complete"] is True
    assert result["summary"]["promotion_authority_status"] == "PASS"
    assert result["summary"]["promotion_eligible"] is True
    assert result["promotion_authority"]["blockers"] == []


def test_stale_audit_sha_blocks_promotion_without_changing_eval_result():
    evidence = complete_integrated_evidence()
    add_bound_promotion_authority(evidence)
    evidence["promotion"]["audit_b"]["candidate_sha"] = "d" * 40

    result = evaluate_s0_integrated(evidence)

    assert result["summary"]["evaluation_complete"] is True
    assert result["summary"]["overall_status"] == "PASS"
    assert result["summary"]["promotion_authority_status"] == "FAIL"
    assert result["summary"]["promotion_eligible"] is False
    assert any("audit_b.candidate_sha does not match" in item for item in result["promotion_authority"]["blockers"])


def test_candidate_sha_must_be_exact_git_object_id():
    evidence = complete_integrated_evidence()
    evidence["candidate"]["sha"] = "short-sha"

    result = evaluate_s0_integrated(evidence)

    assert gate(result, "s0.identity")["status"] == "FAIL"
    assert result["summary"]["evaluation_complete"] is False
    assert result["summary"]["promotion_eligible"] is False


def test_d01_d04_vocab_mismatch_fails():
    evidence = complete_integrated_evidence()
    evidence["tokenizer"]["vocab_size"] = 259
    evidence["tokenizer"]["max_token_id"] = 258

    result = evaluate_s0_integrated(evidence)
    compatibility = gate(result, "s0.tokenizer_model_vocab")

    assert compatibility["status"] == "FAIL"
    assert compatibility["evidence"]["model_vocab_size"] == 256
    assert compatibility["evidence"]["tokenizer_vocab_size"] == 259
    assert compatibility["evidence"]["max_token_id"] == 258
    assert result["summary"]["evaluation_complete"] is False
    assert result["summary"]["promotion_eligible"] is False
    assert result["summary"]["overall_status"] == "FAIL"


def test_max_token_id_must_fit_even_when_vocab_sizes_match():
    evidence = complete_integrated_evidence()
    evidence["tokenizer"]["max_token_id"] = 256

    result = evaluate_s0_integrated(evidence)

    assert gate(result, "s0.tokenizer_model_vocab")["status"] == "FAIL"


def test_missing_tokenizer_evidence_is_not_tested():
    evidence = complete_integrated_evidence()
    evidence.pop("tokenizer")

    result = evaluate_s0_integrated(evidence)

    assert gate(result, "s0.tokenizer_model_vocab")["status"] == "NOT_TESTED"
    assert result["summary"]["evaluation_complete"] is False
    assert result["summary"]["promotion_eligible"] is False
