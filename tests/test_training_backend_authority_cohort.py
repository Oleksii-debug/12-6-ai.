from __future__ import annotations

from twelve_six.training.backend_qualification import (
    REFERENCE_BACKEND,
    REFERENCE_ROLE,
    SCHEMA,
    assess_backend_qualification,
    run_project_native_resume_probe,
    stable_report_sha256,
)

SHA_A = "a" * 40
SHA_C = "c" * 40
SHA256 = "b" * 64


def _authority(claim: str, *, git_sha: str = SHA_A) -> dict:
    return {
        "claim": claim,
        "repository": "Oleksii-debug/12-6-ai.",
        "git_sha": git_sha,
        "evidence_sha256": SHA256,
        "workflow_run_id": 123,
        "workflow_conclusion": "success",
        "terminal": True,
    }


def _complete_report() -> dict:
    runtime = run_project_native_resume_probe()
    parity = {
        "modelspec_parity": True,
        "optimizer_semantics_parity": True,
        "causal_objective_parity": True,
        "valid_target_accounting_parity": True,
        "gradient_accumulation_parity": True,
        "precision_semantics_parity": True,
        "checkpoint_save_load_resume": True,
        "fresh_instance_recovery": True,
        "determinism_same_seed": True,
        "rollback_path": True,
    }
    report = {
        "schema": SCHEMA,
        "backend_id": REFERENCE_BACKEND,
        "role": REFERENCE_ROLE,
        "exact_version": runtime["exact_version"],
        "license": "BSD-3-Clause",
        "parity": parity,
        "parity_authorities": {key: _authority(key) for key in parity},
        "runtime_probe": runtime,
        "bounded_benchmark": {
            "backend_id": REFERENCE_BACKEND,
            "exact_version": runtime["exact_version"],
            "device": "cpu",
            "tokens_per_second": 1.0,
            "step_time_seconds": 1.0,
            "peak_ram_bytes": 1,
            "measurement_method": "TEST_TERMINAL_MEASUREMENT_FIXTURE",
            "model_is_test_fixture_only": True,
            "model_scale_throughput_claimed": False,
            "authority": _authority("bounded_benchmark"),
        },
        "fresh_process_recovery": {
            "backend_id": REFERENCE_BACKEND,
            "exact_version": runtime["exact_version"],
            "device": "cpu",
            "fresh_process_only": True,
            "model_is_test_fixture_only": True,
            "proven": True,
            "authority": _authority("fresh_process_recovery"),
        },
        "truth_boundary": {
            "canonical_base_random_init_only": True,
            "foreign_pretrained_weights_used": False,
            "teacher_logits_used": False,
            "evaluation_or_final_test_used_for_training": False,
            "paid_compute_authorized": False,
            "material_training_authorized": False,
            "backend_promotion_authorized": False,
            "stage_promotion_authorized": False,
        },
    }
    report["report_sha256"] = stable_report_sha256(report)
    return report


def _rehash(report: dict) -> None:
    report.pop("report_sha256", None)
    report["report_sha256"] = stable_report_sha256(report)


def _assert_cohort_mismatch_fails_closed(report: dict) -> None:
    _rehash(report)
    result = assess_backend_qualification(report)
    assert not result.contract_valid
    assert not result.mechanics_reference_proven
    assert not result.ready_for_candidate_parity_comparison
    assert "authority_git_sha_cohort_mismatch" in result.blockers


def test_parity_authority_from_different_head_breaks_cohort() -> None:
    report = _complete_report()
    report["parity_authorities"]["optimizer_semantics_parity"]["git_sha"] = SHA_C
    _assert_cohort_mismatch_fails_closed(report)


def test_benchmark_authority_from_different_head_breaks_cohort() -> None:
    report = _complete_report()
    report["bounded_benchmark"]["authority"]["git_sha"] = SHA_C
    _assert_cohort_mismatch_fails_closed(report)


def test_recovery_authority_from_different_head_breaks_cohort() -> None:
    report = _complete_report()
    report["fresh_process_recovery"]["authority"]["git_sha"] = SHA_C
    _assert_cohort_mismatch_fails_closed(report)
