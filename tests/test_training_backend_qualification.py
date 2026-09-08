from __future__ import annotations

import copy

from twelve_six.training.backend_qualification import (
    REFERENCE_BACKEND,
    REFERENCE_ROLE,
    SCHEMA,
    assess_backend_qualification,
    run_project_native_benchmark,
    run_project_native_resume_probe,
    stable_report_sha256,
)

SHA40 = "a" * 40
SHA64 = "b" * 64


def _authority() -> dict:
    return {
        "repository": "Oleksii-debug/12-6-ai.",
        "git_sha": SHA40,
        "evidence_sha256": SHA64,
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
        "parity_authorities": {key: _authority() for key in parity},
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
            "authority": _authority(),
        },
        "fresh_process_recovery": {
            "proven": True,
            "authority": _authority(),
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


def test_project_native_resume_probe_is_real_and_equivalent() -> None:
    probe = run_project_native_resume_probe(seed=126)
    assert probe["backend_id"] == REFERENCE_BACKEND
    assert probe["optimizer_steps"] == 2
    assert probe["optimized_targets"] == 8
    assert probe["checkpoint_state_bytes"] > 0
    assert probe["elapsed_seconds"] > 0
    assert probe["finite_loss"] is True
    assert probe["resume_equivalent"] is True
    assert probe["fresh_instance_only"] is True
    assert probe["model_is_test_fixture_only"] is True


def test_project_native_benchmark_measures_bounded_cpu_workload() -> None:
    benchmark = run_project_native_benchmark(steps=4, seed=126)
    assert benchmark["backend_id"] == REFERENCE_BACKEND
    assert benchmark["device"] == "cpu"
    assert benchmark["steps"] == 4
    assert benchmark["optimized_targets"] == 16
    assert benchmark["measurement_method"] == "PROCESS_MAX_RSS_RUSAGE"
    assert benchmark["tokens_per_second"] > 0
    assert benchmark["step_time_seconds"] > 0
    assert benchmark["peak_ram_bytes"] > 0
    assert benchmark["model_is_test_fixture_only"] is True
    assert benchmark["model_scale_throughput_claimed"] is False


def test_benchmark_rejects_invalid_step_count() -> None:
    for value in (0, -1, True):
        try:
            run_project_native_benchmark(steps=value)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid benchmark step count must fail closed")


def test_complete_reference_report_can_reach_candidate_comparison_gate() -> None:
    result = assess_backend_qualification(_complete_report())
    assert result.contract_valid
    assert result.mechanics_reference_proven
    assert result.fresh_process_recovery_proven
    assert result.benchmark_complete
    assert result.ready_for_candidate_parity_comparison
    assert result.blockers == ()


def test_inprocess_resume_cannot_impersonate_fresh_process_authority() -> None:
    report = _complete_report()
    report["fresh_process_recovery"] = {"proven": False, "authority": None}
    _rehash(report)
    result = assess_backend_qualification(report)
    assert result.mechanics_reference_proven
    assert not result.fresh_process_recovery_proven
    assert not result.ready_for_candidate_parity_comparison
    assert "fresh_process_recovery_authority_missing" in result.blockers


def test_missing_real_benchmark_fails_closed_without_erasing_mechanics() -> None:
    report = _complete_report()
    report["bounded_benchmark"] = {
        "tokens_per_second": None,
        "step_time_seconds": None,
        "peak_ram_bytes": None,
        "measurement_method": None,
        "authority": None,
    }
    _rehash(report)
    result = assess_backend_qualification(report)
    assert result.mechanics_reference_proven
    assert not result.benchmark_complete
    assert not result.ready_for_candidate_parity_comparison
    assert "bounded_throughput_memory_benchmark_missing" in result.blockers
    assert "bounded_benchmark_scope_mismatch" in result.blockers


def test_cpu_fixture_benchmark_cannot_impersonate_gpu_or_model_scale_evidence() -> None:
    for key, value in (
        ("device", "cuda"),
        ("model_is_test_fixture_only", False),
        ("model_scale_throughput_claimed", True),
        ("exact_version", "different-runtime"),
    ):
        report = _complete_report()
        report["bounded_benchmark"][key] = value
        _rehash(report)
        result = assess_backend_qualification(report)
        assert result.mechanics_reference_proven
        assert not result.benchmark_complete
        assert not result.ready_for_candidate_parity_comparison
        assert "bounded_benchmark_scope_mismatch" in result.blockers


def test_optimizer_semantics_drift_invalidates_reference_mechanics() -> None:
    report = _complete_report()
    report["parity"]["optimizer_semantics_parity"] = False
    _rehash(report)
    result = assess_backend_qualification(report)
    assert not result.mechanics_reference_proven
    assert not result.ready_for_candidate_parity_comparison
    assert "optimizer_semantics_parity_not_proven" in result.blockers


def test_nonterminal_parity_authority_invalidates_mechanics() -> None:
    report = _complete_report()
    authority = copy.deepcopy(report["parity_authorities"]["optimizer_semantics_parity"])
    authority["workflow_conclusion"] = "failure"
    report["parity_authorities"]["optimizer_semantics_parity"] = authority
    _rehash(report)
    result = assess_backend_qualification(report)
    assert not result.mechanics_reference_proven
    assert not result.ready_for_candidate_parity_comparison
    assert "optimizer_semantics_parity_authority_missing" in result.blockers


def test_report_identity_is_immutable() -> None:
    report = _complete_report()
    tampered = copy.deepcopy(report)
    tampered["runtime_probe"]["optimized_targets"] += 1
    result = assess_backend_qualification(tampered)
    assert not result.contract_valid
    assert "report_identity_mismatch" in result.blockers


def test_truth_boundary_cannot_self_authorize_backend_or_training() -> None:
    for key in (
        "paid_compute_authorized",
        "material_training_authorized",
        "backend_promotion_authorized",
        "stage_promotion_authorized",
    ):
        report = _complete_report()
        report["truth_boundary"][key] = True
        _rehash(report)
        result = assess_backend_qualification(report)
        assert not result.mechanics_reference_proven
        assert f"truth_boundary_{key}_must_be_false" in result.blockers
