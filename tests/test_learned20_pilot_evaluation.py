from __future__ import annotations

from copy import deepcopy

from twelve_six.learned20_pilot_evaluation import validate_terminal_pilot_evaluation


def _evidence() -> dict:
    pilot = {
        "terminal": True,
        "identity": "pilot-v1",
        "evaluation_firewall_identity": "eval-firewall-v1",
        "result_checkpoint_identity": "pilot-checkpoint-v1",
    }
    d06 = {
        "pilot_identity": pilot["identity"],
        "evaluation_firewall_identity": pilot["evaluation_firewall_identity"],
        "random_init_baseline_identity": "random-init-model341-v1",
        "heldout_metrics": {
            "UA": {
                "target_count": 100,
                "random_init_mean_nll": 5.8,
                "trained_mean_nll": 5.2,
            },
            "EN": {
                "target_count": 200,
                "random_init_mean_nll": 5.7,
                "trained_mean_nll": 5.1,
            },
            "CODE": {
                "target_count": 100,
                "random_init_mean_nll": 5.9,
                "trained_mean_nll": 5.4,
            },
        },
        "weighted_random_init_mean_nll": 5.775,
        "weighted_trained_mean_nll": 5.2,
        "selection_trajectory": [
            {"optimizer_step": 0, "optimized_target_exposure": 0, "mean_nll": 5.775},
            {"optimizer_step": 10, "optimized_target_exposure": 40960, "mean_nll": 5.35},
            {"optimizer_step": 20, "optimized_target_exposure": 81920, "mean_nll": 5.2},
        ],
        "inference_probe": {
            "prompt_suite_identity": "d06-pilot-probes-v1",
            "output_fingerprint": "sha256:" + "a" * 64,
            "fresh_process_reload": True,
            "checkpoint_identity": pilot["result_checkpoint_identity"],
        },
        "memorization_diagnostic": {
            "policy_identity": "d06-memorization-v1",
            "training_exact_match_rate": 0.03,
            "heldout_exact_match_rate": 0.0,
            "passed": True,
        },
        "throughput_optimized_targets_per_second": 1250.5,
        "peak_memory_bytes": 1_500_000_000,
    }
    pilot["d06_evaluation"] = d06
    return {"bounded_pilot": pilot}


def test_terminal_pilot_requires_numeric_d06_scientific_evidence() -> None:
    assert validate_terminal_pilot_evaluation(_evidence()) == []


def test_terminal_pilot_cannot_replace_numeric_evidence_with_booleans() -> None:
    evidence = _evidence()
    evidence["bounded_pilot"]["d06_evaluation"] = {
        "pilot_identity": "pilot-v1",
        "evaluation_firewall_identity": "eval-firewall-v1",
    }
    blockers = validate_terminal_pilot_evaluation(evidence)
    assert "bounded_pilot.d06.heldout_metrics_missing" in blockers
    assert "bounded_pilot.d06.selection_trajectory_insufficient" in blockers
    assert "bounded_pilot.d06.inference_probe_missing" in blockers
    assert "bounded_pilot.d06.memorization_diagnostic_missing" in blockers


def test_weighted_heldout_metrics_are_recomputed_and_must_improve() -> None:
    evidence = _evidence()
    evidence["bounded_pilot"]["d06_evaluation"]["weighted_trained_mean_nll"] = 1.0
    blockers = validate_terminal_pilot_evaluation(evidence)
    assert "bounded_pilot.d06.weighted_trained_mean_nll_mismatch" in blockers

    evidence = _evidence()
    for metric in evidence["bounded_pilot"]["d06_evaluation"]["heldout_metrics"].values():
        metric["trained_mean_nll"] = metric["random_init_mean_nll"] + 0.1
    evidence["bounded_pilot"]["d06_evaluation"]["weighted_trained_mean_nll"] = 5.875
    blockers = validate_terminal_pilot_evaluation(evidence)
    assert "bounded_pilot.d06.heldout_not_better_than_random_init" in blockers


def test_all_three_heldout_strata_are_mandatory() -> None:
    evidence = _evidence()
    del evidence["bounded_pilot"]["d06_evaluation"]["heldout_metrics"]["UA"]
    blockers = validate_terminal_pilot_evaluation(evidence)
    assert "bounded_pilot.d06.heldout_metrics.UA_missing" in blockers


def test_selection_trajectory_must_be_monotonic_and_improve() -> None:
    evidence = _evidence()
    trajectory = evidence["bounded_pilot"]["d06_evaluation"]["selection_trajectory"]
    trajectory[2]["optimizer_step"] = trajectory[1]["optimizer_step"]
    trajectory[2]["mean_nll"] = trajectory[0]["mean_nll"]
    blockers = validate_terminal_pilot_evaluation(evidence)
    assert "bounded_pilot.d06.selection_trajectory_steps_not_strict" in blockers
    assert "bounded_pilot.d06.selection_trajectory_not_improving" in blockers


def test_inference_probe_must_reload_exact_result_checkpoint() -> None:
    evidence = _evidence()
    probe = evidence["bounded_pilot"]["d06_evaluation"]["inference_probe"]
    probe["checkpoint_identity"] = "stale-checkpoint"
    probe["fresh_process_reload"] = False
    blockers = validate_terminal_pilot_evaluation(evidence)
    assert "bounded_pilot.d06.inference_probe.checkpoint_identity_mismatch" in blockers
    assert "bounded_pilot.d06.inference_probe.fresh_process_reload_not_proven" in blockers


def test_memorization_diagnostic_is_fail_closed() -> None:
    evidence = _evidence()
    diagnostic = evidence["bounded_pilot"]["d06_evaluation"]["memorization_diagnostic"]
    diagnostic["training_exact_match_rate"] = 1.2
    diagnostic["passed"] = False
    blockers = validate_terminal_pilot_evaluation(evidence)
    assert (
        "bounded_pilot.d06.memorization_diagnostic.training_exact_match_rate_invalid"
        in blockers
    )
    assert "bounded_pilot.d06.memorization_diagnostic_not_passed" in blockers


def test_nonterminal_pilot_does_not_claim_d06_terminal_evidence() -> None:
    evidence = _evidence()
    evidence["bounded_pilot"]["terminal"] = False
    del evidence["bounded_pilot"]["d06_evaluation"]
    assert validate_terminal_pilot_evaluation(evidence) == []


def test_pilot_and_firewall_identity_drift_is_rejected() -> None:
    evidence = deepcopy(_evidence())
    d06 = evidence["bounded_pilot"]["d06_evaluation"]
    d06["pilot_identity"] = "other-pilot"
    d06["evaluation_firewall_identity"] = "other-firewall"
    blockers = validate_terminal_pilot_evaluation(evidence)
    assert "bounded_pilot.d06.pilot_identity_mismatch" in blockers
    assert "bounded_pilot.d06.evaluation_firewall_identity_mismatch" in blockers
