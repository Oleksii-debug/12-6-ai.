from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any

from twelve_six.learned20_pilot_evaluation import (
    MEMORIZATION_POLICY_V1_IDENTITY,
    validate_terminal_pilot_evaluation as _validate_terminal_pilot_evaluation,
)


def _sha256_identity(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _reseal_fresh_process_receipt(receipt: dict[str, Any]) -> None:
    receipt["checkpoint_provenance_identity"] = _sha256_identity(
        {
            key: receipt.get(key)
            for key in (
                "checkpoint_identity",
                "git_sha",
                "model_spec_sha256",
                "tokenizer_config_sha256",
                "tokenizer_vocab_sha256",
                "dataset_manifest_sha256",
                "run_manifest_sha256",
                "optimizer_step",
                "tokens_seen",
            )
        }
    )
    receipt["process_run_identity"] = _sha256_identity(
        {
            "checkpoint_identity": receipt.get("checkpoint_identity"),
            "prompt_suite_identity": receipt.get("prompt_suite_identity"),
            "prompt_payload_sha256": receipt.get("prompt_payload_sha256"),
            "generation_config_identity": receipt.get("generation_config_identity"),
            "output_fingerprint": receipt.get("output_fingerprint"),
            "child_pid": receipt.get("child_pid"),
            "parent_pid": receipt.get("parent_pid"),
            "challenge_sha256": receipt.get("challenge_sha256"),
        }
    )
    receipt["receipt_identity"] = _sha256_identity(
        {
            key: receipt.get(key)
            for key in sorted(set(receipt) - {"receipt_identity"})
        }
    )


def _fresh_process_receipt(checkpoint_identity: str) -> dict[str, Any]:
    receipt: dict[str, Any] = {
        "schema_version": "d07-fresh-process-inference-v1",
        "producer_identity": "twelve_six.inference.fresh_process",
        "checkpoint_identity": checkpoint_identity,
        "checkpoint_provenance_identity": "",
        "git_sha": "1" * 40,
        "model_spec_sha256": "2" * 64,
        "tokenizer_config_sha256": "3" * 64,
        "tokenizer_vocab_sha256": "4" * 64,
        "dataset_manifest_sha256": "5" * 64,
        "run_manifest_sha256": "6" * 64,
        "optimizer_step": 10,
        "tokens_seen": 40960,
        "prompt_suite_identity": "d06-pilot-probes-v1",
        "prompt_payload_sha256": "sha256:" + "7" * 64,
        "generation_config_identity": "sha256:" + "8" * 64,
        "output_fingerprint": "sha256:" + "a" * 64,
        "process_run_identity": "",
        "child_pid": 2222,
        "parent_pid": 1111,
        "challenge_sha256": "sha256:" + "b" * 64,
        "receipt_identity": "",
    }
    _reseal_fresh_process_receipt(receipt)
    return receipt


def _fresh_process_expectations(receipt: dict[str, Any]) -> dict[str, Any]:
    return {
        "receipt_identity": receipt["receipt_identity"],
        "checkpoint_identity": receipt["checkpoint_identity"],
        "prompt_suite_identity": receipt["prompt_suite_identity"],
        "prompt_payload_sha256": receipt["prompt_payload_sha256"],
        "generation_config_identity": receipt["generation_config_identity"],
    }


def validate_terminal_pilot_evaluation(evidence: dict[str, Any]) -> list[str]:
    return _validate_terminal_pilot_evaluation(
        evidence,
        fresh_process_expectations=evidence.get("_test_fresh_process_expectations"),
    )


def _evidence() -> dict[str, Any]:
    checkpoint_identity = "c" * 64
    receipt = _fresh_process_receipt(checkpoint_identity)
    pilot: dict[str, Any] = {
        "terminal": True,
        "identity": "pilot-v1",
        "evaluation_firewall_identity": "eval-firewall-v1",
        "result_checkpoint_identity": checkpoint_identity,
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
                "scored_utf8_bytes": 200,
                "bpb_predicted_tokens": 100,
                "random_init_bpb_total_nll_nats": 580.0,
                "trained_bpb_total_nll_nats": 520.0,
                "random_init_bpb": 4.183815618577994,
                "trained_bpb": 3.7510071063113046,
            },
            "EN": {
                "target_count": 200,
                "random_init_mean_nll": 5.7,
                "trained_mean_nll": 5.1,
                "scored_utf8_bytes": 400,
                "bpb_predicted_tokens": 200,
                "random_init_bpb_total_nll_nats": 1140.0,
                "trained_bpb_total_nll_nats": 1020.0,
                "random_init_bpb": 4.111680866533546,
                "trained_bpb": 3.678872354266856,
            },
            "CODE": {
                "target_count": 100,
                "random_init_mean_nll": 5.9,
                "trained_mean_nll": 5.4,
                "scored_utf8_bytes": 200,
                "bpb_predicted_tokens": 100,
                "random_init_bpb_total_nll_nats": 590.0,
                "trained_bpb_total_nll_nats": 540.0,
                "random_init_bpb": 4.2559503706224415,
                "trained_bpb": 3.895276610400201,
            },
        },
        "weighted_random_init_mean_nll": 5.775,
        "weighted_trained_mean_nll": 5.2,
        "selection_trajectory": [
            {
                "event_identity": "selection-event-0",
                "checkpoint_identity": "random-init-checkpoint-v1",
                "optimizer_step": 0,
                "optimized_target_exposure": 0,
                "mean_nll": 5.775,
            },
            {
                "event_identity": "selection-event-10",
                "checkpoint_identity": pilot["result_checkpoint_identity"],
                "optimizer_step": 10,
                "optimized_target_exposure": 40960,
                "mean_nll": 5.35,
            },
            {
                "event_identity": "selection-event-20",
                "checkpoint_identity": "pilot-final-checkpoint-v1",
                "optimizer_step": 20,
                "optimized_target_exposure": 81920,
                "mean_nll": 5.2,
            },
        ],
        "inference_probe": {
            "prompt_suite_identity": "d06-pilot-probes-v1",
            "output_fingerprint": receipt["output_fingerprint"],
            "fresh_process_reload": True,
            "checkpoint_identity": pilot["result_checkpoint_identity"],
            "fresh_process_receipt": receipt,
        },
        "memorization_diagnostic": {
            "policy_identity": MEMORIZATION_POLICY_V1_IDENTITY,
            "training_sample_count": 100,
            "training_exact_match_count": 3,
            "training_exact_match_rate": 0.03,
            "heldout_sample_count": 100,
            "heldout_exact_match_count": 0,
            "heldout_exact_match_rate": 0.0,
            "max_training_exact_match_rate": 0.05,
            "max_heldout_exact_match_rate": 0.0,
            "passed": True,
        },
        "throughput_optimized_targets_per_second": 1250.5,
        "peak_memory_bytes": 1_500_000_000,
    }
    pilot["d06_evaluation"] = d06
    return {
        "bounded_pilot": pilot,
        "_test_fresh_process_expectations": _fresh_process_expectations(receipt),
    }


def test_terminal_pilot_requires_numeric_d06_scientific_evidence() -> None:
    assert validate_terminal_pilot_evaluation(_evidence()) == []


def test_terminal_pilot_requires_externally_supplied_fresh_process_expectations() -> None:
    evidence = _evidence()
    blockers = _validate_terminal_pilot_evaluation(evidence)
    assert (
        "bounded_pilot.d06.inference_probe.fresh_process_expectations_missing"
        in blockers
    )


def test_legacy_synthetic_probe_cannot_grant_fresh_process_credit() -> None:
    evidence = _evidence()
    checkpoint_identity = evidence["bounded_pilot"]["result_checkpoint_identity"]
    evidence["bounded_pilot"]["d06_evaluation"]["inference_probe"] = {
        "prompt_suite_identity": "d06-pilot-probes-v1",
        "output_fingerprint": "sha256:" + "a" * 64,
        "fresh_process_reload": True,
        "checkpoint_identity": checkpoint_identity,
    }
    blockers = validate_terminal_pilot_evaluation(evidence)
    assert "bounded_pilot.d06.inference_probe.fresh_process_receipt_missing" in blockers


def test_coherent_receipt_reseal_cannot_replace_authorized_receipt() -> None:
    evidence = _evidence()
    receipt = evidence["bounded_pilot"]["d06_evaluation"]["inference_probe"][
        "fresh_process_receipt"
    ]
    receipt["prompt_suite_identity"] = "substituted-suite"
    _reseal_fresh_process_receipt(receipt)
    blockers = validate_terminal_pilot_evaluation(evidence)
    assert (
        "bounded_pilot.d06.inference_probe.d07.fresh_process.prompt_suite_identity_mismatch"
        in blockers
    )
    assert (
        "bounded_pilot.d06.inference_probe.d07.fresh_process.receipt_identity_expected_mismatch"
        in blockers
    )


def test_wrong_expected_receipt_identity_is_rejected() -> None:
    evidence = _evidence()
    expectations = deepcopy(evidence["_test_fresh_process_expectations"])
    expectations["receipt_identity"] = "sha256:" + "f" * 64
    blockers = _validate_terminal_pilot_evaluation(
        evidence,
        fresh_process_expectations=expectations,
    )
    assert (
        "bounded_pilot.d06.inference_probe.d07.fresh_process.receipt_identity_expected_mismatch"
        in blockers
    )


def test_stale_checkpoint_in_receipt_is_rejected_even_after_reseal() -> None:
    evidence = _evidence()
    receipt = evidence["bounded_pilot"]["d06_evaluation"]["inference_probe"][
        "fresh_process_receipt"
    ]
    receipt["checkpoint_identity"] = "d" * 64
    _reseal_fresh_process_receipt(receipt)
    blockers = validate_terminal_pilot_evaluation(evidence)
    assert (
        "bounded_pilot.d06.inference_probe.d07.fresh_process.checkpoint_identity_mismatch"
        in blockers
    )


def test_prompt_and_generation_config_substitution_are_rejected() -> None:
    evidence = _evidence()
    receipt = evidence["bounded_pilot"]["d06_evaluation"]["inference_probe"][
        "fresh_process_receipt"
    ]
    receipt["prompt_payload_sha256"] = "sha256:" + "d" * 64
    receipt["generation_config_identity"] = "sha256:" + "e" * 64
    _reseal_fresh_process_receipt(receipt)
    blockers = validate_terminal_pilot_evaluation(evidence)
    assert (
        "bounded_pilot.d06.inference_probe.d07.fresh_process.prompt_payload_sha256_mismatch"
        in blockers
    )
    assert (
        "bounded_pilot.d06.inference_probe.d07.fresh_process.generation_config_identity_mismatch"
        in blockers
    )


def test_same_process_receipt_is_rejected_even_after_reseal() -> None:
    evidence = _evidence()
    receipt = evidence["bounded_pilot"]["d06_evaluation"]["inference_probe"][
        "fresh_process_receipt"
    ]
    receipt["child_pid"] = receipt["parent_pid"]
    _reseal_fresh_process_receipt(receipt)
    blockers = validate_terminal_pilot_evaluation(evidence)
    assert (
        "bounded_pilot.d06.inference_probe.d07.fresh_process.process_not_fresh"
        in blockers
    )


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


def test_bpb_is_recomputed_from_additive_totals() -> None:
    evidence = _evidence()
    ua = evidence["bounded_pilot"]["d06_evaluation"]["heldout_metrics"]["UA"]
    ua["trained_bpb"] = 0.01
    blockers = validate_terminal_pilot_evaluation(evidence)
    assert "bounded_pilot.d06.heldout_metrics.UA.trained_bpb_mismatch" in blockers

    evidence = _evidence()
    ua = evidence["bounded_pilot"]["d06_evaluation"]["heldout_metrics"]["UA"]
    ua["scored_utf8_bytes"] = 0
    blockers = validate_terminal_pilot_evaluation(evidence)
    assert "bounded_pilot.d06.heldout_metrics.UA.scored_utf8_bytes_invalid" in blockers


def test_all_three_heldout_strata_are_mandatory_and_closed_world() -> None:
    evidence = _evidence()
    del evidence["bounded_pilot"]["d06_evaluation"]["heldout_metrics"]["UA"]
    blockers = validate_terminal_pilot_evaluation(evidence)
    assert "bounded_pilot.d06.heldout_metrics.UA_missing" in blockers
    assert "bounded_pilot.d06.heldout_metrics.strata_set_mismatch" in blockers

    evidence = _evidence()
    heldout = evidence["bounded_pilot"]["d06_evaluation"]["heldout_metrics"]
    heldout["EXTRA"] = deepcopy(heldout["UA"])
    blockers = validate_terminal_pilot_evaluation(evidence)
    assert "bounded_pilot.d06.heldout_metrics.strata_set_mismatch" in blockers


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


def test_memorization_diagnostic_recomputes_rates_from_exact_counts() -> None:
    evidence = _evidence()
    diagnostic = evidence["bounded_pilot"]["d06_evaluation"]["memorization_diagnostic"]
    diagnostic["training_exact_match_rate"] = 0.01
    blockers = validate_terminal_pilot_evaluation(evidence)
    assert (
        "bounded_pilot.d06.memorization_diagnostic.training_exact_match_rate_mismatch"
        in blockers
    )


def test_memorization_summary_boolean_cannot_override_policy_thresholds() -> None:
    evidence = _evidence()
    diagnostic = evidence["bounded_pilot"]["d06_evaluation"]["memorization_diagnostic"]
    diagnostic["max_training_exact_match_rate"] = 0.02
    diagnostic["passed"] = True
    blockers = validate_terminal_pilot_evaluation(evidence)
    assert (
        "bounded_pilot.d06.memorization_diagnostic.max_training_exact_match_rate_policy_mismatch"
        in blockers
    )
    assert "bounded_pilot.d06.memorization_diagnostic_not_passed" in blockers


def test_memorization_policy_cannot_be_weakened_by_candidate() -> None:
    evidence = _evidence()
    diagnostic = evidence["bounded_pilot"]["d06_evaluation"]["memorization_diagnostic"]
    diagnostic["training_exact_match_count"] = 100
    diagnostic["training_exact_match_rate"] = 1.0
    diagnostic["max_training_exact_match_rate"] = 1.0
    diagnostic["passed"] = True
    blockers = validate_terminal_pilot_evaluation(evidence)
    assert (
        "bounded_pilot.d06.memorization_diagnostic.max_training_exact_match_rate_policy_mismatch"
        in blockers
    )
    assert "bounded_pilot.d06.memorization_diagnostic_not_passed" in blockers


def test_memorization_policy_identity_substitution_is_rejected() -> None:
    evidence = _evidence()
    diagnostic = evidence["bounded_pilot"]["d06_evaluation"]["memorization_diagnostic"]
    diagnostic["policy_identity"] = "sha256:" + "f" * 64
    blockers = validate_terminal_pilot_evaluation(evidence)
    assert "bounded_pilot.d06.memorization_diagnostic.policy_identity_mismatch" in blockers


def test_memorization_threshold_bool_alias_is_rejected() -> None:
    evidence = _evidence()
    diagnostic = evidence["bounded_pilot"]["d06_evaluation"]["memorization_diagnostic"]
    diagnostic["max_training_exact_match_rate"] = True
    diagnostic["passed"] = True
    blockers = validate_terminal_pilot_evaluation(evidence)
    assert (
        "bounded_pilot.d06.memorization_diagnostic.max_training_exact_match_rate_invalid"
        in blockers
    )
    assert "bounded_pilot.d06.memorization_diagnostic_not_passed" in blockers


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


def test_terminal_provenance_wrapper_blocks_long_training_without_d06_evidence(
    monkeypatch,
) -> None:
    import twelve_six.learned20_pilot_authority as pilot_authority

    monkeypatch.setattr(
        pilot_authority,
        "assess_launch_with_checkpoint_provenance",
        lambda contract, evidence, *, material_cost: {
            "pilot_ready": True,
            "long_training_ready": True,
            "pilot_blockers": [],
            "long_training_blockers": [],
        },
    )
    monkeypatch.setattr(
        pilot_authority,
        "validate_bounded_pilot_authority",
        lambda evidence: [],
    )

    result = pilot_authority.assess_launch_with_terminal_provenance(
        {},
        {"bounded_pilot": {"terminal": True, "identity": "pilot-v1"}},
        material_cost=False,
    )

    assert result["pilot_ready"] is True
    assert result["long_training_ready"] is False
    assert "bounded_pilot.d06_evaluation_missing" in result["long_training_blockers"]


def test_launch_contract_is_the_only_source_of_d07_expectations(monkeypatch) -> None:
    import twelve_six.learned20_pilot_authority as pilot_authority

    expected = _evidence()["_test_fresh_process_expectations"]
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        pilot_authority,
        "assess_launch_with_checkpoint_provenance",
        lambda contract, evidence, *, material_cost: {
            "pilot_ready": True,
            "long_training_ready": True,
            "pilot_blockers": [],
            "long_training_blockers": [],
        },
    )
    monkeypatch.setattr(
        pilot_authority,
        "validate_evaluation_firewall_provenance",
        lambda evidence: [],
    )
    monkeypatch.setattr(
        pilot_authority,
        "validate_bounded_pilot_authority",
        lambda evidence: [],
    )

    def _capture(evidence, *, fresh_process_expectations=None):
        captured["expectations"] = fresh_process_expectations
        return []

    monkeypatch.setattr(
        pilot_authority,
        "validate_terminal_pilot_evaluation",
        _capture,
    )
    pilot_authority.assess_launch_with_terminal_provenance(
        {"d07_fresh_process_expectations": expected},
        {"d07_fresh_process_expectations": {"receipt_identity": "attacker"}},
        material_cost=False,
    )
    assert captured["expectations"] == expected
