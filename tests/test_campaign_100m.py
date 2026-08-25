from __future__ import annotations

from pathlib import Path

import pytest

from twelve_six.campaign_100m import (
    GPU_PILOT_SCHEMA,
    S2_EVIDENCE_SCHEMA,
    S2_EXPECTED_PARAMETERS,
    S2_MODEL_SHA256,
    S4_D11_EXPECTED_PARAMETERS,
    S4_D11_MODEL_SHA256,
    project_budget,
    run_s4_gpu_pilot,
    s2_d11_model_spec,
    s4_d11_model_spec,
    wrap_s3_gpu_pilot,
)
from twelve_six.campaign_100m_authority import qualify_campaign_main_launch

SOURCE = "a" * 40


def _pilot(*, measured_100m: bool = False) -> dict[str, object]:
    return {
        "schema": GPU_PILOT_SCHEMA,
        "source_sha": SOURCE,
        "hourly_cost_eur": 2.0,
        "report_sha256": "b" * 64,
        "measurement": {
            "measured_end_to_end_optimized_tokens_per_second": 100_000.0,
            "checkpoint_payload_bytes": 123,
            "checkpoint_id": "checkpoint-test",
            "restored_checkpoint_id": "checkpoint-test",
        },
        "truth_boundary": {
            "100m_throughput_measured": measured_100m,
            "projection_requires_100m_pilot_recalibration": not measured_100m,
        },
    }


def test_exact_s2_and_s4_geometry_identities() -> None:
    s2 = s2_d11_model_spec()
    assert s2.parameter_count() == S2_EXPECTED_PARAMETERS
    assert s2.identity_sha256() == S2_MODEL_SHA256

    s4 = s4_d11_model_spec()
    assert s4.parameter_count() == S4_D11_EXPECTED_PARAMETERS
    assert s4.identity_sha256() == S4_D11_MODEL_SHA256
    assert s4.n_heads == 12
    assert s4.n_kv_heads == 4
    assert s4.max_seq_len == 4096


def test_budget_projection_uses_measured_pilot_rate_and_authority() -> None:
    preliminary = project_budget(_pilot(), "eur_2k")
    expected_hours = 2_000_000_000 / 100_000.0 / 3600.0
    assert preliminary["projected_accelerator_hours"] == pytest.approx(expected_hours)
    assert preliminary["projected_accelerator_cost_eur"] == pytest.approx(expected_hours * 2.0)
    assert preliminary["within_compute_cap"] is True
    assert preliminary["projection_authority"] == (
        "10M_PRELIMINARY_EXTRAPOLATION_RECALIBRATION_REQUIRED"
    )

    measured = project_budget(_pilot(measured_100m=True), "eur_2k")
    assert measured["projection_authority"] == "100M_MEASURED_PILOT"


def test_wrap_gpu_pilot_rejects_cpu_evidence() -> None:
    s3 = {
        "schema": "12-6.s3-10m-engineering-evidence.v1",
        "source_sha": SOURCE,
        "candidate": {"analytic_parameters": 9_999_680},
        "runtime": {"device": "cpu"},
        "execution": {
            "optimized_tokens": 1000,
            "train_backward_update_and_checkpoint_seconds": 1.0,
            "last_loss": 1.0,
            "last_grad_norm": 1.0,
        },
        "checkpoint": {"records": [{"step": 1}]},
        "truth_boundary": {"gpu_execution": False},
    }
    with pytest.raises(ValueError, match="CUDA"):
        wrap_s3_gpu_pilot(
            s3,
            source_sha=SOURCE,
            provider_label="test",
            hardware_label="test-gpu",
            hourly_cost_eur=1.0,
            rate_evidence="test fixture",
        )


def test_paid_s4_gpu_pilot_requires_explicit_authorization_before_cuda() -> None:
    with pytest.raises(PermissionError, match="explicit paid-compute authorization"):
        run_s4_gpu_pilot(
            repo_root=Path("."),
            source_sha=SOURCE,
            checkpoint_root=None,
            provider_label="test-provider",
            hardware_label="test-gpu",
            hourly_cost_eur=1.0,
            rate_evidence="test rate",
            compute_class="paid",
            paid_compute_authorized=False,
        )


def _qualification_inputs() -> dict[str, dict[str, object]]:
    tokenizer = {
        "schema": "12-6.tokenizer-freeze.v1",
        "status": "FROZEN",
        "source_sha": SOURCE,
        "algorithm": "bytelevel_bpe",
        "vocab_size": 32768,
        "round_trip_pass": True,
        "repeatability_pass": True,
        "heldout_fertility_measured": True,
        "artifact_sha256": "c" * 64,
        "ordered_vocab_sha256": "d" * 64,
    }
    corpus = {
        "schema": "12-6.corpus-freeze.v1",
        "status": "FROZEN",
        "source_sha": SOURCE,
        "manifest_sha256": "e" * 64,
        "eligible_train_tokens": 2_000_000_000,
        "rights_review_complete": True,
        "contamination_gate_pass": True,
        "reproducible_build_pass": True,
        "tokenizer_artifact_sha256": "c" * 64,
    }
    evaluation = {
        "schema": "12-6.evaluation-freeze.v1",
        "status": "FROZEN",
        "source_sha": SOURCE,
        "training_use": False,
        "random_init_control": True,
        "validation_manifest_sha256": "f" * 64,
        "test_manifest_sha256": "1" * 64,
        "contamination_registry_sha256": "2" * 64,
        "protocol_sha256": "3" * 64,
        "capability_registry_sha256": "4" * 64,
        "tokenizer_artifact_sha256": "c" * 64,
        "corpus_manifest_sha256": "e" * 64,
    }
    return {
        "s2": {
            "schema": S2_EVIDENCE_SCHEMA,
            "source_sha": SOURCE,
            "execution": {"optimizer_steps": 1, "parameter_changed": True},
        },
        "s3": {
            "schema": "12-6.s3-10m-engineering-evidence.v1",
            "source_sha": SOURCE,
            "execution": {"optimizer_steps": 1},
            "checkpoint": {"records": [{"step": 1}]},
        },
        "s4": {
            "schema": "12-6.campaign47-s4-100m-preflight.v1",
            "source_sha": SOURCE,
            "candidate": {"instantiated_trainable_parameters": 99_797_760},
        },
        "pilot": _pilot(measured_100m=True),
        "tokenizer": tokenizer,
        "corpus": corpus,
        "evaluation": evaluation,
    }


def _qualify(evidence: dict[str, dict[str, object]], *, authorized: bool) -> dict[str, object]:
    return qualify_campaign_main_launch(
        source_sha=SOURCE,
        variant_name="eur_2k",
        s2_evidence=evidence["s2"],
        s3_evidence=evidence["s3"],
        s4_preflight=evidence["s4"],
        gpu_pilot=evidence["pilot"],
        tokenizer_freeze=evidence["tokenizer"],
        corpus_freeze=evidence["corpus"],
        evaluation_freeze=evidence["evaluation"],
        paid_compute_authorized=authorized,
    )


def test_paid_launch_is_fail_closed_without_explicit_authorization() -> None:
    report = _qualify(_qualification_inputs(), authorized=False)
    assert report["technical_ready"] is True
    assert report["main_launch_ready"] is False
    assert report["action"] == "BLOCKED_NO_PAYMENT_LAUNCH"


def test_explicit_authorization_only_unlocks_after_all_technical_gates() -> None:
    evidence = _qualification_inputs()
    report = _qualify(evidence, authorized=True)
    assert report["technical_ready"] is True
    assert report["main_launch_ready"] is True
    assert report["action"] == "PREPARED_FOR_EXPLICIT_PAYMENT_LAUNCH"

    evidence["corpus"]["rights_review_complete"] = False
    blocked = _qualify(evidence, authorized=True)
    assert blocked["main_launch_ready"] is False
    assert blocked["checks"]["corpus_frozen_and_eligible"] is False


def test_evaluation_freeze_is_part_of_main_launch_authority() -> None:
    evidence = _qualification_inputs()
    evidence["evaluation"]["training_use"] = True
    report = _qualify(evidence, authorized=True)
    assert report["checks"]["evaluation_registry_frozen"] is False
    assert report["technical_ready"] is False
    assert report["main_launch_ready"] is False
