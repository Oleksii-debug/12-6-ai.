from __future__ import annotations

import copy
import json
from pathlib import Path

from twelve_six.portable_run_packet import (
    CONTRACT_FIELD_LOCATIONS,
    assess_portable_run_packet,
    validate_portable_run_contract,
)

ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / "configs/research/r01_portable_local_free_run_packet_v1.json"
ROADMAP = ROOT / "configs/research/r01_accelerated_scaling_roadmap_v2.json"
SHA40 = "a" * 40
SHA64 = "b" * 64


def _load() -> dict:
    return json.loads(PACKET.read_text(encoding="utf-8"))


def _authority() -> dict:
    return {
        "repository": "Oleksii-debug/12-6-ai.",
        "git_sha": SHA40,
        "evidence_sha256": SHA64,
        "workflow_run_id": 123,
        "workflow_conclusion": "success",
        "terminal": True,
    }


def _ready_postpack() -> dict:
    return {
        "schema_version": "12-6.d04-deterministic-double-pack-proof.v1",
        "proof_identity_sha256": SHA64,
        "terminal_corpus_authority_identity_sha256": SHA64,
        "terminal_record_inventory_digest_sha256": SHA64,
        "terminal_payload_inventory_digest_sha256": SHA64,
        "stage_bindings": {
            "normalization": SHA64,
            "evaluation_reservations": SHA64,
            "dedup": SHA64,
            "split": SHA64,
            "packing": SHA64,
        },
        "tokenizer_identity_sha256": SHA64,
        "packing_identity_sha256": SHA64,
        "ledger_identity_sha256": SHA64,
        "canonical_build_sha256": SHA64,
        "one_pass_unique_loss_positions": 1000,
        "independent_builds_byte_identical": True,
        "training_authorized_by_proof": False,
    }


def _ready_fresh_packet() -> dict:
    data = _load()
    identities = data["identities"]
    identities["source_git_sha"] = SHA40
    for field in (
        "initspec_sha256",
        "tokenizer_sha256",
        "corpus_manifest_sha256",
        "split_sha256",
        "packing_sha256",
        "unique_loss_ledger_sha256",
    ):
        identities[field] = SHA64
    for field in data["authorities"]:
        if field != "parent_checkpoint":
            data["authorities"][field] = _authority()
    data["postpack"].update(_ready_postpack())
    data["recipe"].update(
        {
            "training_config_sha256": SHA64,
            "optimizer_scheduler_precision": {
                "optimizer": "AdamW",
                "scheduler": "cosine_with_warmup",
                "precision": "fp32",
            },
            "seed": 20260907,
            "target_unique_loss_positions": 1000,
            "maximum_total_exposures": 1000,
            "available_unique_loss_positions": 1000,
        }
    )
    data["checkpoint"].update(
        {
            "stop_resume_policy_sha256": SHA64,
            "session_time_limit_minutes": 120,
            "first_checkpoint_deadline_minutes": 15,
            "checkpoint_every_steps": 50,
        }
    )
    data["evaluation"].update(
        {
            "evaluation_schedule_sha256": SHA64,
            "final_test_reservation_sha256": SHA64,
        }
    )
    data["runtime"].update(
        {
            "backend_id": "PROJECT_NATIVE_PYTORCH",
            "python_version": "3.11.11",
            "framework_version": "2.8.0",
            "environment_lock_sha256": SHA64,
            "device_type": "cpu",
        }
    )
    data["resource"]["provider"] = "OWNER_LAPTOP"
    data["output"]["artifact_store_uri"] = "file:///tmp/twelve-six-run"
    data["status"] = "READY_CANDIDATE"
    return data


def _ready_resume_packet() -> dict:
    data = _ready_fresh_packet()
    data["checkpoint"]["mode"] = "RESUME"
    data["checkpoint"]["lineage"] = {
        "parent_checkpoint_sha256": SHA64,
        "parent_manifest_sha256": SHA64,
        "previous_run_id": "R01-SESSION-001",
        "source_provider": "OWNER_LAPTOP",
        "cross_provider_transfer": True,
        "resume_validated": True,
    }
    data["authorities"]["parent_checkpoint"] = _authority()
    data["resource"].update({"resource_class": "FREE_GPU", "provider": "KAGGLE"})
    data["runtime"]["device_type"] = "cuda"
    data["output"]["artifact_store_uri"] = "https://artifacts.example/sha256"
    return data


def test_current_template_is_valid_but_not_launch_ready() -> None:
    data = _load()
    assert validate_portable_run_contract(data) == []
    result = assess_portable_run_packet(data)
    assert result.contract_valid
    assert not result.ready_for_initial_local_free_launch
    assert not result.ready_for_cross_provider_resume
    assert "source_git_sha_invalid" in result.launch_blockers
    assert "resource_provider_unbound" in result.launch_blockers
    assert "final_test_reservation_sha256_invalid" in result.launch_blockers
    assert "postpack_proof_authority_invalid" in result.launch_blockers
    assert "postpack_proof_identity_sha256_invalid" in result.launch_blockers


def test_packet_fields_exactly_match_accelerated_roadmap() -> None:
    roadmap = json.loads(ROADMAP.read_text(encoding="utf-8"))
    declared = set(roadmap["portable_training_runner"]["required_run_packet_fields"])
    assert declared == set(CONTRACT_FIELD_LOCATIONS)
    assert declared == set(_load()["contract_fields"])


def test_complete_fresh_packet_is_ready_only_for_initial_launch() -> None:
    result = assess_portable_run_packet(_ready_fresh_packet())
    assert result.contract_valid
    assert result.ready_for_initial_local_free_launch
    assert not result.ready_for_cross_provider_resume
    assert result.launch_blockers == ()
    assert result.resume_blockers == ("checkpoint_mode_is_not_resume",)


def test_final_test_reservation_authority_is_required_and_exact() -> None:
    data = _ready_fresh_packet()
    data["authorities"]["final_test_reservation"] = None
    result = assess_portable_run_packet(data)
    assert not result.ready_for_initial_local_free_launch
    assert "final_test_reservation_authority_invalid" in result.launch_blockers

    data = _ready_fresh_packet()
    data["evaluation"]["final_test_reservation_sha256"] = "c" * 64
    result = assess_portable_run_packet(data)
    assert not result.ready_for_initial_local_free_launch
    assert "final_test_reservation_authority_mismatch" in result.launch_blockers


def test_postpack_proof_authority_and_lineage_must_match_packet() -> None:
    data = _ready_fresh_packet()
    data["authorities"]["postpack_proof"] = None
    result = assess_portable_run_packet(data)
    assert not result.ready_for_initial_local_free_launch
    assert "postpack_proof_authority_invalid" in result.launch_blockers

    data = _ready_fresh_packet()
    data["postpack"]["ledger_identity_sha256"] = "c" * 64
    result = assess_portable_run_packet(data)
    assert not result.ready_for_initial_local_free_launch
    assert "postpack_ledger_identity_mismatch" in result.launch_blockers

    data = _ready_fresh_packet()
    data["postpack"]["one_pass_unique_loss_positions"] = 999
    result = assess_portable_run_packet(data)
    assert not result.ready_for_initial_local_free_launch
    assert "postpack_unique_loss_positions_mismatch" in result.launch_blockers


def test_postpack_proof_cannot_self_authorize_training() -> None:
    data = _ready_fresh_packet()
    data["postpack"]["training_authorized_by_proof"] = True
    errors = validate_portable_run_contract(data)
    assert "postpack_proof_must_not_self_authorize_training" in errors


def test_unique_exposure_cannot_be_inflated_by_replay() -> None:
    data = _ready_fresh_packet()
    data["recipe"]["max_exposures_per_unique_position"] = 2
    data["recipe"]["maximum_total_exposures"] = 2000
    result = assess_portable_run_packet(data)
    assert not result.ready_for_initial_local_free_launch
    assert "max_exposures_per_unique_position_must_be_one" in result.launch_blockers
    assert "maximum_total_exposures_must_equal_unique_target" in result.launch_blockers


def test_target_unique_positions_must_fit_terminal_ledger() -> None:
    data = _ready_fresh_packet()
    data["recipe"]["target_unique_loss_positions"] = 1001
    data["recipe"]["maximum_total_exposures"] = 1001
    result = assess_portable_run_packet(data)
    assert "target_unique_loss_positions_exceed_ledger" in result.launch_blockers


def test_paid_resource_or_nonzero_cost_fails_contract() -> None:
    data = _load()
    data["resource"].update(
        {
            "resource_class": "PAID_GPU",
            "maximum_cost_usd": 1,
            "materially_paid": True,
        }
    )
    errors = validate_portable_run_contract(data)
    assert "resource_class_must_be_local_free_or_free_gpu" in errors
    assert "maximum_cost_usd_must_be_zero" in errors
    assert "materially_paid_must_be_false" in errors


def test_final_test_access_or_teacher_logits_fail_contract() -> None:
    data = _load()
    data["evaluation"]["final_test_payload_access"] = True
    data["truth_boundary"]["hidden_teacher_logits_used"] = True
    errors = validate_portable_run_contract(data)
    assert "final_test_payload_access_must_be_false" in errors
    assert "truth_boundary_hidden_teacher_logits_used_must_be_false" in errors


def test_credentials_cannot_be_embedded_in_packet() -> None:
    data = _load()
    data["runtime"]["api_key"] = "not-a-real-key"
    errors = validate_portable_run_contract(data)
    assert "embedded_secret_forbidden:root.runtime.api_key" in errors


def test_backend_requires_exact_runtime_and_terminal_authority() -> None:
    data = _ready_fresh_packet()
    data["runtime"]["framework_version"] = None
    data["authorities"]["backend"] = None
    result = assess_portable_run_packet(data)
    assert "runtime_framework_version_missing" in result.launch_blockers
    assert "backend_authority_invalid" in result.launch_blockers


def test_first_checkpoint_must_precede_session_limit() -> None:
    data = _ready_fresh_packet()
    data["checkpoint"]["first_checkpoint_deadline_minutes"] = 120
    result = assess_portable_run_packet(data)
    assert "first_checkpoint_deadline_must_precede_session_limit" in result.launch_blockers


def test_complete_cross_provider_resume_packet_is_ready() -> None:
    result = assess_portable_run_packet(_ready_resume_packet())
    assert result.contract_valid
    assert not result.ready_for_initial_local_free_launch
    assert result.ready_for_cross_provider_resume
    assert result.resume_blockers == ()


def test_cross_provider_resume_rejects_same_provider_or_missing_parent() -> None:
    data = _ready_resume_packet()
    data["checkpoint"]["lineage"]["source_provider"] = "KAGGLE"
    data["checkpoint"]["lineage"]["parent_checkpoint_sha256"] = None
    result = assess_portable_run_packet(data)
    assert not result.ready_for_cross_provider_resume
    assert "cross_provider_source_and_target_must_differ" in result.resume_blockers
    assert "parent_checkpoint_sha256_invalid" in result.resume_blockers


def test_mutating_copy_does_not_change_template() -> None:
    original = _load()
    mutated = copy.deepcopy(original)
    mutated["truth_boundary"]["foreign_pretrained_or_aligned_weights_used"] = True
    assert validate_portable_run_contract(original) == []
    assert validate_portable_run_contract(mutated)
