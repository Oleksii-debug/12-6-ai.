from __future__ import annotations

import copy
import json
from pathlib import Path

from twelve_six.learned20m_readiness import scientific_authority_token
from twelve_six.portable_run_binding import bind_portable_run_packet
from twelve_six.portable_run_packet import assess_portable_run_packet

ROOT = Path(__file__).resolve().parents[1]
READINESS = ROOT / "configs/research/r01_learned20m_launch_readiness_v1.json"
PACKET = ROOT / "configs/research/r01_portable_local_free_run_packet_v1.json"
OVERLAY = ROOT / "configs/research/r01_portable_session_overlay_v1.json"
SHA40 = "a" * 40
SHA64 = "b" * 64


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _authority(*, git_sha: str = SHA40, **extra: object) -> dict:
    authority = {
        "repository": "Oleksii-debug/12-6-ai.",
        "git_sha": git_sha,
        "evidence_sha256": SHA64,
        "workflow_run_id": 123,
        "workflow_conclusion": "success",
        "terminal": True,
    }
    authority.update(extra)
    return authority


def _ready_readiness() -> dict:
    data = _load(READINESS)
    evidence = data["evidence"]
    evidence["code"]["git_sha"] = SHA40
    evidence["corpus"].update(
        {
            "manifest_sha256": SHA64,
            "split_sha256": SHA64,
            "packing_sha256": SHA64,
            "two_clean_builds_identical": True,
            "authority": _authority(),
        }
    )
    evidence["tokenizer"].update(
        {
            "identity_sha256": SHA64,
            "decision": "BYTE_BASELINE_RETAINED",
            "authority": _authority(),
        }
    )
    evidence["loss_ledger"].update(
        {
            "identity_sha256": SHA64,
            "unique_causal_loss_positions": 1000,
            "authority": _authority(),
            "data_budget_authority": _authority(),
            "data_budget_status": "QUALIFIED",
        }
    )
    evidence["checkpoint_integrity"].update(
        {"authority": _authority(), "status": "PASS"}
    )
    evidence["evaluation"].update(
        {
            "firewall_authority": _authority(),
            "selection_validation_authority": _authority(),
            "status": "PASS",
        }
    )
    evidence["training_recipe"].update(
        {
            "authority": _authority(),
            "status": "QUALIFIED",
            "seed_count": 1,
            "config_sha256": SHA64,
            "stopping_policy_sha256": SHA64,
            "requested_unique_loss_positions": 1000,
            "requested_total_training_exposures": 1000,
            "max_exposures_per_unique_position": 1,
        }
    )
    return data


def _verified_tokens(readiness: dict) -> list[str]:
    evidence = readiness["evidence"]
    role_authorities = (
        ("corpus", evidence["corpus"]["authority"]),
        ("tokenizer", evidence["tokenizer"]["authority"]),
        ("loss_ledger", evidence["loss_ledger"]["authority"]),
        ("data_budget", evidence["loss_ledger"]["data_budget_authority"]),
        ("checkpoint_integrity", evidence["checkpoint_integrity"]["authority"]),
        ("evaluation_firewall", evidence["evaluation"]["firewall_authority"]),
        (
            "selection_validation",
            evidence["evaluation"]["selection_validation_authority"],
        ),
        ("training_recipe", evidence["training_recipe"]["authority"]),
    )
    tokens: list[str] = []
    for role, authority in role_authorities:
        token = scientific_authority_token(role, authority, require_workflow=True)
        assert token is not None
        tokens.append(token)
    return tokens


def _ready_overlay() -> dict:
    data = _load(OVERLAY)
    model = _load(READINESS)["model_authority"]
    data["status"] = "READY_CANDIDATE"
    data["scientific_bindings"].update(
        {
            "initspec_sha256": SHA64,
            "seed": 20260907,
            "optimizer_scheduler_precision": {
                "optimizer": "AdamW",
                "scheduler": "cosine_with_warmup",
                "precision": "fp32",
            },
            "authorities": {
                "code": _authority(),
                "model": _authority(
                    git_sha=model["git_sha"],
                    modelspec_sha256=model["modelspec_sha256"],
                ),
                "backend": _authority(
                    backend_id="PROJECT_NATIVE_PYTORCH",
                    environment_lock_sha256=SHA64,
                ),
            },
        }
    )
    data["checkpoint"].update(
        {
            "session_time_limit_minutes": 120,
            "first_checkpoint_deadline_minutes": 15,
            "checkpoint_every_steps": 50,
        }
    )
    data["evaluation"]["evaluation_schedule_sha256"] = SHA64
    data["runtime"].update(
        {
            "backend_id": "PROJECT_NATIVE_PYTORCH",
            "python_version": "3.11.11",
            "framework_version": "2.8.0",
            "environment_lock_sha256": SHA64,
            "device_type": "cpu",
        }
    )
    data["resource"]["provider"] = "OTHER_FREE"
    data["output"]["artifact_store_uri"] = "https://artifacts.example/sha256"
    return data


def _same_provider_resume_overlay() -> dict:
    data = _ready_overlay()
    data["checkpoint"].update(
        {
            "mode": "RESUME",
            "lineage": {
                "parent_checkpoint_sha256": SHA64,
                "parent_manifest_sha256": SHA64,
                "previous_run_id": "R01-GITHUB-SESSION-001",
                "source_provider": "OTHER_FREE",
                "cross_provider_transfer": False,
                "resume_validated": True,
            },
            "parent_checkpoint_authority": _authority(),
        }
    )
    return data


def _bind(overlay: dict):
    readiness = _ready_readiness()
    return bind_portable_run_packet(
        readiness,
        _load(PACKET),
        overlay,
        verified_scientific_authorities=_verified_tokens(readiness),
    )


def test_same_provider_resume_fails_closed_without_trusted_recovery_binding() -> None:
    result = _bind(_same_provider_resume_overlay())
    assert not result.binding_ready
    assert result.mode == "RESUME"
    assert result.packet is None
    assert result.blockers == ("packet:trusted_parent_recovery_binding_missing",)

    fresh = _bind(_ready_overlay())
    assert fresh.binding_ready
    assert fresh.packet is not None
    packet = copy.deepcopy(fresh.packet)
    packet["checkpoint"]["mode"] = "RESUME"
    packet["checkpoint"]["lineage"] = copy.deepcopy(
        _same_provider_resume_overlay()["checkpoint"]["lineage"]
    )
    packet["authorities"]["parent_checkpoint"] = _authority()
    assessment = assess_portable_run_packet(packet)
    assert assessment.contract_valid
    assert not assessment.ready_for_same_provider_fresh_process_resume
    assert not assessment.ready_for_cross_provider_resume
    assert assessment.same_provider_resume_blockers == (
        "trusted_parent_recovery_binding_missing",
    )
    assert "cross_provider_transfer_not_declared" in assessment.resume_blockers
    assert "cross_provider_source_and_target_must_differ" in assessment.resume_blockers


def test_same_provider_resume_cannot_self_authorize_resealed_lineage() -> None:
    overlay = _same_provider_resume_overlay()
    overlay["checkpoint"]["lineage"].update(
        {
            "parent_checkpoint_sha256": "c" * 64,
            "parent_manifest_sha256": "d" * 64,
            "previous_run_id": "R01-RESEALED-SESSION-999",
            "source_provider": "OTHER_FREE",
            "resume_validated": True,
        }
    )
    overlay["checkpoint"]["parent_checkpoint_authority"] = _authority(
        git_sha="c" * 40
    )

    result = _bind(overlay)
    assert not result.binding_ready
    assert result.packet is None
    assert result.blockers == ("packet:trusted_parent_recovery_binding_missing",)


def test_same_provider_resume_rejects_provider_mismatch_and_cross_transfer_claim() -> None:
    mismatch = _same_provider_resume_overlay()
    mismatch["checkpoint"]["lineage"]["source_provider"] = "OWNER_LAPTOP"
    result = _bind(mismatch)
    assert not result.binding_ready
    assert result.packet is None
    assert "packet:same_provider_source_and_target_must_match" in result.blockers
    assert "packet:trusted_parent_recovery_binding_missing" in result.blockers

    transfer = _same_provider_resume_overlay()
    transfer["checkpoint"]["lineage"]["cross_provider_transfer"] = True
    result = _bind(transfer)
    assert not result.binding_ready
    assert result.packet is None
    assert "packet:cross_provider_source_and_target_must_differ" in result.blockers


def test_same_provider_resume_requires_validated_parent_identity_and_authority() -> None:
    mutations = (
        ("parent_checkpoint_sha256", None, "packet:parent_checkpoint_sha256_invalid"),
        ("parent_manifest_sha256", None, "packet:parent_manifest_sha256_invalid"),
        ("previous_run_id", "", "packet:previous_run_id_missing"),
        ("resume_validated", False, "packet:parent_checkpoint_resume_not_validated"),
    )
    for field, value, expected in mutations:
        overlay = _same_provider_resume_overlay()
        overlay["checkpoint"]["lineage"][field] = value
        result = _bind(overlay)
        assert not result.binding_ready
        assert result.packet is None
        assert expected in result.blockers
        assert "packet:trusted_parent_recovery_binding_missing" in result.blockers

    overlay = _same_provider_resume_overlay()
    overlay["checkpoint"]["parent_checkpoint_authority"] = None
    result = _bind(overlay)
    assert not result.binding_ready
    assert result.packet is None
    assert "packet:parent_checkpoint_authority_invalid" in result.blockers
    assert "packet:trusted_parent_recovery_binding_missing" in result.blockers


def test_same_provider_resume_keeps_zero_cost_resource_boundary() -> None:
    overlay = _same_provider_resume_overlay()
    overlay["resource"]["maximum_cost_usd"] = 1
    result = _bind(overlay)
    assert not result.binding_ready
    assert result.packet is None
    assert "overlay:overlay_resource_maximum_cost_usd_must_be_zero" in result.blockers

    overlay = _same_provider_resume_overlay()
    overlay["resource"]["materially_paid"] = True
    result = _bind(overlay)
    assert not result.binding_ready
    assert result.packet is None
    assert "overlay:overlay_resource_materially_paid_must_be_false" in result.blockers


def test_packet_container_type_drift_fails_closed_instead_of_raising() -> None:
    assert assess_portable_run_packet([]).contract_errors == ("packet_root_must_be_object",)

    ready = _bind(_ready_overlay())
    assert ready.binding_ready
    assert ready.packet is not None
    for path, replacement, expected in (
        (("status",), [], "status_invalid"),
        (("resource", "resource_class"), [], "resource_class_must_be_local_free_or_free_gpu"),
        (("resource", "provider"), {}, "resource_provider_invalid"),
        (("checkpoint", "mode"), [], "checkpoint_mode_invalid"),
        (("runtime", "backend_id"), [], "backend_id_not_qualified_candidate"),
    ):
        packet = copy.deepcopy(ready.packet)
        if len(path) == 1:
            packet[path[0]] = replacement
        else:
            packet[path[0]][path[1]] = replacement
        assessment = assess_portable_run_packet(packet)
        assert not assessment.contract_valid or not assessment.ready_for_initial_local_free_launch
        assert expected in (assessment.contract_errors + assessment.launch_blockers)

    packet = copy.deepcopy(ready.packet)
    packet["contract_fields"][0] = {}
    assessment = assess_portable_run_packet(packet)
    assert not assessment.contract_valid
    assert "contract_fields_must_be_strings" in assessment.contract_errors