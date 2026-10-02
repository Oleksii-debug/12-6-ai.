from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

from twelve_six.learned20m_readiness import scientific_role_metadata
from twelve_six.portable_run_binding import (
    bind_portable_run_packet,
    bind_preoptimizer_to_run_binding,
    canonical_sha256,
    validate_authenticated_portable_execution,
    validate_session_overlay_contract,
)
from twelve_six.portable_run_packet import validate_portable_run_contract
from twelve_six.preoptimizer_authority import PREOPTIMIZER_SCHEMA
from twelve_six.readiness_trust_root import (
    authenticated_trusted_launch_bundle,
    trusted_readiness_bundle_sha256,
)
from twelve_six.tokenization.decision_authority import DECISION as TOKENIZER_DECISION

ROOT = Path(__file__).resolve().parents[1]
READINESS = ROOT / "configs/research/r01_learned20m_launch_readiness_v1.json"
PACKET = ROOT / "configs/research/r01_portable_local_free_run_packet_v1.json"
OVERLAY = ROOT / "configs/research/r01_portable_session_overlay_v1.json"
SHA40 = "a" * 40
SHA64 = "b" * 64
MODEL341_INITSPEC_SHA256 = "86483c6df623e80cab2f73aba718863fce18af6fe3b12430c1348414d92b48a5"
LEARN345_POLICY_IDENTITY_SHA256 = "84152a673c4ed8fd34f4b81b03a96b4a3f5b40a22d961f436cbed11af23000e7"

_SCIENTIFIC_AUTHORITIES = (
    ("code", ("code", "authority")),
    ("corpus", ("corpus", "authority")),
    ("tokenizer", ("tokenizer", "authority")),
    ("loss_ledger", ("loss_ledger", "authority")),
    ("data_budget", ("loss_ledger", "data_budget_authority")),
    ("checkpoint_integrity", ("checkpoint_integrity", "authority")),
    ("evaluation_firewall", ("evaluation", "firewall_authority")),
    ("selection_validation", ("evaluation", "selection_validation_authority")),
    ("training_recipe", ("training_recipe", "authority")),
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _authority(*, git_sha: str = SHA40, **extra: object) -> dict:
    result = {
        "repository": "Oleksii-debug/12-6-ai.",
        "git_sha": git_sha,
        "evidence_sha256": SHA64,
        "workflow_run_id": 123,
        "workflow_conclusion": "success",
        "terminal": True,
    }
    result.update(extra)
    return result


def _ready_readiness() -> dict:
    data = _load(READINESS)
    evidence = data["evidence"]
    evidence["code"].update({"git_sha": SHA40, "authority": _authority()})
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
            "decision": TOKENIZER_DECISION,
            "decision_identity_sha256": "3" * 64,
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


def _authority_at(readiness: dict, path: tuple[str, ...]) -> object:
    value: object = readiness["evidence"]
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _ready_overlay() -> dict:
    data = _load(OVERLAY)
    model = _load(READINESS)["model_authority"]
    data["status"] = "READY_CANDIDATE"
    data["scientific_bindings"].update(
        {
            "initspec_sha256": MODEL341_INITSPEC_SHA256,
            "seed": 20260826,
            "optimizer_scheduler_precision": {
                "optimizer": "AdamW",
                "scheduler": "constant",
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
    data["resource"]["provider"] = "OWNER_LAPTOP"
    data["output"]["artifact_store_uri"] = "file:///tmp/twelve-six-artifacts"
    return data


def _portable_execution(readiness: dict, overlay: dict | None = None) -> dict:
    model = readiness["model_authority"]
    recipe = readiness["evidence"]["training_recipe"]
    session = copy.deepcopy(overlay if overlay is not None else _ready_overlay())
    session.pop("schema_version")
    session.pop("overlay_id")
    session.pop("status")
    return {
        "model": {
            "modelspec_sha256": model["modelspec_sha256"],
            "initspec_sha256": MODEL341_INITSPEC_SHA256,
            "parameter_count": model["parameter_count"],
            "canonical_base": model["canonical_base"],
        },
        "training": {
            "training_config_sha256": recipe["config_sha256"],
            "stopping_policy_sha256": recipe["stopping_policy_sha256"],
            "policy_identity_sha256": LEARN345_POLICY_IDENTITY_SHA256,
            "optimizer": "AdamW",
            "learning_rate": 0.00022,
            "betas": [0.9, 0.95],
            "eps": 1e-08,
            "weight_decay": 0.1,
            "gradient_clip_norm": 1.0,
            "scheduler": "constant",
            "warmup_steps": 0,
            "sequence_length": 128,
            "micro_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "precision": "fp32",
            "seed_vector": {
                "model_init": 20260826,
                "data_order": 20260826,
                "dataloader": 20260826,
            },
        },
        "session": session,
    }


def _preoptimizer(readiness: dict) -> dict:
    evidence = readiness["evidence"]
    tokenizer = evidence["tokenizer"]["identity_sha256"]
    packing = evidence["corpus"]["packing_sha256"]
    ledger = evidence["loss_ledger"]["identity_sha256"]
    positions = evidence["loss_ledger"]["unique_causal_loss_positions"]
    return {
        "schema": PREOPTIMIZER_SCHEMA,
        "launch_input": {
            "schema": "12-6.learned20m-launch-input-authority.v2",
            "authority_identity_sha256": "1" * 64,
            "tokenizer_identity_sha256": tokenizer,
            "packing_identity_sha256": packing,
            "unique_loss_ledger_identity_sha256": ledger,
            "one_pass_unique_nonignored_causal_loss_positions": positions,
        },
        "loss_bearing_content": {
            "schema": "12-6.d04-loss-bearing-content-manifest.v2",
            "manifest_identity_sha256": "2" * 64,
            "tokenizer_identity_sha256": tokenizer,
            "packing_identity_sha256": packing,
            "unique_loss_ledger_identity_sha256": ledger,
            "one_pass_unique_nonignored_causal_loss_positions": positions,
        },
        "tokenizer_decision": {
            "schema": "12-6.d04-learned20m-tokenizer-decision.v1",
            "decision": TOKENIZER_DECISION,
            "decision_identity_sha256": "3" * 64,
            "tokenizer_identity_sha256": tokenizer,
        },
        "ordered_exposure": {
            "plan_identity_sha256": "5" * 64,
            "preflight_identity_sha256": "6" * 64,
            "unique_loss_ledger_identity_sha256": ledger,
            "one_pass_unique_nonignored_causal_loss_positions": positions,
        },
        "resource_evidence": {
            "evidence_identity_sha256": "4" * 64,
            "resource_class": "LOCAL_FREE",
            "mechanics_scope": "forward+causal_ce+backward_only",
            "median_causal_targets_per_second": 609.5197637935258,
            "process_hwm_mib_approx": 478.56,
            "mechanics_only_lower_bound_seconds": 32812.71779527559,
            "cross_host_extrapolation_allowed": False,
            "paid_compute_used": False,
            "authorized_optimized_target_exposure": 0,
            "optimizer_updates_executed_on_real_targets": 0,
        },
    }


def _trusted_bundle(readiness: dict, overlay: dict | None = None) -> dict:
    evidence = readiness["evidence"]
    scientific: dict[str, object] = {}
    for role, path in _SCIENTIFIC_AUTHORITIES:
        scientific[role] = {
            "authority": copy.deepcopy(_authority_at(readiness, path)),
            "metadata": scientific_role_metadata(role, evidence),
        }
    return {
        "schema_version": 3,
        "scientific_authorities": scientific,
        "verified_authorization_refs": [],
        "portable_execution": _portable_execution(readiness, overlay),
        "preoptimizer_authorities": _preoptimizer(readiness),
    }


def _verified_inputs(
    readiness: dict,
    overlay: dict | None = None,
) -> tuple[set[str], set[str], dict, str, dict]:
    bundle = _trusted_bundle(readiness, overlay)
    expected = trusted_readiness_bundle_sha256(bundle)
    assert expected is not None
    resolved = authenticated_trusted_launch_bundle(
        bundle,
        expected_identity_sha256=expected,
    )
    assert resolved is not None
    scientific, refs, execution, preoptimizer = resolved
    assert execution is not None
    assert preoptimizer is not None
    return scientific, refs, bundle, expected, execution


def _run_builder(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ROOT / "tools/build_r01_portable_run_packet.py"), *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_portable_template_is_closed_world_for_final_test_adjacent_fields() -> None:
    template = _load(PACKET)
    assert validate_portable_run_contract(template) == []

    root_extra = copy.deepcopy(template)
    root_extra["final_test_manifest"] = {"sha256": SHA64}
    assert "packet_top_level_fields_mismatch" in validate_portable_run_contract(root_extra)

    evaluation_extra = copy.deepcopy(template)
    evaluation_extra["evaluation"]["final_test_payload_sha256"] = SHA64
    assert "evaluation_fields_mismatch" in validate_portable_run_contract(evaluation_extra)

    truth_extra = copy.deepcopy(template)
    truth_extra["truth_boundary"]["final_test_payload_count"] = 0
    assert "truth_boundary_fields_mismatch" in validate_portable_run_contract(truth_extra)


def test_portable_template_rejects_unknown_nested_runtime_and_recipe_fields() -> None:
    template = _load(PACKET)

    runtime_extra = copy.deepcopy(template)
    runtime_extra["runtime"]["environment_override"] = "caller-selected"
    assert "runtime_fields_mismatch" in validate_portable_run_contract(runtime_extra)

    recipe_extra = copy.deepcopy(template)
    recipe_extra["recipe"]["teacher_logits_sha256"] = SHA64
    assert "recipe_fields_mismatch" in validate_portable_run_contract(recipe_extra)

    optimizer_extra = copy.deepcopy(template)
    optimizer_extra["recipe"]["optimizer_scheduler_precision"]["loss_scaler"] = "caller"
    assert (
        "optimizer_scheduler_precision_fields_mismatch"
        in validate_portable_run_contract(optimizer_extra)
    )


def test_parent_checkpoint_authority_slot_exists_only_for_resume() -> None:
    fresh = _load(PACKET)
    assert fresh["checkpoint"]["mode"] == "FRESH_START"
    assert "parent_checkpoint" not in fresh["authorities"]
    assert validate_portable_run_contract(fresh) == []

    invalid_fresh = copy.deepcopy(fresh)
    invalid_fresh["authorities"]["parent_checkpoint"] = None
    assert "authority_slots_mismatch" in validate_portable_run_contract(invalid_fresh)

    resume = copy.deepcopy(fresh)
    resume["checkpoint"]["mode"] = "RESUME"
    resume["authorities"]["parent_checkpoint"] = None
    assert validate_portable_run_contract(resume) == []


def test_checked_in_overlay_contract_is_valid_but_deliberately_blocked() -> None:
    overlay = _load(OVERLAY)
    assert validate_session_overlay_contract(overlay) == []
    result = bind_portable_run_packet(_load(READINESS), _load(PACKET), overlay)
    assert not result.binding_ready
    assert result.packet is None
    assert "overlay:status_not_ready_candidate" in result.blockers
    assert "readiness:data_budget_not_qualified" in result.blockers


def test_ready_looking_packet_without_external_verification_remains_blocked() -> None:
    readiness = _ready_readiness()
    result = bind_portable_run_packet(readiness, _load(PACKET), _ready_overlay())
    assert not result.readiness_ready
    assert not result.binding_ready
    assert result.packet is None
    assert "readiness:terminal_corpus_authority_unverified" in result.blockers
    assert "readiness:training_recipe_authority_unverified" in result.blockers
    assert "binding:authenticated_portable_execution_missing" in result.blockers


def test_external_scientific_verification_without_execution_root_remains_blocked() -> None:
    readiness = _ready_readiness()
    overlay = _ready_overlay()
    tokens, refs, _, _, _ = _verified_inputs(readiness)
    result = bind_portable_run_packet(
        readiness,
        _load(PACKET),
        overlay,
        verified_scientific_authorities=tokens,
        verified_authorization_refs=refs,
    )
    assert result.readiness_ready
    assert not result.binding_ready
    assert result.packet is None
    assert "binding:authenticated_portable_execution_missing" in result.blockers


def test_ready_fresh_binding_is_exact_and_does_not_mutate_inputs() -> None:
    readiness = _ready_readiness()
    template = _load(PACKET)
    overlay = _ready_overlay()
    originals = copy.deepcopy((readiness, template, overlay))
    tokens, refs, _, _, execution = _verified_inputs(readiness)
    result = bind_portable_run_packet(
        readiness,
        template,
        overlay,
        expected_portable_execution=execution,
        verified_scientific_authorities=tokens,
        verified_authorization_refs=refs,
    )
    assert result.binding_ready
    assert result.mode == "FRESH_START"
    assert result.packet_contract_valid
    assert result.blockers == ()
    assert result.packet is not None
    assert result.packet["identities"]["source_git_sha"] == SHA40
    assert result.packet["binding"]["readiness_sha256"] == canonical_sha256(readiness)
    assert result.packet["binding"]["session_overlay_sha256"] == canonical_sha256(overlay)
    assert result.packet["binding"]["portable_execution_sha256"] == canonical_sha256(execution)
    assert result.packet_sha256 == canonical_sha256(result.packet)
    assert (readiness, template, overlay) == originals


def test_ready_packet_execution_projection_is_closed_world_and_cross_bound() -> None:
    readiness = _ready_readiness()
    overlay = _ready_overlay()
    tokens, refs, _, _, execution = _verified_inputs(readiness)
    result = bind_portable_run_packet(
        readiness,
        _load(PACKET),
        overlay,
        expected_portable_execution=execution,
        verified_scientific_authorities=tokens,
        verified_authorization_refs=refs,
    )
    assert result.binding_ready
    assert result.packet is not None
    packet = result.packet
    assert validate_portable_run_contract(packet) == []

    unknown = copy.deepcopy(packet)
    unknown["recipe"]["execution_projection"]["final_test_payload_uri"] = (
        "file:///forbidden"
    )
    assert (
        "execution_projection_fields_mismatch"
        in validate_portable_run_contract(unknown)
    )

    missing = copy.deepcopy(packet)
    missing["recipe"].pop("execution_projection")
    assert "execution_projection_missing" in validate_portable_run_contract(missing)

    config_drift = copy.deepcopy(packet)
    config_drift["recipe"]["execution_projection"]["training_config_sha256"] = (
        "c" * 64
    )
    assert (
        "execution_projection_training_config_sha256_mismatch"
        in validate_portable_run_contract(config_drift)
    )

    optimizer_drift = copy.deepcopy(packet)
    optimizer_drift["recipe"]["execution_projection"]["optimizer"] = "SGD"
    assert (
        "execution_projection_optimizer_mismatch"
        in validate_portable_run_contract(optimizer_drift)
    )

    seed_alias = copy.deepcopy(packet)
    seed_alias["recipe"]["execution_projection"]["seed_vector"]["model_init"] = True
    assert (
        "execution_projection_seed_vector_model_init_invalid"
        in validate_portable_run_contract(seed_alias)
    )


def test_ready_packet_execution_projection_rejects_nonfinite_and_invalid_ranges() -> None:
    readiness = _ready_readiness()
    overlay = _ready_overlay()
    tokens, refs, _, _, execution = _verified_inputs(readiness)
    result = bind_portable_run_packet(
        readiness,
        _load(PACKET),
        overlay,
        expected_portable_execution=execution,
        verified_scientific_authorities=tokens,
        verified_authorization_refs=refs,
    )
    assert result.binding_ready
    assert result.packet is not None

    for field, invalid in (
        ("learning_rate", float("nan")),
        ("eps", float("inf")),
        ("gradient_clip_norm", float("-inf")),
    ):
        packet = copy.deepcopy(result.packet)
        packet["recipe"]["execution_projection"][field] = invalid
        assert (
            f"execution_projection_{field}_invalid"
            in validate_portable_run_contract(packet)
        )

    negative_decay = copy.deepcopy(result.packet)
    negative_decay["recipe"]["execution_projection"]["weight_decay"] = -0.1
    assert (
        "execution_projection_weight_decay_must_be_nonnegative"
        in validate_portable_run_contract(negative_decay)
    )

    invalid_betas = copy.deepcopy(result.packet)
    invalid_betas["recipe"]["execution_projection"]["betas"] = [0.9, float("nan")]
    assert (
        "execution_projection_betas_invalid"
        in validate_portable_run_contract(invalid_betas)
    )


def test_ready_cross_provider_resume_binds_parent_lineage() -> None:
    readiness = _ready_readiness()
    overlay = _ready_overlay()
    overlay["checkpoint"].update(
        {
            "mode": "RESUME",
            "lineage": {
                "parent_checkpoint_sha256": SHA64,
                "parent_manifest_sha256": SHA64,
                "previous_run_id": "R01-SESSION-001",
                "source_provider": "OWNER_LAPTOP",
                "cross_provider_transfer": True,
                "resume_validated": True,
            },
            "parent_checkpoint_authority": _authority(),
        }
    )
    overlay["resource"].update({"resource_class": "FREE_GPU", "provider": "KAGGLE"})
    overlay["runtime"]["device_type"] = "cuda"
    overlay["output"]["artifact_store_uri"] = "https://artifacts.example/sha256"
    tokens, refs, _, _, execution = _verified_inputs(readiness, overlay)
    result = bind_portable_run_packet(
        readiness,
        _load(PACKET),
        overlay,
        expected_portable_execution=execution,
        verified_scientific_authorities=tokens,
        verified_authorization_refs=refs,
    )
    assert result.binding_ready
    assert result.mode == "RESUME"
    assert result.packet is not None
    assert result.packet["checkpoint"]["lineage"]["previous_run_id"] == "R01-SESSION-001"


def test_authority_identity_mismatch_fails_closed() -> None:
    readiness = _ready_readiness()
    overlay = _ready_overlay()
    overlay["scientific_bindings"]["authorities"]["code"]["git_sha"] = "c" * 40
    overlay["scientific_bindings"]["authorities"]["model"]["modelspec_sha256"] = "d" * 64
    tokens, refs, _, _, execution = _verified_inputs(readiness)
    result = bind_portable_run_packet(
        readiness,
        _load(PACKET),
        overlay,
        expected_portable_execution=execution,
        verified_scientific_authorities=tokens,
        verified_authorization_refs=refs,
    )
    assert not result.binding_ready
    assert result.packet is None
    assert "binding:authenticated_session_projection_mismatch" in result.blockers
    assert "binding:code_authority_git_sha_mismatch" in result.blockers
    assert "binding:model_authority_modelspec_sha256_mismatch" in result.blockers


def test_backend_authority_must_bind_backend_and_environment() -> None:
    readiness = _ready_readiness()
    overlay = _ready_overlay()
    authority = overlay["scientific_bindings"]["authorities"]["backend"]
    authority["backend_id"] = "LITGPT"
    authority["environment_lock_sha256"] = "c" * 64
    tokens, refs, _, _, execution = _verified_inputs(readiness)
    result = bind_portable_run_packet(
        readiness,
        _load(PACKET),
        overlay,
        expected_portable_execution=execution,
        verified_scientific_authorities=tokens,
        verified_authorization_refs=refs,
    )
    assert not result.binding_ready
    assert "binding:authenticated_session_projection_mismatch" in result.blockers
    assert "binding:backend_authority_backend_id_mismatch" in result.blockers
    assert "binding:backend_authority_environment_lock_sha256_mismatch" in result.blockers


def test_embedded_secret_and_overlay_drift_are_rejected() -> None:
    readiness = _ready_readiness()
    tokens, refs, _, _, execution = _verified_inputs(readiness)

    secret = _ready_overlay()
    secret["scientific_bindings"]["authorities"]["code"]["api_key"] = "forbidden"
    result = bind_portable_run_packet(
        readiness,
        _load(PACKET),
        secret,
        expected_portable_execution=execution,
        verified_scientific_authorities=tokens,
        verified_authorization_refs=refs,
    )
    assert not result.binding_ready
    assert not result.packet_contract_valid
    assert any("embedded_secret_forbidden" in blocker for blocker in result.blockers)

    drift = _ready_overlay()
    drift["runtime"]["container_image"] = "unreviewed:latest"
    result = bind_portable_run_packet(
        readiness,
        _load(PACKET),
        drift,
        expected_portable_execution=execution,
        verified_scientific_authorities=tokens,
        verified_authorization_refs=refs,
    )
    assert not result.binding_ready
    assert "overlay:overlay_runtime_container_image_unexpected" in result.blockers


def test_binding_does_not_relax_one_pass_unique_exposure_rule() -> None:
    readiness = _ready_readiness()
    recipe = readiness["evidence"]["training_recipe"]
    recipe["requested_unique_loss_positions"] = 500
    recipe["requested_total_training_exposures"] = 1000
    recipe["max_exposures_per_unique_position"] = 2
    tokens, refs, _, _, execution = _verified_inputs(readiness)
    result = bind_portable_run_packet(
        readiness,
        _load(PACKET),
        _ready_overlay(),
        expected_portable_execution=execution,
        verified_scientific_authorities=tokens,
        verified_authorization_refs=refs,
    )
    assert result.readiness_ready
    assert not result.binding_ready
    assert "packet:max_exposures_per_unique_position_must_be_one" in result.blockers
    assert "packet:maximum_total_exposures_must_equal_unique_target" in result.blockers


def test_cli_writes_once_only_after_ready_binding(tmp_path: Path) -> None:
    readiness = _ready_readiness()
    tokens, _, bundle, expected, _ = _verified_inputs(readiness)
    readiness_path = tmp_path / "readiness.json"
    template_path = tmp_path / "packet.json"
    overlay_path = tmp_path / "overlay.json"
    bindings_path = tmp_path / "trusted-bindings.json"
    output_path = tmp_path / "bound.json"
    for path, value in (
        (readiness_path, readiness),
        (template_path, _load(PACKET)),
        (overlay_path, _ready_overlay()),
        (bindings_path, bundle),
    ):
        path.write_text(json.dumps(value), encoding="utf-8")

    args = [
        "--readiness",
        str(readiness_path),
        "--template",
        str(template_path),
        "--overlay",
        str(overlay_path),
        "--trusted-bindings",
        str(bindings_path),
        "--expected-trusted-bindings-sha256",
        expected,
        "--output",
        str(output_path),
    ]

    first = _run_builder(args)
    assert first.returncode == 0, first.stderr or first.stdout
    output_text = output_path.read_text(encoding="utf-8")
    assert json.loads(output_text)["status"] == "READY_CANDIDATE"
    assert all(token not in output_text for token in tokens)
    assert all(token not in first.stdout for token in tokens)

    second = _run_builder(args)
    assert second.returncode == 2
    assert "refusing to overwrite existing output" in second.stdout


def test_cli_rejects_trusted_bundle_without_external_expected_root(tmp_path: Path) -> None:
    readiness = _ready_readiness()
    _, _, bundle, _, _ = _verified_inputs(readiness)
    readiness_path = tmp_path / "readiness.json"
    bindings_path = tmp_path / "trusted-bindings.json"
    readiness_path.write_text(json.dumps(readiness), encoding="utf-8")
    bindings_path.write_text(json.dumps(bundle), encoding="utf-8")
    result = _run_builder(
        [
            "--readiness",
            str(readiness_path),
            "--template",
            str(PACKET),
            "--overlay",
            str(OVERLAY),
            "--trusted-bindings",
            str(bindings_path),
        ]
    )
    assert result.returncode == 2
    assert "--expected-trusted-bindings-sha256" in result.stdout


def test_cli_no_longer_accepts_raw_verified_token_injection() -> None:
    result = _run_builder(["--verified-scientific-authority-token", "a" * 64])
    assert result.returncode == 2
    assert "unrecognized arguments" in result.stderr


def test_cli_never_writes_current_blocked_inputs(tmp_path: Path) -> None:
    output_path = tmp_path / "must-not-exist.json"
    args = [
        "--readiness",
        str(READINESS),
        "--template",
        str(PACKET),
        "--overlay",
        str(OVERLAY),
        "--output",
        str(output_path),
    ]
    result = _run_builder(args)
    assert result.returncode == 1, result.stderr or result.stdout
    assert not output_path.exists()


def test_preoptimizer_finalization_rehashes_exact_runtime_packet_and_binds_d10_root() -> None:
    readiness = _ready_readiness()
    overlay = _ready_overlay()
    tokens, refs, _, expected, execution = _verified_inputs(readiness, overlay)
    preliminary = bind_portable_run_packet(
        readiness,
        _load(PACKET),
        overlay,
        expected_portable_execution=execution,
        verified_scientific_authorities=tokens,
        verified_authorization_refs=refs,
    )
    assert preliminary.binding_ready
    assert preliminary.packet is not None
    assert "launch_input_authority_identity_sha256" not in preliminary.packet["binding"]
    assert preliminary.packet["binding"]["tokenizer_decision_identity_sha256"] == "3" * 64

    final = bind_preoptimizer_to_run_binding(
        preliminary,
        _preoptimizer(readiness),
        trusted_readiness_bundle_sha256=expected,
    )
    assert final.binding_ready
    assert final.packet is not None
    assert final.packet is not preliminary.packet
    assert final.packet["binding"]["launch_input_authority_identity_sha256"] == "1" * 64
    assert final.launch_input_authority_identity_sha256 == "1" * 64
    assert final.loss_bearing_manifest_identity_sha256 == "2" * 64
    assert final.exposure_plan_identity_sha256 == "5" * 64
    assert final.packet["binding"]["loss_bearing_manifest_identity_sha256"] == "2" * 64
    assert final.packet["binding"]["exposure_plan_identity_sha256"] == "5" * 64
    assert final.trusted_readiness_bundle_sha256 == expected
    assert final.preoptimizer_authorities_sha256 == final.packet["binding"][
        "preoptimizer_authorities_sha256"
    ]
    assert final.packet_sha256 == canonical_sha256(final.packet)
    assert final.packet_sha256 != preliminary.packet_sha256



def test_preoptimizer_finalization_rejects_tokenizer_decision_identity_substitution() -> None:
    readiness = _ready_readiness()
    overlay = _ready_overlay()
    tokens, refs, _, expected, execution = _verified_inputs(readiness, overlay)
    preliminary = bind_portable_run_packet(
        readiness,
        _load(PACKET),
        overlay,
        expected_portable_execution=execution,
        verified_scientific_authorities=tokens,
        verified_authorization_refs=refs,
    )
    substituted = _preoptimizer(readiness)
    substituted["tokenizer_decision"]["decision_identity_sha256"] = "9" * 64
    try:
        bind_preoptimizer_to_run_binding(
            preliminary,
            substituted,
            trusted_readiness_bundle_sha256=expected,
        )
    except ValueError as exc:
        assert "tokenizer_decision_authority_identity_packet_mismatch" in str(exc)
    else:
        raise AssertionError("tokenizer decision identity substitution must fail closed")


def test_matched_overlay_and_execution_unknown_backend_authority_field_fail_closed() -> None:
    readiness = _ready_readiness()
    overlay = _ready_overlay()
    tokens, refs, _, _, execution = _verified_inputs(readiness)
    forbidden_field = "final_test_payload_uri"
    overlay["scientific_bindings"]["authorities"]["backend"][forbidden_field] = (
        "file:///forbidden"
    )
    execution["session"]["scientific_bindings"]["authorities"]["backend"][
        forbidden_field
    ] = "file:///forbidden"

    result = bind_portable_run_packet(
        readiness,
        _load(PACKET),
        overlay,
        expected_portable_execution=execution,
        verified_scientific_authorities=tokens,
        verified_authorization_refs=refs,
    )

    assert not result.binding_ready
    assert result.packet is None
    assert (
        "overlay:overlay_authority_backend_final_test_payload_uri_unexpected"
        in result.blockers
    )
    assert (
        "binding:authenticated_portable_execution_session:"
        "overlay_authority_backend_final_test_payload_uri_unexpected"
        in result.blockers
    )


def test_portable_packet_authorities_are_role_specific_closed_world() -> None:
    template = _load(PACKET)
    template["authorities"]["code"] = _authority()
    template["authorities"]["model"] = _authority(modelspec_sha256=SHA64)
    template["authorities"]["backend"] = _authority(
        backend_id="PROJECT_NATIVE_PYTORCH",
        environment_lock_sha256=SHA64,
    )
    errors = validate_portable_run_contract(template)
    assert "code_authority_fields_mismatch" not in errors
    assert "model_authority_fields_mismatch" not in errors
    assert "backend_authority_fields_mismatch" not in errors

    for role in ("code", "model", "backend"):
        tampered = copy.deepcopy(template)
        tampered["authorities"][role]["final_test_payload_uri"] = "file:///forbidden"
        assert (
            f"{role}_authority_fields_mismatch"
            in validate_portable_run_contract(tampered)
        )


def test_resume_parent_checkpoint_authority_is_closed_world() -> None:
    packet = _load(PACKET)
    packet["checkpoint"]["mode"] = "RESUME"
    packet["authorities"]["parent_checkpoint"] = _authority()
    assert (
        "parent_checkpoint_authority_fields_mismatch"
        not in validate_portable_run_contract(packet)
    )

    packet["authorities"]["parent_checkpoint"]["final_test_payload_uri"] = (
        "file:///forbidden"
    )
    assert (
        "parent_checkpoint_authority_fields_mismatch"
        in validate_portable_run_contract(packet)
    )


def test_fresh_overlay_rejects_unused_parent_checkpoint_authority() -> None:
    overlay = _ready_overlay()
    overlay["checkpoint"]["parent_checkpoint_authority"] = _authority()
    assert (
        "overlay_parent_checkpoint_authority_forbidden_for_fresh_start"
        in validate_session_overlay_contract(overlay)
    )


def test_authenticated_training_projection_rejects_bool_int_aliases() -> None:
    readiness = _ready_readiness()
    cases = (
        ("warmup_steps", False),
        ("micro_batch_size", True),
        ("gradient_accumulation_steps", True),
        ("gradient_clip_norm", True),
    )
    for field, invalid in cases:
        execution = _portable_execution(readiness)
        execution["training"][field] = invalid
        errors = validate_authenticated_portable_execution(execution)
        assert (
            f"authenticated_portable_execution_training_{field}_mismatch"
            in errors
        )


def test_authenticated_training_projection_rejects_numeric_type_drift() -> None:
    readiness = _ready_readiness()
    execution = _portable_execution(readiness)
    execution["training"]["gradient_clip_norm"] = 1
    errors = validate_authenticated_portable_execution(execution)
    assert (
        "authenticated_portable_execution_training_gradient_clip_norm_mismatch"
        in errors
    )
