from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

from twelve_six.learned20m_readiness import scientific_role_metadata
from twelve_six.portable_run_binding import (
    bind_portable_run_packet,
    canonical_sha256,
    validate_session_overlay_contract,
)
from twelve_six.readiness_trust_root import (
    authenticated_trusted_readiness_inputs,
    trusted_readiness_bundle_sha256,
)

ROOT = Path(__file__).resolve().parents[1]
READINESS = ROOT / "configs/research/r01_learned20m_launch_readiness_v1.json"
PACKET = ROOT / "configs/research/r01_portable_local_free_run_packet_v1.json"
OVERLAY = ROOT / "configs/research/r01_portable_session_overlay_v1.json"
SHA40 = "a" * 40
SHA64 = "b" * 64

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


def _authority_at(readiness: dict, path: tuple[str, ...]) -> object:
    value: object = readiness["evidence"]
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _trusted_bundle(readiness: dict) -> dict:
    evidence = readiness["evidence"]
    scientific: dict[str, object] = {}
    for role, path in _SCIENTIFIC_AUTHORITIES:
        scientific[role] = {
            "authority": copy.deepcopy(_authority_at(readiness, path)),
            "metadata": scientific_role_metadata(role, evidence),
        }
    return {
        "schema_version": 1,
        "scientific_authorities": scientific,
        "verified_authorization_refs": [],
    }


def _verified_inputs(readiness: dict) -> tuple[set[str], set[str], dict, str]:
    bundle = _trusted_bundle(readiness)
    expected = trusted_readiness_bundle_sha256(bundle)
    assert expected is not None
    resolved = authenticated_trusted_readiness_inputs(
        bundle,
        expected_identity_sha256=expected,
    )
    assert resolved is not None
    scientific, refs = resolved
    return scientific, refs, bundle, expected


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
    data["resource"]["provider"] = "OWNER_LAPTOP"
    data["output"]["artifact_store_uri"] = "file:///tmp/twelve-six-artifacts"
    return data


def _run_builder(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ROOT / "tools/build_r01_portable_run_packet.py"), *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


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


def test_ready_fresh_binding_is_exact_and_does_not_mutate_inputs() -> None:
    readiness = _ready_readiness()
    template = _load(PACKET)
    overlay = _ready_overlay()
    originals = copy.deepcopy((readiness, template, overlay))
    tokens, refs, _, _ = _verified_inputs(readiness)
    result = bind_portable_run_packet(
        readiness,
        template,
        overlay,
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
    assert result.packet_sha256 == canonical_sha256(result.packet)
    assert (readiness, template, overlay) == originals


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
    tokens, refs, _, _ = _verified_inputs(readiness)
    result = bind_portable_run_packet(
        readiness,
        _load(PACKET),
        overlay,
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
    tokens, refs, _, _ = _verified_inputs(readiness)
    result = bind_portable_run_packet(
        readiness,
        _load(PACKET),
        overlay,
        verified_scientific_authorities=tokens,
        verified_authorization_refs=refs,
    )
    assert not result.binding_ready
    assert result.packet is None
    assert "binding:code_authority_git_sha_mismatch" in result.blockers
    assert "binding:model_authority_modelspec_sha256_mismatch" in result.blockers


def test_backend_authority_must_bind_backend_and_environment() -> None:
    readiness = _ready_readiness()
    overlay = _ready_overlay()
    authority = overlay["scientific_bindings"]["authorities"]["backend"]
    authority["backend_id"] = "LITGPT"
    authority["environment_lock_sha256"] = "c" * 64
    tokens, refs, _, _ = _verified_inputs(readiness)
    result = bind_portable_run_packet(
        readiness,
        _load(PACKET),
        overlay,
        verified_scientific_authorities=tokens,
        verified_authorization_refs=refs,
    )
    assert not result.binding_ready
    assert "binding:backend_authority_backend_id_mismatch" in result.blockers
    assert "binding:backend_authority_environment_lock_sha256_mismatch" in result.blockers


def test_embedded_secret_and_overlay_drift_are_rejected() -> None:
    readiness = _ready_readiness()
    tokens, refs, _, _ = _verified_inputs(readiness)

    secret = _ready_overlay()
    secret["scientific_bindings"]["authorities"]["code"]["api_key"] = "forbidden"
    result = bind_portable_run_packet(
        readiness,
        _load(PACKET),
        secret,
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
    tokens, refs, _, _ = _verified_inputs(readiness)
    result = bind_portable_run_packet(
        readiness,
        _load(PACKET),
        _ready_overlay(),
        verified_scientific_authorities=tokens,
        verified_authorization_refs=refs,
    )
    assert result.readiness_ready
    assert not result.binding_ready
    assert "packet:max_exposures_per_unique_position_must_be_one" in result.blockers
    assert "packet:maximum_total_exposures_must_equal_unique_target" in result.blockers


def test_cli_writes_once_only_after_ready_binding(tmp_path: Path) -> None:
    readiness = _ready_readiness()
    tokens, _, bundle, expected = _verified_inputs(readiness)
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
    _, _, bundle, _ = _verified_inputs(readiness)
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
