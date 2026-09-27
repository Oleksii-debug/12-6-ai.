from __future__ import annotations

import json
import shutil
import subprocess
from copy import deepcopy
from pathlib import Path

import pytest

import twelve_six.learned20m_current_run_authority as current_run
from twelve_six.learned20m_current_run_authority import (
    CURRENT_RUN_POINTER_REF,
    CurrentRunAuthorityInspection,
    activate_current_run_authority,
    build_current_run_identity,
    build_current_run_pointer_state,
    decode_current_run_pointer_state,
    inspect_current_run_authority,
    retire_current_run_authority,
    validate_current_run_identity,
    validate_current_run_pointer_state,
    verify_candidate_against_current_run,
)
from twelve_six.learned20m_global_training_lease import (
    GlobalLeaseInspection,
    global_training_run_lease_ref,
)
from twelve_six.learned20m_training_lease import (
    canonical_json_bytes,
    launch_manifest_sha256,
)


def _manifest(*, source_git_sha: str = "b" * 40, binding: str = "a" * 64) -> dict:
    hashes = {
        "modelspec_sha256": "1" * 64,
        "initspec_sha256": "2" * 64,
        "tokenizer_sha256": "3" * 64,
        "corpus_manifest_sha256": "4" * 64,
        "split_sha256": "5" * 64,
        "packing_sha256": "6" * 64,
        "unique_loss_ledger_sha256": "7" * 64,
        "training_config_sha256": "8" * 64,
        "portable_run_packet_sha256": "9" * 64,
        "portable_run_binding_sha256": binding,
    }
    return {
        "schema_version": 1,
        "manifest_id": "R01-LEARNED20M-LAUNCH-MANIFEST-V1",
        "stage": "LEARNED_20M",
        "identities": {"source_git_sha": source_git_sha, **hashes},
        "recipe": {
            "optimizer_scheduler_precision": "AdamW|cosine|fp32",
            "seed": 1337,
            "target_unique_loss_positions": 20_000_000,
            "maximum_total_exposures": 24_000_000,
        },
        "checkpoint": {
            "lineage": "learned20m-v1",
            "checkpoint_contract_sha256": "c" * 64,
        },
        "evaluation": {
            "firewall_sha256": "d" * 64,
            "final_test_payload_access": False,
        },
        "resource": {
            "resource_class": "LOCAL_FREE",
            "maximum_cost_usd": 0,
            "materially_paid": False,
        },
        "authorities": {
            "training": {
                "reference": "github:issue/653#comment-training",
                "evidence_sha256": "e" * 64,
            },
            "compute": {
                "reference": "github:issue/653#comment-compute",
                "evidence_sha256": "f" * 64,
            },
        },
        "execution_backend": "PROJECT_NATIVE_PYTORCH",
    }


def _identity(
    *,
    run_id: str = "run-a",
    recovery_manifest: str = "1" * 64,
    binding: str = "a" * 64,
    source_git_sha: str = "b" * 40,
) -> dict:
    return build_current_run_identity(
        run_id=run_id,
        recovery_run_manifest_sha256=recovery_manifest,
        recovery_attempt_authority_sha256="2" * 64,
        portable_run_binding_sha256=binding,
        source_git_sha=source_git_sha,
    )


def _global_inspection(
    manifest: dict,
    *,
    run_id: str = "run-a",
    remote_tip: str = "c" * 40,
) -> GlobalLeaseInspection:
    return GlobalLeaseInspection(
        present=True,
        valid=True,
        ref=global_training_run_lease_ref(manifest),
        remote_tip=remote_tip,
        launch_manifest_sha256=launch_manifest_sha256(manifest),
        run_id=run_id,
        lease_status="RUNNING",
        renewal_sequence=0,
        blockers=(),
    )


def _git(*args: str, cwd: Path | None = None) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


@pytest.fixture
def git_pair(tmp_path: Path) -> tuple[Path, Path, Path]:
    if shutil.which("git") is None:
        pytest.skip("git executable unavailable")
    remote = tmp_path / "remote.git"
    writer_a = tmp_path / "writer-a"
    writer_b = tmp_path / "writer-b"
    _git("init", "--bare", str(remote))
    _git("init", str(writer_a))
    _git("init", str(writer_b))
    return remote, writer_a, writer_b


def test_current_run_identity_is_closed_world_and_self_hashed() -> None:
    identity = _identity()
    assert validate_current_run_identity(identity) == ()
    assert identity["identity_sha256"] == current_run._identity_digest(identity)

    substituted = deepcopy(identity)
    substituted["recovery_run_manifest_sha256"] = "3" * 64
    assert "current_run_identity_self_hash_mismatch" in validate_current_run_identity(
        substituted
    )

    widened = deepcopy(identity)
    widened["training_authorized"] = True
    assert "current_run_identity_fields_mismatch" in validate_current_run_identity(
        widened
    )


def test_pointer_state_binds_incumbent_manifest_global_lease_and_run() -> None:
    manifest = _manifest()
    identity = _identity()
    global_lease = _global_inspection(manifest)
    state = build_current_run_pointer_state(
        manifest, global_lease, identity, generation=1
    )

    assert validate_current_run_pointer_state(state) == ()
    assert state["launch_manifest_sha256"] == launch_manifest_sha256(manifest)
    assert state["global_lease_ref"] == global_training_run_lease_ref(manifest)
    assert state["current_run_identity"]["run_id"] == "run-a"

    wrong_run = _global_inspection(manifest, run_id="run-b")
    with pytest.raises(ValueError, match="global_lease_run_id_mismatch"):
        build_current_run_pointer_state(manifest, wrong_run, identity, generation=1)

    wrong_ref = deepcopy(global_lease)
    object.__setattr__(wrong_ref, "ref", "refs/heads/attacker")
    with pytest.raises(ValueError, match="global_lease_ref_mismatch"):
        build_current_run_pointer_state(manifest, wrong_ref, identity, generation=1)


def test_pointer_decoder_rejects_noncanonical_and_unknown_fields() -> None:
    manifest = _manifest()
    state = build_current_run_pointer_state(
        manifest, _global_inspection(manifest), _identity(), generation=1
    )
    assert decode_current_run_pointer_state(canonical_json_bytes(state)) == state

    pretty = json.dumps(state, indent=2, sort_keys=True).encode()
    with pytest.raises(ValueError, match="current_run_pointer_not_canonical"):
        decode_current_run_pointer_state(pretty)

    widened = deepcopy(state)
    widened["optimizer_start_permitted"] = True
    widened["pointer_identity_sha256"] = current_run._pointer_digest(widened)
    with pytest.raises(ValueError, match="current_run_pointer_fields_mismatch"):
        decode_current_run_pointer_state(canonical_json_bytes(widened))


def test_candidate_b_cannot_self_select_namespace_when_a_is_current() -> None:
    manifest_a = _manifest()
    identity_a = _identity()
    state_a = build_current_run_pointer_state(
        manifest_a, _global_inspection(manifest_a), identity_a, generation=1
    )
    inspection = CurrentRunAuthorityInspection(
        present=True,
        valid=True,
        active=True,
        ref=CURRENT_RUN_POINTER_REF,
        remote_tip="d" * 40,
        generation=1,
        pointer_identity_sha256=state_a["pointer_identity_sha256"],
        launch_manifest_sha256=state_a["launch_manifest_sha256"],
        global_lease_ref=state_a["global_lease_ref"],
        global_lease_remote_tip=state_a["global_lease_remote_tip"],
        run_id="run-a",
        recovery_run_manifest_sha256="1" * 64,
        current_run_identity_sha256=identity_a["identity_sha256"],
        blockers=(),
    )

    assert (
        verify_candidate_against_current_run(
            inspection,
            launch_manifest_sha256_value=launch_manifest_sha256(manifest_a),
            run_id="run-a",
            recovery_run_manifest_sha256="1" * 64,
            current_run_identity_sha256=identity_a["identity_sha256"],
        )
        == ()
    )

    manifest_b = _manifest(binding="f" * 64)
    identity_b = _identity(
        run_id="run-b", recovery_manifest="4" * 64, binding="f" * 64
    )
    blockers = verify_candidate_against_current_run(
        inspection,
        launch_manifest_sha256_value=launch_manifest_sha256(manifest_b),
        run_id="run-b",
        recovery_run_manifest_sha256="4" * 64,
        current_run_identity_sha256=identity_b["identity_sha256"],
    )
    assert set(blockers) == {
        "current_run_launch_manifest_mismatch",
        "current_run_id_mismatch",
        "current_run_recovery_manifest_mismatch",
        "current_run_identity_mismatch",
    }


def test_fixed_pointer_activation_retirement_and_generation(
    git_pair: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote, writer_a, writer_b = git_pair
    manifest_a = _manifest()
    identity_a = _identity()
    inspection_a = _global_inspection(manifest_a)

    monkeypatch.setattr(
        current_run,
        "inspect_global_training_run_lease",
        lambda repo_root, remote_value, manifest: inspection_a,
    )
    first = activate_current_run_authority(
        writer_a,
        str(remote),
        manifest_a,
        identity_a,
        expected_pointer_tip=None,
    )
    assert first.committed is True
    assert first.post_write_reread_verified is True
    assert first.generation == 1

    current = inspect_current_run_authority(writer_b, str(remote))
    assert current.present is True
    assert current.valid is True
    assert current.active is True
    assert current.ref == CURRENT_RUN_POINTER_REF
    assert current.run_id == "run-a"

    duplicate = activate_current_run_authority(
        writer_b,
        str(remote),
        manifest_a,
        identity_a,
        expected_pointer_tip=first.written_remote_tip,
    )
    assert duplicate.committed is False
    assert duplicate.blockers == ("current_run_already_active",)

    retired = retire_current_run_authority(
        writer_a,
        str(remote),
        expected_pointer_tip=first.written_remote_tip,
        expected_current_run_identity_sha256=identity_a["identity_sha256"],
    )
    assert retired.committed is True
    assert retired.post_write_reread_verified is True
    assert inspect_current_run_authority(writer_b, str(remote)).active is False

    stale = activate_current_run_authority(
        writer_b,
        str(remote),
        manifest_a,
        identity_a,
        expected_pointer_tip=first.written_remote_tip,
    )
    assert stale.committed is False
    assert stale.blockers == ("current_run_pointer_expected_tip_mismatch",)


def test_retired_pointer_can_advance_only_from_exact_latest_tip(
    git_pair: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote, writer_a, writer_b = git_pair
    manifest_a = _manifest()
    identity_a = _identity()
    manifest_b = _manifest(source_git_sha="c" * 40, binding="f" * 64)
    identity_b = _identity(
        run_id="run-b",
        recovery_manifest="4" * 64,
        binding="f" * 64,
        source_git_sha="c" * 40,
    )

    def inspect_global(repo_root, remote_value, manifest):
        if launch_manifest_sha256(manifest) == launch_manifest_sha256(manifest_a):
            return _global_inspection(manifest_a)
        return _global_inspection(manifest_b, run_id="run-b", remote_tip="e" * 40)

    monkeypatch.setattr(current_run, "inspect_global_training_run_lease", inspect_global)

    first = activate_current_run_authority(
        writer_a, str(remote), manifest_a, identity_a, expected_pointer_tip=None
    )
    retired = retire_current_run_authority(
        writer_a,
        str(remote),
        expected_pointer_tip=first.written_remote_tip,
        expected_current_run_identity_sha256=identity_a["identity_sha256"],
    )
    assert retired.committed is True

    second = activate_current_run_authority(
        writer_b,
        str(remote),
        manifest_b,
        identity_b,
        expected_pointer_tip=retired.written_remote_tip,
    )
    assert second.committed is True
    assert second.generation == 2
    current = inspect_current_run_authority(writer_a, str(remote))
    assert current.active is True
    assert current.generation == 2
    assert current.run_id == "run-b"
    assert current.launch_manifest_sha256 == launch_manifest_sha256(manifest_b)


def test_boolean_generation_and_source_or_binding_substitution_fail_closed() -> None:
    manifest = _manifest()
    identity = _identity()
    with pytest.raises(ValueError, match="generation_must_be_positive_integer"):
        build_current_run_pointer_state(
            manifest, _global_inspection(manifest), identity, generation=True
        )

    wrong_source = _identity(source_git_sha="c" * 40)
    with pytest.raises(ValueError, match="current_run_source_git_sha_mismatch"):
        build_current_run_pointer_state(
            manifest, _global_inspection(manifest), wrong_source, generation=1
        )

    wrong_binding = _identity(binding="f" * 64)
    with pytest.raises(ValueError, match="current_run_portable_binding_mismatch"):
        build_current_run_pointer_state(
            manifest, _global_inspection(manifest), wrong_binding, generation=1
        )
