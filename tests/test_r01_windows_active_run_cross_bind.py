from __future__ import annotations

import json
from pathlib import Path

import pytest

from twelve_six.windows_operator_preflight import (
    EXIT_ERROR,
    OperatorPreflightError,
    _canonical_sha256,
    _load_bound_run_identity,
    _load_current_run_identity,
    _load_run_identity,
    main,
)


def _manifest(run_id: str, *, git_sha: str = "c" * 40, world_size: int = 1) -> dict:
    return {
        "run_id": run_id,
        "candidate": {"git_sha": git_sha},
        "recovery": {"topology": {"world_size": world_size}},
    }


def _state(manifest: dict, *, phase: str = "RUNNING") -> dict:
    topology = manifest["recovery"]["topology"]
    payload = {
        "schema_version": "12-6.training-recovery.v1",
        "authority": "LOCAL_PROCESS_RECOVERY_POLICY_NOT_DISTRIBUTED_ELASTICITY",
        "run_id": manifest["run_id"],
        "run_manifest_sha256": _canonical_sha256(manifest),
        "source_sha": manifest["candidate"]["git_sha"],
        "topology_sha256": _canonical_sha256(topology),
        "world_size": topology["world_size"],
        "policy": {
            "checkpoint_every_steps": 1,
            "retain_last": 2,
            "max_restarts": 3,
            "max_preemptions": 8,
            "target_checkpoint_overhead_fraction": 0.05,
            "max_recovery_window_seconds": 900.0,
            "require_exact_topology_resume": True,
        },
        "phase": phase,
        "attempt": 1,
        "restarts_used": 0,
        "preemptions_seen": 0,
        "journal_reconstructed": False,
        "last_known_good": None,
        "checkpoint_count": 0,
        "invalid_checkpoint_directories": [],
        "failure_count": 0,
    }
    payload["state_sha256"] = _canonical_sha256(payload)
    return payload


def _write_root(tmp_path: Path, manifest: dict, *, phase: str = "RUNNING") -> Path:
    root = tmp_path / f"root-{manifest['run_id']}"
    root.mkdir()
    (root / "run-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (root / "recovery-state.json").write_text(
        json.dumps(_state(manifest, phase=phase)), encoding="utf-8"
    )
    return root


def test_shape_valid_manifest_b_cannot_replace_active_manifest_a(tmp_path: Path) -> None:
    active = _manifest("run-a", git_sha="a" * 40)
    active_root = _write_root(tmp_path, active)
    supplied = _manifest("run-b", git_sha="b" * 40)
    supplied_path = tmp_path / "synthetic-b.json"
    supplied_path.write_text(json.dumps(supplied), encoding="utf-8")

    # Candidate parsing alone deliberately carries no current-run authority.
    assert _load_run_identity(supplied_path) == ("run-b", _canonical_sha256(supplied))
    with pytest.raises(OperatorPreflightError, match="run_manifest_not_current_run"):
        _load_bound_run_identity(supplied_path, recovery_root=active_root)

    assert _load_bound_run_identity(
        active_root / "run-manifest.json", recovery_root=active_root
    ) == ("run-a", _canonical_sha256(active))


def test_recovery_state_is_independent_self_hashed_authority(tmp_path: Path) -> None:
    manifest = _manifest("run-a")
    root = _write_root(tmp_path, manifest)
    state_path = root / "recovery-state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["run_id"] = "run-b"
    state_path.write_text(json.dumps(state), encoding="utf-8")

    with pytest.raises(
        OperatorPreflightError, match="current_run_recovery_state_sha256_invalid"
    ):
        _load_current_run_identity(root)


def test_resealed_state_cannot_disagree_with_persisted_manifest(tmp_path: Path) -> None:
    manifest = _manifest("run-a", git_sha="a" * 40)
    root = _write_root(tmp_path, manifest)
    state_path = root / "recovery-state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["source_sha"] = "b" * 40
    state.pop("state_sha256")
    state["state_sha256"] = _canonical_sha256(state)
    state_path.write_text(json.dumps(state), encoding="utf-8")

    with pytest.raises(OperatorPreflightError, match="current_run_state_source_sha_mismatch"):
        _load_current_run_identity(root)


def test_terminal_recovery_root_cannot_authorize_new_stop(tmp_path: Path) -> None:
    manifest = _manifest("run-a")
    root = _write_root(tmp_path, manifest, phase="COMPLETED")
    with pytest.raises(OperatorPreflightError, match="current_run_recovery_state_not_active"):
        _load_current_run_identity(root)

    state = _state(manifest, phase="RUNNING")
    (root / "recovery-state.json").write_text(json.dumps(state), encoding="utf-8")
    (root / "terminal.json").write_text("{}", encoding="utf-8")
    with pytest.raises(OperatorPreflightError, match="terminal_marker_present"):
        _load_current_run_identity(root)


def test_request_stop_rejects_manifest_b_before_marker_side_effect(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    active = _manifest("run-a", git_sha="a" * 40)
    active_root = _write_root(tmp_path, active)
    supplied = _manifest("run-b", git_sha="b" * 40)
    supplied_path = tmp_path / "synthetic-b.json"
    supplied_path.write_text(json.dumps(supplied), encoding="utf-8")
    state_dir = tmp_path / "operator-state"

    # The checked-in profile/packet paths are supplied by the real repository in CI.
    root = Path(__file__).resolve().parents[1]
    result = main(
        [
            "--profile",
            str(root / "configs/research/r01_windows_local_free_operator_v1.json"),
            "--packet",
            str(root / "configs/research/r01_portable_local_free_run_packet_v1.json"),
            "--state-dir",
            str(state_dir),
            "--json",
            "request-stop",
            "--target",
            "20m",
            "--recovery-root",
            str(active_root),
            "--run-manifest",
            str(supplied_path),
        ]
    )
    assert result == EXIT_ERROR
    assert "run_manifest_not_current_run" in capsys.readouterr().out
    assert not state_dir.exists()
