from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from twelve_six.windows_operator_preflight import (
    OperatorPreflightError,
    _canonical_sha256,
    _load_run_identity,
    read_stop_status,
    request_safe_stop,
)


def _canonical_manifest(
    *,
    run_id: Any = "run-a",
    git_sha: Any = "c" * 40,
    topology: Any = None,
) -> dict[str, Any]:
    if topology is None:
        topology = {"world_size": 1}
    return {
        "run_id": run_id,
        "candidate": {"git_sha": git_sha},
        "recovery": {"topology": topology},
    }


def _write_manifest(tmp_path: Path, manifest: dict[str, Any]) -> Path:
    path = tmp_path / "run-manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    return path


def _identity(manifest: dict[str, Any]) -> tuple[str, str]:
    return str(manifest["run_id"]), _canonical_sha256(manifest)


def _request(tmp_path: Path) -> tuple[Path, str, str]:
    state = tmp_path / "state"
    manifest = _canonical_manifest()
    run_id, manifest_sha = _identity(manifest)
    request_safe_stop(
        state,
        profile_sha256="a" * 64,
        packet_sha256="b" * 64,
        target="20m",
        run_id=run_id,
        run_manifest_sha256=manifest_sha,
        requested_at_utc="2026-09-14T20:00:00Z",
    )
    return state, run_id, manifest_sha


def test_run_id_only_json_cannot_mint_current_run_authority(tmp_path: Path) -> None:
    path = _write_manifest(tmp_path, {"run_id": "run-a"})
    with pytest.raises(OperatorPreflightError, match="safe_stop_run_manifest_candidate_invalid"):
        _load_run_identity(path)


@pytest.mark.parametrize(
    ("manifest", "error"),
    [
        (
            {
                "run_id": "run-a",
                "candidate": {},
                "recovery": {"topology": {"world_size": 1}},
            },
            "safe_stop_run_manifest_candidate_git_sha_invalid",
        ),
        (
            _canonical_manifest(git_sha="C" * 40),
            "safe_stop_run_manifest_candidate_git_sha_invalid",
        ),
        (
            {"run_id": "run-a", "candidate": {"git_sha": "c" * 40}},
            "safe_stop_run_manifest_recovery_invalid",
        ),
        (
            _canonical_manifest(topology={}),
            "safe_stop_run_manifest_topology_invalid",
        ),
        (
            _canonical_manifest(topology={"world_size": True}),
            "safe_stop_run_manifest_world_size_invalid",
        ),
        (
            _canonical_manifest(topology={"world_size": 0}),
            "safe_stop_run_manifest_world_size_invalid",
        ),
    ],
)
def test_run_manifest_must_match_recovery_store_identity_contract(
    tmp_path: Path, manifest: dict[str, Any], error: str
) -> None:
    path = _write_manifest(tmp_path, manifest)
    with pytest.raises(OperatorPreflightError, match=error):
        _load_run_identity(path)


@pytest.mark.parametrize("git_sha", ["c" * 40, "d" * 64])
def test_recovery_store_compatible_manifest_binds_exact_identity(
    tmp_path: Path, git_sha: str
) -> None:
    manifest = _canonical_manifest(git_sha=git_sha, topology={"world_size": 2})
    run_id, digest = _load_run_identity(_write_manifest(tmp_path, manifest))
    assert run_id == "run-a"
    assert digest == _canonical_sha256(manifest)


def test_self_resealed_unknown_marker_field_fails_closed(tmp_path: Path) -> None:
    state, run_id, manifest_sha = _request(tmp_path)
    marker_path = state / "STOP_REQUEST.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker.pop("marker_sha256")
    marker["future_field"] = "self-resealed-but-unauthorized"
    marker["marker_sha256"] = _canonical_sha256(marker)
    marker_path.write_text(json.dumps(marker), encoding="utf-8")

    status = read_stop_status(
        state,
        profile_sha256="a" * 64,
        packet_sha256="b" * 64,
        target="20m",
        run_id=run_id,
        run_manifest_sha256=manifest_sha,
    )
    assert status["status"] == "INVALID"
    assert status["marker"] is None
    assert "safe_stop_marker_keys_mismatch" in status["errors"]


@pytest.mark.parametrize(
    "timestamp",
    [
        None,
        True,
        7,
        "",
        "2026-09-14",
        "2026-09-14T20:00Z",
        "2026-09-14T20:00:00+00:00",
        "2026-09-14T20:00:00",
    ],
)
def test_self_resealed_noncanonical_requested_at_utc_fails_closed(
    tmp_path: Path, timestamp: Any
) -> None:
    state, run_id, manifest_sha = _request(tmp_path)
    marker_path = state / "STOP_REQUEST.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker.pop("marker_sha256")
    marker["requested_at_utc"] = timestamp
    marker["marker_sha256"] = _canonical_sha256(marker)
    marker_path.write_text(json.dumps(marker), encoding="utf-8")

    status = read_stop_status(
        state,
        profile_sha256="a" * 64,
        packet_sha256="b" * 64,
        target="20m",
        run_id=run_id,
        run_manifest_sha256=manifest_sha,
    )
    assert status["status"] == "INVALID"
    assert "safe_stop_requested_at_utc_invalid" in status["errors"]


def test_request_rejects_bad_timestamp_before_state_side_effect(tmp_path: Path) -> None:
    state = tmp_path / "state"
    manifest = _canonical_manifest()
    run_id, manifest_sha = _identity(manifest)
    with pytest.raises(OperatorPreflightError, match="safe_stop_requested_at_utc_invalid"):
        request_safe_stop(
            state,
            profile_sha256="a" * 64,
            packet_sha256="b" * 64,
            target="20m",
            run_id=run_id,
            run_manifest_sha256=manifest_sha,
            requested_at_utc="2026-09-14T20:00:00+00:00",
        )
    assert not state.exists()


def test_canonical_timestamp_with_fractional_seconds_remains_valid(tmp_path: Path) -> None:
    state = tmp_path / "state"
    manifest = _canonical_manifest()
    run_id, manifest_sha = _identity(manifest)
    result = request_safe_stop(
        state,
        profile_sha256="a" * 64,
        packet_sha256="b" * 64,
        target="20m",
        run_id=run_id,
        run_manifest_sha256=manifest_sha,
        requested_at_utc="2026-09-14T20:00:00.123456Z",
    )
    assert result["status"] == "REQUESTED"
    status = read_stop_status(
        state,
        profile_sha256="a" * 64,
        packet_sha256="b" * 64,
        target="20m",
        run_id=run_id,
        run_manifest_sha256=manifest_sha,
    )
    assert status["status"] == "REQUESTED"
