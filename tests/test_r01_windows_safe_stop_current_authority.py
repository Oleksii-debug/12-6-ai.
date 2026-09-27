from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import twelve_six.windows_operator_preflight as operator
from twelve_six.learned20m_current_run_authority import (
    CurrentRunAuthorityInspection,
)
from twelve_six.windows_operator_preflight import (
    OperatorPreflightError,
    _TRUTH_BOUNDARY,
    _bind_candidate_to_current_run,
    _canonical_sha256,
    read_stop_status,
    request_safe_stop,
)


def _inspection(
    *,
    run_id: str = "run-a",
    recovery_sha: str = "2" * 64,
    identity_sha: str = "3" * 64,
    valid: bool = True,
    active: bool = True,
) -> CurrentRunAuthorityInspection:
    return CurrentRunAuthorityInspection(
        present=True,
        valid=valid,
        active=active,
        ref="refs/heads/ts6-current-training-run-v1",
        remote_tip="1" * 40,
        generation=1,
        pointer_identity_sha256="4" * 64,
        launch_manifest_sha256="5" * 64,
        global_lease_ref="refs/heads/ts6-training-run-lease-v1/" + "5" * 64,
        global_lease_remote_tip="6" * 40,
        global_lease_state_sha256="7" * 64,
        global_lease_expires_at_utc="2026-09-27T23:59:59Z",
        run_id=run_id,
        recovery_run_manifest_sha256=recovery_sha,
        current_run_identity_sha256=identity_sha,
        blockers=(),
    )


def _request(state: Path) -> tuple[str, str]:
    run_id = "run-a"
    run_manifest_sha256 = "2" * 64
    request_safe_stop(
        state,
        profile_sha256="a" * 64,
        packet_sha256="b" * 64,
        target="20m",
        run_id=run_id,
        run_manifest_sha256=run_manifest_sha256,
        requested_at_utc="2026-09-27T18:00:00Z",
    )
    return run_id, run_manifest_sha256


def test_candidate_manifest_is_input_only_and_must_match_fixed_current_run() -> None:
    current = _inspection()
    assert _bind_candidate_to_current_run(
        current,
        candidate_run_id="run-a",
        candidate_run_manifest_sha256="2" * 64,
    ) == ("run-a", "2" * 64, "3" * 64)

    with pytest.raises(
        OperatorPreflightError, match="safe_stop_candidate_run_id_not_current"
    ):
        _bind_candidate_to_current_run(
            current,
            candidate_run_id="run-b",
            candidate_run_manifest_sha256="2" * 64,
        )
    with pytest.raises(
        OperatorPreflightError, match="safe_stop_candidate_run_manifest_not_current"
    ):
        _bind_candidate_to_current_run(
            current,
            candidate_run_id="run-a",
            candidate_run_manifest_sha256="8" * 64,
        )


def test_inactive_current_run_never_authorizes_stop_binding() -> None:
    with pytest.raises(
        OperatorPreflightError, match="safe_stop_current_run_authority_not_active"
    ):
        _bind_candidate_to_current_run(
            _inspection(valid=False, active=False),
            candidate_run_id="run-a",
            candidate_run_manifest_sha256="2" * 64,
        )


@pytest.mark.parametrize(
    ("path", "replacement", "error"),
    [
        (
            ("checkpoint_or_training_success_claimed",),
            0,
            "safe_stop_checkpoint_or_training_success_claimed_mismatch",
        ),
        (
            ("truth_boundary", "training_executed"),
            0,
            "safe_stop_truth_boundary_mismatch",
        ),
        (
            ("truth_boundary", "authorized_optimized_target_exposure"),
            False,
            "safe_stop_truth_boundary_mismatch",
        ),
        (
            ("truth_boundary", "authorized_optimized_target_exposure"),
            0.0,
            "safe_stop_truth_boundary_mismatch",
        ),
    ],
)
def test_self_resealed_scalar_aliases_fail_closed(
    tmp_path: Path,
    path: tuple[str, ...],
    replacement: object,
    error: str,
) -> None:
    state = tmp_path / "state"
    run_id, run_manifest_sha256 = _request(state)
    marker_path = state / "STOP_REQUEST.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker.pop("marker_sha256")
    target = marker
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = replacement
    marker["marker_sha256"] = _canonical_sha256(marker)
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    status = read_stop_status(
        state,
        profile_sha256="a" * 64,
        packet_sha256="b" * 64,
        target="20m",
        run_id=run_id,
        run_manifest_sha256=run_manifest_sha256,
    )
    assert status["status"] == "INVALID"
    assert error in status["errors"]


def test_write_failure_leaves_no_final_marker_and_retry_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = tmp_path / "state"
    original_write = os.write

    def fail_write(fd: int, data: object) -> int:
        del fd, data
        raise OSError("synthetic write failure")

    monkeypatch.setattr(os, "write", fail_write)
    with pytest.raises(OperatorPreflightError, match="safe_stop_publish_failed"):
        _request(state)
    assert not (state / "STOP_REQUEST.json").exists()
    monkeypatch.setattr(os, "write", original_write)
    run_id, run_manifest_sha256 = _request(state)
    status = read_stop_status(
        state,
        profile_sha256="a" * 64,
        packet_sha256="b" * 64,
        target="20m",
        run_id=run_id,
        run_manifest_sha256=run_manifest_sha256,
    )
    assert status["status"] == "REQUESTED"


def test_fsync_failure_leaves_no_final_marker_and_retry_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = tmp_path / "state"
    original_fsync = os.fsync

    def fail_fsync(fd: int) -> None:
        del fd
        raise OSError("synthetic fsync failure")

    monkeypatch.setattr(os, "fsync", fail_fsync)
    with pytest.raises(OperatorPreflightError, match="safe_stop_publish_failed"):
        _request(state)
    assert not (state / "STOP_REQUEST.json").exists()
    monkeypatch.setattr(os, "fsync", original_fsync)
    _request(state)
    assert (state / "STOP_REQUEST.json").is_file()


def test_truth_boundary_fixture_still_has_exact_json_scalar_types() -> None:
    assert _TRUTH_BOUNDARY["authorized_optimized_target_exposure"].__class__ is int
    assert _TRUTH_BOUNDARY["training_executed"].__class__ is bool



def test_status_without_marker_does_not_require_remote_current_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = tmp_path / "missing-state"

    def forbidden_resolver(**kwargs: object) -> tuple[str, str, str]:
        del kwargs
        raise AssertionError("current-run authority must not be read for no marker")

    monkeypatch.setattr(operator, "_resolve_current_run_identity", forbidden_resolver)
    code = operator.main(["--state-dir", str(state), "--json", "status"])
    assert code in {operator.EXIT_OK, operator.EXIT_BLOCKED}


def test_status_with_marker_fails_closed_without_current_run_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = tmp_path / "state"
    state.mkdir()
    (state / "STOP_REQUEST.json").write_text("{}", encoding="utf-8")

    def blocked_resolver(**kwargs: object) -> tuple[str, str, str]:
        del kwargs
        raise OperatorPreflightError("synthetic_current_run_unavailable")

    monkeypatch.setattr(operator, "_resolve_current_run_identity", blocked_resolver)
    code = operator.main(["--state-dir", str(state), "--json", "status"])
    assert code == operator.EXIT_ERROR



def test_temp_path_substitution_never_publishes_success_or_unlinks_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = tmp_path / "state"
    original_link = os.link
    replacement_paths: list[Path] = []

    def substitute_then_link(
        source: os.PathLike[str] | str,
        destination: os.PathLike[str] | str,
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> None:
        assert src_dir_fd is None
        assert dst_dir_fd is None
        source_path = Path(source)
        source_path.unlink()
        source_path.write_bytes(b"attacker-replacement")
        replacement_paths.append(source_path)
        original_link(
            source_path,
            destination,
            follow_symlinks=follow_symlinks,
        )

    monkeypatch.setattr(os, "link", substitute_then_link)
    with pytest.raises(
        OperatorPreflightError, match="safe_stop_publish_identity_mismatch"
    ):
        _request(state)

    assert replacement_paths
    assert replacement_paths[0].read_bytes() == b"attacker-replacement"
    assert (state / "STOP_REQUEST.json").read_bytes() == b"attacker-replacement"
