from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from twelve_six.windows_operator_preflight import (
    EXIT_BLOCKED,
    SAFE_STOP_SCHEMA,
    MachineFacts,
    OperatorPreflightError,
    _canonical_sha256,
    _load_bound_inputs,
    _load_run_identity,
    _read_json_file,
    _validate_existing_marker,
    assess_machine,
    read_stop_status,
    request_safe_stop,
    validate_operator_profile,
)

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "configs/research/r01_windows_local_free_operator_v1.json"
PACKET = ROOT / "configs/research/r01_portable_local_free_run_packet_v1.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_manifest(run_id: Any = "run-a", **changes: Any) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "candidate": {"git_sha": "c" * 40},
        "recovery": {"topology": {"world_size": 1}},
    }
    manifest.update(changes)
    return manifest


def _run_identity(run_id: str = "run-a", **changes: Any) -> tuple[str, str]:
    manifest = _run_manifest(run_id, **changes)
    return run_id, _canonical_sha256(manifest)


def _owner_like_facts(**changes: object) -> MachineFacts:
    values = {
        "os_name": "Windows",
        # CPython 3.11 can expose Windows 11 through the legacy display release "10".
        "os_release": "10",
        "machine": "AMD64",
        "logical_cpus": 12,
        "ram_bytes": 16 * 1024**3,
        "free_disk_bytes": 100 * 1024**3,
        "python_version": "3.11.9",
        "windows_version_major": 10,
        "windows_version_minor": 0,
        "windows_version_build": 22631,
    }
    values.update(changes)
    return MachineFacts(**values)


def test_checked_in_profile_is_authority_neutral_and_valid() -> None:
    profile = _load(PROFILE)
    packet = _load(PACKET)
    assert validate_operator_profile(profile, packet) == []
    assert profile["targets"]["20m"]["launch_authorized_by_profile"] is False
    assert profile["targets"]["20m"]["training_authorized_by_profile"] is False
    assert profile["targets"]["100m"]["qualification_only"] is True
    assert profile["safe_stop"]["run_identity_required"] is True
    assert (
        profile["safe_stop"]["run_manifest_hash_semantics"]
        == "CANONICAL_SORTED_COMPACT_UTF8_JSON_SHA256"
    )
    assert profile["truth_boundary"]["authorized_optimized_target_exposure"] == 0
    assert profile["truth_boundary"]["training_executed"] is False
    assert "external_llm_or_api_used_for_data_or_intelligence" not in profile["truth_boundary"]


def test_windows_11_native_build_passes_even_when_display_release_is_10() -> None:
    result = assess_machine(
        _load(PROFILE), _load(PACKET), _owner_like_facts(), target="20m"
    )
    assert result["status"] == "PREFLIGHT_PASS"
    assert result["operator_preflight_passed"] is True
    assert result["launch_authorized"] is False
    assert result["training_authorized"] is False
    assert result["full_training_sufficiency_claimed"] is False
    assert "external_llm_or_api_used_for_data_or_intelligence" not in result["truth_boundary"]
    native = next(
        item for item in result["checks"] if item["name"] == "windows_native_version"
    )
    assert native["passed"] is True
    assert native["observed"]["build"] == 22631
    assert native["observed"]["display_release"] == "10"


@pytest.mark.parametrize(
    ("changes", "failed_check"),
    [
        (
            {
                "os_name": "Linux",
                "os_release": "6.8",
                "windows_version_major": None,
                "windows_version_minor": None,
                "windows_version_build": None,
            },
            "os",
        ),
        ({"windows_version_build": 19045}, "windows_native_version"),
        ({"windows_version_major": True}, "windows_native_version"),
        ({"windows_version_minor": True}, "windows_native_version"),
        ({"windows_version_build": True}, "windows_native_version"),
        ({"windows_version_build": None}, "windows_native_version"),
        ({"logical_cpus": 1}, "logical_cpus"),
        ({"ram_bytes": 8 * 1024**3 - 1}, "ram_bytes"),
        ({"free_disk_bytes": 5 * 1024**3 - 1}, "free_disk_bytes"),
        ({"logical_cpus": True}, "logical_cpus"),
        ({"ram_bytes": True}, "ram_bytes"),
        ({"free_disk_bytes": True}, "free_disk_bytes"),
    ],
)
def test_machine_checks_fail_closed(changes: dict, failed_check: str) -> None:
    result = assess_machine(
        _load(PROFILE), _load(PACKET), _owner_like_facts(**changes), target="20m"
    )
    assert result["status"] == "BLOCKED"
    check = next(item for item in result["checks"] if item["name"] == failed_check)
    assert check["passed"] is False
    assert result["launch_authorized"] is False
    assert result["training_authorized"] is False


def test_100m_can_never_be_promoted_beyond_qualification_only() -> None:
    result = assess_machine(
        _load(PROFILE), _load(PACKET), _owner_like_facts(), target="100m"
    )
    assert result["status"] == "QUALIFICATION_ONLY"
    assert result["qualification_only"] is True
    assert result["launch_authorized"] is False
    assert result["training_authorized"] is False


def test_profile_rejects_authority_widening_and_bool_numeric_aliases() -> None:
    packet = _load(PACKET)
    for mutate in (
        lambda p: p["targets"]["20m"].__setitem__(
            "training_authorized_by_profile", True
        ),
        lambda p: p["machine_policy"].__setitem__("minimum_ram_bytes", True),
        lambda p: p["machine_policy"].__setitem__("minimum_windows_build", True),
        lambda p: p["safe_stop"].__setitem__("run_identity_required", False),
        lambda p: p["safe_stop"].__setitem__(
            "run_manifest_hash_semantics", "RAW_FILE_SHA256"
        ),
        lambda p: p["truth_boundary"].__setitem__("training_executed", True),
    ):
        profile = copy.deepcopy(_load(PROFILE))
        mutate(profile)
        assert validate_operator_profile(profile, packet)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p["truth_boundary"].__setitem__(
            "foreign_pretrained_or_aligned_weights_used", True
        ),
        lambda p: p["truth_boundary"].__setitem__("final_test_payload_accessed", True),
        lambda p: p["runtime"].__setitem__("api_key", "secret-value"),
        lambda p: p["checkpoint"].__setitem__("atomic_publish_required", False),
    ],
)
def test_canonical_portable_packet_contract_is_enforced(mutate) -> None:
    profile = _load(PROFILE)
    packet = copy.deepcopy(_load(PACKET))
    mutate(packet)
    errors = validate_operator_profile(profile, packet)
    assert any(error.startswith("portable_packet_contract:") for error in errors)


def test_bound_input_loader_rejects_canonical_packet_contract_drift(
    tmp_path: Path,
) -> None:
    profile_path = tmp_path / "profile.json"
    packet_path = tmp_path / "packet.json"
    profile_path.write_text(PROFILE.read_text(encoding="utf-8"), encoding="utf-8")
    packet = _load(PACKET)
    packet["evaluation"]["final_test_payload_access"] = True
    packet_path.write_text(json.dumps(packet), encoding="utf-8")
    with pytest.raises(OperatorPreflightError, match="portable_packet_contract"):
        _load_bound_inputs(profile_path, packet_path)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity", "1e9999"])
def test_strict_json_rejects_non_finite_numbers(
    tmp_path: Path, constant: str
) -> None:
    path = tmp_path / "bad.json"
    path.write_text(f'{{"value": {constant}}}', encoding="utf-8")
    with pytest.raises(OperatorPreflightError, match="invalid_or_unreadable_json"):
        _read_json_file(path)


@pytest.mark.parametrize(
    ("source", "duplicate"),
    [
        ('{"execution_mode":"LOCAL_FREE","execution_mode":"PAID"}', "execution_mode"),
        (
            '{"truth_boundary":{"training_executed":false,"training_executed":true}}',
            "training_executed",
        ),
        ('{"runtime":{"api_key":null,"api_key":"secret"}}', "api_key"),
    ],
)
def test_strict_json_rejects_duplicate_keys_at_any_depth(
    tmp_path: Path, source: str, duplicate: str
) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text(source, encoding="utf-8")
    with pytest.raises(OperatorPreflightError, match=f"duplicate_object_key:{duplicate}"):
        _read_json_file(path)


def test_run_manifest_hash_matches_recovery_store_hash_json_semantics() -> None:
    manifest = {
        "run_id": "run-α",
        "z": 2,
        "a": {"β": 1, "nested": [3, 2, 1]},
    }
    canonical = json.dumps(
        manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    assert _canonical_sha256(manifest) == hashlib.sha256(canonical).hexdigest()


def test_load_run_identity_hashes_semantic_manifest_not_file_whitespace(
    tmp_path: Path,
) -> None:
    manifest = _run_manifest("run-current", note="unicode-β")
    path = tmp_path / "run-manifest.json"
    path.write_text(json.dumps(manifest, indent=4, ensure_ascii=False), encoding="utf-8")
    run_id, digest = _load_run_identity(path)
    assert run_id == "run-current"
    assert digest == _canonical_sha256(manifest)
    assert digest != hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("run_id", [None, "", "   ", True, 7])
def test_load_run_identity_rejects_missing_or_malformed_run_id(
    tmp_path: Path, run_id: Any
) -> None:
    path = tmp_path / "run-manifest.json"
    path.write_text(json.dumps(_run_manifest(run_id)), encoding="utf-8")
    with pytest.raises(OperatorPreflightError, match="run_manifest_identity_invalid"):
        _load_run_identity(path)


@pytest.mark.parametrize(
    "source",
    [
        '{"run_id":"run-a","run_id":"run-b"}',
        '{"run_id":"run-a","value":NaN}',
        '{"run_id":"run-a","value":Infinity}',
    ],
)
def test_load_run_identity_uses_strict_json_decoder(tmp_path: Path, source: str) -> None:
    path = tmp_path / "run-manifest.json"
    path.write_text(source, encoding="utf-8")
    with pytest.raises(OperatorPreflightError, match="invalid_or_unreadable_json"):
        _load_run_identity(path)


def test_safe_stop_is_exclusive_content_authenticated_run_bound_and_idempotent(
    tmp_path: Path,
) -> None:
    profile_sha = "a" * 64
    packet_sha = "b" * 64
    state = tmp_path / "state"
    run_id, run_manifest_sha = _run_identity()

    first = request_safe_stop(
        state,
        profile_sha256=profile_sha,
        packet_sha256=packet_sha,
        target="20m",
        run_id=run_id,
        run_manifest_sha256=run_manifest_sha,
        requested_at_utc="2026-09-14T04:30:00Z",
    )
    second = request_safe_stop(
        state,
        profile_sha256=profile_sha,
        packet_sha256=packet_sha,
        target="20m",
        run_id=run_id,
        run_manifest_sha256=run_manifest_sha,
        requested_at_utc="2026-09-14T04:31:00Z",
    )
    assert first["status"] == "REQUESTED"
    assert second["status"] == "ALREADY_REQUESTED"
    assert first["marker_sha256"] == second["marker_sha256"]
    assert first["run_id"] == second["run_id"] == run_id
    assert first["run_manifest_sha256"] == second["run_manifest_sha256"] == run_manifest_sha

    status = read_stop_status(
        state,
        profile_sha256=profile_sha,
        packet_sha256=packet_sha,
        target="20m",
        run_id=run_id,
        run_manifest_sha256=run_manifest_sha,
    )
    assert status["status"] == "REQUESTED"
    assert status["marker"]["schema"] == SAFE_STOP_SCHEMA
    assert status["marker"]["run_id"] == run_id
    assert status["marker"]["run_manifest_sha256"] == run_manifest_sha
    assert status["marker"]["checkpoint_or_training_success_claimed"] is False
    assert (
        "external_llm_or_api_used_for_data_or_intelligence"
        not in status["marker"]["truth_boundary"]
    )
    assert (
        _validate_existing_marker(
            status["marker"],
            profile_sha256=profile_sha,
            packet_sha256=packet_sha,
            target="20m",
            run_id=run_id,
            run_manifest_sha256=run_manifest_sha,
        )
        == []
    )


def test_existing_marker_without_current_run_identity_never_reports_requested(
    tmp_path: Path,
) -> None:
    state = tmp_path / "state"
    run_id, run_manifest_sha = _run_identity()
    request_safe_stop(
        state,
        profile_sha256="a" * 64,
        packet_sha256="b" * 64,
        target="20m",
        run_id=run_id,
        run_manifest_sha256=run_manifest_sha,
        requested_at_utc="2026-09-14T04:30:00Z",
    )
    status = read_stop_status(
        state,
        profile_sha256="a" * 64,
        packet_sha256="b" * 64,
        target="20m",
    )
    assert status["status"] == "INVALID"
    assert status["marker"] is None
    assert "safe_stop_current_run_id_invalid" in status["errors"]
    assert "safe_stop_current_run_manifest_sha256_invalid" in status["errors"]


def test_absent_marker_remains_not_requested_without_run_identity(tmp_path: Path) -> None:
    state = tmp_path / "state"
    state.mkdir()
    assert read_stop_status(
        state,
        profile_sha256="a" * 64,
        packet_sha256="b" * 64,
        target="20m",
    ) == {"status": "NOT_REQUESTED", "marker": None, "errors": []}


def test_safe_stop_rejects_stale_wrong_run_and_resealed_manifest(tmp_path: Path) -> None:
    state = tmp_path / "state"
    run_id, run_manifest_sha = _run_identity("run-a")
    request_safe_stop(
        state,
        profile_sha256="a" * 64,
        packet_sha256="b" * 64,
        target="20m",
        run_id=run_id,
        run_manifest_sha256=run_manifest_sha,
        requested_at_utc="2026-09-14T04:30:00Z",
    )

    run_b_id, run_b_sha = _run_identity("run-b")
    wrong_run = read_stop_status(
        state,
        profile_sha256="a" * 64,
        packet_sha256="b" * 64,
        target="20m",
        run_id=run_b_id,
        run_manifest_sha256=run_b_sha,
    )
    assert wrong_run["status"] == "INVALID"
    assert "safe_stop_run_id_mismatch" in wrong_run["errors"]
    assert "safe_stop_run_manifest_sha256_mismatch" in wrong_run["errors"]

    resealed_same_run_sha = _canonical_sha256(
        _run_manifest("run-a", stopping={"max_steps": 2})
    )
    resealed = read_stop_status(
        state,
        profile_sha256="a" * 64,
        packet_sha256="b" * 64,
        target="20m",
        run_id="run-a",
        run_manifest_sha256=resealed_same_run_sha,
    )
    assert resealed["status"] == "INVALID"
    assert "safe_stop_run_manifest_sha256_mismatch" in resealed["errors"]

    with pytest.raises(OperatorPreflightError, match="existing_safe_stop_marker_invalid"):
        request_safe_stop(
            state,
            profile_sha256="a" * 64,
            packet_sha256="b" * 64,
            target="20m",
            run_id=run_b_id,
            run_manifest_sha256=run_b_sha,
        )


def test_old_v1_marker_is_invalid_even_if_self_hash_is_resealed(tmp_path: Path) -> None:
    state = tmp_path / "state"
    run_id, run_manifest_sha = _run_identity()
    request_safe_stop(
        state,
        profile_sha256="a" * 64,
        packet_sha256="b" * 64,
        target="20m",
        run_id=run_id,
        run_manifest_sha256=run_manifest_sha,
        requested_at_utc="2026-09-14T04:30:00Z",
    )
    marker_path = state / "STOP_REQUEST.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker.pop("marker_sha256")
    marker["schema"] = "12-6.windows-local-free-safe-stop-request.v1"
    marker["marker_sha256"] = _canonical_sha256(marker)
    marker_path.write_text(json.dumps(marker), encoding="utf-8")

    status = read_stop_status(
        state,
        profile_sha256="a" * 64,
        packet_sha256="b" * 64,
        target="20m",
        run_id=run_id,
        run_manifest_sha256=run_manifest_sha,
    )
    assert status["status"] == "INVALID"
    assert "safe_stop_schema_mismatch" in status["errors"]


def test_safe_stop_rejects_corrupt_or_wrong_binding(tmp_path: Path) -> None:
    state = tmp_path / "state"
    run_id, run_manifest_sha = _run_identity()
    request_safe_stop(
        state,
        profile_sha256="a" * 64,
        packet_sha256="b" * 64,
        target="20m",
        run_id=run_id,
        run_manifest_sha256=run_manifest_sha,
        requested_at_utc="2026-09-14T04:30:00Z",
    )
    assert (
        read_stop_status(
            state,
            profile_sha256="c" * 64,
            packet_sha256="b" * 64,
            target="20m",
            run_id=run_id,
            run_manifest_sha256=run_manifest_sha,
        )["status"]
        == "INVALID"
    )
    marker = state / "STOP_REQUEST.json"
    marker.write_text("{broken", encoding="utf-8")
    with pytest.raises(OperatorPreflightError, match="existing_safe_stop_marker_invalid"):
        request_safe_stop(
            state,
            profile_sha256="a" * 64,
            packet_sha256="b" * 64,
            target="20m",
            run_id=run_id,
            run_manifest_sha256=run_manifest_sha,
        )


@pytest.mark.parametrize(
    ("run_id", "run_manifest_sha256"),
    [
        ("", "c" * 64),
        ("   ", "c" * 64),
        ("run-a", "C" * 64),
        ("run-a", "c" * 63),
        ("run-a", "g" * 64),
    ],
)
def test_request_safe_stop_rejects_invalid_identity_before_state_side_effect(
    tmp_path: Path, run_id: str, run_manifest_sha256: str
) -> None:
    state = tmp_path / "state"
    with pytest.raises(OperatorPreflightError, match="run_manifest_identity_invalid"):
        request_safe_stop(
            state,
            profile_sha256="a" * 64,
            packet_sha256="b" * 64,
            target="20m",
            run_id=run_id,
            run_manifest_sha256=run_manifest_sha256,
        )
    assert not state.exists()


@pytest.mark.parametrize(
    "malformed",
    [
        '{"target":"20m","target":"100m"}',
        '{"marker_sha256":"x","marker_sha256":"y"}',
        '{"requested_at_utc":NaN}',
    ],
)
def test_safe_stop_status_uses_strict_json_decoder(
    tmp_path: Path, malformed: str
) -> None:
    state = tmp_path / "state"
    state.mkdir()
    (state / "STOP_REQUEST.json").write_text(malformed, encoding="utf-8")
    result = read_stop_status(
        state,
        profile_sha256="a" * 64,
        packet_sha256="b" * 64,
        target="20m",
    )
    assert result == {
        "status": "INVALID",
        "marker": None,
        "errors": ["safe_stop_marker_invalid_json"],
    }


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlink unsupported")
def test_state_dir_symlink_is_rejected(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    try:
        link.symlink_to(real, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation unavailable")
    run_id, run_manifest_sha = _run_identity()
    with pytest.raises(OperatorPreflightError, match="state_dir_must_not_be_symlink"):
        request_safe_stop(
            link,
            profile_sha256="a" * 64,
            packet_sha256="b" * 64,
            target="20m",
            run_id=run_id,
            run_manifest_sha256=run_manifest_sha,
        )


def test_cli_request_stop_requires_exact_run_manifest_before_side_effect(
    tmp_path: Path,
) -> None:
    state = tmp_path / "state"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "twelve_six.windows_operator_preflight",
            "--state-dir",
            str(state),
            "request-stop",
            "--target",
            "20m",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )
    assert proc.returncode == 2
    assert "--run-manifest" in proc.stderr
    assert not state.exists()


def test_cli_is_line_or_json_stable_and_fails_closed_on_non_windows() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "twelve_six.windows_operator_preflight",
            "--json",
            "verify",
            "--target",
            "20m",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )
    payload = json.loads(proc.stdout)
    if sys.platform == "win32":
        assert proc.returncode in (0, EXIT_BLOCKED)
    else:
        assert proc.returncode == EXIT_BLOCKED
        assert payload["status"] == "BLOCKED"
    assert payload["launch_authorized"] is False
    assert payload["training_authorized"] is False