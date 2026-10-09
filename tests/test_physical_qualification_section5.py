from __future__ import annotations

import base64
import hashlib
import io
import json
import subprocess
from pathlib import Path

import pytest

from twelve_six import physical_qualification
from twelve_six.physical_qualification import (
    ActionExecution,
    ExecutionMode,
    ExternalResourceEvidence,
    HostInventory,
    QualificationAction,
    QualificationPacket,
    QualificationVerdict,
    ResourceKind,
    execute_qualification,
    load_verified_signed_packet,
    run_bounded_pytest,
    verify_qualification_evidence,
    write_evidence_bundle,
)
from twelve_six.sil_qualification import GitState


_GIT_SHA = "a" * 40
_AGENT_BYTES = b"section-5-agent-source"
_AGENT_SHA = hashlib.sha256(_AGENT_BYTES).hexdigest()
_KEY_ID = "physical-root-1"
_HOST_KEY_ID = "physical-host-1"


def _fake_signature(message: bytes) -> bytes:
    left = hashlib.sha256(_KEY_ID.encode("ascii") + message).digest()
    right = hashlib.sha256(message + _KEY_ID.encode("ascii")).digest()
    return left + right


def _fake_verify(key_id: str, message: bytes, signature: bytes) -> bool:
    return key_id == _KEY_ID and signature == _fake_signature(message)


def _host_signature(message: bytes) -> bytes:
    left = hashlib.sha256(_HOST_KEY_ID.encode("ascii") + message).digest()
    right = hashlib.sha256(message + _HOST_KEY_ID.encode("ascii")).digest()
    return left + right


def _fake_evidence_signer(key_id: str, message: bytes) -> bytes:
    assert key_id == _HOST_KEY_ID
    return _host_signature(message)


def _fake_evidence_verify(key_id: str, message: bytes, signature: bytes) -> bool:
    return key_id == _HOST_KEY_ID and signature == _host_signature(message)


def _action(
    *,
    action_id: str = "cpu-regression",
    resources: tuple[ResourceKind, ...] = (ResourceKind.CPU,),
    max_output_bytes: int = 1024,
) -> QualificationAction:
    return QualificationAction(
        action_id=action_id,
        pytest_targets=("tests/test_physical_qualification_section5.py",),
        timeout_seconds=60,
        max_output_bytes=max_output_bytes,
        required_resources=resources,
    )


def _packet(
    *,
    mode: ExecutionMode = ExecutionMode.REAL_HOST,
    actions: tuple[QualificationAction, ...] | None = None,
    artifacts: tuple[str, ...] = (),
) -> QualificationPacket:
    return QualificationPacket(
        schema_version="12-6.physical-qualification-packet.v1",
        packet_id="section5-test",
        target_git_sha=_GIT_SHA,
        agent_source_sha256=_AGENT_SHA,
        execution_mode=mode,
        allowed_os_families=("LINUX", "WINDOWS"),
        not_before_epoch_seconds=100,
        expires_epoch_seconds=200,
        actions=(_action(),) if actions is None else actions,
        artifact_paths=artifacts,
    )


def _write_signed_packet(path: Path, packet: QualificationPacket, *, corrupt: bool = False) -> None:
    message = json.dumps(
        packet.to_dict(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    signature = _fake_signature(message)
    if corrupt:
        signature = b"x" * 64
    payload = {
        "schema_version": "12-6.signed-physical-qualification-packet.v1",
        "packet": packet.to_dict(),
        "signature": {
            "algorithm": "ED25519",
            "key_id": _KEY_ID,
            "signature_b64": base64.b64encode(signature).decode("ascii"),
        },
    }
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _load(tmp_path: Path, packet: QualificationPacket):
    path = tmp_path / "packet.json"
    _write_signed_packet(path, packet)
    return load_verified_signed_packet(
        path,
        signature_verifier=_fake_verify,
        now_epoch_seconds=150,
    )


def _inventory(*, cuda: bool = False) -> HostInventory:
    return HostInventory(
        os_family="LINUX",
        platform_system="Linux",
        platform_release="test",
        machine="x86_64",
        python_version="3.13.0",
        python_executable="/opt/12-6/python-test",
        cpu_logical_count=8,
        ram_total_bytes=16 * 1024**3,
        disk_total_bytes=100 * 1024**3,
        disk_free_bytes=80 * 1024**3,
        torch_version="2.8.0",
        cuda_available=cuda,
        cuda_device_names=("Test GPU",) if cuda else (),
    )


def _pass_runner(action: QualificationAction, root: Path) -> ActionExecution:
    del action, root
    return ActionExecution(return_code=0, stdout=b"ok\n", stderr=b"", duration_ms=7)


def _git_probe(_: Path) -> GitState:
    return GitState(sha=_GIT_SHA, tracked_clean=True)


def _network_probe(_: Path) -> ExternalResourceEvidence:
    return ExternalResourceEvidence(
        resource=ResourceKind.NETWORK,
        adapter_id="network-loopback-v1",
        evidence=b"network-proof-v1",
    )


def _network_probe_verify(adapter_id: str, evidence: bytes) -> bool:
    return adapter_id == "network-loopback-v1" and evidence == b"network-proof-v1"


def test_signed_packet_rejects_bad_signature_and_expiry(tmp_path: Path) -> None:
    path = tmp_path / "packet.json"
    packet = _packet()
    _write_signed_packet(path, packet, corrupt=True)
    with pytest.raises(ValueError, match="signature verification failed"):
        load_verified_signed_packet(
            path,
            signature_verifier=_fake_verify,
            now_epoch_seconds=150,
        )

    _write_signed_packet(path, packet)
    with pytest.raises(ValueError, match="validity window"):
        load_verified_signed_packet(
            path,
            signature_verifier=_fake_verify,
            now_epoch_seconds=201,
        )


def test_action_rejects_shell_or_path_escape() -> None:
    with pytest.raises(ValueError, match="under tests"):
        QualificationAction(
            action_id="bad",
            pytest_targets=("src/twelve_six/foo.py",),
            timeout_seconds=10,
            max_output_bytes=100,
            required_resources=(ResourceKind.CPU,),
        )
    with pytest.raises(ValueError, match="inside the repository"):
        QualificationAction(
            action_id="bad",
            pytest_targets=("tests/../secrets.py",),
            timeout_seconds=10,
            max_output_bytes=100,
            required_resources=(ResourceKind.CPU,),
        )


def test_external_resource_evidence_is_bounded_and_scope_typed() -> None:
    with pytest.raises(ValueError, match="only valid for NETWORK/MODEL/PROVIDER"):
        ExternalResourceEvidence(
            resource=ResourceKind.CPU,
            adapter_id="cpu-not-external",
            evidence=b"x",
        )
    with pytest.raises(ValueError, match="size is invalid or unbounded"):
        ExternalResourceEvidence(
            resource=ResourceKind.NETWORK,
            adapter_id="network-empty",
            evidence=b"",
        )
    with pytest.raises(ValueError, match="size is invalid or unbounded"):
        ExternalResourceEvidence(
            resource=ResourceKind.NETWORK,
            adapter_id="network-oversized",
            evidence=b"x" * (physical_qualification._MAX_RESOURCE_PROBE_BYTES + 1),
        )


def test_simulation_can_never_claim_physical_pass(tmp_path: Path) -> None:
    verified = _load(tmp_path, _packet(mode=ExecutionMode.SIMULATION))
    evidence, _ = execute_qualification(
        verified,
        repo_root=tmp_path,
        host_inventory=_inventory(),
        action_runner=_pass_runner,
        git_probe=_git_probe,
        agent_source_bytes=_AGENT_BYTES,
        evidence_signing_key_id=_HOST_KEY_ID,
        evidence_signer=_fake_evidence_signer,
    )
    assert evidence["verdict"] == QualificationVerdict.SIMULATION_PASS.value
    assert set(evidence["resource_observations"].values()) == {"SIMULATED"}


def test_real_host_pass_requires_exact_proven_resources(tmp_path: Path) -> None:
    action = _action(
        resources=(ResourceKind.CPU, ResourceKind.DISK, ResourceKind.RAM),
    )
    verified = _load(tmp_path, _packet(actions=(action,)))
    evidence, _ = execute_qualification(
        verified,
        repo_root=tmp_path,
        host_inventory=_inventory(),
        action_runner=_pass_runner,
        git_probe=_git_probe,
        agent_source_bytes=_AGENT_BYTES,
        evidence_signing_key_id=_HOST_KEY_ID,
        evidence_signer=_fake_evidence_signer,
    )
    assert evidence["verdict"] == QualificationVerdict.PASS.value
    assert evidence["resource_observations"]["CPU"] == "REAL_PROBED"
    assert evidence["resource_observations"]["RAM"] == "REAL_PROBED"
    assert evidence["resource_observations"]["DISK"] == "REAL_PROBED"
    assert evidence["actions"][0]["argv"][0] == "/opt/12-6/python-test"


def test_unproven_network_or_absent_gpu_blocks_real_pass(tmp_path: Path) -> None:
    actions = (
        _action(
            action_id="gpu",
            resources=(ResourceKind.GPU,),
        ),
        _action(
            action_id="network",
            resources=(ResourceKind.NETWORK,),
        ),
    )
    verified = _load(tmp_path, _packet(actions=actions))
    evidence, _ = execute_qualification(
        verified,
        repo_root=tmp_path,
        host_inventory=_inventory(cuda=False),
        action_runner=_pass_runner,
        git_probe=_git_probe,
        agent_source_bytes=_AGENT_BYTES,
        evidence_signing_key_id=_HOST_KEY_ID,
        evidence_signer=_fake_evidence_signer,
    )
    assert evidence["verdict"] == QualificationVerdict.FAIL.value
    assert "required real resource is not proven: GPU" in evidence["reasons"]
    assert "required real resource is not proven: NETWORK" in evidence["reasons"]


def test_verified_external_network_probe_can_satisfy_real_resource(
    tmp_path: Path,
) -> None:
    action = _action(resources=(ResourceKind.NETWORK,))
    verified = _load(tmp_path, _packet(actions=(action,)))
    evidence, log_bytes = execute_qualification(
        verified,
        repo_root=tmp_path,
        host_inventory=_inventory(),
        action_runner=_pass_runner,
        git_probe=_git_probe,
        agent_source_bytes=_AGENT_BYTES,
        evidence_signing_key_id=_HOST_KEY_ID,
        evidence_signer=_fake_evidence_signer,
        resource_probes={ResourceKind.NETWORK: _network_probe},
        resource_probe_verifiers={ResourceKind.NETWORK: _network_probe_verify},
    )
    assert evidence["verdict"] == QualificationVerdict.PASS.value
    assert evidence["resource_observations"]["NETWORK"] == "REAL_PROBED"
    assert evidence["external_resource_evidence"][0]["resource"] == "NETWORK"

    evidence_path = tmp_path / "network-evidence.json"
    log_path = tmp_path / "network-run.jsonl"
    write_evidence_bundle(
        evidence_path,
        log_path,
        evidence=evidence,
        log_bytes=log_bytes,
    )
    checked = verify_qualification_evidence(
        evidence_path,
        log_path,
        verified_packet=verified,
        agent_source_bytes=_AGENT_BYTES,
        artifact_root=tmp_path,
        require_real_pass=True,
        evidence_signature_verifier=_fake_evidence_verify,
        resource_probe_verifiers={ResourceKind.NETWORK: _network_probe_verify},
    )
    assert checked["verdict"] == "PASS"

    with pytest.raises(ValueError, match="verifier is missing: NETWORK"):
        verify_qualification_evidence(
            evidence_path,
            log_path,
            verified_packet=verified,
            agent_source_bytes=_AGENT_BYTES,
            artifact_root=tmp_path,
            require_real_pass=True,
            evidence_signature_verifier=_fake_evidence_verify,
        )


def test_external_resource_probe_requires_independent_verifier(tmp_path: Path) -> None:
    action = _action(resources=(ResourceKind.NETWORK,))
    verified = _load(tmp_path, _packet(actions=(action,)))

    with pytest.raises(ValueError, match="verifier is missing: NETWORK"):
        execute_qualification(
            verified,
            repo_root=tmp_path,
            host_inventory=_inventory(),
            action_runner=_pass_runner,
            git_probe=_git_probe,
            agent_source_bytes=_AGENT_BYTES,
            evidence_signing_key_id=_HOST_KEY_ID,
            evidence_signer=_fake_evidence_signer,
            resource_probes={ResourceKind.NETWORK: _network_probe},
        )

    with pytest.raises(ValueError, match="verification failed: NETWORK"):
        execute_qualification(
            verified,
            repo_root=tmp_path,
            host_inventory=_inventory(),
            action_runner=_pass_runner,
            git_probe=_git_probe,
            agent_source_bytes=_AGENT_BYTES,
            evidence_signing_key_id=_HOST_KEY_ID,
            evidence_signer=_fake_evidence_signer,
            resource_probes={ResourceKind.NETWORK: _network_probe},
            resource_probe_verifiers={ResourceKind.NETWORK: lambda adapter, raw: False},
        )


def test_agent_source_and_exact_clean_checkout_are_fail_closed(tmp_path: Path) -> None:
    verified = _load(tmp_path, _packet())
    with pytest.raises(ValueError, match="does not authorize"):
        execute_qualification(
            verified,
            repo_root=tmp_path,
            host_inventory=_inventory(),
            action_runner=_pass_runner,
            git_probe=_git_probe,
            agent_source_bytes=b"wrong-agent",
            evidence_signing_key_id=_HOST_KEY_ID,
            evidence_signer=_fake_evidence_signer,
        )

    with pytest.raises(ValueError, match="dirty before qualification"):
        execute_qualification(
            verified,
            repo_root=tmp_path,
            host_inventory=_inventory(),
            action_runner=_pass_runner,
            git_probe=lambda _: GitState(sha=_GIT_SHA, tracked_clean=False),
            agent_source_bytes=_AGENT_BYTES,
            evidence_signing_key_id=_HOST_KEY_ID,
            evidence_signer=_fake_evidence_signer,
        )


def test_runtime_results_require_exact_schema_types(tmp_path: Path) -> None:
    verified = _load(tmp_path, _packet())
    with pytest.raises(ValueError, match="git probe must return exact GitState"):
        execute_qualification(
            verified,
            repo_root=tmp_path,
            host_inventory=_inventory(),
            action_runner=_pass_runner,
            git_probe=lambda _: object(),
            agent_source_bytes=_AGENT_BYTES,
            evidence_signing_key_id=_HOST_KEY_ID,
            evidence_signer=_fake_evidence_signer,
        )

    with pytest.raises(ValueError, match="action runner must return exact ActionExecution"):
        execute_qualification(
            verified,
            repo_root=tmp_path,
            host_inventory=_inventory(),
            action_runner=lambda action, root: object(),
            git_probe=_git_probe,
            agent_source_bytes=_AGENT_BYTES,
            evidence_signing_key_id=_HOST_KEY_ID,
            evidence_signer=_fake_evidence_signer,
        )

    network_action = _action(resources=(ResourceKind.NETWORK,))
    network_verified = _load(tmp_path, _packet(actions=(network_action,)))
    with pytest.raises(ValueError, match="exact ExternalResourceEvidence"):
        execute_qualification(
            network_verified,
            repo_root=tmp_path,
            host_inventory=_inventory(),
            action_runner=_pass_runner,
            git_probe=_git_probe,
            agent_source_bytes=_AGENT_BYTES,
            evidence_signing_key_id=_HOST_KEY_ID,
            evidence_signer=_fake_evidence_signer,
            resource_probes={ResourceKind.NETWORK: lambda _: object()},
            resource_probe_verifiers={ResourceKind.NETWORK: _network_probe_verify},
        )


def test_post_action_tree_mutation_or_output_overflow_fails(tmp_path: Path) -> None:
    verified = _load(tmp_path, _packet(actions=(_action(max_output_bytes=2),)))
    states = iter(
        (
            GitState(sha=_GIT_SHA, tracked_clean=True),
            GitState(sha=_GIT_SHA, tracked_clean=True),
            GitState(sha=_GIT_SHA, tracked_clean=False),
        )
    )
    evidence, _ = execute_qualification(
        verified,
        repo_root=tmp_path,
        host_inventory=_inventory(),
        action_runner=lambda action, root: ActionExecution(
            return_code=0,
            stdout=b"too long",
            stderr=b"",
            duration_ms=1,
        ),
        git_probe=lambda _: next(states),
        agent_source_bytes=_AGENT_BYTES,
        evidence_signing_key_id=_HOST_KEY_ID,
        evidence_signer=_fake_evidence_signer,
    )
    assert evidence["verdict"] == QualificationVerdict.FAIL.value
    assert evidence["actions"][0]["stdout_truncated"] is True
    assert evidence["actions"][0]["post_tracked_clean"] is False


def test_windows_taskkill_is_attempted_after_parent_exit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[tuple[str, ...], dict[str, object]]] = []

    class ExitedParent:
        pid = 4344

        @staticmethod
        def poll() -> int:
            return 0

        @staticmethod
        def kill() -> None:
            raise AssertionError("direct-process fallback must not be used")

    def fake_run(argv: tuple[str, ...], **kwargs: object) -> object:
        calls.append((argv, kwargs))
        return object()

    monkeypatch.setattr(physical_qualification.sys, "platform", "win32")
    monkeypatch.setattr(physical_qualification.subprocess, "run", fake_run)
    monkeypatch.setenv("SystemRoot", str(tmp_path / "Windows"))

    physical_qualification._terminate_process_tree(ExitedParent())
    assert len(calls) == 1
    argv, kwargs = calls[0]
    assert argv[1:] == ("/PID", "4344", "/T", "/F")
    assert argv[0].replace("\\", "/").endswith("/System32/taskkill.exe")
    assert kwargs["shell"] is False
    assert kwargs["check"] is True


def test_process_tree_policy_uses_isolated_posix_group(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[int, int]] = []

    class FakeProcess:
        pid = 4242

        @staticmethod
        def poll() -> None:
            return None

        @staticmethod
        def kill() -> None:
            raise AssertionError("direct-process fallback must not be used")

    monkeypatch.setattr(physical_qualification.sys, "platform", "linux")
    monkeypatch.setattr(
        physical_qualification.os,
        "killpg",
        lambda pid, sig: calls.append((pid, sig)),
    )

    assert physical_qualification._popen_process_group_kwargs() == {"start_new_session": True}
    physical_qualification._terminate_process_tree(FakeProcess())
    assert calls == [(4242, physical_qualification.signal.SIGKILL)]


def test_posix_process_group_is_killed_after_parent_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, int]] = []

    class ExitedParent:
        pid = 4243

        @staticmethod
        def poll() -> int:
            return 0

        @staticmethod
        def kill() -> None:
            raise AssertionError("direct-process fallback must not be used")

    monkeypatch.setattr(physical_qualification.sys, "platform", "linux")
    monkeypatch.setattr(
        physical_qualification.os,
        "killpg",
        lambda pid, sig: calls.append((pid, sig)),
    )

    physical_qualification._terminate_process_tree(ExitedParent())
    assert calls == [(4243, physical_qualification.signal.SIGKILL)]


def test_process_tree_policy_uses_windows_taskkill(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[tuple[str, ...], dict[str, object]]] = []

    class FakeProcess:
        pid = 4343

        @staticmethod
        def poll() -> None:
            return None

        @staticmethod
        def kill() -> None:
            raise AssertionError("direct-process fallback must not be used")

    def fake_run(argv: tuple[str, ...], **kwargs: object) -> object:
        calls.append((argv, kwargs))
        return object()

    monkeypatch.setattr(physical_qualification.sys, "platform", "win32")
    monkeypatch.setattr(
        physical_qualification.subprocess,
        "CREATE_NEW_PROCESS_GROUP",
        0x200,
        raising=False,
    )
    monkeypatch.setattr(physical_qualification.subprocess, "run", fake_run)
    monkeypatch.setenv("SystemRoot", str(tmp_path / "Windows"))

    assert physical_qualification._popen_process_group_kwargs() == {"creationflags": 0x200}
    physical_qualification._terminate_process_tree(FakeProcess())
    assert len(calls) == 1
    argv, kwargs = calls[0]
    assert argv[1:] == ("/PID", "4343", "/T", "/F")
    assert argv[0].replace("\\", "/").endswith("/System32/taskkill.exe")
    assert kwargs["shell"] is False
    assert kwargs["check"] is True


def test_bounded_pytest_rejects_host_python_pytest_env_overrides(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_must_execute.py").write_text(
        "def test_must_execute() -> None:\n    assert False\n",
        encoding="utf-8",
    )
    subprocess.run(("git", "init", "-q"), cwd=tmp_path, check=True)
    subprocess.run(("git", "add", "--", "tests/test_must_execute.py"), cwd=tmp_path, check=True)
    monkeypatch.setenv("PYTEST_ADDOPTS", "--collect-only")
    monkeypatch.setenv("PYTEST_PLUGINS", "untrusted_host_plugin")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "host-pythonpath"))
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "host-git-dir"))
    monkeypatch.setenv("GIT_WORK_TREE", str(tmp_path / "host-work-tree"))
    monkeypatch.setenv("GIT_INDEX_FILE", str(tmp_path / "host-index"))
    monkeypatch.setenv("TWELVE_SIX_SAFE_SENTINEL", "preserved")

    child_env = physical_qualification._bounded_pytest_env()
    assert "PYTEST_ADDOPTS" not in child_env
    assert "PYTEST_PLUGINS" not in child_env
    assert "PYTHONPATH" not in child_env
    assert "GIT_DIR" not in child_env
    assert "GIT_WORK_TREE" not in child_env
    assert "GIT_INDEX_FILE" not in child_env
    assert child_env["GIT_CONFIG_NOSYSTEM"] == "1"
    assert child_env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"
    assert child_env["PYTHONNOUSERSITE"] == "1"
    assert child_env["PYTHONHASHSEED"] == "0"
    assert child_env["TWELVE_SIX_SAFE_SENTINEL"] == "preserved"

    action = QualificationAction(
        action_id="host-env-bypass",
        pytest_targets=("tests/test_must_execute.py",),
        timeout_seconds=30,
        max_output_bytes=16 * 1024,
        required_resources=(),
    )
    execution = run_bounded_pytest(action, tmp_path)

    assert execution.return_code != 0


def test_run_bounded_pytest_enforces_signed_capture_limit_in_flight(
    tmp_path: Path,
) -> None:
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_noisy.py").write_text(
        "def test_noisy() -> None:\n"
        "    print('x' * 200_000)\n"
        "    assert False\n",
        encoding="utf-8",
    )
    subprocess.run(("git", "init", "-q"), cwd=tmp_path, check=True)
    subprocess.run(("git", "add", "--", "tests/test_noisy.py"), cwd=tmp_path, check=True)
    action = QualificationAction(
        action_id="bounded-noisy",
        pytest_targets=("tests/test_noisy.py",),
        timeout_seconds=30,
        max_output_bytes=1024,
        required_resources=(),
    )

    execution = run_bounded_pytest(action, tmp_path)

    assert execution.return_code != 0
    assert len(execution.stdout) <= 1025
    assert len(execution.stderr) <= 1025
    assert 1025 in {len(execution.stdout), len(execution.stderr)}


def test_tracked_target_check_rejects_ambient_git_repository_redirect(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    target = tmp_path / "target"
    spoof = tmp_path / "spoof"
    for root in (target, spoof):
        (root / "tests").mkdir(parents=True)
        (root / "tests" / "test_redirect.py").write_text(
            "def test_redirect() -> None:\n    assert True\n",
            encoding="utf-8",
        )
        subprocess.run(("git", "init", "-q"), cwd=root, check=True)

    subprocess.run(
        ("git", "add", "--", "tests/test_redirect.py"),
        cwd=spoof,
        check=True,
    )
    monkeypatch.setenv("GIT_DIR", str(spoof / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(spoof))
    monkeypatch.setenv("GIT_INDEX_FILE", str(spoof / ".git" / "index"))

    action = QualificationAction(
        action_id="git-redirect-bypass",
        pytest_targets=("tests/test_redirect.py",),
        timeout_seconds=30,
        max_output_bytes=1024,
        required_resources=(),
    )

    with pytest.raises(ValueError, match="not exactly tracked"):
        run_bounded_pytest(action, target)


def test_run_bounded_pytest_rejects_untracked_test_lookalike(tmp_path: Path) -> None:
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_untracked.py").write_text(
        "def test_untracked() -> None:\n    assert True\n",
        encoding="utf-8",
    )
    subprocess.run(("git", "init", "-q"), cwd=tmp_path, check=True)
    action = QualificationAction(
        action_id="untracked-lookalike",
        pytest_targets=("tests/test_untracked.py",),
        timeout_seconds=30,
        max_output_bytes=1024,
        required_resources=(),
    )

    with pytest.raises(ValueError, match="not exactly tracked"):
        run_bounded_pytest(action, tmp_path)


def test_artifact_hashes_and_log_are_independently_verified(tmp_path: Path) -> None:
    artifact = tmp_path / "evidence.bin"
    artifact.write_bytes(b"physical-result")
    verified = _load(tmp_path, _packet(artifacts=("evidence.bin",)))
    evidence, log_bytes = execute_qualification(
        verified,
        repo_root=tmp_path,
        host_inventory=_inventory(),
        action_runner=_pass_runner,
        git_probe=_git_probe,
        agent_source_bytes=_AGENT_BYTES,
        evidence_signing_key_id=_HOST_KEY_ID,
        evidence_signer=_fake_evidence_signer,
    )
    evidence_path = tmp_path / "evidence.json"
    log_path = tmp_path / "run.jsonl"
    write_evidence_bundle(
        evidence_path,
        log_path,
        evidence=evidence,
        log_bytes=log_bytes,
    )
    checked = verify_qualification_evidence(
        evidence_path,
        log_path,
        verified_packet=verified,
        agent_source_bytes=_AGENT_BYTES,
        artifact_root=tmp_path,
        require_real_pass=True,
        evidence_signature_verifier=_fake_evidence_verify,
    )
    assert checked["verdict"] == "PASS"

    log_path.write_bytes(log_bytes + b"x")
    with pytest.raises(ValueError, match="log hash mismatch"):
        verify_qualification_evidence(
            evidence_path,
            log_path,
            verified_packet=verified,
            agent_source_bytes=_AGENT_BYTES,
            artifact_root=tmp_path,
            require_real_pass=True,
            evidence_signature_verifier=_fake_evidence_verify,
        )


def test_artifact_substitution_and_simulation_real_gate_are_rejected(tmp_path: Path) -> None:
    artifact = tmp_path / "evidence.bin"
    artifact.write_bytes(b"physical-result")
    verified = _load(
        tmp_path,
        _packet(
            mode=ExecutionMode.SIMULATION,
            artifacts=("evidence.bin",),
        ),
    )
    evidence, log_bytes = execute_qualification(
        verified,
        repo_root=tmp_path,
        host_inventory=_inventory(),
        action_runner=_pass_runner,
        git_probe=_git_probe,
        agent_source_bytes=_AGENT_BYTES,
        evidence_signing_key_id=_HOST_KEY_ID,
        evidence_signer=_fake_evidence_signer,
    )
    evidence_path = tmp_path / "evidence.json"
    log_path = tmp_path / "run.jsonl"
    write_evidence_bundle(
        evidence_path,
        log_path,
        evidence=evidence,
        log_bytes=log_bytes,
    )
    with pytest.raises(ValueError, match="real physical PASS is required"):
        verify_qualification_evidence(
            evidence_path,
            log_path,
            verified_packet=verified,
            agent_source_bytes=_AGENT_BYTES,
            artifact_root=tmp_path,
            require_real_pass=True,
            evidence_signature_verifier=_fake_evidence_verify,
        )

    artifact.write_bytes(b"substituted")
    with pytest.raises(ValueError, match="artifact hash/size mismatch"):
        verify_qualification_evidence(
            evidence_path,
            log_path,
            verified_packet=verified,
            agent_source_bytes=_AGENT_BYTES,
            artifact_root=tmp_path,
            require_real_pass=False,
            evidence_signature_verifier=_fake_evidence_verify,
        )


def test_forged_self_consistent_evidence_without_host_signature_is_rejected(
    tmp_path: Path,
) -> None:
    verified = _load(tmp_path, _packet())
    evidence, log_bytes = execute_qualification(
        verified,
        repo_root=tmp_path,
        host_inventory=_inventory(),
        action_runner=_pass_runner,
        git_probe=_git_probe,
        agent_source_bytes=_AGENT_BYTES,
        evidence_signing_key_id=_HOST_KEY_ID,
        evidence_signer=_fake_evidence_signer,
    )
    evidence_path = tmp_path / "evidence.json"
    log_path = tmp_path / "run.jsonl"
    write_evidence_bundle(
        evidence_path,
        log_path,
        evidence=evidence,
        log_bytes=log_bytes,
    )

    payload = json.loads(evidence_path.read_text(encoding="utf-8"))
    payload["resource_observations"]["PROVIDER"] = "REAL_PROBED"
    attestation = payload.pop("evidence_attestation")
    payload.pop("evidence_identity_sha256")
    message = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    payload["evidence_identity_sha256"] = hashlib.sha256(message).hexdigest()
    payload["evidence_attestation"] = attestation
    evidence_path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="host attestation verification failed"):
        verify_qualification_evidence(
            evidence_path,
            log_path,
            verified_packet=verified,
            agent_source_bytes=_AGENT_BYTES,
            artifact_root=tmp_path,
            require_real_pass=False,
            evidence_signature_verifier=_fake_evidence_verify,
        )


def test_duplicate_json_members_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "packet.json"
    path.write_text(
        '{"schema_version":"x","schema_version":"y","packet":{},"signature":{}}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="strict unambiguous"):
        load_verified_signed_packet(
            path,
            signature_verifier=_fake_verify,
            now_epoch_seconds=150,
        )


def test_json_and_artifact_limits_are_enforced_without_unbounded_path_reads(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    oversized_json = tmp_path / "oversized.json"
    with oversized_json.open("wb") as stream:
        stream.truncate(physical_qualification._MAX_JSON_BYTES + 1)

    oversized_artifact = tmp_path / "oversized.bin"
    with oversized_artifact.open("wb") as stream:
        stream.truncate(physical_qualification._MAX_ARTIFACT_BYTES + 1)

    def forbid_read_bytes(self: Path) -> bytes:
        raise AssertionError(f"unbounded Path.read_bytes() used for {self}")

    monkeypatch.setattr(Path, "read_bytes", forbid_read_bytes)

    with pytest.raises(ValueError, match="exceeds maximum encoded size"):
        physical_qualification._strict_json_object(
            oversized_json,
            label="bounded-json",
        )
    with pytest.raises(ValueError, match="artifacts exceed total byte bound"):
        physical_qualification._collect_artifacts(
            tmp_path,
            ("oversized.bin",),
        )


def test_evidence_verifier_streams_bounded_files_without_path_read_bytes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    verified = _load(tmp_path, _packet())
    evidence, log_bytes = execute_qualification(
        verified,
        repo_root=tmp_path,
        host_inventory=_inventory(),
        action_runner=_pass_runner,
        git_probe=_git_probe,
        agent_source_bytes=_AGENT_BYTES,
        evidence_signing_key_id=_HOST_KEY_ID,
        evidence_signer=_fake_evidence_signer,
    )
    evidence_path = tmp_path / "bounded-evidence.json"
    log_path = tmp_path / "bounded-run.jsonl"
    write_evidence_bundle(
        evidence_path,
        log_path,
        evidence=evidence,
        log_bytes=log_bytes,
    )

    def forbid_read_bytes(self: Path) -> bytes:
        raise AssertionError(f"unbounded Path.read_bytes() used for {self}")

    monkeypatch.setattr(Path, "read_bytes", forbid_read_bytes)

    checked = verify_qualification_evidence(
        evidence_path,
        log_path,
        verified_packet=verified,
        agent_source_bytes=_AGENT_BYTES,
        artifact_root=tmp_path,
        require_real_pass=True,
        evidence_signature_verifier=_fake_evidence_verify,
    )
    assert checked["verdict"] == QualificationVerdict.PASS.value


def test_physical_identity_objects_revalidate_stale_state_and_enum_wires() -> None:
    packet = _packet()
    action = packet.actions[0]
    object.__setattr__(action, "timeout_seconds", 0)
    with pytest.raises(ValueError, match="action timeout exceeds"):
        packet.identity_sha256()

    inventory = _inventory()
    object.__setattr__(inventory, "disk_free_bytes", -1)
    with pytest.raises(ValueError, match="disk_free_bytes must be a non-negative integer"):
        inventory.identity_sha256()

    evidence = ExternalResourceEvidence(
        ResourceKind.NETWORK,
        "network-adapter",
        b"proof",
    )
    object.__setattr__(evidence, "evidence", b"")
    with pytest.raises(ValueError, match="evidence size is invalid"):
        evidence.to_dict()

    clean_packet = _packet()
    original_mode_value = ExecutionMode.REAL_HOST.value
    object.__setattr__(ExecutionMode.REAL_HOST, "_value_", "FORGED_REAL_HOST")
    try:
        with pytest.raises(ValueError, match="ExecutionMode wire value is non-canonical"):
            clean_packet.identity_sha256()
    finally:
        object.__setattr__(ExecutionMode.REAL_HOST, "_value_", original_mode_value)


def test_closed_signed_qualification_schemas_reject_behavioral_subclasses(tmp_path: Path) -> None:
    class ForgedStr(str):
        pass

    class ForgedBytes(bytes):
        pass

    class ForgedTuple(tuple):
        def __iter__(self):
            raise AssertionError("behavioral tuple must not be iterated")

    with pytest.raises(ValueError, match="exact lowercase SHA-256"):
        physical_qualification._require_sha256("identity", ForgedStr("a" * 64))

    with pytest.raises(ValueError, match="pytest_targets must contain"):
        QualificationAction(
            action_id="closed-action",
            pytest_targets=ForgedTuple(
                ("tests/test_physical_qualification_section5.py",)
            ),
            timeout_seconds=60,
            max_output_bytes=1024,
            required_resources=(ResourceKind.CPU,),
        )

    packet = _packet()
    with pytest.raises(ValueError, match="execution_mode must be an ExecutionMode"):
        QualificationPacket(
            schema_version=packet.schema_version,
            packet_id=packet.packet_id,
            target_git_sha=packet.target_git_sha,
            agent_source_sha256=packet.agent_source_sha256,
            execution_mode="REAL_HOST",
            allowed_os_families=packet.allowed_os_families,
            not_before_epoch_seconds=packet.not_before_epoch_seconds,
            expires_epoch_seconds=packet.expires_epoch_seconds,
            actions=packet.actions,
            artifact_paths=packet.artifact_paths,
        )

    with pytest.raises(ValueError, match="external resource evidence must be bytes"):
        ExternalResourceEvidence(
            ResourceKind.NETWORK,
            "network-adapter",
            ForgedBytes(b"probe"),
        )

    with pytest.raises(ValueError, match="action output must be bytes"):
        ActionExecution(0, ForgedBytes(b"out"), b"", 1)

    verified = _load(tmp_path, packet)
    with pytest.raises(ValueError, match="created by signature verification"):
        physical_qualification.VerifiedSignedPacket(
            verified.packet,
            verified.signing_key_id,
            verified.signature_sha256,
            verified.signed_bundle_identity_sha256,
        )

    class ForgedVerified(physical_qualification.VerifiedSignedPacket):
        pass

    forged_verified = ForgedVerified(
        verified.packet,
        verified.signing_key_id,
        verified.signature_sha256,
        verified.signed_bundle_identity_sha256,
        _verification_token=physical_qualification._VERIFIED_PACKET_TOKEN,
    )
    with pytest.raises(ValueError, match="exact VerifiedSignedPacket"):
        execute_qualification(
            forged_verified,
            repo_root=tmp_path,
            host_inventory=_inventory(),
            action_runner=_pass_runner,
            git_probe=_git_probe,
            agent_source_bytes=_AGENT_BYTES,
            evidence_signing_key_id=_HOST_KEY_ID,
            evidence_signer=_fake_evidence_signer,
        )


def test_verifier_callbacks_require_exact_boolean_decisions(tmp_path: Path) -> None:
    packet = _packet()
    packet_path = tmp_path / "truthy-packet.json"
    _write_signed_packet(packet_path, packet)

    def truthy_signature_verifier(
        key_id: str,
        message: bytes,
        signature: bytes,
    ) -> object:
        del key_id, message, signature
        return "truthy-not-bool"

    with pytest.raises(ValueError, match="signature verifier must return bool"):
        load_verified_signed_packet(
            packet_path,
            signature_verifier=truthy_signature_verifier,
            now_epoch_seconds=150,
        )

    external = ExternalResourceEvidence(
        ResourceKind.NETWORK,
        "network-adapter",
        b"network-proof",
    )
    with pytest.raises(ValueError, match="external resource verifier must return bool"):
        physical_qualification._collect_external_resource_evidence(
            required={ResourceKind.NETWORK},
            repo_root=tmp_path,
            execution_mode=ExecutionMode.REAL_HOST,
            resource_probes={
                ResourceKind.NETWORK: lambda root: external,
            },
            resource_probe_verifiers={
                ResourceKind.NETWORK: lambda adapter_id, raw: "truthy-not-bool",
            },
        )

    class ForgedDict(dict):
        pass

    with pytest.raises(ValueError, match="must be exact dictionaries"):
        physical_qualification._validate_external_resource_maps(
            ForgedDict(),
            {},
        )

    verified = _load(tmp_path, packet)
    evidence, log_bytes = execute_qualification(
        verified,
        repo_root=tmp_path,
        host_inventory=_inventory(),
        action_runner=_pass_runner,
        git_probe=_git_probe,
        agent_source_bytes=_AGENT_BYTES,
        evidence_signing_key_id=_HOST_KEY_ID,
        evidence_signer=_fake_evidence_signer,
    )
    evidence_path = tmp_path / "truthy-evidence.json"
    log_path = tmp_path / "truthy-run.jsonl"
    write_evidence_bundle(
        evidence_path,
        log_path,
        evidence=evidence,
        log_bytes=log_bytes,
    )

    with pytest.raises(ValueError, match="signature verifier must return bool"):
        verify_qualification_evidence(
            evidence_path,
            log_path,
            verified_packet=verified,
            agent_source_bytes=_AGENT_BYTES,
            artifact_root=tmp_path,
            require_real_pass=True,
            evidence_signature_verifier=truthy_signature_verifier,
        )

def test_signed_physical_authority_ignores_rebound_class_methods(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    action = _action(
        resources=(ResourceKind.CPU, ResourceKind.NETWORK),
    )
    packet = _packet(actions=(action,))
    verified = _load(tmp_path, packet)
    inventory = _inventory()
    external = _network_probe(tmp_path)

    def fail_rebound(*args, **kwargs):
        del args, kwargs
        raise AssertionError("rebound class authority must not be dispatched")

    monkeypatch.setattr(QualificationAction, "__post_init__", fail_rebound)
    monkeypatch.setattr(QualificationAction, "to_dict", fail_rebound)
    monkeypatch.setattr(QualificationAction, "logical_argv", fail_rebound)
    monkeypatch.setattr(QualificationPacket, "__post_init__", fail_rebound)
    monkeypatch.setattr(QualificationPacket, "to_dict", fail_rebound)
    monkeypatch.setattr(QualificationPacket, "identity_sha256", fail_rebound)
    monkeypatch.setattr(
        physical_qualification.VerifiedSignedPacket,
        "__post_init__",
        fail_rebound,
    )
    monkeypatch.setattr(HostInventory, "__post_init__", fail_rebound)
    monkeypatch.setattr(HostInventory, "to_dict", fail_rebound)
    monkeypatch.setattr(HostInventory, "identity_sha256", fail_rebound)
    monkeypatch.setattr(ExternalResourceEvidence, "__post_init__", fail_rebound)
    monkeypatch.setattr(ExternalResourceEvidence, "to_dict", fail_rebound)

    evidence, log_bytes = execute_qualification(
        verified,
        repo_root=tmp_path,
        host_inventory=inventory,
        action_runner=_pass_runner,
        git_probe=_git_probe,
        agent_source_bytes=_AGENT_BYTES,
        evidence_signing_key_id=_HOST_KEY_ID,
        evidence_signer=_fake_evidence_signer,
        resource_probes={ResourceKind.NETWORK: lambda root: external},
        resource_probe_verifiers={ResourceKind.NETWORK: _network_probe_verify},
    )
    evidence_path = tmp_path / "rebind-evidence.json"
    log_path = tmp_path / "rebind-log.jsonl"
    write_evidence_bundle(
        evidence_path,
        log_path,
        evidence=evidence,
        log_bytes=log_bytes,
    )
    checked = verify_qualification_evidence(
        evidence_path,
        log_path,
        verified_packet=verified,
        agent_source_bytes=_AGENT_BYTES,
        artifact_root=tmp_path,
        require_real_pass=True,
        evidence_signature_verifier=_fake_evidence_verify,
        resource_probe_verifiers={ResourceKind.NETWORK: _network_probe_verify},
    )
    assert checked["verdict"] == QualificationVerdict.PASS.value

    object.__setattr__(verified.packet.actions[0], "timeout_seconds", 0)
    with pytest.raises(ValueError, match="action timeout exceeds"):
        execute_qualification(
            verified,
            repo_root=tmp_path,
            host_inventory=inventory,
            action_runner=_pass_runner,
            git_probe=_git_probe,
            agent_source_bytes=_AGENT_BYTES,
            evidence_signing_key_id=_HOST_KEY_ID,
            evidence_signer=_fake_evidence_signer,
            resource_probes={ResourceKind.NETWORK: lambda root: external},
            resource_probe_verifiers={ResourceKind.NETWORK: _network_probe_verify},
        )



def test_operator_keyboard_interrupt_terminates_qualified_process_tree(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    action = QualificationAction(
        action_id="keyboard-interrupt",
        pytest_targets=("tests/test_physical_qualification_section5.py",),
        timeout_seconds=10,
        max_output_bytes=256,
        required_resources=(),
    )

    class FakeProcess:
        def __init__(self) -> None:
            self.stdout = io.BytesIO(b"")
            self.stderr = io.BytesIO(b"")
            self.wait_calls = 0

        def wait(self, timeout: int | None = None) -> int:
            self.wait_calls += 1
            if self.wait_calls == 1:
                raise KeyboardInterrupt
            return 130

    process = FakeProcess()
    terminated: list[FakeProcess] = []
    monkeypatch.setattr(
        physical_qualification, "_validate_checked_in_pytest_targets", lambda *_: None
    )
    monkeypatch.setattr(
        physical_qualification, "_popen_process_group_kwargs", dict
    )
    monkeypatch.setattr(
        physical_qualification.subprocess, "Popen", lambda *_, **__: process
    )
    monkeypatch.setattr(
        physical_qualification,
        "_terminate_process_tree",
        lambda item: terminated.append(item),
    )

    with pytest.raises(KeyboardInterrupt):
        run_bounded_pytest(action, tmp_path)
    assert terminated == [process]
    assert process.wait_calls == 2
