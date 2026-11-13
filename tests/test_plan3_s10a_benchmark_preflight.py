import hashlib
import json
import os
from types import SimpleNamespace

import pytest

from tools import benchmark_plan3_native as tool
from twelve_six.checkpoint import hash_json
from twelve_six.training.resource_envelope import HostResources


def test_cpu_observation_reports_denials_without_allocating_or_writing(
    tmp_path, monkeypatch, capsys,
):
    observed = []

    def sample(scratch, device):
        observed.append((scratch, device))
        return HostResources(device, 2_000_000_000, 2_000_000_000, 4_000_000_000)

    monkeypatch.setattr(tool, "observe_host_resources", sample)
    assert tool.main(["SCALE-500M", "--scratch", str(tmp_path)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert observed == [(tmp_path.resolve(), "cpu")]
    assert result["capacity"]["parameter_count"] == 494_204_160
    assert result["capacity"]["disposition"] == "CAPACITY_DENIED"
    assert "BENCHMARK_HOST_MEMORY_BELOW_RESTORE_ENVELOPE" in result["capacity"][
        "benchmark_capacity_denials"
    ]
    assert result["physical_measurement"] == "NOT_RUN"
    assert result["model_allocated"] is result["campaign_authorized"] is False
    assert result["optimizer_steps_executed"] == result["checkpoint_files_written"] == 0
    assert result.pop("receipt_sha256") == hash_json(result)
    assert list(tmp_path.iterdir()) == []


def test_gpu_selection_keeps_recipe_identity_and_does_not_launch(tmp_path, monkeypatch):
    observed = []

    def sample(scratch, device):
        observed.append((scratch, device))
        return HostResources(device, 100_000_000_000, 100_000_000_000, 100_000_000_000)

    monkeypatch.setattr(tool, "observe_host_resources", sample)
    cpu = tool.observe_profile(tool.ROOT, tmp_path, "SCALE-500M", "cpu")
    gpu = tool.observe_profile(tool.ROOT, tmp_path, "SCALE-500M", "cuda:3")
    assert observed == [(tmp_path.resolve(), "cpu"), (tmp_path.resolve(), "cuda:3")]
    assert gpu["capacity"]["disposition"] == "CAPACITY_SCREEN_PASSED"
    assert gpu["capacity"]["benchmark_device_required_bytes"] != cpu["capacity"][
        "benchmark_device_required_bytes"
    ]
    assert gpu["fixture"] == cpu["fixture"]
    assert gpu["physical_measurement"] == "NOT_RUN"
    assert gpu["model_allocated"] is gpu["paid_compute_used"] is False
    assert list(tmp_path.iterdir()) == []


def test_uncommitted_profile_is_rejected_before_any_host_probe(
    tmp_path, monkeypatch,
):
    relative = tool.PROFILES["SCALE-500M"]
    config = tmp_path / relative
    config.parent.mkdir(parents=True)
    accepted = (tool.ROOT / relative).read_bytes().replace(b"\r\n", b"\n")
    config.write_bytes(accepted + b"\n")

    def unexpected(*_args):
        pytest.fail("changed profile reached host observation")

    def git(_root, *args):
        return accepted if args[0] == "show" else b"a" * 40

    monkeypatch.setattr(tool, "_git", git)
    monkeypatch.setattr(tool, "observe_host_resources", unexpected)
    with pytest.raises(ValueError, match="differs from the source commit"):
        tool.observe_profile(tmp_path, tmp_path, "SCALE-500M", "cpu")


def _physical_args(tmp_path):
    return {
        "source_sha": "a" * 40,
        "runtime_lock_sha": "b" * 64,
        "install_report": tmp_path / "install.json",
        "install_report_sha": "c" * 64,
        "runtime_release_version": 2,
    }


def test_missing_explicit_mode_never_dispatches_physical_measurement(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(tool, "measure_profile", lambda *_, **__: pytest.fail("measurement dispatched"))
    args = ["SCALE-500M", "--scratch", str(tmp_path), "--device", "cuda:0"]
    assert tool.main([*args, "--source-sha", "a" * 40]) == 2
    assert "physical benchmark inputs require --benchmark" in capsys.readouterr().err
    assert tool.main([*args, "--benchmark"]) == 2
    assert "benchmark mode requires" in capsys.readouterr().err


@pytest.mark.parametrize(("platform", "device"), [("win32", "cuda:0"), ("linux", "cpu"), ("linux", "cuda")])
def test_physical_gate_refuses_non_linux_or_implicit_gpu(tmp_path, monkeypatch, platform, device):
    monkeypatch.setattr(tool, "sys", SimpleNamespace(platform=platform))
    with pytest.raises(ValueError, match="Linux and explicit"):
        tool.measure_profile(tmp_path, tmp_path, "SCALE-500M", device, **_physical_args(tmp_path))


def test_failed_capacity_denies_before_checkpoint_or_worker(tmp_path, monkeypatch):
    monkeypatch.setattr(tool, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(tool, "_verify_measurement_inputs", lambda *_: {})
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)

    def dispatch(operation, *_):
        assert operation is tool._benchmark_preflight
        return {
            "source_git_sha": "a" * 40,
            "capacity": {"disposition": "CAPACITY_DENIED", "benchmark_capacity_denials": ["DISK"]},
        }

    monkeypatch.setattr(tool, "_in_fresh_process", dispatch)
    with pytest.raises(RuntimeError, match="capacity denied: DISK"):
        tool.measure_profile(tmp_path, tmp_path, "SCALE-500M", "cuda:0", **_physical_args(tmp_path))
    assert list(tmp_path.iterdir()) == []


def test_checkpoint_context_uses_filesystem_identity_across_reboot():
    before = {"boot_id": "first", "device": {"uuid": "gpu"}, "checkpoint_filesystem": {"st_dev": 1, "f_fsid": 2}}
    after = {**before, "boot_id": "second", "checkpoint_filesystem": {"st_dev": 3, "f_fsid": 2}}
    assert tool.checkpoint_context_matches(before, after)
    after["checkpoint_filesystem"]["f_fsid"] = None
    assert not tool.checkpoint_context_matches(before, after)
    after = {**before, "checkpoint_filesystem": {"st_dev": 1, "f_fsid": None}}
    assert tool.checkpoint_context_matches(before, after)


def test_physical_mode_requires_an_explicit_trusted_release(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(tool, "measure_profile", lambda *_, **__: pytest.fail("measurement dispatched"))
    args = [
        "SCALE-500M", "--scratch", str(tmp_path), "--device", "cuda:0", "--benchmark",
        "--source-sha", "a" * 40, "--runtime-lock-sha256", "b" * 64,
        "--install-report", str(tmp_path / "install.json"), "--install-report-sha256", "c" * 64,
    ]
    assert tool.main(args) == 2
    assert "trusted runtime release version" in capsys.readouterr().err


def test_physical_measurement_uses_ordered_workers_and_removes_scratch(tmp_path, monkeypatch):
    monkeypatch.setattr(tool, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    lock_pins = []

    def verify(*args):
        lock_pins.append(args)
        return {"profile_id": "linux-x86_64"}

    monkeypatch.setattr(tool, "_verify_measurement_inputs", verify)
    preflight = {
        "source_git_sha": "a" * 40,
        "capacity": {"disposition": "CAPACITY_SCREEN_PASSED", "benchmark_capacity_denials": []},
    }
    context = {"boot_id": "same", "device": {"uuid": "gpu"}, "checkpoint_filesystem": {"st_dev": 1, "f_fsid": 2}}
    measured = {"boundary": {"checkpoint_id": "e" * 64, "tokens_seen": 31}, "execution_context": context}
    restored = {"checkpoint_id": "e" * 64, "tokens_seen": 31, "committed_step": 1, "execution_context": context}
    calls = []

    def dispatch(operation, *args):
        calls.append(operation)
        if operation is tool._benchmark_preflight:
            return preflight
        checkpoint = args[-2]
        if operation is tool._measurement_worker:
            assert not checkpoint.exists()
            checkpoint.mkdir()
            return measured
        assert args[-3] == measured["boundary"]
        assert checkpoint.is_dir()
        return restored

    monkeypatch.setattr(tool, "_in_fresh_process", dispatch)
    pins = _physical_args(tmp_path)
    receipt = tool.measure_profile(tmp_path, tmp_path, "SCALE-500M", "cuda:0", **pins)
    assert calls == [tool._benchmark_preflight, tool._measurement_worker, tool._restore_worker]
    assert lock_pins == [(tmp_path.resolve(), *pins.values())]
    assert receipt["measurement"] == measured
    assert receipt["fresh_process_restore"] == restored
    assert receipt["campaign_authorized"] is False
    assert receipt.pop("receipt_sha256") == hash_json(receipt)
    assert list(tmp_path.iterdir()) == []


def test_exact_lock_and_pip_report_reject_changed_archive(tmp_path, monkeypatch):
    locked_hash = "a" * 64
    lock = f"torch==2.13.0 --hash=sha256:{locked_hash}\n".encode()
    runtime_path = "runtime.lock.txt"
    profile_path = "profile.json"
    installed_version = "2.13.0"
    monkeypatch.setattr(tool, "_git", lambda _root, action, *_: b"a" * 40 if action == "rev-parse" else b"")
    monkeypatch.setattr(tool, "_committed_file", lambda *_: lock)
    verified_versions = []
    monkeypatch.setattr(
        tool.runtime_release, "verify_runtime_release",
        lambda _, *, release_version: verified_versions.append(release_version),
    )
    monkeypatch.setattr(tool.dependency_lock, "assert_exact_python", lambda: None)
    monkeypatch.setattr(tool.dependency_lock, "current_profile_id", lambda: "linux-x86_64")

    def validate_index(**_):
        return {"profiles": {"linux-x86_64": {"path": profile_path}}}

    monkeypatch.setattr(tool.dependency_lock, "validate_lock_index", validate_index)
    monkeypatch.setattr(tool.inspect, "getfile", lambda _: str(tmp_path / "loaded.py"))
    monkeypatch.setattr(tool.dependency_lock, "validate_profile_manifest", lambda **_: {
        "locks": {"runtime": {"path": runtime_path, "package_count": 1}},
    })
    monkeypatch.setattr(tool.metadata, "version", lambda _: installed_version)
    report_path = tmp_path / "install.json"

    def report(archive_hash):
        value = {"version": "1", "install": [{
            "metadata": {"name": "torch", "version": installed_version},
            "download_info": {
                "url": "file:///wheels/torch-2.13.0.whl",
                "archive_info": {"hashes": {"sha256": archive_hash}},
            },
        }]}
        raw = json.dumps(value).encode()
        report_path.write_bytes(raw)
        return hashlib.sha256(raw).hexdigest()

    lock_hash = hashlib.sha256(lock).hexdigest()
    verified = tool._verify_measurement_inputs(
        tmp_path, "a" * 40, lock_hash, report_path, report(locked_hash), 2,
    )
    assert verified["verified_distribution_versions"] == 1
    assert verified["runtime_release_version"] == 2
    assert verified["runtime_release_sha256"] == tool.runtime_release.RELEASE_V2_SHA256
    with pytest.raises(ValueError, match="wheel differs"):
        tool._verify_measurement_inputs(tmp_path, "a" * 40, lock_hash, report_path, report("b" * 64), 2)
    assert verified_versions == [2, 2]


def test_spawn_waits_for_distinct_fresh_workers():
    first = tool._in_fresh_process(os.getpid)
    second = tool._in_fresh_process(os.getpid)
    assert len({first, second, os.getpid()}) == 3
