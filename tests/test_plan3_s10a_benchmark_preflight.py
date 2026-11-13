import json

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
