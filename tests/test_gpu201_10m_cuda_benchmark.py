from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _load_module():
    path = Path(__file__).resolve().parents[1] / "tools" / "run_gpu201_10m_cuda_benchmark.py"
    spec = importlib.util.spec_from_file_location("gpu201_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_no_gpu_report_contains_no_target_device_numbers(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module()
    monkeypatch.setattr(
        module,
        "runtime_identity",
        lambda: {
            "python": "3.11.16",
            "pytorch": "2.13.0+cu130",
            "pytorch_cuda_runtime": "13.0",
            "cuda_available": False,
            "visible_cuda_devices": 0,
            "platform": "test",
        },
    )
    report = module.no_gpu_report("a" * 40)
    assert report["status"] == "NOT_RUN_NO_GPU"
    assert report["benchmark_executed"] is False
    assert report["target_device_numbers_present"] is False
    assert report["cpu_extrapolation_present"] is False
    assert report["paid_compute"] is False
    assert report["torch_compile"]["enabled"] is False


def test_gpu199_not_run_is_rejected(tmp_path: Path) -> None:
    module = _load_module()
    path = tmp_path / "gpu199.json"
    path.write_text(
        json.dumps(
            {
                "swarm_worker_id": module.GPU199_WORKER_ID,
                "status": "NOT_RUN_NO_GPU",
                "selected_precision": "fp32",
                "runtime": {"cuda_device_name": "Example GPU"},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(module.GPU201Error, match="no executed CUDA result"):
        module.load_gpu199(path, {"cuda_device_name": "Example GPU"})


def test_gpu199_must_bind_current_device(tmp_path: Path) -> None:
    module = _load_module()
    path = tmp_path / "gpu199.json"
    path.write_text(
        json.dumps(
            {
                "swarm_worker_id": module.GPU199_WORKER_ID,
                "status": "PASS_DEVICE_BOUND",
                "decision": {"selected_precision": "bf16"},
                "runtime": {"cuda_device_name": "GPU A"},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(module.GPU201Error, match="device binding"):
        module.load_gpu199(path, {"cuda_device_name": "GPU B"})


def test_gpu199_valid_handoff_is_hash_bound(tmp_path: Path) -> None:
    module = _load_module()
    path = tmp_path / "gpu199.json"
    path.write_text(
        json.dumps(
            {
                "swarm_worker_id": module.GPU199_WORKER_ID,
                "status": "PASS_DEVICE_BOUND",
                "decision": {"selected_precision": "fp32"},
                "runtime": {"cuda_device_name": "GPU A"},
            }
        ),
        encoding="utf-8",
    )
    result = module.load_gpu199(path, {"cuda_device_name": "GPU A"})
    assert result["selected_precision"] == "fp32"
    assert len(result["evidence_sha256"]) == 64


def test_cpu_only_purpose_environment_is_rejected(tmp_path: Path) -> None:
    module = _load_module()
    path = tmp_path / "environment.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "12-6.purpose-environment-evidence.v1",
                "profile_id": "linux-x86_64-cuda-training",
                "source_sha": "b" * 40,
                "profile": {"profile_sha256": "c" * 64},
                "verification": {
                    "exact_hash_install": "PASS",
                    "project_wheel_install": "PASS",
                    "registry_validation": "PASS",
                    "runtime_probe": "PASS",
                },
                "runtime_probe": {
                    "cuda_available": False,
                    "gpu_execution": "NOT_RUN_NO_GPU",
                    "torch_cuda": "13.0",
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(module.GPU201Error, match="did not execute on a GPU"):
        module.load_environment(
            path,
            "b" * 40,
            {"pytorch_cuda_runtime": "13.0"},
        )
