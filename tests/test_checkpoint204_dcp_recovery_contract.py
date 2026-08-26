from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "tools/run_checkpoint204_dcp_recovery.py"


def _module():
    spec = importlib.util.spec_from_file_location("checkpoint204_probe", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_last_known_good_rejects_newer_invalid_generation(tmp_path: Path) -> None:
    module = _module()
    valid = tmp_path / "generation-000002"
    invalid = tmp_path / "generation-000003"
    valid.mkdir()
    invalid.mkdir()

    def verifier(path: Path):
        if path.name == invalid.name:
            raise ValueError("corrupted generation")
        return {"status": "PASS"}

    selected, rejected = module.select_last_known_good(tmp_path, verifier=verifier)
    assert selected == valid
    assert rejected == [{"generation": invalid.name, "reason": "ValueError"}]


def test_last_known_good_fails_closed_without_verified_generation(tmp_path: Path) -> None:
    module = _module()
    candidate = tmp_path / "generation-000004"
    candidate.mkdir()

    def verifier(_path: Path):
        raise RuntimeError("invalid")

    with pytest.raises(RuntimeError, match="no verified last-known-good"):
        module.select_last_known_good(tmp_path, verifier=verifier)


def test_preflight_report_is_machine_readable_and_never_fakes_cuda(tmp_path: Path) -> None:
    report_path = tmp_path / "checkpoint204-preflight.json"
    source_sha = "c" * 40
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--source-sha",
            source_sha,
            "--output",
            str(report_path),
            "--preflight-only",
        ],
        cwd=ROOT,
        check=True,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["schema_version"] == "12-6.checkpoint204-dcp-cuda-recovery.v1"
    assert report["source_sha"] == source_sha
    assert report["checkpoint_format_changed"] is False
    assert report["async_dcp_enabled"] is False
    hardware = report["hardware"]
    runnable = hardware["cuda_device_count"] >= 2 and hardware["nccl_available"]
    assert report["claims"]["cuda_validated"] is runnable
    assert report["status"] == (
        "READY_FOR_REAL_CUDA_RECOVERY" if runnable else "CUDA_UNTESTED_NO_MULTI_GPU"
    )
    assert report["claims"]["paid_compute_used"] is False
