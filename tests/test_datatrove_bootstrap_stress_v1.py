from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

VALIDATOR = Path(__file__).parents[1] / "tools" / "validate_datatrove_bootstrap_stress_v1.py"


def base_manifest() -> dict:
    return {
        "schema_version": 1,
        "component": {"id": "DATATROVE", "package": "datatrove", "version": "0.10.0"},
        "upstream": {
            "repository": "https://github.com/huggingface/datatrove",
            "tag": "v0.10.0",
            "commit": "7024aecca2f9ffb7b7cf0d02c0c823b8b24cf664",
            "wheel_sha256": "c7bb75deed2c3e88fb5138f8ea075a170ee98d6c94fc263829609091ea9c2b5d",
            "sdist_sha256": "e31f89bdccb30ef0796854f5842ff52b4b224c28b2d5b110088e84071ea05c40",
        },
        "rights": {
            "software_license": "Apache-2.0",
            "source_notice_present": False,
            "dataset_rights_inherited": False,
            "model_weight_rights_applicable": False,
            "training_authority_from_code_license": False,
        },
        "environment": {
            "python": "3.13.5", "os": "Linux", "arch": "x86_64", "cpu_count": 5,
            "gpu_detected": False, "package_managers": {"uv": {}}, "network_to_package_index": "FAIL",
        },
        "bootstrap": {
            "isolated_env_created": True, "exact_install_attempted": True, "installed": False,
            "install_command": ["uv", "pip", "install", "datatrove==0.10.0"], "result_code": 2,
        },
        "runtime": {"status": "NOT_EXECUTED", "execution_mode": "real"},
        "parity": {"status": "NOT_EXECUTED"},
        "benchmark": {"runs_required": 2, "status": "NOT_EXECUTED", "paid_compute": False},
        "canonical_base_safety": {
            "canonical_base_modified": False, "foreign_pretrained_weights_used": False,
            "foreign_instruction_or_alignment_behavior_used": False, "tokenizer_modified": False,
            "training_executed": False, "benchmark_final_test_data_accessed": False,
            "model_checkpoint_modified": False,
        },
        "decision": "RETEST_RUNTIME_REQUIRED",
    }


def run_validator(tmp_path: Path, payload: dict) -> subprocess.CompletedProcess[str]:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    return subprocess.run([sys.executable, str(VALIDATOR), str(manifest)], text=True, capture_output=True, check=False)


def test_valid_blocked_manifest_is_accepted(tmp_path: Path) -> None:
    result = run_validator(tmp_path, base_manifest())
    assert result.returncode == 0, result.stdout + result.stderr


def test_wrong_upstream_commit_fails_closed(tmp_path: Path) -> None:
    payload = base_manifest()
    payload["upstream"]["commit"] = "0" * 40
    result = run_validator(tmp_path, payload)
    assert result.returncode != 0


def test_installed_without_exact_version_fails_closed(tmp_path: Path) -> None:
    payload = base_manifest()
    payload["bootstrap"]["installed"] = True
    payload["bootstrap"]["installed_version"] = "0.9.0"
    payload["bootstrap"]["dependency_lock_status"] = "LOCKED"
    result = run_validator(tmp_path, payload)
    assert result.returncode != 0


def test_installed_without_lock_fails_closed(tmp_path: Path) -> None:
    payload = base_manifest()
    payload["bootstrap"]["installed"] = True
    payload["bootstrap"]["installed_version"] = "0.10.0"
    result = run_validator(tmp_path, payload)
    assert result.returncode != 0


def test_mock_runtime_fails_closed(tmp_path: Path) -> None:
    payload = base_manifest()
    payload["runtime"] = {"status": "PASS", "execution_mode": "mock"}
    result = run_validator(tmp_path, payload)
    assert result.returncode != 0


def test_canonical_base_contamination_fails_closed(tmp_path: Path) -> None:
    payload = copy.deepcopy(base_manifest())
    payload["canonical_base_safety"]["foreign_pretrained_weights_used"] = True
    result = run_validator(tmp_path, payload)
    assert result.returncode != 0


def test_not_executed_cannot_be_adopted(tmp_path: Path) -> None:
    payload = base_manifest()
    payload["decision"] = "ADOPTABLE_COMPONENT"
    result = run_validator(tmp_path, payload)
    assert result.returncode != 0
