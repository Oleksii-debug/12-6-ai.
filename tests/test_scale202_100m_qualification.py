from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


REPO = Path(__file__).parents[1]
TOOL = REPO / "tools/run_scale202_100m_qualification.py"


def _module():
    spec = importlib.util.spec_from_file_location("scale202_runner", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exact_100m_data_tokenizer_contract() -> None:
    module = _module()
    contract = module.contract_snapshot(REPO)
    assert contract["parameter_count"] == 99_897_600
    assert contract["model_spec_sha256"] == (
        "6103d0d457e25206c11871f09aef1f2e23860329c060379c9f956b3851740170"
    )
    assert contract["init_spec_sha256"] == (
        "86483c6df623e80cab2f73aba718863fce18af6fe3b12430c1348414d92b48a5"
    )
    assert contract["tokenizer"]["version"] == "s0-byte-v1"
    assert contract["tokenizer"]["vocab_size"] == 256
    assert contract["corpus_identity_sha256"] == (
        "422f545da93526ceba2c0ff738e0b9deac65f35bfb31d87d2aab068cda091fc8"
    )
    assert contract["train_validation_content_overlap"] == 0


def test_execution_gate_refuses_no_gpu() -> None:
    module = _module()
    decision = module.execution_decision(
        {"cuda_available": False, "visible_cuda_devices": 0, "devices": []}
    )
    assert decision["mode"] == "none"
    assert decision["status"] == "NOT_RUN_NO_GPU"


def test_execution_gate_requires_native_bf16() -> None:
    module = _module()
    decision = module.execution_decision(
        {
            "cuda_available": True,
            "devices": [
                {
                    "index": 0,
                    "free_bytes": 16 * 1024**3,
                    "native_bf16_supported": False,
                }
            ],
        }
    )
    assert decision["status"] == "NOT_RUN_NO_NATIVE_BF16"


def test_execution_gate_selects_single_gpu_only_with_headroom() -> None:
    module = _module()
    decision = module.execution_decision(
        {
            "cuda_available": True,
            "devices": [
                {
                    "index": 0,
                    "free_bytes": module.MIN_FREE_BYTES - 1,
                    "native_bf16_supported": True,
                },
                {
                    "index": 1,
                    "free_bytes": module.MIN_FREE_BYTES + 1024,
                    "native_bf16_supported": True,
                },
            ],
        }
    )
    assert decision["mode"] == "single_gpu"
    assert decision["device_index"] == 1
    assert decision["required_free_bytes"] == module.MIN_FREE_BYTES


def test_execution_gate_marks_real_multi_gpu_fsdp2_candidate() -> None:
    module = _module()
    decision = module.execution_decision(
        {
            "cuda_available": True,
            "devices": [
                {
                    "index": 0,
                    "free_bytes": module.MIN_FREE_BYTES - 1,
                    "native_bf16_supported": True,
                },
                {
                    "index": 1,
                    "free_bytes": module.MIN_FREE_BYTES - 2,
                    "native_bf16_supported": True,
                },
            ],
        }
    )
    assert decision["mode"] == "fsdp2_candidate"
    assert decision["status"] == "NOT_RUN_SINGLE_GPU_HEADROOM_FSDP2_CANDIDATE"


def test_main_writes_truthful_no_gpu_evidence_without_model_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    monkeypatch.setattr(
        module,
        "hardware_snapshot",
        lambda: {
            "python": "test",
            "torch": "test",
            "torch_cuda_build": None,
            "cuda_available": False,
            "visible_cuda_devices": 0,
            "devices": [],
            "nvidia_smi_present": False,
            "nvidia_smi_summary": None,
            "process_peak_rss_bytes": 123,
            "cuda_visible_devices": None,
        },
    )
    output = tmp_path / "evidence.json"
    work = tmp_path / "work"
    assert module.main(
        [
            "--repo-root",
            str(REPO),
            "--output",
            str(output),
            "--work-dir",
            str(work),
        ]
    ) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "NOT_RUN_NO_GPU"
    assert payload["contract"]["parameter_count"] == 99_897_600
    assert payload["memory_gate"]["required_free_bytes"] == module.MIN_FREE_BYTES
    assert payload["paid_compute"] is False
    assert not work.exists()
