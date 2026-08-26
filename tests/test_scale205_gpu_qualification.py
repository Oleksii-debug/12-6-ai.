from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / "tools/run_scale205_activation_checkpoint_gpu.py"
    spec = importlib.util.spec_from_file_location("scale205_gpu_qualification", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_measured_headroom_rule_uses_reserved_hbm_and_inclusive_80_percent() -> None:
    module = _module()
    result = module.decide_checkpoint_policy(
        uncheckpointed_status="PASS",
        peak_reserved_bytes=8_000,
        usable_hbm_bytes=10_000,
    )
    assert result["decision"] == "none"
    assert result["uncheckpointed_reserved_fraction_of_usable_hbm"] == 0.8

    result = module.decide_checkpoint_policy(
        uncheckpointed_status="PASS",
        peak_reserved_bytes=8_001,
        usable_hbm_bytes=10_000,
    )
    assert result["decision"] == "per_block"


def test_oom_forces_per_block_without_fabricating_peak() -> None:
    module = _module()
    result = module.decide_checkpoint_policy(
        uncheckpointed_status="OOM",
        peak_reserved_bytes=None,
        usable_hbm_bytes=24 * 1024**3,
    )
    assert result == {
        "decision": "per_block",
        "reason": "uncheckpointed_oom",
        "uncheckpointed_reserved_fraction_of_usable_hbm": None,
        "threshold": 0.8,
    }


def test_100m_search_is_bounded_largest_tokens_first() -> None:
    module = _module()
    rows = module.ordered_100m_setups(4096)
    assert rows[0] == (4, 4096)
    assert rows[-1] == (1, 256)
    scores = [batch * (sequence - 1) for batch, sequence in rows]
    assert scores == sorted(scores, reverse=True)
    assert len(rows) == 15


def test_no_cuda_is_not_run_and_never_gpu_pass(monkeypatch, tmp_path: Path) -> None:
    module = _module()
    monkeypatch.setattr(
        module,
        "_hardware",
        lambda: {
            "python": "test",
            "platform": "test",
            "machine": "x86_64",
            "torch": "test",
            "torch_cuda_build": None,
            "cuda_available": False,
            "cuda_device_count": 0,
            "nvidia_smi_present": False,
        },
    )
    output = tmp_path / "qualification.json"
    args = argparse.Namespace(
        output=output,
        authorized_free_gpu=False,
        dtype="bf16",
        stage_10m=Path("configs/stages/s3_10m.json"),
        stage_100m=Path("configs/stages/s4_100m_accelerator.candidate.json"),
    )
    assert module._parent(args) == 0
    evidence = json.loads(output.read_text(encoding="utf-8"))
    assert evidence["status"] == "NOT_RUN_NO_GPU"
    assert evidence["scales"] == {}
    assert "CPU evidence is not promoted to GPU PASS" in evidence["conclusion"]
