from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_TOOL = Path(__file__).parents[1] / "tools" / "run_scale205_activation_checkpoint_gpu.py"
_spec = importlib.util.spec_from_file_location("scale205_gpu", _TOOL)
assert _spec is not None and _spec.loader is not None
scale205 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(scale205)


def test_headroom_rule_uses_measured_usable_hbm_boundary() -> None:
    at_limit = scale205._headroom_decision(800, 1000)
    above = scale205._headroom_decision(801, 1000)
    assert at_limit["rule_selection"] == "none"
    assert above["rule_selection"] == "per_block"
    assert at_limit["headroom_fraction_limit"] == pytest.approx(0.80)


def test_runtime_identity_requires_exact_d08_torch_cuda_pair() -> None:
    ok, errors = scale205._runtime_identity_ok(
        {"torch": "2.13.0+cu130", "torch_cuda": "13.0"}
    )
    assert ok is True
    assert errors == []
    ok, errors = scale205._runtime_identity_ok(
        {"torch": "2.13.0+cu130", "torch_cuda": "12.8"}
    )
    assert ok is False
    assert errors


def test_oom_classifier_is_fail_closed() -> None:
    assert scale205._oom(RuntimeError("CUDA out of memory")) is True
    assert scale205._oom(RuntimeError("unrelated numerical failure")) is False
