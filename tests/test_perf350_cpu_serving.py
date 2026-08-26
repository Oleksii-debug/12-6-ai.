from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "tools"
        / "benchmark_perf350_20m_cpu_serving.py"
    )
    module_spec = importlib.util.spec_from_file_location(
        "benchmark_perf350_20m_cpu_serving", path
    )
    assert module_spec is not None
    assert module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = module
    module_spec.loader.exec_module(module)
    return module


def test_mechanics_surrogate_is_bounded_20m_gqa() -> None:
    module = _load_module()
    model_spec = module.MECHANICS_SPEC
    assert model_spec.parameter_count() == 19_935_488
    assert abs(model_spec.parameter_count() - 20_000_000) / 20_000_000 < 0.005
    assert model_spec.n_heads == 8
    assert model_spec.n_kv_heads == 2
    assert model_spec.max_seq_len == 1024
    assert model_spec.n_layers == 24


def test_thread_selection_uses_five_percent_tie_band_and_fewer_threads() -> None:
    module = _load_module()
    results = [
        {"case": {"threads": 1}, "decode": {"aggregate_tokens_per_second": 100.0}},
        {"case": {"threads": 2}, "decode": {"aggregate_tokens_per_second": 104.0}},
        {"case": {"threads": 4}, "decode": {"aggregate_tokens_per_second": 105.0}},
    ]
    assert module.select_threads(results) == 1


def test_candidate_threads_are_bounded() -> None:
    module = _load_module()
    assert module.candidate_threads({"cpu_affinity_count": 3}) == [1, 2, 3]
    assert module.candidate_threads({"cpu_affinity_count": 64}) == [1, 2, 4, 8]
