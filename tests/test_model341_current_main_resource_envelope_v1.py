from __future__ import annotations

import copy
import json
import warnings
from pathlib import Path

import pytest
import torch

import tools.model341_current_main_resource_envelope_v1 as probe
from tools.model341_current_main_resource_envelope_v1 import (
    EXPECTED_PARAMETER_COUNT,
    TRUTH_BOUNDARY,
    run_probe,
    validate_probe,
    validate_source_root,
)

ROOT = Path(__file__).resolve().parents[1]


def test_current_main_resource_probe_source_root_is_exact() -> None:
    validate_source_root(ROOT)


def test_current_main_resource_probe_rejects_authority_widening() -> None:
    report = run_probe(ROOT, warmup_samples=0, measured_samples=1, intraop_threads=1)
    report["truth_boundary"] = copy.deepcopy(TRUTH_BOUNDARY)
    report["truth_boundary"]["authorized_optimized_target_exposure"] = 1
    with pytest.raises(ValueError, match="truth boundary mismatch"):
        validate_probe(report)


def test_current_main_resource_probe_rejects_bool_zero_alias() -> None:
    report = run_probe(ROOT, warmup_samples=0, measured_samples=1, intraop_threads=1)
    report["measurement"]["optimizer_updates"] = False
    with pytest.raises(ValueError, match="optimizer_updates must be exact integer zero"):
        validate_probe(report)


def test_current_main_resource_probe_truth_is_scoped_to_the_probe() -> None:
    assert "external_llm_or_api_used_for_data_or_intelligence" not in TRUTH_BOUNDARY
    assert TRUTH_BOUNDARY["probe_uses_real_corpus_or_final_test_payload"] is False
    assert TRUTH_BOUNDARY["probe_uses_deterministic_synthetic_token_ids_only"] is True
    assert TRUTH_BOUNDARY["external_model_or_api_called_by_probe"] is False


def test_current_main_resource_probe_restores_torch_process_state() -> None:
    previous_threads = torch.get_num_threads()
    previous_rng = torch.get_rng_state().clone()
    requested_threads = 1 if previous_threads != 1 else 2

    run_probe(
        ROOT,
        warmup_samples=0,
        measured_samples=1,
        intraop_threads=requested_threads,
    )

    assert torch.get_num_threads() == previous_threads
    assert torch.equal(torch.get_rng_state(), previous_rng)


def test_process_hwm_unavailable_is_honest_and_valid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(probe, "_resource", None)
    monkeypatch.setattr(probe.sys, "platform", "unsupported-platform")
    value, source = probe._process_hwm_mib_approx()
    assert value is None
    assert source == "unavailable"

    report = run_probe(ROOT, warmup_samples=0, measured_samples=1, intraop_threads=1)
    report["measurement"]["process_hwm_mib_approx"] = None
    report["measurement"]["process_hwm_source"] = "unavailable"
    validate_probe(report)


def test_windows_memory_probe_fails_closed_without_windows_api(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(probe.sys, "platform", "win32")
    if hasattr(probe.ctypes, "windll"):
        monkeypatch.delattr(probe.ctypes, "windll")
    assert probe._windows_peak_working_set_mib() is None


def test_current_main_resource_probe_preserves_weights_and_emits_measurement() -> None:
    report = run_probe(ROOT, warmup_samples=1, measured_samples=3, intraop_threads=2)
    validate_probe(report)
    assert report["model"]["parameter_count"] == EXPECTED_PARAMETER_COUNT
    assert report["measurement"]["parameter_fingerprint_unchanged"] is True
    assert report["measurement"]["optimizer_updates"] == 0
    assert report["measurement"]["model_updates"] == 0
    assert report["truth_boundary"] == TRUTH_BOUNDARY

    compact = json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False)
    warnings.warn(f"MODEL341_CURRENT_MAIN_REPAIRED_MEASUREMENT={compact}", stacklevel=1)
