from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

import tools.model341_current_main_resource_envelope_v2 as probe
from tools.model341_current_main_resource_envelope_v2 import (
    EXPECTED_PARAMETER_COUNT,
    git_blob_sha1,
    TRUTH_BOUNDARY,
    run_probe,
    validate_probe,
    validate_source_root,
)

ROOT = Path(__file__).resolve().parents[1]
LEGACY_V1_TOOL_BLOB_SHA1 = "6a668c6885c8578b23cfc50d94c6add149b3d978"


def test_historical_v1_probe_tool_remains_byte_frozen() -> None:
    legacy = ROOT / "tools/model341_current_main_resource_envelope_v1.py"
    assert git_blob_sha1(legacy) == LEGACY_V1_TOOL_BLOB_SHA1


def test_current_main_resource_probe_source_root_is_exact() -> None:
    validate_source_root(ROOT)


def test_current_main_resource_probe_rejects_authority_widening() -> None:
    report = run_probe(ROOT, warmup_samples=0, measured_samples=1, intraop_threads=1)
    report["truth_boundary"] = copy.deepcopy(TRUTH_BOUNDARY)
    report["truth_boundary"]["authorized_optimized_target_exposure"] = 1
    with pytest.raises(
        ValueError, match="truth boundary authorized_optimized_target_exposure mismatch"
    ):
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


def test_windows_memory_probe_fails_closed_without_windows_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    assert type(report["runtime"]["torch"]) is str
    assert report["truth_boundary"] == TRUTH_BOUNDARY

    compact = json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False)
    assert compact


def test_current_main_resource_probe_rejects_elapsed_sample_reseal() -> None:
    report = run_probe(ROOT, warmup_samples=0, measured_samples=3, intraop_threads=1)
    report["measurement"]["elapsed_seconds"] = [
        value * 2.0 for value in report["measurement"]["elapsed_seconds"]
    ]
    with pytest.raises(ValueError, match="median seconds do not match elapsed samples"):
        validate_probe(report)


def test_current_main_resource_probe_rejects_elapsed_sample_count_mismatch() -> None:
    report = run_probe(ROOT, warmup_samples=0, measured_samples=2, intraop_threads=1)
    report["measurement"]["elapsed_seconds"].pop()
    with pytest.raises(ValueError, match="elapsed sample count mismatch"):
        validate_probe(report)


def test_current_main_resource_probe_rejects_fingerprint_field_tamper() -> None:
    report = run_probe(ROOT, warmup_samples=0, measured_samples=1, intraop_threads=1)
    report["measurement"]["parameter_fingerprint_before_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="parameter fingerprint changed"):
        validate_probe(report)


def test_current_main_resource_probe_rejects_non_hex_fingerprint() -> None:
    report = run_probe(ROOT, warmup_samples=0, measured_samples=1, intraop_threads=1)
    report["measurement"]["parameter_fingerprint_before_sha256"] = "z" * 64
    report["measurement"]["parameter_fingerprint_after_sha256"] = "z" * 64
    with pytest.raises(ValueError, match="must be a SHA-256 hex digest"):
        validate_probe(report)


def test_current_main_resource_probe_rejects_planning_reseal() -> None:
    report = run_probe(ROOT, warmup_samples=0, measured_samples=1, intraop_threads=1)
    report["planning"]["mechanics_only_lower_bound_seconds_example"] *= 0.5
    with pytest.raises(ValueError, match="planning lower-bound seconds mismatch"):
        validate_probe(report)


def test_current_main_resource_probe_rejects_bool_measurement_size_alias() -> None:
    report = run_probe(ROOT, warmup_samples=0, measured_samples=1, intraop_threads=1)
    report["measurement"]["parameter_bytes"] = True
    with pytest.raises(ValueError, match="parameter_bytes must be an exact positive integer"):
        validate_probe(report)


def test_current_main_resource_probe_rejects_model_report_reseal() -> None:
    report = run_probe(ROOT, warmup_samples=0, measured_samples=1, intraop_threads=1)
    report["model"]["precision"] = "fp16"
    with pytest.raises(ValueError, match="model report precision mismatch"):
        validate_probe(report)


def test_current_main_resource_probe_rejects_planning_exclusion_widening() -> None:
    report = run_probe(ROOT, warmup_samples=0, measured_samples=1, intraop_threads=1)
    report["planning"]["excluded"].remove("optimizer.step")
    with pytest.raises(ValueError, match="planning exclusions mismatch"):
        validate_probe(report)


def test_current_main_resource_probe_rejects_parameter_byte_reseal() -> None:
    report = run_probe(ROOT, warmup_samples=0, measured_samples=1, intraop_threads=1)
    report["measurement"]["parameter_bytes"] += 4
    with pytest.raises(ValueError, match="parameter_bytes mismatch for exact fp32 model"):
        validate_probe(report)


def test_current_main_resource_probe_rejects_gradient_byte_reseal() -> None:
    report = run_probe(ROOT, warmup_samples=0, measured_samples=1, intraop_threads=1)
    report["measurement"]["gradient_bytes"] -= 4
    with pytest.raises(ValueError, match="gradient_bytes mismatch for exact fp32 model"):
        validate_probe(report)


def test_current_main_resource_probe_rejects_synthetic_loss_reseal() -> None:
    report = run_probe(ROOT, warmup_samples=0, measured_samples=3, intraop_threads=1)
    report["measurement"]["synthetic_loss_median"] *= 0.5
    with pytest.raises(ValueError, match="synthetic loss median does not match samples"):
        validate_probe(report)


def test_current_main_resource_probe_rejects_synthetic_loss_sample_count_mismatch() -> None:
    report = run_probe(ROOT, warmup_samples=0, measured_samples=2, intraop_threads=1)
    report["measurement"]["synthetic_loss_samples"].pop()
    with pytest.raises(ValueError, match="synthetic loss sample count mismatch"):
        validate_probe(report)


def test_current_main_resource_probe_rejects_unknown_top_level_claim() -> None:
    report = run_probe(ROOT, warmup_samples=0, measured_samples=1, intraop_threads=1)
    report["training_executed"] = True
    with pytest.raises(ValueError, match="report keys mismatch"):
        validate_probe(report)


def test_current_main_resource_probe_rejects_unknown_measurement_claim() -> None:
    report = run_probe(ROOT, warmup_samples=0, measured_samples=1, intraop_threads=1)
    report["measurement"]["full_training_hours"] = 1.0
    with pytest.raises(ValueError, match="measurement keys mismatch"):
        validate_probe(report)


def test_windows_hwm_path_does_not_fall_back_to_unix_resource(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(probe.sys, "platform", "win32")
    monkeypatch.setattr(probe, "_windows_peak_working_set_mib", lambda: None)
    value, source = probe._process_hwm_mib_approx()
    assert value is None
    assert source == "unavailable"


def test_current_main_resource_probe_rejects_hwm_platform_mismatch() -> None:
    report = run_probe(ROOT, warmup_samples=0, measured_samples=1, intraop_threads=1)
    report["runtime"]["platform"] = "win32"
    report["measurement"]["process_hwm_source"] = "ru_maxrss_kib"
    with pytest.raises(ValueError, match="ru_maxrss_kib platform mismatch"):
        validate_probe(report)


def test_unix_hwm_path_fails_closed_when_resource_probe_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenResource:
        RUSAGE_SELF = 0

        @staticmethod
        def getrusage(_who: int):
            raise OSError("probe unavailable")

    monkeypatch.setattr(probe.sys, "platform", "linux")
    monkeypatch.setattr(probe, "_resource", BrokenResource())
    assert probe._process_hwm_mib_approx() == (None, "unavailable")


def test_unknown_platform_hwm_fails_closed_even_with_resource(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class AvailableResource:
        RUSAGE_SELF = 0

        class Usage:
            ru_maxrss = 1234

        @staticmethod
        def getrusage(_who: int):
            return AvailableResource.Usage()

    monkeypatch.setattr(probe.sys, "platform", "unsupported-platform")
    monkeypatch.setattr(probe, "_resource", AvailableResource())
    assert probe._process_hwm_mib_approx() == (None, "unavailable")


def test_current_main_resource_probe_rejects_truth_boundary_int_as_bool_alias() -> None:
    report = run_probe(ROOT, warmup_samples=0, measured_samples=1, intraop_threads=1)
    report["truth_boundary"]["tokenizer_fit_authorized"] = 0
    with pytest.raises(ValueError, match="truth boundary tokenizer_fit_authorized mismatch"):
        validate_probe(report)


def test_current_main_resource_probe_rejects_truth_boundary_bool_as_int_alias() -> None:
    report = run_probe(ROOT, warmup_samples=0, measured_samples=1, intraop_threads=1)
    report["truth_boundary"]["authorized_optimized_target_exposure"] = False
    with pytest.raises(
        ValueError, match="truth boundary authorized_optimized_target_exposure mismatch"
    ):
        validate_probe(report)


def test_current_main_resource_probe_does_not_seed_cuda_rng(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_cuda_seed(_seed: int) -> None:
        raise AssertionError("CPU-only probe must not seed CUDA RNG")

    monkeypatch.setattr(torch.cuda, "manual_seed_all", forbidden_cuda_seed)
    report = run_probe(ROOT, warmup_samples=0, measured_samples=1, intraop_threads=1)
    validate_probe(report)


def test_current_main_resource_probe_rejects_runtime_model_import_path_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(probe.model_module, "__file__", str(ROOT / "wrong-model.py"))
    with pytest.raises(ValueError, match="runtime model import path mismatch"):
        validate_source_root(ROOT)


def test_current_main_resource_probe_rejects_non_fp32_default_dtype() -> None:
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        with pytest.raises(ValueError, match="probe requires torch.float32 default dtype"):
            run_probe(ROOT, warmup_samples=0, measured_samples=1, intraop_threads=1)
    finally:
        torch.set_default_dtype(previous)


def test_current_main_resource_probe_rejects_non_cpu_default_device() -> None:
    previous = torch.get_default_device()
    try:
        torch.set_default_device("meta")
        with pytest.raises(ValueError, match="probe requires CPU default device"):
            run_probe(ROOT, warmup_samples=0, measured_samples=1, intraop_threads=1)
    finally:
        torch.set_default_device(previous)


def test_current_main_resource_probe_enables_grad_inside_no_grad() -> None:
    with torch.no_grad():
        report = run_probe(ROOT, warmup_samples=0, measured_samples=1, intraop_threads=1)
    validate_probe(report)


def test_current_main_resource_probe_disables_outer_cpu_autocast() -> None:
    with torch.autocast("cpu", dtype=torch.bfloat16):
        report = run_probe(ROOT, warmup_samples=0, measured_samples=1, intraop_threads=1)
    validate_probe(report)
    assert report["measurement"]["parameter_bytes"] == EXPECTED_PARAMETER_COUNT * 4
    assert report["measurement"]["gradient_bytes"] == EXPECTED_PARAMETER_COUNT * 4


def test_git_blob_sha1_normalizes_windows_crlf(tmp_path: Path) -> None:
    lf_path = tmp_path / "lf.txt"
    crlf_path = tmp_path / "crlf.txt"
    lf_path.write_bytes(b"alpha\nbeta\n")
    crlf_path.write_bytes(b"alpha\r\nbeta\r\n")
    assert git_blob_sha1(lf_path) == git_blob_sha1(crlf_path)


def test_git_blob_sha1_rejects_bare_cr(tmp_path: Path) -> None:
    path = tmp_path / "bad.txt"
    path.write_bytes(b"alpha\rbeta\n")
    with pytest.raises(ValueError, match="unsupported bare CR"):
        git_blob_sha1(path)
