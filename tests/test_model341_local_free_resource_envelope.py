from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tools.model341_local_free_resource_envelope import (
    CURRENT_CARRIER_HEAD,
    EXPECTED_BLOBS,
    canonical_json_sha256,
    run_probe,
    validate_report,
    validate_repo_execution_blobs,
)

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "reports/model341_local_free_resource_envelope_v4.json"


def _load() -> dict:
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


def _reseal(report: dict) -> dict:
    report = copy.deepcopy(report)
    report.pop("report_identity_sha256", None)
    report["report_identity_sha256"] = canonical_json_sha256(report)
    return report


def test_checked_in_resource_envelope_validates_exact_current_blobs() -> None:
    report = _load()
    validate_report(report)
    validate_repo_execution_blobs(ROOT)
    assert report["measurement_origin"]["current_carrier_head"] == CURRENT_CARRIER_HEAD
    assert report["measurement_origin"]["execution_bearing_blobs"] == EXPECTED_BLOBS


def test_bool_int_alias_cannot_fake_zero_optimizer_updates() -> None:
    report = _load()
    report["measurement"]["optimizer_updates"] = False
    with pytest.raises(ValueError, match="exact JSON integer"):
        validate_report(_reseal(report))


def test_truth_boundary_cannot_promote_real_target_exposure() -> None:
    report = _load()
    report["truth_boundary"]["authorized_optimized_target_exposure"] = 1
    with pytest.raises(ValueError, match="truth boundary widened"):
        validate_report(_reseal(report))


def test_planning_arithmetic_tamper_fails_closed() -> None:
    report = _load()
    report["planning"]["mechanics_only_lower_bound_seconds"] = 32_812.0
    with pytest.raises(ValueError, match="planning seconds arithmetic mismatch"):
        validate_report(_reseal(report))


def test_execution_bearing_blob_substitution_fails_closed() -> None:
    report = _load()
    report["measurement_origin"]["execution_bearing_blobs"]["src/twelve_six/model.py"] = "0" * 40
    with pytest.raises(ValueError, match="execution-bearing blob drift"):
        validate_report(_reseal(report))


def test_unknown_field_fails_closed() -> None:
    report = _load()
    report["measurement"]["training_ready"] = True
    with pytest.raises(ValueError, match="keys mismatch"):
        validate_report(_reseal(report))


def test_fresh_exact_model_probe_executes_without_parameter_or_optimizer_update() -> None:
    probe = run_probe(
        ROOT,
        warmup_samples=0,
        measured_samples=1,
        intraop_threads=1,
        interop_threads=1,
    )
    assert probe["parameter_fingerprint_unchanged"] is True
    assert probe["optimizer_object_created"] is False
    assert probe["optimizer_updates"] == 0
    assert probe["model_updates"] == 0
    assert probe["truth_boundary"]["authorized_optimized_target_exposure"] == 0
    assert probe["truth_boundary"]["training_executed"] is False
    assert probe["truth_boundary"]["learned_weights_created"] is False
