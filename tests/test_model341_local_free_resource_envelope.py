from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tools.model341_local_free_resource_envelope import (
    CURRENT_CARRIER_HEAD,
    EXPECTED_BLOBS,
    MEASUREMENT_AUTHORITY_SHA256,
    MEASUREMENT_AUTHORITY_SOURCE_COMMENT_ID,
    MEASUREMENT_AUTHORITY_SOURCE_ISSUE,
    canonical_json_sha256,
    run_probe,
    validate_repo_execution_blobs,
    validate_report,
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
    assert MEASUREMENT_AUTHORITY_SOURCE_ISSUE == 1262
    assert MEASUREMENT_AUTHORITY_SOURCE_COMMENT_ID == 5634725801
    assert MEASUREMENT_AUTHORITY_SHA256 == (
        "5090abe87bdad694274c4a183117e41a35869ab852301048a81f244e39c89965"
    )


def test_bool_int_alias_cannot_fake_zero_optimizer_updates() -> None:
    report = _load()
    report["measurement"]["optimizer_updates"] = False
    with pytest.raises(ValueError, match="measurement authority tuple mismatch"):
        validate_report(_reseal(report))


def test_truth_boundary_cannot_promote_real_target_exposure() -> None:
    report = _load()
    report["truth_boundary"]["authorized_optimized_target_exposure"] = 1
    with pytest.raises(ValueError, match="truth_boundary.authorized_optimized_target_exposure mismatch"):
        validate_report(_reseal(report))


def test_planning_arithmetic_tamper_fails_closed() -> None:
    report = _load()
    report["planning"]["mechanics_only_lower_bound_seconds"] = 32_812.0
    with pytest.raises(ValueError, match="planning seconds arithmetic mismatch"):
        validate_report(_reseal(report))


def test_median_throughput_consistency_is_unconditional_without_raw_samples() -> None:
    report = _load()
    report["measurement"]["median_causal_targets_per_second"] = 600.0
    report["planning"]["mechanics_only_lower_bound_seconds"] = 20_000_000 / 600.0
    report["planning"]["mechanics_only_lower_bound_hours"] = (
        report["planning"]["mechanics_only_lower_bound_seconds"] / 3600.0
    )
    with pytest.raises(ValueError, match="median/throughput arithmetic mismatch"):
        validate_report(_reseal(report))


def test_coherent_telemetry_and_planning_reseal_cannot_replace_prepublished_measurement() -> None:
    report = _load()
    measurement = report["measurement"]
    measurement["median_forward_loss_backward_ms"] = 400.0
    measurement["median_causal_targets_per_second"] = 127.0 / 0.4
    measurement["synthetic_loss"] = 6.25
    measurement["process_hwm_mib_approx"] = 512.0
    report["planning"]["mechanics_only_lower_bound_seconds"] = (
        20_000_000 / measurement["median_causal_targets_per_second"]
    )
    report["planning"]["mechanics_only_lower_bound_hours"] = (
        report["planning"]["mechanics_only_lower_bound_seconds"] / 3600.0
    )
    with pytest.raises(ValueError, match="measurement authority tuple mismatch"):
        validate_report(_reseal(report))


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("synthetic_loss", 6.0),
        ("process_hwm_mib_approx", 500.0),
    ],
)
def test_independently_published_measurement_fields_cannot_be_self_resealed(
    field: str, replacement: float
) -> None:
    report = _load()
    report["measurement"][field] = replacement
    with pytest.raises(ValueError, match="measurement authority tuple mismatch"):
        validate_report(_reseal(report))


def test_execution_bearing_blob_substitution_fails_closed() -> None:
    report = _load()
    report["measurement_origin"]["execution_bearing_blobs"]["src/twelve_six/model.py"] = "0" * 40
    with pytest.raises(ValueError, match="execution_bearing_blobs.src/twelve_six/model.py mismatch"):
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
