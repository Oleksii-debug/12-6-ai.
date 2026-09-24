from __future__ import annotations

import copy
import json
import warnings
from pathlib import Path

import pytest

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


def test_current_main_resource_probe_preserves_weights_and_emits_measurement() -> None:
    report = run_probe(ROOT, warmup_samples=1, measured_samples=3, intraop_threads=2)
    validate_probe(report)
    assert report["model"]["parameter_count"] == EXPECTED_PARAMETER_COUNT
    assert report["measurement"]["parameter_fingerprint_unchanged"] is True
    assert report["measurement"]["optimizer_updates"] == 0
    assert report["measurement"]["model_updates"] == 0
    assert report["truth_boundary"] == TRUTH_BOUNDARY

    compact = json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False)
    warnings.warn(f"MODEL341_CURRENT_MAIN_MEASUREMENT={compact}", stacklevel=1)
