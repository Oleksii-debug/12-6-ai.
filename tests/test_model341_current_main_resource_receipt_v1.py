from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tools.model341_current_main_resource_receipt_v1 import (
    EXPECTED_CAPTURE,
    EXPECTED_PROBE_REPORT_SHA256,
    canonical_json_sha256,
    validate_probe_tool_blob,
    validate_receipt,
    validate_receipt_file,
)

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "reports/model341_current_main_resource_envelope_v1.json"


def _load() -> dict:
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


def test_checked_in_resource_receipt_is_exact_and_valid() -> None:
    receipt = validate_receipt_file(REPORT_PATH, root=ROOT)
    assert receipt["capture"] == EXPECTED_CAPTURE
    assert receipt["probe_report_sha256"] == EXPECTED_PROBE_REPORT_SHA256
    assert canonical_json_sha256(receipt["probe_report"]) == EXPECTED_PROBE_REPORT_SHA256


def test_captured_probe_tool_blob_is_still_exact() -> None:
    validate_probe_tool_blob(ROOT)


def test_coherent_probe_reseal_cannot_replace_captured_ci_measurement() -> None:
    receipt = _load()
    receipt["probe_report"]["measurement"]["synthetic_loss_median"] += 0.125
    receipt["probe_report_sha256"] = canonical_json_sha256(receipt["probe_report"])

    with pytest.raises(ValueError, match="captured probe report identity mismatch"):
        validate_receipt(receipt)


def test_ci_binding_substitution_fails_closed() -> None:
    receipt = _load()
    receipt["capture"] = copy.deepcopy(EXPECTED_CAPTURE)
    receipt["capture"]["probe_head_sha"] = "0" * 40

    with pytest.raises(ValueError, match="CI capture binding mismatch"):
        validate_receipt(receipt)


def test_unknown_receipt_field_fails_closed() -> None:
    receipt = _load()
    receipt["unexpected"] = "not-authority"

    with pytest.raises(ValueError, match="top-level key set mismatch"):
        validate_receipt(receipt)


def test_report_payload_tamper_with_stale_hash_fails_closed() -> None:
    receipt = _load()
    receipt["probe_report"]["truth_boundary"]["authorized_optimized_target_exposure"] = 1

    with pytest.raises(ValueError, match="captured probe report content mismatch"):
        validate_receipt(receipt)
