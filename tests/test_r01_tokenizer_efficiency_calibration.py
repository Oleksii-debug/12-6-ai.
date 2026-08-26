from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/r01_tokenizer_efficiency_calibration.py"

spec = importlib.util.spec_from_file_location("r01_tokenizer_efficiency", TOOL)
assert spec is not None and spec.loader is not None
calibration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(calibration)


def _payload() -> dict:
    return {
        "corpus_identity": "corpus-v1-test",
        "tokenizer_identity": "tokenizer-test",
        "records": [
            {"record_id": "ua-1", "stratum": "UA", "utf8_bytes": 240, "content_tokens": 80},
            {"record_id": "en-1", "stratum": "EN", "utf8_bytes": 200, "content_tokens": 100},
            {
                "record_id": "code-1",
                "stratum": "CODE",
                "utf8_bytes": 180,
                "content_tokens": 120,
            },
        ],
    }


def test_calibration_is_deterministic_and_stratum_specific() -> None:
    first = calibration.calibrate(_payload())
    second = calibration.calibrate(_payload())
    assert first == second
    assert first["by_stratum"]["UA"]["bytes_per_content_token"] == 3.0
    assert first["by_stratum"]["EN"]["bytes_per_content_token"] == 2.0
    assert first["by_stratum"]["CODE"]["bytes_per_content_token"] == 1.5
    assert first["result_identity_sha256"] == second["result_identity_sha256"]


def test_calibration_never_derives_training_or_compute_authority() -> None:
    result = calibration.calibrate(_payload())
    boundary = result["truth_boundary"]
    assert boundary["measurement_only"] is True
    assert boundary["training_budget_derived"] is False
    assert boundary["long_training_authorized"] is False
    assert boundary["paid_compute_authorized"] is False
    assert boundary["semantic_context_equivalence_claimed"] is False
    assert boundary["flop_equivalence_claimed"] is False


def test_missing_stratum_fails_closed() -> None:
    payload = _payload()
    payload["records"] = payload["records"][:-1]
    try:
        calibration.calibrate(payload)
    except ValueError as exc:
        assert "all UA/EN/CODE strata are required" in str(exc)
    else:
        raise AssertionError("missing CODE stratum must fail")


def test_duplicate_record_id_fails_closed() -> None:
    payload = _payload()
    payload["records"][1]["record_id"] = "ua-1"
    try:
        calibration.calibrate(payload)
    except ValueError as exc:
        assert "duplicate record_id" in str(exc)
    else:
        raise AssertionError("duplicate record_id must fail")


def test_boolean_or_nonpositive_counts_fail_closed() -> None:
    for field, bad_value in (("utf8_bytes", True), ("content_tokens", 0)):
        payload = copy.deepcopy(_payload())
        payload["records"][0][field] = bad_value
        try:
            calibration.calibrate(payload)
        except ValueError as exc:
            assert field in str(exc)
        else:
            raise AssertionError(f"invalid {field} must fail")
