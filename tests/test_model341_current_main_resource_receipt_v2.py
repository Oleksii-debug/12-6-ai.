from __future__ import annotations

import ast
import copy
import json
from pathlib import Path

import pytest

import tools.model341_current_main_resource_receipt_v2 as receipt

ROOT = Path(__file__).resolve().parents[1]


def _minimal_candidate() -> dict:
    return {
        "schema": receipt.RECEIPT_SCHEMA,
        "status": receipt.RECEIPT_STATUS,
        "capture": {},
        "probe_report_sha256": "0" * 64,
        "probe_report": {},
    }


def test_v2_receipt_fails_closed_until_external_capture_is_published() -> None:
    receipt.validate_probe_artifacts(ROOT)
    with pytest.raises(ValueError, match="capture authority is not published"):
        receipt.validate_receipt(_minimal_candidate(), root=ROOT)


def test_v2_receipt_does_not_top_level_import_probe_before_blob_check() -> None:
    source = (ROOT / "tools/model341_current_main_resource_receipt_v2.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    assert "tools.model341_current_main_resource_envelope_v2" not in imported


def test_v2_receipt_rejects_modified_probe_bytes_before_dynamic_import(
    tmp_path: Path,
) -> None:
    probe_source = ROOT / receipt.PROBE_TOOL_RELATIVE_PATH
    test_source = ROOT / receipt.FOCUSED_TEST_RELATIVE_PATH
    probe_target = tmp_path / receipt.PROBE_TOOL_RELATIVE_PATH
    test_target = tmp_path / receipt.FOCUSED_TEST_RELATIVE_PATH
    probe_target.parent.mkdir(parents=True)
    test_target.parent.mkdir(parents=True)
    probe_target.write_bytes(probe_source.read_bytes() + b"\n")
    test_target.write_bytes(test_source.read_bytes())

    with pytest.raises(ValueError, match="v2 probe tool blob mismatch"):
        receipt.validate_probe_artifacts(tmp_path)


def test_v2_receipt_rejects_modified_focused_test_bytes(tmp_path: Path) -> None:
    probe_source = ROOT / receipt.PROBE_TOOL_RELATIVE_PATH
    test_source = ROOT / receipt.FOCUSED_TEST_RELATIVE_PATH
    probe_target = tmp_path / receipt.PROBE_TOOL_RELATIVE_PATH
    test_target = tmp_path / receipt.FOCUSED_TEST_RELATIVE_PATH
    probe_target.parent.mkdir(parents=True)
    test_target.parent.mkdir(parents=True)
    probe_target.write_bytes(probe_source.read_bytes())
    test_target.write_bytes(test_source.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="v2 focused test blob mismatch"):
        receipt.validate_probe_artifacts(tmp_path)


def test_v2_receipt_git_blob_identity_is_windows_checkout_safe(tmp_path: Path) -> None:
    lf = tmp_path / "lf.py"
    crlf = tmp_path / "crlf.py"
    lf.write_bytes(b"alpha\nbeta\n")
    crlf.write_bytes(b"alpha\r\nbeta\r\n")
    assert receipt.git_blob_sha1(lf) == receipt.git_blob_sha1(crlf)


def test_v2_receipt_git_blob_identity_rejects_bare_cr(tmp_path: Path) -> None:
    path = tmp_path / "bad.py"
    path.write_bytes(b"alpha\rbeta\n")
    with pytest.raises(ValueError, match="unsupported bare CR"):
        receipt.git_blob_sha1(path)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_v2_receipt_strict_json_rejects_nonfinite_constants(constant: str) -> None:
    with pytest.raises(ValueError, match="non-finite JSON constant rejected"):
        receipt.strict_json_loads('{"value":' + constant + "}")


def test_v2_receipt_strict_json_rejects_overflow() -> None:
    with pytest.raises(ValueError, match="non-finite JSON number rejected"):
        receipt.strict_json_loads('{"value":1e400}')


def test_v2_receipt_strict_json_rejects_duplicate_keys() -> None:
    with pytest.raises(ValueError, match="duplicate JSON object member: schema"):
        receipt.strict_json_loads('{"schema":"a","schema":"b"}')


def test_v2_authority_payload_changes_under_coherent_timing_reseal() -> None:
    candidate = _minimal_candidate()
    candidate["capture"] = {"run_id": 1}
    candidate["probe_report"] = {
        "measurement": {
            "elapsed_seconds": [1.0, 1.0, 1.0],
            "median_forward_loss_backward_seconds": 1.0,
            "median_causal_targets_per_second": 127.0,
        },
        "planning": {
            "mechanics_only_lower_bound_seconds_example": 20_000_000 / 127.0,
            "mechanics_only_lower_bound_hours_example": 20_000_000 / 127.0 / 3600.0,
        },
    }
    candidate["probe_report_sha256"] = receipt.canonical_json_sha256(
        candidate["probe_report"]
    )
    original = receipt.canonical_json_sha256(receipt.measurement_authority_payload(candidate))

    resealed = copy.deepcopy(candidate)
    resealed["probe_report"]["measurement"]["elapsed_seconds"] = [2.0, 2.0, 2.0]
    resealed["probe_report"]["measurement"]["median_forward_loss_backward_seconds"] = 2.0
    resealed["probe_report"]["measurement"]["median_causal_targets_per_second"] = 63.5
    resealed["probe_report"]["planning"]["mechanics_only_lower_bound_seconds_example"] = (
        20_000_000 / 63.5
    )
    resealed["probe_report"]["planning"]["mechanics_only_lower_bound_hours_example"] = (
        20_000_000 / 63.5 / 3600.0
    )
    resealed["probe_report_sha256"] = receipt.canonical_json_sha256(
        resealed["probe_report"]
    )
    assert (
        receipt.canonical_json_sha256(receipt.measurement_authority_payload(resealed))
        != original
    )


def test_v2_authority_payload_changes_under_coherent_fingerprint_substitution() -> None:
    candidate = _minimal_candidate()
    candidate["capture"] = {"run_id": 1}
    candidate["probe_report"] = {
        "measurement": {
            "parameter_fingerprint_before_sha256": "1" * 64,
            "parameter_fingerprint_after_sha256": "1" * 64,
            "parameter_fingerprint_unchanged": True,
        }
    }
    candidate["probe_report_sha256"] = receipt.canonical_json_sha256(
        candidate["probe_report"]
    )
    original = receipt.canonical_json_sha256(receipt.measurement_authority_payload(candidate))

    resealed = copy.deepcopy(candidate)
    resealed["probe_report"]["measurement"]["parameter_fingerprint_before_sha256"] = "0" * 64
    resealed["probe_report"]["measurement"]["parameter_fingerprint_after_sha256"] = "0" * 64
    resealed["probe_report_sha256"] = receipt.canonical_json_sha256(
        resealed["probe_report"]
    )
    assert (
        receipt.canonical_json_sha256(receipt.measurement_authority_payload(resealed))
        != original
    )


def test_v2_authority_payload_changes_under_coherent_loss_reseal() -> None:
    candidate = _minimal_candidate()
    candidate["capture"] = {"run_id": 1}
    candidate["probe_report"] = {
        "measurement": {
            "synthetic_loss_samples": [5.0, 5.0, 5.0],
            "synthetic_loss_median": 5.0,
        }
    }
    candidate["probe_report_sha256"] = receipt.canonical_json_sha256(
        candidate["probe_report"]
    )
    original = receipt.canonical_json_sha256(receipt.measurement_authority_payload(candidate))

    resealed = copy.deepcopy(candidate)
    resealed["probe_report"]["measurement"]["synthetic_loss_samples"] = [6.0, 6.0, 6.0]
    resealed["probe_report"]["measurement"]["synthetic_loss_median"] = 6.0
    resealed["probe_report_sha256"] = receipt.canonical_json_sha256(
        resealed["probe_report"]
    )
    assert (
        receipt.canonical_json_sha256(receipt.measurement_authority_payload(resealed))
        != original
    )
