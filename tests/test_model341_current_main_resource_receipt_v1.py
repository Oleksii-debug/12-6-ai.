from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import tools.model341_current_main_resource_receipt_v1 as receipt_module
from tools.model341_current_main_resource_receipt_v1 import (
    EXPECTED_CAPTURE,
    EXPECTED_PROBE_REPORT_SHA256,
    EXPECTED_RUNTIME_PROJECT,
    MEASUREMENT_AUTHORITY_COMMENT_ID,
    MEASUREMENT_AUTHORITY_SHA256,
    canonical_json_sha256,
    measurement_authority_payload,
    runtime_project_projection,
    validate_current_checkout_compatibility,
    validate_probe_tool_blob,
    validate_receipt,
    validate_receipt_file,
)

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "reports/model341_current_main_resource_envelope_v1.json"


def _load() -> dict:
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


def _write_raw_receipt(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "receipt.json"
    path.write_text(text, encoding="utf-8")
    return path


def test_checked_in_resource_receipt_is_exact_and_valid() -> None:
    # Validate the sealed old report; current-source compatibility is separate.
    receipt = validate_receipt_file(REPORT_PATH)
    assert receipt["capture"] == EXPECTED_CAPTURE
    assert receipt["probe_report_sha256"] == EXPECTED_PROBE_REPORT_SHA256
    assert canonical_json_sha256(receipt["probe_report"]) == EXPECTED_PROBE_REPORT_SHA256


def test_captured_probe_tool_blob_is_still_exact() -> None:
    validate_probe_tool_blob(ROOT)


def test_prepublished_measurement_authority_matches_checked_in_receipt() -> None:
    receipt = _load()
    assert MEASUREMENT_AUTHORITY_COMMENT_ID == 5851221028
    assert (
        canonical_json_sha256(measurement_authority_payload(receipt))
        == MEASUREMENT_AUTHORITY_SHA256
    )


def test_coherent_probe_reseal_cannot_replace_captured_ci_measurement() -> None:
    receipt = _load()
    receipt["probe_report"]["measurement"]["synthetic_loss_median"] += 0.125
    receipt["probe_report_sha256"] = canonical_json_sha256(receipt["probe_report"])

    with pytest.raises(ValueError, match="captured probe report identity mismatch"):
        validate_receipt(receipt)


def test_prepublished_authority_rejects_coherent_incommit_reseal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = _load()
    receipt["probe_report"]["measurement"]["synthetic_loss_median"] += 0.125
    resealed = canonical_json_sha256(receipt["probe_report"])
    receipt["probe_report_sha256"] = resealed
    monkeypatch.setattr(receipt_module, "EXPECTED_PROBE_REPORT_SHA256", resealed)

    with pytest.raises(ValueError, match="prepublished measurement authority mismatch"):
        receipt_module.validate_receipt(receipt)


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


def test_raw_receipt_rejects_duplicate_top_level_key(tmp_path: Path) -> None:
    raw = REPORT_PATH.read_text(encoding="utf-8")
    ambiguous = raw.replace(
        "{\n",
        '{\n  "schema": "attacker-controlled-shadow",\n',
        1,
    )
    path = _write_raw_receipt(tmp_path, ambiguous)

    with pytest.raises(ValueError, match="duplicate JSON object member: schema"):
        validate_receipt_file(path)


def test_raw_receipt_rejects_duplicate_nested_authority_key(tmp_path: Path) -> None:
    raw = REPORT_PATH.read_text(encoding="utf-8")
    ambiguous = raw.replace(
        '"capture": {\n',
        '"capture": {\n'
        '    "probe_head_sha": "0000000000000000000000000000000000000000",\n',
        1,
    )
    path = _write_raw_receipt(tmp_path, ambiguous)

    with pytest.raises(ValueError, match="duplicate JSON object member: probe_head_sha"):
        validate_receipt_file(path)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_raw_receipt_rejects_nonfinite_json_constants(
    tmp_path: Path,
    constant: str,
) -> None:
    raw = REPORT_PATH.read_text(encoding="utf-8")
    invalid = raw.replace('"pytest_warnings": 1,', f'"pytest_warnings": {constant},', 1)
    path = _write_raw_receipt(tmp_path, invalid)

    with pytest.raises(ValueError, match="non-finite JSON constant rejected"):
        validate_receipt_file(path)


def test_current_checkout_runtime_compatibility_is_explicit() -> None:
    # Historical MODEL-341 evidence cannot authorize changed model.py bytes.
    with pytest.raises(ValueError, match="current checkout model.py identity mismatch"):
        validate_current_checkout_compatibility(ROOT)


def test_runtime_projection_ignores_packaging_only_metadata() -> None:
    pyproject = {
        "project": {
            "name": "twelve-six-ai",
            "version": "0.2.0.dev0",
            "requires-python": ">=3.11",
            "dependencies": ["numpy>=1.26", "safetensors>=0.5", "torch>=2.5"],
            "optional-dependencies": {
                "dev": ["pytest>=8", "ruff>=0.12", "setuptools>=75", "wheel"],
            },
            "scripts": {"twelve-six-windows": "twelve_six.windows_operator_cli:main"},
        },
        "tool": {"setuptools": {"data-files": {"share/twelve-six-ai": ["config.json"]}}},
    }

    assert runtime_project_projection(pyproject) == EXPECTED_RUNTIME_PROJECT


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("requires-python", ">=3.12"),
        ("dependencies", ["numpy>=1.26", "safetensors>=0.5", "torch>=2.6"]),
    ],
)
def test_runtime_projection_rejects_execution_environment_drift(
    field: str,
    replacement: object,
) -> None:
    project = copy.deepcopy(EXPECTED_RUNTIME_PROJECT)
    project[field] = replacement
    projection = runtime_project_projection({"project": project})

    assert projection != EXPECTED_RUNTIME_PROJECT


def test_current_checkout_compatibility_rejects_dependency_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_source = ROOT / "src/twelve_six/model.py"
    model_target = tmp_path / "src/twelve_six/model.py"
    model_target.parent.mkdir(parents=True)
    model_target.write_bytes(model_source.read_bytes())

    (tmp_path / "pyproject.toml").write_text(
        """
[project]
requires-python = ">=3.11"
dependencies = ["numpy>=1.26", "safetensors>=0.5", "torch>=99"]
""".strip(),
        encoding="utf-8",
    )

    # Isolate the dependency guard; model-byte rejection is tested above.
    monkeypatch.setattr(
        receipt_module, "MODEL_BLOB_SHA1", receipt_module.git_blob_sha1(model_target)
    )
    with pytest.raises(
        ValueError,
        match="current checkout runtime project dependency projection mismatch",
    ):
        validate_current_checkout_compatibility(tmp_path)


def test_raw_receipt_rejects_overflowed_json_number(tmp_path: Path) -> None:
    raw = REPORT_PATH.read_text(encoding="utf-8")
    invalid = raw.replace('"pytest_warnings": 1,', '"pytest_warnings": 1e400,', 1)
    path = _write_raw_receipt(tmp_path, invalid)

    with pytest.raises(ValueError, match="non-finite JSON number rejected"):
        validate_receipt_file(path)


def test_current_checkout_compatibility_rejects_model_byte_drift(
    tmp_path: Path,
) -> None:
    model_source = ROOT / "src/twelve_six/model.py"
    model_target = tmp_path / "src/twelve_six/model.py"
    model_target.parent.mkdir(parents=True)
    model_target.write_bytes(model_source.read_bytes() + b"\n")

    (tmp_path / "pyproject.toml").write_text(
        """
[project]
requires-python = ">=3.11"
dependencies = ["numpy>=1.26", "safetensors>=0.5", "torch>=2.5"]
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="current checkout model.py identity mismatch"):
        validate_current_checkout_compatibility(tmp_path)


def test_raw_receipt_rejects_non_object_document(tmp_path: Path) -> None:
    path = _write_raw_receipt(tmp_path, "[]")

    with pytest.raises(ValueError, match="receipt must be an object"):
        validate_receipt_file(path)
