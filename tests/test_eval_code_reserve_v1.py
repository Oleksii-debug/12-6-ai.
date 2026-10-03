from __future__ import annotations

import copy
import importlib.util
import json
import os
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "tools/validate_eval_code_reserve_v1.py"
MATERIALIZER = ROOT / "tools/materialize_eval_code_reserve_v1.py"
MANIFEST = ROOT / "configs/evaluation/eval_code_reserve_v1.json"
EVIDENCE = ROOT / "evidence/eval647/code_selection_source_materialization_v1.json"

validator_spec = importlib.util.spec_from_file_location("eval647_validator", VALIDATOR)
assert validator_spec is not None and validator_spec.loader is not None
validator = importlib.util.module_from_spec(validator_spec)
validator_spec.loader.exec_module(validator)

materializer_spec = importlib.util.spec_from_file_location("eval647_materializer", MATERIALIZER)
assert materializer_spec is not None and materializer_spec.loader is not None
materializer = importlib.util.module_from_spec(materializer_spec)
materializer_spec.loader.exec_module(materializer)


def _manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _evidence() -> dict:
    return json.loads(EVIDENCE.read_text(encoding="utf-8"))


def test_contract_is_source_sealed_but_not_authorized() -> None:
    result = validator.validate_document(_manifest())
    assert result["reserved_objects"] == 2
    assert result["independent_families"] == 2
    assert result["selection_validation_records_authorized"] == 0
    assert result["status"] == "EXACT_RAW_OBJECTS_SEALED_PENDING_PROJECT_OVERLAP_AUDIT"


def test_committed_source_evidence_binds_sealed_contract() -> None:
    validator.validate_materialization_evidence(_manifest(), _evidence())


def test_training_promotion_fails_closed() -> None:
    mutated = copy.deepcopy(_manifest())
    mutated["objects"][0]["training_allowed"] = True
    with pytest.raises(ValueError):
        validator.validate_document(mutated)


def test_final_test_access_fails_closed() -> None:
    mutated = copy.deepcopy(_manifest())
    mutated["reservation"]["final_test_payload_access_allowed"] = True
    with pytest.raises(ValueError):
        validator.validate_document(mutated)


def test_blob_identity_mutation_fails_closed() -> None:
    mutated = copy.deepcopy(_manifest())
    mutated["objects"][1]["git_blob_sha1"] = "0" * 40
    with pytest.raises(ValueError):
        validator.validate_document(mutated)


def test_raw_sha_mutation_fails_closed() -> None:
    mutated = copy.deepcopy(_manifest())
    mutated["objects"][0]["raw_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        validator.validate_document(mutated)


def test_source_evidence_tamper_fails_closed() -> None:
    mutated = copy.deepcopy(_evidence())
    mutated["objects"][0]["raw_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        validator.validate_materialization_evidence(_manifest(), mutated)


def test_license_evidence_tamper_fails_closed() -> None:
    mutated = copy.deepcopy(_evidence())
    mutated["objects"][0]["license_raw_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        validator.validate_materialization_evidence(_manifest(), mutated)


def test_object_set_identity_tamper_fails_closed() -> None:
    mutated = copy.deepcopy(_evidence())
    mutated["object_set_identity_sha256"] = "0" * 64
    body = copy.deepcopy(mutated)
    body.pop("evidence_identity_sha256")
    mutated["evidence_identity_sha256"] = validator.hashlib.sha256(
        validator._canonical_bytes(body)
    ).hexdigest()
    original_identity = validator.EXPECTED_EVIDENCE_IDENTITY
    validator.EXPECTED_EVIDENCE_IDENTITY = mutated["evidence_identity_sha256"]
    try:
        with pytest.raises(ValueError, match="object-set identity drift"):
            validator.validate_materialization_evidence(_manifest(), mutated)
    finally:
        validator.EXPECTED_EVIDENCE_IDENTITY = original_identity


def test_wrong_source_bytes_fail_before_credit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(materializer, "_fetch", lambda *_: b"wrong")
    with pytest.raises(RuntimeError, match="raw byte-size drift"):
        materializer.materialize(_manifest())


@pytest.mark.skipif(os.environ.get("GITHUB_ACTIONS") != "true", reason="needs GitHub network")
def test_live_pinned_source_materialization_is_deterministic() -> None:
    first = materializer.materialize(_manifest())
    second = materializer.materialize(_manifest())
    assert first == second
    assert first["objects"] == _evidence()["objects"]
    assert first["object_set_identity_sha256"] == _evidence()["object_set_identity_sha256"]
    assert first["reserved_object_count"] == 2
    assert first["independent_family_count"] == 2
    assert first["raw_payload_persisted_in_repository"] is False
    assert first["selection_validation_records_authorized"] == 0
    discovered = {
        item["repository"]: {
            "raw_bytes": item["raw_bytes"],
            "raw_sha256": item["raw_sha256"],
            "git_blob_sha1": item["git_blob_sha1"],
            "license_raw_sha256": item["license_raw_sha256"],
            "license_git_blob_sha1": item["license_git_blob_sha1"],
        }
        for item in first["objects"]
    }
    warnings.warn(
        "EVAL647_LIVE_SOURCE_DISCOVERY=" + json.dumps(discovered, sort_keys=True),
        stacklevel=1,
    )



@pytest.mark.parametrize(
    "raw",
    [
        '{"value":NaN}',
        '{"value":Infinity}',
        '{"value":-Infinity}',
        '{"value":1e400}',
        '{"value":-1e400}',
    ],
)
def test_strict_authority_loader_rejects_non_finite_numbers(
    tmp_path: Path,
    raw: str,
) -> None:
    path = tmp_path / "authority.json"
    path.write_text(raw, encoding="utf-8")
    with pytest.raises(ValueError):
        validator._load_mapping(path)


def test_strict_authority_loader_rejects_nested_duplicate_members(
    tmp_path: Path,
) -> None:
    path = tmp_path / "authority.json"
    path.write_text(
        '{"outer":{"training_allowed":false,"training_allowed":true}}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate_json_key:training_allowed"):
        validator._load_mapping(path)


def test_strict_authority_loader_preserves_valid_finite_object(
    tmp_path: Path,
) -> None:
    path = tmp_path / "authority.json"
    path.write_text(
        '{"count":2,"ratio":1e-3,"nested":{"allowed":false}}',
        encoding="utf-8",
    )
    assert validator._load_mapping(path) == {
        "count": 2,
        "ratio": 1e-3,
        "nested": {"allowed": False},
    }


def test_strict_authority_loader_requires_object_root(tmp_path: Path) -> None:
    path = tmp_path / "authority.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="must contain a JSON object"):
        validator._load_mapping(path)



def test_programmatic_reservation_rejects_nested_non_finite_float() -> None:
    mutated = copy.deepcopy(_manifest())
    mutated["truth_boundary"]["diagnostic"] = float("inf")
    with pytest.raises(ValueError, match="contains non-finite float"):
        validator.validate_document(mutated)


def test_programmatic_evidence_rejects_nested_non_finite_float() -> None:
    mutated = copy.deepcopy(_evidence())
    mutated["truth_boundary"]["diagnostic"] = float("nan")
    with pytest.raises(ValueError, match="contains non-finite float"):
        validator.validate_materialization_evidence(_manifest(), mutated)


@pytest.mark.parametrize("kind", ("array", "object"))
def test_deep_external_authority_json_fails_closed(
    tmp_path: Path, kind: str,
) -> None:
    depth = 10_000
    nested = ("[" * depth + "0" + "]" * depth) if kind == "array" else (
        '{"item":' * depth + "0" + "}" * depth
    )
    path = tmp_path / f"deep-{kind}.json"
    path.write_text('{"authority":' + nested + "}", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON nesting exceeds decoder limit"):
        validator._load_mapping(path)


@pytest.mark.parametrize("kind", ("array", "object", "cycle"))
def test_programmatic_authority_nesting_fails_closed_without_python_recursion(
    kind: str,
) -> None:
    value: object = 0
    if kind == "cycle":
        loop: list[object] = []
        loop.append(loop)
        value = loop
    else:
        for _ in range(200):
            value = [value] if kind == "array" else {"item": value}
    doc = _manifest()
    doc["untrusted_extra"] = value
    with pytest.raises(ValueError, match="JSON nesting limit exceeded"):
        validator.validate_document(doc)


def test_programmatic_shallow_finite_authority_still_validates() -> None:
    assert validator.validate_document(_manifest())["reserved_objects"] == 2
    validator.validate_materialization_evidence(_manifest(), _evidence())


@pytest.mark.parametrize("literal", ("1e-9999", "-1e-9999", "2.5e-9999"))
def test_strict_authority_loader_rejects_nonzero_underflow(
    tmp_path: Path, literal: str,
) -> None:
    path = tmp_path / "underflow.json"
    path.write_text('{"value":' + literal + "}", encoding="utf-8")
    with pytest.raises(ValueError, match="nonzero_json_number_underflowed_to_zero"):
        validator._load_mapping(path)


@pytest.mark.parametrize(
    ("literal", "expected"),
    [("0e-9999", 0.0), ("-0e-9999", -0.0), ("1e-3", 0.001)],
)
def test_strict_authority_loader_preserves_lexical_zero_and_finite_numbers(
    tmp_path: Path, literal: str, expected: float,
) -> None:
    path = tmp_path / "finite.json"
    path.write_text('{"value":' + literal + "}", encoding="utf-8")
    assert validator._load_mapping(path)["value"] == expected


@pytest.mark.parametrize(
    ("section", "field", "replacement", "message"),
    [
        ("root", "issue", 647.0, "issue binding drift"),
        ("reservation", "minimum_independent_families", 2.0, "family minimum drift"),
        ("reservation", "historical_training_exposure_required", False, "historical training boundary drift"),
        ("reservation", "historical_tokenizer_fit_exposure_required", 0.0, "historical tokenizer boundary drift"),
        ("reservation", "training_overlap_required", False, "overlap boundary drift"),
        ("truth_boundary", "selection_validation_records_authorized", False, "selection records prematurely authorized"),
        ("truth_boundary", "optimizer_updates_authorized", 0.0, "optimizer updates prematurely authorized"),
    ],
)
def test_reservation_rejects_equal_value_numeric_type_aliases(
    section: str, field: str, replacement: object, message: str,
) -> None:
    doc = _manifest()
    target = doc if section == "root" else doc[section]
    target[field] = replacement
    with pytest.raises(ValueError, match=message):
        validator.validate_document(doc)


@pytest.mark.parametrize("replacement", (10438.0, "10438", False))
def test_reserved_object_raw_byte_identity_rejects_numeric_aliases(
    replacement: object,
) -> None:
    doc = _manifest()
    doc["objects"][0]["expected_raw_bytes"] = replacement
    with pytest.raises(ValueError, match="identity drift"):
        validator.validate_document(doc)


def test_reserved_object_requires_object_shape() -> None:
    doc = _manifest()
    doc["objects"][0] = 0
    with pytest.raises(ValueError, match="reserved object must be an object"):
        validator.validate_document(doc)
