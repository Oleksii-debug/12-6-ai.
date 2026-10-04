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
    document = json.loads(MANIFEST.read_text(encoding="utf-8"))
    # Re-signed negative fixtures patch the validator's pinned evidence ID.
    # Match the synthetic test document reference to that test-only identity;
    # the committed contract and production pinned identity never change.
    document["materialization_evidence"]["identity_sha256"] = (
        validator.EXPECTED_EVIDENCE_IDENTITY
    )
    return document


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


@pytest.mark.parametrize(
    ("section", "bad"),
    [
        ("root", []),
        ("predecessor", []),
        ("reservation", "not-an-object"),
        ("materialization_evidence", None),
        ("truth_boundary", 0),
    ],
)
def test_reservation_malformed_nested_shape_is_controlled(
    section: str, bad: object,
) -> None:
    document = _manifest()
    if section == "root":
        document = bad
    else:
        document[section] = bad
    with pytest.raises(ValueError, match="must be a JSON object|must be an object"):
        validator.validate_document(document)


@pytest.mark.parametrize(
    "mutation",
    ["missing", "empty", "reversed", "extra", "wrong_type"],
)
def test_pending_successor_gates_cannot_be_erased_or_reordered(
    mutation: str,
) -> None:
    document = _manifest()
    gates = document["remaining_successor_gates"]
    if mutation == "missing":
        document.pop("remaining_successor_gates")
    elif mutation == "empty":
        document["remaining_successor_gates"] = []
    elif mutation == "reversed":
        document["remaining_successor_gates"] = list(reversed(gates))
    elif mutation == "extra":
        document["remaining_successor_gates"].append("FIT_OR_TRAINING_ALLOWED")
    else:
        document["remaining_successor_gates"] = "all-cleared"
    message = (
        "reservation contract fields are not closed-world"
        if mutation == "missing" else "remaining successor gates drift"
    )
    with pytest.raises(ValueError, match=message):
        validator.validate_document(document)


def _resign_evidence(evidence: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    body = copy.deepcopy(evidence)
    body.pop("evidence_identity_sha256")
    identity = validator.hashlib.sha256(validator._canonical_bytes(body)).hexdigest()
    evidence["evidence_identity_sha256"] = identity
    monkeypatch.setattr(validator, "EXPECTED_EVIDENCE_IDENTITY", identity)


@pytest.mark.parametrize(
    "mutation",
    ["missing", "empty", "reversed", "extra", "wrong_type"],
)
def test_resealed_evidence_cannot_change_pending_gates(
    mutation: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _evidence()
    gates = evidence["remaining_gates"]
    if mutation == "missing":
        evidence.pop("remaining_gates")
    elif mutation == "empty":
        evidence["remaining_gates"] = []
    elif mutation == "reversed":
        evidence["remaining_gates"] = list(reversed(gates))
    elif mutation == "extra":
        evidence["remaining_gates"].append("ALL_GATES_COMPLETE")
    else:
        evidence["remaining_gates"] = "done"
    _resign_evidence(evidence, monkeypatch)
    message = (
        "materialization evidence fields are not closed-world"
        if mutation == "missing" else "evidence remaining gates drift"
    )
    with pytest.raises(ValueError, match=message):
        validator.validate_materialization_evidence(_manifest(), evidence)


@pytest.mark.parametrize(
    ("field", "bad", "message"),
    [
        ("objects", [[], _evidence()["objects"][1]], "reserved object must be an object"),
        ("truth_boundary", [], "evidence truth boundary must be an object"),
    ],
)
def test_resealed_evidence_rejects_malformed_nested_shapes(
    field: str, bad: object, message: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _evidence()
    evidence[field] = bad
    _resign_evidence(evidence, monkeypatch)
    with pytest.raises(ValueError, match=message):
        validator.validate_materialization_evidence(_manifest(), evidence)


def test_direct_evidence_validator_rejects_malformed_reservation_shape() -> None:
    document = _manifest()
    document["reservation"] = []
    with pytest.raises(ValueError, match="reservation must be a JSON object"):
        validator.validate_materialization_evidence(document, _evidence())


@pytest.mark.parametrize(
    "invalid", [object(), (1, 2), 1 + 2j, {"nested": {1: "bad"}}],
)
def test_programmatic_authority_rejects_non_json_types(invalid: object) -> None:
    document = _manifest()
    document["unexpected"] = invalid
    with pytest.raises(ValueError, match="not a JSON scalar|non-string JSON key"):
        validator.validate_document(document)


def test_programmatic_authority_rejects_invalid_utf8() -> None:
    document = _manifest()
    document["unexpected"] = chr(0xD800)
    with pytest.raises(ValueError, match="contains invalid UTF-8"):
        validator.validate_document(document)


@pytest.mark.parametrize(
    ("section", "injected", "message"),
    [
        ("root", "training_allowed", "reservation contract"),
        ("predecessor", "untrusted_head_sha", "predecessor"),
        ("reservation", "evaluation_authorized", "reservation"),
        ("object", "training_authorized", "reserved object"),
        ("materialization_evidence", "alternative_identity", "evidence reference"),
        ("truth_boundary", "final_test_allowed", "reservation truth boundary"),
    ],
)
def test_reservation_rejects_injected_contradictory_fields(
    section: str, injected: str, message: str,
) -> None:
    doc = _manifest()
    if section == "root":
        target = doc
    elif section == "object":
        target = doc["objects"][0]
    else:
        target = doc[section]
    target[injected] = True
    with pytest.raises(ValueError, match=message + " fields are not closed-world"):
        validator.validate_document(doc)


@pytest.mark.parametrize(
    ("section", "injected", "message"),
    [
        ("root", "training_allowed", "materialization evidence"),
        ("object", "future_training_authorized", "evidence reserved object"),
        ("truth_boundary", "final_test_allowed", "evidence truth boundary"),
    ],
)
def test_resealed_evidence_rejects_injected_contradictory_fields(
    section: str, injected: str, message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _evidence()
    target = (
        evidence if section == "root" else
        evidence["objects"][0] if section == "object" else
        evidence["truth_boundary"]
    )
    target[injected] = True
    _resign_evidence(evidence, monkeypatch)
    with pytest.raises(ValueError, match=message + " fields are not closed-world"):
        validator.validate_materialization_evidence(_manifest(), evidence)


@pytest.mark.parametrize(
    ("section", "replacement", "message"),
    [
        ("completed_gate", "ALL_GATES_COMPLETE", "completed-gate mismatch"),
        ("reservation_authority_issue", False, "reservation authority issue drift"),
    ],
)
def test_resealed_evidence_rejects_false_completion_and_authority_issue(
    section: str, replacement: object, message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _evidence()
    evidence[section] = replacement
    _resign_evidence(evidence, monkeypatch)
    with pytest.raises(ValueError, match=message):
        validator.validate_materialization_evidence(_manifest(), evidence)


def test_resealed_evidence_rejects_duplicate_source_family(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _evidence()
    evidence["objects"][1] = copy.deepcopy(evidence["objects"][0])
    payload = {
        "reservation_effective_at_utc": _manifest()["reservation"]["effective_at_utc"],
        "objects": evidence["objects"],
    }
    evidence["object_set_identity_sha256"] = validator.hashlib.sha256(
        validator._canonical_bytes(payload)
    ).hexdigest()
    _resign_evidence(evidence, monkeypatch)
    with pytest.raises(ValueError, match="duplicate evidence repository"):
        validator.validate_materialization_evidence(_manifest(), evidence)


def test_resealed_evidence_rejects_reserved_object_purpose_promotion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _evidence()
    evidence["objects"][0]["evaluation_use"] = "final_test"
    payload = {
        "reservation_effective_at_utc": _manifest()["reservation"]["effective_at_utc"],
        "objects": evidence["objects"],
    }
    evidence["object_set_identity_sha256"] = validator.hashlib.sha256(
        validator._canonical_bytes(payload)
    ).hexdigest()
    _resign_evidence(evidence, monkeypatch)
    with pytest.raises(ValueError, match="evidence purpose drift"):
        validator.validate_materialization_evidence(_manifest(), evidence)


@pytest.mark.parametrize("target", ["contract", "evidence"])
def test_evaluation_authority_oversize_blocks_before_validation(
    tmp_path: Path, target: str,
) -> None:
    contract = tmp_path / "контракт з пробілами.json"
    evidence = tmp_path / "evidence.json"
    contract.write_bytes(MANIFEST.read_bytes())
    evidence.write_bytes(EVIDENCE.read_bytes())
    blocked = contract if target == "contract" else evidence
    blocked.write_bytes(b"{}" + b" " * validator.MAX_INPUT_BYTES)
    with pytest.raises(ValueError, match="exceeds byte limit"):
        validator.validate(contract, evidence)
    assert blocked.read_bytes().startswith(b"{}")


def test_evaluation_loader_accepts_exact_byte_limit(tmp_path: Path) -> None:
    path = tmp_path / "limit.json"
    raw = b'{"data":0}'
    path.write_bytes(raw + b" " * (validator.MAX_INPUT_BYTES - len(raw)))
    assert validator._load_mapping(path) == {"data": 0}


def test_evaluation_loader_rejects_excessive_nodes(tmp_path: Path) -> None:
    path = tmp_path / "too-many-nodes.json"
    # Root mapping + list + primitive nodes exceed the fixed node bound.
    raw = '{"data":[' + ",".join(["0"] * validator.MAX_JSON_NODES) + "]}"
    path.write_text(raw, encoding="utf-8")
    with pytest.raises(ValueError, match="exceeds node limit"):
        validator._load_mapping(path)


@pytest.mark.parametrize("kind", ["invalid_utf8", "surrogate_key", "surrogate_value"])
def test_evaluation_loader_rejects_invalid_unicode(
    tmp_path: Path, kind: str,
) -> None:
    path = tmp_path / "unicode.json"
    raw = {
        "invalid_utf8": b'{"data":"\xff"}',
        "surrogate_key": br'{"\ud800":"data"}',
        "surrogate_value": br'{"data":"\ud800"}',
    }[kind]
    path.write_bytes(raw)
    with pytest.raises((ValueError, UnicodeError)):
        validator._load_mapping(path)


def test_bounded_loader_preserves_committed_eval647_authority() -> None:
    result = validator.validate(MANIFEST, EVIDENCE)
    assert result["reserved_objects"] == 2
    assert result["selection_validation_records_authorized"] == 0


@pytest.mark.parametrize("target", ["contract", "evidence"])
@pytest.mark.parametrize("failure", ["oversize", "invalid_utf8", "surrogate", "nonfinite"])
def test_eval647_cli_reports_one_zero_credit_error(
    tmp_path: Path, target: str, failure: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    contract = tmp_path / "контракт із пробілами.json"
    evidence = tmp_path / "доказ із пробілами.json"
    contract.write_bytes(MANIFEST.read_bytes())
    evidence.write_bytes(EVIDENCE.read_bytes())
    bad_bytes = {
        "oversize": b"{}" + b" " * validator.MAX_INPUT_BYTES,
        "invalid_utf8": b'{"bad":"\xff"}',
        "surrogate": br'{"bad":"\ud800"}',
        "nonfinite": b'{"bad":1e400}',
    }[failure]
    (contract if target == "contract" else evidence).write_bytes(bad_bytes)
    monkeypatch.setattr(validator, "DEFAULT_MANIFEST", contract)
    monkeypatch.setattr(validator, "DEFAULT_EVIDENCE", evidence)

    assert validator.main() == 2
    output = capsys.readouterr()
    assert output.err == ""
    assert len(output.out.splitlines()) == 1
    result = json.loads(output.out)
    assert result["status"] == "BLOCKED_INVALID_EVAL647_AUTHORITY"
    assert result["selection_validation_records_authorized"] == 0
    assert result["model_training_authorized"] is False
    assert result["final_test_outcomes_read"] is False
    assert (contract if target == "contract" else evidence).read_bytes() == bad_bytes


def test_eval647_cli_keeps_existing_valid_output(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(validator, "DEFAULT_MANIFEST", MANIFEST)
    monkeypatch.setattr(validator, "DEFAULT_EVIDENCE", EVIDENCE)
    assert validator.main() == 0
    output = capsys.readouterr()
    assert output.err == ""
    result = json.loads(output.out)
    assert result["reserved_objects"] == 2
    assert result["selection_validation_records_authorized"] == 0


def test_programmatic_authority_shared_dag_has_node_budget() -> None:
    document = _manifest()
    shared: object = 0
    for _ in range(15):
        shared = [shared, shared]
    document["predecessor"]["head_sha"] = shared
    with pytest.raises(ValueError, match="exceeds node limit"):
        validator.validate_document(document)


@pytest.mark.parametrize("bad", [[], {}])
def test_resealed_evidence_rejects_unhashable_repository(
    bad: object, monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _evidence()
    evidence["objects"][0]["repository"] = bad
    _resign_evidence(evidence, monkeypatch)
    with pytest.raises(ValueError, match="evidence repository must be a string"):
        validator.validate_materialization_evidence(_manifest(), evidence)


def test_parsed_value_node_limit_does_not_count_json_mapping_keys(
    tmp_path: Path,
) -> None:
    document = {"items": {str(i): 0 for i in range(validator.MAX_JSON_NODES - 2)}}
    path = tmp_path / "node-limit.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    assert validator._load_mapping(path) == document


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("reservation_training", "training accidentally allowed"),
        ("truth_training", "model training prematurely authorized"),
        ("object_sha", "identity drift for jd/tenacity:raw_sha256"),
        ("evidence_identity", "evidence identity drift"),
    ],
)
def test_direct_evidence_validation_rejects_invalid_contract(
    mutation: str, error: str,
) -> None:
    document = _manifest()
    if mutation == "reservation_training":
        document["reservation"]["training_allowed"] = True
    elif mutation == "truth_training":
        document["truth_boundary"]["model_training_authorized"] = True
    elif mutation == "object_sha":
        document["objects"][0]["raw_sha256"] = "0" * 64
    else:
        document["materialization_evidence"]["identity_sha256"] = "0" * 64
    with pytest.raises(ValueError, match=error):
        validator.validate_materialization_evidence(document, _evidence())
