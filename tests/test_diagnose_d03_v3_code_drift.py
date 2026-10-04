"""Bounded diagnostic regressions; no training or source authority is granted."""

from __future__ import annotations

import marshal
from types import CodeType

import pytest

from tools.diagnose_d03_v3_code_drift import compare_code_objects


def _function(source: str) -> tuple[CodeType, object]:
    namespace: dict[str, object] = {}
    exec(compile(source, "exact/historical/v3.py", "exec", dont_inherit=True), namespace)
    candidate = namespace["candidate"]
    return candidate.__code__, candidate


def test_independent_compiles_of_same_nested_function_agree() -> None:
    source = "def candidate(items):\n    return [value * 3 for value in items]\n"
    live, function = _function(source)
    canonical, _ = _function(source)
    for _ in range(100):
        function(range(20))
    report = compare_code_objects(live, canonical)
    assert report["classification"] == "NO_CODE_MISMATCH_OBSERVED"
    assert report["structural_fields_equal"] is True
    assert report["marshal_equal"] is True
    assert report["different_field_paths"] == []
    assert report["attestation_override_allowed"] is False


def test_changed_nested_executable_code_is_identified_without_payload() -> None:
    live, _ = _function("def candidate(items):\n    return [value * 3 for value in items]\n")
    canonical, _ = _function("def candidate(items):\n    return [value * 4 for value in items]\n")
    report = compare_code_objects(live, canonical)
    assert report["classification"] == "STRUCTURAL_CODE_MISMATCH"
    assert any("co_consts" in path for path in report["different_field_paths"])
    assert report["attestation_override_allowed"] is False
    assert "value * 3" not in str(report)
    assert "value * 4" not in str(report)


def test_changed_code_flags_detected() -> None:
    live, _ = _function("def candidate(value):\n    return value + 1\n")
    modified = live.replace(co_flags=live.co_flags ^ 0x20)
    report = compare_code_objects(live, modified)
    assert report["classification"] == "STRUCTURAL_CODE_MISMATCH"
    assert "code.co_flags" in report["different_field_paths"]
    assert report["marshal_equal"] is False


def test_literal_data_never_appears_in_report() -> None:
    secret = "PRIVATE_SOURCE_LITERAL_NOT_FOR_OUTPUT"
    code, _ = _function(f"def candidate():\n    return {secret!r}\n")
    report = compare_code_objects(code, code)
    assert secret not in str(report)
    assert report["canonical_corpus_credit"] == 0
    assert report["training_authorized"] is False


@pytest.mark.parametrize("invalid", [None, 0, False, b"code"])
def test_non_code_objects_fail_closed(invalid: object) -> None:
    code, _ = _function("def candidate():\n    return 1\n")
    with pytest.raises(TypeError, match="code objects"):
        compare_code_objects(code, invalid)


def test_marshal_reference_alias_difference_is_never_a_bypass() -> None:
    """Equal constant values can have different marshal reference encoding."""
    code, _ = _function("def candidate():\n    return None\n")
    shared = "private-unprinted-" + ("z" * 500)
    distinct = shared.encode("utf-8").decode("utf-8")
    assert shared == distinct and shared is not distinct
    live = code.replace(co_consts=(None, shared, shared))
    canonical = code.replace(co_consts=(None, shared, distinct))
    assert live.co_consts == canonical.co_consts
    if marshal.dumps(live) == marshal.dumps(canonical):
        pytest.skip("this interpreter does not distinguish these alias encodings")
    report = compare_code_objects(live, canonical)
    assert report["classification"] == "SERIALIZATION_MISMATCH_UNRESOLVED"
    assert report["structural_fields_equal"] is True
    assert report["marshal_equal"] is False
    assert report["attestation_override_allowed"] is False
    assert shared not in str(report)


def test_adversarially_nested_code_is_bounded_and_untrusted() -> None:
    code, _ = _function("def candidate():\n    return 1\n")
    nested = code
    for _ in range(40):
        nested = nested.replace(co_consts=(nested,))
    report = compare_code_objects(nested, nested)
    assert report["classification"] == "INCOMPLETE_DIAGNOSTIC"
    assert report["diagnostic_limited"] is True
    assert report["structural_fields_equal"] is False
    assert report["attestation_override_allowed"] is False
