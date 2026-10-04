"""Regression for bounded nested frozenset intake in the V3 read-only diagnostic.

Synthetic code objects only. Never grants attestation or corpus/training credit.
"""

from __future__ import annotations

from types import CodeType

from tools.diagnose_d03_v3_code_drift import compare_code_objects


def _code_with_constant(value: object) -> CodeType:
    module = compile("def candidate():\n    return 1\n", "synthetic/v3.py", "exec")
    code = next(item for item in module.co_consts if type(item) is CodeType)
    return code.replace(co_consts=(value,))


def _assert_incomplete(result: dict[str, object]) -> None:
    assert result["classification"] == "INCOMPLETE_DIAGNOSTIC"
    assert result["diagnostic_limited"] is True
    assert result["structural_fields_equal"] is False
    assert result["marshal_equal"] is None
    assert result["live_marshal_sha256"] is None
    assert result["canonical_marshal_sha256"] is None
    assert result["attestation_override_allowed"] is False
    assert result["canonical_corpus_credit"] == 0
    assert result["training_authorized"] is False


def test_deep_tuple_inside_frozenset_is_bounded_before_marshal() -> None:
    value: object = 1
    for _ in range(40):
        value = (value,)
    code = _code_with_constant(frozenset({value}))
    _assert_incomplete(compare_code_objects(code, code))


def test_wide_tuple_inside_frozenset_is_bounded_by_total_nodes() -> None:
    code = _code_with_constant(frozenset({tuple(range(6_000))}))
    _assert_incomplete(compare_code_objects(code, code))


def test_small_nested_frozenset_remains_diagnosable() -> None:
    code = _code_with_constant(frozenset({(1, (2, 3))}))
    result = compare_code_objects(code, code)
    assert result["classification"] == "NO_CODE_MISMATCH_OBSERVED"
    assert result["marshal_equal"] is True
    assert result["structural_fields_equal"] is True
    assert result["attestation_override_allowed"] is False


def test_different_small_nested_frozensets_are_structural_mismatch() -> None:
    left = _code_with_constant(frozenset({(1, (2, 3))}))
    right = _code_with_constant(frozenset({(1, (2, 4))}))
    result = compare_code_objects(left, right)
    assert result["classification"] == "STRUCTURAL_CODE_MISMATCH"
    assert result["marshal_equal"] is False
    assert result["attestation_override_allowed"] is False
