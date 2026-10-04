"""Regression for bounded nested frozenset intake in the V3 read-only diagnostic.

Synthetic code objects only. Never grants attestation or corpus/training credit.
"""

from __future__ import annotations

import marshal
from types import CodeType

import pytest

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


@pytest.mark.parametrize(
    "error", [TypeError, ValueError, RecursionError, OverflowError],
)
def test_member_serialization_failure_is_incomplete(
    monkeypatch: pytest.MonkeyPatch, error: type[Exception],
) -> None:
    code = _code_with_constant(frozenset({(1, (2, 3))}))
    original = marshal.dumps

    def fail_member(value: object) -> bytes:
        if type(value) is tuple:
            raise error("synthetic member serialization failure")
        return original(value)

    monkeypatch.setattr(marshal, "dumps", fail_member)
    _assert_incomplete(compare_code_objects(code, code))


def test_tuple_length_mismatch_cannot_hide_deep_unvisited_member() -> None:
    inner: object = 1
    for _ in range(40):
        inner = (inner,)
    left = _code_with_constant((inner,))
    right = _code_with_constant((inner, 2))
    _assert_incomplete(compare_code_objects(left, right))


def test_type_mismatch_cannot_hide_deep_unvisited_member() -> None:
    inner: object = 1
    for _ in range(40):
        inner = (inner,)
    left = _code_with_constant((1,))
    right = _code_with_constant((inner,))
    _assert_incomplete(compare_code_objects(left, right))


def test_small_length_and_type_mismatches_remain_diagnosable() -> None:
    left = _code_with_constant((1,))
    different_length = _code_with_constant((1, 2))
    different_type = _code_with_constant(((1,),))
    for right in (different_length, different_type):
        result = compare_code_objects(left, right)
        assert result["classification"] == "STRUCTURAL_CODE_MISMATCH"
        assert result["marshal_equal"] is False
        assert result["diagnostic_limited"] is False
        assert result["attestation_override_allowed"] is False
