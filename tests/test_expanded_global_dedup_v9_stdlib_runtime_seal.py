from __future__ import annotations

import builtins
import copy
from collections.abc import Mapping
from typing import Any

import pytest

from twelve_six.data import expanded_global_dedup_v9 as v9


def _never_matcher_audit(
    inventory: Mapping[str, Any], payloads: Mapping[str, bytes]
) -> Mapping[str, Any]:
    del inventory, payloads
    raise AssertionError("matcher callback must not execute after runtime dependency drift")


def _never_matcher_verify(report: Mapping[str, Any]) -> None:
    del report
    raise AssertionError("matcher verifier must not execute after runtime dependency drift")


def test_builtin_float_substitution_fails_before_matcher_callbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_float = builtins.float

    def substituted_float(value: object = 0.0) -> float:
        del value
        return original_float(0.0)

    monkeypatch.setattr(builtins, "float", substituted_float)

    with pytest.raises(
        v9.ExpandedDedupError,
        match=r"stdlib runtime builtin replaced: builtins\.float",
    ):
        v9._verify_matcher_semantic_closure(
            _never_matcher_audit,
            _never_matcher_verify,
        )


def test_copy_deepcopy_substitution_fails_before_matcher_callbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def substituted_deepcopy(value: object, memo: object = None) -> object:
        del memo
        return value

    monkeypatch.setattr(copy, "deepcopy", substituted_deepcopy)

    with pytest.raises(
        v9.ExpandedDedupError,
        match=r"stdlib runtime member replaced: copy\.deepcopy",
    ):
        v9._verify_matcher_semantic_closure(
            _never_matcher_audit,
            _never_matcher_verify,
        )


def test_builtin_runtime_seal_covers_threshold_and_collection_primitives() -> None:
    frozen = {name: value for name, value in v9._FROZEN_BUILTIN_CALLABLES}
    for name in (
        "any",
        "dict",
        "float",
        "frozenset",
        "int",
        "isinstance",
        "len",
        "list",
        "map",
        "min",
        "range",
        "round",
        "set",
        "sorted",
        "str",
        "sum",
        "tuple",
    ):
        assert frozen[name] is builtins.__dict__[name]
