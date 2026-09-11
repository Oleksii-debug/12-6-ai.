from __future__ import annotations

import sys
from types import ModuleType

import pytest

from twelve_six.data.expanded_global_dedup_v9 import (
    ExpandedDedupError,
    _verify_matcher_semantic_closure,
)

V3_NAME = "twelve_six.data.cross_source_capacity_audit_v3"
V1_NAME = "twelve_six.data.cross_source_capacity_audit"
DATA232_NAME = "twelve_six.data._data232_decontamination_matching"


def _callback(module_name: str):
    def callback(*args, **kwargs):
        raise AssertionError("matcher callback must not execute")

    callback.__module__ = module_name
    return callback


def _helper():
    def helper(*args, **kwargs):
        return None

    helper.__module__ = DATA232_NAME
    return helper


def _install_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[ModuleType, ModuleType, ModuleType, object, object]:
    v3 = ModuleType(V3_NAME)
    v1 = ModuleType(V1_NAME)
    data232 = ModuleType(DATA232_NAME)
    audit = _callback(V3_NAME)
    verify = _callback(V3_NAME)
    v3.audit_payloads = audit
    v3.verify_report = verify

    normalize = _helper()
    skeleton = _helper()
    thresholds = {"sentinel": 1.0}
    token_re = object()
    data232.normalize_for_contamination = normalize
    data232.code_skeleton_tokens = skeleton
    data232.DEFAULT_THRESHOLDS = thresholds
    data232.TOKEN_RE = token_re
    v1.normalize_for_contamination = normalize
    v1.code_skeleton_tokens = skeleton
    v1.DEFAULT_THRESHOLDS = thresholds
    v1.TOKEN_RE = token_re
    v3.v1 = v1

    monkeypatch.setitem(sys.modules, V3_NAME, v3)
    monkeypatch.setitem(sys.modules, V1_NAME, v1)
    monkeypatch.setitem(sys.modules, DATA232_NAME, data232)
    return v3, v1, data232, audit, verify


def test_same_named_v1_module_substitution_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    v3, _, data232, audit, verify = _install_runtime(monkeypatch)
    forged_v1 = ModuleType(V1_NAME)
    forged_v1.normalize_for_contamination = _helper()
    forged_v1.code_skeleton_tokens = _helper()
    forged_v1.DEFAULT_THRESHOLDS = data232.DEFAULT_THRESHOLDS
    forged_v1.TOKEN_RE = data232.TOKEN_RE
    v3.v1 = forged_v1

    with pytest.raises(
        ExpandedDedupError,
        match="terminal V3 base matcher dependency object replaced",
    ):
        _verify_matcher_semantic_closure(audit, verify)


def test_value_equal_data232_global_substitution_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, v1, data232, audit, verify = _install_runtime(monkeypatch)
    substituted_thresholds = dict(data232.DEFAULT_THRESHOLDS)
    assert substituted_thresholds == data232.DEFAULT_THRESHOLDS
    assert substituted_thresholds is not data232.DEFAULT_THRESHOLDS
    v1.DEFAULT_THRESHOLDS = substituted_thresholds

    with pytest.raises(
        ExpandedDedupError,
        match="terminal V1 DATA-232 dependency object replaced: DEFAULT_THRESHOLDS",
    ):
        _verify_matcher_semantic_closure(audit, verify)
