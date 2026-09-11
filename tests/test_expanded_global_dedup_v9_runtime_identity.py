from __future__ import annotations

from types import ModuleType

import pytest

from twelve_six.data import _data232_decontamination_matching as data232
from twelve_six.data import cross_source_capacity_audit as v1
from twelve_six.data import cross_source_capacity_audit_v3 as v3
from twelve_six.data.expanded_global_dedup_v9 import (
    ExpandedDedupError,
    _verify_matcher_semantic_closure,
)


def _spoofed_helper(name: str):
    original = getattr(data232, name)

    def wrapper(*args, **kwargs):
        return original(*args, **kwargs)

    wrapper.__name__ = name
    wrapper.__module__ = data232.__name__
    return wrapper


def test_same_named_v1_module_substitution_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    forged_v1 = ModuleType(v1.__name__)
    forged_v1.normalize_for_contamination = _spoofed_helper("normalize_for_contamination")
    forged_v1.code_skeleton_tokens = _spoofed_helper("code_skeleton_tokens")
    forged_v1.DEFAULT_THRESHOLDS = data232.DEFAULT_THRESHOLDS
    forged_v1.TOKEN_RE = data232.TOKEN_RE

    monkeypatch.setattr(v3, "v1", forged_v1)

    with pytest.raises(
        ExpandedDedupError,
        match="terminal V3 base matcher dependency object replaced",
    ):
        _verify_matcher_semantic_closure(v3.audit_payloads, v3.verify_report)


def test_value_equal_data232_global_substitution_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    substituted_thresholds = dict(data232.DEFAULT_THRESHOLDS)
    assert substituted_thresholds == data232.DEFAULT_THRESHOLDS
    assert substituted_thresholds is not data232.DEFAULT_THRESHOLDS

    monkeypatch.setattr(v1, "DEFAULT_THRESHOLDS", substituted_thresholds)

    with pytest.raises(
        ExpandedDedupError,
        match="terminal V1 DATA-232 dependency object replaced: DEFAULT_THRESHOLDS",
    ):
        _verify_matcher_semantic_closure(v3.audit_payloads, v3.verify_report)
