from __future__ import annotations

import re
import sys
import unicodedata
from types import FunctionType, ModuleType

import pytest

import twelve_six.data.expanded_global_dedup_v9 as v9
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


def _template(*args, **kwargs):
    return None


def _never_execute(*args, **kwargs):
    raise AssertionError("matcher callback must not execute")


def _conditional_pair_matches(*args, **kwargs):
    if args and "rada" in repr(args[-1]).lower():
        return [{"match_type": "forged"}]
    return []


class _EqualitySpoofFloat:
    def __eq__(self, other: object) -> bool:
        return True

    def __float__(self) -> float:
        return 0.0


class _CapacityMatchSpoof(str):
    pass


def _bind_function(module: ModuleType, name: str, template) -> object:
    function = FunctionType(template.__code__, module.__dict__, name)
    function.__module__ = module.__name__
    setattr(module, name, function)
    return function


def _install_full_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[ModuleType, ModuleType, ModuleType, object, object]:
    current_v3 = ModuleType(V3_NAME)
    current_v1 = ModuleType(V1_NAME)
    current_data232 = ModuleType(DATA232_NAME)
    reference_v3 = ModuleType("_reference_v3")
    reference_v1 = ModuleType("_reference_v1")
    reference_data232 = ModuleType("_reference_data232")

    for current, reference, names in (
        (current_v3, reference_v3, v9._V3_RUNTIME_FUNCTIONS),
        (current_v1, reference_v1, v9._V1_RUNTIME_FUNCTIONS),
        (current_data232, reference_data232, v9._DATA232_RUNTIME_FUNCTIONS),
    ):
        for name in names:
            template = _never_execute if name in {"audit_payloads", "verify_report"} else _template
            _bind_function(current, name, template)
            _bind_function(reference, name, template)

    for name, value in {
        "SCHEMA": "v1-schema",
        "ALGORITHM": "v1-algorithm",
        "COLLAPSE_MATCH_TYPES": {"raw_exact"},
        "STATUS_SCOPES": {"terminal": {"REGISTRY_TERMINAL"}},
    }.items():
        setattr(current_v1, name, value)
        setattr(reference_v1, name, value.copy() if isinstance(value, (dict, set)) else value)

    for name, value in {
        "SCHEMA": "v3-schema",
        "INVENTORY_SCHEMA": "v3-inventory",
        "ALGORITHM": "v3-algorithm",
        "TERMINAL_STATUSES": {"REGISTRY_TERMINAL"},
        "RUST_BOOK_PROSE_POLICY": "prose",
        "RELATION_MATCH_TYPES": {"mirror": "lineage_mirror"},
        "LINEAGE_COLLAPSE_MATCH_TYPES": {"lineage_mirror"},
    }.items():
        setattr(current_v3, name, value)
        setattr(reference_v3, name, value.copy() if isinstance(value, (dict, set)) else value)
    current_v3.CAPACITY_COLLAPSE_MATCH_TYPES = {"raw_exact", "lineage_mirror"}

    for name in v9._V3_IDENTITY_GLOBALS:
        sentinel = object()
        setattr(current_v3, name, sentinel)
        setattr(reference_v3, name, sentinel)

    thresholds = {"natural_near_jaccard": 0.80}
    invisible = {1: None}
    keywords = {"if"}
    current_data232.DEFAULT_THRESHOLDS = thresholds
    reference_data232.DEFAULT_THRESHOLDS = dict(thresholds)
    current_data232.INVISIBLE = invisible
    reference_data232.INVISIBLE = dict(invisible)
    current_data232.KEYWORDS = keywords
    reference_data232.KEYWORDS = set(keywords)

    regex_values = {
        "TOKEN_RE": re.compile(r"\w+"),
        "CODE_TOKEN_RE": re.compile(r"\w+"),
        "LINE_COMMENT": re.compile(r"#.*$"),
        "BLOCK_COMMENT": re.compile(r"/\*.*?\*/"),
        "STRING": re.compile(r"'[^']*'"),
    }
    for name, value in regex_values.items():
        setattr(current_data232, name, value)
        setattr(reference_data232, name, re.compile(value.pattern, value.flags))

    current_data232.re = reference_data232.re = re
    current_data232.unicodedata = reference_data232.unicodedata = unicodedata

    for name in v9._V1_IDENTITY_GLOBALS:
        sentinel = object()
        setattr(current_v1, name, sentinel)
        setattr(reference_v1, name, sentinel)

    for name in (
        "normalize_for_contamination",
        "code_skeleton_tokens",
        "DEFAULT_THRESHOLDS",
        "TOKEN_RE",
    ):
        setattr(current_v1, name, getattr(current_data232, name))

    current_v3.v1 = current_v1
    audit = current_v3.audit_payloads
    verify = current_v3.verify_report

    monkeypatch.setitem(sys.modules, V3_NAME, current_v3)
    monkeypatch.setitem(sys.modules, V1_NAME, current_v1)
    monkeypatch.setitem(sys.modules, DATA232_NAME, current_data232)
    monkeypatch.setattr(
        v9,
        "_module_source_blob",
        lambda module: v9._EXPECTED_MATCHER_BLOBS[module.__name__],
    )
    references = {
        DATA232_NAME: reference_data232,
        V1_NAME: reference_v1,
        V3_NAME: reference_v3,
    }
    monkeypatch.setattr(
        v9,
        "_load_reference_module",
        lambda module, *, label: references[module.__name__],
    )
    return current_v3, current_v1, current_data232, audit, verify


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


def test_conditional_v1_pair_matches_substitution_is_rejected_before_matcher_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, v1, _, audit, verify = _install_full_runtime(monkeypatch)
    _bind_function(v1, "_pair_matches", _conditional_pair_matches)

    with pytest.raises(
        ExpandedDedupError,
        match="V1 runtime function code replaced: _pair_matches",
    ):
        _verify_matcher_semantic_closure(audit, verify)


def test_in_place_threshold_equality_spoof_is_rejected_before_matcher_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, data232, audit, verify = _install_full_runtime(monkeypatch)
    spoof = _EqualitySpoofFloat()
    data232.DEFAULT_THRESHOLDS["natural_near_jaccard"] = spoof

    assert spoof == 0.80
    assert float(spoof) == 0.0
    assert data232.DEFAULT_THRESHOLDS == {"natural_near_jaccard": 0.80}

    with pytest.raises(
        ExpandedDedupError,
        match="DATA-232 runtime value replaced: DEFAULT_THRESHOLDS",
    ):
        _verify_matcher_semantic_closure(audit, verify)


def test_capacity_collapse_member_type_spoof_is_rejected_before_matcher_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    v3, _, _, audit, verify = _install_full_runtime(monkeypatch)
    v3.CAPACITY_COLLAPSE_MATCH_TYPES = {
        _CapacityMatchSpoof("raw_exact"),
        "lineage_mirror",
    }

    assert v3.CAPACITY_COLLAPSE_MATCH_TYPES == {"raw_exact", "lineage_mirror"}

    with pytest.raises(
        ExpandedDedupError,
        match="V3 runtime value replaced: CAPACITY_COLLAPSE_MATCH_TYPES",
    ):
        _verify_matcher_semantic_closure(audit, verify)
