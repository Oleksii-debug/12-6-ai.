"""Bounded diagnostic regressions; no training or source authority is granted."""

from __future__ import annotations

import hashlib
import importlib.util
import marshal
import py_compile
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from types import CodeType, FunctionType

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


def test_deep_code_is_bounded_before_marshal(monkeypatch: pytest.MonkeyPatch) -> None:
    code, _ = _function("def candidate():\n    return 1\n")
    nested = code
    for _ in range(40):
        nested = nested.replace(co_consts=(nested,))

    def forbidden_marshal(_value: object) -> bytes:
        raise AssertionError("over-depth code must not be marshalled")

    monkeypatch.setattr(marshal, "dumps", forbidden_marshal)
    report = compare_code_objects(nested, nested)
    assert report["classification"] == "INCOMPLETE_DIAGNOSTIC"
    assert report["diagnostic_limited"] is True
    assert report["marshal_equal"] is None
    assert report["live_marshal_sha256"] is None
    assert report["canonical_marshal_sha256"] is None
    assert report["attestation_override_allowed"] is False


def test_same_nan_constant_is_not_false_structural_drift() -> None:
    code, _ = _function("def candidate():\n    return 1\n")
    candidate = code.replace(co_consts=(float("nan"),))
    report = compare_code_objects(candidate, candidate)
    assert report["classification"] == "NO_CODE_MISMATCH_OBSERVED"
    assert report["structural_fields_equal"] is True
    assert report["marshal_equal"] is True
    assert report["attestation_override_allowed"] is False


def test_complex_signed_zero_is_real_structural_difference() -> None:
    code, _ = _function("def candidate():\n    return 1\n")
    live = code.replace(co_consts=(complex(0.0, -0.0),))
    canonical = code.replace(co_consts=(complex(0.0, 0.0),))
    assert live.co_consts == canonical.co_consts
    assert marshal.dumps(live) != marshal.dumps(canonical)
    report = compare_code_objects(live, canonical)
    assert report["classification"] == "STRUCTURAL_CODE_MISMATCH"
    assert report["marshal_equal"] is False
    assert report["attestation_override_allowed"] is False


def test_partial_marshal_failure_discards_both_digest_claims(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    code, _ = _function("def candidate():\n    return 1\n")
    original = marshal.dumps
    calls = 0

    def fail_second(value: object) -> bytes:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("simulated marshal limit")
        return original(value)

    monkeypatch.setattr(marshal, "dumps", fail_second)
    report = compare_code_objects(code, code)
    assert report["classification"] == "INCOMPLETE_DIAGNOSTIC"
    assert report["marshal_equal"] is None
    assert report["live_marshal_sha256"] is None
    assert report["canonical_marshal_sha256"] is None
    assert report["attestation_override_allowed"] is False


def test_exact_historical_v3_pyc_matches_recompilation_after_warmup(tmp_path: Path) -> None:
    """Isolate pinned V3 bytecode from Caselaw transport and physical data."""
    root = Path(__file__).resolve().parents[1]
    if not (root / ".git").exists() or shutil.which("git") is None:
        pytest.skip("historical Git graph is unavailable outside a repository checkout")
    historical_sha = "d3333ec1b4a508df232a5aefccd6686adda745fb"
    historical_path = "src/twelve_six/data/cross_source_capacity_audit_v3.py"
    proc = subprocess.run(
        ["git", "-C", str(root), "show", f"{historical_sha}:{historical_path}"],
        capture_output=True,
        check=True,
        timeout=30,
    )
    source = proc.stdout
    git_blob = b"blob " + str(len(source)).encode("ascii") + b"\0" + source
    assert hashlib.sha1(git_blob, usedforsecurity=False).hexdigest() == (
        "11490b1803e0aa2266d8ac0053676efcfb0f91ba"
    )

    path = tmp_path / "cross_source_capacity_audit_v3.py"
    path.write_bytes(source)
    pyc = tmp_path / "historical_v3.pyc"
    py_compile.compile(str(path), cfile=str(pyc), dfile=str(path), doraise=True)
    archived = pyc.read_bytes()
    assert archived[:4] == importlib.util.MAGIC_NUMBER
    pyc_module = marshal.loads(archived[16:])
    current_module = compile(source, str(path), "exec", dont_inherit=True)

    def lineage_code(module: CodeType) -> CodeType:
        matches = [
            value for value in module.co_consts
            if type(value) is CodeType and value.co_name == "_lineage_matches"
        ]
        assert len(matches) == 1
        return matches[0]

    live = lineage_code(pyc_module)
    canonical = lineage_code(current_module)
    runtime = FunctionType(live, {"defaultdict": defaultdict, "RELATION_MATCH_TYPES": {}})
    fingerprints = [
        {
            "row": {
                "source_id": f"offline-{index}",
                "stable_object_id": f"offline-{index // 2}",
                "source_family": "offline-diagnostic-only",
            }
        }
        for index in range(16)
    ]
    for _ in range(100):
        matches = runtime(fingerprints, ())
        assert len(matches) == 8

    report = compare_code_objects(live, canonical)
    assert report["structural_fields_equal"] is True
    assert report["classification"] in {
        "NO_CODE_MISMATCH_OBSERVED", "SERIALIZATION_MISMATCH_UNRESOLVED"
    }
    assert report["attestation_override_allowed"] is False
    assert report["canonical_corpus_credit"] == 0


def test_nested_frozenset_is_limited_before_marshal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    code, _ = _function("def candidate():\n    return 1\n")
    nested = frozenset({frozenset({1})})
    live = code.replace(co_consts=(nested,))

    def forbidden_marshal(_value: object) -> bytes:
        raise AssertionError("nested frozen constants must not be marshalled")

    monkeypatch.setattr(marshal, "dumps", forbidden_marshal)
    report = compare_code_objects(live, live)
    assert report["classification"] == "INCOMPLETE_DIAGNOSTIC"
    assert report["diagnostic_limited"] is True
    assert report["marshal_equal"] is None
    assert report["live_marshal_sha256"] is None
    assert report["canonical_marshal_sha256"] is None
    assert report["attestation_override_allowed"] is False


def test_simple_scalar_frozenset_remains_comparable() -> None:
    code, _ = _function("def candidate():\n    return 1\n")
    frozen = frozenset({1, 2, 3})
    left = code.replace(co_consts=(frozen,))
    right = code.replace(co_consts=(frozenset({3, 2, 1}),))
    report = compare_code_objects(left, right)
    assert report["classification"] == "NO_CODE_MISMATCH_OBSERVED"
    assert report["marshal_equal"] is True
    assert report["attestation_override_allowed"] is False

@pytest.mark.parametrize("difference", ["tuple-length", "element-type"])
def test_mismatched_code_subtree_is_bounded_before_marshal(
    difference: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    code, _ = _function("def candidate():\n    return 1\n")
    deeply_nested = code
    for _ in range(40):
        deeply_nested = deeply_nested.replace(co_consts=(deeply_nested,))
    left = code.replace(co_consts=(0,))
    right_consts = (
        (0, deeply_nested) if difference == "tuple-length" else (deeply_nested,)
    )
    right = code.replace(co_consts=right_consts)

    def forbidden_marshal(_value: object) -> bytes:
        raise AssertionError("unexamined code subtree must not be marshalled")

    monkeypatch.setattr(marshal, "dumps", forbidden_marshal)
    report = compare_code_objects(left, right)
    assert report["classification"] == "INCOMPLETE_DIAGNOSTIC"
    assert report["diagnostic_limited"] is True
    assert report["marshal_equal"] is None
    assert report["live_marshal_sha256"] is None
    assert report["canonical_marshal_sha256"] is None
    assert report["attestation_override_allowed"] is False


def test_ordinary_tuple_length_mismatch_still_reports_structural_change() -> None:
    code, _ = _function("def candidate():\n    return 1\n")
    left = code.replace(co_consts=(1,))
    right = code.replace(co_consts=(1, 2))
    report = compare_code_objects(left, right)
    assert report["classification"] == "STRUCTURAL_CODE_MISMATCH"
    assert any(path.endswith(":length") for path in report["different_field_paths"])
    assert report["diagnostic_limited"] is False
    assert report["attestation_override_allowed"] is False

@pytest.mark.parametrize(
    "constant",
    [
        frozenset(range(6_000)),
        b"x" * 1_048_577,
        "x" * 1_048_577,
        frozenset({b"x" * 1_048_577}),
        1 << (8 * 1_048_576 + 1),
    ],
    ids=["frozen-node-budget", "bytes-width", "string-width",
         "frozen-scalar-width", "integer-bit-width"],
)
def test_oversized_scalar_or_frozen_set_never_reaches_marshal(
    constant: object, monkeypatch: pytest.MonkeyPatch,
) -> None:
    code, _ = _function("def candidate():\n    return 1\n")
    oversized = code.replace(co_consts=(constant,))

    def forbidden_marshal(_value: object) -> bytes:
        raise AssertionError("over-budget constants must not be marshalled")

    monkeypatch.setattr(marshal, "dumps", forbidden_marshal)
    report = compare_code_objects(oversized, oversized)
    assert report["classification"] == "INCOMPLETE_DIAGNOSTIC"
    assert report["diagnostic_limited"] is True
    assert report["marshal_equal"] is None
    assert report["live_marshal_sha256"] is None
    assert report["canonical_marshal_sha256"] is None
    assert report["attestation_override_allowed"] is False

@pytest.mark.parametrize("frozen", [False, True], ids=["tuple", "frozenset"])
def test_aggregate_scalar_budget_prevents_large_marshal(
    frozen: bool, monkeypatch: pytest.MonkeyPatch,
) -> None:
    code, _ = _function("def candidate():\n    return 1\n")
    values = tuple(bytes([index]) + b"x" * 899_999 for index in range(20))
    constant = frozenset(values) if frozen else values
    oversized = code.replace(co_consts=(constant,))

    def forbidden_marshal(_value: object) -> bytes:
        raise AssertionError("large aggregate constants must not reach marshal")

    monkeypatch.setattr(marshal, "dumps", forbidden_marshal)
    report = compare_code_objects(oversized, oversized)
    assert report["classification"] == "INCOMPLETE_DIAGNOSTIC"
    assert report["diagnostic_limited"] is True
    assert report["marshal_equal"] is None
    assert report["attestation_override_allowed"] is False


def test_moderate_scalar_constant_still_compares_normally() -> None:
    code, _ = _function("def candidate():\n    return 1\n")
    moderate = code.replace(co_consts=(b"x" * 200_000,))
    report = compare_code_objects(moderate, moderate)
    assert report["classification"] == "NO_CODE_MISMATCH_OBSERVED"
    assert report["marshal_equal"] is True
