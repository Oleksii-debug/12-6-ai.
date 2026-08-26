from __future__ import annotations

import json
from pathlib import Path

import pytest

from twelve_six.next100_041_cpython_code_admission import (
    CpythonAdmissionError,
    _assert_code_only_path,
    _family_identity,
    _near_jaccard,
    _parse_python,
    _scan_privacy_and_secrets,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = REPO_ROOT / "configs/data/next100_041_cpython_code_rights_policy_v1.json"


def _policy() -> dict:
    return json.loads(POLICY_PATH.read_text(encoding="utf-8"))


def test_policy_is_bounded_code_only_and_one_family() -> None:
    policy = _policy()
    assert policy["worker_id"] == "NEXT100-041-CODE-CPYTHON"
    assert policy["upstream"]["canonical_family_id"] == "github:python/cpython"
    assert policy["upstream"]["commit"] == "9036982ed73d17848d45b60b7550f097371214e4"
    assert policy["license"]["license_id"] == "PSF-2.0"
    assert policy["limits"]["max_files"] == 3
    assert len(policy["inventory"]) == 3
    assert {item["path"] for item in policy["inventory"]} == {
        "Lib/graphlib.py",
        "Lib/fnmatch.py",
        "Lib/bisect.py",
    }
    assert all(item["source_kind"] == "source_code" for item in policy["inventory"])
    assert all(item["purpose"] == "pretraining" for item in policy["inventory"])
    assert policy["training_purpose_authority"]["evaluation"] == "NOT_ADMITTED"
    assert policy["training_purpose_authority"]["reserved_for_evaluation"] is False


def test_code_only_path_boundary_rejects_docs_tests_and_non_python() -> None:
    _assert_code_only_path("Lib/graphlib.py")
    for path in ("Doc/library/graphlib.rst", "Lib/test/test_graphlib.py", "Modules/gcmodule.c"):
        with pytest.raises(CpythonAdmissionError):
            _assert_code_only_path(path)


def test_privacy_and_secret_scan_fails_closed() -> None:
    assert _scan_privacy_and_secrets(b"def f():\n    return 1\n", "Lib/f.py")["passed"] is True
    with pytest.raises(CpythonAdmissionError):
        _scan_privacy_and_secrets(b"# maintainer@example.com\n", "Lib/f.py")
    with pytest.raises(CpythonAdmissionError):
        _scan_privacy_and_secrets(b"ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ123456\n", "Lib/f.py")


def test_parse_validity_and_near_duplicate_signal() -> None:
    assert _parse_python("def f(x):\n    return x + 1\n", "Lib/f.py")["passed"] is True
    with pytest.raises(SyntaxError):
        _parse_python("def f(:\n", "Lib/bad.py")
    assert _near_jaccard("def f(x): return x + 1", "def f(x): return x + 1") == 1.0
    assert _near_jaccard("def f(x): return x + 1", "class Z: pass") < 0.85


def test_family_identity_collapses_all_cpython_files() -> None:
    identity = _family_identity(_policy())
    assert len(identity) == 64
    assert all(ch in "0123456789abcdef" for ch in identity)
