from __future__ import annotations

import json
from pathlib import Path

import pytest

from twelve_six.next100_044_fastapi_code_admission import (
    FastapiAdmissionError,
    _assert_code_only_path,
    _family_identity,
    _near_jaccard,
    _parse_python,
    _scan_privacy_and_secrets,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = REPO_ROOT / "configs/data/next100_044_fastapi_code_rights_policy_v1.json"


def _policy() -> dict:
    return json.loads(POLICY_PATH.read_text(encoding="utf-8"))


def test_policy_is_bounded_fastapi_code_only_and_one_family() -> None:
    policy = _policy()
    assert policy["worker_id"] == "NEXT100-044-CODE-FASTAPI"
    assert policy["upstream"]["canonical_family_id"] == "github:fastapi/fastapi"
    assert policy["upstream"]["commit"] == "49033471594ea5d99a80abdf1043231b7791ee49"
    assert policy["license"]["license_id"] == "MIT"
    assert policy["license"]["blob_sha1"] == "3e92463e6bd522a2a21e5f0a80d8089d6c4be20d"
    assert policy["limits"]["max_files"] == 3
    assert {item["path"] for item in policy["inventory"]} == {
        "fastapi/sse.py",
        "fastapi/exceptions.py",
        "fastapi/datastructures.py",
    }
    assert len({item["blob_sha1"] for item in policy["inventory"]}) == 3
    assert all(item["source_kind"] == "source_code" for item in policy["inventory"])
    assert all(item["purpose"] == "pretraining" for item in policy["inventory"])
    assert policy["training_purpose_authority"]["decision"] == "ALLOWED"
    assert policy["training_purpose_authority"]["evaluation"] == "NOT_ADMITTED"
    assert policy["training_purpose_authority"]["reserved_for_evaluation"] is False
    assert policy["evaluation_boundary"]["code_record_count"] == 0
    mixed = policy["explicit_exclusions"]["mixed_provenance_review_required"]
    assert any("fastapi/encoders.py" in item for item in mixed)


def test_code_only_path_boundary_rejects_tests_docs_and_unselected_code() -> None:
    _assert_code_only_path("fastapi/sse.py")
    _assert_code_only_path("fastapi/exceptions.py")
    _assert_code_only_path("fastapi/datastructures.py")
    for path in (
        "tests/test_encoders.py",
        "docs/en/docs/index.md",
        "docs_src/response_model/tutorial001.py",
        "fastapi/encoders.py",
        "fastapi/applications.py",
        "fastapi/py.typed",
    ):
        with pytest.raises(FastapiAdmissionError):
            _assert_code_only_path(path)


def test_privacy_secret_and_private_endpoint_scan_fails_closed() -> None:
    assert _scan_privacy_and_secrets(b"def f():\n    return 1\n", "fastapi/f.py")["passed"] is True
    with pytest.raises(FastapiAdmissionError):
        _scan_privacy_and_secrets(b"# maintainer@example.com\n", "fastapi/f.py")
    with pytest.raises(FastapiAdmissionError):
        _scan_privacy_and_secrets(b"ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ123456\n", "fastapi/f.py")
    with pytest.raises(FastapiAdmissionError):
        _scan_privacy_and_secrets(b'url = "http://10.0.0.4/internal"\n', "fastapi/f.py")


def test_parse_validity_and_near_duplicate_signal() -> None:
    assert _parse_python("def f(x):\n    return x + 1\n", "fastapi/f.py")["passed"] is True
    with pytest.raises(SyntaxError):
        _parse_python("def f(:\n", "fastapi/bad.py")
    assert _near_jaccard("def f(x): return x + 1", "def f(x): return x + 1") == 1.0
    assert _near_jaccard("def f(x): return x + 1", "class Z: pass") < 0.85


def test_family_identity_collapses_all_fastapi_files() -> None:
    policy = _policy()
    identity = _family_identity(policy)
    assert len(identity) == 64
    assert all(ch in "0123456789abcdef" for ch in identity)
    assert policy["upstream"]["canonical_family_id"] not in policy["current_registry_binding"]["existing_families"]


def test_redistribution_notice_and_eval_boundary_are_explicit() -> None:
    policy = _policy()
    conditions = policy["license"]["redistribution_conditions"]
    assert "copyright notice" in conditions
    assert "permission notice" in conditions
    assert policy["evaluation_boundary"]["code_component"] == "EVAL-292"
    assert policy["evaluation_boundary"]["fastapi_inventory_reserved"] is False
