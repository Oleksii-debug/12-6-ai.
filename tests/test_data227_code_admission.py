from pathlib import Path

import pytest

from twelve_six.data227_code_admission import (
    BANNED_PATH_PARTS,
    Data227Error,
    _assert_path_allowed,
    _load_policy,
    _near_jaccard,
    _rights,
)


def test_policy_has_two_new_independent_families_and_explicit_use_decisions() -> None:
    policy = _load_policy(Path("."))
    decisions = policy["decisions"]
    assert {item["source_family"] for item in decisions} == {
        "github:encode/httpx",
        "github:psf/requests",
    }
    serialized = str(decisions)
    assert "itsdangerous" not in serialized
    assert "pytest-dev/pluggy" not in serialized
    assert all(item["uses"]["model_training"] == "ALLOWED" for item in decisions)
    assert all(item["uses"]["redistribution"] == "ALLOWED" for item in decisions)
    assert {item["license_id"] for item in decisions} == {"BSD-3-Clause", "Apache-2.0"}


def test_data24_rights_decision_is_machine_readable_and_source_bound() -> None:
    decision = _load_policy(Path("."))["decisions"][0]
    rights = _rights(decision, license_sha256="0" * 64, policy_sha256="1" * 64)
    assert rights.allows_model_training is True
    assert rights.uses is not None
    assert rights.uses.model_training == "ALLOWED"
    assert rights.uses.redistribution == "ALLOWED"
    assert {item.evidence_kind for item in rights.evidence_refs} == {
        "license_text",
        "policy_decision",
    }
    assert all(item.source_id == decision["source_id"] for item in rights.evidence_refs)
    assert all(
        item.source_version == f"git:{decision['commit']}" for item in rights.evidence_refs
    )


@pytest.mark.parametrize(
    "path",
    [
        "vendor/pkg/a.py",
        "src/generated/a.py",
        "build/out/a.py",
        "node_modules/pkg/index.js",
        "web/bundle.min.js",
        "binary/module.so",
    ],
)
def test_excluded_paths_fail_closed(path: str) -> None:
    with pytest.raises(Data227Error):
        _assert_path_allowed(path)


def test_normal_source_paths_are_allowed() -> None:
    _assert_path_allowed("httpx/_content.py")
    _assert_path_allowed("src/requests/_internal_utils.py")
    assert "vendor" in BANNED_PATH_PARTS


def test_near_duplicate_metric_is_fail_closed_at_high_similarity() -> None:
    left = "def alpha(x):\n    return x + 1\n"
    right = "def beta(y):\n    return y * 7\n"
    identical = _near_jaccard(left, left)
    distinct = _near_jaccard(left, right)
    assert identical == 1.0
    assert distinct < 0.85
