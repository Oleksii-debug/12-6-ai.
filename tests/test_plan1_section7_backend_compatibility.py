"""Nonpromoting Plan-1 S7 compatibility policy / adversarial tests."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from twelve_six.backend_compatibility import (
    CompatibilityError,
    StaticBackendMatrix,
    load_static_backend_matrix,
    matrix_blob_sha256,
    parse_static_backend_matrix,
)

ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = ROOT / "configs/compatibility/plan1_backend_matrix_v1.json"
RAW = MATRIX_PATH.read_bytes()
LOCK_SHA = "870587414a9108fc5d924a0caad37cd228515ad32df6639538deba48dc901b5a"


def parse(raw: bytes = RAW) -> StaticBackendMatrix:
    return parse_static_backend_matrix(raw, expected_lock_sha256=LOCK_SHA)


def changed(**changes: object) -> bytes:
    payload = json.loads(RAW)
    payload.update(changes)
    return json.dumps(payload).encode()


def test_real_repository_lock_validation_reused_not_forked() -> None:
    matrix = load_static_backend_matrix(root=ROOT)
    assert matrix == parse()
    assert len(matrix_blob_sha256(RAW)) == 64


def test_all_locked_platform_profiles_and_no_optimistic_gpu_support() -> None:
    matrix = parse()
    for profile in ("linux-aarch64", "linux-x86_64", "windows-x86_64"):
        baseline = matrix.lookup(
            profile, "python", "dependency_environment",
            expected_lock_sha256=LOCK_SHA,
        )
        assert (baseline.status, baseline.qualification) == ("supported", "UNQUALIFIED")
        experimental = matrix.lookup(
            profile, "cpu", "fixture_runtime",
            expected_lock_sha256=LOCK_SHA,
        )
        assert (experimental.status, experimental.qualification) == (
            "experimental", "UNQUALIFIED"
        )
        cuda = matrix.lookup(profile, "cuda", "training", expected_lock_sha256=LOCK_SHA)
        assert (cuda.status, cuda.qualification) == ("unsupported", "UNQUALIFIED")


@pytest.mark.parametrize("query", [
    ("linux-x86_64", "rocm", "training"),
    ("windows-aarch64", "python", "dependency_environment"),
    ("linux-x86_64", "cuda", "experimental-training"),
    ("linux-x86_64", "", "training"),
    ("linux-x86_64", True, "training"),
])
def test_unknown_is_unsupported_unqualified(query: tuple[object, object, object]) -> None:
    answer = parse().lookup(*query, expected_lock_sha256=LOCK_SHA)
    assert (answer.status, answer.qualification, answer.evidence_ref) == (
        "unsupported", "UNQUALIFIED", None
    )


def test_evidence_pointer_cannot_grant_readiness() -> None:
    matrix = parse()
    entry = matrix.entries[0]
    bound = replace(entry, evidence_ref="a" * 64)
    changed_entries = (bound, *matrix.entries[1:])
    new = replace(matrix, entries=changed_entries)
    answer = new.lookup(entry.profile, entry.backend, entry.feature,
                        expected_lock_sha256=LOCK_SHA)
    assert answer.evidence_ref == "a" * 64
    assert answer.qualification == "UNQUALIFIED"


@pytest.mark.parametrize("mutation", [
    {"schema_version": "12-6.static-backend-compatibility.v2"},
    {"python_version": "3.12"},
    {"lock_index_sha256": "f" * 64},
    {"lock_index_sha256": True},
    {"extra": "not allowed"},
    {"entries": []},
    {"entries": {}},
])
def test_bad_schema_lock_binding_and_missing_profiles_fail_closed(
    mutation: dict[str, object],
) -> None:
    with pytest.raises(CompatibilityError):
        parse(changed(**mutation))


def test_duplicate_json_keys_nonfinite_and_invalid_utf8_rejected() -> None:
    with pytest.raises(CompatibilityError):
        parse(RAW[:-2] + b',"schema_version":"oops"}')
    with pytest.raises(CompatibilityError):
        parse(b'{"x":NaN}')
    with pytest.raises(CompatibilityError):
        parse(b"\xff" + RAW)


def test_duplicate_or_unsorted_entries_rejected() -> None:
    obj = json.loads(RAW)
    obj["entries"] += [obj["entries"][-1]]
    with pytest.raises(CompatibilityError):
        parse(json.dumps(obj).encode())
    obj = json.loads(RAW)
    obj["entries"].reverse()
    with pytest.raises(CompatibilityError):
        parse(json.dumps(obj).encode())


@pytest.mark.parametrize("mut", [
    {"status": "ready"},
    {"status": "SUPPORTED"},
    {"evidence_ref": "unexpected"},
    {"feature": "../training"},
    {"backend": "CUDA"},
    {"profile": "macos-arm64"},
])
def test_untrusted_entry_fields_rejected(mut: dict[str, object]) -> None:
    obj = json.loads(RAW)
    obj["entries"][0].update(mut)
    with pytest.raises(CompatibilityError):
        parse(json.dumps(obj).encode())


def test_frozen_object_mutation_cannot_weaken_readback() -> None:
    matrix = parse()
    object.__setattr__(matrix.entries[0], "status", "READY")
    with pytest.raises(CompatibilityError):
        matrix.lookup("linux-aarch64", "cpu", "fixture_runtime",
                      expected_lock_sha256=LOCK_SHA)


def test_parser_denies_oversize_raw_and_bool_schema() -> None:
    with pytest.raises(CompatibilityError):
        parse(b"x" * 65537)
    obj = json.loads(RAW)
    obj["schema_version"] = True
    with pytest.raises(CompatibilityError):
        parse(json.dumps(obj).encode())
