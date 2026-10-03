from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from twelve_six.data.permissive_repo_source_authority import (
    SourceAuthorityError,
    authority_identity,
    load_and_validate_source_authority,
    validate_source_authority,
)

ROOT = Path(__file__).resolve().parents[1]
AUTHORITY_PATH = ROOT / "configs" / "data" / "scipy_v118_source_authority_v1.json"


def _document() -> dict:
    return json.loads(AUTHORITY_PATH.read_text(encoding="utf-8"))


def _resign(document: dict) -> dict:
    document["authority_sha256"] = authority_identity(document)
    return document


def test_authority_is_deterministic_and_candidate_only() -> None:
    summary = load_and_validate_source_authority(AUTHORITY_PATH)
    assert summary == {
        "authority_id": "scipy-v1.18.0-bounded-first-party-v1",
        "authority_sha256": "a9ddc66c826f32299cb9e69aa64d3e7e7526869391e3e2ab3162abf43cf886a9",
        "files": 2,
        "candidate_raw_bytes": 78307,
        "canonical_credit_bytes": 0,
        "ready_for_corpus_credit": False,
    }
    assert authority_identity(_document()) == summary["authority_sha256"]


def test_tampered_manifest_identity_fails() -> None:
    document = _document()
    document["capacity"]["candidate_raw_bytes"] += 1
    with pytest.raises(SourceAuthorityError, match="authority_sha256 mismatch"):
        validate_source_authority(document)


def test_eval_permission_cannot_be_enabled_even_when_resigned() -> None:
    document = _resign(copy.deepcopy(_document()))
    document["purpose"]["evaluation_allowed"] = True
    _resign(document)
    with pytest.raises(SourceAuthorityError, match="purpose boundary drift"):
        validate_source_authority(document)


def test_canonical_credit_cannot_be_claimed_pre_materialization() -> None:
    document = copy.deepcopy(_document())
    document["capacity"]["canonical_credit_bytes"] = 1
    _resign(document)
    with pytest.raises(SourceAuthorityError, match="canonical credit must remain zero"):
        validate_source_authority(document)


def test_upstream_pin_drift_fails_closed() -> None:
    document = copy.deepcopy(_document())
    document["upstream"]["commit_sha"] = "0" * 40
    _resign(document)
    with pytest.raises(SourceAuthorityError, match="upstream pin drift"):
        validate_source_authority(document)


def test_third_party_or_test_path_is_rejected() -> None:
    document = copy.deepcopy(_document())
    document["allowlist"][0]["path"] = "scipy/optimize/tests/example.py"
    commit = document["upstream"]["commit_sha"]
    document["allowlist"][0]["raw_url"] = (
        f"https://raw.githubusercontent.com/scipy/scipy/{commit}/scipy/optimize/tests/example.py"
    )
    _resign(document)
    with pytest.raises(SourceAuthorityError, match="forbidden provenance path"):
        validate_source_authority(document)


def test_duplicate_allowlist_path_is_rejected() -> None:
    document = copy.deepcopy(_document())
    document["allowlist"][1] = copy.deepcopy(document["allowlist"][0])
    document["capacity"]["candidate_raw_bytes"] = 2 * document["allowlist"][0]["raw_bytes"]
    _resign(document)
    with pytest.raises(SourceAuthorityError, match="duplicate allowlist path"):
        validate_source_authority(document)


def test_candidate_byte_arithmetic_is_bound() -> None:
    document = copy.deepcopy(_document())
    document["capacity"]["candidate_raw_bytes"] += 7
    _resign(document)
    with pytest.raises(SourceAuthorityError, match="candidate byte arithmetic drift"):
        validate_source_authority(document)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("schema_version",), True, "schema_version"),
        (("license", "whole_repository_credit_forbidden"), 1, "license boundary"),
        (("purpose", "training_allowed"), 1, "purpose boundary"),
        (("purpose", "evaluation_allowed"), 0, "purpose boundary"),
        (("gates", "requires_materialization"), 1, "downstream gates"),
        (("gates", "requires_global_cross_source_dedup"), 1, "downstream gates"),
        (("capacity", "candidate_raw_bytes"), 78307.0, "candidate byte arithmetic"),
        (("capacity", "canonical_credit_bytes"), False, "canonical credit"),
        (("allowlist", 0, "git_blob_sha1"), "a" * 40, "unapproved SciPy"),
        (("allowlist", 0, "raw_bytes"), 25258, "unapproved SciPy"),
    ],
)
def test_resigned_source_authority_rejects_aliases_and_forged_pins(
    path: tuple[str | int, ...], value: object, message: str,
) -> None:
    document = copy.deepcopy(_document())
    target = document
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    if path == ("allowlist", 0, "raw_bytes"):
        document["capacity"]["candidate_raw_bytes"] += 1
    _resign(document)
    with pytest.raises(SourceAuthorityError, match=message):
        validate_source_authority(document)


def test_resigned_source_authority_rejects_new_uninspected_code_path() -> None:
    document = copy.deepcopy(_document())
    entry = document["allowlist"][0]
    entry["path"] = "scipy/optimize/_uninspected.py"
    commit = document["upstream"]["commit_sha"]
    entry["raw_url"] = (
        f"https://raw.githubusercontent.com/scipy/scipy/{commit}/{entry['path']}"
    )
    _resign(document)
    with pytest.raises(SourceAuthorityError, match="unapproved SciPy"):
        validate_source_authority(document)


def test_resigned_source_authority_rejects_missing_pinned_code_file() -> None:
    document = copy.deepcopy(_document())
    document["allowlist"].pop()
    document["capacity"]["candidate_raw_bytes"] = document["allowlist"][0]["raw_bytes"]
    _resign(document)
    with pytest.raises(SourceAuthorityError, match="two pinned first-party files"):
        validate_source_authority(document)


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (b'{"a":1,"a":2}', "duplicate authority JSON key"),
        (b'{"a":{"b":1,"b":2}}', "duplicate authority JSON key"),
        (b'{"a":NaN}', "non-finite"),
        (b'{"a":Infinity}', "non-finite"),
        (b'{"a":1e9999}', "not finite"),
        (b'{"a":1e-9999}', "underflowed"),
        (b'{"a":-1e-9999}', "underflowed"),
        (b'{"a":' + b"9" * 65 + b"}", "digit limit"),
        (b'{"a":' + b"[" * 65 + b"0" + b"]" * 65 + b"}", "structure limit"),
        (
            b'{"a":' + b"[" * 10000 + b"0" + b"]" * 10000 + b"}",
            "nesting limit",
        ),
        (
            b'{"a":[' + b",".join([b"0"] * 10010) + b"]}",
            "structure limit",
        ),
        (br'{"\ud800":1}', "surrogates not allowed"),
        (br'{"a":"\ud800"}', "surrogates not allowed"),
        (b'{"a":"\xff"}', "decode"),
        (b" " * 1_048_577, "byte limit"),
        (b"[]", "authority root must be an object"),
    ],
)
def test_authority_loader_rejects_unsafe_external_json(
    tmp_path: Path, raw: bytes, message: str,
) -> None:
    source = tmp_path / "недовірений маніфест із пробілами.json"
    source.write_bytes(raw)
    with pytest.raises(SourceAuthorityError, match=message):
        load_and_validate_source_authority(source)


def test_invalid_real_cli_manifest_never_issues_source_credit(tmp_path: Path) -> None:
    source = tmp_path / "дублікати ключів.json"
    source.write_bytes(b'{"authority_id":"safe","authority_id":"forged"}')
    cli = ROOT / "tools" / "run_scipy_source_authority.py"
    run = subprocess.run(
        [sys.executable, str(cli), "--authority", str(source)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert run.returncode == 2
    assert run.stderr == ""
    response = json.loads(run.stdout)
    assert response["status"] == "BLOCKED_INVALID_AUTHORITY"
    assert response["canonical_credit_bytes"] == 0
    assert response["ready_for_corpus_credit"] is False


@pytest.mark.parametrize(
    "mutation",
    ["extra-root", "extra-capacity", "allowlist-order", "extra-gate-free-text"],
)
def test_resigned_canonical_authority_rejects_extra_or_reordered_metadata(
    mutation: str,
) -> None:
    document = copy.deepcopy(_document())
    if mutation == "extra-root":
        document["unreviewed_extension"] = "must-not-be-authority"
    elif mutation == "extra-capacity":
        document["capacity"]["unreviewed_bytes"] = 78307
    elif mutation == "allowlist-order":
        document["allowlist"].reverse()
    else:
        document["gates"]["requires_global_cross_source_dedup"] = "yes"
        _resign(document)
        with pytest.raises(SourceAuthorityError, match="downstream gates"):
            validate_source_authority(document)
        return
    _resign(document)
    with pytest.raises(SourceAuthorityError, match="canonical SciPy authority identity drift"):
        validate_source_authority(document)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("allowlist", 1, "git_blob_sha1"), "0" * 40, "unapproved SciPy"),
        (
            ("allowlist", 1, "raw_url"),
            "https://raw.githubusercontent.com/scipy/scipy/main/scipy/optimize/_minimize.py",
            "raw URL is not exact-commit pinned",
        ),
        (("allowlist", 0, "raw_bytes"), 25257.0, "invalid raw byte count"),
        (("allowlist", 0, "path"), "scipy/optimize/../secret.py", "unsafe allowlist path"),
        (("allowlist", 0, "path"), "/scipy/optimize/_constraints.py", "unsafe allowlist path"),
        (
            ("allowlist", 0, "path"),
            "scipy\\optimize\\_constraints.py",
            "outside bounded scipy/optimize scope",
        ),
    ],
)
def test_resigned_source_authority_rejects_secondary_blob_and_path_escapes(
    path: tuple[str | int, ...], value: object, message: str,
) -> None:
    document = copy.deepcopy(_document())
    target = document
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    _resign(document)
    with pytest.raises(SourceAuthorityError, match=message):
        validate_source_authority(document)
