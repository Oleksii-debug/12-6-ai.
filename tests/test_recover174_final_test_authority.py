from __future__ import annotations

import gzip
import json
import shutil
from pathlib import Path

import pytest

from twelve_six.data.recover174_final_test_authority import (
    AUTHORITY_ID,
    CANONICAL_FINAL_TEST_SET_IDENTITY_SHA256,
    CURRENT_AUTHORITY_PATH,
    RECOVER174_AUTHORITY_PATH,
    RECOVER174_SEED_PATH,
    Recover174FinalTestAuthorityError,
    SOURCE_MEMBERSHIP_IDENTITY_SHA256,
    load_final_test_reserved_set,
    verify_current_final_test_authority,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA = "a" * 40


def _copy_authority_inputs(tmp_path: Path) -> None:
    for relative in (RECOVER174_SEED_PATH, RECOVER174_AUTHORITY_PATH, CURRENT_AUTHORITY_PATH):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO_ROOT / relative, target)


def test_exact_recover174_final_test_exposes_current_reserved_contract() -> None:
    payloads, reserved = load_final_test_reserved_set(REPO_ROOT, source_sha=SOURCE_SHA)

    assert len(payloads) == 16
    assert {row["modality"] for row in payloads} == {"ua", "en"}
    assert sum(row["modality"] == "ua" for row in payloads) == 8
    assert sum(row["modality"] == "en" for row in payloads) == 8

    assert reserved["authority_id"] == AUTHORITY_ID
    assert reserved["identity_sha256"] == CANONICAL_FINAL_TEST_SET_IDENTITY_SHA256
    assert (
        reserved["source_membership_identity_sha256"]
        == SOURCE_MEMBERSHIP_IDENTITY_SHA256
    )
    assert reserved["role"] == "final_test"
    assert reserved["source_sha"] == SOURCE_SHA
    assert len(reserved["members"]) == 16
    assert all(member["training_prohibited"] is True for member in reserved["members"])
    assert all(member["outcomes_included"] is False for member in reserved["members"])
    assert all("text" not in member for member in reserved["members"])

    payload_by_id = {row["record_id"]: row for row in payloads}
    member_by_id = {row["record_id"]: row for row in reserved["members"]}
    assert set(payload_by_id) == set(member_by_id)


def test_verification_is_text_free_and_independently_pinned() -> None:
    summary = verify_current_final_test_authority(
        REPO_ROOT,
        source_sha=SOURCE_SHA,
        expected_identity_sha256=CANONICAL_FINAL_TEST_SET_IDENTITY_SHA256,
        expected_membership_identity_sha256=SOURCE_MEMBERSHIP_IDENTITY_SHA256,
    )
    assert summary["documents"] == 16
    assert summary["modality_documents"] == {"ua": 8, "en": 8}
    assert summary["raw_text_persisted"] is False
    assert summary["outcomes_included"] is False
    assert summary["training_prohibited"] is True
    assert summary["local_free_only"] is True

    with pytest.raises(
        Recover174FinalTestAuthorityError,
        match="independently expected final-test identity mismatch",
    ):
        verify_current_final_test_authority(
            REPO_ROOT,
            source_sha=SOURCE_SHA,
            expected_identity_sha256="0" * 64,
        )


def test_exact_seed_tamper_fails_closed(tmp_path: Path) -> None:
    _copy_authority_inputs(tmp_path)
    seed_path = tmp_path / RECOVER174_SEED_PATH
    rows = gzip.decompress(seed_path.read_bytes()).splitlines()
    first = json.loads(rows[0])
    first["text"] += " tampered"
    rows[0] = json.dumps(first, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    seed_path.write_bytes(gzip.compress(b"\n".join(rows) + b"\n", mtime=0))

    with pytest.raises(
        Recover174FinalTestAuthorityError,
        match="seed Git blob mismatch",
    ):
        load_final_test_reserved_set(tmp_path, source_sha=SOURCE_SHA)


def test_durable_membership_tamper_fails_closed(tmp_path: Path) -> None:
    _copy_authority_inputs(tmp_path)
    evidence_path = tmp_path / CURRENT_AUTHORITY_PATH
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence["membership"]["members"][0]["content_sha256"] = "0" * 64
    evidence_path.write_text(
        json.dumps(evidence, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        Recover174FinalTestAuthorityError,
        match="authority evidence self-hash mismatch",
    ):
        load_final_test_reserved_set(tmp_path, source_sha=SOURCE_SHA)
