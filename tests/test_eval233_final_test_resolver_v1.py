from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from twelve_six.data.eval233_final_test_resolver_v1 import (
    AUTHORITY_PATH,
    DATA232_FINAL_TEST_ID,
    EVAL233_FINAL_SET_ID,
    EXPECTED_DOCUMENTS,
    EXPECTED_MODALITY_COUNTS,
    SEED_PATH,
    FinalTestResolverError,
    resolve_eval233_final_test,
)

ROOT = Path(__file__).resolve().parents[1]


def test_exact_terminal_final_test_resolves_to_16_bound_matcher_rows():
    rows, reserved_set, evidence = resolve_eval233_final_test(ROOT)
    assert len(rows) == EXPECTED_DOCUMENTS
    assert {row["modality"] for row in rows} == {"ua", "en"}
    assert sum(row["modality"] == "ua" for row in rows) == 8
    assert sum(row["modality"] == "en" for row in rows) == 8
    assert reserved_set["identity_sha256"] == DATA232_FINAL_TEST_ID
    assert reserved_set["source_membership_identity_sha256"] == EVAL233_FINAL_SET_ID
    assert reserved_set["role"] == "final_test"
    assert len(reserved_set["members"]) == EXPECTED_DOCUMENTS
    assert evidence["documents"] == EXPECTED_DOCUMENTS
    assert evidence["modality_documents"] == EXPECTED_MODALITY_COUNTS
    assert evidence["final_test_payload_accessed_for_decontamination"] is True
    assert evidence["final_test_outcomes_read"] is False
    assert evidence["authorized_training_exposure"] == 0


def test_resolver_durable_outputs_do_not_contain_final_test_text_or_outcome_values():
    rows, reserved_set, evidence = resolve_eval233_final_test(ROOT)
    durable = json.dumps(
        {"reserved_set": reserved_set, "evidence": evidence},
        ensure_ascii=False,
        sort_keys=True,
    )
    for row in rows:
        assert row["text"] not in durable
    assert '"text"' not in durable
    assert evidence["final_test_outcomes_read"] is False
    assert evidence["selection_or_hyperparameter_use"] is False
    expected_member_keys = {
        "record_id",
        "source_id",
        "source_family",
        "modality",
        "content_sha256",
        "utf8_bytes",
        "training_prohibited",
        "outcomes_included",
    }
    for member in reserved_set["members"]:
        assert set(member) == expected_member_keys
        assert member["outcomes_included"] is False
        assert member["training_prohibited"] is True


def _copy_inputs(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    seed_target = root / SEED_PATH
    authority_target = root / AUTHORITY_PATH
    seed_target.parent.mkdir(parents=True, exist_ok=True)
    authority_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / SEED_PATH, seed_target)
    shutil.copyfile(ROOT / AUTHORITY_PATH, authority_target)
    return root


def test_seed_byte_mutation_fails_before_decompression(tmp_path: Path):
    root = _copy_inputs(tmp_path)
    path = root / SEED_PATH
    raw = bytearray(path.read_bytes())
    raw[-1] ^= 1
    path.write_bytes(raw)
    with pytest.raises(FinalTestResolverError, match="seed Git blob identity drift"):
        resolve_eval233_final_test(root)


def test_source_authority_mutation_fails_exact_blob_binding(tmp_path: Path):
    root = _copy_inputs(tmp_path)
    path = root / AUTHORITY_PATH
    authority = json.loads(path.read_text(encoding="utf-8"))
    authority["unsupported_claims"]["production_readiness"] = True
    path.write_text(json.dumps(authority, sort_keys=True), encoding="utf-8")
    with pytest.raises(
        FinalTestResolverError,
        match="source-authority Git blob identity drift",
    ):
        resolve_eval233_final_test(root)


def test_resolver_evidence_identity_is_deterministic_and_self_consistent():
    _, _, first = resolve_eval233_final_test(ROOT)
    _, _, second = resolve_eval233_final_test(ROOT)
    assert first == second
    claimed = first["resolver_identity_sha256"]
    body = dict(first)
    body.pop("resolver_identity_sha256")
    raw = json.dumps(
        body,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    assert claimed == hashlib.sha256(raw).hexdigest()
