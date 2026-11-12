"""Materialize Plan 2 validation and test manifests without opening heldout records."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from tools import plan2_public_domain_books_v1 as books
from tools import plan2_s16_corpus_admission_v1 as corpus
from tools import plan2_s16_independent_eval_binding_v1 as selection
from twelve_six.data import eval233_final_test_resolver_v1 as final_test
from twelve_six.data import eval303_selection_payload_resolver_v1 as eval303

SCHEMA = "12-6.plan2-s17-reserved-role-manifest.v1"
TEST_MEMBERS = "evidence/plan2_s17/eval233_reserved_members_v1.json"
TEST_MEMBERS_SHA256 = "94b8e925206a0f2ed1e3bb33165e0073e0cd70b961b1846fd31d38f958e80385"
TEST_PROJECTION_SHA256 = "2eb91ade9a4a655f0c29fe4cf80fe796bad0bcecef2de57fdd15ae6b00e5374f"


def _read_lf_git_text(root: Path, relative: str) -> bytes:
    """Undo Windows checkout line endings before checking pinned Git text."""
    return books.read_checked(root, relative).replace(b"\r\n", b"\n")


def _selection_members(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw = _read_lf_git_text(root, selection.SELECTION_MEMBERS)
    config_raw = _read_lf_git_text(root, selection.SELECTION_CONFIG)
    selection.need(
        books.git_blob(config_raw) == selection.SELECTION_GIT_BLOB
        and books.git_blob(raw) == selection.SELECTION_MEMBER_GIT_BLOB
        and books.sha(raw) == eval303.EVAL303_MEMBERSHIP_SHA256,
        "EVAL-303 membership or authority changed",
    )
    config = json.loads(config_raw)
    rows = selection.inspect_selection_rows(raw, config)
    return [
        {
            "record_id_sha256": books.sha(row["record_id"].encode()),
            "source_id_sha256": books.sha(row["source_id"].encode()),
            "source_family_sha256": books.sha(row["source_family"].encode()),
            "language": row["language"],
            "content_sha256": row["content_sha256"],
            "utf8_bytes": row["utf8_bytes"],
        }
        for row in rows
    ], {
        "id": "eval303-selection-validation",
        "head_sha": eval303.EVAL303_TERMINAL_SOURCE_SHA,
        "identity_sha256": eval303.EVAL303_SELECTION_ID,
        "membership_sha256": eval303.EVAL303_MEMBERSHIP_SHA256,
        "membership_git_blob_sha1": selection.SELECTION_MEMBER_GIT_BLOB,
        "authority_git_blob_sha1": selection.SELECTION_GIT_BLOB,
    }


def _final_test_members(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    # This projection was produced once by the historical DATA-232 resolver.
    # S17 verifies custody of the sealed seed without decompressing it.
    raw = _read_lf_git_text(root, TEST_MEMBERS)
    selection.need(books.sha(raw) == TEST_MEMBERS_SHA256, "EVAL-233 member projection changed")
    proof = json.loads(raw)
    expected = {
        "schema_version": "12-6.plan2-s17-eval233-reserved-members.v1",
        "source_head_sha": final_test.EVAL233_HEAD,
        "source_final_set_identity_sha256": final_test.EVAL233_FINAL_SET_ID,
        "source_authority_identity_sha256": final_test.SOURCE_AUTHORITY_ID,
        "data232_final_test_identity_sha256": final_test.DATA232_FINAL_TEST_ID,
        "source_authority_git_blob_sha1": final_test.AUTHORITY_GIT_BLOB_SHA1,
        "seed_git_blob_sha1": final_test.SEED_GIT_BLOB_SHA1,
        "seed_sha256": final_test.SEED_SHA256,
        "member_projection_sha256": TEST_PROJECTION_SHA256,
    }
    selection.need(
        {key: value for key, value in proof.items() if key != "members"} == expected
        and books.sha(books.canonical(proof["members"]).removesuffix(b"\n")) == TEST_PROJECTION_SHA256,
        "EVAL-233 member projection does not match the historical authority",
    )
    seed = books.read_checked(root, final_test.SEED_PATH.as_posix())
    authority = _read_lf_git_text(root, final_test.AUTHORITY_PATH.as_posix())
    selection.need(
        books.git_blob(seed) == final_test.SEED_GIT_BLOB_SHA1
        and books.sha(seed) == final_test.SEED_SHA256
        and books.git_blob(authority) == final_test.AUTHORITY_GIT_BLOB_SHA1,
        "EVAL-233 sealed seed or authority changed",
    )
    return [
        {
            "record_id_sha256": row["record_id_sha256"],
            "source_id_sha256": row["source_id_sha256"],
            "source_family_sha256": row["source_family_sha256"],
            "language": {"ua": "uk", "en": "en"}[row["modality"]],
            "content_sha256": row["content_sha256"],
            "utf8_bytes": row["utf8_bytes"],
        }
        for row in proof["members"]
    ], {
        "id": "eval233-final-test",
        "head_sha": final_test.EVAL233_HEAD,
        "identity_sha256": final_test.EVAL233_FINAL_SET_ID,
        "data232_identity_sha256": final_test.DATA232_FINAL_TEST_ID,
        "member_projection_file_sha256": TEST_MEMBERS_SHA256,
        "historical_member_projection_sha256": TEST_PROJECTION_SHA256,
        "source_authority_identity_sha256": final_test.SOURCE_AUTHORITY_ID,
        "source_authority_git_blob_sha1": final_test.AUTHORITY_GIT_BLOB_SHA1,
        "seed_git_blob_sha1": final_test.SEED_GIT_BLOB_SHA1,
        "seed_sha256": final_test.SEED_SHA256,
    }


def reserved_role_manifests(root: Path, admission: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Bind the admitted S16 corpus to both independently reserved sets."""
    admission_id = admission["manifest_sha256"]
    selection.need(
        admission["schema"] == corpus.SCHEMA
        and admission["physical_s3_s9_admission_passed"] is True
        and admission["heldout_payloads_exported"] is False
        and books.sha(books.canonical({
            key: value for key, value in admission.items() if key != "manifest_sha256"
        })) == admission_id
        and admission["s8_selection_sha256"] == eval303.EVAL303_SELECTION_ID
        and admission["s8_final_test_sha256"] == final_test.EVAL233_FINAL_SET_ID,
        "S16 admission or its reserved-set identities changed",
    )
    sources = {"validation": _selection_members(root), "test": _final_test_members(root)}
    manifests = {}
    seen_ids: set[str] = set()
    seen_content: set[str] = set()
    seen_families = {
        books.sha(family.encode()) for family in admission["s9_mixture"]["selected_bytes_by_family"]
    }
    for role, (rows, authority) in sources.items():
        members = sorted(rows, key=lambda row: row["record_id_sha256"])
        counts = dict(sorted(Counter(row["language"] for row in members).items()))
        expected_counts = {"en": 2 if role == "validation" else 8, "uk": 8}
        ids = {row["record_id_sha256"] for row in members}
        content = {row["content_sha256"] for row in members}
        families = {row["source_family_sha256"] for row in members}
        selection.need(
            counts == expected_counts == admission["role_counts"][role]
            and len(ids) == len(content) == len(members)
            and ids.isdisjoint(seen_ids)
            and content.isdisjoint(seen_content)
            and families.isdisjoint(seen_families),
            "reserved role membership, language counts or split isolation changed",
        )
        seen_ids.update(ids)
        seen_content.update(content)
        seen_families.update(families)
        core = {
            "schema_version": SCHEMA,
            "role": role,
            "purpose": "selection-validation" if role == "validation" else "final-test",
            "s16_corpus_manifest_sha256": admission_id,
            "s16_decontamination_report_sha256": admission["s8_data232_report_sha256"],
            "authority": authority,
            "records": len(members),
            "language_counts": counts,
            "members": members,
            "member_projection_sha256": books.sha(books.canonical(members)),
            "training_eligible": False,
            "tokenizer_fit_eligible": False,
            "selection_eligible": role == "validation",
            "final_test_outcomes_read": False,
            "payload_exported": False,
        }
        manifests[role] = {**core, "manifest_sha256": books.sha(books.canonical(core))}
    return manifests
