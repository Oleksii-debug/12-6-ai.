"""Verify the physical corpus CLI and its publication boundary."""
import hashlib
import json
import sys
from pathlib import Path

import pytest

from tools import plan2_s16_corpus_admission_v1 as admission

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def replay():
    survivors = []
    for modality, family, source, count in (
        ("uk", "uk-books", "uk-books", 23),
        ("uk", "uk-archives", "uk-archives", 22),
        ("en", "en-science", "arxiv_abstracts", 18),
        ("en", "en-books", "en-books", 17),
        ("code", "code-project-one", "code-project-one", 10),
        ("code", "code-project-two", "code-project-two", 10),
    ):
        for index in range(count):
            record_id = f"{family}:{index}"
            survivors.append({
                "record_id": record_id, "source_id": source, "source_family": family,
                "modality": modality, "text": f"{record_id:100}",
            })
    receipt = {
        "manifest_sha256": "1" * 64,
        "decontamination": {
            "clean": True,
            "candidate_dedup_projection": {"surviving_records": len(survivors)},
            "data232_report_sha256": "2" * 64,
        },
        "source_evidence": {},
        "canonical_privacy_policy": "test",
    }
    holdouts = {
        "all_source_families_disjoint_across_roles": True,
        "evaluation_use_rights_separately_admitted": True,
        "real_final_test_outcomes_read": False,
        "physical_training_candidate_sha256": "3" * 64,
        "eval303_selection_identity_sha256": "4" * 64,
        "eval233_final_set_identity_sha256": "5" * 64,
        "physical_source_rights_sha256": "6" * 64,
        "role_language_records": {"validation": {"uk": 2, "en": 2}, "test": {"uk": 2, "en": 2}},
    }
    return receipt, holdouts, survivors


def _run_cli(monkeypatch, destination, replay):
    monkeypatch.setattr(admission.candidate, "replay", lambda *_: replay)
    terms = {
        family: {"license": "test", "notice": "fixture"}
        for family in (
            "uk-books", "uk-archives", "en-science", "en-books",
            "code-project-one", "code-project-two",
        )
    }
    monkeypatch.setattr(admission, "_source_terms", lambda *_: terms)
    monkeypatch.setattr(sys, "argv", [
        "plan2-s16", "--root", str(ROOT), "--out-dir", str(destination),
        "--selection-reconstructed", "uk.jsonl", "uk-authority", "en.jsonl", "en-authority",
        "--franko", "franko", "--arxiv", "arxiv", "--languk", "languk",
        "--code-report", "code-report", "--code-checkouts", "checkouts",
    ])
    return admission.main()


def test_cli_publishes_the_same_verified_train_payload_after_restart_and_rebuild(
    tmp_path, monkeypatch, replay,
):
    first = tmp_path / "first"
    rebuilt = tmp_path / "rebuilt"
    assert _run_cli(monkeypatch, first, replay) == 0
    manifest_bytes = (first / admission.MANIFEST).read_bytes()
    payload = (first / admission.DATA).read_bytes()
    manifest = json.loads(manifest_bytes)
    rows = [json.loads(row) for row in payload.splitlines()]

    assert manifest["payload_sha256"] == hashlib.sha256(payload).hexdigest()
    assert manifest["records"] == len(rows) == 100
    assert manifest["s9_mixture"]["selected_bytes_by_stratum"] == {
        "ua": 4500, "en": 3500, "code": 2000,
    }
    assert manifest["heldout_payloads_exported"] is False
    assert manifest["production_release_authorized"] is False
    assert set(manifest["selected_source_terms"]) == {row["source_family"] for row in rows}
    expected_documents = {row["text"] for row in replay[2]}
    assert {row["text"] for row in rows} == expected_documents

    assert _run_cli(monkeypatch, first, replay) == 0
    reversed_replay = (*replay[:2], list(reversed(replay[2])))
    assert _run_cli(monkeypatch, rebuilt, reversed_replay) == 0
    assert (first / admission.MANIFEST).read_bytes() == (
        rebuilt / admission.MANIFEST
    ).read_bytes() == manifest_bytes
    assert (first / admission.DATA).read_bytes() == (rebuilt / admission.DATA).read_bytes() == payload


@pytest.mark.parametrize(
    "fault", ["only_one_uk_family", "missing_uk_validation", "reserved_overlap", "unmapped_grant"],
)
def test_cli_rejects_bad_composition_before_creating_the_output_directory(
    fault, tmp_path, monkeypatch, replay,
):
    receipt, holdouts, survivors = replay
    if fault == "only_one_uk_family":
        for row in survivors:
            if row["modality"] == "uk":
                row["source_family"] = "uk-books"
    elif fault == "missing_uk_validation":
        holdouts["role_language_records"]["validation"].pop("uk")
    elif fault == "unmapped_grant":
        for row in survivors:
            if row["source_family"] == "code-project-two":
                row["source_family"] = "unmapped-code-project"
    else:
        receipt["decontamination"]["clean"] = False

    destination = tmp_path / "denied"
    assert _run_cli(monkeypatch, destination, replay) == 2
    assert not destination.exists()


def test_cli_cannot_create_payload_under_a_changed_existing_manifest(tmp_path, monkeypatch, replay):
    stale = tmp_path / admission.MANIFEST
    stale.write_bytes(b'{"production_release_authorized":true}\n')
    assert _run_cli(monkeypatch, tmp_path, replay) == 2
    assert stale.read_bytes() == b'{"production_release_authorized":true}\n'
    assert not (tmp_path / admission.DATA).exists()
