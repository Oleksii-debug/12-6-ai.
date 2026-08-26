from __future__ import annotations

from pathlib import Path

import pytest

from twelve_six.data.real_corpus_holdout import (
    RealCorpusHoldoutError,
    build_exclusion_proof,
    build_fixed_tokenizer_no_fit_proof,
    build_immutable_holdout,
    hash_json,
    load_heldout_rows,
)

SHA = "1" * 64


def _records():
    return [
        {
            "record_id": "ua-1",
            "modality": "ua",
            "source_id": "rada",
            "source_family": "government-law",
            "source_version": "v1",
            "source_snapshot_sha256": "2" * 64,
            "text": "Україна має закон. Україна має право.",
            "source_kind": "EXTERNAL_REAL",
            "evaluation_use_authority_ref": "D03:rights:ua",
            "provenance_ref": "source://rada/v1",
        },
        {
            "record_id": "en-1",
            "modality": "en",
            "source_id": "manual",
            "source_family": "technical-prose",
            "source_version": "v2",
            "source_snapshot_sha256": "3" * 64,
            "text": "The manual defines stable editorial rules for text.",
            "source_kind": "EXTERNAL_REAL",
            "evaluation_use_authority_ref": "D03:rights:en",
            "provenance_ref": "source://manual/v2",
        },
        {
            "record_id": "code-1",
            "modality": "code",
            "source_id": "code-source",
            "source_family": "python-stdlib",
            "source_version": "v3",
            "source_snapshot_sha256": "4" * 64,
            "text": "def add(a, b):\n    return a + b\n",
            "source_kind": "EXTERNAL_REAL",
            "evaluation_use_authority_ref": "D03:rights:code",
            "provenance_ref": "source://code/v3",
        },
    ]


def _build(path: Path):
    return build_immutable_holdout(
        _records(),
        path,
        suite_name="eval131-test",
        evaluation_corpus_identity_sha256=SHA,
        benchmark_registry_sha256="5" * 64,
        decontamination_reference_bundle_sha256="6" * 64,
        decontamination_report_sha256="7" * 64,
    )


def test_immutable_three_modality_holdout_and_exclusion_proofs(tmp_path: Path):
    manifest = _build(tmp_path / "heldout")
    same = _build(tmp_path / "heldout")
    assert same == manifest
    checked, rows = load_heldout_rows(tmp_path / "heldout")
    assert checked["heldout_identity_sha256"] == manifest["heldout_identity_sha256"]
    assert {row["modality"] for row in rows} == {"ua", "en", "code"}
    assert set(manifest["source_families"]) == {
        "government-law",
        "technical-prose",
        "python-stdlib",
    }
    training = build_exclusion_proof(
        [{"record_id": "train-1", "text": "different training material"}],
        manifest,
        purpose="MODEL_TRAINING",
        candidate_identity_sha256="8" * 64,
    )
    tokenizer = build_fixed_tokenizer_no_fit_proof(
        manifest, tokenizer_identity_sha256="9" * 64
    )
    assert training["status"] == "PASS_EXCLUDED_BEFORE_USE"
    assert tokenizer["status"] == "NOT_APPLICABLE_FIXED_TOKENIZER_NO_FIT_CORPUS"
    unsigned = {k: v for k, v in training.items() if k != "proof_sha256"}
    assert training["proof_sha256"] == hash_json(unsigned)


def test_project_authored_or_overlapping_material_fails_closed(tmp_path: Path):
    records = _records()
    records[0] = dict(records[0], source_kind="PROJECT_AUTHORED")
    with pytest.raises(RealCorpusHoldoutError, match="EXTERNAL_REAL"):
        build_immutable_holdout(
            records,
            tmp_path / "bad",
            suite_name="bad",
            evaluation_corpus_identity_sha256=SHA,
            benchmark_registry_sha256="5" * 64,
            decontamination_reference_bundle_sha256="6" * 64,
            decontamination_report_sha256="7" * 64,
        )

    manifest = _build(tmp_path / "heldout")
    with pytest.raises(RealCorpusHoldoutError, match="overlaps"):
        build_exclusion_proof(
            [{"record_id": "train-1", "text": _records()[1]["text"]}],
            manifest,
            purpose="MODEL_TRAINING",
            candidate_identity_sha256="8" * 64,
        )


def test_existing_immutable_bytes_cannot_be_replaced(tmp_path: Path):
    directory = tmp_path / "heldout"
    _build(directory)
    path = directory / "ua.jsonl"
    path.write_text(path.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
    with pytest.raises(RealCorpusHoldoutError, match="bytes differ"):
        _build(directory)
