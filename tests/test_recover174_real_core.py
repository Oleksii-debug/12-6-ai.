from __future__ import annotations

import gzip
import json
from pathlib import Path

from twelve_six.recover174_real_core import (
    ADMITTED_MODALITIES,
    BLOCKED_MODALITY,
    EXPECTED_AUTHORITY_ID,
    EXPECTED_SEED_SHA256,
    _authority,
    build_partial_holdout,
)
from twelve_six.data.real_corpus_holdout import sha256_file


def test_recover174_frozen_seed_and_authority_are_exact() -> None:
    repo = Path(".")
    assert sha256_file(repo / "data/evaluation/recover174_real_holdout_seed.jsonl.gz") == (
        EXPECTED_SEED_SHA256
    )
    authority = _authority(repo)
    assert authority["authority_identity_sha256"] == EXPECTED_AUTHORITY_ID
    assert authority["admitted_modalities"] == list(ADMITTED_MODALITIES)
    assert authority["blocked_modalities"][BLOCKED_MODALITY]["status"] == (
        "BLOCKED_NO_EVALUATION_USE_AUTHORITY"
    )


def test_recover174_partial_holdout_is_immutable_ua_en_only(tmp_path: Path) -> None:
    manifest, rows = build_partial_holdout(Path("."), tmp_path / "heldout")
    assert manifest["modalities"] == list(ADMITTED_MODALITIES)
    assert manifest["truth_boundary"]["full_ua_en_code_authority"] is False
    assert manifest["blocked_modalities"][BLOCKED_MODALITY]["status"] == (
        "BLOCKED_NO_EVALUATION_USE_AUTHORITY"
    )
    assert len(rows) == 16
    assert {row["modality"] for row in rows} == set(ADMITTED_MODALITIES)
    assert BLOCKED_MODALITY not in {row["modality"] for row in rows}
    assert all(row["source_kind"] == "EXTERNAL_REAL" for row in rows)
    assert all(row["evaluation_use_authority_ref"] for row in rows)

    before = {
        path.name: path.read_bytes()
        for path in (tmp_path / "heldout").iterdir()
    }
    manifest2, rows2 = build_partial_holdout(Path("."), tmp_path / "heldout")
    after = {
        path.name: path.read_bytes()
        for path in (tmp_path / "heldout").iterdir()
    }
    assert manifest2 == manifest
    assert rows2 == rows
    assert after == before


def test_recover174_seed_contains_only_authorized_snapshots() -> None:
    authority = _authority(Path("."))
    with gzip.open(
        Path("data/evaluation/recover174_real_holdout_seed.jsonl.gz"),
        "rt", encoding="utf-8"
    ) as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    assert len(rows) == 16
    for row in rows:
        source = authority["sources"][row["source_id"]]
        assert source["evaluation_status"] == "APPROVED_FOR_HELDOUT_EVALUATION"
        assert row["source_snapshot_sha256"] in source[
            "admitted_source_snapshots_sha256"
        ]
        assert row["modality"] in ADMITTED_MODALITIES
