from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from twelve_six.eval233_holdout import Eval233Error, build, git_blob_sha1, hash_json, verify


def _fixture(root: Path) -> tuple[str, str, str, bytes]:
    seed_path = root / "data/evaluation/recover174_real_holdout_seed.jsonl.gz"
    authority_path = root / "configs/evaluation/recover174_source_authority_v1.json"
    seed_path.parent.mkdir(parents=True)
    authority_path.parent.mkdir(parents=True)
    authority = {
        "admitted_modalities": ["ua", "en"],
        "blocked_modalities": {"code": {"status": "BLOCKED_NO_EVALUATION_USE_AUTHORITY"}},
        "sources": {},
    }
    rows: list[bytes] = []
    for modality in ("ua", "en"):
        source_id = f"{modality}.fixture"
        snapshots = []
        for index in range(4):
            snapshot = f"{index + (10 if modality == 'en' else 0):064x}"
            snapshots.append(snapshot)
            row = {
                "record_id": f"{modality}-{index}",
                "modality": modality,
                "source_id": source_id,
                "source_family": f"family-{modality}",
                "source_version": "v1",
                "source_snapshot_sha256": snapshot,
                "source_kind": "EXTERNAL_REAL",
                "evaluation_use_authority_ref": "fixture",
                "provenance_ref": "fixture",
                "text": f"{modality}-text-{index}",
            }
            rows.append(json.dumps(row, separators=(",", ":")).encode() + b"\n")
        authority["sources"][source_id] = {
            "evaluation_status": "APPROVED_FOR_HELDOUT_EVALUATION",
            "raw_sha256": [("1" if modality == "ua" else "2") * 64],
            "source_identity_sha256": ("3" if modality == "ua" else "4") * 64,
            "admitted_source_snapshots_sha256": snapshots,
        }
    authority_id = hash_json(authority)
    authority["authority_identity_sha256"] = authority_id
    authority_blob = json.dumps(authority, sort_keys=True).encode()
    seed_blob = gzip.compress(b"".join(rows), mtime=0)
    authority_path.write_bytes(authority_blob)
    seed_path.write_bytes(seed_blob)
    return git_blob_sha1(seed_blob), git_blob_sha1(authority_blob), authority_id, seed_blob


def test_exact_final_test_preserved_and_selection_remains_empty(tmp_path: Path) -> None:
    seed_sha, auth_sha, auth_id, seed_blob = _fixture(tmp_path)
    out = tmp_path / "out"
    manifest = build(
        tmp_path,
        out,
        source_sha="a" * 40,
        expected_seed_git_blob_sha1=seed_sha,
        expected_authority_git_blob_sha1=auth_sha,
        expected_authority_identity=auth_id,
    )
    assert (out / "final-test/recover174_real_holdout_seed.jsonl.gz").read_bytes() == seed_blob
    selection = json.loads((out / "selection-validation/manifest.json").read_text())
    final = json.loads((out / "final-test/manifest.json").read_text())
    assert selection["documents"] == 0
    assert selection["records"] == []
    assert selection["invented_from_final_test"] is False
    assert final["documents"] == 8
    assert final["modality_documents"] == {"ua": 4, "en": 4}
    assert final["selection_eligible"] is False
    assert final["tokenizer_fit_eligible"] is False
    assert final["hyperparameter_selection_eligible"] is False
    assert manifest["code"]["documents"] == 0
    assert manifest["code"]["evaluation_use_explicitly_authorized"] is False
    assert manifest["decontamination"]["scan_executed"] is False
    assert manifest["decontamination"]["evaluation_release_allowed"] is False


def test_exact_rerun_is_immutable_and_tamper_fails(tmp_path: Path) -> None:
    seed_sha, auth_sha, auth_id, _ = _fixture(tmp_path)
    out = tmp_path / "out"
    kwargs = {
        "source_sha": "b" * 40,
        "expected_seed_git_blob_sha1": seed_sha,
        "expected_authority_git_blob_sha1": auth_sha,
        "expected_authority_identity": auth_id,
    }
    assert build(tmp_path, out, **kwargs) == build(tmp_path, out, **kwargs)
    target = out / "final-test/recover174_real_holdout_seed.jsonl.gz"
    target.write_bytes(target.read_bytes() + b"x")
    with pytest.raises(Eval233Error, match="immutable output bytes differ"):
        build(tmp_path, out, **kwargs)
    with pytest.raises(Eval233Error):
        verify(out)


def test_provenance_defect_fails_closed(tmp_path: Path) -> None:
    seed_sha, auth_sha, auth_id, _ = _fixture(tmp_path)
    seed_path = tmp_path / "data/evaluation/recover174_real_holdout_seed.jsonl.gz"
    rows = gzip.decompress(seed_path.read_bytes()).splitlines(keepends=True)
    first = json.loads(rows[0])
    first["source_snapshot_sha256"] = "f" * 64
    rows[0] = json.dumps(first, separators=(",", ":")).encode() + b"\n"
    tampered = gzip.compress(b"".join(rows), mtime=0)
    seed_path.write_bytes(tampered)
    with pytest.raises(Eval233Error, match="not admitted"):
        build(
            tmp_path,
            tmp_path / "out",
            source_sha="c" * 40,
            expected_seed_git_blob_sha1=git_blob_sha1(tampered),
            expected_authority_git_blob_sha1=auth_sha,
            expected_authority_identity=auth_id,
        )
