from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from twelve_six.eval233_holdout import Eval233Error, build, git_blob_sha1, hash_json, verify


def _fixture(root: Path) -> tuple[str, str, str, list[bytes]]:
    seed = root / "data/evaluation/recover174_real_holdout_seed.jsonl.gz"
    auth = root / "configs/evaluation/recover174_source_authority_v1.json"
    seed.parent.mkdir(parents=True)
    auth.parent.mkdir(parents=True)
    authority = {
        "admitted_modalities": ["ua", "en"],
        "blocked_modalities": {"code": {"status": "BLOCKED_NO_EVALUATION_USE_AUTHORITY"}},
        "sources": {},
    }
    rows = []
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
    auth_bytes = json.dumps(authority, sort_keys=True).encode()
    seed_bytes = gzip.compress(b"".join(rows), mtime=0)
    auth.write_bytes(auth_bytes)
    seed.write_bytes(seed_bytes)
    return git_blob_sha1(seed_bytes), git_blob_sha1(auth_bytes), authority_id, rows


def test_preserves_bytes_and_seals_final_test(tmp_path: Path) -> None:
    seed_sha, auth_sha, auth_id, rows = _fixture(tmp_path)
    out = tmp_path / "out"
    manifest = build(
        tmp_path,
        out,
        source_sha="a" * 40,
        expected_seed_git_blob_sha1=seed_sha,
        expected_authority_git_blob_sha1=auth_sha,
        expected_authority_identity=auth_id,
    )
    emitted = []
    for purpose in ("selection-validation", "final-test"):
        for modality in ("ua", "en"):
            emitted += (out / purpose / f"{modality}.jsonl").read_bytes().splitlines(keepends=True)
    assert sorted(emitted) == sorted(rows)
    assert manifest["code"]["documents"] == 0
    assert manifest["decontamination"]["evaluation_release_allowed"] is False
    final = json.loads((out / "final-test/manifest.json").read_text())
    assert final["selection_eligible"] is False
    assert final["tokenizer_fit_eligible"] is False
    assert final["hyperparameter_selection_eligible"] is False


def test_immutable_rerun_and_tamper_failure(tmp_path: Path) -> None:
    seed_sha, auth_sha, auth_id, _ = _fixture(tmp_path)
    out = tmp_path / "out"
    kwargs = {
        "source_sha": "b" * 40,
        "expected_seed_git_blob_sha1": seed_sha,
        "expected_authority_git_blob_sha1": auth_sha,
        "expected_authority_identity": auth_id,
    }
    assert build(tmp_path, out, **kwargs) == build(tmp_path, out, **kwargs)
    target = out / "selection-validation/ua.jsonl"
    target.write_bytes(target.read_bytes() + b"{}\n")
    with pytest.raises(Eval233Error):
        verify(out)
