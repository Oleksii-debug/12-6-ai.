"""Synthetic S18 integrity-only regression; no release or training authority."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools import plan2_s18_release_integrity_v1 as gate


def _encode(obj):
    return (json.dumps(obj, sort_keys=True, ensure_ascii=False,
                       separators=(",", ":")) + "\n").encode("utf-8")


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _bundle(tmp_path):
    root = tmp_path / "bundle"
    root.mkdir()
    docs = {name: {"fixture": True} for name in gate.REQUIRED}
    raw = {name: _encode(value) for name, value in docs.items()}
    dataset = _sha(raw["dataset-manifest.json"])
    tokenizer = _sha(raw["tokenizer-manifest.json"])
    raw["shards-manifest.json"] = _encode({"dataset_sha256": dataset,
                                             "tokenizer_sha256": tokenizer})
    shards = _sha(raw["shards-manifest.json"])
    raw["exposure-manifest.json"] = _encode({"shards_sha256": shards})
    exposure = _sha(raw["exposure-manifest.json"])
    raw["rights-attribution.json"] = _encode({"dataset_sha256": dataset})
    rights = _sha(raw["rights-attribution.json"])
    raw["plan9-binding.json"] = _encode({
        "dataset_sha256": dataset, "tokenizer_sha256": tokenizer,
        "shards_sha256": shards, "exposure_sha256": exposure,
        "rights_sha256": rights, "campaign_authorized": False,
    })
    for name, content in raw.items():
        (root / name).write_bytes(content)
    manifest = _encode({"schema_version": gate.SCHEMA,
                        "producer_commit_sha1": "a" * 40,
                        "producer_tree_sha1": "b" * 40,
                        "members": {name: _sha(content) for name, content in raw.items()}})
    (root / gate.MANIFEST).write_bytes(manifest)
    return root, _sha(manifest)


@pytest.mark.parametrize("insertion", ["file", "empty_dir", "nested_file"])
def test_late_extra_member_denied(tmp_path: Path, monkeypatch, insertion: str):
    root, trusted = _bundle(tmp_path)
    original = gate._sha256_file
    injected = False

    def add_after_inventory(path):
        nonlocal injected
        if path.name == "dataset-manifest.json" and not injected:
            injected = True
            if insertion == "file":
                (root / "rogue.bin").write_bytes(b"unlisted")
            elif insertion == "empty_dir":
                (root / "rogue-empty").mkdir()
            else:
                (root / "rogue-dir").mkdir()
                (root / "rogue-dir" / "rogue.bin").write_bytes(b"unlisted")
        return original(path)

    monkeypatch.setattr(gate, "_sha256_file", add_after_inventory)
    with pytest.raises(gate.IntegrityDenied, match="unlisted|inventory"):
        gate.verify(root, trusted)
    assert injected


def test_unchanged_fixture_is_never_release_authority(tmp_path: Path):
    root, trusted = _bundle(tmp_path)
    report = gate.verify(root, trusted)
    assert report["decision"] == "BYTE_INTEGRITY_ONLY_NOT_RELEASE"
    assert report["production_release_authorized"] is False
    assert report["plan9_campaign_authorized"] is False
    assert report["terminal_done"] is False
