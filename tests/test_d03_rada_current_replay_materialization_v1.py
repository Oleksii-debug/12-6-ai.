from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "tools" / "materialize_d03_rada_current_snapshot_candidate_v1.py"


def _load():
    spec = importlib.util.spec_from_file_location(
        "materialize_d03_rada_current_snapshot_candidate_test",
        MODULE,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_materializer_constants_bind_exact_physical_replay() -> None:
    mod = _load()
    assert mod.SOURCE_ARCHIVE_BYTES == 46_774_786
    assert mod.SOURCE_ARCHIVE_SHA256 == (
        "9f937b3283dd2d9508a0a92442fc007e7e6d373a17d3d50fa428e324cb2fa290"
    )
    assert mod.NORMALIZER_GIT_BLOB_SHA1 == (
        "55ef5f5e7f09042489fa2642d099aedffaddb740"
    )
    assert mod.QP_TOOL_GIT_BLOB_SHA1 == (
        "69491874e60a1b846d683fff3c10214a869f18ff"
    )
    assert mod.EXPECTED_ACCEPTED_CHUNKS == 101_733
    assert mod.EXPECTED_ACCEPTED_PAYLOAD_BYTES == 192_393_157
    assert mod.EXPECTED_ACCEPTED_JSONL_BYTES == 224_897_989
    assert mod.EXPECTED_ACCEPTED_JSONL_SHA256 == (
        "8d1343708b3ce1747d32c3b551d6fb8ab7123be5b37f74fff3c457b6439159a7"
    )
    assert mod.EXPECTED_EXACT_DUPLICATE_HASHES == 765


def test_git_blob_sha_matches_git_object_formula() -> None:
    mod = _load()
    payload = b"exact mechanics\n"
    expected = hashlib.sha1(  # noqa: S324 - test of Git identity
        f"blob {len(payload)}\0".encode("ascii") + payload
    ).hexdigest()
    assert mod._git_blob_sha1(payload) == expected


def test_load_json_requires_exact_transport_identity(tmp_path: Path) -> None:
    mod = _load()
    path = tmp_path / "evidence.json"
    raw = b'{"x":1}\n'
    path.write_bytes(raw)

    assert mod._load_json_with_transport(
        path,
        label="fixture",
        expected_sha256=hashlib.sha256(raw).hexdigest(),
    ) == {"x": 1}

    with pytest.raises(
        mod.RadaCurrentReplayMaterializationError,
        match="transport identity drift",
    ):
        mod._load_json_with_transport(
            path,
            label="fixture",
            expected_sha256="0" * 64,
        )


def test_read_regular_bytes_rejects_path_descriptor_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load()
    path = tmp_path / "source.bin"
    path.write_bytes(b"payload")
    real_lstat = os.lstat

    def substituted_lstat(candidate: Path):
        observed = real_lstat(candidate)
        if candidate == path:
            return SimpleNamespace(
                st_mode=observed.st_mode,
                st_dev=observed.st_dev + 1,
                st_ino=observed.st_ino,
            )
        return observed

    monkeypatch.setattr(os, "lstat", substituted_lstat)
    with pytest.raises(
        mod.RadaCurrentReplayMaterializationError,
        match="path changed before descriptor lock",
    ):
        mod._read_regular_bytes(path, label="fixture")


def test_create_only_output_rejects_existing_and_dangling_symlink(
    tmp_path: Path,
) -> None:
    mod = _load()
    existing = tmp_path / "existing.json"
    existing.write_bytes(b"old")
    with pytest.raises(
        mod.RadaCurrentReplayMaterializationError,
        match="refusing to overwrite",
    ):
        mod._write_create_only(existing, b"new")
    assert existing.read_bytes() == b"old"

    dangling = tmp_path / "dangling.json"
    dangling.symlink_to(tmp_path / "missing")
    with pytest.raises(
        mod.RadaCurrentReplayMaterializationError,
        match="refusing to overwrite",
    ):
        mod._write_create_only(dangling, b"new")


def test_exact_mechanics_module_loader_rejects_blob_drift(tmp_path: Path) -> None:
    mod = _load()
    source = tmp_path / "fixture.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    raw = source.read_bytes()
    expected = mod._git_blob_sha1(raw)

    loaded = mod._load_module(source, "fixture_exact_module", expected_blob=expected)
    assert loaded.VALUE == 1

    with pytest.raises(
        mod.RadaCurrentReplayMaterializationError,
        match="Git blob drift",
    ):
        mod._load_module(
            source,
            "fixture_drifted_module",
            expected_blob="0" * 40,
        )


def test_summary_contract_contains_only_hash_counts_and_zero_authority() -> None:
    mod = _load()
    source = MODULE.read_text(encoding="utf-8")
    assert '"raw_text_emitted_to_summary": False' in source
    assert '"production_qp_authority_established": False' in source
    assert '"canonical_capacity_credited": 0' in source
    assert '"training_authorized_bytes": 0' in source
    assert '"authorized_optimized_target_exposure": 0' in source
    assert '"tokenizer_fit_authorized": False' in source
    assert '"training_executed": False' in source
    assert '"final_test_outcomes_read": False' in source
    assert '"paid_compute_used": False' in source
    assert "accepted_jsonl" not in json.dumps(
        {
            "candidate_output_is_ephemeral": True,
            "raw_text_emitted_to_summary": False,
        }
    )
