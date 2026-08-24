from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from twelve_six.checkpoint import hash_json
from twelve_six.inference import portable_artifact
from twelve_six.inference.portable_artifact import (
    ARTIFACT_SCHEMA,
    INFERENCE_EVIDENCE_SCHEMA,
    PortableArtifactError,
    validate_portable_artifact_manifest,
    validate_portable_runtime_artifact,
)

SOURCE_SHA = "1" * 40
CHECKPOINT_ID = "2" * 64


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _rebuild_manifest(root: Path, *, source_sha: str = SOURCE_SHA) -> dict[str, Any]:
    records = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name == "artifact-manifest.json":
            continue
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    manifest: dict[str, Any] = {
        "schema": ARTIFACT_SCHEMA,
        "source_sha": source_sha,
        "files": records,
        "promotion_claim": False,
        "windows_nvda_live_pass": False,
    }
    manifest["manifest_sha256"] = hash_json(manifest)
    _write_json(root / "artifact-manifest.json", manifest)
    return manifest


def _write_fake_bundle(root: Path) -> None:
    inference: dict[str, Any] = {
        "schema": INFERENCE_EVIDENCE_SCHEMA,
        "status": "PASS",
        "candidate": {
            "repository": portable_artifact.REPOSITORY,
            "sha": SOURCE_SHA,
            "canonical_base": "random_init",
            "pretraining_only": True,
            "foreign_pretrained_weights": False,
            "behavioral_alignment_weights": False,
        },
        "checkpoint": {
            "checkpoint_id": CHECKPOINT_ID,
            "serialization_pickle": False,
            "corrupt_checkpoint_rejected": True,
        },
        "inference": {
            "parity": {"passed": True, "max_abs_error": 0.0, "max_rel_error": 0.0},
            "openai_compatible_raw_completion_equal": True,
            "chat_semantics": False,
        },
        "artifact": {
            "retained_for_external_execution": True,
            "windows_nvda_live_pass": False,
        },
        "truth_boundary": {
            "paid_compute": False,
            "candidate_or_stable_promotion": False,
            "windows_nvda_live_execution": "NOT_TESTED",
        },
    }
    inference["report_sha256"] = hash_json(inference)
    _write_json(root / "runtime/inference_evidence.json", inference)

    checkpoint = root / "runtime/checkpoint"
    checkpoint.mkdir(parents=True)
    for name, payload in {
        "manifest.json": b"{}\n",
        "MANIFEST.sha256": b"0" * 64 + b"\n",
        "weights.safetensors": b"weights",
        "state.safetensors": b"state",
        "state.json": b"{}\n",
    }.items():
        (checkpoint / name).write_bytes(payload)

    _write_json(root / "locked-environment-linux-x86_64.json", {})
    cli = {
        "mode": "greedy",
        "backend": {
            "backend": "first_party_torch",
            "git_sha": SOURCE_SHA,
            "checkpoint_id": CHECKPOINT_ID,
        },
    }
    _write_json(root / "cli-prompt.json", cli)
    _write_json(root / "cli-stdin.json", cli)
    _write_json(
        root / "server-response.json",
        {
            "object": "text_completion",
            "model": "12-6-base-s0",
            "choices": [{"index": 0, "text": "x"}],
        },
    )
    wheel = root / "dist/twelve_six-0.0.0-py3-none-any.whl"
    wheel.parent.mkdir(parents=True)
    wheel.write_bytes(b"portable wheel fixture")
    _rebuild_manifest(root)


def test_portable_manifest_recomputes_every_file(tmp_path: Path) -> None:
    _write_fake_bundle(tmp_path)
    result = validate_portable_artifact_manifest(
        tmp_path,
        expected_source_sha=SOURCE_SHA,
    )

    assert result["source_sha"] == SOURCE_SHA
    assert result["file_count"] == 11
    assert result["wheel"]["path"].endswith(".whl")

    wheel = tmp_path / result["wheel"]["path"]
    wheel.write_bytes(wheel.read_bytes() + b"tamper")
    with pytest.raises(PortableArtifactError, match="size mismatch"):
        validate_portable_artifact_manifest(tmp_path)


def test_portable_manifest_rejects_traversal_and_untracked_files(tmp_path: Path) -> None:
    _write_fake_bundle(tmp_path)
    manifest_path = tmp_path / "artifact-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    wheel_record = next(item for item in manifest["files"] if item["path"].endswith(".whl"))
    wheel_record["path"] = "../escape.whl"
    manifest.pop("manifest_sha256")
    manifest["manifest_sha256"] = hash_json(manifest)
    _write_json(manifest_path, manifest)

    with pytest.raises(PortableArtifactError, match="unsafe traversal"):
        validate_portable_artifact_manifest(tmp_path)

    extra_root = tmp_path / "extra"
    extra_root.mkdir()
    _write_fake_bundle(extra_root)
    (extra_root / "unexpected.bin").write_bytes(b"extra")
    with pytest.raises(PortableArtifactError, match="unmanifested"):
        validate_portable_artifact_manifest(extra_root)


def test_portable_manifest_rejects_symlink(tmp_path: Path) -> None:
    _write_fake_bundle(tmp_path)
    target = tmp_path / "server-response.json"
    link = tmp_path / "server-alias.json"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlink creation is not available in this test environment")
    _rebuild_manifest(tmp_path)

    with pytest.raises(PortableArtifactError, match="symlink"):
        validate_portable_artifact_manifest(tmp_path)


def test_deep_portable_validation_cross_binds_runtime_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_fake_bundle(tmp_path)

    monkeypatch.setattr(
        portable_artifact,
        "verify_checkpoint",
        lambda _path: {
            "checkpoint_id": CHECKPOINT_ID,
            "identity": {"git_sha": SOURCE_SHA},
        },
    )
    monkeypatch.setattr(
        portable_artifact,
        "validate_locked_environment_evidence",
        lambda _evidence, *, source_sha: {
            "profile_id": "linux-x86_64",
            "python_version": "3.11.16",
            "environment_evidence_sha256": "3" * 64,
            "source_sha": source_sha,
        },
    )

    report = validate_portable_runtime_artifact(
        tmp_path,
        expected_source_sha=SOURCE_SHA,
    )
    assert report["status"] == "PASS"
    assert report["checkpoint_id"] == CHECKPOINT_ID
    assert report["windows_handoff"] == {
        "artifact_only": True,
        "repository_checkout_required": False,
        "required_dependency_profile": "windows-x86_64",
        "hash_locked_windows_profile_available": False,
        "runtime_status": "BLOCKED_BY_MISSING_HASH_LOCKED_WINDOWS_RUNTIME",
        "nvda_status": "NOT_TESTED",
    }
    unhashed = dict(report)
    claimed = unhashed.pop("validation_sha256")
    assert hash_json(unhashed) == claimed


def test_deep_portable_validation_rejects_cross_surface_identity_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_fake_bundle(tmp_path)
    cli_path = tmp_path / "cli-prompt.json"
    cli = json.loads(cli_path.read_text(encoding="utf-8"))
    cli["backend"]["git_sha"] = "4" * 40
    _write_json(cli_path, cli)
    _rebuild_manifest(tmp_path)

    monkeypatch.setattr(
        portable_artifact,
        "verify_checkpoint",
        lambda _path: {
            "checkpoint_id": CHECKPOINT_ID,
            "identity": {"git_sha": SOURCE_SHA},
        },
    )
    monkeypatch.setattr(
        portable_artifact,
        "validate_locked_environment_evidence",
        lambda _evidence, *, source_sha: {"source_sha": source_sha},
    )

    with pytest.raises(PortableArtifactError, match="cli-prompt.json source SHA mismatch"):
        validate_portable_runtime_artifact(tmp_path, expected_source_sha=SOURCE_SHA)
