from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import twelve_six.inference.artifact_transport as transport
from twelve_six.inference.artifact_transport import S0ArtifactTransportError

SOURCE_SHA = "a" * 40
CHECKPOINT_ID = "b" * 64
EVIDENCE_SHA = "c" * 64


def _payload(tmp_path: Path) -> Path:
    root = tmp_path / "payload"
    checkpoint = root / "checkpoint"
    checkpoint.mkdir(parents=True)
    for name in transport.REQUIRED_CHECKPOINT_FILES:
        (checkpoint / name).write_bytes(f"fixture:{name}\n".encode())
    (root / "s0-generation-evidence.json").write_text(
        json.dumps({"candidate_sha": SOURCE_SHA, "evidence_sha256": EVIDENCE_SHA}),
        encoding="utf-8",
    )
    return root


def _patch_semantic_validators(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        transport,
        "verify_checkpoint",
        lambda _path: {
            "checkpoint_id": CHECKPOINT_ID,
            "identity": {"git_sha": SOURCE_SHA},
        },
    )
    monkeypatch.setattr(
        transport,
        "validate_s0_generation_artifact",
        lambda evidence, *, checkpoint_path: None,
    )


def test_transport_manifest_round_trip_and_source_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_semantic_validators(monkeypatch)
    root = _payload(tmp_path)
    manifest = transport.build_s0_artifact_transport_manifest(root, source_sha=SOURCE_SHA)

    validated = transport.validate_s0_artifact_transport_manifest(
        root,
        manifest,
        expected_source_sha=SOURCE_SHA,
    )
    assert validated["checkpoint"]["checkpoint_id"] == CHECKPOINT_ID
    assert validated["generation_evidence"]["evidence_sha256"] == EVIDENCE_SHA
    assert validated["transport_claims"]["promotion_authority"] is False

    with pytest.raises(S0ArtifactTransportError, match="source SHA mismatch"):
        transport.validate_s0_artifact_transport_manifest(
            root,
            manifest,
            expected_source_sha="d" * 40,
        )


def test_downloaded_byte_tamper_add_delete_and_manifest_tamper_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_semantic_validators(monkeypatch)
    root = _payload(tmp_path)
    manifest = transport.build_s0_artifact_transport_manifest(root, source_sha=SOURCE_SHA)

    weights = root / "checkpoint/weights.safetensors"
    original = weights.read_bytes()
    weights.write_bytes(original + b"tamper")
    with pytest.raises(S0ArtifactTransportError, match="byte inventory mismatch"):
        transport.validate_s0_artifact_transport_manifest(root, manifest)
    weights.write_bytes(original)

    extra = root / "checkpoint/untracked.bin"
    extra.write_bytes(b"unexpected")
    with pytest.raises(S0ArtifactTransportError, match="byte inventory mismatch"):
        transport.validate_s0_artifact_transport_manifest(root, manifest)
    extra.unlink()

    state = root / "checkpoint/state.json"
    saved_state = state.read_bytes()
    state.unlink()
    with pytest.raises(S0ArtifactTransportError, match="byte inventory mismatch"):
        transport.validate_s0_artifact_transport_manifest(root, manifest)
    state.write_bytes(saved_state)

    tampered_manifest = copy.deepcopy(manifest)
    tampered_manifest["transport_claims"]["promotion_authority"] = True
    with pytest.raises(S0ArtifactTransportError, match="self-hash mismatch"):
        transport.validate_s0_artifact_transport_manifest(root, tampered_manifest)


def test_symlink_nodes_are_rejected_before_transport(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_semantic_validators(monkeypatch)
    root = _payload(tmp_path)
    target = root / "checkpoint/weights.safetensors"
    link = root / "checkpoint/alias.safetensors"
    try:
        link.symlink_to(target.name)
    except OSError:
        pytest.skip("symlinks unavailable on this filesystem")

    with pytest.raises(S0ArtifactTransportError, match="symlink rejected"):
        transport.build_s0_artifact_transport_manifest(root, source_sha=SOURCE_SHA)
