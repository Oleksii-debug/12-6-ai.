import json
from pathlib import Path

import numpy as np
import pytest

from twelve_six.checkpoint import (
    CheckpointIdentity,
    CheckpointIntegrityError,
    hash_json,
    save_checkpoint,
    verify_checkpoint,
)
from twelve_six.checkpoint.core import MANIFEST_CHECKSUM_NAME, MANIFEST_NAME, canonical_json_bytes


class Model:
    def __init__(self):
        self.value = np.array([1.0, 2.0], dtype=np.float32)

    def state_dict(self):
        return {"value": self.value}

    def load_state_dict(self, state, strict=True):
        self.value = state["value"]


def checkpoint_identity() -> CheckpointIdentity:
    return CheckpointIdentity(
        git_sha="d" * 40,
        model_spec={"model_type": "strict-schema-test", "vocab_size": 2},
        parameter_count=2,
        tokenizer_hash="1" * 64,
        tokenizer_vocab_hash="2" * 64,
        dataset_manifest_hash="3" * 64,
        run_manifest_hash="4" * 64,
        training_config={"steps": 0},
        seed=7,
        precision="float32",
        step=0,
        tokens_seen=0,
        optimizer={"name": "none"},
        scheduler=None,
        environment_lock_hash="5" * 64,
    )


def make_checkpoint(tmp_path: Path) -> Path:
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(checkpoint, model=Model(), identity=checkpoint_identity())
    verify_checkpoint(checkpoint)
    return checkpoint


def rewrite_manifest(
    checkpoint: Path,
    mutate,
    *,
    recompute_checkpoint_id: bool = False,
) -> None:
    path = checkpoint / MANIFEST_NAME
    manifest = json.loads(path.read_text(encoding="utf-8"))
    mutate(manifest)
    if recompute_checkpoint_id:
        manifest["checkpoint_id"] = hash_json(
            {"identity": manifest["identity"], "files": manifest["files"]}
        )
    payload = canonical_json_bytes(manifest) + b"\n"
    path.write_bytes(payload)
    import hashlib

    digest = hashlib.sha256(payload).hexdigest()
    (checkpoint / MANIFEST_CHECKSUM_NAME).write_text(
        f"{digest}  {MANIFEST_NAME}\n", encoding="ascii"
    )


def test_checkpoint_v1_baseline_manifest_is_accepted(tmp_path: Path):
    checkpoint = make_checkpoint(tmp_path)
    manifest = verify_checkpoint(checkpoint)
    assert manifest["format"] == "12-6-checkpoint"
    assert manifest["format_version"] == 1


def test_checkpoint_v1_rejects_unknown_top_level_field_even_when_rechecksummed(tmp_path: Path):
    checkpoint = make_checkpoint(tmp_path)
    rewrite_manifest(checkpoint, lambda manifest: manifest.__setitem__("future_semantics", {}))

    with pytest.raises(CheckpointIntegrityError, match="manifest keys"):
        verify_checkpoint(checkpoint)


def test_checkpoint_v1_rejects_missing_identity_field_even_with_new_checkpoint_id(tmp_path: Path):
    checkpoint = make_checkpoint(tmp_path)

    def mutate(manifest):
        manifest["identity"].pop("precision")

    rewrite_manifest(checkpoint, mutate, recompute_checkpoint_id=True)
    with pytest.raises(CheckpointIntegrityError, match="identity keys"):
        verify_checkpoint(checkpoint)


def test_checkpoint_v1_rejects_unknown_identity_field_even_with_new_checkpoint_id(tmp_path: Path):
    checkpoint = make_checkpoint(tmp_path)

    def mutate(manifest):
        manifest["identity"]["undeclared_resume_semantics"] = "silent-drift"

    rewrite_manifest(checkpoint, mutate, recompute_checkpoint_id=True)
    with pytest.raises(CheckpointIntegrityError, match="identity keys"):
        verify_checkpoint(checkpoint)


@pytest.mark.parametrize("bad_parameter_count", [0, -1, True, "2"])
def test_checkpoint_v1_rejects_invalid_parameter_count_with_consistent_hashes(
    tmp_path: Path,
    bad_parameter_count,
):
    checkpoint = make_checkpoint(tmp_path)

    def mutate(manifest):
        manifest["identity"]["parameter_count"] = bad_parameter_count

    rewrite_manifest(checkpoint, mutate, recompute_checkpoint_id=True)
    with pytest.raises(CheckpointIntegrityError, match="parameter_count"):
        verify_checkpoint(checkpoint)


def test_checkpoint_v1_rejects_non_mapping_environment_with_consistent_hashes(tmp_path: Path):
    checkpoint = make_checkpoint(tmp_path)

    def mutate(manifest):
        manifest["identity"]["environment"] = None
        manifest["identity"]["environment_hash"] = hash_json(None)

    rewrite_manifest(checkpoint, mutate, recompute_checkpoint_id=True)
    with pytest.raises(CheckpointIntegrityError, match="environment"):
        verify_checkpoint(checkpoint)


def test_checkpoint_v1_rejects_serialization_semantic_drift(tmp_path: Path):
    checkpoint = make_checkpoint(tmp_path)

    def mutate(manifest):
        manifest["serialization"]["pickle"] = True

    rewrite_manifest(checkpoint, mutate)
    with pytest.raises(CheckpointIntegrityError, match="serialization"):
        verify_checkpoint(checkpoint)


def test_checkpoint_v1_rejects_unknown_serialization_field(tmp_path: Path):
    checkpoint = make_checkpoint(tmp_path)

    def mutate(manifest):
        manifest["serialization"]["unsafe_extension"] = "ignored-by-old-reader"

    rewrite_manifest(checkpoint, mutate)
    with pytest.raises(CheckpointIntegrityError, match="serialization keys"):
        verify_checkpoint(checkpoint)


def test_checkpoint_v1_rejects_unknown_file_record_field_even_with_new_checkpoint_id(
    tmp_path: Path,
):
    checkpoint = make_checkpoint(tmp_path)

    def mutate(manifest):
        manifest["files"]["weights.safetensors"]["codec"] = "implicit"

    rewrite_manifest(checkpoint, mutate, recompute_checkpoint_id=True)
    with pytest.raises(CheckpointIntegrityError, match="file record keys"):
        verify_checkpoint(checkpoint)


@pytest.mark.parametrize(
    "created_at",
    [
        "not-a-timestamp",
        "2026-08-25T00:00:00+03:00",
        123,
        None,
    ],
)
def test_checkpoint_v1_rejects_invalid_created_at_utc(tmp_path: Path, created_at):
    checkpoint = make_checkpoint(tmp_path)
    rewrite_manifest(checkpoint, lambda manifest: manifest.__setitem__("created_at_utc", created_at))

    with pytest.raises(CheckpointIntegrityError, match="created_at_utc"):
        verify_checkpoint(checkpoint)


def test_checkpoint_v1_rejects_boolean_format_version(tmp_path: Path):
    checkpoint = make_checkpoint(tmp_path)
    rewrite_manifest(checkpoint, lambda manifest: manifest.__setitem__("format_version", True))

    with pytest.raises(CheckpointIntegrityError, match="format_version"):
        verify_checkpoint(checkpoint)
