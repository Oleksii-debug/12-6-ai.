from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import load as load_safetensors_bytes
from safetensors.numpy import save as save_safetensors_bytes

from twelve_six.checkpoint.core import (
    CheckpointIdentity,
    CheckpointIntegrityError,
    hash_json,
    load_checkpoint,
    load_verified_checkpoint,
    prepare_checkpoint_load,
    save_checkpoint,
    verify_checkpoint,
)
from twelve_six.checkpoint.state_tree import pack_state_tree, unpack_state_tree


class Model:
    def __init__(self, value: list[float]) -> None:
        self.weights = np.asarray(value, dtype=np.float64).copy()
        self.loads = 0

    def state_dict(self) -> dict[str, np.ndarray]:
        return {"weights": self.weights.copy()}

    def load_state_dict(self, state: dict[str, np.ndarray], strict: bool = True) -> None:
        assert not strict or set(state) == {"weights"}
        self.loads += 1
        self.weights = state["weights"].copy()


def identity(step: int = 0) -> CheckpointIdentity:
    return CheckpointIdentity(
        git_sha="a" * 40,
        model_spec={"kind": "test", "width": 3},
        parameter_count=3,
        tokenizer_hash="b" * 64,
        tokenizer_vocab_hash="c" * 64,
        dataset_manifest_hash="d" * 64,
        run_manifest_hash="e" * 64,
        training_config={"steps": 2},
        seed=7,
        precision="float64",
        step=step,
        tokens_seen=step * 3,
        optimizer={"name": "none"},
        scheduler=None,
        environment_lock_hash="f" * 64,
    )


def _rewrite_manifest_for_payload(checkpoint: Path, name: str) -> None:
    manifest_path = checkpoint / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    data = (checkpoint / name).read_bytes()
    manifest["files"][name] = {
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
    }
    manifest["checkpoint_id"] = hash_json(
        {"identity": manifest["identity"], "files": manifest["files"]}
    )
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    (checkpoint / "MANIFEST.sha256").write_text(
        f"{manifest_sha}  manifest.json\n", encoding="ascii"
    )


def test_existing_checkpoint_is_immutable_even_with_overwrite(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(checkpoint, model=Model([1, 2, 3]), identity=identity())
    before = {path.name: path.read_bytes() for path in checkpoint.iterdir()}

    with pytest.raises(FileExistsError, match="immutable"):
        save_checkpoint(
            checkpoint,
            model=Model([9, 9, 9]),
            identity=identity(1),
            overwrite=True,
        )

    after = {path.name: path.read_bytes() for path in checkpoint.iterdir()}
    assert before == after
    verify_checkpoint(checkpoint)


def test_untracked_file_fails_closed(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(checkpoint, model=Model([1, 2, 3]), identity=identity())
    (checkpoint / "notes.txt").write_text("not part of checkpoint", encoding="utf-8")

    with pytest.raises(CheckpointIntegrityError, match="inventory mismatch"):
        verify_checkpoint(checkpoint)


def test_symlink_payload_fails_closed(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(checkpoint, model=Model([1, 2, 3]), identity=identity())
    external = tmp_path / "external.safetensors"
    external.write_bytes((checkpoint / "weights.safetensors").read_bytes())
    (checkpoint / "weights.safetensors").unlink()
    try:
        os.symlink(external, checkpoint / "weights.safetensors")
    except (OSError, NotImplementedError):
        pytest.skip("symlink unavailable")

    with pytest.raises(CheckpointIntegrityError, match="non-symlink"):
        verify_checkpoint(checkpoint)


def test_verified_snapshot_is_not_changed_by_source_tamper(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(checkpoint, model=Model([1, 2, 3]), identity=identity())
    verified = prepare_checkpoint_load(checkpoint)

    # The verified snapshot, not a later re-open of the directory, is load authority.
    (checkpoint / "weights.safetensors").write_bytes(b"tampered after verified snapshot")
    target = Model([9, 9, 9])
    load_verified_checkpoint(verified, model=target, restore_rng=False)

    np.testing.assert_array_equal(target.weights, [1, 2, 3])
    assert target.loads == 1


def test_checksum_valid_malformed_state_tree_fails_before_model_mutation(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(
        checkpoint,
        model=Model([1, 2, 3]),
        identity=identity(),
        trainer_state={"x": 1},
    )
    bad_tree = {
        "__kind__": "mapping",
        "items": [
            [
                "rng",
                {
                    "__kind__": "tensor",
                    "key": "does-not-exist",
                    "backend": "numpy",
                    "device": None,
                },
            ]
        ],
    }
    (checkpoint / "state.json").write_text(
        json.dumps(bad_tree, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    _rewrite_manifest_for_payload(checkpoint, "state.json")
    target = Model([9, 9, 9])
    before = target.weights.copy()

    with pytest.raises(CheckpointIntegrityError, match="state tree"):
        load_checkpoint(checkpoint, model=target, restore_rng=False)

    np.testing.assert_array_equal(target.weights, before)
    assert target.loads == 0


def test_checksum_valid_malformed_safetensors_fails_before_model_mutation(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(checkpoint, model=Model([1, 2, 3]), identity=identity())
    (checkpoint / "state.safetensors").write_bytes(b"not-safetensors")
    _rewrite_manifest_for_payload(checkpoint, "state.safetensors")
    target = Model([9, 9, 9])
    before = target.weights.copy()

    with pytest.raises(CheckpointIntegrityError, match="state.safetensors"):
        load_checkpoint(checkpoint, model=target, restore_rng=False)

    np.testing.assert_array_equal(target.weights, before)
    assert target.loads == 0


def test_rng_preflight_occurs_before_model_mutation(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(checkpoint, model=Model([1, 2, 3]), identity=identity())

    tree = json.loads((checkpoint / "state.json").read_text(encoding="utf-8"))
    arrays = load_safetensors_bytes((checkpoint / "state.safetensors").read_bytes())
    state = copy.deepcopy(unpack_state_tree(tree, arrays))
    state["rng"]["python"] = ("bad-version",)
    packed = pack_state_tree(state)
    (checkpoint / "state.safetensors").write_bytes(save_safetensors_bytes(packed.tensors))
    (checkpoint / "state.json").write_text(
        json.dumps(packed.tree, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    _rewrite_manifest_for_payload(checkpoint, "state.safetensors")
    _rewrite_manifest_for_payload(checkpoint, "state.json")
    target = Model([9, 9, 9])
    before = target.weights.copy()

    with pytest.raises(Exception, match="Python RNG"):
        load_checkpoint(checkpoint, model=target, restore_rng=True)

    np.testing.assert_array_equal(target.weights, before)
    assert target.loads == 0
