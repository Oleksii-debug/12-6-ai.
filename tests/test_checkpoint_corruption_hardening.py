from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import load as load_safetensors_bytes
from safetensors.numpy import save as save_safetensors_bytes

from twelve_six.checkpoint.core import (
    CheckpointCompatibilityError,
    CheckpointIdentity,
    CheckpointIntegrityError,
    hash_json,
    load_checkpoint,
    save_checkpoint,
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


class MomentumOptimizer:
    def __init__(self, model: Model) -> None:
        self.model = model
        self.velocity = np.zeros_like(model.weights)
        self.loads = 0

    def state_dict(self) -> dict[str, object]:
        return {"velocity": self.velocity.copy()}

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.loads += 1
        self.velocity = np.asarray(state["velocity"]).copy()


def identity(*, step: int = 0, tokens_seen: int = 0) -> CheckpointIdentity:
    return CheckpointIdentity(
        git_sha="a" * 40,
        model_spec={"kind": "d05-hardening-test", "width": 3},
        parameter_count=3,
        tokenizer_hash="b" * 64,
        tokenizer_vocab_hash="c" * 64,
        dataset_manifest_hash="d" * 64,
        run_manifest_hash="e" * 64,
        training_config={"steps": 2},
        seed=7,
        precision="float64",
        step=step,
        tokens_seen=tokens_seen,
        optimizer={"name": "MomentumOptimizer"},
        scheduler=None,
        environment_lock_hash="f" * 64,
    )


def _rebind_manifest(checkpoint: Path) -> None:
    manifest_path = checkpoint / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name in manifest["files"]:
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


def _replace_combined_state(checkpoint: Path, combined_state: object) -> None:
    packed = pack_state_tree(combined_state)
    (checkpoint / "state.safetensors").write_bytes(save_safetensors_bytes(packed.tensors))
    (checkpoint / "state.json").write_text(
        json.dumps(packed.tree, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    _rebind_manifest(checkpoint)


def test_model_dtype_corruption_rejects_before_model_mutation(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(checkpoint, model=Model([1, 2, 3]), identity=identity())

    arrays = load_safetensors_bytes((checkpoint / "weights.safetensors").read_bytes())
    arrays["weights"] = arrays["weights"].astype(np.float32)
    (checkpoint / "weights.safetensors").write_bytes(save_safetensors_bytes(arrays))
    _rebind_manifest(checkpoint)

    target = Model([9, 9, 9])
    before = target.weights.copy()
    with pytest.raises(CheckpointCompatibilityError, match="dtype mismatch"):
        load_checkpoint(checkpoint, model=target, restore_rng=False)

    np.testing.assert_array_equal(target.weights, before)
    assert target.loads == 0


def test_wrong_shaped_optimizer_state_rejects_before_model_mutation(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    source_model = Model([1, 2, 3])
    source_optimizer = MomentumOptimizer(source_model)
    source_optimizer.velocity[:] = [0.1, 0.2, 0.3]
    save_checkpoint(
        checkpoint,
        model=source_model,
        optimizer=source_optimizer,
        identity=identity(),
    )

    tree = json.loads((checkpoint / "state.json").read_text(encoding="utf-8"))
    state_arrays = load_safetensors_bytes((checkpoint / "state.safetensors").read_bytes())
    combined_state = copy.deepcopy(unpack_state_tree(tree, state_arrays))
    combined_state["optimizer"]["velocity"] = np.asarray([7.0, 8.0], dtype=np.float64)
    _replace_combined_state(checkpoint, combined_state)

    target_model = Model([9, 9, 9])
    target_optimizer = MomentumOptimizer(target_model)
    model_before = target_model.weights.copy()
    optimizer_before = target_optimizer.velocity.copy()

    with pytest.raises(CheckpointCompatibilityError, match="optimizer.*shape mismatch"):
        load_checkpoint(
            checkpoint,
            model=target_model,
            optimizer=target_optimizer,
            restore_rng=False,
        )

    np.testing.assert_array_equal(target_model.weights, model_before)
    np.testing.assert_array_equal(target_optimizer.velocity, optimizer_before)
    assert target_model.loads == 0
    assert target_optimizer.loads == 0


def test_negative_resume_counters_reject_after_consistent_manifest_rebinding(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(
        checkpoint,
        model=Model([1, 2, 3]),
        identity=identity(step=2, tokens_seen=6),
    )

    manifest_path = checkpoint / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["identity"]["step"] = -1
    manifest["identity"]["tokens_seen"] = -3
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    _rebind_manifest(checkpoint)

    target = Model([9, 9, 9])
    before = target.weights.copy()
    with pytest.raises(CheckpointIntegrityError, match="step.*tokens_seen.*non-negative"):
        load_checkpoint(checkpoint, model=target, restore_rng=False)

    np.testing.assert_array_equal(target.weights, before)
    assert target.loads == 0


def test_expected_resume_counters_are_bound_before_model_mutation(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(
        checkpoint,
        model=Model([1, 2, 3]),
        identity=identity(step=2, tokens_seen=6),
    )

    target = Model([9, 9, 9])
    before = target.weights.copy()
    with pytest.raises(CheckpointCompatibilityError, match="identity mismatch"):
        load_checkpoint(
            checkpoint,
            model=target,
            restore_rng=False,
            expected_step=3,
            expected_tokens_seen=7,
        )

    np.testing.assert_array_equal(target.weights, before)
    assert target.loads == 0
