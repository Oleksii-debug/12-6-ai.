from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.numpy import load as load_safetensors_bytes
from safetensors.numpy import save as save_safetensors_bytes

from twelve_six.checkpoint import (
    CheckpointCompatibilityError,
    CheckpointIdentity,
    CheckpointIntegrityError,
    hash_json,
    load_checkpoint,
    save_checkpoint,
    verify_checkpoint,
)
from twelve_six.checkpoint.state_tree import pack_state_tree, unpack_state_tree


class NumpyModel:
    def __init__(self, values: list[float], *, dtype: np.dtype = np.dtype(np.float64)) -> None:
        self.weights = np.asarray(values, dtype=dtype).copy()
        self.loads = 0

    def state_dict(self) -> dict[str, np.ndarray]:
        return {"weights": self.weights.copy()}

    def load_state_dict(self, state: dict[str, np.ndarray], strict: bool = True) -> None:
        assert not strict or set(state) == {"weights"}
        self.loads += 1
        self.weights = state["weights"].copy()


def _identity(*, parameter_count: int, precision: str = "float64") -> CheckpointIdentity:
    return CheckpointIdentity(
        git_sha="a" * 40,
        model_spec={"kind": "d05-corruption-test", "parameter_count": parameter_count},
        parameter_count=parameter_count,
        tokenizer_hash="b" * 64,
        tokenizer_vocab_hash="c" * 64,
        dataset_manifest_hash="d" * 64,
        run_manifest_hash="e" * 64,
        training_config={"steps": 2, "precision": precision},
        seed=7,
        precision=precision,
        step=1,
        tokens_seen=parameter_count,
        optimizer={"name": "test"},
        scheduler=None,
        environment_lock_hash="f" * 64,
    )


def _rewrite_manifest(checkpoint: Path) -> None:
    manifest_path = checkpoint / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name in ("weights.safetensors", "state.safetensors", "state.json"):
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
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    (checkpoint / "MANIFEST.sha256").write_text(
        f"{digest}  manifest.json\n", encoding="ascii"
    )


def test_dtype_corruption_rejects_before_model_mutation(tmp_path: Path) -> None:
    checkpoint = tmp_path / "dtype"
    source = NumpyModel([1.0, 2.0, 3.0])
    save_checkpoint(checkpoint, model=source, identity=_identity(parameter_count=3))

    arrays = load_safetensors_bytes((checkpoint / "weights.safetensors").read_bytes())
    arrays["weights"] = arrays["weights"].astype(np.float32)
    (checkpoint / "weights.safetensors").write_bytes(save_safetensors_bytes(arrays))
    _rewrite_manifest(checkpoint)

    target = NumpyModel([9.0, 9.0, 9.0])
    before = target.weights.copy()
    with pytest.raises(CheckpointCompatibilityError, match="dtype mismatch"):
        load_checkpoint(checkpoint, model=target, restore_rng=False)

    np.testing.assert_array_equal(target.weights, before)
    assert target.loads == 0


def test_negative_counters_reject_after_consistent_manifest_rebind(tmp_path: Path) -> None:
    checkpoint = tmp_path / "counter"
    save_checkpoint(
        checkpoint,
        model=NumpyModel([1.0, 2.0, 3.0]),
        identity=_identity(parameter_count=3),
    )

    manifest_path = checkpoint / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["identity"]["step"] = -1
    manifest["identity"]["tokens_seen"] = -32
    manifest["checkpoint_id"] = hash_json(
        {"identity": manifest["identity"], "files": manifest["files"]}
    )
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    (checkpoint / "MANIFEST.sha256").write_text(
        f"{digest}  manifest.json\n", encoding="ascii"
    )

    with pytest.raises(CheckpointIntegrityError, match="step and tokens_seen"):
        verify_checkpoint(checkpoint)


def test_wrong_shaped_sgd_momentum_rejects_before_any_live_target_mutation(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "optimizer"
    source = torch.nn.Linear(2, 2, bias=False)
    source_optimizer = torch.optim.SGD(source.parameters(), lr=0.1, momentum=0.9)
    for parameter in source.parameters():
        parameter.grad = torch.ones_like(parameter)
    source_optimizer.step()
    source_optimizer.zero_grad(set_to_none=True)
    save_checkpoint(
        checkpoint,
        model=source,
        optimizer=source_optimizer,
        identity=_identity(parameter_count=4, precision="float32"),
    )

    tree = json.loads((checkpoint / "state.json").read_text(encoding="utf-8"))
    tensors = load_safetensors_bytes((checkpoint / "state.safetensors").read_bytes())
    state = copy.deepcopy(unpack_state_tree(tree, tensors))
    optimizer_state = state["optimizer"]
    parameter_id = next(iter(optimizer_state["state"]))
    optimizer_state["state"][parameter_id]["momentum_buffer"] = torch.zeros(1)
    packed = pack_state_tree(state)
    (checkpoint / "state.safetensors").write_bytes(save_safetensors_bytes(packed.tensors))
    (checkpoint / "state.json").write_text(
        json.dumps(packed.tree, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    _rewrite_manifest(checkpoint)

    target = torch.nn.Linear(2, 2, bias=False)
    target_optimizer = torch.optim.SGD(target.parameters(), lr=0.1, momentum=0.9)
    before_model = {name: tensor.detach().clone() for name, tensor in target.state_dict().items()}
    before_optimizer = copy.deepcopy(target_optimizer.state_dict())

    with pytest.raises(CheckpointCompatibilityError, match="optimizer state tensor shape mismatch"):
        load_checkpoint(
            checkpoint,
            model=target,
            optimizer=target_optimizer,
            restore_rng=False,
        )

    for name, tensor in target.state_dict().items():
        assert torch.equal(tensor, before_model[name])
    assert target_optimizer.state_dict() == before_optimizer
