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

from twelve_six.checkpoint.core import (
    CheckpointCompatibilityError,
    CheckpointIdentity,
    hash_json,
    load_checkpoint,
    save_checkpoint,
)
from twelve_six.checkpoint.state_tree import pack_state_tree, unpack_state_tree


def _identity() -> CheckpointIdentity:
    return CheckpointIdentity(
        git_sha="a" * 40,
        model_spec={"kind": "d05-torch-sgd-regression", "width": 4},
        parameter_count=16,
        tokenizer_hash="b" * 64,
        tokenizer_vocab_hash="c" * 64,
        dataset_manifest_hash="d" * 64,
        run_manifest_hash="e" * 64,
        training_config={"steps": 2},
        seed=7,
        precision="float32",
        step=1,
        tokens_seen=16,
        optimizer={"name": "SGD", "momentum": 0.9},
        scheduler=None,
        environment_lock_hash="f" * 64,
    )


def _rebind_manifest(checkpoint: Path) -> None:
    manifest_path = checkpoint / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name in manifest["files"]:
        payload = (checkpoint / name).read_bytes()
        manifest["files"][name] = {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
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


def _saved_sgd_checkpoint(tmp_path: Path) -> tuple[Path, object]:
    source_model = torch.nn.Linear(4, 4, bias=False)
    source_optimizer = torch.optim.SGD(source_model.parameters(), lr=0.1, momentum=0.9)

    source_model.weight.grad = torch.ones_like(source_model.weight)
    source_optimizer.step()
    source_optimizer.zero_grad(set_to_none=True)

    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(
        checkpoint,
        model=source_model,
        optimizer=source_optimizer,
        identity=_identity(),
    )

    tree = json.loads((checkpoint / "state.json").read_text(encoding="utf-8"))
    state_arrays = load_safetensors_bytes((checkpoint / "state.safetensors").read_bytes())
    combined_state = copy.deepcopy(unpack_state_tree(tree, state_arrays))
    return checkpoint, combined_state


def _first_momentum_buffer(combined_state: object) -> tuple[dict[str, object], object]:
    assert isinstance(combined_state, dict)
    optimizer = combined_state["optimizer"]
    assert isinstance(optimizer, dict)
    optimizer_state = optimizer["state"]
    assert isinstance(optimizer_state, dict)
    first_parameter_id = next(iter(optimizer_state))
    parameter_state = optimizer_state[first_parameter_id]
    assert isinstance(parameter_state, dict)
    return parameter_state, parameter_state["momentum_buffer"]


def test_one_element_sgd_momentum_rejects_before_model_mutation(tmp_path: Path) -> None:
    checkpoint, combined_state = _saved_sgd_checkpoint(tmp_path)
    parameter_state, momentum = _first_momentum_buffer(combined_state)

    if isinstance(momentum, torch.Tensor):
        corrupt_momentum = torch.zeros((1,), dtype=momentum.dtype)
    elif isinstance(momentum, np.ndarray):
        corrupt_momentum = np.zeros((1,), dtype=momentum.dtype)
    else:  # pragma: no cover - fail loudly if the state-tree contract changes.
        raise AssertionError(f"unexpected momentum_buffer type: {type(momentum)!r}")
    parameter_state["momentum_buffer"] = corrupt_momentum
    _replace_combined_state(checkpoint, combined_state)

    target_model = torch.nn.Linear(4, 4, bias=False)
    target_optimizer = torch.optim.SGD(target_model.parameters(), lr=0.1, momentum=0.9)
    model_before = target_model.weight.detach().clone()

    with pytest.raises(CheckpointCompatibilityError, match="optimizer.*shape mismatch"):
        load_checkpoint(
            checkpoint,
            model=target_model,
            optimizer=target_optimizer,
            restore_rng=False,
        )

    assert torch.equal(target_model.weight.detach(), model_before)
    assert target_optimizer.state == {}


def test_sgd_momentum_dtype_corruption_rejects_before_model_mutation(tmp_path: Path) -> None:
    checkpoint, combined_state = _saved_sgd_checkpoint(tmp_path)
    parameter_state, momentum = _first_momentum_buffer(combined_state)

    if isinstance(momentum, torch.Tensor):
        corrupt_momentum = momentum.to(dtype=torch.float64)
    elif isinstance(momentum, np.ndarray):
        corrupt_momentum = momentum.astype(np.float64)
    else:  # pragma: no cover - fail loudly if the state-tree contract changes.
        raise AssertionError(f"unexpected momentum_buffer type: {type(momentum)!r}")
    parameter_state["momentum_buffer"] = corrupt_momentum
    _replace_combined_state(checkpoint, combined_state)

    target_model = torch.nn.Linear(4, 4, bias=False)
    target_optimizer = torch.optim.SGD(target_model.parameters(), lr=0.1, momentum=0.9)
    model_before = target_model.weight.detach().clone()

    with pytest.raises(CheckpointCompatibilityError, match="optimizer.*dtype mismatch"):
        load_checkpoint(
            checkpoint,
            model=target_model,
            optimizer=target_optimizer,
            restore_rng=False,
        )

    assert torch.equal(target_model.weight.detach(), model_before)
    assert target_optimizer.state == {}
