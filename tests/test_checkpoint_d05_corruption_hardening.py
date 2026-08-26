from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from safetensors.numpy import load_file as load_safetensors_file
from safetensors.numpy import save_file as save_safetensors_file

from twelve_six.checkpoint import (
    CheckpointCompatibilityError,
    CheckpointIdentity,
    CheckpointIntegrityError,
    hash_json,
    load_checkpoint,
    save_checkpoint,
    verify_checkpoint,
)
from twelve_six.checkpoint.core import (
    MANIFEST_CHECKSUM_NAME,
    MANIFEST_NAME,
    STATE_TENSORS_NAME,
    STATE_TREE_NAME,
    WEIGHTS_NAME,
    canonical_json_bytes,
)


def _identity(*, step: int = 1, tokens_seen: int = 8, optimizer: dict[str, Any] | None = None):
    return CheckpointIdentity(
        git_sha="a" * 40,
        model_spec={"kind": "d05-corruption-test", "in": 4, "out": 3},
        parameter_count=15,
        tokenizer_hash="b" * 64,
        tokenizer_vocab_hash="c" * 64,
        dataset_manifest_hash="d" * 64,
        run_manifest_hash="e" * 64,
        training_config={"batch_size": 2, "max_steps": 4},
        seed=1337,
        precision="float32",
        step=step,
        tokens_seen=tokens_seen,
        optimizer=optimizer or {"name": "none"},
        scheduler=None,
        environment_lock_hash="f" * 64,
    )


def _rewrite_manifest(checkpoint: Path, mutate) -> None:
    manifest_path = checkpoint / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mutate(manifest)
    manifest["checkpoint_id"] = hash_json(
        {"identity": manifest["identity"], "files": manifest["files"]}
    )
    payload = canonical_json_bytes(manifest) + b"\n"
    manifest_path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    (checkpoint / MANIFEST_CHECKSUM_NAME).write_text(
        f"{digest}  {MANIFEST_NAME}\n", encoding="ascii"
    )


def _rebind_payload(manifest: dict[str, Any], checkpoint: Path, name: str) -> None:
    payload = checkpoint / name
    manifest["files"][name] = {
        "sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
        "bytes": payload.stat().st_size,
    }


def _state_tensor_key_for_named_slot(tree: Any, slot_name: str) -> str:
    if isinstance(tree, dict):
        if tree.get("__kind__") == "mapping":
            for key, value in tree.get("items", []):
                if (
                    key == slot_name
                    and isinstance(value, dict)
                    and value.get("__kind__") == "tensor"
                ):
                    return value["key"]
                try:
                    return _state_tensor_key_for_named_slot(value, slot_name)
                except KeyError:
                    pass
        for value in tree.values():
            try:
                return _state_tensor_key_for_named_slot(value, slot_name)
            except KeyError:
                pass
    elif isinstance(tree, list):
        for value in tree:
            try:
                return _state_tensor_key_for_named_slot(value, slot_name)
            except KeyError:
                pass
    raise KeyError(slot_name)


def _assert_model_unchanged(model, before) -> None:
    torch = pytest.importorskip("torch")
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)


def test_dtype_corruption_rejects_before_model_mutation(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    torch.manual_seed(11)
    source = torch.nn.Linear(4, 3)
    checkpoint = tmp_path / "dtype"
    save_checkpoint(checkpoint, model=source, identity=_identity())

    weights_path = checkpoint / WEIGHTS_NAME
    arrays = load_safetensors_file(str(weights_path))
    name = next(iter(arrays))
    arrays[name] = arrays[name].astype(np.float64)
    save_safetensors_file(arrays, str(weights_path))

    _rewrite_manifest(
        checkpoint,
        lambda manifest: _rebind_payload(manifest, checkpoint, WEIGHTS_NAME),
    )

    target = torch.nn.Linear(4, 3)
    before = {name: tensor.detach().clone() for name, tensor in target.state_dict().items()}

    with pytest.raises(CheckpointCompatibilityError, match="dtype mismatch"):
        load_checkpoint(checkpoint, model=target, restore_rng=False)
    _assert_model_unchanged(target, before)


def test_wrong_shaped_sgd_momentum_rejects_before_any_live_mutation(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    torch.manual_seed(12)
    source = torch.nn.Linear(4, 3)
    source_optimizer = torch.optim.SGD(source.parameters(), lr=0.1, momentum=0.9)
    loss = source(torch.randn(2, 4)).pow(2).mean()
    loss.backward()
    source_optimizer.step()

    checkpoint = tmp_path / "optimizer"
    save_checkpoint(
        checkpoint,
        model=source,
        optimizer=source_optimizer,
        identity=_identity(optimizer={"name": "SGD", "lr": 0.1, "momentum": 0.9}),
    )

    tree = json.loads((checkpoint / STATE_TREE_NAME).read_text(encoding="utf-8"))
    momentum_key = _state_tensor_key_for_named_slot(tree, "momentum_buffer")
    state_path = checkpoint / STATE_TENSORS_NAME
    state_arrays = load_safetensors_file(str(state_path))
    state_arrays[momentum_key] = np.zeros((1,), dtype=state_arrays[momentum_key].dtype)
    save_safetensors_file(state_arrays, str(state_path))

    _rewrite_manifest(
        checkpoint,
        lambda manifest: _rebind_payload(manifest, checkpoint, STATE_TENSORS_NAME),
    )

    target = torch.nn.Linear(4, 3)
    target_optimizer = torch.optim.SGD(target.parameters(), lr=0.1, momentum=0.9)
    before = {name: tensor.detach().clone() for name, tensor in target.state_dict().items()}

    with pytest.raises(CheckpointCompatibilityError, match="shape .* parameter"):
        load_checkpoint(
            checkpoint,
            model=target,
            optimizer=target_optimizer,
            restore_rng=False,
        )
    _assert_model_unchanged(target, before)
    assert target_optimizer.state_dict()["state"] == {}


def test_negative_checkpoint_counters_reject_even_when_manifest_is_rebound(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    checkpoint = tmp_path / "counters"
    save_checkpoint(checkpoint, model=torch.nn.Linear(4, 3), identity=_identity())

    def mutate(manifest: dict[str, Any]) -> None:
        manifest["identity"]["step"] = -1
        manifest["identity"]["tokens_seen"] = -32

    _rewrite_manifest(checkpoint, mutate)

    with pytest.raises(CheckpointIntegrityError, match="step and tokens_seen"):
        verify_checkpoint(checkpoint)
