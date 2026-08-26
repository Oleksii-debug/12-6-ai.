from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import load as load_safetensors_bytes
from safetensors.numpy import save as save_safetensors_bytes

torch = pytest.importorskip("torch")

from twelve_six.checkpoint import (
    CheckpointCompatibilityError,
    CheckpointIdentity,
    CheckpointIntegrityError,
    hash_json,
    load_checkpoint,
    save_checkpoint,
)
from twelve_six.checkpoint.state_tree import pack_state_tree, unpack_state_tree


def _identity(*, step: int = 1, tokens_seen: int = 4) -> CheckpointIdentity:
    return CheckpointIdentity(
        git_sha="a" * 40,
        model_spec={"kind": "linear-test", "in": 4, "out": 3},
        parameter_count=15,
        tokenizer_hash="b" * 64,
        tokenizer_vocab_hash="c" * 64,
        dataset_manifest_hash="d" * 64,
        run_manifest_hash="e" * 64,
        training_config={"steps": 2},
        seed=7,
        precision="float32",
        step=step,
        tokens_seen=tokens_seen,
        optimizer={"name": "SGD", "momentum": 0.9},
        scheduler=None,
        environment_lock_hash="f" * 64,
    )


def _write_manifest(checkpoint: Path, manifest: dict) -> None:
    manifest["checkpoint_id"] = hash_json(
        {"identity": manifest["identity"], "files": manifest["files"]}
    )
    manifest_path = checkpoint / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    (checkpoint / "MANIFEST.sha256").write_text(
        f"{digest}  manifest.json\n", encoding="ascii"
    )


def _rewrite_payload_records(checkpoint: Path, *names: str) -> None:
    manifest_path = checkpoint / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name in names:
        payload = (checkpoint / name).read_bytes()
        manifest["files"][name] = {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
        }
    _write_manifest(checkpoint, manifest)


def _clone_model_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def _assert_model_unchanged(
    model: torch.nn.Module, before: dict[str, torch.Tensor]
) -> None:
    after = model.state_dict()
    assert set(after) == set(before)
    for name, value in before.items():
        assert torch.equal(after[name], value), name


def test_dtype_corruption_rejects_before_model_mutation(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    source = torch.nn.Linear(4, 3)
    save_checkpoint(checkpoint, model=source, identity=_identity())

    weights_path = checkpoint / "weights.safetensors"
    arrays = dict(load_safetensors_bytes(weights_path.read_bytes()))
    victim = sorted(arrays)[0]
    assert arrays[victim].dtype == np.float32
    arrays[victim] = arrays[victim].astype(np.float64)
    weights_path.write_bytes(save_safetensors_bytes(arrays))
    _rewrite_payload_records(checkpoint, "weights.safetensors")

    target = torch.nn.Linear(4, 3)
    before = _clone_model_state(target)
    with pytest.raises(CheckpointCompatibilityError, match="dtype mismatch"):
        load_checkpoint(checkpoint, model=target, restore_rng=False)

    _assert_model_unchanged(target, before)


def test_wrong_shaped_sgd_momentum_rejects_before_model_mutation(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    source = torch.nn.Linear(4, 3)
    optimizer = torch.optim.SGD(source.parameters(), lr=0.1, momentum=0.9)
    loss = source(torch.ones(2, 4)).sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    save_checkpoint(
        checkpoint,
        model=source,
        optimizer=optimizer,
        identity=_identity(),
    )

    tree_path = checkpoint / "state.json"
    tensors_path = checkpoint / "state.safetensors"
    tree = json.loads(tree_path.read_text(encoding="utf-8"))
    arrays = load_safetensors_bytes(tensors_path.read_bytes())
    combined = unpack_state_tree(tree, arrays)
    optimizer_state = combined["optimizer"]["state"]
    assert optimizer_state
    first_state = next(iter(optimizer_state.values()))
    momentum = first_state["momentum_buffer"]
    first_state["momentum_buffer"] = torch.zeros(
        1, dtype=momentum.dtype, device=momentum.device
    )

    packed = pack_state_tree(combined)
    tensors_path.write_bytes(save_safetensors_bytes(packed.tensors))
    tree_path.write_text(
        json.dumps(packed.tree, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    _rewrite_payload_records(checkpoint, "state.safetensors", "state.json")

    target = torch.nn.Linear(4, 3)
    target_optimizer = torch.optim.SGD(target.parameters(), lr=0.1, momentum=0.9)
    before = _clone_model_state(target)
    assert target_optimizer.state == {}

    with pytest.raises(CheckpointCompatibilityError, match="optimizer state shape mismatch"):
        load_checkpoint(
            checkpoint,
            model=target,
            optimizer=target_optimizer,
            restore_rng=False,
        )

    _assert_model_unchanged(target, before)
    assert target_optimizer.state == {}


def test_negative_identity_counters_reject_after_consistent_rebind(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    source = torch.nn.Linear(4, 3)
    save_checkpoint(checkpoint, model=source, identity=_identity())

    manifest_path = checkpoint / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["identity"]["step"] = -1
    manifest["identity"]["tokens_seen"] = -32
    _write_manifest(checkpoint, manifest)

    target = torch.nn.Linear(4, 3)
    before = _clone_model_state(target)
    with pytest.raises(CheckpointIntegrityError, match="identity.step"):
        load_checkpoint(checkpoint, model=target, restore_rng=False)

    _assert_model_unchanged(target, before)
