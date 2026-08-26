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
    verify_checkpoint,
)
from twelve_six.checkpoint.state_tree import pack_state_tree, unpack_state_tree


def _identity(*, parameter_count: int, step: int = 1, tokens_seen: int = 8) -> CheckpointIdentity:
    return CheckpointIdentity(
        git_sha="a" * 40,
        model_spec={"kind": "d05-regression", "parameter_count": parameter_count},
        parameter_count=parameter_count,
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


def _rewrite_manifest(checkpoint: Path, *payload_names: str) -> None:
    manifest_path = checkpoint / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name in payload_names:
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


def _torch_linear(torch):
    return torch.nn.Linear(3, 2, bias=False)


def test_model_dtype_mismatch_fails_before_mutation(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    source = _torch_linear(torch)
    checkpoint = tmp_path / "dtype"
    save_checkpoint(
        checkpoint,
        model=source,
        identity=_identity(parameter_count=sum(p.numel() for p in source.parameters())),
    )

    arrays = load_safetensors_bytes((checkpoint / "weights.safetensors").read_bytes())
    name = next(iter(arrays))
    arrays[name] = arrays[name].astype(np.float64)
    (checkpoint / "weights.safetensors").write_bytes(save_safetensors_bytes(arrays))
    _rewrite_manifest(checkpoint, "weights.safetensors")

    target = _torch_linear(torch)
    before = {name: tensor.detach().clone() for name, tensor in target.state_dict().items()}
    with pytest.raises(CheckpointCompatibilityError, match="dtype mismatch"):
        load_checkpoint(checkpoint, model=target, restore_rng=False)
    for name, tensor in target.state_dict().items():
        torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)


def test_negative_counters_fail_after_consistent_rebinding(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    model = _torch_linear(torch)
    checkpoint = tmp_path / "counter"
    save_checkpoint(
        checkpoint,
        model=model,
        identity=_identity(parameter_count=sum(p.numel() for p in model.parameters())),
    )

    manifest_path = checkpoint / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["identity"]["step"] = -1
    manifest["identity"]["tokens_seen"] = -8
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

    with pytest.raises(CheckpointIntegrityError, match="non-negative integers"):
        verify_checkpoint(checkpoint)


def test_bad_sgd_momentum_shape_fails_before_model_or_optimizer_mutation(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    source = _torch_linear(torch)
    source_optimizer = torch.optim.SGD(source.parameters(), lr=0.03, momentum=0.9)
    loss = source(torch.ones(2, 3)).pow(2).mean()
    loss.backward()
    source_optimizer.step()

    checkpoint = tmp_path / "optimizer"
    save_checkpoint(
        checkpoint,
        model=source,
        optimizer=source_optimizer,
        identity=_identity(parameter_count=sum(p.numel() for p in source.parameters())),
    )

    tree = json.loads((checkpoint / "state.json").read_text(encoding="utf-8"))
    arrays = load_safetensors_bytes((checkpoint / "state.safetensors").read_bytes())
    state = copy.deepcopy(unpack_state_tree(tree, arrays))
    first_parameter_state = next(iter(state["optimizer"]["state"].values()))
    momentum = first_parameter_state["momentum_buffer"]
    first_parameter_state["momentum_buffer"] = torch.zeros(
        1, dtype=momentum.dtype, device=momentum.device
    )
    packed = pack_state_tree(state)
    (checkpoint / "state.safetensors").write_bytes(save_safetensors_bytes(packed.tensors))
    (checkpoint / "state.json").write_text(
        json.dumps(packed.tree, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    _rewrite_manifest(checkpoint, "state.safetensors", "state.json")

    target = _torch_linear(torch)
    target_optimizer = torch.optim.SGD(target.parameters(), lr=0.03, momentum=0.9)
    before = {name: tensor.detach().clone() for name, tensor in target.state_dict().items()}
    before_optimizer = copy.deepcopy(target_optimizer.state_dict())

    with pytest.raises(CheckpointCompatibilityError, match="optimizer state tensor shape mismatch"):
        load_checkpoint(
            checkpoint,
            model=target,
            optimizer=target_optimizer,
            restore_rng=False,
        )

    for name, tensor in target.state_dict().items():
        torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)
    assert target_optimizer.state_dict() == before_optimizer


def test_bfloat16_raw_uint16_checkpoint_representation_remains_supported(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    source = _torch_linear(torch).to(dtype=torch.bfloat16)
    checkpoint = tmp_path / "bf16"
    save_checkpoint(
        checkpoint,
        model=source,
        identity=_identity(parameter_count=sum(p.numel() for p in source.parameters())),
    )

    target = _torch_linear(torch).to(dtype=torch.bfloat16)
    load_checkpoint(checkpoint, model=target, restore_rng=False)
    for name, tensor in target.state_dict().items():
        torch.testing.assert_close(tensor, source.state_dict()[name], rtol=0, atol=0)
