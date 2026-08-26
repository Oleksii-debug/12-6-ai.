from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

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


def _identity(*, step: int = 1, tokens_seen: int = 8) -> CheckpointIdentity:
    return CheckpointIdentity(
        git_sha="a" * 40,
        model_spec={"kind": "torch-linear", "in_features": 3, "out_features": 2},
        parameter_count=8,
        tokenizer_hash="b" * 64,
        tokenizer_vocab_hash="c" * 64,
        dataset_manifest_hash="d" * 64,
        run_manifest_hash="e" * 64,
        training_config={"batch_size": 1, "max_steps": 2},
        seed=7,
        precision="float32",
        step=step,
        tokens_seen=tokens_seen,
        optimizer={"name": "SGD", "lr": 0.1, "momentum": 0.9},
        scheduler=None,
        environment_lock_hash="f" * 64,
    )


def _rebind_manifest(checkpoint: Path) -> None:
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
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    (checkpoint / "MANIFEST.sha256").write_text(
        f"{manifest_sha}  manifest.json\n", encoding="ascii"
    )


def test_model_dtype_mismatch_fails_before_target_mutation(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    source = torch.nn.Linear(3, 2, dtype=torch.float32)
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(checkpoint, model=source, identity=_identity())

    target = torch.nn.Linear(3, 2, dtype=torch.float64)
    with torch.no_grad():
        for parameter in target.parameters():
            parameter.fill_(7.0)
    before = {name: tensor.detach().clone() for name, tensor in target.state_dict().items()}

    with pytest.raises(CheckpointCompatibilityError, match="dtype mismatch"):
        load_checkpoint(checkpoint, model=target, restore_rng=False)

    for name, tensor in target.state_dict().items():
        torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)


def test_wrong_shaped_sgd_momentum_fails_before_model_mutation(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    source = torch.nn.Linear(3, 2)
    optimizer = torch.optim.SGD(source.parameters(), lr=0.1, momentum=0.9)
    loss = source(torch.randn(4, 3)).pow(2).mean()
    loss.backward()
    optimizer.step()

    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(
        checkpoint,
        model=source,
        optimizer=optimizer,
        identity=_identity(),
    )

    tree = json.loads((checkpoint / "state.json").read_text(encoding="utf-8"))
    arrays = load_safetensors_bytes((checkpoint / "state.safetensors").read_bytes())
    combined = copy.deepcopy(unpack_state_tree(tree, arrays))
    first_parameter_state = next(iter(combined["optimizer"]["state"].values()))
    original_buffer = first_parameter_state["momentum_buffer"]
    first_parameter_state["momentum_buffer"] = torch.zeros(
        1, dtype=original_buffer.dtype
    )
    packed = pack_state_tree(combined)
    (checkpoint / "state.safetensors").write_bytes(save_safetensors_bytes(packed.tensors))
    (checkpoint / "state.json").write_text(
        json.dumps(packed.tree, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    _rebind_manifest(checkpoint)

    target = torch.nn.Linear(3, 2)
    target_optimizer = torch.optim.SGD(target.parameters(), lr=0.1, momentum=0.9)
    with torch.no_grad():
        for parameter in target.parameters():
            parameter.fill_(11.0)
    before = {name: tensor.detach().clone() for name, tensor in target.state_dict().items()}

    with pytest.raises(CheckpointCompatibilityError, match="optimizer tensor shape mismatch"):
        load_checkpoint(
            checkpoint,
            model=target,
            optimizer=target_optimizer,
            restore_rng=False,
        )

    for name, tensor in target.state_dict().items():
        torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)


@pytest.mark.parametrize("field", ["step", "tokens_seen"])
def test_negative_manifest_counters_fail_after_consistent_rebinding(
    tmp_path: Path, field: str
) -> None:
    torch = pytest.importorskip("torch")
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(
        checkpoint,
        model=torch.nn.Linear(3, 2),
        identity=_identity(),
    )

    manifest_path = checkpoint / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["identity"][field] = -1
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
