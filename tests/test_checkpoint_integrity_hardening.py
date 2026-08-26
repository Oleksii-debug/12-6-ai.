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


class NumpyModel:
    def __init__(self, values: list[float], *, dtype: np.dtype) -> None:
        self.weights = np.asarray(values, dtype=dtype).copy()
        self.loads = 0

    def state_dict(self) -> dict[str, np.ndarray]:
        return {"weights": self.weights.copy()}

    def load_state_dict(self, state: dict[str, np.ndarray], strict: bool = True) -> None:
        assert not strict or set(state) == {"weights"}
        self.loads += 1
        self.weights = state["weights"].copy()


def _identity(*, step: int = 1, tokens_seen: int = 8) -> CheckpointIdentity:
    return CheckpointIdentity(
        git_sha="a" * 40,
        model_spec={"kind": "checkpoint-hardening-test", "width": 3},
        parameter_count=3,
        tokenizer_hash="b" * 64,
        tokenizer_vocab_hash="c" * 64,
        dataset_manifest_hash="d" * 64,
        run_manifest_hash="e" * 64,
        training_config={"steps": 1},
        seed=7,
        precision="float64",
        step=step,
        tokens_seen=tokens_seen,
        optimizer={"name": "test"},
        scheduler=None,
        environment_lock_hash="f" * 64,
    )


def _rewrite_manifest(checkpoint: Path) -> None:
    manifest_path = checkpoint / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
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


def _rewrite_state_payloads(checkpoint: Path, combined_state: dict) -> None:
    packed = pack_state_tree(combined_state)
    (checkpoint / "state.safetensors").write_bytes(save_safetensors_bytes(packed.tensors))
    (checkpoint / "state.json").write_text(
        json.dumps(packed.tree, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    manifest_path = checkpoint / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name in ("state.safetensors", "state.json"):
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


def test_model_dtype_corruption_is_rejected_before_mutation(tmp_path: Path) -> None:
    checkpoint = tmp_path / "dtype"
    source = NumpyModel([1, 2, 3], dtype=np.dtype("float64"))
    save_checkpoint(checkpoint, model=source, identity=_identity())

    target = NumpyModel([9, 9, 9], dtype=np.dtype("float32"))
    before = target.weights.copy()
    with pytest.raises(CheckpointCompatibilityError, match="dtype mismatch"):
        load_checkpoint(checkpoint, model=target, restore_rng=False)

    np.testing.assert_array_equal(target.weights, before)
    assert target.loads == 0


def test_negative_checkpoint_counters_fail_even_after_consistent_rehash(tmp_path: Path) -> None:
    checkpoint = tmp_path / "counters"
    save_checkpoint(
        checkpoint,
        model=NumpyModel([1, 2, 3], dtype=np.dtype("float64")),
        identity=_identity(),
    )
    manifest_path = checkpoint / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["identity"]["step"] = -1
    manifest["identity"]["tokens_seen"] = -32
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    _rewrite_manifest(checkpoint)

    with pytest.raises(CheckpointIntegrityError, match="non-negative integers"):
        verify_checkpoint(checkpoint)


def test_wrong_shaped_sgd_momentum_fails_before_model_mutation(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    torch.manual_seed(11)
    source_model = torch.nn.Linear(3, 2)
    source_optimizer = torch.optim.SGD(source_model.parameters(), lr=0.1, momentum=0.9)
    loss = source_model(torch.randn(4, 3)).pow(2).mean()
    loss.backward()
    source_optimizer.step()

    checkpoint = tmp_path / "optimizer"
    identity = CheckpointIdentity(
        git_sha="1" * 40,
        model_spec={"kind": "torch-linear", "in": 3, "out": 2},
        parameter_count=sum(parameter.numel() for parameter in source_model.parameters()),
        tokenizer_hash="2" * 64,
        tokenizer_vocab_hash="3" * 64,
        dataset_manifest_hash="4" * 64,
        run_manifest_hash="5" * 64,
        training_config={"steps": 1},
        seed=11,
        precision="float32",
        step=1,
        tokens_seen=12,
        optimizer={"name": "SGD", "lr": 0.1, "momentum": 0.9},
        scheduler=None,
    )
    save_checkpoint(
        checkpoint,
        model=source_model,
        optimizer=source_optimizer,
        identity=identity,
    )

    tree = json.loads((checkpoint / "state.json").read_text(encoding="utf-8"))
    arrays = load_safetensors_bytes((checkpoint / "state.safetensors").read_bytes())
    combined_state = copy.deepcopy(unpack_state_tree(tree, arrays))
    optimizer_state = combined_state["optimizer"]["state"]
    first_parameter_id = next(iter(optimizer_state))
    optimizer_state[first_parameter_id]["momentum_buffer"] = torch.zeros(1)
    _rewrite_state_payloads(checkpoint, combined_state)

    torch.manual_seed(99)
    target_model = torch.nn.Linear(3, 2)
    target_optimizer = torch.optim.SGD(target_model.parameters(), lr=0.1, momentum=0.9)
    before = {name: value.detach().clone() for name, value in target_model.state_dict().items()}

    with pytest.raises(CheckpointCompatibilityError, match="optimizer tensor shape mismatch"):
        load_checkpoint(
            checkpoint,
            model=target_model,
            optimizer=target_optimizer,
            restore_rng=False,
        )

    for name, value in target_model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
