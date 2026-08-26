from __future__ import annotations

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


class NumpyModel:
    def __init__(self, value: list[float], *, dtype: np.dtype[np.floating] = np.dtype("float64")) -> None:
        self.weights = np.asarray(value, dtype=dtype).copy()
        self.loads = 0

    def state_dict(self) -> dict[str, np.ndarray]:
        return {"weights": self.weights.copy()}

    def load_state_dict(self, state: dict[str, np.ndarray], strict: bool = True) -> None:
        assert not strict or set(state) == {"weights"}
        self.loads += 1
        self.weights = state["weights"].copy()


def identity(*, parameter_count: int = 3, optimizer: str = "none") -> CheckpointIdentity:
    return CheckpointIdentity(
        git_sha="a" * 40,
        model_spec={"kind": "corruption-test", "parameter_count": parameter_count},
        parameter_count=parameter_count,
        tokenizer_hash="b" * 64,
        tokenizer_vocab_hash="c" * 64,
        dataset_manifest_hash="d" * 64,
        run_manifest_hash="e" * 64,
        training_config={"steps": 2},
        seed=7,
        precision="float32" if optimizer != "none" else "float64",
        step=1,
        tokens_seen=3,
        optimizer={"name": optimizer},
        scheduler=None,
        environment_lock_hash="f" * 64,
    )


def _write_manifest(checkpoint: Path, manifest: dict[str, object]) -> None:
    manifest_path = checkpoint / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    (checkpoint / "MANIFEST.sha256").write_text(
        f"{manifest_sha}  manifest.json\n", encoding="ascii"
    )


def _rebind_payloads(checkpoint: Path, *names: str) -> None:
    manifest_path = checkpoint / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name in names:
        data = (checkpoint / name).read_bytes()
        manifest["files"][name] = {
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
        }
    manifest["checkpoint_id"] = hash_json(
        {"identity": manifest["identity"], "files": manifest["files"]}
    )
    _write_manifest(checkpoint, manifest)


def _rebind_identity(checkpoint: Path, field: str, value: object) -> None:
    manifest_path = checkpoint / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["identity"][field] = value
    manifest["checkpoint_id"] = hash_json(
        {"identity": manifest["identity"], "files": manifest["files"]}
    )
    _write_manifest(checkpoint, manifest)


def test_model_dtype_mismatch_fails_before_model_mutation(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(
        checkpoint,
        model=NumpyModel([1, 2, 3], dtype=np.dtype("float64")),
        identity=identity(),
    )
    arrays = load_safetensors_bytes((checkpoint / "weights.safetensors").read_bytes())
    arrays["weights"] = arrays["weights"].astype(np.float32)
    (checkpoint / "weights.safetensors").write_bytes(save_safetensors_bytes(arrays))
    _rebind_payloads(checkpoint, "weights.safetensors")

    target = NumpyModel([9, 9, 9], dtype=np.dtype("float64"))
    before = target.weights.copy()
    with pytest.raises(CheckpointCompatibilityError, match="dtype mismatch"):
        load_checkpoint(checkpoint, model=target, restore_rng=False)

    np.testing.assert_array_equal(target.weights, before)
    assert target.loads == 0


@pytest.mark.parametrize("field", ["step", "tokens_seen"])
def test_negative_manifest_counter_fails_before_model_mutation(
    tmp_path: Path, field: str
) -> None:
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(checkpoint, model=NumpyModel([1, 2, 3]), identity=identity())
    _rebind_identity(checkpoint, field, -1)

    target = NumpyModel([9, 9, 9])
    before = target.weights.copy()
    with pytest.raises(CheckpointIntegrityError, match=field):
        load_checkpoint(checkpoint, model=target, restore_rng=False)

    np.testing.assert_array_equal(target.weights, before)
    assert target.loads == 0


def test_wrong_shaped_sgd_momentum_fails_before_model_mutation(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")

    source = torch.nn.Linear(2, 1, bias=False)
    source_optimizer = torch.optim.SGD(source.parameters(), lr=0.1, momentum=0.9)
    loss = source(torch.ones(1, 2)).sum()
    loss.backward()
    source_optimizer.step()

    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(
        checkpoint,
        model=source,
        optimizer=source_optimizer,
        identity=identity(parameter_count=2, optimizer="sgd"),
    )

    tree = json.loads((checkpoint / "state.json").read_text(encoding="utf-8"))
    arrays = load_safetensors_bytes((checkpoint / "state.safetensors").read_bytes())
    combined_state = unpack_state_tree(tree, arrays)
    optimizer_state = combined_state["optimizer"]
    parameter_id = next(iter(optimizer_state["state"]))
    optimizer_state["state"][parameter_id]["momentum_buffer"] = torch.zeros(3)
    packed = pack_state_tree(combined_state)
    (checkpoint / "state.safetensors").write_bytes(save_safetensors_bytes(packed.tensors))
    (checkpoint / "state.json").write_text(
        json.dumps(packed.tree, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    _rebind_payloads(checkpoint, "state.safetensors", "state.json")

    target = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        target.weight.fill_(9.0)
    target_optimizer = torch.optim.SGD(target.parameters(), lr=0.1, momentum=0.9)
    before = target.weight.detach().clone()

    with pytest.raises(CheckpointCompatibilityError, match="optimizer state shape mismatch"):
        load_checkpoint(
            checkpoint,
            model=target,
            optimizer=target_optimizer,
            restore_rng=False,
        )

    assert torch.equal(target.weight.detach(), before)
    assert target_optimizer.state == {}
