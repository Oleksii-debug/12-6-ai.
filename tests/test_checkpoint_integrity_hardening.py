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
    verify_checkpoint,
)
from twelve_six.checkpoint.state_tree import pack_state_tree, unpack_state_tree


class NumpyModel:
    def __init__(self, values: list[float]) -> None:
        self.weights = np.asarray(values, dtype=np.float64).copy()
        self.loads = 0

    def state_dict(self) -> dict[str, np.ndarray]:
        return {"weights": self.weights.copy()}

    def load_state_dict(self, state: dict[str, np.ndarray], strict: bool = True) -> None:
        assert not strict or set(state) == {"weights"}
        self.loads += 1
        self.weights = state["weights"].copy()


def identity(*, step: int = 1, tokens_seen: int = 8) -> CheckpointIdentity:
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


def _rewrite_payload_record(checkpoint: Path, name: str) -> None:
    manifest_path = checkpoint / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload = (checkpoint / name).read_bytes()
    manifest["files"][name] = {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
    }
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    _rewrite_manifest(checkpoint)


def test_checksum_valid_model_dtype_corruption_fails_before_mutation(tmp_path: Path) -> None:
    checkpoint = tmp_path / "dtype-corrupt"
    save_checkpoint(checkpoint, model=NumpyModel([1.0, 2.0, 3.0]), identity=identity())

    arrays = load_safetensors_bytes((checkpoint / "weights.safetensors").read_bytes())
    arrays["weights"] = arrays["weights"].astype(np.float32)
    (checkpoint / "weights.safetensors").write_bytes(save_safetensors_bytes(arrays))
    _rewrite_payload_record(checkpoint, "weights.safetensors")

    target = NumpyModel([9.0, 9.0, 9.0])
    before = target.weights.copy()
    with pytest.raises(CheckpointCompatibilityError, match="dtype mismatch"):
        load_checkpoint(checkpoint, model=target, restore_rng=False)

    np.testing.assert_array_equal(target.weights, before)
    assert target.loads == 0


@pytest.mark.parametrize("field", ["step", "tokens_seen"])
def test_rehashed_negative_progress_counter_is_rejected(
    tmp_path: Path,
    field: str,
) -> None:
    checkpoint = tmp_path / f"negative-{field}"
    save_checkpoint(checkpoint, model=NumpyModel([1.0, 2.0, 3.0]), identity=identity())

    manifest_path = checkpoint / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["identity"][field] = -1
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    _rewrite_manifest(checkpoint)

    with pytest.raises(CheckpointIntegrityError, match=field):
        verify_checkpoint(checkpoint)


def test_wrong_shape_sgd_momentum_fails_before_model_or_optimizer_mutation(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    torch.manual_seed(17)

    source_model = torch.nn.Linear(3, 2)
    source_optimizer = torch.optim.SGD(source_model.parameters(), lr=0.05, momentum=0.9)
    loss = source_model(torch.randn(4, 3)).pow(2).mean()
    loss.backward()
    source_optimizer.step()

    checkpoint = tmp_path / "optimizer-corrupt"
    torch_identity = CheckpointIdentity(
        git_sha="1" * 40,
        model_spec={"kind": "torch-linear-d05", "in": 3, "out": 2},
        parameter_count=sum(parameter.numel() for parameter in source_model.parameters()),
        tokenizer_hash="2" * 64,
        tokenizer_vocab_hash="3" * 64,
        dataset_manifest_hash="4" * 64,
        run_manifest_hash="5" * 64,
        training_config={"steps": 1},
        seed=17,
        precision="float32",
        step=1,
        tokens_seen=12,
        optimizer={"name": "SGD", "lr": 0.05, "momentum": 0.9},
        scheduler=None,
    )
    save_checkpoint(
        checkpoint,
        model=source_model,
        optimizer=source_optimizer,
        identity=torch_identity,
    )

    tree = json.loads((checkpoint / "state.json").read_text(encoding="utf-8"))
    arrays = load_safetensors_bytes((checkpoint / "state.safetensors").read_bytes())
    state = unpack_state_tree(tree, arrays)
    first_parameter_id = next(iter(state["optimizer"]["state"]))
    momentum = state["optimizer"]["state"][first_parameter_id]["momentum_buffer"]
    state["optimizer"]["state"][first_parameter_id]["momentum_buffer"] = torch.zeros(
        momentum.numel() + 1,
        dtype=momentum.dtype,
    )
    packed = pack_state_tree(state)
    (checkpoint / "state.safetensors").write_bytes(save_safetensors_bytes(packed.tensors))
    (checkpoint / "state.json").write_text(
        json.dumps(packed.tree, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    _rewrite_payload_record(checkpoint, "state.safetensors")
    _rewrite_payload_record(checkpoint, "state.json")

    torch.manual_seed(99)
    target_model = torch.nn.Linear(3, 2)
    target_optimizer = torch.optim.SGD(target_model.parameters(), lr=0.05, momentum=0.9)
    before = {
        name: tensor.detach().clone()
        for name, tensor in target_model.state_dict().items()
    }

    with pytest.raises(CheckpointCompatibilityError, match="optimizer.*shape mismatch"):
        load_checkpoint(
            checkpoint,
            model=target_model,
            optimizer=target_optimizer,
            restore_rng=False,
        )

    for name, tensor in target_model.state_dict().items():
        torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)
    assert target_optimizer.state == {}
