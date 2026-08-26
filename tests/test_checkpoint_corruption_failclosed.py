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


def _identity(*, parameter_count: int, step: int = 1, tokens_seen: int = 8) -> CheckpointIdentity:
    return CheckpointIdentity(
        git_sha="a" * 40,
        model_spec={"kind": "checkpoint-corruption-probe", "parameters": parameter_count},
        parameter_count=parameter_count,
        tokenizer_hash="b" * 64,
        tokenizer_vocab_hash="c" * 64,
        dataset_manifest_hash="d" * 64,
        run_manifest_hash="e" * 64,
        training_config={"probe": True},
        seed=17,
        precision="float64-test",
        step=step,
        tokens_seen=tokens_seen,
        optimizer={"name": "SGD", "lr": 0.01, "momentum": 0.9},
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
    checksum = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    (checkpoint / "MANIFEST.sha256").write_text(
        f"{checksum}  manifest.json\n", encoding="ascii"
    )


def _rewrite_state(checkpoint: Path, state: object) -> None:
    packed = pack_state_tree(state)
    (checkpoint / "state.safetensors").write_bytes(save_safetensors_bytes(packed.tensors))
    (checkpoint / "state.json").write_text(
        json.dumps(packed.tree, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    _rewrite_manifest(checkpoint)


def test_model_dtype_corruption_is_rejected_before_mutation(tmp_path: Path) -> None:
    checkpoint = tmp_path / "dtype-corrupt"
    save_checkpoint(
        checkpoint,
        model=NumpyModel([1.0, 2.0, 3.0]),
        identity=_identity(parameter_count=3),
    )
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


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [("step", -1), ("tokens_seen", -32), ("seed", -1), ("parameter_count", 0)],
)
def test_manifest_scalar_identity_corruption_is_rejected(
    tmp_path: Path, field: str, bad_value: int
) -> None:
    checkpoint = tmp_path / f"identity-{field}"
    save_checkpoint(
        checkpoint,
        model=NumpyModel([1.0, 2.0, 3.0]),
        identity=_identity(parameter_count=3),
    )
    manifest_path = checkpoint / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["identity"][field] = bad_value
    manifest["checkpoint_id"] = hash_json(
        {"identity": manifest["identity"], "files": manifest["files"]}
    )
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    checksum = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    (checkpoint / "MANIFEST.sha256").write_text(
        f"{checksum}  manifest.json\n", encoding="ascii"
    )

    target = NumpyModel([9.0, 9.0, 9.0])
    before = target.weights.copy()
    with pytest.raises(CheckpointIntegrityError, match=field):
        load_checkpoint(checkpoint, model=target, restore_rng=False)

    np.testing.assert_array_equal(target.weights, before)
    assert target.loads == 0


def test_torch_sgd_momentum_shape_corruption_is_rejected_before_any_live_mutation(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    torch.manual_seed(7)
    source = torch.nn.Linear(3, 2)
    source_optimizer = torch.optim.SGD(source.parameters(), lr=0.01, momentum=0.9)
    loss = source(torch.randn(4, 3)).pow(2).mean()
    loss.backward()
    source_optimizer.step()

    checkpoint = tmp_path / "optimizer-corrupt"
    save_checkpoint(
        checkpoint,
        model=source,
        optimizer=source_optimizer,
        identity=_identity(parameter_count=sum(p.numel() for p in source.parameters())),
    )

    tree = json.loads((checkpoint / "state.json").read_text(encoding="utf-8"))
    tensors = load_safetensors_bytes((checkpoint / "state.safetensors").read_bytes())
    state = copy.deepcopy(unpack_state_tree(tree, tensors))
    optimizer_state = state["optimizer"]
    first_parameter_state = next(iter(optimizer_state["state"].values()))
    momentum = first_parameter_state["momentum_buffer"]
    first_parameter_state["momentum_buffer"] = momentum.reshape(-1)[:1].clone()
    _rewrite_state(checkpoint, state)

    torch.manual_seed(99)
    target = torch.nn.Linear(3, 2)
    target_optimizer = torch.optim.SGD(target.parameters(), lr=9.0, momentum=0.9)
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
        torch.testing.assert_close(tensor, before_model[name], rtol=0, atol=0)
    assert target_optimizer.state_dict() == before_optimizer
