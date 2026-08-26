from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
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


_DEFAULT_DTYPE = np.dtype(np.float64)


class Model:
    def __init__(self, values: list[float], *, dtype: np.dtype = _DEFAULT_DTYPE) -> None:
        self.weights = np.asarray(values, dtype=dtype).copy()
        self.loads = 0

    def state_dict(self) -> dict[str, np.ndarray]:
        return {"weights": self.weights.copy()}

    def load_state_dict(self, state: dict[str, np.ndarray], strict: bool = True) -> None:
        assert not strict or set(state) == {"weights"}
        self.loads += 1
        self.weights = state["weights"].copy()


class MomentumSGD:
    def __init__(self, model: Model) -> None:
        self.model = model
        self.lr = 0.03
        self.momentum = 0.8
        self.velocity = np.zeros_like(model.weights)
        self.loads = 0

    def state_dict(self) -> dict[str, object]:
        return {
            "lr": self.lr,
            "momentum": self.momentum,
            "velocity": self.velocity.copy(),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.loads += 1
        self.lr = float(state["lr"])
        self.momentum = float(state["momentum"])
        self.velocity = np.asarray(state["velocity"]).copy()


def identity(*, precision: str = "float64-test") -> CheckpointIdentity:
    return CheckpointIdentity(
        git_sha="a" * 40,
        model_spec={"kind": "d05-corruption-test", "width": 3},
        parameter_count=3,
        tokenizer_hash="b" * 64,
        tokenizer_vocab_hash="c" * 64,
        dataset_manifest_hash="d" * 64,
        run_manifest_hash="e" * 64,
        training_config={"steps": 8},
        seed=17,
        precision=precision,
        step=3,
        tokens_seen=24,
        optimizer={"name": "MomentumSGD", "lr": 0.03, "momentum": 0.8},
        scheduler=None,
        environment_lock_hash="f" * 64,
    )


def _write_rebound_manifest(checkpoint: Path, manifest: dict[str, object]) -> None:
    manifest_path = checkpoint / "manifest.json"
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


def _rebind_payloads(checkpoint: Path, *names: str) -> None:
    manifest = json.loads((checkpoint / "manifest.json").read_text(encoding="utf-8"))
    for name in names:
        data = (checkpoint / name).read_bytes()
        manifest["files"][name] = {
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
        }
    _write_rebound_manifest(checkpoint, manifest)


def _rewrite_combined_state(checkpoint: Path, mutate) -> None:
    tree = json.loads((checkpoint / "state.json").read_text(encoding="utf-8"))
    arrays = load_safetensors_bytes((checkpoint / "state.safetensors").read_bytes())
    state = unpack_state_tree(tree, arrays)
    mutate(state)
    packed = pack_state_tree(state)
    (checkpoint / "state.safetensors").write_bytes(save_safetensors_bytes(packed.tensors))
    (checkpoint / "state.json").write_text(
        json.dumps(packed.tree, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    _rebind_payloads(checkpoint, "state.safetensors", "state.json")


def test_numpy_dtype_mismatch_rejected_before_model_mutation(tmp_path: Path) -> None:
    checkpoint = tmp_path / "dtype-numpy"
    source = Model([1.0, 2.0, 3.0], dtype=np.dtype(np.float32))
    save_checkpoint(
        checkpoint,
        model=source,
        identity=identity(precision="float32-test"),
    )

    target = Model([9.0, 9.0, 9.0], dtype=np.dtype(np.float64))
    before = target.weights.copy()
    with pytest.raises(CheckpointCompatibilityError, match="dtype mismatch"):
        load_checkpoint(checkpoint, model=target, restore_rng=False)

    np.testing.assert_array_equal(target.weights, before)
    assert target.loads == 0


def test_torch_dtype_mismatch_rejected_before_model_mutation(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    checkpoint = tmp_path / "dtype-torch"
    source = torch.nn.Linear(3, 1, bias=False).float()
    torch_identity = CheckpointIdentity(
        git_sha="a" * 40,
        model_spec={"kind": "torch-linear", "in": 3, "out": 1},
        parameter_count=sum(param.numel() for param in source.parameters()),
        tokenizer_hash="b" * 64,
        tokenizer_vocab_hash="c" * 64,
        dataset_manifest_hash="d" * 64,
        run_manifest_hash="e" * 64,
        training_config={"steps": 1},
        seed=17,
        precision="float32",
        step=1,
        tokens_seen=3,
        optimizer={"name": "none"},
        scheduler=None,
        environment_lock_hash="f" * 64,
    )
    save_checkpoint(checkpoint, model=source, identity=torch_identity)

    target = torch.nn.Linear(3, 1, bias=False).double()
    before = {name: value.detach().clone() for name, value in target.state_dict().items()}
    with pytest.raises(CheckpointCompatibilityError, match="dtype mismatch"):
        load_checkpoint(checkpoint, model=target, restore_rng=False)

    for name, value in target.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_custom_optimizer_wrong_shape_rejected_before_any_live_mutation(tmp_path: Path) -> None:
    checkpoint = tmp_path / "optimizer-custom"
    source_model = Model([1.0, 2.0, 3.0])
    source_optimizer = MomentumSGD(source_model)
    source_optimizer.velocity[:] = [0.1, 0.2, 0.3]
    save_checkpoint(
        checkpoint,
        model=source_model,
        optimizer=source_optimizer,
        identity=identity(),
    )

    def corrupt(state) -> None:
        state["optimizer"]["velocity"] = np.asarray([7.0], dtype=np.float64)

    _rewrite_combined_state(checkpoint, corrupt)

    target_model = Model([9.0, 9.0, 9.0])
    target_optimizer = MomentumSGD(target_model)
    before_weights = target_model.weights.copy()
    before_velocity = target_optimizer.velocity.copy()

    with pytest.raises(CheckpointCompatibilityError, match="shape mismatch"):
        load_checkpoint(
            checkpoint,
            model=target_model,
            optimizer=target_optimizer,
            restore_rng=False,
        )

    np.testing.assert_array_equal(target_model.weights, before_weights)
    np.testing.assert_array_equal(target_optimizer.velocity, before_velocity)
    assert target_model.loads == 0
    assert target_optimizer.loads == 0


def test_torch_sgd_wrong_momentum_shape_rejected_before_model_mutation(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    checkpoint = tmp_path / "optimizer-torch"
    source_model = torch.nn.Linear(3, 1, bias=False)
    source_optimizer = torch.optim.SGD(source_model.parameters(), lr=0.1, momentum=0.9)
    source_optimizer.zero_grad(set_to_none=True)
    source_model(torch.ones(2, 3)).sum().backward()
    source_optimizer.step()

    torch_identity = CheckpointIdentity(
        git_sha="a" * 40,
        model_spec={"kind": "torch-linear", "in": 3, "out": 1},
        parameter_count=sum(param.numel() for param in source_model.parameters()),
        tokenizer_hash="b" * 64,
        tokenizer_vocab_hash="c" * 64,
        dataset_manifest_hash="d" * 64,
        run_manifest_hash="e" * 64,
        training_config={"steps": 1},
        seed=17,
        precision="float32",
        step=1,
        tokens_seen=6,
        optimizer={"name": "SGD", "lr": 0.1, "momentum": 0.9},
        scheduler=None,
        environment_lock_hash="f" * 64,
    )
    save_checkpoint(
        checkpoint,
        model=source_model,
        optimizer=source_optimizer,
        identity=torch_identity,
    )

    def corrupt(state) -> None:
        optimizer_state = state["optimizer"]["state"]
        parameter_id = next(iter(optimizer_state))
        momentum = optimizer_state[parameter_id]["momentum_buffer"]
        optimizer_state[parameter_id]["momentum_buffer"] = torch.zeros(
            (1,), dtype=momentum.dtype
        )

    _rewrite_combined_state(checkpoint, corrupt)

    target_model = torch.nn.Linear(3, 1, bias=False)
    target_optimizer = torch.optim.SGD(target_model.parameters(), lr=0.1, momentum=0.9)
    before = {name: value.detach().clone() for name, value in target_model.state_dict().items()}

    with pytest.raises(CheckpointCompatibilityError, match="shape mismatch"):
        load_checkpoint(
            checkpoint,
            model=target_model,
            optimizer=target_optimizer,
            restore_rng=False,
        )

    for name, value in target_model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    assert target_optimizer.state == {}


@pytest.mark.parametrize("field", ["step", "tokens_seen"])
def test_negative_resume_counter_rejected_even_when_manifest_is_rebound(
    tmp_path: Path, field: str
) -> None:
    checkpoint = tmp_path / f"counter-{field}"
    save_checkpoint(checkpoint, model=Model([1.0, 2.0, 3.0]), identity=identity())

    manifest = json.loads((checkpoint / "manifest.json").read_text(encoding="utf-8"))
    manifest["identity"][field] = -1
    _write_rebound_manifest(checkpoint, manifest)

    with pytest.raises(CheckpointIntegrityError, match=rf"identity\.{field}"):
        verify_checkpoint(checkpoint)
