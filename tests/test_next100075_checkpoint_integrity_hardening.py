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
    def __init__(self, value: list[float], *, dtype: np.dtype = np.dtype("float64")) -> None:
        self.weights = np.asarray(value, dtype=dtype).copy()
        self.loads = 0

    def state_dict(self) -> dict[str, np.ndarray]:
        return {"weights": self.weights.copy()}

    def load_state_dict(self, state: dict[str, np.ndarray], strict: bool = True) -> None:
        assert not strict or set(state) == {"weights"}
        self.loads += 1
        self.weights = state["weights"].copy()


def _identity(*, parameter_count: int = 3, step: int = 1, tokens_seen: int = 3) -> CheckpointIdentity:
    return CheckpointIdentity(
        git_sha="a" * 40,
        model_spec={"kind": "next100075-regression"},
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
        optimizer={"name": "SGD", "lr": 0.1, "momentum": 0.9},
        scheduler=None,
        environment_lock_hash="f" * 64,
    )


def _rewrite_manifest(checkpoint: Path, manifest: dict[str, object]) -> None:
    manifest["checkpoint_id"] = hash_json(
        {"identity": manifest["identity"], "files": manifest["files"]}
    )
    payload = json.dumps(
        manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8") + b"\n"
    (checkpoint / "manifest.json").write_bytes(payload)
    manifest_sha = hashlib.sha256(payload).hexdigest()
    (checkpoint / "MANIFEST.sha256").write_text(
        f"{manifest_sha}  manifest.json\n", encoding="ascii"
    )


def _rewrite_state_payloads(checkpoint: Path, combined_state: dict[str, object]) -> None:
    packed = pack_state_tree(combined_state)
    state_tensor_bytes = save_safetensors_bytes(packed.tensors)
    state_tree_bytes = (
        json.dumps(packed.tree, sort_keys=True, separators=(",", ":")).encode("utf-8")
        + b"\n"
    )
    (checkpoint / "state.safetensors").write_bytes(state_tensor_bytes)
    (checkpoint / "state.json").write_bytes(state_tree_bytes)

    manifest = json.loads((checkpoint / "manifest.json").read_text(encoding="utf-8"))
    manifest["files"]["state.safetensors"] = {
        "sha256": hashlib.sha256(state_tensor_bytes).hexdigest(),
        "bytes": len(state_tensor_bytes),
    }
    manifest["files"]["state.json"] = {
        "sha256": hashlib.sha256(state_tree_bytes).hexdigest(),
        "bytes": len(state_tree_bytes),
    }
    _rewrite_manifest(checkpoint, manifest)


def test_model_dtype_mismatch_rejects_before_mutation(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    source = NumpyModel([1.0, 2.0, 3.0], dtype=np.dtype("float64"))
    save_checkpoint(checkpoint, model=source, identity=_identity())

    target = NumpyModel([9.0, 9.0, 9.0], dtype=np.dtype("float32"))
    before = target.weights.copy()

    with pytest.raises(CheckpointCompatibilityError, match="dtype mismatch"):
        load_checkpoint(checkpoint, model=target, restore_rng=False)

    np.testing.assert_array_equal(target.weights, before)
    assert target.loads == 0


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [("step", -1), ("tokens_seen", -32), ("step", True), ("tokens_seen", False)],
)
def test_manifest_counters_fail_closed_after_consistent_rehash(
    tmp_path: Path, field: str, bad_value: object
) -> None:
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(checkpoint, model=NumpyModel([1, 2, 3]), identity=_identity())

    manifest = json.loads((checkpoint / "manifest.json").read_text(encoding="utf-8"))
    manifest["identity"][field] = bad_value
    _rewrite_manifest(checkpoint, manifest)

    target = NumpyModel([9, 9, 9])
    before = target.weights.copy()
    with pytest.raises(CheckpointIntegrityError, match="non-negative integers"):
        load_checkpoint(checkpoint, model=target, restore_rng=False)

    np.testing.assert_array_equal(target.weights, before)
    assert target.loads == 0


def test_wrong_shaped_sgd_momentum_rejects_before_model_or_optimizer_mutation(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")

    source_model = torch.nn.Linear(2, 2)
    source_optimizer = torch.optim.SGD(
        source_model.parameters(), lr=0.1, momentum=0.9
    )
    source_optimizer.zero_grad(set_to_none=True)
    source_model(torch.ones(1, 2)).sum().backward()
    source_optimizer.step()

    parameter_count = sum(parameter.numel() for parameter in source_model.parameters())
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(
        checkpoint,
        model=source_model,
        identity=_identity(parameter_count=parameter_count),
        optimizer=source_optimizer,
    )

    tree = json.loads((checkpoint / "state.json").read_text(encoding="utf-8"))
    arrays = load_safetensors_bytes((checkpoint / "state.safetensors").read_bytes())
    combined_state = copy.deepcopy(unpack_state_tree(tree, arrays))
    optimizer_state = combined_state["optimizer"]["state"]
    first_parameter_id = next(iter(optimizer_state))
    optimizer_state[first_parameter_id]["momentum_buffer"] = torch.ones(1)
    _rewrite_state_payloads(checkpoint, combined_state)

    target_model = torch.nn.Linear(2, 2)
    target_optimizer = torch.optim.SGD(
        target_model.parameters(), lr=0.1, momentum=0.9
    )
    before_weights = {
        name: tensor.detach().clone() for name, tensor in target_model.state_dict().items()
    }
    before_optimizer_state = copy.deepcopy(target_optimizer.state_dict())

    with pytest.raises(CheckpointCompatibilityError, match="tensor shape mismatch"):
        load_checkpoint(
            checkpoint,
            model=target_model,
            optimizer=target_optimizer,
            restore_rng=False,
        )

    for name, tensor in target_model.state_dict().items():
        torch.testing.assert_close(tensor, before_weights[name], rtol=0, atol=0)
    assert target_optimizer.state_dict() == before_optimizer_state


def test_scheduler_load_failure_is_preflighted_before_model_mutation(tmp_path: Path) -> None:
    class Scheduler:
        def __init__(self, *, reject: bool = False) -> None:
            self.reject = reject
            self.loaded = False

        def state_dict(self) -> dict[str, object]:
            return {"version": 1}

        def load_state_dict(self, state: dict[str, object]) -> None:
            if self.reject:
                raise ValueError("synthetic incompatible scheduler state")
            self.loaded = True

    checkpoint = tmp_path / "checkpoint"
    source_scheduler = Scheduler()
    save_checkpoint(
        checkpoint,
        model=NumpyModel([1, 2, 3]),
        identity=_identity(),
        scheduler=source_scheduler,
    )

    target = NumpyModel([9, 9, 9])
    scheduler = Scheduler(reject=True)
    before = target.weights.copy()

    with pytest.raises(CheckpointCompatibilityError, match="scheduler state is incompatible"):
        load_checkpoint(
            checkpoint,
            model=target,
            scheduler=scheduler,
            restore_rng=False,
        )

    np.testing.assert_array_equal(target.weights, before)
    assert target.loads == 0
    assert scheduler.loaded is False
