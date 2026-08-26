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
    canonical_json_bytes,
    hash_json,
    load_checkpoint,
    save_checkpoint,
)
from twelve_six.checkpoint.state_tree import pack_state_tree, unpack_state_tree


class NumpyModel:
    def __init__(self, values: list[float], *, dtype: np.dtype = np.dtype("float64")) -> None:
        self.weights = np.asarray(values, dtype=dtype).copy()
        self.loads = 0

    def state_dict(self) -> dict[str, np.ndarray]:
        return {"weights": self.weights.copy()}

    def load_state_dict(self, state: dict[str, np.ndarray], strict: bool = True) -> None:
        assert not strict or set(state) == {"weights"}
        self.loads += 1
        self.weights = state["weights"].copy()


def _identity(*, parameter_count: int = 3, step: int = 1) -> CheckpointIdentity:
    return CheckpointIdentity(
        git_sha="a" * 40,
        model_spec={"kind": "checkpoint-hardening-test"},
        parameter_count=parameter_count,
        tokenizer_hash="b" * 64,
        tokenizer_vocab_hash="c" * 64,
        dataset_manifest_hash="d" * 64,
        run_manifest_hash="e" * 64,
        training_config={"steps": 10},
        seed=7,
        precision="float32",
        step=step,
        tokens_seen=step * 8,
        optimizer={"name": "sgd", "lr": 0.1, "momentum": 0.9},
        scheduler=None,
        environment_lock_hash="f" * 64,
    )


def _rewrite_manifest(checkpoint: Path) -> None:
    manifest_path = checkpoint / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["checkpoint_id"] = hash_json(
        {"identity": manifest["identity"], "files": manifest["files"]}
    )
    manifest_path.write_bytes(canonical_json_bytes(manifest) + b"\n")
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    (checkpoint / "MANIFEST.sha256").write_text(
        f"{digest}  manifest.json\n", encoding="ascii"
    )


def _rebind_payload(checkpoint: Path, name: str) -> None:
    manifest_path = checkpoint / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    data = (checkpoint / name).read_bytes()
    manifest["files"][name] = {
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
    }
    manifest_path.write_bytes(canonical_json_bytes(manifest) + b"\n")
    _rewrite_manifest(checkpoint)


def test_model_dtype_corruption_fails_before_target_mutation(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    source = NumpyModel([1.0, 2.0, 3.0], dtype=np.dtype("float64"))
    save_checkpoint(checkpoint, model=source, identity=_identity())

    arrays = load_safetensors_bytes((checkpoint / "weights.safetensors").read_bytes())
    arrays["weights"] = arrays["weights"].astype(np.float32)
    (checkpoint / "weights.safetensors").write_bytes(save_safetensors_bytes(arrays))
    _rebind_payload(checkpoint, "weights.safetensors")

    target = NumpyModel([9.0, 9.0, 9.0], dtype=np.dtype("float64"))
    before = target.weights.copy()

    with pytest.raises(CheckpointCompatibilityError, match="dtype mismatch"):
        load_checkpoint(checkpoint, model=target, restore_rng=False)

    np.testing.assert_array_equal(target.weights, before)
    assert target.loads == 0


def test_negative_resume_counters_fail_even_after_manifest_rebinding(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(checkpoint, model=NumpyModel([1, 2, 3]), identity=_identity())

    manifest_path = checkpoint / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["identity"]["step"] = -1
    manifest["identity"]["tokens_seen"] = -8
    manifest_path.write_bytes(canonical_json_bytes(manifest) + b"\n")
    _rewrite_manifest(checkpoint)

    target = NumpyModel([9, 9, 9])
    before = target.weights.copy()
    with pytest.raises(CheckpointIntegrityError, match="identity.step"):
        load_checkpoint(checkpoint, model=target, restore_rng=False)

    np.testing.assert_array_equal(target.weights, before)
    assert target.loads == 0


def test_expected_resume_counter_binding_fails_before_target_mutation(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(checkpoint, model=NumpyModel([1, 2, 3]), identity=_identity(step=4))

    target = NumpyModel([9, 9, 9])
    before = target.weights.copy()
    with pytest.raises(CheckpointCompatibilityError, match="step"):
        load_checkpoint(
            checkpoint,
            model=target,
            restore_rng=False,
            expected_step=5,
            expected_tokens_seen=40,
        )

    np.testing.assert_array_equal(target.weights, before)
    assert target.loads == 0


def test_wrong_shaped_sgd_momentum_fails_before_model_or_optimizer_mutation(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")

    checkpoint = tmp_path / "checkpoint"
    source = torch.nn.Linear(3, 2, bias=False)
    source_optimizer = torch.optim.SGD(source.parameters(), lr=0.1, momentum=0.9)
    loss = source(torch.ones(1, 3)).sum()
    loss.backward()
    source_optimizer.step()
    source_optimizer.zero_grad(set_to_none=True)
    save_checkpoint(
        checkpoint,
        model=source,
        optimizer=source_optimizer,
        identity=_identity(parameter_count=6),
    )

    tree = json.loads((checkpoint / "state.json").read_text(encoding="utf-8"))
    arrays = load_safetensors_bytes((checkpoint / "state.safetensors").read_bytes())
    combined = unpack_state_tree(tree, arrays)
    optimizer_slots = combined["optimizer"]["state"]
    first_slot = next(iter(optimizer_slots.values()))
    first_slot["momentum_buffer"] = torch.zeros(1, dtype=source.weight.dtype)
    packed = pack_state_tree(combined)
    (checkpoint / "state.safetensors").write_bytes(save_safetensors_bytes(packed.tensors))
    (checkpoint / "state.json").write_bytes(canonical_json_bytes(packed.tree) + b"\n")
    _rebind_payload(checkpoint, "state.safetensors")
    _rebind_payload(checkpoint, "state.json")

    target = torch.nn.Linear(3, 2, bias=False)
    target_optimizer = torch.optim.SGD(target.parameters(), lr=0.1, momentum=0.9)
    before = target.weight.detach().clone()

    with pytest.raises(CheckpointCompatibilityError, match="optimizer state shape mismatch"):
        load_checkpoint(
            checkpoint,
            model=target,
            optimizer=target_optimizer,
            restore_rng=False,
        )

    torch.testing.assert_close(target.weight, before, rtol=0.0, atol=0.0)
    assert not target_optimizer.state
