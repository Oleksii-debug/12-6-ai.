from __future__ import annotations

import copy
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


class Model:
    def __init__(self, value: list[float], *, dtype: np.dtype = np.dtype(np.float64)) -> None:
        self.weights = np.asarray(value, dtype=dtype).copy()
        self.loads = 0

    def state_dict(self) -> dict[str, np.ndarray]:
        return {"weights": self.weights.copy()}

    def load_state_dict(self, state: dict[str, np.ndarray], strict: bool = True) -> None:
        assert not strict or set(state) == {"weights"}
        self.loads += 1
        self.weights = state["weights"].copy()


def identity(
    *,
    parameter_count: int = 3,
    precision: str = "float64",
    step: int = 2,
    tokens_seen: int = 6,
) -> CheckpointIdentity:
    return CheckpointIdentity(
        git_sha="a" * 40,
        model_spec={"kind": "test", "width": parameter_count},
        parameter_count=parameter_count,
        tokenizer_hash="b" * 64,
        tokenizer_vocab_hash="c" * 64,
        dataset_manifest_hash="d" * 64,
        run_manifest_hash="e" * 64,
        training_config={"steps": 4},
        seed=7,
        precision=precision,
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


@pytest.mark.parametrize("field", ["step", "tokens_seen"])
def test_negative_resume_counters_fail_integrity_even_after_rebinding(
    tmp_path: Path, field: str
) -> None:
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(checkpoint, model=Model([1, 2, 3]), identity=identity())

    manifest_path = checkpoint / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["identity"][field] = -1
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    _rewrite_manifest(checkpoint)

    with pytest.raises(CheckpointIntegrityError, match="scalar invariant"):
        verify_checkpoint(checkpoint)


def test_model_dtype_corruption_fails_before_target_mutation(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(checkpoint, model=Model([1, 2, 3]), identity=identity())

    arrays = load_safetensors_bytes((checkpoint / "weights.safetensors").read_bytes())
    arrays["weights"] = arrays["weights"].astype(np.float32)
    (checkpoint / "weights.safetensors").write_bytes(save_safetensors_bytes(arrays))
    _rewrite_payload_record(checkpoint, "weights.safetensors")

    target = Model([9, 9, 9])
    before = target.weights.copy()
    with pytest.raises(CheckpointCompatibilityError, match="dtype mismatch"):
        load_checkpoint(checkpoint, model=target, restore_rng=False)

    np.testing.assert_array_equal(target.weights, before)
    assert target.loads == 0


def test_expected_resume_counters_are_bound_before_target_mutation(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(checkpoint, model=Model([1, 2, 3]), identity=identity())

    target = Model([9, 9, 9])
    before = target.weights.copy()
    with pytest.raises(CheckpointCompatibilityError, match="resume identity mismatch"):
        load_checkpoint(
            checkpoint,
            model=target,
            restore_rng=False,
            expected_step=3,
            expected_tokens_seen=6,
        )

    np.testing.assert_array_equal(target.weights, before)
    assert target.loads == 0


def test_manifest_parameter_count_must_match_strict_target(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(checkpoint, model=Model([1, 2, 3]), identity=identity())

    manifest_path = checkpoint / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["identity"]["parameter_count"] = 4
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    _rewrite_manifest(checkpoint)

    target = Model([9, 9, 9])
    before = target.weights.copy()
    with pytest.raises(CheckpointCompatibilityError, match="parameter_count"):
        load_checkpoint(checkpoint, model=target, restore_rng=False)

    np.testing.assert_array_equal(target.weights, before)
    assert target.loads == 0


def test_wrong_shaped_sgd_momentum_fails_before_model_or_optimizer_mutation(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")

    source = torch.nn.Linear(3, 1, bias=False)
    source_optimizer = torch.optim.SGD(source.parameters(), lr=0.1, momentum=0.9)
    source_optimizer.zero_grad(set_to_none=True)
    source(torch.ones(1, 3)).sum().backward()
    source_optimizer.step()

    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(
        checkpoint,
        model=source,
        optimizer=source_optimizer,
        identity=identity(parameter_count=3, precision="float32"),
    )

    tree = json.loads((checkpoint / "state.json").read_text(encoding="utf-8"))
    tensors = load_safetensors_bytes((checkpoint / "state.safetensors").read_bytes())
    combined = copy.deepcopy(unpack_state_tree(tree, tensors))
    state_entries = combined["optimizer"]["state"]
    assert len(state_entries) == 1
    param_id = next(iter(state_entries))
    momentum = state_entries[param_id]["momentum_buffer"]
    state_entries[param_id]["momentum_buffer"] = torch.zeros(
        (2,), dtype=momentum.dtype
    )
    packed = pack_state_tree(combined)
    (checkpoint / "state.safetensors").write_bytes(
        save_safetensors_bytes(packed.tensors)
    )
    (checkpoint / "state.json").write_text(
        json.dumps(packed.tree, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    _rewrite_payload_record(checkpoint, "state.safetensors")
    _rewrite_payload_record(checkpoint, "state.json")

    target = torch.nn.Linear(3, 1, bias=False)
    target_optimizer = torch.optim.SGD(target.parameters(), lr=0.1, momentum=0.9)
    before = [parameter.detach().clone() for parameter in target.parameters()]
    assert not target_optimizer.state

    with pytest.raises(CheckpointCompatibilityError, match="tensor shape"):
        load_checkpoint(
            checkpoint,
            model=target,
            optimizer=target_optimizer,
            restore_rng=False,
        )

    for parameter, original in zip(target.parameters(), before, strict=True):
        assert torch.equal(parameter.detach(), original)
    assert not target_optimizer.state


def test_wrong_dtype_sgd_momentum_fails_before_model_mutation(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")

    source = torch.nn.Linear(3, 1, bias=False, dtype=torch.float32)
    source_optimizer = torch.optim.SGD(source.parameters(), lr=0.1, momentum=0.9)
    source_optimizer.zero_grad(set_to_none=True)
    source(torch.ones(1, 3, dtype=torch.float32)).sum().backward()
    source_optimizer.step()

    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(
        checkpoint,
        model=source,
        optimizer=source_optimizer,
        identity=identity(parameter_count=3, precision="float32"),
    )

    tree = json.loads((checkpoint / "state.json").read_text(encoding="utf-8"))
    tensors = load_safetensors_bytes((checkpoint / "state.safetensors").read_bytes())
    combined = copy.deepcopy(unpack_state_tree(tree, tensors))
    state_entries = combined["optimizer"]["state"]
    param_id = next(iter(state_entries))
    momentum = state_entries[param_id]["momentum_buffer"]
    state_entries[param_id]["momentum_buffer"] = momentum.to(torch.float64)
    packed = pack_state_tree(combined)
    (checkpoint / "state.safetensors").write_bytes(
        save_safetensors_bytes(packed.tensors)
    )
    (checkpoint / "state.json").write_text(
        json.dumps(packed.tree, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    _rewrite_payload_record(checkpoint, "state.safetensors")
    _rewrite_payload_record(checkpoint, "state.json")

    target = torch.nn.Linear(3, 1, bias=False, dtype=torch.float32)
    target_optimizer = torch.optim.SGD(target.parameters(), lr=0.1, momentum=0.9)
    before = [parameter.detach().clone() for parameter in target.parameters()]

    with pytest.raises(CheckpointCompatibilityError, match="tensor dtype"):
        load_checkpoint(
            checkpoint,
            model=target,
            optimizer=target_optimizer,
            restore_rng=False,
        )

    for parameter, original in zip(target.parameters(), before, strict=True):
        assert torch.equal(parameter.detach(), original)
    assert not target_optimizer.state
