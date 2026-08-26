from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from twelve_six.checkpoint import (
    CheckpointCompatibilityError,
    CheckpointIdentity,
    CheckpointIntegrityError,
    hash_json,
    load_checkpoint,
    save_checkpoint,
    verify_checkpoint,
)
from twelve_six.checkpoint.hardening import preflight_optimizer_state


class NumpyModel:
    def __init__(self, values: np.ndarray, *, dtype: np.dtype):
        self.weights = np.asarray(values, dtype=dtype).copy()

    def state_dict(self):
        return {"weights": self.weights.copy()}

    def load_state_dict(self, state, strict=True):
        assert not strict or set(state) == {"weights"}
        self.weights = state["weights"].copy()


def identity(*, step: int = 0, tokens_seen: int = 0) -> CheckpointIdentity:
    return CheckpointIdentity(
        git_sha="a" * 40,
        model_spec={"kind": "hardening-test", "width": 3},
        parameter_count=3,
        tokenizer_hash="b" * 64,
        tokenizer_vocab_hash="c" * 64,
        dataset_manifest_hash="d" * 64,
        run_manifest_hash="e" * 64,
        training_config={"batch_size": 1, "max_steps": 8},
        seed=7,
        precision="float64-test",
        step=step,
        tokens_seen=tokens_seen,
        optimizer={"name": "none-for-model-only-test"},
        scheduler=None,
    )


def _rewrite_manifest(path: Path, manifest: dict) -> None:
    manifest["checkpoint_id"] = hash_json(
        {"identity": manifest["identity"], "files": manifest["files"]}
    )
    payload = json.dumps(
        manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8") + b"\n"
    path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    (path.parent / "MANIFEST.sha256").write_text(
        f"{digest}  manifest.json\n", encoding="ascii"
    )


def test_public_load_rejects_model_dtype_mismatch_before_mutation(tmp_path: Path) -> None:
    source = NumpyModel(np.array([0.1, -0.2, 0.3]), dtype=np.float64)
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(checkpoint, model=source, identity=identity())

    target = NumpyModel(np.array([9.0, 8.0, 7.0]), dtype=np.float32)
    before = target.weights.copy()
    with pytest.raises(CheckpointCompatibilityError, match="dtype mismatch"):
        load_checkpoint(checkpoint, model=target, restore_rng=False)
    np.testing.assert_array_equal(target.weights, before)
    assert target.weights.dtype == np.float32


@pytest.mark.parametrize(
    ("field", "value"),
    [("step", -1), ("tokens_seen", -32), ("seed", -1), ("parameter_count", 0)],
)
def test_rebound_manifest_invalid_counters_fail_closed(
    tmp_path: Path, field: str, value: int
) -> None:
    model = NumpyModel(np.array([0.1, -0.2, 0.3]), dtype=np.float64)
    checkpoint = tmp_path / f"checkpoint-{field}"
    save_checkpoint(checkpoint, model=model, identity=identity(step=2, tokens_seen=16))

    manifest_path = checkpoint / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["identity"][field] = value
    _rewrite_manifest(manifest_path, manifest)

    with pytest.raises(CheckpointIntegrityError, match=field):
        verify_checkpoint(checkpoint)


def test_torch_optimizer_wrong_momentum_shape_rejected_without_mutation() -> None:
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(4, 3)
    trained_optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    loss = model(torch.randn(2, 4)).pow(2).mean()
    loss.backward()
    trained_optimizer.step()

    corrupted = copy.deepcopy(trained_optimizer.state_dict())
    parameter_id = corrupted["param_groups"][0]["params"][0]
    corrupted["state"][parameter_id]["momentum_buffer"] = torch.zeros(1)

    fresh_optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    assert fresh_optimizer.state_dict()["state"] == {}
    with pytest.raises(CheckpointCompatibilityError, match="shape mismatch"):
        preflight_optimizer_state(fresh_optimizer, corrupted)
    assert fresh_optimizer.state_dict()["state"] == {}


def test_torch_optimizer_wrong_state_dtype_rejected_without_mutation() -> None:
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(4, 3, dtype=torch.float32)
    trained_optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    loss = model(torch.randn(2, 4)).pow(2).mean()
    loss.backward()
    trained_optimizer.step()

    corrupted = copy.deepcopy(trained_optimizer.state_dict())
    parameter_id = corrupted["param_groups"][0]["params"][0]
    corrupted["state"][parameter_id]["momentum_buffer"] = corrupted["state"][parameter_id][
        "momentum_buffer"
    ].to(torch.float64)

    fresh_optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    with pytest.raises(CheckpointCompatibilityError, match="dtype mismatch"):
        preflight_optimizer_state(fresh_optimizer, corrupted)
    assert fresh_optimizer.state_dict()["state"] == {}
