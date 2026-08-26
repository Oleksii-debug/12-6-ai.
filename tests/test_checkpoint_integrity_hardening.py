from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import twelve_six.checkpoint.core as core
from twelve_six.checkpoint import (
    CheckpointCompatibilityError,
    CheckpointIdentity,
    CheckpointIntegrityError,
    hash_json,
    save_checkpoint,
    verify_checkpoint,
)
from twelve_six.checkpoint.integrity_hardening import _optimizer_preflight


class ArrayModel:
    def __init__(self, dtype: np.dtype):
        self.weights = np.array([1.0, -2.0, 3.0], dtype=dtype)

    def state_dict(self):
        return {"weights": self.weights.copy()}

    def load_state_dict(self, state, strict=True):
        self.weights = state["weights"].copy()


def _identity() -> CheckpointIdentity:
    return CheckpointIdentity(
        git_sha="f" * 40,
        model_spec={"kind": "array-test", "width": 3},
        parameter_count=3,
        tokenizer_hash="a" * 64,
        tokenizer_vocab_hash="b" * 64,
        dataset_manifest_hash="c" * 64,
        run_manifest_hash="d" * 64,
        training_config={"batch_size": 1},
        seed=7,
        precision="float64-test",
        step=1,
        tokens_seen=3,
        optimizer={"name": "none"},
        scheduler=None,
        environment_lock_hash="e" * 64,
    )


def test_direct_core_import_installs_hardening() -> None:
    assert core._NEXT100075_HARDENING_INSTALLED is True


def test_model_dtype_mismatch_fails_closed(tmp_path: Path) -> None:
    source = ArrayModel(np.float64)
    checkpoint = tmp_path / "dtype"
    save_checkpoint(checkpoint, model=source, identity=_identity())

    target = ArrayModel(np.float32)
    before = target.weights.copy()
    with pytest.raises(CheckpointCompatibilityError, match="dtype mismatch"):
        core.load_checkpoint(checkpoint, model=target)
    np.testing.assert_array_equal(target.weights, before)


def test_wrong_shaped_sgd_momentum_is_rejected_preflight() -> None:
    torch = pytest.importorskip("torch")
    parameter = torch.nn.Parameter(torch.zeros(3, dtype=torch.float32))
    optimizer = torch.optim.SGD([parameter], lr=0.1, momentum=0.9)
    saved = optimizer.state_dict()
    saved["state"] = {0: {"momentum_buffer": torch.zeros(2, dtype=torch.float32)}}

    with pytest.raises(CheckpointCompatibilityError, match="shape mismatch"):
        _optimizer_preflight(optimizer, saved, core)


def test_negative_resume_counter_rejected_after_manifest_rebind(tmp_path: Path) -> None:
    checkpoint = tmp_path / "counter"
    save_checkpoint(checkpoint, model=ArrayModel(np.float64), identity=_identity())

    manifest_path = checkpoint / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["identity"]["tokens_seen"] = -1
    manifest["checkpoint_id"] = hash_json(
        {"identity": manifest["identity"], "files": manifest["files"]}
    )
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    (checkpoint / "MANIFEST.sha256").write_text(
        f"{digest}  manifest.json\n", encoding="ascii"
    )

    with pytest.raises(CheckpointIntegrityError, match="tokens_seen"):
        verify_checkpoint(checkpoint)
