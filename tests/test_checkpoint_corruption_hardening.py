from __future__ import annotations

import json

import numpy as np
import pytest

from twelve_six.checkpoint import CheckpointIdentity, CheckpointIntegrityError
from twelve_six.checkpoint import core as checkpoint_core


def _identity(*, step: int = 1, tokens_seen: int = 8) -> CheckpointIdentity:
    return CheckpointIdentity(
        git_sha="a" * 40,
        model_spec={"kind": "corruption-hardening-test", "width": 2},
        parameter_count=4,
        tokenizer_hash="b" * 64,
        tokenizer_vocab_hash="c" * 64,
        dataset_manifest_hash="d" * 64,
        run_manifest_hash="e" * 64,
        training_config={"batch_size": 1, "max_steps": 2},
        seed=7,
        precision="float32",
        step=step,
        tokens_seen=tokens_seen,
        optimizer={"name": "SGD", "lr": 0.1, "momentum": 0.9},
        scheduler=None,
        environment_lock_hash="f" * 64,
    )


def test_numpy_model_dtype_mismatch_is_rejected_without_cast() -> None:
    source = np.ones((2, 2), dtype=np.float64)
    target = np.zeros((2, 2), dtype=np.float32)

    with pytest.raises(checkpoint_core.CheckpointCompatibilityError, match="dtype mismatch"):
        checkpoint_core._materialize_for_target(source, target)

    assert target.dtype == np.float32
    np.testing.assert_array_equal(target, np.zeros((2, 2), dtype=np.float32))


def test_torch_model_dtype_mismatch_is_rejected_without_cast() -> None:
    torch = pytest.importorskip("torch")
    source = np.ones((2, 2), dtype=np.float64)
    target = torch.zeros((2, 2), dtype=torch.float32)

    with pytest.raises(checkpoint_core.CheckpointCompatibilityError, match="dtype mismatch"):
        checkpoint_core._materialize_for_target(source, target)

    torch.testing.assert_close(target, torch.zeros_like(target), rtol=0, atol=0)


def test_bfloat16_uint16_storage_exception_remains_exact() -> None:
    torch = pytest.importorskip("torch")
    target = torch.tensor([1.0, -2.0, 3.5], dtype=torch.bfloat16)
    stored = target.detach().cpu().contiguous().view(torch.uint16).numpy().copy()

    restored = checkpoint_core._materialize_for_target(stored, target)

    assert restored.dtype == torch.bfloat16
    torch.testing.assert_close(restored, target, rtol=0, atol=0)


def test_manifest_negative_counters_are_rejected_on_load_validation() -> None:
    identity = _identity()
    record = checkpoint_core._build_identity(
        identity,
        {
            "python": "test",
            "implementation": "test",
            "platform": "test",
            "machine": "test",
            "packages": {},
        },
    )
    record["step"] = -1
    record["tokens_seen"] = -8

    with pytest.raises(CheckpointIntegrityError, match="step and tokens_seen"):
        checkpoint_core._validate_manifest_identity(record)


def _sgd_with_momentum_state():
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    loss = model(torch.ones(4, 3)).sum()
    loss.backward()
    optimizer.step()
    return torch, model, optimizer


def test_wrong_shaped_sgd_momentum_is_rejected_by_preflight() -> None:
    torch, _model, optimizer = _sgd_with_momentum_state()
    state = optimizer.state_dict()
    first_id = next(iter(state["state"]))
    state["state"][first_id]["momentum_buffer"] = torch.zeros(1)

    with pytest.raises(
        checkpoint_core.CheckpointCompatibilityError,
        match="optimizer state shape mismatch",
    ):
        checkpoint_core._preflight_optimizer_state(optimizer, state)


def test_optimizer_corruption_rejects_before_live_model_mutation(monkeypatch) -> None:
    torch, model, optimizer = _sgd_with_momentum_state()
    model_before = {name: value.detach().clone() for name, value in model.state_dict().items()}

    bad_state = optimizer.state_dict()
    first_id = next(iter(bad_state["state"]))
    bad_state["state"][first_id]["momentum_buffer"] = torch.zeros(1)

    arrays = {
        name: value.detach().cpu().numpy().copy()
        for name, value in model.state_dict().items()
    }
    combined_state = {
        "optimizer": bad_state,
        "scheduler": None,
        "trainer": {},
        "rng": {},
    }
    monkeypatch.setattr(
        checkpoint_core,
        "_decode_verified_state",
        lambda _verified: (arrays, combined_state),
    )

    verified = checkpoint_core.VerifiedCheckpoint(
        _manifest_bytes=json.dumps({"identity": {}}).encode("utf-8"),
        _artifacts={},
    )

    with pytest.raises(
        checkpoint_core.CheckpointCompatibilityError,
        match="optimizer state shape mismatch",
    ):
        checkpoint_core.load_verified_checkpoint(
            verified,
            model=model,
            optimizer=optimizer,
            restore_rng=False,
        )

    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, model_before[name], rtol=0, atol=0)


def test_valid_torch_sgd_state_passes_preflight() -> None:
    _torch, _model, optimizer = _sgd_with_momentum_state()
    checkpoint_core._preflight_optimizer_state(optimizer, optimizer.state_dict())


def test_valid_numpy_state_structure_still_allows_scalar_value_restore() -> None:
    saved = {
        "lr": 0.01,
        "velocity": np.ones((2,), dtype=np.float64),
    }
    current = {
        "lr": 9.0,
        "velocity": np.zeros((2,), dtype=np.float64),
    }

    checkpoint_core._preflight_state_structure(saved, current, path="optimizer")
