from __future__ import annotations

import copy

import numpy as np
import pytest

from twelve_six.checkpoint import CheckpointCompatibilityError, CheckpointIntegrityError
from twelve_six.checkpoint.hardening import (
    _preflight_optimizer_state,
    _validate_identity_counters,
    _validate_model_tensor_dtypes,
)


class _NumpyModel:
    def __init__(self, dtype: np.dtype) -> None:
        self.weight = np.zeros((2, 3), dtype=dtype)

    def state_dict(self):
        return {"weight": self.weight}


def test_negative_step_fails_closed() -> None:
    manifest = {"identity": {"step": -1, "tokens_seen": 0}}
    with pytest.raises(CheckpointIntegrityError, match="identity.step"):
        _validate_identity_counters(manifest)


def test_negative_tokens_seen_fails_closed() -> None:
    manifest = {"identity": {"step": 0, "tokens_seen": -1}}
    with pytest.raises(CheckpointIntegrityError, match="identity.tokens_seen"):
        _validate_identity_counters(manifest)


def test_boolean_resume_counter_is_not_accepted_as_integer() -> None:
    manifest = {"identity": {"step": True, "tokens_seen": 0}}
    with pytest.raises(CheckpointIntegrityError, match="identity.step"):
        _validate_identity_counters(manifest)


def test_numpy_weight_dtype_mismatch_is_rejected_instead_of_cast() -> None:
    model = _NumpyModel(np.dtype(np.float32))
    arrays = {"weight": np.zeros((2, 3), dtype=np.float64)}

    with pytest.raises(CheckpointCompatibilityError, match="dtype mismatch"):
        _validate_model_tensor_dtypes(model, arrays, strict=True)


def test_numpy_weight_exact_dtype_is_accepted() -> None:
    model = _NumpyModel(np.dtype(np.float32))
    arrays = {"weight": np.zeros((2, 3), dtype=np.float32)}

    _validate_model_tensor_dtypes(model, arrays, strict=True)


def test_sgd_wrong_shaped_momentum_is_rejected_before_live_optimizer_mutation() -> None:
    torch = pytest.importorskip("torch")

    parameter = torch.nn.Parameter(torch.ones(3, dtype=torch.float32))
    optimizer = torch.optim.SGD([parameter], lr=0.1, momentum=0.9)
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    corrupt = copy.deepcopy(optimizer.state_dict())
    state_id = next(iter(corrupt["state"]))
    corrupt["state"][state_id]["momentum_buffer"] = torch.ones(2, dtype=torch.float32)

    live_parameter_before = parameter.detach().clone()
    live_optimizer_before = copy.deepcopy(optimizer.state_dict())

    with pytest.raises(CheckpointCompatibilityError, match="shape mismatch"):
        _preflight_optimizer_state(optimizer, corrupt)

    assert torch.equal(parameter.detach(), live_parameter_before)
    assert optimizer.state_dict()["param_groups"] == live_optimizer_before["param_groups"]
    live_state_id = next(iter(optimizer.state_dict()["state"]))
    assert torch.equal(
        optimizer.state_dict()["state"][live_state_id]["momentum_buffer"],
        live_optimizer_before["state"][live_state_id]["momentum_buffer"],
    )


def test_sgd_matching_momentum_state_passes_preflight() -> None:
    torch = pytest.importorskip("torch")

    parameter = torch.nn.Parameter(torch.ones(3, dtype=torch.float32))
    optimizer = torch.optim.SGD([parameter], lr=0.1, momentum=0.9)
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()

    _preflight_optimizer_state(optimizer, copy.deepcopy(optimizer.state_dict()))
