"""Independent adversarial verifier for PR #1336 live start-state authentication.

This is synthetic regression mechanics only. It grants no corpus, tokenizer, optimizer,
training, or learned-weight authority.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
import torch

from twelve_six.data.deterministic_exposure_order import ordered_next_exposure_identity
from twelve_six.training.bounded_pilot import BoundedPilotAuthorizationError


def _load_exact_parent_fixture() -> ModuleType:
    """Reuse PR #1336's own positive-path fixture instead of inventing a parallel gate."""
    path = Path(__file__).with_name("test_bounded_pilot.py")
    spec = importlib.util.spec_from_file_location("_pr1336_exact_fixture", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load exact PR1336 bounded-pilot fixture")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_FIXTURE = _load_exact_parent_fixture()


def _parameter_snapshot(model: torch.nn.Module) -> list[torch.Tensor]:
    return [parameter.detach().clone() for parameter in model.parameters()]


def _assert_same_parameters(
    model: torch.nn.Module,
    expected: list[torch.Tensor],
) -> None:
    actual = [parameter.detach() for parameter in model.parameters()]
    assert len(actual) == len(expected)
    assert all(
        torch.equal(left, right)
        for left, right in zip(actual, expected, strict=True)
    )


def test_same_shape_weight_substitution_is_blocked_before_optimizer_step_1() -> None:
    """Packet metadata must not let arbitrary same-shape live weights pose as random init."""
    guard, plan = _FIXTURE._guard_and_plan()
    trainer = _FIXTURE._trainer()

    # Bind the exact otherwise-valid packet first.  The attack changes only live tensor
    # values afterwards; ModelSpec, InitSpec, parameter count, TrainerConfig, optimizer,
    # packet root, D04 plan, and every currently checked metadata identity stay unchanged.
    binding = _FIXTURE._binding(guard, trainer)
    guard_before = guard.state_dict()

    with torch.no_grad():
        for index, parameter in enumerate(trainer.model.parameters()):
            replacement = torch.full_like(parameter, float(index + 1) / 7.0)
            parameter.copy_(replacement)
    substituted_state = _parameter_snapshot(trainer.model)

    expected_exposure = ordered_next_exposure_identity(
        guard,
        plan,
        batch_index=0,
        expected_plan_identity_sha256=plan["plan_identity_sha256"],
    )

    gate = None
    try:
        with pytest.raises(BoundedPilotAuthorizationError):
            gate = _FIXTURE._gate(
                _FIXTURE.SingleDeviceStepRunner(trainer),
                binding,
                guard,
                plan,
            )
            gate.train_authorized_microbatch(
                _FIXTURE._batch(),
                batch_index=0,
                expected_next_exposure_identity_sha256=expected_exposure,
            )
    finally:
        if gate is not None:
            gate.close()

    assert trainer.optimizer_step == 0
    assert trainer.micro_step == 0
    assert trainer.tokens_seen == 0
    assert guard.state_dict() == guard_before
    assert guard.consumed_loss_positions == 0
    _assert_same_parameters(trainer.model, substituted_state)
