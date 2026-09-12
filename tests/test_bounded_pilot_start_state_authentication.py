"""Executable adversarial verifier for PREQUAL1467-D02-009.

This test intentionally targets PR #1336 exact head b57ebfb1.  It proves the
bounded-pilot gate must authenticate the live parameter tensor start-state, not
only ModelSpec/InitSpec metadata and parameter count.  On the vulnerable head
the adversarial constructor call is expected to pass, making this verifier red.
No optimizer transition or learned-target execution is performed here.
"""

from __future__ import annotations

from copy import deepcopy

import pytest
import torch

import test_bounded_pilot as product_fixture
from twelve_six.training.bounded_pilot import (
    BoundedPilotAuthorizationError,
    BoundedPilotStepRunner,
)
from twelve_six.training.single_gpu import SingleDeviceStepRunner


def _construct_gate(
    trainer: object,
    binding: object,
    guard: object,
    plan: dict,
) -> BoundedPilotStepRunner:
    packet_sha256 = binding.packet_sha256
    assert packet_sha256 is not None
    return BoundedPilotStepRunner(
        SingleDeviceStepRunner(trainer),
        binding=binding,
        expected_packet_sha256=packet_sha256,
        replay_guard=guard,
        exposure_plan=plan,
        expected_plan_identity_sha256=plan["plan_identity_sha256"],
    )


def test_same_shaped_live_weight_substitution_is_blocked_before_step_1() -> None:
    # Prove the inherited positive fixture itself is valid without executing a step.
    control_guard, control_plan = product_fixture._guard_and_plan()
    control_trainer = product_fixture._trainer()
    control_binding = product_fixture._binding(control_guard, control_trainer)
    control_gate = _construct_gate(
        control_trainer,
        control_binding,
        control_guard,
        control_plan,
    )
    control_gate.close()
    assert control_trainer.optimizer_step == 0
    assert control_trainer.tokens_seen == 0
    assert control_guard.consumed_loss_positions == 0

    guard, plan = product_fixture._guard_and_plan()
    trainer = product_fixture._trainer()
    binding = product_fixture._binding(guard, trainer)

    model_sha_before = trainer.model.spec.identity_sha256()
    init_sha_before = trainer.model.init_spec.identity_sha256()
    parameter_count_before = sum(parameter.numel() for parameter in trainer.model.parameters())
    guard_before = deepcopy(guard.state_dict())
    optimizer_step_before = trainer.optimizer_step
    tokens_seen_before = trainer.tokens_seen

    substituted = next(trainer.model.parameters())
    substituted_before = substituted.detach().clone()
    with torch.no_grad():
        substituted.view(-1)[0].add_(1.0)

    # The adversary changes tensor values only: all metadata accepted by b57 is unchanged.
    assert not torch.equal(substituted.detach(), substituted_before)
    assert trainer.model.spec.identity_sha256() == model_sha_before
    assert trainer.model.init_spec.identity_sha256() == init_sha_before
    assert sum(parameter.numel() for parameter in trainer.model.parameters()) == parameter_count_before
    assert trainer.optimizer_step == optimizer_step_before == 0
    assert trainer.tokens_seen == tokens_seen_before == 0
    assert guard.state_dict() == guard_before

    # Required behavior after the Product repair: fail closed at constructor/pre-step boundary.
    # Vulnerable b57 does not authenticate parameter values, so it currently reaches the end of
    # construction and pytest reports DID NOT RAISE.  That red result is the executable finding.
    with pytest.raises(BoundedPilotAuthorizationError, match="BLOCKED_PRE_STEP_1"):
        _construct_gate(trainer, binding, guard, plan)

    assert trainer.optimizer_step == optimizer_step_before
    assert trainer.tokens_seen == tokens_seen_before
    assert guard.state_dict() == guard_before
