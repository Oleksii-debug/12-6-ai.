from __future__ import annotations

from pathlib import Path

import pytest
from test_bounded_pilot import _authority, _batch, _next
from test_bounded_pilot_durable_attempt import _gate, _trainer

from twelve_six.training.bounded_pilot import BoundedPilotRecoveryRequiredError


def test_post_step_optimizer_poison_consumes_d04_but_never_returns_success_receipt(
    tmp_path: Path,
) -> None:
    guard, plan, manifest = _authority(batch_count=1)
    trainer = _trainer(max_steps=1)
    gate, store = _gate(
        tmp_path / "optimizer-poison",
        trainer,
        guard,
        plan,
        manifest,
        run_id="optimizer-poison-no-replay",
    )

    def poison_optimizer_state(optimizer, _args, _kwargs) -> None:
        state = next(iter(optimizer.state.values()))
        state["exp_avg_sq"].view(-1)[0] = float("inf")

    hook = trainer.optimizer.register_step_post_hook(poison_optimizer_state)
    try:
        with pytest.raises(
            BoundedPilotRecoveryRequiredError,
            match="Trainer execution failed after D04 authorization was consumed",
        ):
            gate.train_authorized_microbatch(
                _batch(0),
                batch_index=0,
                expected_next_exposure_identity_sha256=_next(guard, plan, 0),
            )
    finally:
        hook.remove()
        gate.close()

    # Optimizer mutation was entered only after D04 authorization, so exposure must
    # never be rewound even though Trainer refuses to commit a successful step.
    assert guard.consumed_loss_positions == 2
    assert trainer.optimizer_step == 0
    assert trainer.tokens_seen == 2
    recovery = store.open()
    assert recovery["attempt"] == 1
    assert recovery["failure_count"] == 1
    assert recovery["phase"] in {"RECOVERING", "FAILED"}
