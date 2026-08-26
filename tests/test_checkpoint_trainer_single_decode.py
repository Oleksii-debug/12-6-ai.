from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import twelve_six.checkpoint.trainer_adapter as trainer_adapter


class _TrainerProbe:
    def __init__(self) -> None:
        self.loaded_state: Any = None

    def load_state_dict(self, state: Any) -> None:
        self.loaded_state = state


def test_trainer_resume_decodes_verified_snapshot_once(monkeypatch: Any) -> None:
    manifest = {"identity": {"step": 0, "tokens_seen": 0}}
    verified = SimpleNamespace(manifest=manifest)
    decoded_arrays = {"weight": object()}
    trainer_state = {"opaque": "trainer-state"}
    combined_state = {"trainer": trainer_state, "rng": {"python": None}}
    materialized = {"weight": object()}
    events: list[str] = []
    decode_calls = 0

    monkeypatch.setattr(
        trainer_adapter,
        "prepare_checkpoint_load",
        lambda _directory: verified,
    )

    def decode(snapshot: Any) -> tuple[dict[str, object], dict[str, Any]]:
        nonlocal decode_calls
        assert snapshot is verified
        decode_calls += 1
        events.append("decode")
        return decoded_arrays, combined_state

    def preflight_trainer(trainer: Any, state: Any, *, manifest: Any) -> None:
        assert state is trainer_state
        events.append("trainer-preflight")

    def prepare_model(model: Any, arrays: Any, strict: bool) -> dict[str, object]:
        assert arrays is decoded_arrays
        assert strict is True
        events.append("model-preflight")
        return materialized

    def apply_model(model: Any, prepared: Any, strict: bool) -> None:
        assert prepared is materialized
        assert strict is True
        events.append("model-apply")

    monkeypatch.setattr(trainer_adapter, "_decode_verified_state", decode)
    monkeypatch.setattr(trainer_adapter, "_preflight_trainer_state", preflight_trainer)
    monkeypatch.setattr(trainer_adapter, "_prepare_model_weights", prepare_model)
    monkeypatch.setattr(trainer_adapter, "_apply_model_weights", apply_model)

    model = object()
    trainer = _TrainerProbe()
    result = trainer_adapter.load_trainer_checkpoint(
        "unused-checkpoint-path",
        model=model,
        trainer=trainer,
        restore_rng=False,
    )
    events.append("trainer-apply")

    assert decode_calls == 1
    assert events == [
        "decode",
        "trainer-preflight",
        "model-preflight",
        "model-apply",
        "trainer-apply",
    ]
    assert trainer.loaded_state is trainer_state
    assert result.trainer_state is trainer_state
    assert result.rng_state is combined_state["rng"]
