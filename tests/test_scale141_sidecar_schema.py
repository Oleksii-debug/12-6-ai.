from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import twelve_six.scale141_resume_sidecar as sidecar
from twelve_six.scale141_resume_sidecar import ResumeSidecarContext, ResumeSidecarError

LEDGER_HASH = "8" * 64
MATERIALIZATION_HASH = "9" * 64
PACKING_HASH = "a" * 64
CHECKPOINT_MANIFEST_HASH = "b" * 64


def _state_hash(value: dict[str, object]) -> str:
    encoded = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _context() -> ResumeSidecarContext:
    return ResumeSidecarContext(
        generation="generation-00000001",
        checkpoint_manifest_sha256=CHECKPOINT_MANIFEST_HASH,
        checkpoint_id="checkpoint-1",
        source_sha="2" * 40,
        run_manifest_hash="3" * 64,
        optimizer_step=7,
        tokens_seen=12,
    )


def _d04_data() -> dict[str, str]:
    return {
        "ledger_identity_sha256": LEDGER_HASH,
        "materialization_identity_sha256": MATERIALIZATION_HASH,
        "packing_identity_sha256": PACKING_HASH,
        "exposure_plan_identity_sha256": "c" * 64,
        "ordered_next_exposure_identity_sha256": "d" * 64,
    }


def _valid_state() -> dict[str, object]:
    context = _context()
    state: dict[str, object] = {
        "schema_version": sidecar.D04_EXPOSURE_STATE_SCHEMA,
        "ledger_identity_sha256": LEDGER_HASH,
        "materialization_identity_sha256": MATERIALIZATION_HASH,
        "packing_identity_sha256": PACKING_HASH,
        "authorized_budget": 100,
        "one_pass_maximum": 100,
        "consumed_loss_positions": 12,
        "claim_sequence": 7,
        "claims": {},
        "trainer_state_binding": {
            "checkpoint_generation": context.generation,
            "checkpoint_manifest_sha256": context.checkpoint_manifest_sha256,
            "optimizer_step": context.optimizer_step,
            "trainer_nonignored_target_count": 12,
        },
    }
    state["state_identity_sha256"] = _state_hash(state)
    return state


def _rehash(state: dict[str, object]) -> None:
    payload = dict(state)
    payload.pop("state_identity_sha256", None)
    state["state_identity_sha256"] = _state_hash(payload)


def test_valid_d04_exposure_state_passes_frozen_sidecar_preflight() -> None:
    state = _valid_state()

    observed = sidecar._validate_exposure_state(
        state,
        context=_context(),
        d04_data=_d04_data(),
    )

    assert observed == state


def test_forged_d04_state_self_hash_fails_closed() -> None:
    state = _valid_state()
    state["claim_sequence"] = 8

    with pytest.raises(ResumeSidecarError, match="self-hash mismatch"):
        sidecar._validate_exposure_state(
            state,
            context=_context(),
            d04_data=_d04_data(),
        )


def test_rehashed_unknown_d04_state_field_requires_schema_bump() -> None:
    state = _valid_state()
    state["future_semantics"] = {"replay": "changed"}
    _rehash(state)

    with pytest.raises(ResumeSidecarError, match="fields do not match the V2 schema"):
        sidecar._validate_exposure_state(
            state,
            context=_context(),
            d04_data=_d04_data(),
        )


def test_rehashed_impossible_budget_fails_closed() -> None:
    state = _valid_state()
    state["authorized_budget"] = 101
    _rehash(state)

    with pytest.raises(ResumeSidecarError, match="authorized budget exceeds"):
        sidecar._validate_exposure_state(
            state,
            context=_context(),
            d04_data=_d04_data(),
        )


def test_rehashed_unknown_trainer_binding_field_fails_closed() -> None:
    state = _valid_state()
    binding = dict(state["trainer_state_binding"])
    binding["future_counter"] = 12
    state["trainer_state_binding"] = binding
    _rehash(state)

    with pytest.raises(ResumeSidecarError, match="trainer_state_binding fields"):
        sidecar._validate_exposure_state(
            state,
            context=_context(),
            d04_data=_d04_data(),
        )


def _reference() -> dict[str, object]:
    return {
        "schema": sidecar.SIDECAR_SCHEMA,
        "directory": "resume-states/generation-00000001",
        "file": sidecar.SIDECAR_FILE,
        "file_sha256": "1" * 64,
        "payload_sha256": "2" * 64,
        "checkpoint_manifest_sha256": CHECKPOINT_MANIFEST_HASH,
        "state_identity_sha256": "3" * 64,
        "ordered_next_exposure_identity_sha256": "4" * 64,
    }


def test_rehashed_unknown_resume_reference_field_requires_schema_bump() -> None:
    reference = _reference()
    reference["future_reference"] = "5" * 64

    with pytest.raises(ResumeSidecarError, match="fields do not match the V1 schema"):
        sidecar.validate_resume_reference(reference, generation=1)


def test_unknown_sidecar_payload_field_fails_before_binding_use(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = tmp_path / "resume-states" / "generation-00000001"
    directory.mkdir(parents=True)
    monkeypatch.setattr(
        sidecar,
        "_read_payload",
        lambda *_args: {"future_payload": "self-rehashed-but-unknown"},
    )

    with pytest.raises(ResumeSidecarError, match="payload fields do not match the V1 schema"):
        sidecar.load_resume_sidecar(
            tmp_path,
            generation=1,
            checkpoint_path=tmp_path / "checkpoint",
            manifest={},
            reference=_reference(),
        )
