from __future__ import annotations

import copy
from pathlib import Path

import pytest

from twelve_six.checkpoint import (
    D04_RESUME_BINDING_SCHEMA,
    CheckpointCompatibilityError,
    CheckpointIdentity,
    bind_d04_resume_identity,
    hash_json,
    load_trainer_checkpoint,
    save_trainer_checkpoint,
)

_BINDING = {
    "ledger_identity_sha256": "1" * 64,
    "materialization_identity_sha256": "2" * 64,
    "packing_identity_sha256": "3" * 64,
    "exposure_plan_identity_sha256": "4" * 64,
    "ordered_next_exposure_identity_sha256": "5" * 64,
}
_EXPECTED_ARGUMENTS = {
    "ledger_identity_sha256": "expected_ledger_identity_sha256",
    "materialization_identity_sha256": "expected_materialization_identity_sha256",
    "packing_identity_sha256": "expected_packing_identity_sha256",
    "exposure_plan_identity_sha256": "expected_exposure_plan_identity_sha256",
    "ordered_next_exposure_identity_sha256": (
        "expected_ordered_next_exposure_identity_sha256"
    ),
}


class _TrainerProbe:
    def __init__(self) -> None:
        self.config = {"seed": 17, "precision": "fp32"}
        self.loads = 0
        self.velocity = [0.0, 0.0]

    def state_dict(self) -> dict[str, object]:
        return {
            "micro_step": 1,
            "optimizer_step": 1,
            "tokens_seen": 3,
            "optimizer": {"velocity": [1.0, 2.0]},
            "scheduler": None,
            "scaler": None,
            "config": copy.deepcopy(self.config),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        if state["config"] != self.config:
            raise ValueError("config mismatch")
        velocity = state["optimizer"]["velocity"]
        if not isinstance(velocity, list) or len(velocity) != 2:
            raise ValueError("velocity geometry mismatch")
        self.loads += 1
        self.velocity = list(velocity)


def _run_manifest() -> dict[str, object]:
    return {
        "run_id": "d04-resume-fixture",
        "data": dict(_BINDING),
        "training": {"seed": 17},
    }


def _identity(parameter_count: int, manifest: dict[str, object]) -> CheckpointIdentity:
    base = CheckpointIdentity(
        git_sha="a" * 40,
        model_spec={"kind": "d04-resume-probe", "parameters": parameter_count},
        parameter_count=parameter_count,
        tokenizer_hash="b" * 64,
        tokenizer_vocab_hash="c" * 64,
        dataset_manifest_hash="d" * 64,
        run_manifest_hash=hash_json(manifest),
        training_config={
            "data": {
                "dataset_manifest_sha256": "d" * 64,
                "packing_sha256": "e" * 64,
            },
            "training": {"seed": 17},
        },
        seed=17,
        precision="fp32",
        step=1,
        tokens_seen=3,
        optimizer={"name": "probe"},
        scheduler=None,
        environment_lock_hash="f" * 64,
    )
    return bind_d04_resume_identity(base, run_manifest=manifest)


def _expected(**overrides: str) -> dict[str, str]:
    values = {
        argument: _BINDING[field]
        for field, argument in _EXPECTED_ARGUMENTS.items()
    }
    values.update(overrides)
    return values


def test_binding_is_transitively_carried_by_run_and_training_identity() -> None:
    manifest = _run_manifest()
    identity = _identity(6, manifest)
    data = identity.training_config["data"]

    assert identity.run_manifest_hash == hash_json(manifest)
    assert data["resume_binding_schema"] == D04_RESUME_BINDING_SCHEMA
    for field, value in _BINDING.items():
        assert data[field] == value


def test_binding_rejects_incomplete_or_wrong_run_manifest() -> None:
    manifest = _run_manifest()
    raw = CheckpointIdentity(
        git_sha="a" * 40,
        model_spec={"kind": "probe"},
        parameter_count=1,
        tokenizer_hash="b" * 64,
        tokenizer_vocab_hash="c" * 64,
        dataset_manifest_hash="d" * 64,
        run_manifest_hash=hash_json(manifest),
        training_config={"data": {}},
        seed=17,
        precision="fp32",
        step=0,
        tokens_seen=0,
        optimizer={"name": "probe"},
        scheduler=None,
    )

    incomplete = copy.deepcopy(manifest)
    incomplete["data"].pop("ordered_next_exposure_identity_sha256")
    incomplete_raw = copy.deepcopy(raw)
    incomplete_raw = CheckpointIdentity(
        **{
            **incomplete_raw.__dict__,
            "run_manifest_hash": hash_json(incomplete),
        }
    )
    with pytest.raises(CheckpointCompatibilityError, match="ordered_next_exposure"):
        bind_d04_resume_identity(incomplete_raw, run_manifest=incomplete)

    wrong = copy.deepcopy(manifest)
    wrong["run_id"] = "different-run"
    with pytest.raises(CheckpointCompatibilityError, match="run manifest does not match"):
        bind_d04_resume_identity(raw, run_manifest=wrong)


@pytest.mark.parametrize(("field", "argument"), _EXPECTED_ARGUMENTS.items())
def test_wrong_d04_handoff_rejects_before_model_or_trainer_mutation(
    tmp_path: Path,
    field: str,
    argument: str,
) -> None:
    torch = pytest.importorskip("torch")
    torch.manual_seed(101)
    source_model = torch.nn.Linear(3, 2, bias=False)
    source_trainer = _TrainerProbe()
    manifest = _run_manifest()
    checkpoint = tmp_path / field
    save_trainer_checkpoint(
        checkpoint,
        model=source_model,
        trainer=source_trainer,
        identity=_identity(sum(p.numel() for p in source_model.parameters()), manifest),
    )

    torch.manual_seed(202)
    target_model = torch.nn.Linear(3, 2, bias=False)
    target_trainer = _TrainerProbe()
    before = {
        name: tensor.detach().clone()
        for name, tensor in target_model.state_dict().items()
    }
    expectations = _expected(**{argument: "0" * 64})

    with pytest.raises(CheckpointCompatibilityError, match="D04 resume binding mismatch"):
        load_trainer_checkpoint(
            checkpoint,
            model=target_model,
            trainer=target_trainer,
            restore_rng=False,
            expected_step=1,
            expected_tokens_seen=3,
            **expectations,
        )

    for name, tensor in target_model.state_dict().items():
        torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)
    assert target_trainer.loads == 0
    assert target_trainer.velocity == [0.0, 0.0]


def test_exact_d04_handoff_allows_resume(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    torch.manual_seed(303)
    source_model = torch.nn.Linear(3, 2, bias=False)
    source_trainer = _TrainerProbe()
    manifest = _run_manifest()
    checkpoint = tmp_path / "exact"
    save_trainer_checkpoint(
        checkpoint,
        model=source_model,
        trainer=source_trainer,
        identity=_identity(sum(p.numel() for p in source_model.parameters()), manifest),
    )

    torch.manual_seed(404)
    target_model = torch.nn.Linear(3, 2, bias=False)
    target_trainer = _TrainerProbe()
    load_trainer_checkpoint(
        checkpoint,
        model=target_model,
        trainer=target_trainer,
        restore_rng=False,
        expected_step=1,
        expected_tokens_seen=3,
        **_expected(),
    )

    for name, tensor in target_model.state_dict().items():
        torch.testing.assert_close(
            tensor,
            source_model.state_dict()[name],
            rtol=0,
            atol=0,
        )
    assert target_trainer.loads == 1
    assert target_trainer.velocity == [1.0, 2.0]
