from __future__ import annotations

import hashlib
import json
import random
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

from twelve_six.checkpoint import (
    D04_RESUME_BINDING_SCHEMA,
    CheckpointCompatibilityError,
    CheckpointIdentity,
    load_trainer_checkpoint,
    save_trainer_checkpoint,
)
from twelve_six.scale141_recovery import (
    RecoveryLifecycleError,
    RecoveryPointerUpdateInterrupted,
    cleanup_recovery_generations,
    publish_recovery_generation,
    resolve_recovery_generation,
)
from twelve_six.scale141_resume_sidecar import ResumeSidecarContext
from twelve_six.training import Trainer, TrainerConfig
from twelve_six.trusted_parent_recovery_binding import (
    trusted_parent_recovery_binding_from_resolution,
    trusted_recovery_authority_token,
)

SOURCE_SHA = "2" * 40
RUN_HASH = "3" * 64
ENV_HASH = "4" * 64
TOKENIZER_HASH = "5" * 64
VOCAB_HASH = "6" * 64
DATA_HASH = "7" * 64
LEDGER_HASH = "8" * 64
MATERIALIZATION_HASH = "9" * 64
PACKING_IDENTITY_HASH = "a" * 64
EXPOSURE_PLAN_HASH = "b" * 64
ORDERED_NEXT_HASH = "c" * 64
SEGMENT_HASH = "d" * 64


class TinyLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(16, 8)
        self.projection = nn.Linear(8, 16, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.projection(self.embedding(input_ids))


def _stack() -> tuple[TinyLM, Trainer, TrainerConfig]:
    cfg = TrainerConfig(max_steps=8, learning_rate=1e-3, seed=211)
    torch.manual_seed(cfg.seed)
    model = TinyLM()
    trainer = Trainer(model, cfg, device="cpu")
    return model, trainer, cfg


def _identity(model: TinyLM, trainer: Trainer, cfg: TrainerConfig) -> CheckpointIdentity:
    return CheckpointIdentity(
        git_sha=SOURCE_SHA,
        model_spec={"kind": "scale141-d04-sidecar-probe", "vocab": 16, "width": 8},
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        tokenizer_hash=TOKENIZER_HASH,
        tokenizer_vocab_hash=VOCAB_HASH,
        dataset_manifest_hash=DATA_HASH,
        run_manifest_hash=RUN_HASH,
        training_config={
            "trainer": asdict(cfg),
            "data": {
                "resume_binding_schema": D04_RESUME_BINDING_SCHEMA,
                "ledger_identity_sha256": LEDGER_HASH,
                "materialization_identity_sha256": MATERIALIZATION_HASH,
                "packing_identity_sha256": PACKING_IDENTITY_HASH,
                "exposure_plan_identity_sha256": EXPOSURE_PLAN_HASH,
                "ordered_next_exposure_identity_sha256": ORDERED_NEXT_HASH,
            },
        },
        seed=cfg.seed,
        precision=cfg.precision,
        step=trainer.optimizer_step,
        tokens_seen=trainer.tokens_seen,
        optimizer={
            "name": "AdamW",
            "learning_rate": cfg.learning_rate,
            "betas": list(cfg.betas),
            "eps": cfg.eps,
            "weight_decay": cfg.weight_decay,
        },
        scheduler=None,
        environment_lock_hash=ENV_HASH,
    )


def _save(path: Path, model: TinyLM, trainer: Trainer, cfg: TrainerConfig):
    return save_trainer_checkpoint(
        path,
        model=model,
        trainer=trainer,
        identity=_identity(model, trainer, cfg),
    )


def _state_hash(value: dict[str, object]) -> str:
    encoded = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _stable_digest(value: Any) -> str:
    digest = hashlib.sha256()

    def visit(item: Any) -> None:
        if torch.is_tensor(item):
            tensor = item.detach().cpu().contiguous()
            digest.update(b"T")
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(repr(tuple(tensor.shape)).encode("ascii"))
            digest.update(tensor.numpy().tobytes())
            return
        if isinstance(item, Mapping):
            digest.update(b"D")
            for key in sorted(item, key=lambda candidate: repr(candidate)):
                visit(key)
                visit(item[key])
            return
        if isinstance(item, list):
            digest.update(b"L")
            for child in item:
                visit(child)
            return
        if isinstance(item, tuple):
            digest.update(b"U")
            for child in item:
                visit(child)
            return
        digest.update(type(item).__name__.encode("utf-8"))
        digest.update(b":")
        digest.update(repr(item).encode("utf-8"))

    visit(value)
    return digest.hexdigest()


def _resume_state(context: ResumeSidecarContext) -> dict[str, object]:
    consumed = context.tokens_seen
    value: dict[str, object] = {
        "schema_version": "12-6.unique-loss-exposure-state.v2",
        "ledger_identity_sha256": LEDGER_HASH,
        "materialization_identity_sha256": MATERIALIZATION_HASH,
        "packing_identity_sha256": PACKING_IDENTITY_HASH,
        "authorized_budget": 1_000_000,
        "one_pass_maximum": 1_000_000,
        "consumed_loss_positions": consumed,
        "claim_sequence": context.optimizer_step,
        "claims": ({SEGMENT_HASH: [[0, consumed]]} if consumed else {}),
        "trainer_state_binding": {
            "checkpoint_generation": context.generation,
            "checkpoint_manifest_sha256": context.checkpoint_manifest_sha256,
            "optimizer_step": context.optimizer_step,
            "trainer_nonignored_target_count": consumed,
        },
    }
    value["state_identity_sha256"] = _state_hash(value)
    return value


def _publish(
    root: Path,
    model: TinyLM,
    trainer: Trainer,
    cfg: TrainerConfig,
    *,
    builder=_resume_state,
    failpoint: str | None = None,
):
    return publish_recovery_generation(
        root,
        save_generation=lambda path: _save(path, model, trainer, cfg),
        expected_source_sha=SOURCE_SHA,
        expected_run_manifest_hash=RUN_HASH,
        expected_step=trainer.optimizer_step,
        expected_tokens_seen=trainer.tokens_seen,
        build_resume_state=builder,
        failpoint=failpoint,
    )


def _step(trainer: Trainer, offset: int = 0) -> None:
    values = torch.tensor([[1 + offset, 2 + offset, 3 + offset, 4 + offset]]) % 16
    trainer.train_microbatch({"input_ids": values})


def test_sidecar_publishes_after_manifest_and_resolves_exact_d04_state(tmp_path: Path) -> None:
    model, trainer, cfg = _stack()
    _step(trainer)
    root = tmp_path / "recovery"

    reference = _publish(root, model, trainer, cfg)
    resolution = resolve_recovery_generation(root, expected_reference=reference)

    assert resolution.resume_state is not None
    binding = resolution.resume_state["trainer_state_binding"]
    sidecar_reference = reference["resume_state"]
    assert binding["checkpoint_generation"] == "generation-00000001"
    assert binding["checkpoint_manifest_sha256"] == sidecar_reference[
        "checkpoint_manifest_sha256"
    ]
    assert binding["optimizer_step"] == trainer.optimizer_step
    assert resolution.resume_state["state_identity_sha256"] == sidecar_reference[
        "state_identity_sha256"
    ]
    assert sidecar_reference["ordered_next_exposure_identity_sha256"] == ORDERED_NEXT_HASH

    fresh_model, fresh_trainer, _ = _stack()
    load_trainer_checkpoint(
        resolution.path,
        model=fresh_model,
        trainer=fresh_trainer,
        restore_rng=False,
        expected_checkpoint_id=reference["checkpoint_id"],
        expected_manifest_sha256=reference["manifest_sha256"],
        expected_git_sha=SOURCE_SHA,
        expected_run_manifest_hash=RUN_HASH,
        expected_step=trainer.optimizer_step,
        expected_tokens_seen=trainer.tokens_seen,
        expected_ledger_identity_sha256=LEDGER_HASH,
        expected_materialization_identity_sha256=MATERIALIZATION_HASH,
        expected_packing_identity_sha256=PACKING_IDENTITY_HASH,
        expected_exposure_plan_identity_sha256=EXPOSURE_PLAN_HASH,
        expected_ordered_next_exposure_identity_sha256=ORDERED_NEXT_HASH,
    )
    assert fresh_trainer.optimizer_step == trainer.optimizer_step
    assert fresh_trainer.tokens_seen == trainer.tokens_seen


@pytest.mark.parametrize("mutation", ["missing", "truncated"])
def test_missing_or_truncated_current_sidecar_fails_closed(
    tmp_path: Path, mutation: str
) -> None:
    model, trainer, cfg = _stack()
    _step(trainer)
    root = tmp_path / mutation
    _publish(root, model, trainer, cfg)
    state_path = root / "resume-states/generation-00000001/state.json"

    if mutation == "missing":
        state_path.unlink()
    else:
        state_path.write_text("{", encoding="utf-8")

    with pytest.raises(RecoveryLifecycleError, match="resume sidecar"):
        resolve_recovery_generation(root)


def test_swapped_sidecar_from_other_generation_fails_closed(tmp_path: Path) -> None:
    model, trainer, cfg = _stack()
    root = tmp_path / "swapped"
    _step(trainer)
    _publish(root, model, trainer, cfg)
    first_bytes = (root / "resume-states/generation-00000001/state.json").read_bytes()

    _step(trainer, 1)
    _publish(root, model, trainer, cfg)
    second_path = root / "resume-states/generation-00000002/state.json"
    second_path.write_bytes(first_bytes)

    with pytest.raises(RecoveryLifecycleError, match="resume sidecar"):
        resolve_recovery_generation(root)


def test_wrong_checkpoint_binding_never_advances_pointer(tmp_path: Path) -> None:
    model, trainer, cfg = _stack()
    _step(trainer)
    root = tmp_path / "wrong-binding"

    def wrong_binding(context: ResumeSidecarContext) -> dict[str, object]:
        state = _resume_state(context)
        state["trainer_state_binding"]["checkpoint_manifest_sha256"] = "0" * 64
        state_without_hash = dict(state)
        state_without_hash.pop("state_identity_sha256")
        state["state_identity_sha256"] = _state_hash(state_without_hash)
        return state

    with pytest.raises(RecoveryLifecycleError, match="sidecar publication failed"):
        _publish(root, model, trainer, cfg, builder=wrong_binding)

    assert not (root / "current.json").exists()
    assert (root / "generations/generation-00000001").is_dir()
    assert not (root / "resume-states/generation-00000001").exists()


def test_interruption_after_sidecar_keeps_last_known_good_and_cleanup_removes_orphan_pair(
    tmp_path: Path,
) -> None:
    model, trainer, cfg = _stack()
    root = tmp_path / "interrupted"
    _step(trainer)
    first = _publish(root, model, trainer, cfg)

    _step(trainer, 1)
    with pytest.raises(RecoveryPointerUpdateInterrupted, match="after D04 sidecar"):
        _publish(
            root,
            model,
            trainer,
            cfg,
            failpoint="after_sidecar_before_pointer",
        )

    current = resolve_recovery_generation(root, expected_reference=first)
    assert current.reference["generation"] == 1
    assert (root / "generations/generation-00000002").is_dir()
    assert (root / "resume-states/generation-00000002").is_dir()

    cleaned = cleanup_recovery_generations(root, keep=1)
    assert "generation-00000002" in cleaned["removed"]
    assert not (root / "generations/generation-00000002").exists()
    assert not (root / "resume-states/generation-00000002").exists()
    assert resolve_recovery_generation(root, expected_reference=first).resume_state is not None



def test_metadata_compatible_alternate_checkpoint_rejected_before_mutation(
    tmp_path: Path,
) -> None:
    model_a, trainer_a, cfg_a = _stack()
    _step(trainer_a)
    reference_a = _publish(tmp_path / "trusted", model_a, trainer_a, cfg_a)

    model_b, trainer_b, cfg_b = _stack()
    _step(trainer_b)
    with torch.no_grad():
        first_parameter = next(model_b.parameters())
        first_parameter.view(-1)[0].add_(0.125)
    reference_b = _publish(tmp_path / "alternate", model_b, trainer_b, cfg_b)

    assert reference_b["checkpoint_id"] != reference_a["checkpoint_id"]
    assert reference_b["manifest_sha256"] != reference_a["manifest_sha256"]
    resolution_b = resolve_recovery_generation(
        tmp_path / "alternate",
        expected_reference=reference_b,
    )

    target_model, target_trainer, _ = _stack()
    model_before = {
        name: tensor.detach().clone()
        for name, tensor in target_model.state_dict().items()
    }
    counters_before = (
        target_trainer.micro_step,
        target_trainer.optimizer_step,
        target_trainer.tokens_seen,
    )
    rng_before = torch.get_rng_state().clone()

    with pytest.raises(
        CheckpointCompatibilityError,
        match="checkpoint_id does not match the independently expected D05 identity",
    ):
        load_trainer_checkpoint(
            resolution_b.path,
            model=target_model,
            trainer=target_trainer,
            restore_rng=True,
            expected_checkpoint_id=reference_a["checkpoint_id"],
            expected_manifest_sha256=reference_a["manifest_sha256"],
            expected_git_sha=SOURCE_SHA,
            expected_run_manifest_hash=RUN_HASH,
            expected_step=trainer_a.optimizer_step,
            expected_tokens_seen=trainer_a.tokens_seen,
            expected_ledger_identity_sha256=LEDGER_HASH,
            expected_materialization_identity_sha256=MATERIALIZATION_HASH,
            expected_packing_identity_sha256=PACKING_IDENTITY_HASH,
            expected_exposure_plan_identity_sha256=EXPOSURE_PLAN_HASH,
            expected_ordered_next_exposure_identity_sha256=ORDERED_NEXT_HASH,
        )

    for name, tensor in target_model.state_dict().items():
        assert torch.equal(tensor, model_before[name])
    assert (
        target_trainer.micro_step,
        target_trainer.optimizer_step,
        target_trainer.tokens_seen,
    ) == counters_before
    assert torch.equal(torch.get_rng_state(), rng_before)


def test_fresh_process_restores_exact_d05_d04_trainer_and_rng_state(
    tmp_path: Path,
) -> None:
    random.seed(211)
    np.random.seed(211)
    model, trainer, cfg = _stack()
    _step(trainer)
    root = tmp_path / "fresh-process"
    reference = _publish(root, model, trainer, cfg)
    resolution = resolve_recovery_generation(root, expected_reference=reference)
    placeholder_authority = {
        "repository": "Oleksii-debug/12-6-ai.",
        "git_sha": SOURCE_SHA,
        "evidence_sha256": "0" * 64,
        "workflow_run_id": 1811,
        "workflow_conclusion": "success",
        "terminal": True,
    }
    projected = trusted_parent_recovery_binding_from_resolution(
        resolution,
        provider_class="OTHER_FREE",
        provider_id="GITHUB_ACTIONS_STANDARD_LINUX_X64",
        provider_session_id="fresh-process-b",
        previous_provider_session_id="publisher-process-a",
        terminal_recovery_authority=placeholder_authority,
    )
    authority = dict(placeholder_authority)
    authority["evidence_sha256"] = projected["binding_sha256"]
    trusted = trusted_parent_recovery_binding_from_resolution(
        resolution,
        provider_class="OTHER_FREE",
        provider_id="GITHUB_ACTIONS_STANDARD_LINUX_X64",
        provider_session_id="fresh-process-b",
        previous_provider_session_id="publisher-process-a",
        terminal_recovery_authority=authority,
    )

    expected = {
        "model_state_sha256": _stable_digest(model.state_dict()),
        "trainer_state_sha256": _stable_digest(trainer.state_dict()),
        "micro_step": trainer.micro_step,
        "optimizer_step": trainer.optimizer_step,
        "tokens_seen": trainer.tokens_seen,
        "checkpoint_id": reference["checkpoint_id"],
        "manifest_sha256": reference["manifest_sha256"],
        "d04_state_identity_sha256": reference["resume_state"][
            "state_identity_sha256"
        ],
        "ordered_next_exposure_identity_sha256": reference["resume_state"][
            "ordered_next_exposure_identity_sha256"
        ],
        "trusted_parent_binding_sha256": trusted["binding_sha256"],
        "trusted_parent_checkpoint_id": trusted["checkpoint_id"],
        "trusted_parent_manifest_sha256": trusted["checkpoint_manifest_sha256"],
        "trusted_recovery_authority_token": trusted_recovery_authority_token(authority),
        "rng_probe": {
            "python": random.random(),
            "numpy": float(np.random.random()),
            "torch": float(torch.rand(1).item()),
        },
    }

    _step(trainer, 1)
    with pytest.raises(RecoveryPointerUpdateInterrupted, match="after D04 sidecar"):
        _publish(
            root,
            model,
            trainer,
            cfg,
            failpoint="after_sidecar_before_pointer",
        )
    still_current = resolve_recovery_generation(root, expected_reference=reference)
    assert still_current.reference["checkpoint_id"] == reference["checkpoint_id"]
    assert (root / "generations/generation-00000002").is_dir()
    assert (root / "resume-states/generation-00000002").is_dir()

    reference_path = tmp_path / "reference.json"
    reference_path.write_text(
        json.dumps(reference, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    worker = (
        Path(__file__).parent
        / "helpers"
        / "pr1811_fresh_resume_worker.py"
    )
    completed = subprocess.run(
        [sys.executable, str(worker), str(root), str(reference_path)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    observed = json.loads(completed.stdout)
    assert observed == expected


@pytest.mark.parametrize(
    ("expected_field", "message"),
    [
        (
            "expected_checkpoint_id",
            "checkpoint_id does not match the independently expected D05 identity",
        ),
        (
            "expected_manifest_sha256",
            "manifest SHA-256 does not match the independently expected D05 identity",
        ),
    ],
)
def test_exact_d05_identity_mismatch_rejects_before_restore_mutation(
    tmp_path: Path,
    expected_field: str,
    message: str,
) -> None:
    model, trainer, cfg = _stack()
    _step(trainer)
    root = tmp_path / expected_field
    reference = _publish(root, model, trainer, cfg)
    resolution = resolve_recovery_generation(root, expected_reference=reference)

    fresh_model, fresh_trainer, _ = _stack()
    model_before = {
        name: tensor.detach().clone()
        for name, tensor in fresh_model.state_dict().items()
    }
    trainer_before = (
        fresh_trainer.micro_step,
        fresh_trainer.optimizer_step,
        fresh_trainer.tokens_seen,
    )
    rng_before = torch.get_rng_state().clone()

    kwargs = {
        "expected_checkpoint_id": reference["checkpoint_id"],
        "expected_manifest_sha256": reference["manifest_sha256"],
    }
    actual = kwargs[expected_field]
    replacement = "0" * 64 if actual != "0" * 64 else "f" * 64
    kwargs[expected_field] = replacement

    with pytest.raises(CheckpointCompatibilityError, match=message):
        load_trainer_checkpoint(
            resolution.path,
            model=fresh_model,
            trainer=fresh_trainer,
            restore_rng=True,
            expected_git_sha=SOURCE_SHA,
            expected_run_manifest_hash=RUN_HASH,
            expected_step=trainer.optimizer_step,
            expected_tokens_seen=trainer.tokens_seen,
            expected_ledger_identity_sha256=LEDGER_HASH,
            expected_materialization_identity_sha256=MATERIALIZATION_HASH,
            expected_packing_identity_sha256=PACKING_IDENTITY_HASH,
            expected_exposure_plan_identity_sha256=EXPOSURE_PLAN_HASH,
            expected_ordered_next_exposure_identity_sha256=ORDERED_NEXT_HASH,
            **kwargs,
        )

    for name, tensor in fresh_model.state_dict().items():
        assert torch.equal(tensor, model_before[name])
    assert (
        fresh_trainer.micro_step,
        fresh_trainer.optimizer_step,
        fresh_trainer.tokens_seen,
    ) == trainer_before
    assert torch.equal(torch.get_rng_state(), rng_before)
