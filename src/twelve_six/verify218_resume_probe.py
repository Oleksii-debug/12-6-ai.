"""Fresh-process D02/D05 recovery probe for VERIFY-218.

This module restores a retained LEARN-217 recovery checkpoint into a fresh model
and Trainer, validates optimizer tensor metadata and counters, and exits.  It is
verification-only: it never executes a forward/backward training transition or
an optimizer/scheduler update.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from twelve_six.checkpoint import load_trainer_checkpoint, verify_checkpoint
from twelve_six.model import ModelSpec, TwelveSixDecoder
from twelve_six.training import Trainer, TrainerConfig


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _trainer_config(identity: Mapping[str, Any]) -> TrainerConfig:
    training_config = identity.get("training_config")
    _require(isinstance(training_config, Mapping), "checkpoint training_config missing")
    raw = training_config.get("trainer")
    _require(isinstance(raw, Mapping), "checkpoint trainer config missing")
    values = dict(raw)
    if isinstance(values.get("betas"), list):
        values["betas"] = tuple(values["betas"])
    return TrainerConfig(**values)


def _optimizer_state_metadata(trainer: Trainer) -> dict[str, int]:
    populated_parameters = 0
    tensor_leaves = 0
    scalar_steps = 0
    for parameter, state in trainer.optimizer.state.items():
        _require(isinstance(state, Mapping), "optimizer parameter state must be a mapping")
        populated_parameters += 1
        for field, value in state.items():
            if not isinstance(value, torch.Tensor):
                continue
            tensor_leaves += 1
            if value.ndim == 0:
                if field == "step":
                    scalar = float(value.detach().cpu().item())
                    _require(
                        math.isfinite(scalar) and scalar >= 0,
                        "optimizer step tensor must be finite and non-negative",
                    )
                    scalar_steps += 1
                continue
            _require(
                tuple(value.shape) == tuple(parameter.shape),
                f"optimizer {field} shape mismatch: {tuple(value.shape)} != {tuple(parameter.shape)}",
            )
            _require(
                value.dtype == parameter.dtype,
                f"optimizer {field} dtype mismatch: {value.dtype} != {parameter.dtype}",
            )
    _require(populated_parameters > 0, "restored optimizer state is empty")
    _require(tensor_leaves > 0, "restored optimizer has no tensor state")
    return {
        "populated_parameters": populated_parameters,
        "tensor_leaves": tensor_leaves,
        "scalar_step_tensors": scalar_steps,
    }


def probe(checkpoint: Path) -> dict[str, Any]:
    checkpoint = checkpoint.resolve()
    checked = verify_checkpoint(checkpoint)
    identity = checked.get("identity")
    _require(isinstance(identity, Mapping), "checkpoint identity missing")
    spec_raw = identity.get("model_spec")
    _require(isinstance(spec_raw, dict), "checkpoint ModelSpec missing")
    spec = ModelSpec.from_dict(spec_raw)
    config = _trainer_config(identity)

    model = TwelveSixDecoder(spec)
    trainer = Trainer(model, config, device="cpu")
    training_config = identity.get("training_config")
    assert isinstance(training_config, Mapping)
    loaded = load_trainer_checkpoint(
        checkpoint,
        model=model,
        trainer=trainer,
        strict_model=True,
        restore_rng=True,
        expected_git_sha=str(identity["git_sha"]),
        expected_model_spec_hash=str(identity["model_spec_hash"]),
        expected_init_spec_hash=training_config.get("init_spec_sha256"),
        expected_tokenizer_hash=str(identity["tokenizer_hash"]),
        expected_tokenizer_vocab_hash=str(identity["tokenizer_vocab_hash"]),
        expected_dataset_manifest_hash=str(identity["dataset_manifest_hash"]),
        expected_run_manifest_hash=str(identity["run_manifest_hash"]),
        expected_training_config_hash=identity.get("training_config_hash"),
        expected_environment_lock_hash=identity.get("environment_lock_hash"),
        expected_seed=identity.get("seed"),
    )
    _require(
        loaded.manifest.get("checkpoint_id") == checked.get("checkpoint_id"),
        "loaded checkpoint identity drift",
    )
    _require(trainer.optimizer_step == int(identity["step"]), "optimizer_step restore mismatch")
    _require(trainer.tokens_seen == int(identity["tokens_seen"]), "tokens_seen restore mismatch")
    trainer.assert_checkpoint_safe()
    optimizer_metadata = _optimizer_state_metadata(trainer)
    return {
        "pid": os.getpid(),
        "checkpoint_id": checked["checkpoint_id"],
        "optimizer_step": trainer.optimizer_step,
        "tokens_seen": trainer.tokens_seen,
        "optimizer_state": optimizer_metadata,
        "rng_restore_requested": True,
        "checkpoint_safe_after_restore": True,
        "training_executed": False,
        "optimizer_updates": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(probe(args.checkpoint), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
