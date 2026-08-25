#!/usr/bin/env python3
"""Execute a controlled NaN/Inf update against the real S1 decoder."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import torch

from twelve_six.checkpoint.trainer_adapter import save_trainer_checkpoint
from twelve_six.model import TwelveSixDecoder, load_stage_config
from twelve_six.training import (
    NonFiniteTrainingError,
    Trainer,
    TrainerConfig,
    TrainingStateInvalidError,
)


class InjectNonFiniteAdamW(torch.optim.AdamW):
    """Probe-only AdamW that poisons a parameter after a selected real step."""

    def __init__(self, params, *, poison: str, poison_on_step: int) -> None:
        super().__init__(params, lr=3e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
        self.poison = poison
        self.poison_on_step = poison_on_step
        self.step_calls = 0

    def step(self, closure=None):
        result = super().step(closure)
        self.step_calls += 1
        if self.step_calls == self.poison_on_step:
            value = float("nan") if self.poison == "nan" else float("inf")
            with torch.no_grad():
                parameter = self.param_groups[0]["params"][0]
                parameter.view(-1)[0] = value
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-config", default="configs/stages/s1_100k.json")
    parser.add_argument("--poison", choices=("nan", "inf"), default="nan")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-sha", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(55)
    stage = load_stage_config(args.stage_config)
    model = TwelveSixDecoder(stage.model, stage.init)
    optimizer = InjectNonFiniteAdamW(
        model.parameters(), poison=args.poison, poison_on_step=2
    )
    trainer = Trainer(
        model,
        TrainerConfig(
            learning_rate=3e-4,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.1,
            max_steps=2,
            scheduler="constant",
            gradient_clip_norm=1.0,
            precision="fp32",
            seed=55,
        ),
        optimizer=optimizer,
    )

    tokens = torch.arange(64, dtype=torch.long).reshape(2, 32) % stage.model.vocab_size
    batch = {"input_ids": tokens}

    healthy = trainer.train_microbatch(batch)
    if trainer.optimizer_step != 1 or not healthy.optimizer_stepped:
        raise AssertionError("healthy control update did not commit exactly once")
    committed_before_poison = trainer.optimizer_step

    try:
        trainer.train_microbatch(batch)
    except NonFiniteTrainingError as exc:
        diagnostics = exc.diagnostics
    else:
        raise AssertionError("controlled non-finite update did not fail closed")

    if diagnostics is None or diagnostics.kind != "update":
        raise AssertionError("missing structured update failure diagnostics")
    if trainer.optimizer_step != committed_before_poison:
        raise AssertionError("poisoned optimizer update was logically committed")

    repeat_training_blocked = False
    try:
        trainer.train_microbatch(batch)
    except TrainingStateInvalidError:
        repeat_training_blocked = True

    state_dict_blocked = False
    try:
        trainer.state_dict()
    except TrainingStateInvalidError:
        state_dict_blocked = True

    with tempfile.TemporaryDirectory() as temporary:
        checkpoint_dir = Path(temporary) / "poisoned"
        checkpoint_publication_blocked = False
        try:
            save_trainer_checkpoint(
                checkpoint_dir,
                model=model,
                trainer=trainer,
                identity=None,
            )
        except TrainingStateInvalidError:
            checkpoint_publication_blocked = not checkpoint_dir.exists()

    if not (repeat_training_blocked and state_dict_blocked and checkpoint_publication_blocked):
        raise AssertionError("poisoned Trainer escaped a fail-closed boundary")

    report = {
        "schema_version": 1,
        "worker_id": "TRAIN-55-NUMERIC-FORENSICS",
        "source_sha": args.source_sha,
        "stage": stage.stage,
        "model_parameters": stage.model.parameter_count(),
        "model_identity_sha256": stage.model.identity_sha256(),
        "init_identity_sha256": stage.init.identity_sha256(),
        "precision": "fp32",
        "device": "cpu",
        "injected_nonfinite": args.poison,
        "healthy_optimizer_step": healthy.optimizer_step,
        "committed_before_poison": committed_before_poison,
        "optimizer_step_after_poison": trainer.optimizer_step,
        "repeat_training_blocked": repeat_training_blocked,
        "state_dict_blocked": state_dict_blocked,
        "checkpoint_publication_blocked": checkpoint_publication_blocked,
        "model_forward_semantics_modified": False,
        "raw_training_text_logged": False,
        "diagnostics": diagnostics.to_dict(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
