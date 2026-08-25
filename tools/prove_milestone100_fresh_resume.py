#!/usr/bin/env python3
"""Prove D05 checkpoint resume and evaluation non-mutation in a fresh Python process."""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
from pathlib import Path

from twelve_six.checkpoint import load_trainer_checkpoint
from twelve_six.model import InitSpec, TwelveSixDecoder
from twelve_six.scaling_500k_evidence import _model_state_sha256, _target_spec, _tree_sha256
from twelve_six.scaling_experiment import (
    _byte_stream,
    _file_sha256,
    _make_batch,
    _read_jsonl,
    _trainer_config,
    _validation_loss,
)
from twelve_six.tokenization import BYTE_TOKENIZER_HASH, BYTE_VOCAB_HASH, ByteTokenizer
from twelve_six.training import Trainer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--resume-budget", type=int, default=16384)
    parser.add_argument("--final-budget", type=int, default=65536)
    args = parser.parse_args(argv)

    repo_root = args.repo_root.resolve()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    run = next(item for item in report["seed_runs"] if int(item["seed"]) == args.seed)
    checkpoint = next(
        item for item in run["checkpoints"] if int(item["requested_token_budget"]) == args.resume_budget
    )
    checkpoint_dir = repo_root / checkpoint["path"]

    tokenizer = ByteTokenizer()
    spec = _target_spec()
    init_spec = InitSpec()
    tokens_per_step = int(report["controls"]["tokens_per_optimizer_step"])
    max_steps = math.ceil(args.final_budget / tokens_per_step)
    trainer_config = _trainer_config(max_steps=max_steps, seed=args.seed)
    model = TwelveSixDecoder(spec, init_spec)
    trainer = Trainer(model, trainer_config, device="cpu")

    load_trainer_checkpoint(
        checkpoint_dir,
        model=model,
        trainer=trainer,
        restore_rng=True,
        expected_git_sha=args.source_sha,
        expected_model_spec_hash=report["model_identity_sha256"],
        expected_tokenizer_hash=BYTE_TOKENIZER_HASH,
        expected_tokenizer_vocab_hash=BYTE_VOCAB_HASH,
        expected_dataset_manifest_hash=report["data"]["dataset_manifest_sha256"],
        expected_run_manifest_hash=report["run_manifest_sha256"],
        expected_seed=args.seed,
    )
    if _model_state_sha256(model) != checkpoint["model_state_sha256"]:
        raise RuntimeError("fresh-process loaded model hash differs from retained checkpoint evidence")
    if _tree_sha256(trainer.state_dict()) != checkpoint["trainer_state_sha256"]:
        raise RuntimeError("fresh-process loaded trainer state differs from retained checkpoint evidence")

    validation_records = _read_jsonl(repo_root / "data/s0/packaged/validation.jsonl")
    before_eval_model = _model_state_sha256(model)
    before_eval_trainer = _tree_sha256(trainer.state_dict())
    before_eval_mode = model.training
    validation_loss, validation_tokens = _validation_loss(model, validation_records, tokenizer)
    after_eval_model = _model_state_sha256(model)
    after_eval_trainer = _tree_sha256(trainer.state_dict())
    after_eval_mode = model.training
    evaluation_non_mutating = (
        before_eval_model == after_eval_model
        and before_eval_trainer == after_eval_trainer
        and before_eval_mode == after_eval_mode
    )
    if not evaluation_non_mutating:
        raise RuntimeError("held-out evaluation mutated model/trainer/mode state")

    train_records = _read_jsonl(repo_root / "data/s0/packaged/train.jsonl")
    train_stream = _byte_stream(train_records, tokenizer)
    step_index = trainer.optimizer_step
    batch = _make_batch(
        train_stream,
        step=step_index,
        batch_size=int(report["controls"]["batch_size"]),
        sequence_length=int(report["controls"]["sequence_length"]),
    )
    before_resume_model = _model_state_sha256(model)
    before_steps = trainer.optimizer_step
    before_tokens = trainer.tokens_seen
    metrics = trainer.train_microbatch({"input_ids": batch})
    after_resume_model = _model_state_sha256(model)
    if trainer.optimizer_step != before_steps + 1:
        raise RuntimeError("fresh-process resume did not advance exactly one optimizer step")
    if after_resume_model == before_resume_model:
        raise RuntimeError("fresh-process resumed optimizer step did not change model state")

    evidence = {
        "schema_version": "12-6.milestone100-fresh-process-resume.v1",
        "fresh_process": True,
        "pid": os.getpid(),
        "parent_pid": os.getppid(),
        "python": platform.python_version(),
        "source_sha": args.source_sha,
        "seed": args.seed,
        "checkpoint_path": checkpoint["path"],
        "checkpoint_manifest_sha256": checkpoint["manifest_sha256"],
        "loaded_model_state_sha256": before_resume_model,
        "loaded_trainer_state_sha256": before_eval_trainer,
        "loaded_optimizer_steps": before_steps,
        "loaded_tokens_seen": before_tokens,
        "held_out_validation_loss": validation_loss,
        "held_out_validation_tokens": validation_tokens,
        "evaluation_non_mutating": evaluation_non_mutating,
        "evaluation_model_hash_before": before_eval_model,
        "evaluation_model_hash_after": after_eval_model,
        "evaluation_trainer_hash_before": before_eval_trainer,
        "evaluation_trainer_hash_after": after_eval_trainer,
        "evaluation_mode_before": before_eval_mode,
        "evaluation_mode_after": after_eval_mode,
        "continued_optimizer_steps": trainer.optimizer_step,
        "continued_tokens_seen": trainer.tokens_seen,
        "continued_train_loss": metrics.update_loss,
        "continued_grad_norm": metrics.grad_norm,
        "continued_model_state_sha256": after_resume_model,
        "dataset_manifest_sha256": _file_sha256(repo_root / "data/s0/packaged/manifest.json"),
        "status": "PASS",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
