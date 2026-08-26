#!/usr/bin/env python3
"""Exact RECOVER-171 launcher for the frozen RESEARCH-123 harness."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import research123_data25_adapter as adapter

# The MILESTONE-150 ancestry intentionally does not carry TRAIN-53's old
# batch_noise_probe module. Install only the two compatibility symbols that the
# frozen RESEARCH-123 file imports, then configure the DATA-25 recovery contract.
adapter.install_batch_noise_probe_stub()

import research123_real_tn_scaling as experiment  # noqa: E402

adapter.configure_experiment(experiment)

# Make fresh-process resume return through this exact launcher.
experiment.__file__ = __file__


def checkpoint_identity(
    *,
    source_sha,
    spec,
    init_spec,
    tokenizer,
    data,
    run_manifest_hash,
    config,
    trainer,
    lock_hash,
):
    """Bind checkpoints to the actual seq128 DATA-25 recovery contract."""
    training_config = {
        "trainer": experiment.asdict(config),
        "init_spec_sha256": init_spec.identity_sha256(),
        "data": {
            "corpus_identity_sha256": data["corpus_identity_sha256"],
            "evaluation_subset_identity_sha256": data["dataset_identity_sha256"],
            "packing_version": adapter.m100.PACKING_VERSION,
            "sequence_length": adapter.SEQUENCE_LENGTH,
            "cross_document": False,
            "packing_sha256": data["training_trace_sha256"],
        },
    }
    return experiment.CheckpointIdentity(
        git_sha=source_sha,
        model_spec=spec.to_dict(),
        parameter_count=spec.parameter_count(),
        tokenizer_hash=tokenizer.identity.config_sha256,
        tokenizer_vocab_hash=tokenizer.identity.vocab_sha256,
        dataset_manifest_hash=str(data["intake_manifest_sha256"]),
        run_manifest_hash=run_manifest_hash,
        training_config=training_config,
        seed=config.seed,
        precision=config.precision,
        step=trainer.optimizer_step,
        tokens_seen=trainer.tokens_seen,
        optimizer={
            "name": "AdamW",
            "learning_rate": config.learning_rate,
            "betas": list(config.betas),
            "eps": config.eps,
            "weight_decay": config.weight_decay,
            "gradient_clip_norm": config.gradient_clip_norm,
        },
        scheduler={"name": config.scheduler, "warmup_steps": config.warmup_steps},
        environment_lock_hash=lock_hash,
    )


experiment._checkpoint_identity = checkpoint_identity

if "--resume-child" in sys.argv:
    incumbent_trainer_config = experiment.TrainerConfig

    def normalized_trainer_config(**kwargs):
        if isinstance(kwargs.get("betas"), list):
            kwargs["betas"] = tuple(kwargs["betas"])
        return incumbent_trainer_config(**kwargs)

    experiment.TrainerConfig = normalized_trainer_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-sha")
    parser.add_argument("--output-dir", type=Path, default=Path("evidence/recover171-research123"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--resume-child", type=Path)
    parser.add_argument("--validate", type=Path)
    args = parser.parse_args()

    if args.resume_child is not None:
        return experiment._run_resume_child(args.resume_child)
    if args.validate is not None:
        report = json.loads(args.validate.read_text(encoding="utf-8"))
        experiment.validate_report(report)
        print(json.dumps({"status": "PASS", "report_sha256": report["report_sha256"]}, sort_keys=True))
        return 0

    experiment._require(args.source_sha is not None, "--source-sha is required")
    experiment._require(args.torch_threads >= 1, "--torch-threads must be >=1")
    report = experiment.run_experiment(
        source_sha=args.source_sha,
        output_dir=args.output_dir,
        torch_threads=args.torch_threads,
    )
    experiment.validate_report(report)
    output = args.output or (args.output_dir / "research123-data25-tn-scaling.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "PASS_DATA25_BOUNDED_ONLY",
                "report": str(output),
                "report_sha256": report["report_sha256"],
                "selected": report["selected_learned_base_checkpoint"],
                "recommendation": report["recommendation"],
                "ten_million_status": report["ten_million_status"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
