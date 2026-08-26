#!/usr/bin/env python3
"""Exact RECOVER-171 launcher for the frozen RESEARCH-123 harness."""

from __future__ import annotations

import sys

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
            "packing_version": experiment.m100.PACKING_VERSION if hasattr(experiment, "m100") else "12-6.packing.v1",
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

if __name__ == "__main__":
    raise SystemExit(experiment.main())
