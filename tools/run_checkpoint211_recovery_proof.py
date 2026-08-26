#!/usr/bin/env python3
"""Execute bounded CHECKPOINT-211 recovery-lifecycle proof; never runs 10M training."""
from __future__ import annotations

import argparse
import copy
import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import nn

from twelve_six.checkpoint import (
    CheckpointIdentity,
    CheckpointIntegrityError,
    hash_json,
    load_trainer_checkpoint,
    save_trainer_checkpoint,
    sha256_file,
    verify_checkpoint,
)
from twelve_six.scale141_recovery import (
    RecoveryPointerUpdateInterrupted,
    cleanup_recovery_generations,
    publish_recovery_generation,
    resolve_recovery_generation,
)
from twelve_six.training import Trainer, TrainerConfig

SCHEMA = "12-6.checkpoint211-recovery-proof.v1"
RUN_HASH = hash_json({"worker": "CHECKPOINT-211-SCALE141-RECOVERY-IMMUTABILITY", "proof": "bounded"})
TOKENIZER_HASH = "5" * 64
VOCAB_HASH = "6" * 64
DATA_HASH = "7" * 64


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


def _identity(
    source_sha: str,
    environment_hash: str,
    model: TinyLM,
    trainer: Trainer,
    cfg: TrainerConfig,
) -> CheckpointIdentity:
    return CheckpointIdentity(
        git_sha=source_sha,
        model_spec={"kind": "checkpoint211-tiny-lm", "vocab": 16, "width": 8},
        parameter_count=sum(p.numel() for p in model.parameters()),
        tokenizer_hash=TOKENIZER_HASH,
        tokenizer_vocab_hash=VOCAB_HASH,
        dataset_manifest_hash=DATA_HASH,
        run_manifest_hash=RUN_HASH,
        training_config={"trainer": asdict(cfg), "proof": "checkpoint211"},
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
        environment_lock_hash=environment_hash,
    )


def _save(
    path: Path,
    *,
    source_sha: str,
    environment_hash: str,
    model: TinyLM,
    trainer: Trainer,
    cfg: TrainerConfig,
    overwrite: bool = False,
):
    return save_trainer_checkpoint(
        path,
        model=model,
        trainer=trainer,
        identity=_identity(source_sha, environment_hash, model, trainer, cfg),
        overwrite=overwrite,
    )


def _publish(
    root: Path,
    *,
    source_sha: str,
    environment_hash: str,
    model: TinyLM,
    trainer: Trainer,
    cfg: TrainerConfig,
    failpoint: str | None = None,
):
    return publish_recovery_generation(
        root,
        save_generation=lambda destination: _save(
            destination,
            source_sha=source_sha,
            environment_hash=environment_hash,
            model=model,
            trainer=trainer,
            cfg=cfg,
        ),
        expected_source_sha=source_sha,
        expected_run_manifest_hash=RUN_HASH,
        expected_step=trainer.optimizer_step,
        expected_tokens_seen=trainer.tokens_seen,
        failpoint=failpoint,
    )


def _step(trainer: Trainer, offset: int) -> None:
    values = torch.tensor([[1 + offset, 2 + offset, 3 + offset, 4 + offset]]) % 16
    metrics = trainer.train_microbatch({"input_ids": values})
    if not metrics.optimizer_stepped:
        raise RuntimeError("bounded proof expected one committed optimizer step")


def _hash_checkpoint_files(path: Path) -> dict[str, str]:
    return {
        child.name: sha256_file(child)
        for child in sorted(path.iterdir())
        if child.is_file()
    }


def _load_environment_hash(path: Path) -> str:
    value = json.loads(path.read_text(encoding="utf-8"))
    identity = value.get("identity_sha256")
    if not isinstance(identity, str) or len(identity) != 64:
        raise RuntimeError("universal bootstrap environment manifest identity missing")
    return identity


def run(source_sha: str, environment_manifest: Path, output: Path) -> dict[str, Any]:
    if len(source_sha) != 40:
        raise RuntimeError("proof requires exact 40-character source SHA")
    environment_hash = _load_environment_hash(environment_manifest)
    work = output / "work"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    model, trainer, cfg = _stack()

    _step(trainer, 0)
    legacy = work / "legacy-recovery-latest"
    _save(
        legacy,
        source_sha=source_sha,
        environment_hash=environment_hash,
        model=model,
        trainer=trainer,
        cfg=cfg,
        overwrite=True,
    )
    historical_failure = None
    try:
        _save(
            legacy,
            source_sha=source_sha,
            environment_hash=environment_hash,
            model=model,
            trainer=trainer,
            cfg=cfg,
            overwrite=True,
        )
    except FileExistsError as exc:
        historical_failure = str(exc)
    if historical_failure is None or "checkpoint-v1 is immutable" not in historical_failure:
        raise RuntimeError("historical recovery-latest overwrite failure was not reproduced")

    root = work / "recovery"
    first = _publish(
        root,
        source_sha=source_sha,
        environment_hash=environment_hash,
        model=model,
        trainer=trainer,
        cfg=cfg,
    )
    first_path = root / "generations" / "generation-00000001"
    first_hashes_before = _hash_checkpoint_files(first_path)

    _step(trainer, 1)
    second = _publish(
        root,
        source_sha=source_sha,
        environment_hash=environment_hash,
        model=model,
        trainer=trainer,
        cfg=cfg,
    )
    first_hashes_after = _hash_checkpoint_files(first_path)
    if first_hashes_before != first_hashes_after:
        raise RuntimeError("older committed recovery generation mutated")

    _step(trainer, 2)
    interrupted = False
    try:
        _publish(
            root,
            source_sha=source_sha,
            environment_hash=environment_hash,
            model=model,
            trainer=trainer,
            cfg=cfg,
            failpoint="before_pointer_replace",
        )
    except RecoveryPointerUpdateInterrupted:
        interrupted = True
    if not interrupted:
        raise RuntimeError("pointer interruption failpoint did not fire")
    after_interruption = resolve_recovery_generation(root, expected_reference=second)
    orphan_path = root / "generations" / "generation-00000003"
    orphan_manifest = verify_checkpoint(orphan_path)

    corrupt_newer = root / "generations" / "generation-00000004"
    corrupt_newer.mkdir()
    (corrupt_newer / "INCOMPLETE").write_text("not committed\n", encoding="utf-8")
    fallback = resolve_recovery_generation(root, expected_reference=second)

    third = _publish(
        root,
        source_sha=source_sha,
        environment_hash=environment_hash,
        model=model,
        trainer=trainer,
        cfg=cfg,
    )
    if third["generation"] != 5:
        raise RuntimeError("generation numbering reused an orphan/incomplete generation")

    current = resolve_recovery_generation(
        root,
        expected_reference=third,
        expected_source_sha=source_sha,
        expected_run_manifest_hash=RUN_HASH,
        expected_step=trainer.optimizer_step,
        expected_tokens_seen=trainer.tokens_seen,
    )
    expected_optimizer = copy.deepcopy(trainer.optimizer.state_dict())
    rng_expected = torch.rand(4)
    torch.manual_seed(999)
    fresh_model, fresh_trainer, _ = _stack()
    torch.manual_seed(999)
    loaded = load_trainer_checkpoint(
        current.path,
        model=fresh_model,
        trainer=fresh_trainer,
        restore_rng=True,
        expected_git_sha=source_sha,
        expected_run_manifest_hash=RUN_HASH,
        expected_tokenizer_hash=TOKENIZER_HASH,
        expected_tokenizer_vocab_hash=VOCAB_HASH,
        expected_dataset_manifest_hash=DATA_HASH,
        expected_environment_lock_hash=environment_hash,
        expected_seed=cfg.seed,
    )
    if fresh_trainer.optimizer_step != trainer.optimizer_step:
        raise RuntimeError("fresh-process proof optimizer counter mismatch")
    if fresh_trainer.tokens_seen != trainer.tokens_seen:
        raise RuntimeError("fresh-process proof optimized-token counter mismatch")
    if not fresh_trainer.optimizer.state_dict()["state"]:
        raise RuntimeError("optimizer state was not restored")
    if fresh_trainer.optimizer.state_dict()["state"].keys() != expected_optimizer["state"].keys():
        raise RuntimeError("optimizer state key mismatch")
    if not torch.equal(torch.rand(4), rng_expected):
        raise RuntimeError("RNG state did not restore exactly")

    cleanup = cleanup_recovery_generations(root, keep=1)
    post_cleanup = resolve_recovery_generation(root, expected_reference=third)

    corrupt_root = work / "corrupt-current"
    corrupt_model, corrupt_trainer, corrupt_cfg = _stack()
    _step(corrupt_trainer, 0)
    corrupt_ref = _publish(
        corrupt_root,
        source_sha=source_sha,
        environment_hash=environment_hash,
        model=corrupt_model,
        trainer=corrupt_trainer,
        cfg=corrupt_cfg,
    )
    corrupt_path = corrupt_root / "generations" / "generation-00000001" / "weights.safetensors"
    with corrupt_path.open("ab") as handle:
        handle.write(b"tamper")
    corrupt_current_failed_closed = False
    try:
        resolve_recovery_generation(corrupt_root, expected_reference=corrupt_ref)
    except CheckpointIntegrityError:
        corrupt_current_failed_closed = True
    if not corrupt_current_failed_closed:
        raise RuntimeError("corrupt current recovery target did not fail closed")

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "worker": "CHECKPOINT-211-SCALE141-RECOVERY-IMMUTABILITY",
        "source_sha": source_sha,
        "environment_manifest": str(environment_manifest),
        "environment_identity_sha256": environment_hash,
        "launch_gate_required": True,
        "full_10m_retraining_performed": False,
        "paid_compute": False,
        "foreign_weights": False,
        "historical_failure": {
            "reproduced": True,
            "error": historical_failure,
            "legacy_checkpoint_id": verify_checkpoint(legacy)["checkpoint_id"],
        },
        "generation_proof": {
            "first": first,
            "second": second,
            "interrupted_generation": {
                "generation": 3,
                "checkpoint_id": orphan_manifest["checkpoint_id"],
                "pointer_remained_generation": after_interruption.reference["generation"],
            },
            "incomplete_newer_generation": {
                "generation": 4,
                "pointer_fallback_generation": fallback.reference["generation"],
            },
            "current": third,
            "older_generation_bytes_unchanged": first_hashes_before == first_hashes_after,
        },
        "fresh_load": {
            "checkpoint_id": loaded.manifest["checkpoint_id"],
            "optimizer_step": fresh_trainer.optimizer_step,
            "tokens_seen": fresh_trainer.tokens_seen,
            "optimizer_state_restored": True,
            "rng_restored": True,
            "source_run_environment_bound": True,
        },
        "cleanup": cleanup,
        "post_cleanup_checkpoint_id": post_cleanup.reference["checkpoint_id"],
        "corrupt_current_generation_fail_closed": corrupt_current_failed_closed,
    }
    report["identity_sha256"] = hash_json(report)
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "recovery-proof.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--environment-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(
        args.source_sha,
        args.environment_manifest.resolve(),
        args.output.resolve(),
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
