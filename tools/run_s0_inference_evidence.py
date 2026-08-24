"""Train, checkpoint, reload, and retain exact S0 first-party inference evidence."""

from __future__ import annotations

import argparse
import json
import subprocess
from itertools import cycle, islice
from pathlib import Path

import torch

from twelve_six.checkpoint import bind_checkpoint_identity, hash_json, save_trainer_checkpoint, sha256_file
from twelve_six.inference.evidence import (
    collect_first_party_inference_evidence,
    validate_first_party_inference_evidence,
)
from twelve_six.inference.first_party import load_first_party_backend
from twelve_six.integration.s0_runtime import S0TorchInferenceBackend
from twelve_six.model import TwelveSixDecoder, load_stage_config
from twelve_six.packing import (
    PACKING_CONFIG_HASH,
    PACKING_VERSION,
    batch_examples,
    collate_rows,
    iter_packed_examples,
    load_jsonl_records,
)
from twelve_six.tokenization import ByteTokenizer
from twelve_six.training import Trainer, TrainerConfig
from twelve_six.training.s0_evidence import DATASET_MANIFEST_SHA256, TRAIN_JSONL_SHA256


def _git_head(root: Path) -> str:
    try:
        value = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("D05 inference evidence requires a Git checkout") from exc
    return value


def _validate_source_sha(value: str) -> None:
    if len(value) != 40 or value != value.lower() or any(ch not in "0123456789abcdef" for ch in value):
        raise ValueError("source_sha must be a full lowercase 40-hex Git SHA")


def _train_batches(
    root: Path,
    tokenizer: ByteTokenizer,
    *,
    batch_size: int,
) -> list[dict[str, torch.Tensor]]:
    records = tuple(load_jsonl_records(root / "data/s0/packaged/train.jsonl", split="train"))
    examples = tuple(
        iter_packed_examples(
            records,
            tokenizer,
            expected_split="train",
            sequence_length=128,
        )
    )
    if not examples:
        raise RuntimeError("canonical S0 train split produced no packed examples")
    output: list[dict[str, torch.Tensor]] = []
    for group in batch_examples(examples, batch_size=batch_size, drop_last=False):
        rows = collate_rows(group, target_mode="labels")
        output.append(
            {
                "input_ids": torch.tensor(rows["input_ids"], dtype=torch.long),
                "labels": torch.tensor(rows["labels"], dtype=torch.long),
            }
        )
    return output


def run(
    root: Path,
    *,
    source_sha: str,
    output_dir: Path,
    seed: int,
    train_steps: int,
    batch_size: int,
) -> dict[str, object]:
    root = root.resolve()
    _validate_source_sha(source_sha)
    if _git_head(root) != source_sha:
        raise ValueError("source_sha does not equal exact checkout HEAD")
    if train_steps <= 0:
        raise ValueError("train_steps must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    stage = load_stage_config(root / "configs/stages/s0_10k.json")
    tokenizer = ByteTokenizer()
    if stage.canonical_base != "random_init":
        raise RuntimeError("canonical S0 Base must remain random_init")
    if stage.expected_parameters != 10_140 or stage.model.vocab_size != tokenizer.vocab_size:
        raise RuntimeError("canonical S0 model/tokenizer identity drift")

    dataset_manifest = root / "data/s0/packaged/manifest.json"
    train_split = root / "data/s0/packaged/train.jsonl"
    environment_lock = root / "requirements/locks/index.json"
    dataset_manifest_sha = sha256_file(dataset_manifest)
    train_split_sha = sha256_file(train_split)
    environment_lock_sha = sha256_file(environment_lock)
    if dataset_manifest_sha != DATASET_MANIFEST_SHA256:
        raise RuntimeError("canonical D03 dataset manifest identity drift")
    if train_split_sha != TRAIN_JSONL_SHA256:
        raise RuntimeError("canonical D03 train split identity drift")

    config = TrainerConfig(
        learning_rate=3e-2,
        weight_decay=0.0,
        max_steps=train_steps,
        scheduler="constant",
        gradient_accumulation_steps=1,
        gradient_clip_norm=1.0,
        precision="fp32",
        seed=seed,
        deterministic_algorithms=True,
        deterministic_warn_only=False,
    )
    batches = _train_batches(root, tokenizer, batch_size=batch_size)

    torch.manual_seed(seed)
    model = TwelveSixDecoder(stage.model, stage.init)
    trainer = Trainer(model, config, device="cpu")
    result = trainer.run(islice(cycle(batches), train_steps))
    trainer.assert_checkpoint_safe()
    if result.optimizer_steps_completed != train_steps:
        raise RuntimeError("S0 inference evidence training did not reach requested steps")

    run_manifest = {
        "schema_version": 1,
        "run_id": f"s0-d05-inference-evidence-{source_sha[:12]}",
        "stage": "S0",
        "run_kind": "first_party_inference_evidence",
        "state": "COMPLETED_LOCAL_FREE",
        "candidate": {
            "repository": "Oleksii-debug/12-6-ai.",
            "git_sha": source_sha,
            "branch_or_tag": "exact-checkout",
            "modelspec_sha256": hash_json(stage.model.to_dict()),
            "initspec_sha256": hash_json(stage.init.to_dict()),
            "parameter_count": stage.expected_parameters,
        },
        "data": {
            "dataset_manifest_sha256": dataset_manifest_sha,
            "tokenizer_sha256": tokenizer.identity.config_sha256,
            "tokenizer_vocab_sha256": tokenizer.identity.vocab_sha256,
            "tokenizer_version": tokenizer.identity.version,
            "split_identity": f"train:{train_split_sha}",
            "packing_sha256": PACKING_CONFIG_HASH,
            "packing_version": PACKING_VERSION,
        },
        "training": {
            "seed": seed,
            "device": "cpu",
            "precision": "fp32",
            "optimizer": {
                "name": "AdamW",
                "lr": config.learning_rate,
                "betas": list(config.betas),
                "eps": config.eps,
                "weight_decay": config.weight_decay,
            },
            "scheduler": {"name": config.scheduler},
            "context_length": stage.model.max_seq_len,
            "target_steps": train_steps,
            "optimized_tokens": trainer.tokens_seen,
            "batch_size_examples": batch_size,
        },
        "environment": {"lock_sha256": environment_lock_sha},
    }
    identity = bind_checkpoint_identity(
        run_manifest=run_manifest,
        model_spec=stage.model.to_dict(),
        init_spec=stage.init.to_dict(),
        tokenizer_identity=tokenizer.identity.to_dict(),
        packing_identity={
            "version": PACKING_VERSION,
            "config_sha256": PACKING_CONFIG_HASH,
        },
        step=trainer.optimizer_step,
        tokens_seen=trainer.tokens_seen,
        environment_lock_hash=environment_lock_sha,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "checkpoint"
    manifest = save_trainer_checkpoint(
        checkpoint,
        model=model,
        trainer=trainer,
        identity=identity,
    )
    direct = S0TorchInferenceBackend(model, tokenizer)
    reloaded = load_first_party_backend(checkpoint)
    evidence = collect_first_party_inference_evidence(
        direct,
        reloaded,
        prompts=("12-6", "Base"),
        seed=17,
        max_new_tokens=6,
    )
    diagnostics = evidence["backend_diagnostics"]
    if diagnostics["git_sha"] != source_sha:
        raise RuntimeError("reloaded checkpoint Git identity diverged from exact source")
    if diagnostics["checkpoint_id"] != manifest["checkpoint_id"]:
        raise RuntimeError("reloaded checkpoint ID diverged from saved manifest")
    if diagnostics["step"] != train_steps:
        raise RuntimeError("reloaded checkpoint step diverged from completed training")
    validate_first_party_inference_evidence(evidence)

    evidence_path = output_dir / "inference_evidence.json"
    evidence_path.write_text(
        json.dumps(evidence, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return evidence


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run exact S0 trained+reloaded first-party inference evidence"
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--train-steps", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    evidence = run(
        args.repo_root,
        source_sha=args.source_sha,
        output_dir=args.output_dir,
        seed=args.seed,
        train_steps=args.train_steps,
        batch_size=args.batch_size,
    )
    diagnostics = evidence["backend_diagnostics"]
    if args.json:
        print(json.dumps(evidence, ensure_ascii=False, sort_keys=True, allow_nan=False))
    else:
        print(
            "D05 inference evidence: PASS "
            f"checkpoint_id={diagnostics['checkpoint_id']} "
            f"source_sha={diagnostics['git_sha']} "
            f"step={diagnostics['step']} tokens_seen={diagnostics['tokens_seen']} "
            f"evidence_sha256={evidence['evidence_sha256']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
