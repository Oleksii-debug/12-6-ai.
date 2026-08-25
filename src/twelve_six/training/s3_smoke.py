"""Bounded ~10M executable smoke that reuses the proven SCALE-02 interfaces."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
from pathlib import Path
from typing import Any

import torch

from twelve_six.checkpoint import (
    CheckpointIdentity,
    hash_json,
    save_trainer_checkpoint,
    sha256_file,
)
from twelve_six.inference.contracts import GenerationConfig
from twelve_six.inference.first_party import load_first_party_backend
from twelve_six.inference.generation import generate
from twelve_six.model import TwelveSixDecoder, load_stage_config
from twelve_six.packing import PACKING_CONFIG_HASH, PACKING_VERSION
from twelve_six.tokenization import ByteTokenizer

from .config import TrainerConfig
from .s2_preflight import (
    _batch_for_step,
    _directory_bytes,
    _load_texts,
    _snapshot,
    _tensor_bytes,
    _weight_delta,
)
from .trainer import Trainer

SCHEMA = "12-6.s3-10m-executable-smoke.v1"
AUTHORITY = "ENGINEERING_EXECUTABLE_SMOKE_NOT_STAGE_EVIDENCE"
REPOSITORY = "Oleksii-debug/12-6-ai."
FIXTURE_SCOPE = "S0_CONTROLLED_FIXTURE_COMPATIBILITY_ONLY_NOT_S3_CORPUS_OR_TOKENIZER"
CANDIDATE_PATH = "configs/stages/alternatives/s3_10m_byte_gqa.candidate.json"
MODEL_SPEC_SHA256 = "61caa5469123e23b9b72fc2024140bfca84c4c480dcb0a7e712ba800a4f22998"
PARAMETER_COUNT = 10_000_640


def _git_head(repo_root: Path) -> str:
    value = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        text=True,
    ).strip()
    if len(value) != 40 or value != value.lower():
        raise ValueError("S3 smoke requires an exact lowercase Git SHA")
    return value


def _identity(
    *,
    candidate_sha: str,
    stage: Any,
    tokenizer: ByteTokenizer,
    dataset_manifest_sha256: str,
    train_sha256: str,
    environment_lock_sha256: str,
    trainer: Trainer,
) -> CheckpointIdentity:
    training_config = {
        "authority": AUTHORITY,
        "stage": "S3",
        "architecture_status": "ENGINEERING_CANDIDATE_NOT_FROZEN",
        "fixture_scope": FIXTURE_SCOPE,
        "s3_corpus_selected": False,
        "future_tokenizer_selected": False,
        "init_spec_sha256": stage.init.identity_sha256(),
        "training": {
            "seed": trainer.config.seed,
            "precision": trainer.config.precision,
            "max_steps": trainer.config.max_steps,
            "context_length": stage.model.max_seq_len,
        },
        "data": {
            "dataset_manifest_sha256": dataset_manifest_sha256,
            "split_identity": f"controlled-s0-train:{train_sha256}",
            "tokenizer_sha256": tokenizer.identity.config_sha256,
            "tokenizer_vocab_sha256": tokenizer.identity.vocab_sha256,
            "tokenizer_version": tokenizer.identity.version,
            "packing_sha256": PACKING_CONFIG_HASH,
            "packing_version": PACKING_VERSION,
        },
        "environment": {"lock_sha256": environment_lock_sha256},
    }
    run_manifest = {
        "schema": SCHEMA,
        "authority": AUTHORITY,
        "candidate_sha": candidate_sha,
        "model_spec_sha256": stage.model.identity_sha256(),
        "parameter_count": stage.expected_parameters,
        "fixture_scope": FIXTURE_SCOPE,
        "step": trainer.optimizer_step,
        "tokens_seen": trainer.tokens_seen,
    }
    return CheckpointIdentity(
        git_sha=candidate_sha,
        model_spec=stage.model.to_dict(),
        parameter_count=stage.expected_parameters,
        tokenizer_hash=tokenizer.identity.config_sha256,
        tokenizer_vocab_hash=tokenizer.identity.vocab_sha256,
        dataset_manifest_hash=dataset_manifest_sha256,
        run_manifest_hash=hash_json(run_manifest),
        training_config=training_config,
        seed=trainer.config.seed,
        precision=trainer.config.precision,
        step=trainer.optimizer_step,
        tokens_seen=trainer.tokens_seen,
        optimizer={
            "name": "AdamW",
            "lr": trainer.config.learning_rate,
            "betas": list(trainer.config.betas),
            "eps": trainer.config.eps,
            "weight_decay": trainer.config.weight_decay,
        },
        scheduler={"name": trainer.config.scheduler},
        environment_lock_hash=environment_lock_sha256,
    )


def collect_s3_10m_smoke(
    repo_root: Path,
    candidate_sha: str,
    output_dir: Path,
    *,
    seed: int = 20260825,
    sequence_length: int = 32,
    verify_checkout: bool = True,
) -> dict[str, Any]:
    """Execute one real 10M update plus checkpoint-to-first-party inference."""

    repo_root = repo_root.resolve()
    output_dir = output_dir.resolve()
    if verify_checkout and _git_head(repo_root) != candidate_sha:
        raise ValueError("candidate_sha does not equal checkout HEAD")
    stage = load_stage_config(repo_root / CANDIDATE_PATH)
    tokenizer = ByteTokenizer()
    if stage.stage != "S3" or stage.expected_parameters != PARAMETER_COUNT:
        raise ValueError("unexpected S3 10M candidate contract")
    if stage.model.identity_sha256() != MODEL_SPEC_SHA256:
        raise ValueError("S3 10M ModelSpec drift")
    if stage.model.vocab_size != tokenizer.vocab_size:
        raise ValueError("S3 current executable smoke requires byte-v1 vocab compatibility")
    if not 2 <= sequence_length <= stage.model.max_seq_len:
        raise ValueError("invalid S3 smoke sequence length")

    train_path = repo_root / "data/s0/packaged/train.jsonl"
    texts = _load_texts(train_path)
    dataset_manifest_sha256 = sha256_file(repo_root / "data/s0/packaged/manifest.json")
    train_sha256 = sha256_file(train_path)
    environment_lock_sha256 = sha256_file(repo_root / "requirements/locks/index.json")

    random.seed(seed)
    torch.manual_seed(seed)
    model = TwelveSixDecoder(stage.model, stage.init)
    actual_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if actual_parameters != PARAMETER_COUNT:
        raise RuntimeError("instantiated S3 trainable parameter count mismatch")
    before = _snapshot(model)
    trainer = Trainer(
        model,
        TrainerConfig(
            learning_rate=5e-4,
            weight_decay=0.0,
            max_steps=1,
            scheduler="constant",
            gradient_accumulation_steps=1,
            gradient_clip_norm=1.0,
            precision="fp32",
            seed=seed,
            deterministic_algorithms=True,
        ),
        device="cpu",
    )
    batch = _batch_for_step(
        texts,
        tokenizer,
        step=0,
        sequence_length=sequence_length,
    )
    metrics = trainer.train_microbatch(batch)
    if not metrics.optimizer_stepped or metrics.grad_norm is None:
        raise RuntimeError("S3 smoke failed to commit the optimizer update")
    delta = _weight_delta(model, before)
    if delta["changed_parameter_elements"] <= 0:
        raise RuntimeError("S3 smoke update did not change model weights")

    identity = _identity(
        candidate_sha=candidate_sha,
        stage=stage,
        tokenizer=tokenizer,
        dataset_manifest_sha256=dataset_manifest_sha256,
        train_sha256=train_sha256,
        environment_lock_sha256=environment_lock_sha256,
        trainer=trainer,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoint"
    manifest = save_trainer_checkpoint(
        checkpoint_dir,
        model=model,
        trainer=trainer,
        identity=identity,
    )
    backend = load_first_party_backend(checkpoint_dir)
    generation = generate(
        backend,
        "12-6",
        GenerationConfig(max_new_tokens=1, sample=False, seed=seed),
    )
    if len(generation.generated_token_ids) != 1:
        raise RuntimeError("S3 first-party generation did not emit one token")

    model_parameter_bytes = sum(
        parameter.numel() * parameter.element_size() for parameter in model.parameters()
    )
    optimizer_tensor_bytes = _tensor_bytes(trainer.optimizer.state_dict())
    evidence: dict[str, Any] = {
        "schema": SCHEMA,
        "authority": AUTHORITY,
        "repository": REPOSITORY,
        "candidate_sha": candidate_sha,
        "stage": "S3",
        "candidate_config": CANDIDATE_PATH,
        "architecture_status": "ENGINEERING_CANDIDATE_NOT_FROZEN",
        "model": {
            "model_spec_sha256": stage.model.identity_sha256(),
            "parameter_count": actual_parameters,
            "parameter_breakdown": stage.model.parameter_breakdown(),
            "vocab_size": stage.model.vocab_size,
            "max_context_tokens": stage.model.max_seq_len,
            "d_model": stage.model.d_model,
            "n_layers": stage.model.n_layers,
            "n_heads": stage.model.n_heads,
            "n_kv_heads": stage.model.n_kv_heads,
            "head_dim": stage.model.head_dim,
            "d_ff": stage.model.d_ff,
        },
        "training_smoke": {
            "precision": trainer.config.precision,
            "optimizer_steps": trainer.optimizer_step,
            "optimized_tokens": trainer.tokens_seen,
            "loss": metrics.loss,
            "grad_norm": metrics.grad_norm,
            "weight_delta": delta,
        },
        "checkpoint_inference_smoke": {
            "checkpoint_id": manifest["checkpoint_id"],
            "checkpoint_directory_bytes": _directory_bytes(checkpoint_dir),
            "model_parameter_bytes": model_parameter_bytes,
            "optimizer_tensor_bytes": optimizer_tensor_bytes,
            "backend": backend.diagnostics()["backend"],
            "generated_token_ids": list(generation.generated_token_ids),
        },
        "fixture": {
            "scope": FIXTURE_SCOPE,
            "tokenizer_version": tokenizer.identity.version,
            "tokenizer_vocab_size": tokenizer.vocab_size,
        },
        "blockers": {
            "meaningful_s3_training_experiment_ready": False,
            "production_s3_corpus_ready": False,
            "future_s3_tokenizer_selected": False,
        },
        "claims": {
            "paid_compute_used": False,
            "foreign_pretrained_weights_used": False,
            "instruction_or_alignment_training": False,
            "s3_quality_or_capability_evidence": False,
            "s3_corpus_or_tokenizer_frozen": False,
            "candidate_or_stable_promotion": False,
        },
    }
    evidence["evidence_sha256"] = hash_json(evidence)
    (output_dir / "s3-10m-executable-smoke.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return evidence


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run bounded SCALE-02 S3 ~10M smoke.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--candidate-sha", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--sequence-length", type=int, default=32)
    args = parser.parse_args(argv)
    evidence = collect_s3_10m_smoke(
        args.repo_root,
        args.candidate_sha,
        args.output_dir,
        seed=args.seed,
        sequence_length=args.sequence_length,
    )
    print(json.dumps(evidence, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
