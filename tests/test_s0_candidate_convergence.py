from __future__ import annotations

import hashlib
import json
import math
import random
import subprocess
from pathlib import Path

import numpy as np
import torch

from twelve_six.checkpoint import (
    bind_checkpoint_identity,
    hash_json,
    load_trainer_checkpoint,
    save_trainer_checkpoint,
    sha256_file,
)
from twelve_six.inference import GenerationConfig, generate
from twelve_six.inference.first_party import load_first_party_backend
from twelve_six.integration import S0TorchInferenceBackend
from twelve_six.model import TwelveSixDecoder, load_stage_config
from twelve_six.packing import PACKING_CONFIG_HASH, PACKING_VERSION
from twelve_six.stage_gates import evaluate_s0_integrated
from twelve_six.tokenization import ByteTokenizer
from twelve_six.training import Trainer, TrainerConfig, causal_lm_loss

ROOT = Path(__file__).resolve().parents[1]
COMPOSITION = ROOT / "configs/releases/s0_candidate_convergence_20260824.experimental.json"


def _git_head() -> str:
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    assert len(head) == 40
    assert head == head.lower()
    assert all(ch in "0123456789abcdef" for ch in head)
    return head


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _load_rows(path: Path) -> list[dict[str, object]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows
    return rows


def _batches(rows: list[dict[str, object]], tokenizer: ByteTokenizer, max_seq_len: int):
    batches = []
    for row in rows:
        ids = tokenizer.encode(str(row["text"]))[:max_seq_len]
        assert len(ids) >= 2
        tensor = torch.tensor([ids], dtype=torch.long)
        batches.append({"input_ids": tensor, "labels": tensor})
    return batches


@torch.no_grad()
def _mean_loss(model: TwelveSixDecoder, batches) -> float:
    model.eval()
    values = []
    for batch in batches:
        loss = causal_lm_loss(model(batch["input_ids"]).logits, batch["labels"])
        value = float(loss.item())
        assert math.isfinite(value)
        values.append(value)
    return sum(values) / len(values)


def _train_range(trainer: Trainer, batches, start: int, end: int) -> None:
    for step in range(start, end):
        metrics = trainer.train_microbatch(batches[step % len(batches)])
        assert metrics.optimizer_stepped is True
        assert math.isfinite(metrics.loss)
    trainer.assert_checkpoint_safe()


def _run_manifest(
    *,
    head: str,
    stage,
    tokenizer: ByteTokenizer,
    trainer_config: TrainerConfig,
    dataset_manifest_sha256: str,
    train_sha256: str,
    environment_lock_sha256: str,
    target_steps: int,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "run_id": f"s0-d01-candidate-{head[:12]}",
        "stage": "S0",
        "run_kind": "integrated_training_evaluation",
        "state": "RUNNING",
        "candidate": {
            "repository": "Oleksii-debug/12-6-ai.",
            "git_sha": head,
            "branch_or_tag": "d01/s0-candidate-convergence-20260824-b",
            "modelspec_sha256": hash_json(stage.model.to_dict()),
            "initspec_sha256": hash_json(stage.init.to_dict()),
            "parameter_count": stage.expected_parameters,
        },
        "data": {
            "dataset_manifest_sha256": dataset_manifest_sha256,
            "tokenizer_sha256": tokenizer.identity.config_sha256,
            "tokenizer_vocab_sha256": tokenizer.identity.vocab_sha256,
            "tokenizer_version": tokenizer.identity.version,
            "split_identity": f"train:{train_sha256}",
            "packing_sha256": PACKING_CONFIG_HASH,
            "packing_version": PACKING_VERSION,
        },
        "training": {
            "seed": trainer_config.seed,
            "device": "cpu",
            "precision": "fp32",
            "optimizer": {
                "name": "AdamW",
                "lr": trainer_config.learning_rate,
                "betas": list(trainer_config.betas),
                "eps": trainer_config.eps,
                "weight_decay": trainer_config.weight_decay,
            },
            "scheduler": {"name": trainer_config.scheduler},
            "context_length": stage.model.max_seq_len,
            "global_batch_tokens": 1,
            "target_steps": target_steps,
            "target_tokens": 1,
            "checkpoint_interval_steps": target_steps // 2,
        },
        "environment": {"lock_sha256": environment_lock_sha256},
    }


def _bind_identity(
    *,
    run_manifest: dict[str, object],
    stage,
    tokenizer: ByteTokenizer,
    trainer: Trainer,
    environment_lock_sha256: str,
):
    return bind_checkpoint_identity(
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
        environment_lock_hash=environment_lock_sha256,
    )


def test_candidate_manifest_binds_exact_green_parent_ancestry() -> None:
    payload = json.loads(COMPOSITION.read_text(encoding="utf-8"))
    head = _git_head()

    assert payload["status"] == "experimental"
    assert payload["composition_complete"] is True
    assert payload["promotion_eligible"] is False
    assert payload["canonical_base"] == "random_init_pretraining_only"
    assert {row["lane"] for row in payload["components"]} == {
        "D01", "D02", "D03", "D04", "D05", "D06", "D07", "D08"
    }
    assert all(row["ci_conclusion"] == "success" for row in payload["components"])
    assert all(not row["contains_behavioral_weights"] for row in payload["components"])
    assert all(not row["contains_foreign_pretrained_weights"] for row in payload["components"])
    assert payload["audits"]["AUDIT-A"]["verdict"] == "CHANGES_REQUIRED"
    assert payload["audits"]["AUDIT-B"]["verdict"] == "CHANGES_REQUIRED"

    for parent in payload["required_git_ancestry"]:
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", parent, head],
            cwd=ROOT,
            check=True,
        )


def test_exact_head_train_checkpoint_resume_eval_and_first_party_inference(
    tmp_path: Path,
) -> None:
    head = _git_head()
    stage = load_stage_config(ROOT / "configs/stages/s0_10k.json")
    tokenizer = ByteTokenizer()
    train_path = ROOT / "data/s0/packaged/train.jsonl"
    validation_path = ROOT / "data/s0/packaged/validation.jsonl"
    dataset_manifest_path = ROOT / "data/s0/packaged/manifest.json"
    environment_lock_path = ROOT / "requirements/locks/index.json"

    assert stage.canonical_base == "random_init"
    assert stage.expected_parameters == 10_140
    assert stage.model.vocab_size == tokenizer.vocab_size == 256

    train_rows = _load_rows(train_path)
    validation_rows = _load_rows(validation_path)
    train_batches = _batches(train_rows, tokenizer, stage.model.max_seq_len)
    validation_batches = _batches(validation_rows, tokenizer, stage.model.max_seq_len)

    seed = 20260824
    total_steps = 40
    split_step = total_steps // 2
    trainer_config = TrainerConfig(
        learning_rate=3e-2,
        weight_decay=0.0,
        max_steps=total_steps,
        scheduler="constant",
        gradient_accumulation_steps=1,
        gradient_clip_norm=1.0,
        precision="fp32",
        seed=seed,
        deterministic_algorithms=True,
    )

    _seed_all(seed)
    control_model = TwelveSixDecoder(stage.model, stage.init)
    random_train_loss = _mean_loss(control_model, train_batches)
    random_validation_loss = _mean_loss(control_model, validation_batches)
    control_trainer = Trainer(control_model, trainer_config, device="cpu")
    _train_range(control_trainer, train_batches, 0, total_steps)
    trained_train_loss = _mean_loss(control_model, train_batches)
    trained_validation_loss = _mean_loss(control_model, validation_batches)

    assert trained_train_loss < random_train_loss
    assert trained_validation_loss < random_validation_loss

    _seed_all(seed)
    partial_model = TwelveSixDecoder(stage.model, stage.init)
    partial_trainer = Trainer(partial_model, trainer_config, device="cpu")
    _train_range(partial_trainer, train_batches, 0, split_step)

    dataset_manifest_sha256 = sha256_file(dataset_manifest_path)
    train_sha256 = sha256_file(train_path)
    environment_lock_sha256 = sha256_file(environment_lock_path)
    run_manifest = _run_manifest(
        head=head,
        stage=stage,
        tokenizer=tokenizer,
        trainer_config=trainer_config,
        dataset_manifest_sha256=dataset_manifest_sha256,
        train_sha256=train_sha256,
        environment_lock_sha256=environment_lock_sha256,
        target_steps=total_steps,
    )
    split_identity = _bind_identity(
        run_manifest=run_manifest,
        stage=stage,
        tokenizer=tokenizer,
        trainer=partial_trainer,
        environment_lock_sha256=environment_lock_sha256,
    )

    split_checkpoint = tmp_path / "split-checkpoint"
    split_manifest = save_trainer_checkpoint(
        split_checkpoint,
        model=partial_model,
        trainer=partial_trainer,
        identity=split_identity,
    )
    assert split_manifest["serialization"]["pickle"] is False

    restored_model = TwelveSixDecoder(stage.model, stage.init)
    restored_trainer = Trainer(restored_model, trainer_config, device="cpu")
    loaded = load_trainer_checkpoint(
        split_checkpoint,
        model=restored_model,
        trainer=restored_trainer,
        restore_rng=True,
        expected_git_sha=head,
        expected_model_spec_hash=hash_json(stage.model.to_dict()),
        expected_init_spec_hash=hash_json(stage.init.to_dict()),
        expected_tokenizer_hash=tokenizer.identity.config_sha256,
        expected_tokenizer_vocab_hash=tokenizer.identity.vocab_sha256,
        expected_dataset_manifest_hash=dataset_manifest_sha256,
        expected_split_identity=f"train:{train_sha256}",
        expected_packing_hash=PACKING_CONFIG_HASH,
        expected_packing_version=PACKING_VERSION,
        expected_run_manifest_hash=hash_json(run_manifest),
        expected_training_config_hash=split_manifest["identity"]["training_config_hash"],
        expected_environment_lock_hash=environment_lock_sha256,
        expected_seed=seed,
    )
    assert loaded.manifest["identity"]["git_sha"] == head
    assert restored_trainer.optimizer_step == split_step

    _train_range(restored_trainer, train_batches, split_step, total_steps)
    assert restored_trainer.optimizer_step == control_trainer.optimizer_step == total_steps
    assert restored_trainer.tokens_seen == control_trainer.tokens_seen
    for name, actual in restored_model.state_dict().items():
        torch.testing.assert_close(actual, control_model.state_dict()[name], rtol=0.0, atol=0.0)

    final_identity = _bind_identity(
        run_manifest=run_manifest,
        stage=stage,
        tokenizer=tokenizer,
        trainer=restored_trainer,
        environment_lock_sha256=environment_lock_sha256,
    )
    final_checkpoint = tmp_path / "final-checkpoint"
    save_trainer_checkpoint(
        final_checkpoint,
        model=restored_model,
        trainer=restored_trainer,
        identity=final_identity,
    )

    direct_backend = S0TorchInferenceBackend(restored_model, tokenizer)
    reloaded_backend = load_first_party_backend(final_checkpoint)
    generation_config = GenerationConfig(max_new_tokens=8, sample=False, seed=seed)
    direct_generation = generate(direct_backend, "12-6", generation_config)
    reloaded_generation = generate(reloaded_backend, "12-6", generation_config)
    assert reloaded_generation == direct_generation
    assert len(reloaded_generation.generated_token_ids) == 8

    train_hashes = {str(row["content_sha256"]) for row in train_rows}
    validation_hashes = {str(row["content_sha256"]) for row in validation_rows}
    contamination = json.loads(
        (ROOT / "data/s0/contamination_registry.json").read_text(encoding="utf-8")
    )
    forbidden_hashes = set(contamination.get("forbidden_normalized_sha256", []))
    source_registry = json.loads(
        (ROOT / "data/s0/source_registry.json").read_text(encoding="utf-8")
    )
    forbidden_purposes = set(contamination.get("forbidden_source_purposes", []))
    forbidden_source_count = sum(
        1
        for source in source_registry.get("sources", [])
        if isinstance(source, dict) and source.get("purpose") in forbidden_purposes
    )
    generated_ids = list(reloaded_generation.generated_token_ids)
    generation_hash = hashlib.sha256(
        json.dumps(generated_ids, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    evidence = {
        "candidate": {
            "sha": head,
            "id": f"s0-d01-convergence@{head}",
            "random_init": True,
            "model_constructed": True,
            "parameter_count": stage.expected_parameters,
            "model_vocab_size": stage.model.vocab_size,
        },
        "tokenizer": {
            "identity": tokenizer.identity.config_sha256,
            "vocab_size": tokenizer.vocab_size,
            "max_token_id": tokenizer.vocab_size - 1,
        },
        "eval_config": {"id": "d01-exact-head-integrated-contract-v1"},
        "dataset": {
            "identity": json.loads(dataset_manifest_path.read_text(encoding="utf-8"))[
                "dataset_identity_sha256"
            ],
            "heldout_used_for_training": False,
            "train_validation_overlap": len(train_hashes & validation_hashes),
            "validation_examples": len(validation_rows),
            "distinct_train_batches": len(train_batches),
        },
        "metrics": {
            "train_loss_before": random_train_loss,
            "train_loss_after": trained_train_loss,
            "validation_loss_before": random_validation_loss,
            "validation_loss_after": trained_validation_loss,
            "random_validation_loss": random_validation_loss,
            "trained_validation_loss": trained_validation_loss,
        },
        "generation_probes": [
            {
                "id": "first-party-final-checkpoint-greedy",
                "token_count": len(generated_ids),
                "output_sha256": generation_hash,
                "seed": seed,
                "sampler": "greedy",
            }
        ],
        "checkpoint": {"save_load_verified": True, "resume_verified": True},
        "contamination": {
            "checked": True,
            "benchmark_overlap_count": len(train_hashes & forbidden_hashes)
            + forbidden_source_count,
            "heldout_overlap_count": len(train_hashes & validation_hashes),
        },
        "regressions": {"executed": True, "failures": 0},
    }
    gate = evaluate_s0_integrated(evidence)
    assert gate["summary"]["counts"] == {"FAIL": 0, "NOT_TESTED": 0, "PASS": 15}
    assert gate["summary"]["evaluation_complete"] is True
    assert gate["summary"]["promotion_eligible"] is False
    assert gate["summary"]["promotion_authority_status"] == "NOT_TESTED"
