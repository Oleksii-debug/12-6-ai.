"""Exact S0 LOCAL_FREE/CPU training and held-out evidence runner.

This module composes existing D01/D02/D03/D04 contracts. It does not own model,
dataset, tokenizer, packing, checkpoint, evaluation-gate, or release semantics.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import time
from dataclasses import asdict
from itertools import cycle, islice
from pathlib import Path
from typing import Any

import torch

from twelve_six.model import TwelveSixDecoder, load_stage_config
from twelve_six.packing import (
    PACKING_CONFIG_HASH,
    batch_examples,
    collate_rows,
    iter_packed_examples,
    load_jsonl_records,
)
from twelve_six.tokenization import ByteTokenizer

from .config import TrainerConfig
from .loss import causal_lm_loss
from .trainer import (
    NonFiniteTrainingError,
    StepMetrics,
    Trainer,
    TrainingStateInvalidError,
)

DATASET_MANIFEST_SHA256 = "b085a7ab56510575a11a80824fcff3a95a17f237d46d1be820e59d1289f220c2"
DATASET_IDENTITY_SHA256 = "bab60119d49e93303c972b77900fcb5553817f754cbc5d9a58019228cfa0ca89"
TRAIN_JSONL_SHA256 = "61d24b7138df56527d201cea405d11c9f607684b4a9593dfa20c599cc2ee6998"
VALIDATION_JSONL_SHA256 = "57f18a846dcca75955a82612382d4635ba9583965aa6628e77626cd2a3eb19c5"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_hash(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _validate_source_sha(source_sha: str) -> None:
    if len(source_sha) != 40 or any(ch not in "0123456789abcdef" for ch in source_sha):
        raise ValueError("source_sha must be a full lowercase 40-hex Git SHA")


def _tensor_batches(
    root: Path,
    *,
    split: str,
    tokenizer: ByteTokenizer,
    batch_size: int,
) -> tuple[list[dict[str, torch.Tensor]], tuple[str, ...], int]:
    records = tuple(
        load_jsonl_records(root / f"data/s0/packaged/{split}.jsonl", split=split)
    )
    examples = tuple(
        iter_packed_examples(
            records,
            tokenizer,
            expected_split=split,
            sequence_length=128,
        )
    )
    if not examples or any(example.split != split for example in examples):
        raise RuntimeError(f"packing did not preserve split={split!r}")

    output: list[dict[str, torch.Tensor]] = []
    for group in batch_examples(examples, batch_size=batch_size, drop_last=False):
        rows = collate_rows(group, target_mode="labels")
        output.append(
            {
                "input_ids": torch.tensor(rows["input_ids"], dtype=torch.long),
                "labels": torch.tensor(rows["labels"], dtype=torch.long),
            }
        )
    return output, tuple(record.record_id for record in records), sum(
        example.num_loss_tokens for example in examples
    )


@torch.no_grad()
def _evaluate(
    model: TwelveSixDecoder,
    batches: list[dict[str, torch.Tensor]],
) -> tuple[float, int]:
    model.eval()
    weighted_loss = 0.0
    token_count = 0
    for batch in batches:
        logits = model(batch["input_ids"]).logits
        labels = batch["labels"]
        tokens = int(labels[:, 1:].ne(-100).sum().item())
        loss = causal_lm_loss(logits, labels)
        if not torch.isfinite(loss).item():
            raise FloatingPointError("held-out evaluation produced non-finite loss")
        weighted_loss += float(loss.item()) * tokens
        token_count += tokens
    if token_count <= 0:
        raise RuntimeError("evaluation split contains zero scoreable tokens")
    return weighted_loss / token_count, token_count


def _weight_snapshot(model: TwelveSixDecoder) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().clone() for name, parameter in model.named_parameters()}


def _weight_delta(
    model: TwelveSixDecoder,
    before: dict[str, torch.Tensor],
) -> dict[str, float | int]:
    squared = 0.0
    max_abs = 0.0
    changed = 0
    total = 0
    for name, parameter in model.named_parameters():
        delta = parameter.detach().float() - before[name].float()
        squared += float(torch.sum(delta * delta).item())
        max_abs = max(max_abs, float(delta.abs().max().item()))
        changed += int(delta.ne(0).sum().item())
        total += delta.numel()
    return {
        "l2": math.sqrt(squared),
        "max_abs": max_abs,
        "changed_parameter_elements": changed,
        "trainable_parameter_elements": total,
    }


def _failure_recovery_probe(
    stage_path: Path,
    batch: dict[str, torch.Tensor],
    *,
    seed: int,
) -> dict[str, bool]:
    """Exercise real-D01 NaN/Inf poisoning and fresh-Trainer recovery boundaries."""
    stage = load_stage_config(stage_path)
    probe_config = TrainerConfig(
        learning_rate=3e-2,
        max_steps=2,
        seed=seed,
        precision="fp32",
        deterministic_algorithms=True,
    )

    def run_fault(value: float) -> bool:
        torch.manual_seed(seed)
        model = TwelveSixDecoder(stage.model, stage.init)
        trainer = Trainer(model, probe_config, device="cpu")
        trainer.train_microbatch(batch)
        safe_model = {
            name: tensor.detach().clone() for name, tensor in model.state_dict().items()
        }
        safe_state = trainer.state_dict()
        first_token = int(batch["input_ids"][0, 0].item())
        with torch.no_grad():
            model.token_embedding.weight[first_token, 0] = value
        try:
            trainer.train_microbatch(batch)
        except NonFiniteTrainingError:
            pass
        else:
            raise AssertionError("non-finite fault did not fail before optimizer continuation")
        try:
            trainer.state_dict()
        except TrainingStateInvalidError:
            pass
        else:
            raise AssertionError("poisoned trainer remained checkpoint-serializable")

        torch.manual_seed(seed)
        restored_model = TwelveSixDecoder(stage.model, stage.init)
        restored_model.load_state_dict(safe_model)
        restored_trainer = Trainer(restored_model, probe_config, device="cpu")
        restored_trainer.load_state_dict(safe_state)
        resumed = restored_trainer.train_microbatch(batch)
        return bool(resumed.optimizer_stepped and math.isfinite(resumed.loss))

    return {
        "nan_fail_closed_and_fresh_recovery": run_fault(float("nan")),
        "inf_fail_closed_and_fresh_recovery": run_fault(float("inf")),
    }


def run_s0_training_evidence(
    root: str | Path,
    *,
    source_sha: str,
    seed: int = 1337,
    max_steps: int = 40,
    batch_size: int = 3,
) -> dict[str, Any]:
    """Run the real 10,140-param S0 model against exact D03/D04 train/validation splits."""
    _validate_source_sha(source_sha)
    root = Path(root).resolve()
    stage_path = root / "configs/stages/s0_10k.json"
    dataset_path = root / "data/s0/packaged/manifest.json"
    train_path = root / "data/s0/packaged/train.jsonl"
    validation_path = root / "data/s0/packaged/validation.jsonl"

    if _sha256_file(dataset_path) != DATASET_MANIFEST_SHA256:
        raise RuntimeError("D03 dataset manifest SHA-256 mismatch")
    if _sha256_file(train_path) != TRAIN_JSONL_SHA256:
        raise RuntimeError("D03 train split SHA-256 mismatch")
    if _sha256_file(validation_path) != VALIDATION_JSONL_SHA256:
        raise RuntimeError("D03 validation split SHA-256 mismatch")
    dataset_manifest = json.loads(dataset_path.read_text(encoding="utf-8"))
    if dataset_manifest["dataset_identity_sha256"] != DATASET_IDENTITY_SHA256:
        raise RuntimeError("D03 dataset semantic identity mismatch")

    stage = load_stage_config(stage_path)
    tokenizer = ByteTokenizer()
    if stage.expected_parameters != 10_140 or stage.model.vocab_size != tokenizer.vocab_size:
        raise RuntimeError("S0 model/tokenizer contract mismatch")

    train_batches, train_ids, train_tokens_per_epoch = _tensor_batches(
        root, split="train", tokenizer=tokenizer, batch_size=batch_size
    )
    validation_batches, validation_ids, validation_tokens = _tensor_batches(
        root, split="validation", tokenizer=tokenizer, batch_size=batch_size
    )
    if set(train_ids) & set(validation_ids):
        raise RuntimeError("train/validation record identity overlap")

    config = TrainerConfig(
        learning_rate=3e-2,
        weight_decay=0.0,
        max_steps=max_steps,
        scheduler="constant",
        gradient_accumulation_steps=1,
        gradient_clip_norm=1.0,
        precision="fp32",
        seed=seed,
        deterministic_algorithms=True,
        deterministic_warn_only=False,
    )

    # Critical lineage ordering: declared seed is applied before scratch model construction.
    torch.manual_seed(seed)
    model = TwelveSixDecoder(stage.model, stage.init)
    initial_weights = _weight_snapshot(model)
    trainer = Trainer(model, config, device="cpu")

    initial_train_loss, initial_train_eval_tokens = _evaluate(model, train_batches)
    initial_validation_loss, initial_validation_eval_tokens = _evaluate(
        model, validation_batches
    )
    validation_step_before_training = trainer.optimizer_step

    step_metrics: list[StepMetrics] = []
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    run_result = trainer.run(
        islice(cycle(train_batches), max_steps),
        on_metrics=step_metrics.append,
    )
    process_cpu_seconds = time.process_time() - cpu_start
    wall_seconds = time.perf_counter() - wall_start

    final_train_loss, final_train_eval_tokens = _evaluate(model, train_batches)
    validation_step_before_eval = trainer.optimizer_step
    final_validation_loss, final_validation_eval_tokens = _evaluate(
        model, validation_batches
    )
    validation_step_after_eval = trainer.optimizer_step

    grad_norms = [metric.grad_norm for metric in step_metrics if metric.grad_norm is not None]
    if not grad_norms or not all(math.isfinite(value) for value in grad_norms):
        raise RuntimeError("training did not emit finite gradient norms")
    delta = _weight_delta(model, initial_weights)
    if delta["changed_parameter_elements"] <= 0:
        raise RuntimeError("real S0 training produced no model weight changes")
    if not final_train_loss < initial_train_loss:
        raise RuntimeError("real S0 training did not reduce full-train loss")
    measured_losses = (
        initial_train_loss,
        final_train_loss,
        initial_validation_loss,
        final_validation_loss,
    )
    if not all(math.isfinite(value) for value in measured_losses):
        raise RuntimeError("non-finite measured S0 loss")

    failure_probe = _failure_recovery_probe(stage_path, train_batches[0], seed=seed + 1)

    identity_payload = {
        "repository": "Oleksii-debug/12-6-ai.",
        "source_sha": source_sha,
        "stage": "S0",
        "modelspec_sha256": stage.model.identity_sha256(),
        "initspec_sha256": stage.init.identity_sha256(),
        "parameter_count": stage.expected_parameters,
        "tokenizer_config_sha256": tokenizer.identity.config_sha256,
        "tokenizer_vocab_sha256": tokenizer.identity.vocab_sha256,
        "tokenizer_version": tokenizer.identity.version,
        "packing_config_sha256": PACKING_CONFIG_HASH,
        "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
        "dataset_identity_sha256": DATASET_IDENTITY_SHA256,
        "train_jsonl_sha256": TRAIN_JSONL_SHA256,
        "validation_jsonl_sha256": VALIDATION_JSONL_SHA256,
        "training_config": asdict(config),
        "batch_size_examples": batch_size,
    }
    payload: dict[str, Any] = {
        "schema_version": "12-6.s0-real-training-evidence.v1",
        "authority": "LOCAL_FREE_OR_FREE_HOSTED_CPU_EVIDENCE_NOT_PROMOTION",
        "identity": identity_payload,
        "identity_sha256": _canonical_hash(identity_payload),
        "seed_ordering": {
            "seed_applied_before_model_construction": True,
            "trainer_reapplies_training_rng_seed": True,
            "resume_policy": "restore_rng_from_verified_checkpoint_not_reseed",
        },
        "split_isolation": {
            "optimized_split": "train",
            "train_record_ids": list(train_ids),
            "validation_record_ids": list(validation_ids),
            "record_id_overlap": [],
            "train_tokens_per_full_epoch": train_tokens_per_epoch,
            "validation_scoreable_tokens": validation_tokens,
            "validation_optimizer_step_before_training": validation_step_before_training,
            "validation_optimizer_step_before_final_eval": validation_step_before_eval,
            "validation_optimizer_step_after_final_eval": validation_step_after_eval,
            "validation_optimized_tokens": 0,
        },
        "training": {
            "optimizer_steps": run_result.optimizer_steps_completed,
            "microbatches_consumed": run_result.microbatches_consumed,
            "optimized_tokens": run_result.tokens_consumed,
            "trainer_tokens_seen": trainer.tokens_seen,
            "initial_train_loss": initial_train_loss,
            "final_train_loss": final_train_loss,
            "initial_validation_loss": initial_validation_loss,
            "final_validation_loss": final_validation_loss,
            "initial_train_eval_tokens": initial_train_eval_tokens,
            "final_train_eval_tokens": final_train_eval_tokens,
            "initial_validation_eval_tokens": initial_validation_eval_tokens,
            "final_validation_eval_tokens": final_validation_eval_tokens,
            "gradient_norm_min": min(grad_norms),
            "gradient_norm_max": max(grad_norms),
            "last_step_loss": step_metrics[-1].loss,
            "weight_delta": delta,
        },
        "failure_semantics": failure_probe,
        "runtime": {
            "wall_seconds": wall_seconds,
            "process_cpu_seconds": process_cpu_seconds,
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "platform": platform.platform(),
            "device": "cpu",
        },
        "claims": {
            "foreign_pretrained_weights_used": False,
            "instruction_or_alignment_training": False,
            "paid_compute_authorized_or_used": False,
            "candidate_or_stable_promotion": False,
        },
    }
    payload["evidence_sha256"] = _canonical_hash(payload)
    return payload
