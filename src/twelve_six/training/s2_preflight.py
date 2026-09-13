"""Executable ~1M engineering preflight for the current 12-6 stack.

The S2 geometry in this module is intentionally byte-tokenizer compatible so the
existing D04 -> D05 -> D07 first-party path can execute end to end today. The S0
fixture and byte tokenizer remain compatibility-only inputs, not an S2 corpus or
future tokenizer selection.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch

from twelve_six.checkpoint import (
    CheckpointIdentity,
    hash_json,
    load_trainer_checkpoint,
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
from .loss import causal_lm_loss
from .trainer import StepMetrics, Trainer

SCHEMA = "12-6.s2-1m-executable-preflight.v1"
AUTHORITY = "ENGINEERING_EXECUTABLE_PREFLIGHT_NOT_STAGE_EVIDENCE"
REPOSITORY = "Oleksii-debug/12-6-ai."
FIXTURE_SCOPE = "S0_CONTROLLED_FIXTURE_COMPATIBILITY_ONLY_NOT_S2_CORPUS_OR_TOKENIZER"
CANDIDATE_PATH = "configs/stages/alternatives/s2_1m_byte_gqa.candidate.json"
MODEL_SPEC_SHA256 = "18284b303eb31cef5191ddb3ed4ddba5ce51789aadf4b14cc90d4226c5c527b5"
PARAMETER_COUNT = 992_896
MODEL_VOCAB = 256
MAX_CONTEXT = 512


def _exact_git_sha(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) in {40, 64}
        and value == value.lower()
        and all(ch in "0123456789abcdef" for ch in value)
    )


def _git_head(repo_root: Path) -> str:
    try:
        value = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("S2 executable preflight requires a Git checkout") from exc
    if not _exact_git_sha(value):
        raise ValueError("git HEAD is not an exact lowercase Git object id")
    return value


def _load_texts(path: Path) -> list[str]:
    texts: list[str] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise TypeError(f"{path}:{line_number} must contain a JSON object")
        text = row.get("text")
        if not isinstance(text, str) or not text:
            raise ValueError(f"{path}:{line_number} must contain non-empty text")
        texts.append(text)
    if not texts:
        raise ValueError("controlled fixture contains no training text")
    return texts


def _batch_for_step(
    texts: Sequence[str],
    tokenizer: ByteTokenizer,
    *,
    step: int,
    sequence_length: int,
) -> dict[str, torch.Tensor]:
    token_ids = tokenizer.encode(texts[step % len(texts)])[:sequence_length]
    if len(token_ids) < 2:
        raise ValueError("controlled fixture record must encode to at least two tokens")
    ids = torch.tensor([token_ids], dtype=torch.long)
    return {"input_ids": ids, "labels": ids}


@torch.no_grad()
def _loss_on_batch(model: TwelveSixDecoder, batch: Mapping[str, torch.Tensor]) -> float:
    model.eval()
    logits = model(batch["input_ids"]).logits
    loss = causal_lm_loss(logits, batch["labels"])
    if not torch.isfinite(loss).item():
        raise RuntimeError("preflight evaluation produced non-finite loss")
    return float(loss.item())


def _snapshot(model: TwelveSixDecoder) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }


def _weight_delta(
    model: TwelveSixDecoder,
    before: Mapping[str, torch.Tensor],
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


def _tensor_bytes(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, Mapping):
        return sum(_tensor_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_bytes(item) for item in value)
    return 0


def _directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.iterdir() if item.is_file())


def _nested_equal(left: Any, right: Any) -> bool:
    if is_dataclass(left) and not isinstance(left, type):
        left = asdict(left)
    if is_dataclass(right) and not isinstance(right, type):
        right = asdict(right)
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and torch.equal(left, right)
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            return False
        return left.keys() == right.keys() and all(
            _nested_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if not isinstance(left, (list, tuple)) or not isinstance(right, (list, tuple)):
            return False
        return len(left) == len(right) and all(
            _nested_equal(l_item, r_item)
            for l_item, r_item in zip(left, right, strict=True)
        )
    return left == right


def _models_equal(left: TwelveSixDecoder, right: TwelveSixDecoder) -> bool:
    left_state = left.state_dict()
    right_state = right.state_dict()
    return left_state.keys() == right_state.keys() and all(
        torch.equal(left_state[key], right_state[key]) for key in left_state
    )


def _train_range(
    trainer: Trainer,
    texts: Sequence[str],
    tokenizer: ByteTokenizer,
    *,
    start_step: int,
    end_step: int,
    sequence_length: int,
) -> list[StepMetrics]:
    metrics: list[StepMetrics] = []
    for step in range(start_step, end_step):
        item = trainer.train_microbatch(
            _batch_for_step(
                texts,
                tokenizer,
                step=step,
                sequence_length=sequence_length,
            )
        )
        if not item.optimizer_stepped or item.grad_norm is None:
            raise RuntimeError("S2 preflight requires a committed finite optimizer step")
        if not math.isfinite(item.loss) or not math.isfinite(item.grad_norm):
            raise RuntimeError("S2 preflight produced non-finite training metrics")
        metrics.append(item)
    trainer.assert_checkpoint_safe()
    return metrics


def _checkpoint_identity(
    *,
    candidate_sha: str,
    stage: Any,
    tokenizer: ByteTokenizer,
    dataset_manifest_sha256: str,
    train_sha256: str,
    environment_lock_sha256: str,
    trainer: Trainer,
) -> CheckpointIdentity:
    init_sha256 = hash_json(stage.init.to_dict())
    training_config = {
        "authority": AUTHORITY,
        "stage": "S2",
        "architecture_status": "ENGINEERING_CANDIDATE_NOT_FROZEN",
        "fixture_scope": FIXTURE_SCOPE,
        "s2_corpus_selected": False,
        "future_tokenizer_selected": False,
        "init_spec_sha256": init_sha256,
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
        "repository": REPOSITORY,
        "candidate_sha": candidate_sha,
        "stage": "S2",
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


def collect_s2_1m_preflight(
    repo_root: Path,
    candidate_sha: str,
    output_dir: Path,
    *,
    total_steps: int = 4,
    split_step: int = 2,
    seed: int = 20260825,
    sequence_length: int = 128,
    verify_checkout: bool = True,
) -> dict[str, Any]:
    """Run current S2 mechanics without claiming corpus, tokenizer, or stage readiness."""

    repo_root = repo_root.resolve()
    output_dir = output_dir.resolve()
    if not _exact_git_sha(candidate_sha):
        raise ValueError("candidate_sha must be a full lowercase Git object id")
    if verify_checkout and _git_head(repo_root) != candidate_sha:
        raise ValueError("candidate_sha does not equal checkout HEAD")
    if not isinstance(total_steps, int) or isinstance(total_steps, bool) or total_steps < 2:
        raise ValueError("total_steps must be an integer >= 2")
    if (
        not isinstance(split_step, int)
        or isinstance(split_step, bool)
        or not 0 < split_step < total_steps
    ):
        raise ValueError("split_step must be strictly between 0 and total_steps")
    if not 2 <= sequence_length <= MAX_CONTEXT:
        raise ValueError(f"sequence_length must be in [2, {MAX_CONTEXT}]")

    stage_path = repo_root / CANDIDATE_PATH
    stage = load_stage_config(stage_path)
    tokenizer = ByteTokenizer()
    if stage.stage != "S2" or stage.canonical_base != "random_init":
        raise ValueError("unexpected S2 candidate contract")
    if stage.expected_parameters != PARAMETER_COUNT:
        raise ValueError("S2 executable candidate parameter count drifted")
    if stage.model.identity_sha256() != MODEL_SPEC_SHA256:
        raise ValueError("S2 executable candidate ModelSpec drifted")
    if stage.model.vocab_size != MODEL_VOCAB or tokenizer.vocab_size != MODEL_VOCAB:
        raise ValueError("S2 byte-compatible model/tokenizer vocabulary contract drifted")
    if stage.model.max_seq_len != MAX_CONTEXT:
        raise ValueError("S2 executable candidate context drifted")

    train_path = repo_root / "data/s0/packaged/train.jsonl"
    dataset_manifest_path = repo_root / "data/s0/packaged/manifest.json"
    environment_lock_path = repo_root / "requirements/locks/index.json"
    texts = _load_texts(train_path)
    dataset_manifest_sha256 = sha256_file(dataset_manifest_path)
    train_sha256 = sha256_file(train_path)
    environment_lock_sha256 = sha256_file(environment_lock_path)

    config = TrainerConfig(
        learning_rate=1e-3,
        weight_decay=0.0,
        max_steps=total_steps,
        scheduler="constant",
        gradient_accumulation_steps=1,
        gradient_clip_norm=1.0,
        precision="fp32",
        seed=seed,
        deterministic_algorithms=True,
    )
    fixed_batch = _batch_for_step(
        texts,
        tokenizer,
        step=0,
        sequence_length=sequence_length,
    )

    random.seed(seed)
    torch.manual_seed(seed)
    baseline_model = TwelveSixDecoder(stage.model, stage.init)
    actual_parameters = sum(
        parameter.numel() for parameter in baseline_model.parameters() if parameter.requires_grad
    )
    if actual_parameters != PARAMETER_COUNT:
        raise RuntimeError("instantiated S2 trainable parameter count mismatch")
    before = _snapshot(baseline_model)
    baseline_trainer = Trainer(baseline_model, config, device="cpu")
    initial_loss = _loss_on_batch(baseline_model, fixed_batch)
    metrics = _train_range(
        baseline_trainer,
        texts,
        tokenizer,
        start_step=0,
        end_step=total_steps,
        sequence_length=sequence_length,
    )
    final_loss = _loss_on_batch(baseline_model, fixed_batch)
    delta = _weight_delta(baseline_model, before)
    if delta["changed_parameter_elements"] <= 0 or delta["l2"] <= 0.0:
        raise RuntimeError("S2 optimizer trajectory did not change model weights")

    context_source = tokenizer.encode("12-6 executable context probe ")
    context_ids = (context_source * ((MAX_CONTEXT // len(context_source)) + 1))[:MAX_CONTEXT]
    context_tensor = torch.tensor([context_ids], dtype=torch.long)
    with torch.no_grad():
        context_logits = baseline_model(context_tensor).logits
    if tuple(context_logits.shape) != (1, MAX_CONTEXT, MODEL_VOCAB):
        raise RuntimeError("S2 full-context forward shape mismatch")

    random.seed(seed)
    torch.manual_seed(seed)
    partial_model = TwelveSixDecoder(stage.model, stage.init)
    partial_trainer = Trainer(partial_model, config, device="cpu")
    _train_range(
        partial_trainer,
        texts,
        tokenizer,
        start_step=0,
        end_step=split_step,
        sequence_length=sequence_length,
    )

    identity = _checkpoint_identity(
        candidate_sha=candidate_sha,
        stage=stage,
        tokenizer=tokenizer,
        dataset_manifest_sha256=dataset_manifest_sha256,
        train_sha256=train_sha256,
        environment_lock_sha256=environment_lock_sha256,
        trainer=partial_trainer,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoint"
    manifest = save_trainer_checkpoint(
        checkpoint_dir,
        model=partial_model,
        trainer=partial_trainer,
        identity=identity,
    )
    checkpoint_bytes = _directory_bytes(checkpoint_dir)

    restored_model = TwelveSixDecoder(stage.model, stage.init)
    restored_trainer = Trainer(restored_model, config, device="cpu")
    load_trainer_checkpoint(
        checkpoint_dir,
        model=restored_model,
        trainer=restored_trainer,
        restore_rng=True,
        expected_git_sha=candidate_sha,
        expected_model_spec_hash=stage.model.identity_sha256(),
        expected_init_spec_hash=stage.init.identity_sha256(),
        expected_tokenizer_hash=tokenizer.identity.config_sha256,
        expected_tokenizer_vocab_hash=tokenizer.identity.vocab_sha256,
        expected_dataset_manifest_hash=dataset_manifest_sha256,
        expected_split_identity=f"controlled-s0-train:{train_sha256}",
        expected_packing_hash=PACKING_CONFIG_HASH,
        expected_packing_version=PACKING_VERSION,
        expected_run_manifest_hash=identity.run_manifest_hash,
        expected_training_config_hash=hash_json(identity.training_config),
        expected_environment_lock_hash=environment_lock_sha256,
        expected_seed=seed,
    )
    _train_range(
        restored_trainer,
        texts,
        tokenizer,
        start_step=split_step,
        end_step=total_steps,
        sequence_length=sequence_length,
    )
    model_state_exact = _models_equal(baseline_model, restored_model)
    trainer_state_exact = _nested_equal(
        baseline_trainer.state_dict(),
        restored_trainer.state_dict(),
    )
    if not model_state_exact or not trainer_state_exact:
        raise RuntimeError("S2 checkpoint resume diverged from uninterrupted control")

    backend = load_first_party_backend(checkpoint_dir)
    generation = generate(
        backend,
        "12-6",
        GenerationConfig(max_new_tokens=2, sample=False, seed=seed),
    )
    if len(generation.generated_token_ids) != 2:
        raise RuntimeError("first-party S2 generation did not emit requested tokens")
    diagnostics = backend.diagnostics()
    if diagnostics["parameter_count"] != PARAMETER_COUNT:
        raise RuntimeError("first-party inference reconstructed wrong S2 parameter count")

    model_parameter_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in partial_model.parameters()
    )
    optimizer_tensor_bytes = _tensor_bytes(partial_trainer.optimizer.state_dict())
    grad_norms = [float(item.grad_norm) for item in metrics if item.grad_norm is not None]

    evidence: dict[str, Any] = {
        "schema": SCHEMA,
        "authority": AUTHORITY,
        "repository": REPOSITORY,
        "candidate_sha": candidate_sha,
        "stage": "S2",
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
            "tied_output": stage.model.tie_word_embeddings,
        },
        "fixture": {
            "scope": FIXTURE_SCOPE,
            "tokenizer_version": tokenizer.identity.version,
            "tokenizer_vocab_size": tokenizer.vocab_size,
            "dataset_manifest_sha256": dataset_manifest_sha256,
            "train_split_sha256": train_sha256,
            "packing_sha256": PACKING_CONFIG_HASH,
            "packing_version": PACKING_VERSION,
        },
        "training": {
            "precision": config.precision,
            "optimizer_steps": baseline_trainer.optimizer_step,
            "optimized_tokens": baseline_trainer.tokens_seen,
            "initial_fixed_batch_loss": initial_loss,
            "final_fixed_batch_loss": final_loss,
            "gradient_norm_min": min(grad_norms),
            "gradient_norm_max": max(grad_norms),
            "weight_delta": delta,
        },
        "context_probe": {
            "sequence_length": MAX_CONTEXT,
            "logits_shape": list(context_logits.shape),
            "finite": bool(torch.isfinite(context_logits).all().item()),
        },
        "checkpoint": {
            "format": manifest["format"],
            "format_version": manifest["format_version"],
            "checkpoint_id": manifest["checkpoint_id"],
            "step": identity.step,
            "tokens_seen": identity.tokens_seen,
            "directory_bytes": checkpoint_bytes,
            "artifact_bytes": {
                name: record["bytes"]
                for name, record in manifest["files"].items()
            },
            "resume_model_state_exact": model_state_exact,
            "resume_trainer_state_exact": trainer_state_exact,
        },
        "tensor_state_footprint": {
            "model_parameter_bytes": model_parameter_bytes,
            "optimizer_tensor_bytes_after_split": optimizer_tensor_bytes,
            "combined_model_optimizer_tensor_bytes": (
                model_parameter_bytes + optimizer_tensor_bytes
            ),
            "note": "Exact tensor-state bytes, not process RSS or peak allocator memory.",
        },
        "first_party_inference": {
            "backend": diagnostics["backend"],
            "parameter_count": diagnostics["parameter_count"],
            "vocab_size": diagnostics["vocab_size"],
            "max_context_tokens": diagnostics["max_context_tokens"],
            "generated_token_ids": list(generation.generated_token_ids),
            "stop_reason": generation.stop_reason,
        },
        "blockers": {
            "meaningful_s2_training_experiment_ready": False,
            "production_s2_corpus_ready": False,
            "future_s2_tokenizer_selected": False,
            "reason": (
                "Only the controlled S0 fixture and byte tokenizer are available for "
                "compatibility mechanics; external corpus/tokenizer selection is not frozen."
            ),
        },
        "claims": {
            "paid_compute_used": False,
            "foreign_pretrained_weights_used": False,
            "instruction_or_alignment_training": False,
            "s2_quality_or_capability_evidence": False,
            "s2_corpus_or_tokenizer_frozen": False,
            "candidate_or_stable_promotion": False,
        },
    }
    evidence["evidence_sha256"] = hash_json(evidence)
    (output_dir / "s2-1m-executable-preflight.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return evidence


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the byte-compatible ~1M 12-6 engineering preflight."
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--candidate-sha", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--total-steps", type=int, default=4)
    parser.add_argument("--split-step", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--sequence-length", type=int, default=128)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    evidence = collect_s2_1m_preflight(
        args.repo_root,
        args.candidate_sha,
        args.output_dir,
        total_steps=args.total_steps,
        split_step=args.split_step,
        seed=args.seed,
        sequence_length=args.sequence_length,
    )
    print(json.dumps(evidence, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
