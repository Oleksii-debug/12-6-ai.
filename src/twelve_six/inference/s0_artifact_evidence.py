"""Retained exact-candidate S0 checkpoint and first-party inference evidence.

This module is evidence orchestration only. It reuses the accepted D01 model,
D02 Trainer, D03 packaged data, D04 tokenizer/packing identities, D05 checkpoint
writer/loader, and D07 generation/parity/OpenAI-compatible completion surfaces.
It does not duplicate model architecture, sampling, checkpoint serialization, or
serving logic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from twelve_six.checkpoint import (
    CheckpointIntegrityError,
    bind_checkpoint_identity,
    hash_json,
    save_trainer_checkpoint,
    sha256_file,
    verify_checkpoint,
)
from twelve_six.integration import S0TorchInferenceBackend
from twelve_six.model import TwelveSixDecoder, load_stage_config
from twelve_six.packing import PACKING_CONFIG_HASH, PACKING_VERSION
from twelve_six.tokenization import ByteTokenizer
from twelve_six.training import Trainer, TrainerConfig, causal_lm_loss

from .contracts import GenerationConfig
from .first_party import load_first_party_backend
from .generation import generate
from .openai_compat import completion_response
from .parity import compare_backends
from .sampling import greedy_token

SCHEMA = "12-6.s0-retained-inference-evidence.v1"
_REPOSITORY = "Oleksii-debug/12-6-ai."
_EXACT_GIT_SHA = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _git_head(repo_root: Path) -> str:
    try:
        value = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("retained inference evidence requires a Git checkout") from exc
    if _EXACT_GIT_SHA.fullmatch(value) is None:
        raise ValueError("git HEAD is not an exact lowercase Git object id")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _load_training_batches(
    path: Path,
    tokenizer: ByteTokenizer,
    *,
    max_seq_len: int,
) -> list[dict[str, torch.Tensor]]:
    batches: list[dict[str, torch.Tensor]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise TypeError(f"{path}:{line_number} must contain a JSON object")
        text = row.get("text")
        if not isinstance(text, str) or not text:
            raise ValueError(f"{path}:{line_number} must contain non-empty text")
        token_ids = tokenizer.encode(text)[:max_seq_len]
        if len(token_ids) < 2:
            raise ValueError(f"{path}:{line_number} must encode to at least two tokens")
        ids = torch.tensor([token_ids], dtype=torch.long)
        batches.append({"input_ids": ids, "labels": ids})
    if not batches:
        raise ValueError(f"{path} contains no training records")
    return batches


@torch.no_grad()
def _mean_loss(model: TwelveSixDecoder, batches: list[dict[str, torch.Tensor]]) -> float:
    previous_mode = model.training
    model.eval()
    values: list[float] = []
    try:
        for batch in batches:
            loss = causal_lm_loss(model(batch["input_ids"]).logits, batch["labels"])
            value = float(loss.detach().cpu().item())
            if not math.isfinite(value) or value < 0:
                raise FloatingPointError("inference-evidence training produced invalid token NLL")
            values.append(value)
    finally:
        model.train(previous_mode)
    return sum(values) / len(values)


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _train(
    trainer: Trainer,
    batches: list[dict[str, torch.Tensor]],
    *,
    steps: int,
) -> None:
    for step in range(steps):
        metrics = trainer.train_microbatch(batches[step % len(batches)])
        if not metrics.optimizer_stepped or not math.isfinite(metrics.loss):
            raise RuntimeError("retained inference evidence did not complete a finite optimizer step")
    trainer.assert_checkpoint_safe()


def _run_manifest(
    *,
    candidate_sha: str,
    stage: Any,
    tokenizer: ByteTokenizer,
    trainer_config: TrainerConfig,
    dataset_manifest_sha256: str,
    train_sha256: str,
    environment_lock_sha256: str,
    train_steps: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_id": f"s0-d05-retained-inference-{candidate_sha[:12]}",
        "stage": "S0",
        "run_kind": "retained_first_party_inference_evidence",
        "state": "COMPLETED_LOCAL_FREE",
        "candidate": {
            "repository": _REPOSITORY,
            "git_sha": candidate_sha,
            "branch_or_tag": "exact-checkout",
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
            "precision": trainer_config.precision,
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
            "target_steps": train_steps,
            "target_tokens": 1,
            "checkpoint_interval_steps": train_steps,
        },
        "environment": {"lock_sha256": environment_lock_sha256},
    }


def _expect_corrupt_checkpoint_rejected(checkpoint: Path) -> str:
    with tempfile.TemporaryDirectory(prefix="twelve-six-corrupt-inference-evidence-") as temp_dir:
        corrupt = Path(temp_dir) / "checkpoint"
        shutil.copytree(checkpoint, corrupt)
        weights = corrupt / "weights.safetensors"
        payload = bytearray(weights.read_bytes())
        if not payload:
            raise RuntimeError("retained checkpoint weights are unexpectedly empty")
        payload[-1] ^= 1
        weights.write_bytes(payload)
        try:
            load_first_party_backend(corrupt)
        except CheckpointIntegrityError as exc:
            return str(exc)
        raise AssertionError("corrupt retained checkpoint was accepted")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def collect_retained_s0_inference_evidence(
    repo_root: Path,
    candidate_sha: str,
    output_dir: Path,
    *,
    train_steps: int = 40,
    seed: int = 20260825,
    verify_checkout: bool = True,
) -> dict[str, Any]:
    """Train S0 locally, retain one strict D05 checkpoint, and prove D07 behavior."""

    repo_root = repo_root.resolve()
    output_dir = output_dir.resolve()
    if _EXACT_GIT_SHA.fullmatch(candidate_sha) is None:
        raise ValueError("candidate_sha must be a full lowercase 40- or 64-hex Git object id")
    if verify_checkout and _git_head(repo_root) != candidate_sha:
        raise ValueError("candidate_sha is stale: it does not equal checkout HEAD")
    if not isinstance(train_steps, int) or isinstance(train_steps, bool) or train_steps < 1:
        raise ValueError("train_steps must be a positive integer")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("output_dir must not contain pre-existing evidence")
    output_dir.mkdir(parents=True, exist_ok=True)

    stage_path = repo_root / "configs/stages/s0_10k.json"
    manifest_path = repo_root / "data/s0/packaged/manifest.json"
    train_path = repo_root / "data/s0/packaged/train.jsonl"
    environment_lock_path = repo_root / "requirements/locks/index.json"

    stage = load_stage_config(stage_path)
    tokenizer = ByteTokenizer()
    if stage.canonical_base != "random_init":
        raise ValueError("S0 canonical Base must remain random_init")
    if stage.model.vocab_size != tokenizer.vocab_size:
        raise ValueError("tokenizer/model vocabulary mismatch")

    dataset_manifest = _load_json(manifest_path)
    outputs = dataset_manifest.get("outputs")
    if not isinstance(outputs, dict) or outputs.get("train.jsonl") != sha256_file(train_path):
        raise ValueError("train.jsonl hash does not match committed dataset manifest")

    batches = _load_training_batches(
        train_path,
        tokenizer,
        max_seq_len=stage.model.max_seq_len,
    )
    trainer_config = TrainerConfig(
        learning_rate=3e-2,
        weight_decay=0.0,
        max_steps=train_steps,
        scheduler="constant",
        gradient_accumulation_steps=1,
        gradient_clip_norm=1.0,
        precision="fp32",
        seed=seed,
        deterministic_algorithms=True,
    )

    _seed_all(seed)
    model = TwelveSixDecoder(stage.model, stage.init)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != stage.expected_parameters:
        raise ValueError("instantiated S0 parameter count does not match frozen stage config")
    train_loss_before = _mean_loss(model, batches)
    trainer = Trainer(model, trainer_config, device="cpu")
    _train(trainer, batches, steps=train_steps)
    train_loss_after = _mean_loss(model, batches)

    dataset_manifest_sha256 = sha256_file(manifest_path)
    train_sha256 = sha256_file(train_path)
    environment_lock_sha256 = sha256_file(environment_lock_path)
    run_manifest = _run_manifest(
        candidate_sha=candidate_sha,
        stage=stage,
        tokenizer=tokenizer,
        trainer_config=trainer_config,
        dataset_manifest_sha256=dataset_manifest_sha256,
        train_sha256=train_sha256,
        environment_lock_sha256=environment_lock_sha256,
        train_steps=train_steps,
    )
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
        environment_lock_hash=environment_lock_sha256,
    )

    checkpoint = output_dir / "checkpoint"
    save_trainer_checkpoint(
        checkpoint,
        model=model,
        trainer=trainer,
        identity=identity,
    )
    checkpoint_manifest = verify_checkpoint(checkpoint)

    direct_backend = S0TorchInferenceBackend(model, tokenizer)
    reloaded_backend = load_first_party_backend(checkpoint)
    prompts = ("12-6", "Base", "Україна")
    parity = compare_backends(
        direct_backend,
        reloaded_backend,
        prompts,
        max_new_tokens=8,
        atol=0.0,
        rtol=0.0,
    )
    if not parity.passed:
        raise RuntimeError("retained checkpoint failed exact direct-vs-reloaded parity")

    greedy_config = GenerationConfig(max_new_tokens=8, sample=False, seed=seed)
    direct_greedy = generate(direct_backend, prompts[0], greedy_config)
    reloaded_greedy = generate(reloaded_backend, prompts[0], greedy_config)
    if direct_greedy != reloaded_greedy:
        raise RuntimeError("retained checkpoint greedy generation diverged after reload")

    sample_config = GenerationConfig(
        max_new_tokens=8,
        sample=True,
        temperature=0.8,
        top_k=32,
        top_p=0.9,
        seed=seed,
    )
    sampled_a = generate(reloaded_backend, prompts[0], sample_config)
    sampled_b = generate(reloaded_backend, prompts[0], sample_config)
    direct_sampled = generate(direct_backend, prompts[0], sample_config)
    if sampled_a != sampled_b or sampled_a != direct_sampled:
        raise RuntimeError("seeded sampling is not repeatable across direct/reloaded S0")

    stop_prompt = "A"
    first_token = greedy_token(reloaded_backend.next_token_logits(reloaded_backend.encode(stop_prompt)))
    token_stopped = generate(
        reloaded_backend,
        stop_prompt,
        GenerationConfig(max_new_tokens=8, stop_token_ids=(first_token,)),
    )
    if token_stopped.stop_reason != "stop_token":
        raise RuntimeError("retained checkpoint token stop semantics failed")
    first_text = reloaded_backend.decode([first_token])
    text_stopped = generate(
        reloaded_backend,
        stop_prompt,
        GenerationConfig(max_new_tokens=8, stop_strings=(first_text,)),
    )
    if text_stopped.stop_reason != "stop_string":
        raise RuntimeError("retained checkpoint text stop semantics failed")

    context_full = generate(
        reloaded_backend,
        "A" * reloaded_backend.max_context_tokens,
        GenerationConfig(max_new_tokens=1),
    )
    if context_full.stop_reason != "context_limit":
        raise RuntimeError("retained checkpoint context-limit stop semantics failed")
    try:
        generate(
            reloaded_backend,
            "A" * (reloaded_backend.max_context_tokens + 1),
            GenerationConfig(max_new_tokens=1),
        )
    except ValueError as exc:
        over_context_error = str(exc)
    else:
        raise AssertionError("over-context retained checkpoint prompt was accepted")

    completion = completion_response(
        reloaded_backend,
        {
            "model": "12-6-base-s0",
            "prompt": prompts[0],
            "max_tokens": 8,
            "temperature": 0,
            "top_p": 1.0,
            "seed": seed,
        },
        response_id="cmpl-s0-retained-evidence",
        created=0,
        model_name="12-6-base-s0",
    )
    choices = completion.get("choices")
    if not isinstance(choices, list) or not choices or choices[0].get("text") != reloaded_greedy.text:
        raise RuntimeError("OpenAI-compatible raw completion diverged from canonical greedy output")

    corrupt_error = _expect_corrupt_checkpoint_rejected(checkpoint)
    diagnostics = reloaded_backend.diagnostics()
    if diagnostics["checkpoint_id"] != checkpoint_manifest["checkpoint_id"]:
        raise RuntimeError("first-party diagnostics checkpoint identity drifted")
    if diagnostics["git_sha"] != candidate_sha:
        raise RuntimeError("first-party diagnostics candidate SHA drifted")

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "PASS",
        "candidate": {
            "repository": _REPOSITORY,
            "sha": candidate_sha,
            "canonical_base": "random_init",
            "pretraining_only": True,
            "foreign_pretrained_weights": False,
            "behavioral_alignment_weights": False,
        },
        "training": {
            "parameter_count": parameter_count,
            "steps": trainer.optimizer_step,
            "tokens_seen": trainer.tokens_seen,
            "seed": seed,
            "train_loss_before": train_loss_before,
            "train_loss_after": train_loss_after,
            "loss_decreased": train_loss_after < train_loss_before,
        },
        "identity": {
            "model_spec_sha256": hash_json(stage.model.to_dict()),
            "init_spec_sha256": hash_json(stage.init.to_dict()),
            "dataset_manifest_sha256": dataset_manifest_sha256,
            "train_sha256": train_sha256,
            "tokenizer_config_sha256": tokenizer.identity.config_sha256,
            "tokenizer_vocab_sha256": tokenizer.identity.vocab_sha256,
            "tokenizer_version": tokenizer.identity.version,
            "packing_sha256": PACKING_CONFIG_HASH,
            "packing_version": PACKING_VERSION,
            "environment_lock_sha256": environment_lock_sha256,
            "run_manifest_sha256": hash_json(run_manifest),
        },
        "checkpoint": {
            "checkpoint_id": checkpoint_manifest["checkpoint_id"],
            "format": checkpoint_manifest["format"],
            "format_version": checkpoint_manifest["format_version"],
            "serialization_pickle": checkpoint_manifest["serialization"]["pickle"],
            "files": checkpoint_manifest["files"],
            "corrupt_checkpoint_rejected": True,
            "corrupt_rejection": corrupt_error,
        },
        "inference": {
            "diagnostics": diagnostics,
            "parity": parity.to_dict(),
            "greedy": {
                "output_sha256": _sha256_text(reloaded_greedy.text),
                "token_ids": list(reloaded_greedy.generated_token_ids),
                "stop_reason": reloaded_greedy.stop_reason,
                "direct_vs_reloaded_equal": True,
            },
            "seeded_sampling": {
                "output_sha256": _sha256_text(sampled_a.text),
                "token_ids": list(sampled_a.generated_token_ids),
                "stop_reason": sampled_a.stop_reason,
                "repeatable": True,
                "direct_vs_reloaded_equal": True,
            },
            "stop_semantics": {
                "token_stop": token_stopped.stop_reason,
                "text_stop": text_stopped.stop_reason,
            },
            "context": {
                "max_context_tokens": reloaded_backend.max_context_tokens,
                "exact_limit_stop": context_full.stop_reason,
                "over_limit_rejected": True,
                "over_limit_error": over_context_error,
            },
            "openai_compatible_raw_completion_equal": True,
            "chat_semantics": False,
        },
        "artifact": {
            "checkpoint_relative_path": "checkpoint",
            "report_relative_path": "inference_evidence.json",
            "retained_for_external_execution": True,
            "windows_nvda_live_pass": False,
        },
        "truth_boundary": {
            "local_free_cpu_only": True,
            "paid_compute": False,
            "candidate_or_stable_promotion": False,
            "audit_verdict": False,
            "external_backend_parity": "NOT_TESTED",
            "windows_nvda_live_execution": "NOT_TESTED",
        },
    }
    report["report_sha256"] = hash_json(report)
    _write_json(output_dir / "inference_evidence.json", report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a retained exact-candidate S0 checkpoint and inference evidence artifact."
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--candidate-sha")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260825)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    candidate_sha = args.candidate_sha or _git_head(repo_root)
    report = collect_retained_s0_inference_evidence(
        repo_root,
        candidate_sha,
        args.output_dir,
        train_steps=args.train_steps,
        seed=args.seed,
    )
    print(
        "inference-evidence: PASS "
        f"candidate={candidate_sha} checkpoint_id={report['checkpoint']['checkpoint_id']} "
        f"steps={report['training']['steps']} parity_steps={report['inference']['parity']['steps_compared']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
