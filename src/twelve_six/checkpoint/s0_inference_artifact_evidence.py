"""Exact S0 trained-checkpoint inference artifact evidence.

This D05 evidence runner composes the existing D01 model, D02 Trainer, D03 data,
D04 tokenizer/packing, D05 checkpoint, and D07 inference/server contracts. It does
not reimplement any architecture, sampling, checkpoint serialization, or HTTP logic.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import subprocess
import threading
from collections.abc import Mapping
from itertools import cycle, islice
from pathlib import Path
from typing import Any

import torch

from twelve_six.inference import GenerationConfig, generate
from twelve_six.inference.first_party import load_first_party_backend
from twelve_six.inference.server import make_server
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
from twelve_six.training.s0_evidence import (
    DATASET_IDENTITY_SHA256,
    DATASET_MANIFEST_SHA256,
    TRAIN_JSONL_SHA256,
    VALIDATION_JSONL_SHA256,
)

from .core import hash_json, sha256_file, verify_checkpoint
from .run_binding import bind_checkpoint_identity
from .trainer_adapter import save_trainer_checkpoint

SCHEMA_VERSION = "12-6.s0-inference-artifact-evidence.v1"
AUTHORITY = "LOCAL_FREE_OR_FREE_HOSTED_CPU_EVIDENCE_NOT_PROMOTION"
REPOSITORY = "Oleksii-debug/12-6-ai."
_SHA_HEX = frozenset("0123456789abcdef")


class S0InferenceArtifactEvidenceError(ValueError):
    """Raised when S0 inference-artifact evidence fails closed."""


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise S0InferenceArtifactEvidenceError(message)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and set(value) <= _SHA_HEX
    )


def _validate_source_sha(value: Any) -> str:
    _require(
        isinstance(value, str)
        and len(value) == 40
        and value == value.lower()
        and set(value) <= _SHA_HEX,
        "source_sha must be a full lowercase 40-hex Git SHA",
    )
    return value


def _git_head(repo_root: Path) -> str:
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise S0InferenceArtifactEvidenceError(
            "exact S0 inference evidence requires a Git checkout"
        ) from exc
    return _validate_source_sha(head)


def _packed_batches(
    repo_root: Path,
    *,
    split: str,
    tokenizer: ByteTokenizer,
    batch_size: int,
) -> tuple[list[dict[str, torch.Tensor]], tuple[str, ...], int]:
    records = tuple(
        load_jsonl_records(
            repo_root / f"data/s0/packaged/{split}.jsonl",
            split=split,
        )
    )
    examples = tuple(
        iter_packed_examples(
            records,
            tokenizer,
            expected_split=split,
            sequence_length=128,
        )
    )
    _require(bool(examples), f"packed {split} split must not be empty")
    _require(
        all(example.split == split for example in examples),
        f"packing must preserve split={split!r}",
    )

    batches: list[dict[str, torch.Tensor]] = []
    for group in batch_examples(examples, batch_size=batch_size, drop_last=False):
        rows = collate_rows(group, target_mode="labels")
        batches.append(
            {
                "input_ids": torch.tensor(rows["input_ids"], dtype=torch.long),
                "labels": torch.tensor(rows["labels"], dtype=torch.long),
            }
        )
    _require(bool(batches), f"packed {split} split produced no batches")
    return (
        batches,
        tuple(record.record_id for record in records),
        sum(example.num_loss_tokens for example in examples),
    )


def _run_manifest(
    *,
    source_sha: str,
    stage: Any,
    tokenizer: ByteTokenizer,
    trainer_config: TrainerConfig,
    environment_lock_sha256: str,
    max_steps: int,
    batch_size: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_id": f"s0-d05-inference-artifact-{source_sha[:12]}",
        "stage": "S0",
        "run_kind": "trained_checkpoint_inference_artifact_evidence",
        "state": "COMPLETED_LOCAL_FREE",
        "candidate": {
            "repository": REPOSITORY,
            "git_sha": source_sha,
            "branch_or_tag": "exact-checkout",
            "modelspec_sha256": hash_json(stage.model.to_dict()),
            "initspec_sha256": hash_json(stage.init.to_dict()),
            "parameter_count": stage.expected_parameters,
        },
        "data": {
            "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
            "tokenizer_sha256": tokenizer.identity.config_sha256,
            "tokenizer_vocab_sha256": tokenizer.identity.vocab_sha256,
            "tokenizer_version": tokenizer.identity.version,
            "split_identity": f"train:{TRAIN_JSONL_SHA256}",
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
            "max_steps": max_steps,
            "batch_size_examples": batch_size,
        },
        "environment": {"lock_sha256": environment_lock_sha256},
    }


def _normalized_generation(result: Any) -> dict[str, Any]:
    token_ids = list(result.generated_token_ids)
    return {
        "token_ids": token_ids,
        "token_ids_sha256": _canonical_hash({"token_ids": token_ids}),
        "text_sha256": hashlib.sha256(result.text.encode("utf-8")).hexdigest(),
        "token_count": len(token_ids),
        "stop_reason": result.stop_reason,
    }


def _checkpoint_inventory(checkpoint_dir: Path) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    for path in sorted(checkpoint_dir.iterdir(), key=lambda item: item.name):
        _require(path.is_file(), "checkpoint artifact inventory must contain files only")
        inventory.append(
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    _require(bool(inventory), "checkpoint artifact inventory must not be empty")
    return inventory


def _http_json(
    host: str,
    port: int,
    method: str,
    path: str,
    payload: Mapping[str, Any] | None = None,
) -> tuple[int, dict[str, Any]]:
    connection = http.client.HTTPConnection(host, port, timeout=10)
    try:
        headers: dict[str, str] = {}
        body: str | None = None
        if payload is not None:
            body = json.dumps(payload, ensure_ascii=False, allow_nan=False)
            headers["Content-Type"] = "application/json"
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        raw = response.read()
    finally:
        connection.close()
    decoded = json.loads(raw.decode("utf-8"))
    _require(isinstance(decoded, dict), "HTTP evidence response must be a JSON object")
    return response.status, decoded


def build_s0_inference_artifact_evidence(
    repo_root: str | Path,
    *,
    source_sha: str,
    output_dir: str | Path,
    seed: int = 1337,
    max_steps: int = 40,
    batch_size: int = 3,
    verify_checkout: bool = True,
) -> dict[str, Any]:
    """Train S0, persist one verified D05 checkpoint, and prove D07 reload/serving."""

    source_sha = _validate_source_sha(source_sha)
    repo_root = Path(repo_root).resolve()
    output_dir = Path(output_dir).resolve()
    _require(seed >= 0, "seed must be non-negative")
    _require(max_steps >= 1, "max_steps must be positive")
    _require(batch_size >= 1, "batch_size must be positive")
    if verify_checkout:
        _require(_git_head(repo_root) == source_sha, "source_sha does not equal checkout HEAD")

    manifest_path = repo_root / "data/s0/packaged/manifest.json"
    train_path = repo_root / "data/s0/packaged/train.jsonl"
    validation_path = repo_root / "data/s0/packaged/validation.jsonl"
    environment_lock_path = repo_root / "requirements/locks/index.json"
    stage_path = repo_root / "configs/stages/s0_10k.json"

    _require(
        sha256_file(manifest_path) == DATASET_MANIFEST_SHA256,
        "D03 dataset manifest SHA-256 mismatch",
    )
    _require(sha256_file(train_path) == TRAIN_JSONL_SHA256, "D03 train SHA-256 mismatch")
    _require(
        sha256_file(validation_path) == VALIDATION_JSONL_SHA256,
        "D03 validation SHA-256 mismatch",
    )
    dataset_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(
        dataset_manifest.get("dataset_identity_sha256") == DATASET_IDENTITY_SHA256,
        "D03 dataset semantic identity mismatch",
    )

    stage = load_stage_config(stage_path)
    tokenizer = ByteTokenizer()
    _require(stage.canonical_base == "random_init", "canonical S0 Base must be random_init")
    _require(stage.expected_parameters == 10_140, "canonical S0 parameter count drift")
    _require(
        stage.model.vocab_size == tokenizer.vocab_size,
        "D01/D04 vocabulary contract mismatch",
    )

    train_batches, train_ids, train_loss_tokens = _packed_batches(
        repo_root,
        split="train",
        tokenizer=tokenizer,
        batch_size=batch_size,
    )
    validation_batches, validation_ids, validation_loss_tokens = _packed_batches(
        repo_root,
        split="validation",
        tokenizer=tokenizer,
        batch_size=batch_size,
    )
    overlap = sorted(set(train_ids) & set(validation_ids))
    _require(not overlap, "train/validation record identity overlap")

    trainer_config = TrainerConfig(
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
    torch.manual_seed(seed)
    model = TwelveSixDecoder(stage.model, stage.init)
    trainer = Trainer(model, trainer_config, device="cpu")
    run_result = trainer.run(islice(cycle(train_batches), max_steps))
    trainer.assert_checkpoint_safe()
    _require(
        run_result.optimizer_steps_completed == max_steps,
        "S0 artifact training did not reach requested optimizer steps",
    )
    _require(
        trainer.optimizer_step == max_steps,
        "Trainer optimizer step disagrees with requested S0 artifact step",
    )

    environment_lock_sha256 = sha256_file(environment_lock_path)
    run_manifest = _run_manifest(
        source_sha=source_sha,
        stage=stage,
        tokenizer=tokenizer,
        trainer_config=trainer_config,
        environment_lock_sha256=environment_lock_sha256,
        max_steps=max_steps,
        batch_size=batch_size,
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

    if output_dir.exists():
        _require(not any(output_dir.iterdir()), "output_dir must be absent or empty")
    else:
        output_dir.mkdir(parents=True)
    checkpoint_dir = output_dir / "checkpoint"
    manifest = save_trainer_checkpoint(
        checkpoint_dir,
        model=model,
        trainer=trainer,
        identity=identity,
    )
    verified_manifest = verify_checkpoint(checkpoint_dir)
    _require(
        verified_manifest["checkpoint_id"] == manifest["checkpoint_id"],
        "post-save checkpoint verification changed checkpoint identity",
    )

    backend = load_first_party_backend(checkpoint_dir)
    diagnostics = backend.diagnostics()
    _require(diagnostics["git_sha"] == source_sha, "first-party backend source identity drift")
    _require(
        diagnostics["checkpoint_id"] == manifest["checkpoint_id"],
        "first-party backend checkpoint identity drift",
    )
    _require(
        diagnostics["max_context_tokens"] == stage.model.max_seq_len,
        "first-party backend context identity drift",
    )

    prompt = "12-6"
    greedy = generate(
        backend,
        prompt,
        GenerationConfig(max_new_tokens=8, sample=False, seed=seed),
    )
    _require(bool(greedy.generated_token_ids), "greedy proof must generate at least one token")
    sampled_a = generate(
        backend,
        prompt,
        GenerationConfig(
            max_new_tokens=8,
            sample=True,
            temperature=0.8,
            top_k=32,
            top_p=0.95,
            seed=seed + 1,
        ),
    )
    sampled_b = generate(
        backend,
        prompt,
        GenerationConfig(
            max_new_tokens=8,
            sample=True,
            temperature=0.8,
            top_k=32,
            top_p=0.95,
            seed=seed + 1,
        ),
    )
    _require(sampled_a == sampled_b, "seeded sampling is not exactly repeatable")

    stop_token = int(greedy.generated_token_ids[0])
    stopped = generate(
        backend,
        prompt,
        GenerationConfig(
            max_new_tokens=8,
            sample=False,
            seed=seed,
            stop_token_ids=(stop_token,),
        ),
    )
    _require(stopped.stop_reason == "stop_token", "token-stop proof did not stop on token")

    context_prompt = "x" * backend.max_context_tokens
    context_limited = generate(
        backend,
        context_prompt,
        GenerationConfig(max_new_tokens=1, sample=False, seed=seed),
    )
    _require(
        context_limited.stop_reason == "context_limit"
        and not context_limited.generated_token_ids,
        "exact-context prompt must stop before backend overflow",
    )
    over_context_rejected = False
    try:
        generate(
            backend,
            context_prompt + "x",
            GenerationConfig(max_new_tokens=1, sample=False, seed=seed),
        )
    except ValueError:
        over_context_rejected = True
    _require(over_context_rejected, "over-context prompt was not rejected fail closed")

    server = make_server(
        backend,
        host="127.0.0.1",
        port=0,
        model_name="12-6-base",
    )
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    try:
        host, port = server.server_address
        health_status, health = _http_json(host, port, "GET", "/healthz")
        completion_status, completion = _http_json(
            host,
            port,
            "POST",
            "/v1/completions",
            {
                "model": "12-6-base",
                "prompt": prompt,
                "max_tokens": 8,
                "temperature": 0,
                "seed": seed,
            },
        )
        chat_status, chat = _http_json(
            host,
            port,
            "POST",
            "/v1/chat/completions",
            {"model": "12-6-base", "messages": []},
        )
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=10)
    _require(not server_thread.is_alive(), "loopback server did not shut down cleanly")
    _require(health_status == 200 and health.get("status") == "ok", "healthz proof failed")
    choices = completion.get("choices")
    _require(
        completion_status == 200
        and isinstance(choices, list)
        and len(choices) == 1
        and isinstance(choices[0], dict),
        "completion server proof did not return one completion choice",
    )
    _require(
        choices[0].get("text") == greedy.text,
        "loopback completion text diverged from canonical greedy generation",
    )
    chat_error = chat.get("error")
    _require(
        chat_status == 404
        and isinstance(chat_error, dict)
        and "chat completions are not supported" in str(chat_error.get("message", "")),
        "raw Base server did not reject chat semantics explicitly",
    )

    inventory = _checkpoint_inventory(checkpoint_dir)
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "authority": AUTHORITY,
        "identity": {
            "repository": REPOSITORY,
            "source_sha": source_sha,
            "modelspec_sha256": stage.model.identity_sha256(),
            "initspec_sha256": stage.init.identity_sha256(),
            "parameter_count": stage.expected_parameters,
            "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
            "dataset_identity_sha256": DATASET_IDENTITY_SHA256,
            "train_jsonl_sha256": TRAIN_JSONL_SHA256,
            "validation_jsonl_sha256": VALIDATION_JSONL_SHA256,
            "tokenizer_config_sha256": tokenizer.identity.config_sha256,
            "tokenizer_vocab_sha256": tokenizer.identity.vocab_sha256,
            "tokenizer_version": tokenizer.identity.version,
            "packing_sha256": PACKING_CONFIG_HASH,
            "packing_version": PACKING_VERSION,
            "environment_lock_sha256": environment_lock_sha256,
            "run_manifest_sha256": hash_json(run_manifest),
        },
        "training": {
            "seed": seed,
            "max_steps": max_steps,
            "batch_size_examples": batch_size,
            "optimizer_steps": trainer.optimizer_step,
            "tokens_seen": trainer.tokens_seen,
            "optimized_split": "train",
            "train_record_ids": list(train_ids),
            "validation_record_ids": list(validation_ids),
            "train_validation_record_overlap": overlap,
            "train_loss_tokens_per_epoch": train_loss_tokens,
            "validation_scoreable_tokens": validation_loss_tokens,
            "validation_batches_constructed": len(validation_batches),
            "validation_optimized_tokens": 0,
        },
        "checkpoint": {
            "relative_path": "checkpoint",
            "checkpoint_id": manifest["checkpoint_id"],
            "serialization_pickle": manifest["serialization"]["pickle"],
            "verified_after_save": True,
            "inventory": inventory,
            "backend_diagnostics": diagnostics,
        },
        "generation": {
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "prompt_tokens": len(greedy.prompt_token_ids),
            "greedy": _normalized_generation(greedy),
            "seeded_sampling": {
                **_normalized_generation(sampled_a),
                "seed": seed + 1,
                "repeat_exact": sampled_a == sampled_b,
            },
            "token_stop": {
                **_normalized_generation(stopped),
                "configured_stop_token": stop_token,
            },
            "context_limit": {
                "max_context_tokens": backend.max_context_tokens,
                "exact_limit_stop_reason": context_limited.stop_reason,
                "exact_limit_generated_tokens": len(context_limited.generated_token_ids),
                "over_limit_rejected": over_context_rejected,
            },
        },
        "server": {
            "loopback_only": True,
            "healthz_verified": True,
            "completion_status": completion_status,
            "completion_text_sha256": hashlib.sha256(
                str(choices[0]["text"]).encode("utf-8")
            ).hexdigest(),
            "completion_matches_greedy": True,
            "chat_status": chat_status,
            "chat_semantics_rejected": True,
            "hidden_system_or_instruction_prompt": False,
        },
        "claims": {
            "canonical_base": "random_init_pretraining_only",
            "foreign_pretrained_weights_used": False,
            "instruction_or_alignment_behavior_added": False,
            "paid_compute_authorized_or_used": False,
            "audit_pass_claimed": False,
            "candidate_or_stable_promotion": False,
            "windows_nvda_live_execution_claimed": False,
            "public_server_hardening_claimed": False,
        },
    }
    report["evidence_sha256"] = _canonical_hash(report)
    return report


def validate_s0_inference_artifact_evidence(
    payload: Mapping[str, Any],
    *,
    checkpoint_dir: str | Path | None = None,
    expected_source_sha: str | None = None,
) -> dict[str, Any]:
    """Validate one self-hashed report and optionally the materialized checkpoint bytes."""

    report = dict(payload)
    _require(report.get("schema_version") == SCHEMA_VERSION, "unexpected evidence schema")
    _require(report.get("authority") == AUTHORITY, "unexpected evidence authority")
    stored_hash = report.pop("evidence_sha256", None)
    _require(_is_sha256(stored_hash), "evidence_sha256 must be lowercase SHA-256")
    _require(stored_hash == _canonical_hash(report), "evidence self-hash mismatch")
    report["evidence_sha256"] = stored_hash

    identity = report.get("identity")
    _require(isinstance(identity, Mapping), "identity must be a mapping")
    source_sha = _validate_source_sha(identity.get("source_sha"))
    _require(identity.get("repository") == REPOSITORY, "repository identity mismatch")
    if expected_source_sha is not None:
        _require(source_sha == _validate_source_sha(expected_source_sha), "source SHA mismatch")
    for field in (
        "modelspec_sha256",
        "initspec_sha256",
        "dataset_manifest_sha256",
        "dataset_identity_sha256",
        "train_jsonl_sha256",
        "validation_jsonl_sha256",
        "tokenizer_config_sha256",
        "tokenizer_vocab_sha256",
        "packing_sha256",
        "environment_lock_sha256",
        "run_manifest_sha256",
    ):
        _require(_is_sha256(identity.get(field)), f"identity.{field} must be SHA-256")
    _require(identity.get("parameter_count") == 10_140, "S0 parameter count drift")

    training = report.get("training")
    _require(isinstance(training, Mapping), "training must be a mapping")
    _require(training.get("optimized_split") == "train", "only train may be optimized")
    _require(training.get("validation_optimized_tokens") == 0, "validation was optimized")
    _require(training.get("train_validation_record_overlap") == [], "split overlap detected")
    _require(
        training.get("optimizer_steps") == training.get("max_steps"),
        "optimizer step evidence is incomplete",
    )

    checkpoint = report.get("checkpoint")
    _require(isinstance(checkpoint, Mapping), "checkpoint must be a mapping")
    _require(_is_sha256(checkpoint.get("checkpoint_id")), "checkpoint_id must be SHA-256")
    _require(checkpoint.get("serialization_pickle") is False, "pickle checkpoint rejected")
    _require(checkpoint.get("verified_after_save") is True, "checkpoint was not verified")

    generation = report.get("generation")
    _require(isinstance(generation, Mapping), "generation must be a mapping")
    sampled = generation.get("seeded_sampling")
    token_stop = generation.get("token_stop")
    context_limit = generation.get("context_limit")
    _require(isinstance(sampled, Mapping), "seeded_sampling must be a mapping")
    _require(sampled.get("repeat_exact") is True, "seeded sampling repeatability failed")
    _require(isinstance(token_stop, Mapping), "token_stop must be a mapping")
    _require(token_stop.get("stop_reason") == "stop_token", "token stop proof failed")
    _require(isinstance(context_limit, Mapping), "context_limit must be a mapping")
    _require(
        context_limit.get("exact_limit_stop_reason") == "context_limit"
        and context_limit.get("exact_limit_generated_tokens") == 0
        and context_limit.get("over_limit_rejected") is True,
        "context-limit proof failed",
    )

    server = report.get("server")
    _require(isinstance(server, Mapping), "server must be a mapping")
    _require(server.get("loopback_only") is True, "server evidence must be loopback-only")
    _require(server.get("healthz_verified") is True, "server health proof failed")
    _require(server.get("completion_status") == 200, "completion server proof failed")
    _require(server.get("completion_matches_greedy") is True, "server parity proof failed")
    _require(server.get("chat_status") == 404, "chat endpoint must remain unsupported")
    _require(server.get("chat_semantics_rejected") is True, "chat semantics were not rejected")
    _require(
        server.get("hidden_system_or_instruction_prompt") is False,
        "raw Base evidence cannot include hidden instruction prompts",
    )

    claims = report.get("claims")
    _require(isinstance(claims, Mapping), "claims must be a mapping")
    _require(
        claims.get("canonical_base") == "random_init_pretraining_only",
        "canonical Base claim drift",
    )
    for field in (
        "foreign_pretrained_weights_used",
        "instruction_or_alignment_behavior_added",
        "paid_compute_authorized_or_used",
        "audit_pass_claimed",
        "candidate_or_stable_promotion",
        "windows_nvda_live_execution_claimed",
        "public_server_hardening_claimed",
    ):
        _require(claims.get(field) is False, f"forbidden overclaim: {field}")

    if checkpoint_dir is not None:
        checkpoint_path = Path(checkpoint_dir).resolve()
        materialized_manifest = verify_checkpoint(checkpoint_path)
        _require(
            materialized_manifest["checkpoint_id"] == checkpoint["checkpoint_id"],
            "materialized checkpoint_id differs from evidence",
        )
        actual_inventory = _checkpoint_inventory(checkpoint_path)
        _require(
            actual_inventory == checkpoint.get("inventory"),
            "materialized checkpoint inventory differs from evidence",
        )
        materialized_identity = materialized_manifest.get("identity")
        _require(
            isinstance(materialized_identity, Mapping)
            and materialized_identity.get("git_sha") == source_sha,
            "materialized checkpoint source SHA differs from evidence",
        )
    return report
