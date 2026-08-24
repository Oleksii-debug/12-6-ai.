"""Retained trained-checkpoint inference evidence for canonical S0 Base.

This module composes existing D01/D02/D03/D04/D05/D07 contracts. It does not
reimplement model architecture, tokenizer semantics, checkpoint serialization,
sampling, parity, CLI, or OpenAI-compatible completion behavior.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from itertools import cycle, islice
from pathlib import Path
from typing import Any

import torch

from twelve_six.checkpoint import (
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
from twelve_six.training import Trainer, TrainerConfig
from twelve_six.training.s0_evidence import (
    DATASET_IDENTITY_SHA256,
    DATASET_MANIFEST_SHA256,
    TRAIN_JSONL_SHA256,
    VALIDATION_JSONL_SHA256,
    _tensor_batches,
)

from .contracts import GenerationConfig, GenerationResult
from .first_party import load_first_party_backend
from .generation import generate
from .openai_compat import completion_response
from .parity import compare_backends

SCHEMA_VERSION = "12-6.s0-trained-inference-evidence.v1"
AUTHORITY = "LOCAL_FREE_OR_FREE_HOSTED_CPU_EVIDENCE_NOT_PROMOTION"
REPOSITORY = "Oleksii-debug/12-6-ai."
MODEL_SPEC_SHA256 = "86c75b31dff05b7b5db9f6ed068c571a6ead01ba663412fe630f5e52b09d9b6b"
INIT_SPEC_SHA256 = "86483c6df623e80cab2f73aba718863fce18af6fe3b12430c1348414d92b48a5"
TOKENIZER_CONFIG_SHA256 = "b04055c1061dd641dcab7cb9d62a931f09b8d1a070140a926ceb4e91d73ca8e1"
TOKENIZER_VOCAB_SHA256 = "905ed40bb42cc4d550e228ff5f24158d504b38e8ed5974dfa3077bd5867ad571"
PARAMETER_COUNT = 10_140
_SHA256_HEX = frozenset("0123456789abcdef")


class S0InferenceEvidenceError(ValueError):
    """Raised when retained S0 inference evidence fails closed."""


def _canonical_hash(value: dict[str, Any]) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _validate_source_sha(source_sha: str) -> None:
    if (
        not isinstance(source_sha, str)
        or len(source_sha) != 40
        or set(source_sha) - _SHA256_HEX
    ):
        raise S0InferenceEvidenceError(
            "source_sha must be a full lowercase 40-hex Git SHA"
        )


def _git_head(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise S0InferenceEvidenceError(
            "trained inference evidence requires a Git checkout"
        ) from exc


def _generation_dict(result: GenerationResult) -> dict[str, Any]:
    payload = {
        "prompt_token_ids": list(result.prompt_token_ids),
        "generated_token_ids": list(result.generated_token_ids),
        "text": result.text,
        "stop_reason": result.stop_reason,
    }
    payload["result_sha256"] = _canonical_hash(payload)
    return payload


def _run_cli(
    checkpoint: Path,
    *,
    prompt: str,
    stdin: bool,
    json_output: bool,
    max_new_tokens: int = 4,
) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "twelve_six.inference.cli",
        "--checkpoint",
        str(checkpoint),
        "--max-new-tokens",
        str(max_new_tokens),
        "--greedy",
    ]
    input_text: str | None = None
    if stdin:
        input_text = prompt
    else:
        command.extend(["--prompt", prompt])
    if json_output:
        command.append("--json")

    completed = subprocess.run(
        command,
        input=input_text,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise S0InferenceEvidenceError(
            f"real-checkpoint CLI failed with exit={completed.returncode}: "
            f"{completed.stderr.strip()}"
        )
    stdout = completed.stdout.rstrip("\n")
    stderr = completed.stderr.rstrip("\n")
    result: dict[str, Any] = {
        "exit_code": completed.returncode,
        "stdout_sha256": hashlib.sha256(stdout.encode("utf-8")).hexdigest(),
        "stderr_sha256": hashlib.sha256(stderr.encode("utf-8")).hexdigest(),
        "stdin_mode": stdin,
        "json_mode": json_output,
        "diagnostics_present": "backend: kind=first_party_torch" in stderr
        and "generation: mode=greedy" in stderr,
    }
    if json_output:
        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise S0InferenceEvidenceError(
                "CLI --json did not emit one valid JSON object"
            ) from exc
        if not isinstance(payload, dict):
            raise S0InferenceEvidenceError("CLI --json payload must be an object")
        result["payload"] = payload
    return result


def _run_manifest(
    *,
    source_sha: str,
    stage: Any,
    tokenizer: ByteTokenizer,
    trainer_config: TrainerConfig,
    environment_lock_sha256: str,
    max_steps: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_id": f"s0-d05-trained-inference-{source_sha[:12]}",
        "stage": "S0",
        "run_kind": "trained_checkpoint_inference_evidence",
        "state": "RUNNING",
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
            "target_steps": max_steps,
        },
        "environment": {"lock_sha256": environment_lock_sha256},
    }


def collect_s0_trained_inference_evidence(
    root: str | Path,
    *,
    source_sha: str,
    output_dir: str | Path,
    seed: int = 1337,
    max_steps: int = 40,
    batch_size: int = 3,
    verify_checkout: bool = True,
) -> dict[str, Any]:
    """Train S0, retain a verified checkpoint, and prove D07 behavior on reload."""

    _validate_source_sha(source_sha)
    if seed < 0:
        raise S0InferenceEvidenceError("seed must be non-negative")
    if max_steps < 1:
        raise S0InferenceEvidenceError("max_steps must be >= 1")
    if batch_size < 1:
        raise S0InferenceEvidenceError("batch_size must be >= 1")

    root = Path(root).resolve()
    output_dir = Path(output_dir).resolve()
    checkpoint = output_dir / "checkpoint"
    evidence_path = output_dir / "s0-trained-inference-evidence.json"
    if output_dir.exists():
        raise S0InferenceEvidenceError(
            "output_dir must not already exist; evidence publication is immutable"
        )
    if verify_checkout and _git_head(root) != source_sha:
        raise S0InferenceEvidenceError(
            "source_sha is stale: it does not equal the checkout HEAD"
        )
    output_dir.mkdir(parents=True)

    stage = load_stage_config(root / "configs/stages/s0_10k.json")
    tokenizer = ByteTokenizer()
    if stage.canonical_base != "random_init":
        raise S0InferenceEvidenceError("canonical S0 Base must remain random_init")
    if stage.expected_parameters != PARAMETER_COUNT:
        raise S0InferenceEvidenceError("S0 parameter count drift")
    if stage.model.identity_sha256() != MODEL_SPEC_SHA256:
        raise S0InferenceEvidenceError("S0 ModelSpec identity drift")
    if stage.init.identity_sha256() != INIT_SPEC_SHA256:
        raise S0InferenceEvidenceError("S0 InitSpec identity drift")
    if tokenizer.identity.config_sha256 != TOKENIZER_CONFIG_SHA256:
        raise S0InferenceEvidenceError("S0 tokenizer config identity drift")
    if tokenizer.identity.vocab_sha256 != TOKENIZER_VOCAB_SHA256:
        raise S0InferenceEvidenceError("S0 tokenizer vocabulary identity drift")
    if sha256_file(root / "data/s0/packaged/manifest.json") != DATASET_MANIFEST_SHA256:
        raise S0InferenceEvidenceError("D03 dataset manifest identity drift")
    if sha256_file(root / "data/s0/packaged/train.jsonl") != TRAIN_JSONL_SHA256:
        raise S0InferenceEvidenceError("D03 train split identity drift")
    if (
        sha256_file(root / "data/s0/packaged/validation.jsonl")
        != VALIDATION_JSONL_SHA256
    ):
        raise S0InferenceEvidenceError("D03 validation split identity drift")

    train_batches, train_ids, _ = _tensor_batches(
        root,
        split="train",
        tokenizer=tokenizer,
        batch_size=batch_size,
    )
    _, validation_ids, _ = _tensor_batches(
        root,
        split="validation",
        tokenizer=tokenizer,
        batch_size=batch_size,
    )
    if set(train_ids) & set(validation_ids):
        raise S0InferenceEvidenceError("train/validation record identity overlap")

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

    # Canonical scratch lineage: declared seed is applied before model construction.
    torch.manual_seed(seed)
    model = TwelveSixDecoder(stage.model, stage.init)
    trainer = Trainer(model, trainer_config, device="cpu")
    run_result = trainer.run(islice(cycle(train_batches), max_steps))
    if run_result.optimizer_steps_completed != max_steps:
        raise S0InferenceEvidenceError("training did not reach requested optimizer steps")
    trainer.assert_checkpoint_safe()

    environment_lock_sha256 = sha256_file(root / "requirements/locks/index.json")
    run_manifest = _run_manifest(
        source_sha=source_sha,
        stage=stage,
        tokenizer=tokenizer,
        trainer_config=trainer_config,
        environment_lock_sha256=environment_lock_sha256,
        max_steps=max_steps,
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
    manifest = save_trainer_checkpoint(
        checkpoint,
        model=model,
        trainer=trainer,
        identity=identity,
    )
    verified_manifest = verify_checkpoint(checkpoint)
    if verified_manifest["checkpoint_id"] != manifest["checkpoint_id"]:
        raise S0InferenceEvidenceError("verified checkpoint identity changed after save")

    direct_backend = S0TorchInferenceBackend(model, tokenizer)
    reloaded_backend = load_first_party_backend(checkpoint)
    diagnostics = reloaded_backend.diagnostics()
    required_diagnostics = {
        "checkpoint_id": manifest["checkpoint_id"],
        "git_sha": source_sha,
        "model_spec_sha256": MODEL_SPEC_SHA256,
        "parameter_count": PARAMETER_COUNT,
        "vocab_size": 256,
        "max_context_tokens": stage.model.max_seq_len,
        "tokenizer_config_sha256": TOKENIZER_CONFIG_SHA256,
        "tokenizer_vocab_sha256": TOKENIZER_VOCAB_SHA256,
        "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
        "step": max_steps,
        "tokens_seen": trainer.tokens_seen,
    }
    for key, expected in required_diagnostics.items():
        if diagnostics.get(key) != expected:
            raise S0InferenceEvidenceError(
                f"reloaded backend diagnostic mismatch for {key}: "
                f"{diagnostics.get(key)!r} != {expected!r}"
            )

    prompts = ("12-6", "Привіт", "def ")
    parity = compare_backends(
        direct_backend,
        reloaded_backend,
        prompts,
        max_new_tokens=8,
        atol=0.0,
        rtol=0.0,
    )
    if not parity.passed or parity.max_abs_error != 0 or parity.max_rel_error != 0:
        raise S0InferenceEvidenceError("direct-vs-reloaded zero-tolerance parity failed")

    greedy_results: list[dict[str, Any]] = []
    for prompt in prompts:
        result = generate(
            reloaded_backend,
            prompt,
            GenerationConfig(max_new_tokens=8, sample=False, seed=seed),
        )
        greedy_results.append(_generation_dict(result))

    sample_config = GenerationConfig(
        max_new_tokens=8,
        sample=True,
        temperature=0.8,
        top_k=32,
        top_p=0.95,
        seed=4242,
    )
    sample_a = generate(reloaded_backend, "12-6", sample_config)
    sample_b = generate(reloaded_backend, "12-6", sample_config)
    if sample_a != sample_b:
        raise S0InferenceEvidenceError("seeded sampling is not repeatable after reload")

    first_token_probe = generate(
        reloaded_backend,
        "12-6",
        GenerationConfig(max_new_tokens=1, sample=False, seed=seed),
    )
    if not first_token_probe.generated_token_ids:
        raise S0InferenceEvidenceError("greedy stop probe generated no token")
    stop_token_id = first_token_probe.generated_token_ids[0]
    token_stop = generate(
        reloaded_backend,
        "12-6",
        GenerationConfig(
            max_new_tokens=8,
            sample=False,
            seed=seed,
            stop_token_ids=(stop_token_id,),
        ),
    )
    if token_stop.stop_reason != "stop_token":
        raise S0InferenceEvidenceError("real-checkpoint token stop did not trigger")

    stop_text = reloaded_backend.decode((stop_token_id,))
    if not stop_text:
        raise S0InferenceEvidenceError("real-checkpoint stop-string probe decoded empty text")
    text_stop = generate(
        reloaded_backend,
        "12-6",
        GenerationConfig(
            max_new_tokens=8,
            sample=False,
            seed=seed,
            stop_strings=(stop_text,),
        ),
    )
    if text_stop.stop_reason != "stop_string":
        raise S0InferenceEvidenceError("real-checkpoint text stop did not trigger")

    exact_context_prompt = "x" * reloaded_backend.max_context_tokens
    context_result = generate(
        reloaded_backend,
        exact_context_prompt,
        GenerationConfig(max_new_tokens=1, sample=False),
    )
    if context_result.stop_reason != "context_limit":
        raise S0InferenceEvidenceError("exact-context prompt did not stop at context_limit")
    over_context_rejected = False
    try:
        generate(
            reloaded_backend,
            exact_context_prompt + "x",
            GenerationConfig(max_new_tokens=1, sample=False),
        )
    except ValueError:
        over_context_rejected = True
    if not over_context_rejected:
        raise S0InferenceEvidenceError("over-context prompt was not rejected")

    cli_json = _run_cli(
        checkpoint,
        prompt="12-6",
        stdin=False,
        json_output=True,
    )
    cli_stdin = _run_cli(
        checkpoint,
        prompt="12-6",
        stdin=True,
        json_output=False,
    )
    if not cli_json["diagnostics_present"] or not cli_stdin["diagnostics_present"]:
        raise S0InferenceEvidenceError("plain CLI diagnostics were not emitted")
    cli_backend = cli_json["payload"].get("backend")
    if not isinstance(cli_backend, dict) or cli_backend.get("checkpoint_id") != manifest[
        "checkpoint_id"
    ]:
        raise S0InferenceEvidenceError("CLI JSON diagnostics lost checkpoint identity")

    direct_greedy_4 = generate(
        reloaded_backend,
        "12-6",
        GenerationConfig(max_new_tokens=4, sample=False, seed=0),
    )
    completion = completion_response(
        reloaded_backend,
        {
            "prompt": "12-6",
            "max_tokens": 4,
            "temperature": 0,
            "top_p": 1.0,
            "seed": 0,
        },
        response_id="cmpl-evidence",
        created=0,
        model_name="12-6-base",
    )
    choice = completion["choices"][0]
    if not isinstance(choice, dict) or choice.get("text") != direct_greedy_4.text:
        raise S0InferenceEvidenceError(
            "OpenAI-compatible raw completion diverged from canonical generation"
        )

    corrupt_rejected = False
    with tempfile.TemporaryDirectory(prefix="twelve-six-d05-corrupt-probe-") as temp_dir:
        corrupt_checkpoint = Path(temp_dir) / "checkpoint"
        shutil.copytree(checkpoint, corrupt_checkpoint)
        weights_path = corrupt_checkpoint / "model.safetensors"
        weights = bytearray(weights_path.read_bytes())
        if not weights:
            raise S0InferenceEvidenceError("model.safetensors is unexpectedly empty")
        weights[-1] ^= 0x01
        weights_path.write_bytes(weights)
        try:
            load_first_party_backend(corrupt_checkpoint)
        except (OSError, RuntimeError, TypeError, ValueError):
            corrupt_rejected = True
    if not corrupt_rejected:
        raise S0InferenceEvidenceError("corrupt checkpoint was accepted by first-party loader")

    artifact_hashes = {
        path.name: sha256_file(path)
        for path in sorted(checkpoint.iterdir())
        if path.is_file()
    }
    evidence: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "authority": AUTHORITY,
        "identity": {
            "repository": REPOSITORY,
            "source_sha": source_sha,
            "stage": "S0",
            "modelspec_sha256": MODEL_SPEC_SHA256,
            "initspec_sha256": INIT_SPEC_SHA256,
            "parameter_count": PARAMETER_COUNT,
            "tokenizer_config_sha256": TOKENIZER_CONFIG_SHA256,
            "tokenizer_vocab_sha256": TOKENIZER_VOCAB_SHA256,
            "tokenizer_version": tokenizer.identity.version,
            "max_context_tokens": reloaded_backend.max_context_tokens,
            "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
            "dataset_identity_sha256": DATASET_IDENTITY_SHA256,
            "train_jsonl_sha256": TRAIN_JSONL_SHA256,
            "validation_jsonl_sha256": VALIDATION_JSONL_SHA256,
            "packing_config_sha256": PACKING_CONFIG_HASH,
            "environment_lock_sha256": environment_lock_sha256,
            "seed": seed,
            "optimizer_steps": trainer.optimizer_step,
            "tokens_seen": trainer.tokens_seen,
        },
        "checkpoint": {
            "relative_path": "checkpoint",
            "checkpoint_id": manifest["checkpoint_id"],
            "retained": True,
            "verified_before_inference": True,
            "artifact_sha256": artifact_hashes,
            "corrupt_copy_rejected": corrupt_rejected,
        },
        "backend_diagnostics": diagnostics,
        "parity": parity.to_dict(),
        "generation": {
            "greedy": greedy_results,
            "seeded_sampling": {
                "config": {
                    "seed": sample_config.seed,
                    "temperature": sample_config.temperature,
                    "top_k": sample_config.top_k,
                    "top_p": sample_config.top_p,
                    "max_new_tokens": sample_config.max_new_tokens,
                },
                "repeatable": sample_a == sample_b,
                "result": _generation_dict(sample_a),
            },
            "stop_semantics": {
                "token_stop_id": stop_token_id,
                "token_stop": _generation_dict(token_stop),
                "text_stop": _generation_dict(text_stop),
            },
            "context_semantics": {
                "exact_context_tokens": reloaded_backend.max_context_tokens,
                "exact_context_stop_reason": context_result.stop_reason,
                "over_context_rejected": over_context_rejected,
            },
        },
        "cli": {
            "prompt_json": cli_json,
            "stdin_plain": cli_stdin,
            "plain_text_no_tui_or_ansi_required": True,
            "windows_nvda_live_tested": False,
        },
        "openai_compatible_handoff": {
            "raw_base_completion_matches_canonical": True,
            "response": completion,
            "chat_or_system_semantics_added": False,
            "network_server_implementation_inherited_from_pr86": True,
        },
        "claims": {
            "foreign_pretrained_weights_used": False,
            "instruction_or_alignment_behavior_added": False,
            "paid_compute_authorized_or_used": False,
            "candidate_or_stable_promotion": False,
            "audit_verdict": False,
            "windows_nvda_live_pass": False,
        },
    }
    evidence["evidence_sha256"] = _canonical_hash(evidence)
    evidence_path.write_text(
        json.dumps(
            evidence,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return evidence


def validate_s0_trained_inference_evidence(
    evidence: dict[str, Any],
    *,
    checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    """Validate evidence and optionally re-verify the retained checkpoint bytes."""

    if evidence.get("schema_version") != SCHEMA_VERSION:
        raise S0InferenceEvidenceError("unexpected trained-inference evidence schema")
    expected_hash = evidence.get("evidence_sha256")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise S0InferenceEvidenceError("evidence_sha256 is missing or malformed")
    unhashed = dict(evidence)
    del unhashed["evidence_sha256"]
    if _canonical_hash(unhashed) != expected_hash:
        raise S0InferenceEvidenceError("trained-inference evidence self-hash mismatch")

    identity = evidence.get("identity")
    if not isinstance(identity, dict):
        raise S0InferenceEvidenceError("identity must be an object")
    expected_identity = {
        "repository": REPOSITORY,
        "modelspec_sha256": MODEL_SPEC_SHA256,
        "initspec_sha256": INIT_SPEC_SHA256,
        "parameter_count": PARAMETER_COUNT,
        "tokenizer_config_sha256": TOKENIZER_CONFIG_SHA256,
        "tokenizer_vocab_sha256": TOKENIZER_VOCAB_SHA256,
        "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
        "dataset_identity_sha256": DATASET_IDENTITY_SHA256,
        "train_jsonl_sha256": TRAIN_JSONL_SHA256,
        "validation_jsonl_sha256": VALIDATION_JSONL_SHA256,
        "packing_config_sha256": PACKING_CONFIG_HASH,
    }
    for key, expected in expected_identity.items():
        if identity.get(key) != expected:
            raise S0InferenceEvidenceError(f"identity mismatch for {key}")
    _validate_source_sha(identity.get("source_sha"))

    checkpoint_record = evidence.get("checkpoint")
    if not isinstance(checkpoint_record, dict):
        raise S0InferenceEvidenceError("checkpoint evidence must be an object")
    if (
        checkpoint_record.get("retained") is not True
        or checkpoint_record.get("verified_before_inference") is not True
        or checkpoint_record.get("corrupt_copy_rejected") is not True
    ):
        raise S0InferenceEvidenceError("checkpoint proof is incomplete")

    parity = evidence.get("parity")
    if not isinstance(parity, dict):
        raise S0InferenceEvidenceError("parity evidence must be an object")
    if (
        parity.get("passed") is not True
        or parity.get("atol") != 0.0
        or parity.get("rtol") != 0.0
        or parity.get("max_abs_error") != 0.0
        or parity.get("max_rel_error") != 0.0
    ):
        raise S0InferenceEvidenceError("zero-tolerance trained checkpoint parity failed")

    generation = evidence.get("generation")
    if not isinstance(generation, dict):
        raise S0InferenceEvidenceError("generation evidence must be an object")
    sampling = generation.get("seeded_sampling")
    stops = generation.get("stop_semantics")
    context = generation.get("context_semantics")
    if not isinstance(sampling, dict) or sampling.get("repeatable") is not True:
        raise S0InferenceEvidenceError("seeded sampling repeatability is missing")
    if not isinstance(stops, dict):
        raise S0InferenceEvidenceError("stop semantics evidence is missing")
    for key, reason in (("token_stop", "stop_token"), ("text_stop", "stop_string")):
        value = stops.get(key)
        if not isinstance(value, dict) or value.get("stop_reason") != reason:
            raise S0InferenceEvidenceError(f"{key} evidence is invalid")
    if (
        not isinstance(context, dict)
        or context.get("exact_context_stop_reason") != "context_limit"
        or context.get("over_context_rejected") is not True
    ):
        raise S0InferenceEvidenceError("context-limit evidence is incomplete")

    cli = evidence.get("cli")
    if not isinstance(cli, dict):
        raise S0InferenceEvidenceError("CLI evidence must be an object")
    for key in ("prompt_json", "stdin_plain"):
        value = cli.get(key)
        if (
            not isinstance(value, dict)
            or value.get("exit_code") != 0
            or value.get("diagnostics_present") is not True
        ):
            raise S0InferenceEvidenceError(f"{key} CLI evidence failed")

    handoff = evidence.get("openai_compatible_handoff")
    if (
        not isinstance(handoff, dict)
        or handoff.get("raw_base_completion_matches_canonical") is not True
        or handoff.get("chat_or_system_semantics_added") is not False
    ):
        raise S0InferenceEvidenceError("raw completion handoff evidence is invalid")

    claims = evidence.get("claims")
    if not isinstance(claims, dict) or any(
        claims.get(key) is not False
        for key in (
            "foreign_pretrained_weights_used",
            "instruction_or_alignment_behavior_added",
            "paid_compute_authorized_or_used",
            "candidate_or_stable_promotion",
            "audit_verdict",
            "windows_nvda_live_pass",
        )
    ):
        raise S0InferenceEvidenceError("truth-boundary claims were weakened")

    if checkpoint is not None:
        checkpoint_path = Path(checkpoint)
        manifest = verify_checkpoint(checkpoint_path)
        if manifest.get("checkpoint_id") != checkpoint_record.get("checkpoint_id"):
            raise S0InferenceEvidenceError("retained checkpoint_id does not match evidence")
        current_hashes = {
            path.name: sha256_file(path)
            for path in sorted(checkpoint_path.iterdir())
            if path.is_file()
        }
        if current_hashes != checkpoint_record.get("artifact_sha256"):
            raise S0InferenceEvidenceError(
                "retained checkpoint bytes do not match evidence inventory"
            )
        backend = load_first_party_backend(checkpoint_path)
        diagnostics = backend.diagnostics()
        if diagnostics.get("git_sha") != identity["source_sha"]:
            raise S0InferenceEvidenceError(
                "retained checkpoint source SHA does not match evidence"
            )

    return {
        "status": "PASS",
        "schema_version": SCHEMA_VERSION,
        "source_sha": identity["source_sha"],
        "checkpoint_id": checkpoint_record["checkpoint_id"],
        "evidence_sha256": expected_hash,
        "zero_tolerance_parity": True,
        "seeded_sampling_repeatable": True,
    }
