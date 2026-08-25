"""Retained real-S0 checkpoint and first-party generation evidence.

This module intentionally reuses the exact D04 S0 candidate orchestration helpers
instead of reimplementing D01 architecture, D02 training, D04 token/data semantics,
D05 checkpoint format, or D07 generation/parity logic.  Its job is narrower: turn
the already-proven ephemeral train->resume->reload path into a durable checkpoint
artifact plus independently verifiable inference evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from twelve_six.checkpoint import hash_json, save_trainer_checkpoint, sha256_file, verify_checkpoint
from twelve_six.integration import S0TorchInferenceBackend
from twelve_six.model import TwelveSixDecoder, load_stage_config
from twelve_six.packing import PACKING_CONFIG_HASH, PACKING_VERSION
from twelve_six.s0_candidate_evaluation import (
    _bind_identity,
    _load_jsonl,
    _make_batches,
    _run_manifest,
    _seed_all,
    _train_range,
    collect_s0_candidate_evidence,
)
from twelve_six.tokenization import ByteTokenizer
from twelve_six.training import Trainer, TrainerConfig

from .contracts import GenerationConfig, GenerationResult
from .first_party import load_first_party_backend
from .generation import generate
from .parity import compare_backends

SCHEMA_VERSION = "12-6.s0-retained-generation-artifact.v1"
AUTHORITY = "LOCAL_FREE_OR_FREE_HOSTED_CPU_EVIDENCE_NOT_PROMOTION"
_REPOSITORY = "Oleksii-debug/12-6-ai."
_SHA256_CHARS = frozenset("0123456789abcdef")


class S0GenerationArtifactError(ValueError):
    """Raised when retained S0 generation evidence fails closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise S0GenerationArtifactError(message)


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _SHA256_CHARS


def _is_git_sha(value: object) -> bool:
    return isinstance(value, str) and len(value) == 40 and set(value) <= _SHA256_CHARS


def _canonical_hash(value: Mapping[str, Any]) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _text_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _token_hash(token_ids: tuple[int, ...]) -> str:
    raw = json.dumps(list(token_ids), separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _generation_record(result: GenerationResult, *, seed: int, mode: str) -> dict[str, Any]:
    return {
        "mode": mode,
        "seed": seed,
        "prompt_tokens": len(result.prompt_token_ids),
        "generated_tokens": len(result.generated_token_ids),
        "generated_token_ids": list(result.generated_token_ids),
        "generated_token_ids_sha256": _token_hash(result.generated_token_ids),
        "decoded_text_sha256": _text_hash(result.text),
        "stop_reason": result.stop_reason,
    }


def _trainer_config(*, train_steps: int, seed: int) -> TrainerConfig:
    return TrainerConfig(
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


def build_s0_generation_artifact(
    repo_root: str | Path,
    *,
    candidate_sha: str,
    checkpoint_out: str | Path,
    train_steps: int = 40,
    seed: int = 20260824,
) -> dict[str, Any]:
    """Build and verify one retained real-S0 checkpoint and generation evidence.

    The function first executes the strict D04 exact-candidate evidence path.  It then
    repeats the same deterministic full training trajectory solely to persist the final
    D05 checkpoint.  The retained checkpoint is accepted only if its checkpoint_id is
    byte/identity-equivalent to the final checkpoint produced by the D04 interrupted
    resume path.  This makes the durable artifact a proof-carrying continuation of the
    already integrated path rather than a parallel training implementation.
    """

    _require(_is_git_sha(candidate_sha), "candidate_sha must be a full lowercase Git SHA")
    _require(isinstance(train_steps, int) and not isinstance(train_steps, bool), "train_steps must be an integer")
    _require(train_steps >= 4 and train_steps % 2 == 0, "train_steps must be an even integer >= 4")
    _require(isinstance(seed, int) and not isinstance(seed, bool) and seed >= 0, "seed must be a non-negative integer")

    root = Path(repo_root).resolve()
    checkpoint_path = Path(checkpoint_out).resolve()
    _require(not checkpoint_path.exists(), "checkpoint_out must not already exist")

    strict_evidence = collect_s0_candidate_evidence(
        root,
        candidate_sha,
        train_steps=train_steps,
        seed=seed,
        verify_checkout=True,
    )
    expected_checkpoint_id = strict_evidence["checkpoint"]["final_checkpoint_id"]

    stage = load_stage_config(root / "configs/stages/s0_10k.json")
    tokenizer = ByteTokenizer()
    _require(stage.canonical_base == "random_init", "S0 canonical Base must remain random_init")
    _require(stage.model.vocab_size == tokenizer.vocab_size, "model/tokenizer vocabulary mismatch")

    train_path = root / "data/s0/packaged/train.jsonl"
    manifest_path = root / "data/s0/packaged/manifest.json"
    environment_lock_path = root / "requirements/locks/index.json"
    train_rows = _load_jsonl(train_path)
    train_batches, _ = _make_batches(
        train_rows,
        tokenizer,
        max_seq_len=stage.model.max_seq_len,
    )
    config = _trainer_config(train_steps=train_steps, seed=seed)

    _seed_all(seed)
    model = TwelveSixDecoder(stage.model, stage.init)
    trainer = Trainer(model, config, device="cpu")
    _train_range(trainer, train_batches, start_step=0, end_step=train_steps)

    dataset_manifest_sha256 = sha256_file(manifest_path)
    train_sha256 = sha256_file(train_path)
    environment_lock_sha256 = sha256_file(environment_lock_path)
    run_manifest = _run_manifest(
        candidate_sha=candidate_sha,
        stage=stage,
        tokenizer=tokenizer,
        trainer_config=config,
        dataset_manifest_sha256=dataset_manifest_sha256,
        train_sha256=train_sha256,
        environment_lock_sha256=environment_lock_sha256,
        train_steps=train_steps,
    )
    identity = _bind_identity(
        run_manifest=run_manifest,
        stage=stage,
        tokenizer=tokenizer,
        trainer=trainer,
        environment_lock_sha256=environment_lock_sha256,
    )
    manifest = save_trainer_checkpoint(
        checkpoint_path,
        model=model,
        trainer=trainer,
        identity=identity,
    )
    _require(
        manifest["checkpoint_id"] == expected_checkpoint_id,
        "retained checkpoint_id differs from strict D04 interrupted-resume final checkpoint",
    )

    verified = verify_checkpoint(checkpoint_path)
    _require(
        verified["checkpoint_id"] == manifest["checkpoint_id"],
        "retained checkpoint failed post-publish identity verification",
    )

    direct_backend = S0TorchInferenceBackend(model, tokenizer)
    reloaded_backend = load_first_party_backend(checkpoint_path)
    diagnostics = reloaded_backend.diagnostics()
    _require(diagnostics["git_sha"] == candidate_sha, "reloaded backend Git identity mismatch")
    _require(
        diagnostics["model_spec_sha256"] == hash_json(stage.model.to_dict()),
        "reloaded backend ModelSpec identity mismatch",
    )
    _require(
        diagnostics["tokenizer_config_sha256"] == tokenizer.identity.config_sha256,
        "reloaded backend tokenizer config identity mismatch",
    )
    _require(
        diagnostics["tokenizer_vocab_sha256"] == tokenizer.identity.vocab_sha256,
        "reloaded backend tokenizer vocab identity mismatch",
    )

    prompts = ("12-6", "Base")
    parity = compare_backends(
        direct_backend,
        reloaded_backend,
        prompts,
        max_new_tokens=8,
        atol=0.0,
        rtol=0.0,
    )
    _require(parity.passed, "retained checkpoint failed zero-tolerance logits/token/decode parity")

    prompt = "12-6"
    greedy_config = GenerationConfig(max_new_tokens=8, sample=False, seed=seed)
    greedy_direct = generate(direct_backend, prompt, greedy_config)
    greedy_reloaded = generate(reloaded_backend, prompt, greedy_config)
    _require(greedy_direct == greedy_reloaded, "greedy direct/reloaded generation mismatch")

    sample_seed = seed + 17
    sample_config = GenerationConfig(
        max_new_tokens=8,
        sample=True,
        temperature=0.8,
        top_k=20,
        top_p=0.95,
        seed=sample_seed,
    )
    sample_a = generate(reloaded_backend, prompt, sample_config)
    sample_b = generate(reloaded_backend, prompt, sample_config)
    _require(sample_a == sample_b, "seeded sampling is not repeatable on retained checkpoint")

    _require(bool(greedy_reloaded.generated_token_ids), "greedy fixture generated zero tokens")
    first_token = greedy_reloaded.generated_token_ids[0]
    token_stop = generate(
        reloaded_backend,
        prompt,
        GenerationConfig(
            max_new_tokens=8,
            sample=False,
            seed=seed,
            stop_token_ids=(first_token,),
        ),
    )
    _require(token_stop.stop_reason == "stop_token", "real checkpoint token stop did not trigger")
    _require(len(token_stop.generated_token_ids) == 1, "token stop did not stop on first matching token")

    first_text = reloaded_backend.decode((first_token,))
    text_stop_verified = False
    if first_text:
        text_stop = generate(
            reloaded_backend,
            prompt,
            GenerationConfig(
                max_new_tokens=8,
                sample=False,
                seed=seed,
                stop_strings=(first_text,),
            ),
        )
        _require(text_stop.stop_reason == "stop_string", "real checkpoint text stop did not trigger")
        text_stop_verified = True

    context_prompt = "x" * reloaded_backend.max_context_tokens
    context_result = generate(
        reloaded_backend,
        context_prompt,
        GenerationConfig(max_new_tokens=1, sample=False, seed=seed),
    )
    _require(context_result.stop_reason == "context_limit", "context boundary did not stop generation")
    over_context_rejected = False
    try:
        generate(
            reloaded_backend,
            context_prompt + "x",
            GenerationConfig(max_new_tokens=1, sample=False, seed=seed),
        )
    except ValueError:
        over_context_rejected = True
    _require(over_context_rejected, "over-context prompt was not rejected fail-closed")

    strict_evidence_sha256 = _canonical_hash(strict_evidence)
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "authority": AUTHORITY,
        "repository": _REPOSITORY,
        "candidate_sha": candidate_sha,
        "stage": "S0",
        "base_semantics": {
            "random_init": True,
            "pretraining_only": True,
            "foreign_pretrained_weights_used": False,
            "instruction_or_alignment_behavior_added": False,
        },
        "strict_candidate_evidence_sha256": strict_evidence_sha256,
        "checkpoint": {
            "checkpoint_id": manifest["checkpoint_id"],
            "matches_strict_d04_final_checkpoint_id": True,
            "git_sha": verified["identity"]["git_sha"],
            "model_spec_sha256": verified["identity"]["model_spec_hash"],
            "init_spec_sha256": verified["identity"]["init_spec_hash"],
            "tokenizer_config_sha256": verified["identity"]["tokenizer_hash"],
            "tokenizer_vocab_sha256": verified["identity"]["tokenizer_vocab_hash"],
            "dataset_manifest_sha256": verified["identity"]["dataset_manifest_hash"],
            "run_manifest_sha256": verified["identity"]["run_manifest_hash"],
            "packing_sha256": PACKING_CONFIG_HASH,
            "packing_version": PACKING_VERSION,
            "step": verified["identity"]["step"],
            "tokens_seen": verified["identity"]["tokens_seen"],
        },
        "backend_diagnostics": diagnostics,
        "parity": parity.to_dict(),
        "generation": {
            "prompt_sha256": _text_hash(prompt),
            "greedy": _generation_record(greedy_reloaded, seed=seed, mode="greedy"),
            "seeded_sample_a": _generation_record(sample_a, seed=sample_seed, mode="sample"),
            "seeded_sample_b": _generation_record(sample_b, seed=sample_seed, mode="sample"),
            "seeded_sampling_repeatable": True,
            "token_stop_verified": True,
            "text_stop_verified": text_stop_verified,
            "context_limit_verified": True,
            "over_context_rejected": True,
        },
        "claims": {
            "local_free_cpu_evidence": True,
            "paid_compute_authorized_or_used": False,
            "promotion_authority": False,
            "candidate_or_stable_promotion": False,
            "live_windows_nvda_execution": False,
        },
    }
    payload["evidence_sha256"] = _canonical_hash(payload)
    validate_s0_generation_artifact(payload, checkpoint_path=checkpoint_path)
    return payload


def validate_s0_generation_artifact(
    evidence: Mapping[str, Any],
    *,
    checkpoint_path: str | Path | None = None,
) -> None:
    """Validate retained generation evidence and optionally the checkpoint bytes."""

    _require(evidence.get("schema_version") == SCHEMA_VERSION, "wrong generation artifact schema")
    _require(evidence.get("authority") == AUTHORITY, "wrong generation artifact authority")
    _require(evidence.get("repository") == _REPOSITORY, "repository identity mismatch")
    _require(_is_git_sha(evidence.get("candidate_sha")), "candidate SHA is invalid")
    _require(_is_sha256(evidence.get("strict_candidate_evidence_sha256")), "strict evidence hash is invalid")

    checkpoint = evidence.get("checkpoint")
    _require(isinstance(checkpoint, Mapping), "checkpoint evidence is missing")
    _require(_is_sha256(checkpoint.get("checkpoint_id")), "checkpoint_id is invalid")
    _require(checkpoint.get("matches_strict_d04_final_checkpoint_id") is True, "strict D04 checkpoint equivalence is not proven")
    _require(checkpoint.get("git_sha") == evidence.get("candidate_sha"), "checkpoint Git SHA mismatch")
    for key in (
        "model_spec_sha256",
        "init_spec_sha256",
        "tokenizer_config_sha256",
        "tokenizer_vocab_sha256",
        "dataset_manifest_sha256",
        "run_manifest_sha256",
        "packing_sha256",
    ):
        _require(_is_sha256(checkpoint.get(key)), f"checkpoint {key} is invalid")

    diagnostics = evidence.get("backend_diagnostics")
    _require(isinstance(diagnostics, Mapping), "backend diagnostics are missing")
    _require(diagnostics.get("checkpoint_id") == checkpoint.get("checkpoint_id"), "backend/checkpoint identity mismatch")
    _require(diagnostics.get("git_sha") == evidence.get("candidate_sha"), "backend Git SHA mismatch")
    _require(diagnostics.get("model_spec_sha256") == checkpoint.get("model_spec_sha256"), "backend ModelSpec mismatch")
    _require(diagnostics.get("tokenizer_config_sha256") == checkpoint.get("tokenizer_config_sha256"), "backend tokenizer config mismatch")
    _require(diagnostics.get("tokenizer_vocab_sha256") == checkpoint.get("tokenizer_vocab_sha256"), "backend tokenizer vocab mismatch")

    parity = evidence.get("parity")
    _require(isinstance(parity, Mapping), "parity evidence is missing")
    _require(parity.get("passed") is True, "parity did not pass")
    _require(parity.get("atol") == 0.0 and parity.get("rtol") == 0.0, "parity tolerances are not exact")
    _require(parity.get("max_abs_error") == 0.0, "parity has non-zero absolute error")
    _require(parity.get("max_rel_error") == 0.0, "parity has non-zero relative error")
    _require(parity.get("failures") == [], "parity contains failures")

    generation = evidence.get("generation")
    _require(isinstance(generation, Mapping), "generation evidence is missing")
    sample_a = generation.get("seeded_sample_a")
    sample_b = generation.get("seeded_sample_b")
    _require(isinstance(sample_a, Mapping) and isinstance(sample_b, Mapping), "seeded samples are missing")
    _require(sample_a == sample_b, "seeded sample records differ")
    for key in (
        "seeded_sampling_repeatable",
        "token_stop_verified",
        "context_limit_verified",
        "over_context_rejected",
    ):
        _require(generation.get(key) is True, f"{key} is not proven")

    claims = evidence.get("claims")
    _require(isinstance(claims, Mapping), "claims are missing")
    _require(claims.get("local_free_cpu_evidence") is True, "LOCAL_FREE boundary missing")
    _require(claims.get("paid_compute_authorized_or_used") is False, "paid-compute boundary violated")
    _require(claims.get("promotion_authority") is False, "artifact must not claim promotion authority")
    _require(claims.get("candidate_or_stable_promotion") is False, "artifact must not self-promote")

    recorded_hash = evidence.get("evidence_sha256")
    _require(_is_sha256(recorded_hash), "evidence_sha256 is invalid")
    unsigned = dict(evidence)
    unsigned.pop("evidence_sha256", None)
    _require(_canonical_hash(unsigned) == recorded_hash, "generation evidence hash mismatch")

    if checkpoint_path is not None:
        verified = verify_checkpoint(Path(checkpoint_path))
        _require(verified["checkpoint_id"] == checkpoint.get("checkpoint_id"), "checkpoint bytes do not match evidence checkpoint_id")
        _require(verified["identity"]["git_sha"] == evidence.get("candidate_sha"), "checkpoint bytes do not match evidence Git SHA")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise S0GenerationArtifactError("evidence JSON must contain an object")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m twelve_six.inference.s0_artifact",
        description="Build or validate a retained real-S0 checkpoint generation artifact.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build")
    build.add_argument("--repo-root", type=Path, default=Path("."))
    build.add_argument("--candidate-sha", required=True)
    build.add_argument("--checkpoint-out", type=Path, required=True)
    build.add_argument("--evidence-out", type=Path, required=True)
    build.add_argument("--train-steps", type=int, default=40)
    build.add_argument("--seed", type=int, default=20260824)

    validate = sub.add_parser("validate")
    validate.add_argument("--checkpoint", type=Path, required=True)
    validate.add_argument("--evidence", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "build":
        evidence = build_s0_generation_artifact(
            args.repo_root,
            candidate_sha=args.candidate_sha,
            checkpoint_out=args.checkpoint_out,
            train_steps=args.train_steps,
            seed=args.seed,
        )
        _write_json(args.evidence_out, evidence)
        print(json.dumps({
            "checkpoint_id": evidence["checkpoint"]["checkpoint_id"],
            "evidence_sha256": evidence["evidence_sha256"],
            "parity_passed": evidence["parity"]["passed"],
        }, sort_keys=True))
        return 0

    evidence = _load_json(args.evidence)
    validate_s0_generation_artifact(evidence, checkpoint_path=args.checkpoint)
    print(json.dumps({
        "checkpoint_id": evidence["checkpoint"]["checkpoint_id"],
        "evidence_sha256": evidence["evidence_sha256"],
        "validated": True,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
