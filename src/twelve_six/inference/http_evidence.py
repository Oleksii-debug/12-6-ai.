"""Exact LOCAL_FREE trained-checkpoint HTTP inference evidence for canonical S0 Base."""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import subprocess
import threading
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
    _evaluate,
    _tensor_batches,
)

from .first_party import load_first_party_backend
from .openai_compat import completion_response
from .parity import compare_backends
from .server import make_server

SCHEMA_VERSION = "12-6.s0-trained-http-inference-evidence.v1"
AUTHORITY = "LOCAL_FREE_OR_FREE_HOSTED_CPU_EVIDENCE_NOT_PROMOTION"
REPOSITORY = "Oleksii-debug/12-6-ai."


def _canonical_hash(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _git_head(repo_root: Path) -> str:
    try:
        value = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("trained HTTP evidence requires a Git checkout") from exc
    return value


def _require_source_sha(source_sha: str, *, actual_head: str) -> None:
    if (
        len(source_sha) != 40
        or source_sha != source_sha.lower()
        or any(ch not in "0123456789abcdef" for ch in source_sha)
    ):
        raise ValueError("source_sha must be a full lowercase 40-hex Git SHA")
    if source_sha != actual_head:
        raise ValueError("source_sha is stale: it does not equal checkout HEAD")


def _run_manifest(
    *,
    source_sha: str,
    stage: Any,
    tokenizer: ByteTokenizer,
    trainer_config: TrainerConfig,
    environment_lock_sha256: str,
    train_steps: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_id": f"s0-d05-http-evidence-{source_sha[:12]}",
        "stage": "S0",
        "run_kind": "trained_first_party_http_inference_evidence",
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
            "target_steps": train_steps,
        },
        "environment": {"lock_sha256": environment_lock_sha256},
    }


def _request_json(
    host: str,
    port: int,
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
) -> tuple[int, dict[str, Any]]:
    body: bytes | None = None
    headers: dict[str, str] = {}
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        headers = {"Content-Type": "application/json", "Content-Length": str(len(body))}
    connection = http.client.HTTPConnection(host, port, timeout=10)
    try:
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        raw = response.read()
        decoded = json.loads(raw.decode("utf-8"))
        if not isinstance(decoded, dict):
            raise RuntimeError("HTTP evidence response must be a JSON object")
        return response.status, decoded
    finally:
        connection.close()


def _completion_semantics(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "object": payload.get("object"),
        "model": payload.get("model"),
        "choices": payload.get("choices"),
        "usage": payload.get("usage"),
    }


def _choice_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise RuntimeError("completion response must contain exactly one choice")
    text = choices[0].get("text")
    if not isinstance(text, str):
        raise RuntimeError("completion choice text must be a string")
    return text


def _error_type(payload: dict[str, Any]) -> str | None:
    error = payload.get("error")
    if not isinstance(error, dict):
        return None
    value = error.get("type")
    return value if isinstance(value, str) else None


def collect_trained_http_inference_evidence(
    repo_root: str | Path,
    *,
    source_sha: str,
    output_dir: str | Path,
    train_steps: int = 40,
    seed: int = 1337,
    max_tokens: int = 8,
) -> dict[str, Any]:
    """Train S0, publish a strict D05 checkpoint, reload, and probe real loopback HTTP."""
    root = Path(repo_root).resolve()
    output = Path(output_dir).resolve()
    _require_source_sha(source_sha, actual_head=_git_head(root))
    if train_steps < 1:
        raise ValueError("train_steps must be >= 1")
    if max_tokens < 1:
        raise ValueError("max_tokens must be >= 1")
    if seed < 0:
        raise ValueError("seed must be non-negative")

    manifest_path = root / "data/s0/packaged/manifest.json"
    train_path = root / "data/s0/packaged/train.jsonl"
    validation_path = root / "data/s0/packaged/validation.jsonl"
    environment_lock_path = root / "requirements/locks/index.json"
    if sha256_file(manifest_path) != DATASET_MANIFEST_SHA256:
        raise RuntimeError("D03 dataset manifest SHA-256 mismatch")
    if sha256_file(train_path) != TRAIN_JSONL_SHA256:
        raise RuntimeError("D03 train split SHA-256 mismatch")
    if sha256_file(validation_path) != VALIDATION_JSONL_SHA256:
        raise RuntimeError("D03 validation split SHA-256 mismatch")
    dataset_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if dataset_manifest.get("dataset_identity_sha256") != DATASET_IDENTITY_SHA256:
        raise RuntimeError("D03 dataset semantic identity mismatch")

    stage = load_stage_config(root / "configs/stages/s0_10k.json")
    tokenizer = ByteTokenizer()
    if stage.canonical_base != "random_init":
        raise RuntimeError("canonical S0 Base must remain random_init")
    if stage.expected_parameters != 10_140:
        raise RuntimeError("unexpected S0 parameter count")
    if stage.model.vocab_size != tokenizer.vocab_size:
        raise RuntimeError("S0 model/tokenizer vocabulary mismatch")

    train_batches, train_ids, _ = _tensor_batches(
        root, split="train", tokenizer=tokenizer, batch_size=3
    )
    validation_batches, validation_ids, _ = _tensor_batches(
        root, split="validation", tokenizer=tokenizer, batch_size=3
    )
    if set(train_ids) & set(validation_ids):
        raise RuntimeError("train/validation record identity overlap")

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
        deterministic_warn_only=False,
    )
    torch.manual_seed(seed)
    model = TwelveSixDecoder(stage.model, stage.init)
    train_loss_before, _ = _evaluate(model, train_batches)
    validation_loss_before, _ = _evaluate(model, validation_batches)
    trainer = Trainer(model, trainer_config, device="cpu")
    run_result = trainer.run(islice(cycle(train_batches), train_steps))
    trainer.assert_checkpoint_safe()
    train_loss_after, _ = _evaluate(model, train_batches)
    validation_step_before = trainer.optimizer_step
    validation_loss_after, _ = _evaluate(model, validation_batches)
    validation_step_after = trainer.optimizer_step
    if validation_step_before != validation_step_after:
        raise RuntimeError("held-out evaluation mutated optimizer step")
    if not train_loss_after < train_loss_before:
        raise RuntimeError("real S0 training did not reduce train loss")

    environment_lock_sha256 = sha256_file(environment_lock_path)
    run_manifest = _run_manifest(
        source_sha=source_sha,
        stage=stage,
        tokenizer=tokenizer,
        trainer_config=trainer_config,
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

    output.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output / "trained-checkpoint"
    checkpoint_manifest = save_trainer_checkpoint(
        checkpoint_dir,
        model=model,
        trainer=trainer,
        identity=identity,
    )
    verified_manifest = verify_checkpoint(checkpoint_dir)
    if verified_manifest != checkpoint_manifest:
        raise RuntimeError("published checkpoint manifest changed after verification")

    backend = load_first_party_backend(checkpoint_dir)
    direct_backend = S0TorchInferenceBackend(model, tokenizer)
    parity = compare_backends(
        direct_backend,
        backend,
        ("S0 parity", "Привіт S0"),
        max_new_tokens=min(max_tokens, 8),
        atol=0.0,
        rtol=0.0,
    )
    if not parity.passed:
        raise RuntimeError(f"direct/reloaded first-party inference parity failed: {parity.failures}")

    model_name = "12-6-base-s0"
    prompt = "S0 HTTP evidence:"
    prompt_sha256 = _text_hash(prompt)
    greedy_request = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "seed": 17,
    }
    direct_greedy = completion_response(
        backend,
        greedy_request,
        response_id="cmpl-direct",
        created=0,
        model_name=model_name,
    )

    server = make_server(backend, host="127.0.0.1", port=0, model_name=model_name)
    actual_port = int(server.server_address[1])
    thread = threading.Thread(
        target=server.serve_forever,
        kwargs={"poll_interval": 0.01},
        daemon=True,
    )
    thread.start()
    try:
        health_status, health = _request_json("127.0.0.1", actual_port, "GET", "/healthz")
        models_status, models = _request_json("127.0.0.1", actual_port, "GET", "/v1/models")
        greedy_status, greedy_http = _request_json(
            "127.0.0.1", actual_port, "POST", "/v1/completions", greedy_request
        )
        greedy_matches_direct = _completion_semantics(greedy_http) == _completion_semantics(
            direct_greedy
        )
        if greedy_status != 200 or not greedy_matches_direct:
            raise RuntimeError("HTTP greedy completion does not match direct canonical completion")

        sampling_request = {
            "model": model_name,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.8,
            "top_p": 0.95,
            "seed": 424242,
        }
        sample_status_a, sample_a = _request_json(
            "127.0.0.1", actual_port, "POST", "/v1/completions", sampling_request
        )
        sample_status_b, sample_b = _request_json(
            "127.0.0.1", actual_port, "POST", "/v1/completions", sampling_request
        )
        sampling_repeatable = _completion_semantics(sample_a) == _completion_semantics(sample_b)
        if sample_status_a != 200 or sample_status_b != 200 or not sampling_repeatable:
            raise RuntimeError("HTTP seeded sampling is not repeatable")

        stop_text = _choice_text(greedy_http)
        if not stop_text:
            raise RuntimeError("greedy completion produced empty text; cannot prove stop-string path")
        stop_request = dict(greedy_request)
        stop_request["stop"] = [stop_text]
        stop_status, stopped = _request_json(
            "127.0.0.1", actual_port, "POST", "/v1/completions", stop_request
        )
        stopped_choice = stopped.get("choices", [{}])[0]
        stop_semantics = bool(
            stop_status == 200
            and isinstance(stopped_choice, dict)
            and stopped_choice.get("finish_reason") == "stop"
            and stopped_choice.get("text") == ""
        )
        if not stop_semantics:
            raise RuntimeError("HTTP stop-string semantics failed")

        exact_context_request = {
            "model": model_name,
            "prompt": "x" * backend.max_context_tokens,
            "max_tokens": 1,
            "temperature": 0,
        }
        context_status, context_payload = _request_json(
            "127.0.0.1", actual_port, "POST", "/v1/completions", exact_context_request
        )
        context_usage = context_payload.get("usage")
        context_choice = context_payload.get("choices", [{}])[0]
        exact_context_semantics = bool(
            context_status == 200
            and isinstance(context_usage, dict)
            and context_usage.get("prompt_tokens") == backend.max_context_tokens
            and context_usage.get("completion_tokens") == 0
            and isinstance(context_choice, dict)
            and context_choice.get("finish_reason") == "length"
        )
        if not exact_context_semantics:
            raise RuntimeError("HTTP exact-context limit semantics failed")

        over_context_request = dict(exact_context_request)
        over_context_request["prompt"] = "x" * (backend.max_context_tokens + 1)
        over_status, over_payload = _request_json(
            "127.0.0.1", actual_port, "POST", "/v1/completions", over_context_request
        )
        over_context_rejected = over_status == 400 and _error_type(over_payload) == "invalid_request_error"
        if not over_context_rejected:
            raise RuntimeError("HTTP over-context prompt did not fail closed")

        messages_status, messages_payload = _request_json(
            "127.0.0.1",
            actual_port,
            "POST",
            "/v1/completions",
            {"model": model_name, "messages": [{"role": "user", "content": "hello"}]},
        )
        messages_rejected = (
            messages_status == 400 and _error_type(messages_payload) == "invalid_request_error"
        )
        chat_status, chat_payload = _request_json(
            "127.0.0.1",
            actual_port,
            "POST",
            "/v1/chat/completions",
            {"model": model_name, "prompt": prompt},
        )
        chat_rejected = chat_status == 404 and _error_type(chat_payload) == "invalid_request_error"
        if not messages_rejected or not chat_rejected:
            raise RuntimeError("raw Base HTTP boundary accepted chat/messages semantics")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        if thread.is_alive():
            raise RuntimeError("loopback HTTP evidence server did not terminate")

    diagnostics = backend.diagnostics()
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "authority": AUTHORITY,
        "candidate": {"repository": REPOSITORY, "source_sha": source_sha, "stage": "S0"},
        "identities": {
            "modelspec_sha256": stage.model.identity_sha256(),
            "initspec_sha256": stage.init.identity_sha256(),
            "parameter_count": stage.expected_parameters,
            "tokenizer_version": tokenizer.identity.version,
            "tokenizer_config_sha256": tokenizer.identity.config_sha256,
            "tokenizer_vocab_sha256": tokenizer.identity.vocab_sha256,
            "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
            "dataset_identity_sha256": DATASET_IDENTITY_SHA256,
            "train_jsonl_sha256": TRAIN_JSONL_SHA256,
            "validation_jsonl_sha256": VALIDATION_JSONL_SHA256,
            "packing_config_sha256": PACKING_CONFIG_HASH,
            "environment_lock_sha256": environment_lock_sha256,
            "run_manifest_sha256": hash_json(run_manifest),
        },
        "training": {
            "seed": seed,
            "optimizer_steps": run_result.optimizer_steps_completed,
            "optimized_tokens": run_result.tokens_consumed,
            "trainer_tokens_seen": trainer.tokens_seen,
            "validation_optimized_tokens": 0,
            "train_loss_before": train_loss_before,
            "train_loss_after": train_loss_after,
            "validation_loss_before": validation_loss_before,
            "validation_loss_after": validation_loss_after,
        },
        "checkpoint": {
            "checkpoint_id": checkpoint_manifest["checkpoint_id"],
            "git_sha": checkpoint_manifest["identity"]["git_sha"],
            "step": checkpoint_manifest["identity"]["step"],
            "tokens_seen": checkpoint_manifest["identity"]["tokens_seen"],
            "files": checkpoint_manifest["files"],
            "pickle": checkpoint_manifest["serialization"]["pickle"],
        },
        "backend": diagnostics,
        "parity": parity.to_dict(),
        "http": {
            "transport": "real_loopback_tcp_http11",
            "host": "127.0.0.1",
            "port_mode": "ephemeral",
            "health": health_status == 200 and health.get("status") == "ok",
            "models": models_status == 200 and models.get("object") == "list",
            "greedy": {
                "status": greedy_status,
                "prompt_utf8_sha256": prompt_sha256,
                "matches_direct_completion": greedy_matches_direct,
                "text_utf8_sha256": _text_hash(_choice_text(greedy_http)),
                "finish_reason": greedy_http["choices"][0]["finish_reason"],
                "usage": greedy_http["usage"],
            },
            "seeded_sampling": {
                "status_a": sample_status_a,
                "status_b": sample_status_b,
                "seed": 424242,
                "same_seed_repeatable": sampling_repeatable,
                "text_utf8_sha256": _text_hash(_choice_text(sample_a)),
            },
            "stop_string": {"status": stop_status, "stop_and_strip_verified": stop_semantics},
            "context": {
                "max_context_tokens": backend.max_context_tokens,
                "exact_limit_verified": exact_context_semantics,
                "over_limit_rejected": over_context_rejected,
            },
            "raw_base_boundary": {
                "messages_rejected": messages_rejected,
                "chat_endpoint_rejected": chat_rejected,
                "hidden_system_or_instruction_template": False,
            },
        },
        "claims": {
            "foreign_pretrained_weights_used": False,
            "instruction_or_alignment_behavior_added": False,
            "paid_compute_authorized_or_used": False,
            "candidate_or_stable_promotion": False,
            "audit_verdict_claimed": False,
            "windows_nvda_live_runtime_claimed": False,
        },
    }
    payload["evidence_sha256"] = _canonical_hash(payload)
    evidence_path = output / "s0-trained-http-inference-evidence.json"
    evidence_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return payload


def validate_trained_http_inference_evidence(
    payload: dict[str, Any], *, expected_source_sha: str | None = None
) -> None:
    """Fail closed on identity drift, self-hash tamper, or a missing proof bit."""
    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("authority") != AUTHORITY:
        raise ValueError("trained HTTP inference evidence schema/authority mismatch")
    declared_hash = payload.get("evidence_sha256")
    if not isinstance(declared_hash, str):
        raise ValueError("trained HTTP inference evidence is missing evidence_sha256")
    unhashed = dict(payload)
    unhashed.pop("evidence_sha256", None)
    if _canonical_hash(unhashed) != declared_hash:
        raise ValueError("trained HTTP inference evidence self-hash mismatch")
    candidate = payload.get("candidate")
    if not isinstance(candidate, dict) or candidate.get("repository") != REPOSITORY:
        raise ValueError("trained HTTP inference evidence repository mismatch")
    source_sha = candidate.get("source_sha")
    if expected_source_sha is not None and source_sha != expected_source_sha:
        raise ValueError("trained HTTP inference evidence source SHA mismatch")
    training = payload.get("training")
    checkpoint = payload.get("checkpoint")
    parity = payload.get("parity")
    http = payload.get("http")
    claims = payload.get("claims")
    if not all(isinstance(item, dict) for item in (training, checkpoint, parity, http, claims)):
        raise ValueError("trained HTTP inference evidence required sections are missing")
    if training.get("validation_optimized_tokens") != 0:
        raise ValueError("validation split was optimized")
    if not float(training.get("train_loss_after", float("inf"))) < float(
        training.get("train_loss_before", float("-inf"))
    ):
        raise ValueError("training loss decrease is not proven")
    if checkpoint.get("git_sha") != source_sha or checkpoint.get("pickle") is not False:
        raise ValueError("checkpoint lineage/serialization boundary mismatch")
    if parity.get("passed") is not True:
        raise ValueError("direct/reloaded parity is not proven")
    required_http_truths = (
        http.get("health"),
        http.get("models"),
        http.get("greedy", {}).get("matches_direct_completion"),
        http.get("seeded_sampling", {}).get("same_seed_repeatable"),
        http.get("stop_string", {}).get("stop_and_strip_verified"),
        http.get("context", {}).get("exact_limit_verified"),
        http.get("context", {}).get("over_limit_rejected"),
        http.get("raw_base_boundary", {}).get("messages_rejected"),
        http.get("raw_base_boundary", {}).get("chat_endpoint_rejected"),
    )
    if any(value is not True for value in required_http_truths):
        raise ValueError("trained HTTP inference evidence has an unproven HTTP contract")
    if any(value is not False for value in claims.values()):
        raise ValueError("trained HTTP inference evidence truth boundary was weakened")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run real trained/reloaded S0 over the loopback /v1/completions transport."
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = collect_trained_http_inference_evidence(
        args.repo_root,
        source_sha=args.source_sha,
        output_dir=args.output_dir,
        train_steps=args.train_steps,
        seed=args.seed,
        max_tokens=args.max_tokens,
    )
    validate_trained_http_inference_evidence(payload, expected_source_sha=args.source_sha)
    if args.json:
        print(json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False))
    else:
        print(
            "trained-http-inference: PASS "
            f"source_sha={args.source_sha} "
            f"checkpoint_id={payload['checkpoint']['checkpoint_id']} "
            f"evidence_sha256={payload['evidence_sha256']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
