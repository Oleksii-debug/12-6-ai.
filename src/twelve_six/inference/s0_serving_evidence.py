"""Exact-source real S0 checkpoint -> first-party -> HTTP serving evidence.

This module composes accepted D01/D02/D03/D04/D05/D07 contracts. It does not
implement model architecture, checkpoint serialization, sampling, or HTTP
transport semantics itself.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import random
import subprocess
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from twelve_six.checkpoint import (
    bind_checkpoint_identity,
    hash_json,
    save_trainer_checkpoint,
    sha256_file,
)
from twelve_six.inference.first_party import load_first_party_backend
from twelve_six.inference.openai_compat import completion_response
from twelve_six.inference.server import make_server
from twelve_six.integration import S0TorchInferenceBackend
from twelve_six.model import TwelveSixDecoder, load_stage_config
from twelve_six.packing import PACKING_CONFIG_HASH, PACKING_VERSION
from twelve_six.tokenization import ByteTokenizer
from twelve_six.training import Trainer, TrainerConfig

SCHEMA_VERSION = "12-6.s0-serving-evidence.v1"
AUTHORITY = "LOCAL_FREE_OR_FREE_HOSTED_CPU_EVIDENCE_NOT_PROMOTION"
REPOSITORY = "Oleksii-debug/12-6-ai."
MODEL_NAME = "12-6-s0-base"
_HEX = frozenset("0123456789abcdef")


class S0ServingEvidenceError(ValueError):
    """Raised when serving evidence is incomplete, stale, or inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise S0ServingEvidenceError(message)


def _validate_git_sha(value: Any) -> str:
    _require(
        isinstance(value, str)
        and len(value) == 40
        and value == value.lower()
        and set(value) <= _HEX,
        "candidate SHA must be a full lowercase 40-hex Git SHA",
    )
    return value


def _git_head(root: Path) -> str:
    try:
        value = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise S0ServingEvidenceError(
            "serving evidence requires an exact Git checkout"
        ) from exc
    return _validate_git_sha(value)


def _canonical_hash(value: Mapping[str, Any]) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _text_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _read_train_batches(
    root: Path,
    tokenizer: ByteTokenizer,
    *,
    max_seq_len: int,
) -> list[dict[str, torch.Tensor]]:
    path = root / "data/s0/packaged/train.jsonl"
    batches: list[dict[str, torch.Tensor]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or not isinstance(row.get("text"), str):
            raise S0ServingEvidenceError(
                f"invalid train record at {path}:{line_number}"
            )
        token_ids = tokenizer.encode(row["text"])[:max_seq_len]
        if len(token_ids) < 2:
            raise S0ServingEvidenceError(
                f"train record at {path}:{line_number} has fewer than two tokens"
            )
        ids = torch.tensor([token_ids], dtype=torch.long)
        batches.append({"input_ids": ids, "labels": ids})
    _require(bool(batches), "committed S0 train split contains no usable records")
    return batches


def _model_state_hash(model: TwelveSixDecoder) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.numpy().tobytes(order="C"))
        digest.update(b"\0")
    return digest.hexdigest()


def _run_manifest(
    root: Path,
    *,
    candidate_sha: str,
    stage: Any,
    tokenizer: ByteTokenizer,
    trainer_config: TrainerConfig,
    train_steps: int,
) -> tuple[dict[str, Any], str]:
    environment_lock_hash = sha256_file(root / "requirements/locks/index.json")
    train_sha = sha256_file(root / "data/s0/packaged/train.jsonl")
    payload = {
        "schema_version": 1,
        "run_id": f"s0-d07-serving-{candidate_sha[:12]}",
        "stage": "S0",
        "run_kind": "real_checkpoint_http_serving_evidence",
        "state": "RUNNING",
        "candidate": {
            "repository": REPOSITORY,
            "git_sha": candidate_sha,
            "branch_or_tag": "exact-checkout",
            "modelspec_sha256": hash_json(stage.model.to_dict()),
            "initspec_sha256": hash_json(stage.init.to_dict()),
            "parameter_count": stage.expected_parameters,
        },
        "data": {
            "dataset_manifest_sha256": sha256_file(
                root / "data/s0/packaged/manifest.json"
            ),
            "tokenizer_sha256": tokenizer.identity.config_sha256,
            "tokenizer_vocab_sha256": tokenizer.identity.vocab_sha256,
            "tokenizer_version": tokenizer.identity.version,
            "split_identity": f"train:{train_sha}",
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
            "target_steps": train_steps,
        },
        "environment": {"lock_sha256": environment_lock_hash},
    }
    return payload, environment_lock_hash


def _http_json(
    host: str,
    port: int,
    *,
    method: str,
    path: str,
    payload: Mapping[str, Any] | None = None,
) -> tuple[int, dict[str, Any]]:
    connection = http.client.HTTPConnection(host, port, timeout=10)
    try:
        body = None
        headers: dict[str, str] = {}
        if payload is not None:
            body = json.dumps(
                payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
            headers = {
                "Content-Type": "application/json",
                "Content-Length": str(len(body)),
            }
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        raw = response.read()
        decoded = json.loads(raw.decode("utf-8"))
        if not isinstance(decoded, dict):
            raise S0ServingEvidenceError(f"{path} did not return a JSON object")
        return response.status, decoded
    finally:
        connection.close()


def _choice_summary(response: Mapping[str, Any]) -> dict[str, Any]:
    choices = response.get("choices")
    _require(
        isinstance(choices, list)
        and len(choices) == 1
        and isinstance(choices[0], Mapping),
        "completion response must contain exactly one choice",
    )
    choice = choices[0]
    text = choice.get("text")
    _require(isinstance(text, str), "completion choice text must be a string")
    usage = response.get("usage")
    _require(isinstance(usage, Mapping), "completion response usage is missing")
    return {
        "text_sha256": _text_hash(text),
        "finish_reason": choice.get("finish_reason"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
    }


def validate_serving_evidence(
    evidence: Mapping[str, Any],
    *,
    expected_candidate_sha: str | None = None,
) -> None:
    """Fail closed unless evidence proves one real trained checkpoint HTTP path."""

    _require(evidence.get("schema_version") == SCHEMA_VERSION, "wrong serving schema")
    _require(evidence.get("authority") == AUTHORITY, "wrong serving evidence authority")
    candidate = evidence.get("candidate")
    _require(isinstance(candidate, Mapping), "candidate evidence is missing")
    candidate_sha = _validate_git_sha(candidate.get("sha"))
    _require(candidate.get("repository") == REPOSITORY, "repository identity mismatch")
    if expected_candidate_sha is not None:
        _require(
            candidate_sha == _validate_git_sha(expected_candidate_sha),
            "serving evidence candidate SHA is stale",
        )

    training = evidence.get("training")
    _require(isinstance(training, Mapping), "training evidence is missing")
    _require(
        isinstance(training.get("optimizer_steps"), int)
        and not isinstance(training.get("optimizer_steps"), bool)
        and training["optimizer_steps"] > 0,
        "serving checkpoint must come from a real optimizer run",
    )
    _require(
        isinstance(training.get("tokens_seen"), int)
        and not isinstance(training.get("tokens_seen"), bool)
        and training["tokens_seen"] > 0,
        "serving checkpoint must record optimized tokens",
    )
    _require(
        training.get("initial_model_state_sha256")
        != training.get("trained_model_state_sha256"),
        "training did not change model state",
    )

    checkpoint = evidence.get("checkpoint")
    _require(isinstance(checkpoint, Mapping), "checkpoint evidence is missing")
    _require(
        checkpoint.get("git_sha") == candidate_sha,
        "checkpoint is not bound to exact candidate SHA",
    )
    _require(checkpoint.get("reload_verified") is True, "checkpoint reload not verified")
    _require(
        checkpoint.get("direct_reloaded_logits_exact") is True,
        "direct/reloaded logits parity failed",
    )

    http = evidence.get("http")
    _require(isinstance(http, Mapping), "HTTP evidence is missing")
    _require(http.get("health_status") == 200, "health endpoint did not succeed")
    _require(http.get("models_status") == 200, "models endpoint did not succeed")
    _require(http.get("greedy_status") == 200, "greedy completion did not succeed")
    _require(http.get("sample_status_a") == 200, "sample completion A did not succeed")
    _require(http.get("sample_status_b") == 200, "sample completion B did not succeed")
    _require(
        http.get("context_overflow_status") == 400,
        "over-context request did not fail closed",
    )
    _require(http.get("chat_status") == 404, "chat endpoint did not remain unsupported")

    parity = evidence.get("parity")
    _require(isinstance(parity, Mapping), "serving parity evidence is missing")
    _require(parity.get("greedy_direct_vs_http") is True, "greedy HTTP parity failed")
    _require(parity.get("sample_direct_vs_http") is True, "sampling HTTP parity failed")
    _require(parity.get("sample_http_repeatable") is True, "seeded HTTP sampling drifted")

    claims = evidence.get("claims")
    _require(isinstance(claims, Mapping), "truth-boundary claims are missing")
    for key in (
        "candidate_or_stable_promotion",
        "audit_pass",
        "paid_compute_authorized_or_used",
        "foreign_pretrained_weights_used",
        "instruction_or_alignment_training",
        "chat_or_hidden_prompt_semantics",
    ):
        _require(claims.get(key) is False, f"prohibited claim is true: {key}")

    expected_hash = evidence.get("evidence_sha256")
    _require(isinstance(expected_hash, str) and len(expected_hash) == 64, "evidence hash missing")
    unhashed = dict(evidence)
    unhashed.pop("evidence_sha256", None)
    _require(_canonical_hash(unhashed) == expected_hash, "serving evidence hash mismatch")


def collect_serving_evidence(
    root: str | Path,
    *,
    candidate_sha: str,
    output_dir: str | Path,
    train_steps: int = 40,
    seed: int = 20260825,
    verify_checkout: bool = True,
) -> dict[str, Any]:
    """Train real S0, checkpoint/reload it, and exercise a real loopback HTTP server."""

    root = Path(root).resolve()
    output_dir = Path(output_dir).resolve()
    candidate_sha = _validate_git_sha(candidate_sha)
    if verify_checkout:
        _require(_git_head(root) == candidate_sha, "candidate SHA does not equal checkout HEAD")
    _require(
        isinstance(train_steps, int)
        and not isinstance(train_steps, bool)
        and train_steps > 0,
        "train_steps must be a positive integer",
    )
    _require(
        isinstance(seed, int) and not isinstance(seed, bool) and 0 <= seed < 2**63,
        "seed must be in [0, 2**63)",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoint"
    _require(
        not checkpoint_dir.exists() and not checkpoint_dir.is_symlink(),
        "serving evidence checkpoint destination already exists",
    )

    stage = load_stage_config(root / "configs/stages/s0_10k.json")
    tokenizer = ByteTokenizer()
    _require(stage.canonical_base == "random_init", "S0 Base is not random initialized")
    _require(stage.model.vocab_size == tokenizer.vocab_size, "model/tokenizer vocab mismatch")
    batches = _read_train_batches(root, tokenizer, max_seq_len=stage.model.max_seq_len)

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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = TwelveSixDecoder(stage.model, stage.init)
    initial_state_hash = _model_state_hash(model)
    trainer = Trainer(model, trainer_config, device="cpu")
    for step in range(train_steps):
        metrics = trainer.train_microbatch(batches[step % len(batches)])
        _require(
            metrics.optimizer_stepped and np.isfinite(metrics.loss),
            "real S0 training failed to complete a finite optimizer step",
        )
    trainer.assert_checkpoint_safe()
    trained_state_hash = _model_state_hash(model)
    _require(initial_state_hash != trained_state_hash, "optimizer run did not change model state")

    run_manifest, environment_lock_hash = _run_manifest(
        root,
        candidate_sha=candidate_sha,
        stage=stage,
        tokenizer=tokenizer,
        trainer_config=trainer_config,
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
        environment_lock_hash=environment_lock_hash,
    )
    manifest = save_trainer_checkpoint(
        checkpoint_dir,
        model=model,
        trainer=trainer,
        identity=identity,
    )
    backend = load_first_party_backend(checkpoint_dir)
    diagnostics = backend.diagnostics()
    _require(diagnostics.get("git_sha") == candidate_sha, "reloaded checkpoint SHA drift")
    _require(
        diagnostics.get("checkpoint_id") == manifest.get("checkpoint_id"),
        "reloaded checkpoint identity drift",
    )

    direct_backend = S0TorchInferenceBackend(model, tokenizer)
    probe_prompt = "12-6 Base serving probe: "
    prompt_ids = tokenizer.encode(probe_prompt)
    direct_logits = list(direct_backend.next_token_logits(prompt_ids))
    reloaded_logits = list(backend.next_token_logits(prompt_ids))
    logits_exact = direct_logits == reloaded_logits
    _require(logits_exact, "trained model and first-party reload logits differ")

    greedy_payload = {
        "model": MODEL_NAME,
        "prompt": probe_prompt,
        "max_tokens": 8,
        "temperature": 0,
        "seed": 17,
    }
    sample_payload = {
        "model": MODEL_NAME,
        "prompt": probe_prompt,
        "max_tokens": 8,
        "temperature": 0.8,
        "top_p": 0.95,
        "seed": 1729,
    }
    direct_greedy = completion_response(
        backend,
        greedy_payload,
        response_id="direct-greedy",
        created=0,
        model_name=MODEL_NAME,
    )
    direct_sample = completion_response(
        backend,
        sample_payload,
        response_id="direct-sample",
        created=0,
        model_name=MODEL_NAME,
    )
    direct_greedy_summary = _choice_summary(direct_greedy)
    direct_sample_summary = _choice_summary(direct_sample)

    server = make_server(
        backend,
        host="127.0.0.1",
        port=0,
        model_name=MODEL_NAME,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address[:2]
        host = str(host)
        port = int(port)
        health_status, health = _http_json(host, port, method="GET", path="/healthz")
        models_status, models = _http_json(host, port, method="GET", path="/v1/models")
        greedy_status, greedy = _http_json(
            host, port, method="POST", path="/v1/completions", payload=greedy_payload
        )
        sample_status_a, sample_a = _http_json(
            host, port, method="POST", path="/v1/completions", payload=sample_payload
        )
        sample_status_b, sample_b = _http_json(
            host, port, method="POST", path="/v1/completions", payload=sample_payload
        )
        context_status, context_error = _http_json(
            host,
            port,
            method="POST",
            path="/v1/completions",
            payload={
                "model": MODEL_NAME,
                "prompt": "x" * (stage.model.max_seq_len + 1),
                "max_tokens": 1,
                "temperature": 0,
            },
        )
        chat_status, chat_error = _http_json(
            host,
            port,
            method="POST",
            path="/v1/chat/completions",
            payload={"model": MODEL_NAME, "messages": [{"role": "user", "content": "x"}]},
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)
    _require(not thread.is_alive(), "HTTP server thread did not shut down cleanly")

    greedy_summary = _choice_summary(greedy)
    sample_summary_a = _choice_summary(sample_a)
    sample_summary_b = _choice_summary(sample_b)
    _require(health.get("model") == MODEL_NAME, "health model identity drift")
    _require(
        isinstance(models.get("data"), list)
        and len(models["data"]) == 1
        and isinstance(models["data"][0], Mapping)
        and models["data"][0].get("id") == MODEL_NAME,
        "models endpoint identity drift",
    )
    _require(isinstance(context_error.get("error"), Mapping), "context failure lacked JSON error")
    _require(isinstance(chat_error.get("error"), Mapping), "chat failure lacked JSON error")

    evidence: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "authority": AUTHORITY,
        "candidate": {
            "repository": REPOSITORY,
            "sha": candidate_sha,
            "canonical_base": "random_init_pretraining_only",
        },
        "training": {
            "optimizer_steps": trainer.optimizer_step,
            "tokens_seen": trainer.tokens_seen,
            "seed": seed,
            "initial_model_state_sha256": initial_state_hash,
            "trained_model_state_sha256": trained_state_hash,
        },
        "checkpoint": {
            "checkpoint_id": manifest["checkpoint_id"],
            "git_sha": manifest["identity"]["git_sha"],
            "model_spec_sha256": manifest["identity"]["model_spec_hash"],
            "tokenizer_config_sha256": manifest["identity"]["tokenizer_hash"],
            "tokenizer_vocab_sha256": manifest["identity"]["tokenizer_vocab_hash"],
            "dataset_manifest_sha256": manifest["identity"]["dataset_manifest_hash"],
            "run_manifest_sha256": manifest["identity"]["run_manifest_hash"],
            "environment_lock_sha256": manifest["identity"]["environment_lock_hash"],
            "step": manifest["identity"]["step"],
            "tokens_seen": manifest["identity"]["tokens_seen"],
            "reload_verified": True,
            "direct_reloaded_logits_exact": logits_exact,
        },
        "probe": {
            "prompt_sha256": _text_hash(probe_prompt),
            "prompt_tokens": len(prompt_ids),
            "greedy_direct": direct_greedy_summary,
            "greedy_http": greedy_summary,
            "sample_direct": direct_sample_summary,
            "sample_http_a": sample_summary_a,
            "sample_http_b": sample_summary_b,
        },
        "http": {
            "host": "127.0.0.1",
            "externally_exposed": False,
            "health_status": health_status,
            "models_status": models_status,
            "greedy_status": greedy_status,
            "sample_status_a": sample_status_a,
            "sample_status_b": sample_status_b,
            "context_overflow_status": context_status,
            "chat_status": chat_status,
        },
        "parity": {
            "greedy_direct_vs_http": direct_greedy_summary == greedy_summary,
            "sample_direct_vs_http": direct_sample_summary == sample_summary_a,
            "sample_http_repeatable": sample_summary_a == sample_summary_b,
        },
        "claims": {
            "candidate_or_stable_promotion": False,
            "audit_pass": False,
            "paid_compute_authorized_or_used": False,
            "foreign_pretrained_weights_used": False,
            "instruction_or_alignment_training": False,
            "chat_or_hidden_prompt_semantics": False,
        },
    }
    evidence["evidence_sha256"] = _canonical_hash(evidence)
    validate_serving_evidence(evidence, expected_candidate_sha=candidate_sha)
    (output_dir / "serving_evidence.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return evidence


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run exact-source real S0 checkpoint -> loopback HTTP serving evidence."
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--candidate-sha", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260825)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    evidence = collect_serving_evidence(
        args.repo_root,
        candidate_sha=args.candidate_sha,
        output_dir=args.output_dir,
        train_steps=args.train_steps,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "candidate_sha": evidence["candidate"]["sha"],
                "checkpoint_id": evidence["checkpoint"]["checkpoint_id"],
                "optimizer_steps": evidence["training"]["optimizer_steps"],
                "tokens_seen": evidence["training"]["tokens_seen"],
                "greedy_http_parity": evidence["parity"]["greedy_direct_vs_http"],
                "sample_http_parity": evidence["parity"]["sample_direct_vs_http"],
                "evidence_sha256": evidence["evidence_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
