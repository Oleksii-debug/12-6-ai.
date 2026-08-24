from __future__ import annotations

import argparse
import json
import random
import subprocess
import threading
from http.client import HTTPConnection
from pathlib import Path
from typing import Any

import torch

from twelve_six.checkpoint import (
    bind_checkpoint_identity,
    hash_json,
    save_trainer_checkpoint,
    sha256_file,
)
from twelve_six.inference.first_party import load_first_party_backend
from twelve_six.inference.openai_compat import completion_response
from twelve_six.inference.parity import compare_backends
from twelve_six.inference.sampling import greedy_token
from twelve_six.inference.server import make_server
from twelve_six.integration import S0TorchInferenceBackend
from twelve_six.model import TwelveSixDecoder, load_stage_config
from twelve_six.packing import PACKING_CONFIG_HASH, PACKING_VERSION
from twelve_six.tokenization import ByteTokenizer
from twelve_six.training import Trainer, TrainerConfig

SCHEMA = "12-6.s0-http-parity-evidence.v1"
REPOSITORY = "Oleksii-debug/12-6-ai."
MODEL_NAME = "12-6-base-s0"


def _is_exact_git_sha(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) in {40, 64}
        and value == value.lower()
        and all(ch in "0123456789abcdef" for ch in value)
    )


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
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
        raise RuntimeError("HTTP parity evidence requires a Git checkout") from exc
    if not _is_exact_git_sha(value):
        raise ValueError("git HEAD is not an exact lowercase Git object id")
    return value


def _load_train_texts(path: Path) -> list[str]:
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
        raise ValueError("S0 training split contains no text")
    return texts


def _normalized_completion(payload: dict[str, object]) -> dict[str, object]:
    result = dict(payload)
    result.pop("id", None)
    result.pop("created", None)
    return result


def _request_json(
    address: tuple[str, int],
    method: str,
    path: str,
    payload: dict[str, object] | None = None,
) -> tuple[int, dict[str, object]]:
    connection = HTTPConnection(*address, timeout=5)
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {} if body is None else {"Content-Type": "application/json"}
    connection.request(method, path, body=body, headers=headers)
    response = connection.getresponse()
    raw = response.read().decode("utf-8")
    connection.close()
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise TypeError("HTTP response JSON must be an object")
    return response.status, parsed


def _evidence_hash(payload: dict[str, Any]) -> str:
    material = dict(payload)
    material.pop("evidence_sha256", None)
    return hash_json(material)


def validate_http_parity_evidence(
    payload: dict[str, Any],
    *,
    expected_candidate_sha: str | None = None,
) -> None:
    if payload.get("schema") != SCHEMA:
        raise ValueError("unexpected HTTP parity evidence schema")
    candidate = payload.get("candidate")
    checkpoint = payload.get("checkpoint")
    training = payload.get("training")
    parity = payload.get("parity")
    http = payload.get("http")
    semantics = payload.get("raw_base_semantics")
    sections = (candidate, checkpoint, training, parity, http, semantics)
    if not all(isinstance(item, dict) for item in sections):
        raise TypeError("HTTP parity evidence sections must be JSON objects")

    candidate_sha = candidate.get("sha")
    if not _is_exact_git_sha(candidate_sha):
        raise ValueError("candidate SHA must be an exact lowercase Git object id")
    if expected_candidate_sha is not None and candidate_sha != expected_candidate_sha:
        raise ValueError("HTTP parity evidence candidate SHA is stale")
    if candidate.get("repository") != REPOSITORY:
        raise ValueError("HTTP parity evidence repository identity mismatch")
    if candidate.get("random_init_pretraining_only") is not True:
        raise ValueError("canonical Base identity is not random-init pretraining-only")

    if not _is_sha256(checkpoint.get("checkpoint_id")):
        raise ValueError("checkpoint_id must be SHA-256")
    optimizer_steps = training.get("optimizer_steps")
    tokens_seen = training.get("tokens_seen")
    if (
        not isinstance(optimizer_steps, int)
        or isinstance(optimizer_steps, bool)
        or optimizer_steps <= 0
        or not isinstance(tokens_seen, int)
        or isinstance(tokens_seen, bool)
        or tokens_seen <= 0
    ):
        raise ValueError("HTTP parity evidence requires a real optimizer update and tokens")
    if training.get("paid_compute") is not False:
        raise ValueError("HTTP parity evidence must remain LOCAL_FREE")

    if parity.get("passed") is not True:
        raise ValueError("direct-vs-reloaded inference parity did not pass")
    if parity.get("max_abs_error") != 0.0 or parity.get("max_rel_error") != 0.0:
        raise ValueError("S0 direct-vs-reloaded parity must be exact")

    required_http = (
        "health_ok",
        "model_list_ok",
        "greedy_matches_direct",
        "sampled_matches_direct",
        "seeded_sampling_repeatable",
        "stop_matches_direct",
        "context_limit_matches_direct",
        "over_context_rejected",
        "chat_rejected",
    )
    if any(http.get(key) is not True for key in required_http):
        raise ValueError("one or more required HTTP parity checks did not pass")

    if semantics.get("hidden_prompt") is not False:
        raise ValueError("raw Base HTTP evidence must not inject a hidden prompt")
    if semantics.get("chat_roles") is not False:
        raise ValueError("raw Base HTTP evidence must not claim chat-role semantics")
    if semantics.get("instruction_template") is not False:
        raise ValueError("raw Base HTTP evidence must not add an instruction template")
    if semantics.get("alignment_behavior") is not False:
        raise ValueError("raw Base HTTP evidence must not add alignment behavior")

    expected_hash = _evidence_hash(payload)
    if payload.get("evidence_sha256") != expected_hash:
        raise ValueError("HTTP parity evidence self-hash mismatch")


def collect_s0_http_parity_evidence(
    repo_root: Path,
    candidate_sha: str,
    output_dir: Path,
    *,
    train_steps: int = 4,
    seed: int = 20260825,
    verify_checkout: bool = True,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    output_dir = output_dir.resolve()
    if not _is_exact_git_sha(candidate_sha):
        raise ValueError("candidate_sha must be a full lowercase Git object id")
    if verify_checkout and _git_head(repo_root) != candidate_sha:
        raise ValueError("candidate_sha does not equal checkout HEAD")
    if not isinstance(train_steps, int) or isinstance(train_steps, bool) or train_steps < 1:
        raise ValueError("train_steps must be a positive integer")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    stage = load_stage_config(repo_root / "configs/stages/s0_10k.json")
    tokenizer = ByteTokenizer()
    if stage.canonical_base != "random_init":
        raise ValueError("S0 canonical Base must remain random_init")
    if stage.model.vocab_size != tokenizer.vocab_size:
        raise ValueError("S0 model/tokenizer vocabulary mismatch")

    train_path = repo_root / "data/s0/packaged/train.jsonl"
    manifest_path = repo_root / "data/s0/packaged/manifest.json"
    lock_path = repo_root / "requirements/locks/index.json"
    train_texts = _load_train_texts(train_path)

    trainer_config = TrainerConfig(
        learning_rate=1e-2,
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
    torch.manual_seed(seed)
    model = TwelveSixDecoder(stage.model, stage.init)
    trainer = Trainer(model, trainer_config, device="cpu")

    for step in range(train_steps):
        text = train_texts[step % len(train_texts)]
        token_ids = tokenizer.encode(text)[: stage.model.max_seq_len]
        if len(token_ids) < 2:
            raise ValueError("S0 training record must encode to at least two tokens")
        batch_ids = torch.tensor([token_ids], dtype=torch.long)
        metrics = trainer.train_microbatch({"input_ids": batch_ids, "labels": batch_ids})
        if not metrics.optimizer_stepped:
            raise RuntimeError("HTTP parity evidence requires one optimizer update per microbatch")
    trainer.assert_checkpoint_safe()

    dataset_manifest_sha256 = sha256_file(manifest_path)
    train_sha256 = sha256_file(train_path)
    environment_lock_sha256 = sha256_file(lock_path)
    model_spec = stage.model.to_dict()
    init_spec = stage.init.to_dict()
    run_manifest = {
        "schema_version": 1,
        "run_id": f"s0-d05-http-parity-{candidate_sha[:12]}",
        "stage": "S0",
        "run_kind": "trained_checkpoint_http_parity",
        "state": "COMPLETED_LOCAL_FREE",
        "candidate": {
            "repository": REPOSITORY,
            "git_sha": candidate_sha,
            "branch_or_tag": "exact-checkout",
            "modelspec_sha256": hash_json(model_spec),
            "initspec_sha256": hash_json(init_spec),
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
            "seed": seed,
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
            "target_steps": train_steps,
            "target_tokens": trainer.tokens_seen,
            "checkpoint_interval_steps": train_steps,
        },
        "environment": {"lock_sha256": environment_lock_sha256},
    }
    identity = bind_checkpoint_identity(
        run_manifest=run_manifest,
        model_spec=model_spec,
        init_spec=init_spec,
        tokenizer_identity=tokenizer.identity.to_dict(),
        packing_identity={
            "version": PACKING_VERSION,
            "config_sha256": PACKING_CONFIG_HASH,
        },
        step=trainer.optimizer_step,
        tokens_seen=trainer.tokens_seen,
        environment_lock_hash=environment_lock_sha256,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoint"
    manifest = save_trainer_checkpoint(
        checkpoint_dir,
        model=model,
        trainer=trainer,
        identity=identity,
    )
    direct_backend = S0TorchInferenceBackend(model, tokenizer)
    reloaded_backend = load_first_party_backend(checkpoint_dir)
    parity_report = compare_backends(
        direct_backend,
        reloaded_backend,
        ("12-6", "Base"),
        max_new_tokens=4,
        atol=0.0,
        rtol=0.0,
    )
    if not parity_report.passed:
        raise RuntimeError("direct-vs-reloaded S0 inference parity failed")

    server = make_server(
        reloaded_backend,
        host="127.0.0.1",
        port=0,
        model_name=MODEL_NAME,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    address = (str(host), int(port))
    try:
        health_status, health = _request_json(address, "GET", "/healthz")
        models_status, models = _request_json(address, "GET", "/v1/models")

        greedy_request = {
            "model": MODEL_NAME,
            "prompt": "12-6",
            "max_tokens": 4,
            "temperature": 0,
            "top_p": 1.0,
            "seed": 17,
        }
        greedy_direct = completion_response(
            reloaded_backend,
            greedy_request,
            response_id="cmpl-direct",
            created=0,
            model_name=MODEL_NAME,
        )
        greedy_status, greedy_http = _request_json(
            address,
            "POST",
            "/v1/completions",
            greedy_request,
        )

        sampled_request = {
            "model": MODEL_NAME,
            "prompt": "12-6",
            "max_tokens": 4,
            "temperature": 0.8,
            "top_p": 0.9,
            "seed": 17,
        }
        sampled_direct = completion_response(
            reloaded_backend,
            sampled_request,
            response_id="cmpl-direct",
            created=0,
            model_name=MODEL_NAME,
        )
        sampled_status_a, sampled_http_a = _request_json(
            address,
            "POST",
            "/v1/completions",
            sampled_request,
        )
        sampled_status_b, sampled_http_b = _request_json(
            address,
            "POST",
            "/v1/completions",
            sampled_request,
        )

        prompt_ids = reloaded_backend.encode("A")
        first_token = greedy_token(reloaded_backend.next_token_logits(prompt_ids))
        stop_text = reloaded_backend.decode([first_token])
        if not stop_text:
            raise RuntimeError("unable to construct deterministic HTTP stop-string probe")
        stop_request = {
            "model": MODEL_NAME,
            "prompt": "A",
            "max_tokens": 8,
            "temperature": 0,
            "top_p": 1.0,
            "seed": 19,
            "stop": stop_text,
        }
        stop_direct = completion_response(
            reloaded_backend,
            stop_request,
            response_id="cmpl-direct",
            created=0,
            model_name=MODEL_NAME,
        )
        stop_status, stop_http = _request_json(
            address,
            "POST",
            "/v1/completions",
            stop_request,
        )

        context_request = {
            "model": MODEL_NAME,
            "prompt": "A" * reloaded_backend.max_context_tokens,
            "max_tokens": 1,
            "temperature": 0,
            "seed": 23,
        }
        context_direct = completion_response(
            reloaded_backend,
            context_request,
            response_id="cmpl-direct",
            created=0,
            model_name=MODEL_NAME,
        )
        context_status, context_http = _request_json(
            address,
            "POST",
            "/v1/completions",
            context_request,
        )
        over_context_status, _ = _request_json(
            address,
            "POST",
            "/v1/completions",
            {
                "model": MODEL_NAME,
                "prompt": "A" * (reloaded_backend.max_context_tokens + 1),
                "max_tokens": 1,
                "temperature": 0,
            },
        )
        chat_status, _ = _request_json(
            address,
            "POST",
            "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "ignored"}]},
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        if thread.is_alive():
            raise RuntimeError("loopback HTTP server did not shut down cleanly")

    greedy_matches_direct = (
        greedy_status == 200
        and _normalized_completion(greedy_http) == _normalized_completion(greedy_direct)
    )
    sampled_matches_direct = (
        sampled_status_a == 200
        and _normalized_completion(sampled_http_a) == _normalized_completion(sampled_direct)
    )
    sampled_repeatable = (
        sampled_status_b == 200
        and _normalized_completion(sampled_http_a) == _normalized_completion(sampled_http_b)
    )
    stop_choices = stop_http.get("choices")
    stop_finish_reason = None
    if isinstance(stop_choices, list) and stop_choices and isinstance(stop_choices[0], dict):
        stop_finish_reason = stop_choices[0].get("finish_reason")
    stop_matches_direct = (
        stop_status == 200
        and _normalized_completion(stop_http) == _normalized_completion(stop_direct)
        and stop_finish_reason == "stop"
    )
    context_choices = context_http.get("choices")
    context_finish_reason = None
    if (
        isinstance(context_choices, list)
        and context_choices
        and isinstance(context_choices[0], dict)
    ):
        context_finish_reason = context_choices[0].get("finish_reason")
    context_matches_direct = (
        context_status == 200
        and _normalized_completion(context_http) == _normalized_completion(context_direct)
        and context_finish_reason == "length"
    )

    model_data = models.get("data")
    model_list_ok = False
    if isinstance(model_data, list) and model_data and isinstance(model_data[0], dict):
        model_list_ok = model_data[0].get("id") == MODEL_NAME

    diagnostics = reloaded_backend.diagnostics()
    evidence: dict[str, Any] = {
        "schema": SCHEMA,
        "candidate": {
            "repository": REPOSITORY,
            "sha": candidate_sha,
            "random_init_pretraining_only": True,
            "model_spec_sha256": diagnostics["model_spec_sha256"],
            "parameter_count": diagnostics["parameter_count"],
            "tokenizer_config_sha256": diagnostics["tokenizer_config_sha256"],
            "tokenizer_vocab_sha256": diagnostics["tokenizer_vocab_sha256"],
            "max_context_tokens": diagnostics["max_context_tokens"],
        },
        "training": {
            "seed": seed,
            "optimizer_steps": trainer.optimizer_step,
            "tokens_seen": trainer.tokens_seen,
            "paid_compute": False,
        },
        "checkpoint": {
            "checkpoint_id": manifest["checkpoint_id"],
            "git_sha": diagnostics["git_sha"],
            "step": diagnostics["step"],
            "tokens_seen": diagnostics["tokens_seen"],
            "serialization_pickle": manifest["serialization"]["pickle"],
            "directory": "checkpoint",
        },
        "parity": parity_report.to_dict(),
        "http": {
            "health_ok": health_status == 200 and health.get("status") == "ok",
            "model_list_ok": models_status == 200 and model_list_ok,
            "greedy_matches_direct": greedy_matches_direct,
            "sampled_matches_direct": sampled_matches_direct,
            "seeded_sampling_repeatable": sampled_repeatable,
            "stop_matches_direct": stop_matches_direct,
            "context_limit_matches_direct": context_matches_direct,
            "over_context_rejected": over_context_status == 400,
            "chat_rejected": chat_status == 404,
            "loopback_only": address[0] == "127.0.0.1",
        },
        "raw_base_semantics": {
            "hidden_prompt": False,
            "chat_roles": False,
            "instruction_template": False,
            "alignment_behavior": False,
        },
        "promotion": {
            "claimed": False,
            "audit_verdict_claimed": False,
            "status": "EXPERIMENTAL",
        },
    }
    evidence["evidence_sha256"] = _evidence_hash(evidence)
    validate_http_parity_evidence(evidence, expected_candidate_sha=candidate_sha)
    evidence_path = output_dir / "s0-http-parity-evidence.json"
    evidence_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return evidence


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train a tiny real S0 CPU checkpoint, reload it through first-party inference, "
            "and prove exact loopback /v1/completions parity."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--candidate-sha", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260825)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    evidence = collect_s0_http_parity_evidence(
        args.repo_root,
        args.candidate_sha,
        args.output_dir,
        train_steps=args.train_steps,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "schema": evidence["schema"],
                "candidate_sha": evidence["candidate"]["sha"],
                "checkpoint_id": evidence["checkpoint"]["checkpoint_id"],
                "evidence_sha256": evidence["evidence_sha256"],
                "http": evidence["http"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
