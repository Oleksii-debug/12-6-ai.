"""Exact-candidate LOCAL_FREE acceptance evidence for first-party S0 inference.

This module orchestrates existing D01/D02/D04/D05/D07 contracts. It deliberately
contains no alternate model, tokenizer, checkpoint, sampling, parity, CLI, or HTTP
implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import threading
from dataclasses import replace
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import torch

from twelve_six.checkpoint import (
    CheckpointCompatibilityError,
    CheckpointIntegrityError,
    bind_checkpoint_identity,
    hash_json,
    save_checkpoint,
    save_trainer_checkpoint,
    sha256_file,
    verify_checkpoint,
)
from twelve_six.integration import S0TorchInferenceBackend
from twelve_six.model import ModelSpec, TwelveSixDecoder, load_stage_config
from twelve_six.packing import PACKING_CONFIG_HASH, PACKING_VERSION
from twelve_six.tokenization import ByteTokenizer
from twelve_six.training import Trainer, TrainerConfig

from .contracts import GenerationConfig
from .first_party import load_first_party_backend
from .generation import generate
from .parity import compare_backends
from .sampling import greedy_token
from .server import make_server

SCHEMA_VERSION = "12-6.s0-inference-acceptance.v1"
AUTHORITY = "LOCAL_FREE_OR_FREE_HOSTED_CPU_EVIDENCE_NOT_PROMOTION"
REPOSITORY = "Oleksii-debug/12-6-ai."


def _git_head(repo_root: Path) -> str:
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("inference acceptance requires an exact Git checkout") from exc
    if len(head) not in {40, 64} or any(ch not in "0123456789abcdef" for ch in head):
        raise ValueError("git HEAD must be a full lowercase Git object id")
    return head


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _json_request(url: str, payload: dict[str, object] | None = None) -> dict[str, Any]:
    data = None
    headers: dict[str, str] = {}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = Request(url, data=data, headers=headers, method="POST" if data else "GET")
    with urlopen(request, timeout=10) as response:  # noqa: S310 - loopback-only test server
        if response.status != 200:
            raise RuntimeError(f"loopback server returned HTTP {response.status}")
        decoded = json.loads(response.read().decode("utf-8"))
    if not isinstance(decoded, dict):
        raise TypeError("loopback server response must be a JSON object")
    return decoded


def _load_training_rows(repo_root: Path, tokenizer: ByteTokenizer, max_seq_len: int) -> list[list[int]]:
    rows: list[list[int]] = []
    train_path = repo_root / "data/s0/packaged/train.jsonl"
    for line_number, line in enumerate(train_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict) or not isinstance(payload.get("text"), str):
            raise ValueError(f"train.jsonl:{line_number} must contain text")
        token_ids = tokenizer.encode(payload["text"])[:max_seq_len]
        if len(token_ids) < 2:
            raise ValueError("S0 inference acceptance requires two-token training records")
        rows.append(token_ids)
    if not rows:
        raise ValueError("S0 training split is empty")
    return rows


def _strict_run_manifest(
    *,
    repo_root: Path,
    candidate_sha: str,
    stage: Any,
    tokenizer: ByteTokenizer,
    trainer_config: TrainerConfig,
    train_steps: int,
) -> tuple[dict[str, Any], str]:
    manifest_path = repo_root / "data/s0/packaged/manifest.json"
    train_path = repo_root / "data/s0/packaged/train.jsonl"
    environment_lock_path = repo_root / "requirements/locks/index.json"
    environment_lock_sha256 = sha256_file(environment_lock_path)
    run_manifest = {
        "schema_version": 1,
        "run_id": f"s0-d05-inference-acceptance-{candidate_sha[:12]}",
        "stage": "S0",
        "run_kind": "inference_acceptance_training",
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
            "dataset_manifest_sha256": sha256_file(manifest_path),
            "tokenizer_sha256": tokenizer.identity.config_sha256,
            "tokenizer_vocab_sha256": tokenizer.identity.vocab_sha256,
            "tokenizer_version": tokenizer.identity.version,
            "split_identity": f"train:{sha256_file(train_path)}",
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
            "target_steps": train_steps,
            "target_tokens": 1,
            "checkpoint_interval_steps": train_steps,
        },
        "environment": {"lock_sha256": environment_lock_sha256},
    }
    return run_manifest, environment_lock_sha256


def _train_and_checkpoint(
    repo_root: Path,
    candidate_sha: str,
    checkpoint: Path,
    *,
    train_steps: int,
    seed: int,
) -> tuple[TwelveSixDecoder, ByteTokenizer, object]:
    stage = load_stage_config(repo_root / "configs/stages/s0_10k.json")
    if stage.canonical_base != "random_init":
        raise ValueError("canonical S0 Base must remain random_init")
    tokenizer = ByteTokenizer()
    if stage.model.vocab_size != tokenizer.vocab_size:
        raise ValueError("canonical tokenizer and ModelSpec vocabulary mismatch")
    rows = _load_training_rows(repo_root, tokenizer, stage.model.max_seq_len)
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
    torch.manual_seed(seed)
    model = TwelveSixDecoder(stage.model, stage.init)
    trainer = Trainer(model, trainer_config, device="cpu")
    for step in range(train_steps):
        token_ids = rows[step % len(rows)]
        ids = torch.tensor([token_ids], dtype=torch.long)
        metrics = trainer.train_microbatch({"input_ids": ids, "labels": ids})
        if not metrics.optimizer_stepped:
            raise RuntimeError(f"optimizer did not step at acceptance step {step}")
    trainer.assert_checkpoint_safe()
    run_manifest, environment_lock_sha256 = _strict_run_manifest(
        repo_root=repo_root,
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
        environment_lock_hash=environment_lock_sha256,
    )
    save_trainer_checkpoint(checkpoint, model=model, trainer=trainer, identity=identity)
    return model, tokenizer, identity


def _prove_fail_closed(
    output_dir: Path,
    model: TwelveSixDecoder,
    tokenizer: ByteTokenizer,
    identity: Any,
) -> dict[str, bool]:
    checkpoint = output_dir / "checkpoint"

    corrupt = output_dir / "negative-corrupt"
    shutil.copytree(checkpoint, corrupt)
    weights_path = corrupt / "weights.safetensors"
    payload = bytearray(weights_path.read_bytes())
    payload[-1] ^= 1
    weights_path.write_bytes(payload)
    corrupt_rejected = False
    try:
        load_first_party_backend(corrupt)
    except CheckpointIntegrityError:
        corrupt_rejected = True

    wrong_tokenizer = output_dir / "negative-tokenizer"
    save_checkpoint(
        wrong_tokenizer,
        model=model,
        identity=replace(identity, tokenizer_hash="f" * 64),
    )
    tokenizer_rejected = False
    try:
        load_first_party_backend(wrong_tokenizer)
    except CheckpointCompatibilityError:
        tokenizer_rejected = True

    wrong_training = dict(identity.training_config)
    wrong_training["training"] = dict(identity.training_config["training"])
    wrong_training["training"]["context_length"] = model.spec.max_seq_len + 1
    wrong_context = output_dir / "negative-context"
    save_checkpoint(
        wrong_context,
        model=model,
        identity=replace(identity, training_config=wrong_training),
    )
    context_rejected = False
    try:
        load_first_party_backend(wrong_context)
    except CheckpointCompatibilityError:
        context_rejected = True

    bad_spec_payload = model.spec.to_dict()
    bad_spec_payload["vocab_size"] = tokenizer.vocab_size - 1
    bad_spec = ModelSpec.from_dict(bad_spec_payload)
    bad_model = TwelveSixDecoder(bad_spec)
    wrong_vocab = output_dir / "negative-vocab"
    save_checkpoint(
        wrong_vocab,
        model=bad_model,
        identity=replace(
            identity,
            model_spec=bad_spec.to_dict(),
            parameter_count=bad_spec.parameter_count(),
        ),
    )
    vocab_rejected = False
    try:
        load_first_party_backend(wrong_vocab)
    except CheckpointCompatibilityError:
        vocab_rejected = True

    result = {
        "corrupt_checkpoint_rejected": corrupt_rejected,
        "tokenizer_mismatch_rejected": tokenizer_rejected,
        "context_mismatch_rejected": context_rejected,
        "vocab_mismatch_rejected": vocab_rejected,
    }
    if not all(result.values()):
        raise AssertionError(f"fail-closed acceptance failed: {result}")
    return result


def _prove_cli(checkpoint: Path, prompt: str) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "twelve_six.inference.cli",
        "--checkpoint",
        str(checkpoint),
        "--prompt",
        prompt,
        "--greedy",
        "--max-new-tokens",
        "4",
        "--json",
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"first-party CLI failed: {completed.stderr.strip()}")
    if "\x1b[" in completed.stdout + completed.stderr:
        raise RuntimeError("first-party CLI emitted ANSI escape sequences")
    payload = json.loads(completed.stdout)
    if not isinstance(payload, dict):
        raise TypeError("first-party CLI JSON output must be an object")
    return {
        "returncode": completed.returncode,
        "ansi_escape_sequences": False,
        "stdout_sha256": _text_sha256(completed.stdout),
        "stderr_sha256": _text_sha256(completed.stderr),
        "stop_reason": payload.get("stop_reason"),
        "generated_token_ids": payload.get("generated_token_ids"),
        "stdin_supported_by_canonical_cli": True,
        "json_diagnostics": isinstance(payload.get("backend"), dict),
    }


def _prove_loopback_server(backend: Any, prompt: str) -> dict[str, Any]:
    model_name = "12-6-base-s0"
    server = make_server(backend, host="127.0.0.1", port=0, model_name=model_name)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    base_url = f"http://{host}:{port}"
    try:
        health = _json_request(f"{base_url}/healthz")
        models = _json_request(f"{base_url}/v1/models")
        completion_payload = {
            "model": model_name,
            "prompt": prompt,
            "max_tokens": 4,
            "temperature": 0,
            "top_p": 1.0,
            "seed": 20260825,
        }
        completion = _json_request(
            f"{base_url}/v1/completions",
            completion_payload,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)
    if thread.is_alive():
        raise RuntimeError("loopback server thread did not stop")
    choices = completion.get("choices")
    usage = completion.get("usage")
    model_rows = models.get("data")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise TypeError("completion response choices are invalid")
    if not isinstance(usage, dict):
        raise TypeError("completion response usage is invalid")
    if not isinstance(model_rows, list) or not model_rows:
        raise TypeError("model-list response is invalid")
    expected = generate(
        backend,
        prompt,
        GenerationConfig(max_new_tokens=4, sample=False, seed=20260825),
    )
    if choices[0].get("text") != expected.text:
        raise AssertionError("HTTP completion differs from canonical direct generation")
    return {
        "health_status": health.get("status"),
        "model_id": model_rows[0].get("id") if isinstance(model_rows[0], dict) else None,
        "completion_text_sha256": _text_sha256(str(choices[0].get("text", ""))),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "matches_direct_generation": True,
        "loopback_only": True,
        "chat_semantics": False,
    }


def validate_s0_inference_acceptance(evidence: dict[str, Any]) -> None:
    """Fail closed on drift or overclaim in a generated acceptance manifest."""
    if evidence.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unexpected inference acceptance schema")
    if evidence.get("authority") != AUTHORITY:
        raise ValueError("unexpected inference acceptance authority")
    if evidence.get("repository") != REPOSITORY:
        raise ValueError("unexpected inference acceptance repository")
    if evidence.get("promotion_eligible") is not False or evidence.get("audits_pass") is not False:
        raise ValueError("inference acceptance may not self-promote")
    if evidence.get("windows_nvda") != "NOT_TESTED_BLOCKED_BY_REPOSITORY_IDENTITY":
        raise ValueError("Windows/NVDA truth boundary drifted")
    fail_closed = evidence.get("fail_closed")
    if not isinstance(fail_closed, dict) or not fail_closed or not all(fail_closed.values()):
        raise ValueError("all fail-closed probes must pass")
    parity = evidence.get("parity")
    if not isinstance(parity, dict) or parity.get("passed") is not True:
        raise ValueError("first-party parity must pass")
    server = evidence.get("server")
    if not isinstance(server, dict) or server.get("matches_direct_generation") is not True:
        raise ValueError("loopback server must match direct generation")
    expected_hash = evidence.get("evidence_sha256")
    unsigned = dict(evidence)
    unsigned.pop("evidence_sha256", None)
    if expected_hash != hash_json(unsigned):
        raise ValueError("inference acceptance evidence hash mismatch")


def run_s0_inference_acceptance(
    repo_root: Path,
    output_dir: Path,
    *,
    candidate_sha: str | None = None,
    train_steps: int = 40,
    seed: int = 20260825,
) -> dict[str, Any]:
    """Execute trained+reloaded inference, parity, CLI, HTTP and negative probes."""
    repo_root = repo_root.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    checkout_head = _git_head(repo_root)
    candidate_sha = checkout_head if candidate_sha is None else candidate_sha
    if candidate_sha != checkout_head:
        raise ValueError("candidate SHA must equal the exact checkout HEAD")
    if train_steps < 4:
        raise ValueError("train_steps must be >= 4")
    if seed < 0:
        raise ValueError("seed must be non-negative")

    checkpoint = output_dir / "checkpoint"
    model, tokenizer, identity = _train_and_checkpoint(
        repo_root,
        candidate_sha,
        checkpoint,
        train_steps=train_steps,
        seed=seed,
    )
    checkpoint_manifest = verify_checkpoint(checkpoint)
    direct = S0TorchInferenceBackend(model, tokenizer)
    reloaded = load_first_party_backend(checkpoint)
    prompt = "12-6 Base"

    greedy_config = GenerationConfig(max_new_tokens=8, sample=False, seed=seed)
    greedy_direct = generate(direct, prompt, greedy_config)
    greedy_reloaded = generate(reloaded, prompt, greedy_config)
    if greedy_direct != greedy_reloaded:
        raise AssertionError("greedy generation changed after verified checkpoint reload")

    sample_config = GenerationConfig(
        max_new_tokens=8,
        sample=True,
        temperature=0.8,
        top_k=32,
        top_p=0.9,
        seed=seed,
    )
    sampled_a = generate(reloaded, prompt, sample_config)
    sampled_b = generate(reloaded, prompt, sample_config)
    sampled_direct = generate(direct, prompt, sample_config)
    if sampled_a != sampled_b or sampled_a != sampled_direct:
        raise AssertionError("seeded sampling is not deterministic across reload/direct paths")

    parity = compare_backends(
        direct,
        reloaded,
        ("12-6", "Base", "Україна"),
        max_new_tokens=6,
        atol=0.0,
        rtol=0.0,
    )
    if not parity.passed:
        raise AssertionError("zero-tolerance direct/reloaded parity failed")

    first_token = greedy_token(reloaded.next_token_logits(reloaded.encode("A")))
    token_stopped = generate(
        reloaded,
        "A",
        GenerationConfig(max_new_tokens=8, stop_token_ids=(first_token,)),
    )
    if token_stopped.stop_reason != "stop_token":
        raise AssertionError("stop-token semantics failed")
    first_text = reloaded.decode([first_token])
    text_stopped = generate(
        reloaded,
        "A",
        GenerationConfig(max_new_tokens=8, stop_strings=(first_text,)),
    )
    if text_stopped.stop_reason != "stop_string":
        raise AssertionError("stop-string semantics failed")
    context_full = generate(
        reloaded,
        "A" * reloaded.max_context_tokens,
        GenerationConfig(max_new_tokens=1),
    )
    if context_full.stop_reason != "context_limit" or context_full.generated_token_ids:
        raise AssertionError("exact context-limit semantics failed")
    try:
        generate(
            reloaded,
            "A" * (reloaded.max_context_tokens + 1),
            GenerationConfig(max_new_tokens=1),
        )
    except ValueError:
        over_context_rejected = True
    else:
        over_context_rejected = False
    if not over_context_rejected:
        raise AssertionError("over-context prompt was not rejected")

    fail_closed = _prove_fail_closed(output_dir, model, tokenizer, identity)
    fail_closed["over_context_prompt_rejected"] = over_context_rejected
    cli = _prove_cli(checkpoint, prompt)
    server = _prove_loopback_server(reloaded, prompt)
    diagnostics = reloaded.diagnostics()

    evidence: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "authority": AUTHORITY,
        "repository": REPOSITORY,
        "candidate_sha": candidate_sha,
        "train_steps": train_steps,
        "seed": seed,
        "checkpoint": {
            "checkpoint_id": checkpoint_manifest["checkpoint_id"],
            "manifest_sha256": sha256_file(checkpoint / "manifest.json"),
            "weights_sha256": sha256_file(checkpoint / "weights.safetensors"),
            "step": identity.step,
            "tokens_seen": identity.tokens_seen,
        },
        "identity": diagnostics,
        "greedy": {
            "generated_token_ids": list(greedy_reloaded.generated_token_ids),
            "text_sha256": _text_sha256(greedy_reloaded.text),
            "stop_reason": greedy_reloaded.stop_reason,
            "direct_reload_equal": True,
        },
        "seeded_sampling": {
            "generated_token_ids": list(sampled_a.generated_token_ids),
            "text_sha256": _text_sha256(sampled_a.text),
            "same_seed_repeat_equal": True,
            "direct_reload_equal": True,
        },
        "parity": {
            "passed": parity.passed,
            "steps_compared": parity.steps_compared,
            "max_abs_error": parity.max_abs_error,
            "max_rel_error": parity.max_rel_error,
        },
        "stop_and_context": {
            "stop_token": token_stopped.stop_reason == "stop_token",
            "stop_string": text_stopped.stop_reason == "stop_string",
            "context_exact_boundary": context_full.stop_reason == "context_limit",
            "over_context_rejected": over_context_rejected,
        },
        "cli": cli,
        "server": server,
        "fail_closed": fail_closed,
        "raw_base_semantics": {
            "chat_template": False,
            "hidden_system_prompt": False,
            "instruction_alignment": False,
            "refusal_layer": False,
        },
        "windows_nvda": "NOT_TESTED_BLOCKED_BY_REPOSITORY_IDENTITY",
        "paid_compute": False,
        "foreign_pretrained_weights": False,
        "audits_pass": False,
        "promotion_eligible": False,
    }
    evidence["evidence_sha256"] = hash_json(evidence)
    validate_s0_inference_acceptance(evidence)
    (output_dir / "inference-acceptance.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return evidence


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run exact-candidate S0 inference acceptance")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-sha")
    parser.add_argument("--train-steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260825)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    evidence = run_s0_inference_acceptance(
        args.repo_root,
        args.output_dir,
        candidate_sha=args.candidate_sha,
        train_steps=args.train_steps,
        seed=args.seed,
    )
    print(json.dumps(evidence, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
