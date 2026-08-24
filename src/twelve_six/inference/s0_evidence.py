"""Exact trained-checkpoint evidence for the canonical first-party S0 inference path.

The collector deliberately reuses the accepted D01 model, D03/D04 data/tokenizer
helpers, D02 Trainer, D05 checkpoint contract and D07 generation/parity code. It
creates no alternative architecture or sampler implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

import torch

from twelve_six.checkpoint import (
    CheckpointCompatibilityError,
    CheckpointError,
    bind_checkpoint_identity,
    hash_json,
    load_checkpoint,
    save_trainer_checkpoint,
    sha256_file,
)
from twelve_six.integration import S0TorchInferenceBackend
from twelve_six.model import TwelveSixDecoder, load_stage_config
from twelve_six.packing import PACKING_CONFIG_HASH, PACKING_VERSION
from twelve_six.s0_candidate_evaluation import (
    _git_head,
    _load_jsonl,
    _make_batches,
    _seed_all,
    _train_range,
)
from twelve_six.tokenization import ByteTokenizer
from twelve_six.training import Trainer, TrainerConfig

from .contracts import GenerationConfig, GenerationResult
from .first_party import load_first_party_backend
from .generation import generate
from .openai_compat import completion_response
from .parity import compare_backends

SCHEMA = "12-6.s0-first-party-inference-evidence.v1"
_REPOSITORY = "Oleksii-debug/12-6-ai."
_EXACT_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _canonical_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _generation_record(result: GenerationResult) -> dict[str, Any]:
    return {
        "prompt_token_ids": list(result.prompt_token_ids),
        "generated_token_ids": list(result.generated_token_ids),
        "generated_text_sha256": _text_sha256(result.text),
        "generated_text_utf8_bytes": len(result.text.encode("utf-8")),
        "stop_reason": result.stop_reason,
    }


def _model_snapshot(model: TwelveSixDecoder) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def _model_matches_snapshot(
    model: TwelveSixDecoder, snapshot: dict[str, torch.Tensor]
) -> bool:
    state = model.state_dict()
    return state.keys() == snapshot.keys() and all(
        torch.equal(state[name], snapshot[name]) for name in state
    )


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
        "run_id": f"s0-d05-inference-evidence-{candidate_sha[:12]}",
        "stage": "S0",
        "run_kind": "trained_checkpoint_inference_evidence",
        "state": "RUNNING",
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
        "environment": {"lock_sha256": environment_lock_sha256},
    }


def _expect_compatibility_rejection_without_mutation(
    checkpoint: Path,
    *,
    stage: Any,
    expected_field: str,
) -> bool:
    model = TwelveSixDecoder(stage.model, stage.init)
    before = _model_snapshot(model)
    kwargs: dict[str, Any] = {expected_field: "0" * 64}
    try:
        load_checkpoint(checkpoint, model=model, restore_rng=False, **kwargs)
    except CheckpointCompatibilityError:
        return _model_matches_snapshot(model, before)
    return False


def _corrupt_checkpoint_rejected(checkpoint: Path) -> bool:
    with tempfile.TemporaryDirectory(prefix="twelve-six-inference-corrupt-") as tmp:
        corrupt = Path(tmp) / "checkpoint"
        shutil.copytree(checkpoint, corrupt)
        weights = corrupt / "model.safetensors"
        data = bytearray(weights.read_bytes())
        if not data:
            raise RuntimeError("model.safetensors is unexpectedly empty")
        data[-1] ^= 0x01
        weights.write_bytes(data)
        try:
            load_first_party_backend(corrupt)
        except CheckpointError:
            return True
    return False


def validate_s0_inference_evidence(
    report: dict[str, Any], *, expected_candidate_sha: str | None = None
) -> None:
    """Fail closed on semantic or self-hash drift in an inference evidence report."""

    if report.get("schema") != SCHEMA:
        raise ValueError("unexpected inference evidence schema")
    candidate = report.get("candidate")
    if not isinstance(candidate, dict):
        raise TypeError("candidate evidence must be an object")
    candidate_sha = candidate.get("sha")
    if not isinstance(candidate_sha, str) or _EXACT_GIT_SHA.fullmatch(candidate_sha) is None:
        raise ValueError("candidate SHA must be exact lowercase 40-hex")
    if expected_candidate_sha is not None and candidate_sha != expected_candidate_sha:
        raise ValueError("candidate SHA does not match expected exact source")
    if candidate.get("repository") != _REPOSITORY:
        raise ValueError("inference evidence repository identity mismatch")
    if candidate.get("canonical_base") != "random_init":
        raise ValueError("canonical Base must remain random_init")

    checkpoint = report.get("checkpoint")
    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint evidence must be an object")
    checkpoint_id = checkpoint.get("checkpoint_id")
    if not isinstance(checkpoint_id, str) or _SHA256.fullmatch(checkpoint_id) is None:
        raise ValueError("checkpoint identity must be SHA-256")

    parity = report.get("parity")
    if not isinstance(parity, dict) or parity.get("passed") is not True:
        raise ValueError("direct-vs-reloaded logits/token/decode parity did not pass")
    if parity.get("failures") != []:
        raise ValueError("parity report contains failures")
    if parity.get("atol") != 0.0 or parity.get("rtol") != 0.0:
        raise ValueError("same-runtime first-party parity must be exact")

    sampling = report.get("seeded_sampling")
    if not isinstance(sampling, dict):
        raise TypeError("seeded sampling evidence must be an object")
    if sampling.get("repeatable") is not True or sampling.get("direct_reload_equal") is not True:
        raise ValueError("seeded sampling repeatability/parity failed")

    stops = report.get("stop_semantics")
    if not isinstance(stops, dict):
        raise TypeError("stop evidence must be an object")
    if stops.get("token_stop_reason") != "stop_token":
        raise ValueError("token stop semantics failed")
    if stops.get("string_stop_reason") != "stop_string":
        raise ValueError("string stop semantics failed")
    if stops.get("context_stop_reason") != "context_limit":
        raise ValueError("context limit semantics failed")
    if stops.get("over_context_prompt_rejected") is not True:
        raise ValueError("over-context prompt did not fail closed")

    fail_closed = report.get("fail_closed")
    if not isinstance(fail_closed, dict) or not all(
        fail_closed.get(key) is True
        for key in (
            "corrupt_checkpoint_rejected",
            "model_identity_mismatch_rejected_before_mutation",
            "tokenizer_identity_mismatch_rejected_before_mutation",
            "vocab_identity_mismatch_rejected_before_mutation",
            "chat_messages_rejected",
        )
    ):
        raise ValueError("one or more fail-closed inference probes failed")

    api = report.get("openai_completion_handoff")
    if not isinstance(api, dict) or api.get("raw_completion_matches_greedy") is not True:
        raise ValueError("raw completion handoff diverged from canonical generation")

    evidence_hash = report.get("evidence_sha256")
    if not isinstance(evidence_hash, str) or _SHA256.fullmatch(evidence_hash) is None:
        raise ValueError("inference evidence self-hash is missing or invalid")
    unhashed = dict(report)
    unhashed.pop("evidence_sha256", None)
    if _canonical_hash(unhashed) != evidence_hash:
        raise ValueError("inference evidence self-hash mismatch")


def collect_s0_inference_evidence(
    repo_root: Path,
    candidate_sha: str,
    output_dir: Path,
    *,
    train_steps: int = 40,
    seed: int = 20260825,
    verify_checkout: bool = True,
) -> dict[str, Any]:
    """Train S0 locally, retain checkpoint-v1, reload it, and prove D07 semantics."""

    repo_root = repo_root.resolve()
    output_dir = output_dir.resolve()
    if _EXACT_GIT_SHA.fullmatch(candidate_sha) is None:
        raise ValueError("candidate_sha must be exact lowercase 40-hex")
    if verify_checkout and _git_head(repo_root) != candidate_sha:
        raise ValueError("candidate_sha is stale: it does not equal checkout HEAD")
    if train_steps < 4:
        raise ValueError("train_steps must be >= 4")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("output_dir must be absent or empty")
    output_dir.mkdir(parents=True, exist_ok=True)

    stage_path = repo_root / "configs/stages/s0_10k.json"
    train_path = repo_root / "data/s0/packaged/train.jsonl"
    manifest_path = repo_root / "data/s0/packaged/manifest.json"
    environment_lock_path = repo_root / "requirements/locks/index.json"

    stage = load_stage_config(stage_path)
    tokenizer = ByteTokenizer()
    if stage.canonical_base != "random_init":
        raise ValueError("S0 canonical Base must remain random_init")
    if stage.model.vocab_size != tokenizer.vocab_size:
        raise ValueError("tokenizer/model vocabulary mismatch")

    train_rows = _load_jsonl(train_path)
    train_batches, _ = _make_batches(
        train_rows, tokenizer, max_seq_len=stage.model.max_seq_len
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
    trainer = Trainer(model, trainer_config, device="cpu")
    _train_range(trainer, train_batches, start_step=0, end_step=train_steps)

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

    checkpoint = output_dir / "checkpoint-v1"
    manifest = save_trainer_checkpoint(
        checkpoint,
        model=model,
        trainer=trainer,
        identity=identity,
    )
    direct_backend = S0TorchInferenceBackend(model, tokenizer)
    reloaded_backend = load_first_party_backend(checkpoint)

    prompts = ("12-6", "Base", "Україна")
    parity_report = compare_backends(
        direct_backend,
        reloaded_backend,
        prompts,
        max_new_tokens=8,
        atol=0.0,
        rtol=0.0,
    )
    if not parity_report.passed:
        raise RuntimeError("exact direct-vs-reloaded inference parity failed")

    greedy_config = GenerationConfig(max_new_tokens=8, sample=False, seed=seed)
    direct_greedy = generate(direct_backend, "12-6", greedy_config)
    reloaded_greedy = generate(reloaded_backend, "12-6", greedy_config)
    if direct_greedy != reloaded_greedy or not reloaded_greedy.generated_token_ids:
        raise RuntimeError("greedy direct/reload generation parity failed")

    sampled_config = GenerationConfig(
        max_new_tokens=8,
        sample=True,
        temperature=0.7,
        top_k=4,
        top_p=1.0,
        seed=seed + 7,
    )
    sampled_a = generate(reloaded_backend, "12-6", sampled_config)
    sampled_b = generate(reloaded_backend, "12-6", sampled_config)
    sampled_direct = generate(direct_backend, "12-6", sampled_config)
    seeded_repeatable = sampled_a == sampled_b
    sampled_direct_reload_equal = sampled_direct == sampled_a
    if not seeded_repeatable or not sampled_direct_reload_equal:
        raise RuntimeError("seeded sampling repeatability/parity failed")

    first_token = reloaded_greedy.generated_token_ids[0]
    token_stop = generate(
        reloaded_backend,
        "12-6",
        GenerationConfig(
            max_new_tokens=8,
            sample=False,
            seed=seed,
            stop_token_ids=(first_token,),
        ),
    )
    if token_stop.stop_reason != "stop_token":
        raise RuntimeError("token stop did not terminate generation")

    if not reloaded_greedy.text:
        raise RuntimeError("greedy fixture produced empty decoded text")
    string_stop = generate(
        reloaded_backend,
        "12-6",
        GenerationConfig(
            max_new_tokens=8,
            sample=False,
            seed=seed,
            stop_strings=(reloaded_greedy.text,),
            strip_stop_strings=True,
        ),
    )
    if string_stop.stop_reason != "stop_string" or string_stop.text:
        raise RuntimeError("string stop/strip semantics failed")

    context_prompt = "a" * (reloaded_backend.max_context_tokens - 1)
    context_result = generate(
        reloaded_backend,
        context_prompt,
        GenerationConfig(max_new_tokens=4, sample=False, seed=seed),
    )
    if context_result.stop_reason != "context_limit":
        raise RuntimeError("context limit did not terminate generation")
    over_context_rejected = False
    try:
        generate(
            reloaded_backend,
            "a" * (reloaded_backend.max_context_tokens + 1),
            GenerationConfig(max_new_tokens=1, sample=False),
        )
    except ValueError:
        over_context_rejected = True

    raw_response = completion_response(
        reloaded_backend,
        {
            "prompt": "12-6",
            "max_tokens": 8,
            "temperature": 0,
            "top_p": 1.0,
            "seed": seed,
        },
        response_id="cmpl-evidence",
        created=0,
        model_name="12-6-base",
    )
    raw_text = raw_response["choices"][0]["text"]
    raw_completion_matches = raw_text == reloaded_greedy.text
    chat_messages_rejected = False
    try:
        completion_response(reloaded_backend, {"messages": []})
    except (TypeError, ValueError):
        chat_messages_rejected = True

    fail_closed = {
        "corrupt_checkpoint_rejected": _corrupt_checkpoint_rejected(checkpoint),
        "model_identity_mismatch_rejected_before_mutation": (
            _expect_compatibility_rejection_without_mutation(
                checkpoint,
                stage=stage,
                expected_field="expected_model_spec_hash",
            )
        ),
        "tokenizer_identity_mismatch_rejected_before_mutation": (
            _expect_compatibility_rejection_without_mutation(
                checkpoint,
                stage=stage,
                expected_field="expected_tokenizer_hash",
            )
        ),
        "vocab_identity_mismatch_rejected_before_mutation": (
            _expect_compatibility_rejection_without_mutation(
                checkpoint,
                stage=stage,
                expected_field="expected_tokenizer_vocab_hash",
            )
        ),
        "chat_messages_rejected": chat_messages_rejected,
    }
    if not all(fail_closed.values()):
        raise RuntimeError("one or more fail-closed probes failed")

    checkpoint_files = {
        path.name: sha256_file(path)
        for path in sorted(checkpoint.iterdir(), key=lambda item: item.name)
        if path.is_file()
    }
    diagnostics = reloaded_backend.diagnostics()
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "candidate": {
            "repository": _REPOSITORY,
            "sha": candidate_sha,
            "canonical_base": stage.canonical_base,
            "model_spec_sha256": hash_json(stage.model.to_dict()),
            "init_spec_sha256": hash_json(stage.init.to_dict()),
            "parameter_count": stage.expected_parameters,
        },
        "training_fixture": {
            "steps": trainer.optimizer_step,
            "tokens_seen": trainer.tokens_seen,
            "seed": seed,
            "dataset_manifest_sha256": dataset_manifest_sha256,
            "train_sha256": train_sha256,
            "environment_lock_sha256": environment_lock_sha256,
            "packing_sha256": PACKING_CONFIG_HASH,
            "packing_version": PACKING_VERSION,
        },
        "checkpoint": {
            "checkpoint_id": manifest["checkpoint_id"],
            "serialization_pickle": manifest["serialization"]["pickle"],
            "files": checkpoint_files,
            "retained_relative_path": "checkpoint-v1",
        },
        "diagnostics": diagnostics,
        "greedy": {
            "direct_reload_equal": direct_greedy == reloaded_greedy,
            "result": _generation_record(reloaded_greedy),
        },
        "seeded_sampling": {
            "seed": sampled_config.seed,
            "temperature": sampled_config.temperature,
            "top_k": sampled_config.top_k,
            "top_p": sampled_config.top_p,
            "repeatable": seeded_repeatable,
            "direct_reload_equal": sampled_direct_reload_equal,
            "result": _generation_record(sampled_a),
        },
        "parity": parity_report.to_dict(),
        "stop_semantics": {
            "token_stop_reason": token_stop.stop_reason,
            "token_stop_generated_count": len(token_stop.generated_token_ids),
            "string_stop_reason": string_stop.stop_reason,
            "string_stop_stripped_to_empty": string_stop.text == "",
            "context_stop_reason": context_result.stop_reason,
            "context_generated_count": len(context_result.generated_token_ids),
            "over_context_prompt_rejected": over_context_rejected,
        },
        "openai_completion_handoff": {
            "object": raw_response["object"],
            "model": raw_response["model"],
            "raw_completion_matches_greedy": raw_completion_matches,
            "chat_semantics_supported": False,
        },
        "fail_closed": fail_closed,
        "truth_boundary": {
            "local_free_cpu_only": True,
            "foreign_pretrained_weights": False,
            "instruction_or_alignment_training": False,
            "promotion_claimed": False,
            "windows_nvda_live_tested": False,
            "public_server_hardening_claimed": False,
        },
    }
    report["evidence_sha256"] = _canonical_hash(report)
    validate_s0_inference_evidence(report, expected_candidate_sha=candidate_sha)
    report_path = output_dir / "inference_evidence.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m twelve_six.inference.s0_evidence",
        description="Collect exact trained+reloaded S0 first-party inference evidence.",
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--candidate-sha", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260825)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = collect_s0_inference_evidence(
        args.repo_root,
        args.candidate_sha,
        args.output_dir,
        train_steps=args.train_steps,
        seed=args.seed,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
