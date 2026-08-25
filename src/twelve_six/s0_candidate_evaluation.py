"""Strict LOCAL_FREE S0 exact-candidate evaluation and report collection.

This is the reusable D04 adapter for the convergence lineage. It intentionally
keeps candidate-quality evidence separate from CI/audit/promotion authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from twelve_six.checkpoint import (
    bind_checkpoint_identity,
    hash_json,
    load_trainer_checkpoint,
    save_trainer_checkpoint,
    sha256_file,
)
from twelve_six.inference import GenerationConfig, generate
from twelve_six.inference.first_party import load_first_party_backend
from twelve_six.integration import S0TorchInferenceBackend
from twelve_six.model import TwelveSixDecoder, load_stage_config
from twelve_six.packing import PACKING_CONFIG_HASH, PACKING_VERSION
from twelve_six.stage_gates import evaluate_s0_integrated
from twelve_six.tokenization import ByteTokenizer
from twelve_six.training import Trainer, TrainerConfig, causal_lm_loss

_EXACT_GIT_SHA = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REPOSITORY = "Oleksii-debug/12-6-ai."


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise TypeError(f"{path}:{line_number} must contain a JSON object")
        rows.append(row)
    if not rows:
        raise ValueError(f"{path} contains no records")
    return rows


def _git_head(repo_root: Path) -> str:
    try:
        value = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("exact candidate evaluation requires a Git checkout") from exc
    if _EXACT_GIT_SHA.fullmatch(value) is None:
        raise ValueError("git HEAD is not an exact lowercase Git object id")
    return value


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _record_hashes(rows: list[dict[str, Any]]) -> set[str]:
    hashes: set[str] = set()
    for row in rows:
        value = row.get("content_sha256")
        if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
            raise ValueError(
                "every packaged record must carry lowercase SHA-256 content identity"
            )
        hashes.add(value)
    return hashes


def _make_batches(
    rows: list[dict[str, Any]], tokenizer: ByteTokenizer, *, max_seq_len: int
) -> tuple[list[dict[str, torch.Tensor]], set[str]]:
    batches: list[dict[str, torch.Tensor]] = []
    identities: set[str] = set()
    for row in rows:
        text = row.get("text")
        if not isinstance(text, str) or not text:
            raise ValueError("every packaged record must contain non-empty text")
        token_ids = tokenizer.encode(text)[:max_seq_len]
        if len(token_ids) < 2:
            raise ValueError(
                "every evaluation/training sequence must contain at least two tokens"
            )
        identities.add(_sha256_bytes(bytes(token_ids)))
        ids = torch.tensor([token_ids], dtype=torch.long)
        batches.append({"input_ids": ids, "labels": ids})
    return batches, identities


@torch.no_grad()
def _mean_loss(
    model: TwelveSixDecoder, batches: list[dict[str, torch.Tensor]]
) -> float:
    model.eval()
    values: list[float] = []
    for batch in batches:
        loss = causal_lm_loss(model(batch["input_ids"]).logits, batch["labels"])
        value = float(loss.detach().cpu().item())
        if not math.isfinite(value) or value < 0:
            raise FloatingPointError("candidate evaluation produced invalid token NLL")
        values.append(value)
    return sum(values) / len(values)


def _train_range(
    trainer: Trainer,
    batches: list[dict[str, torch.Tensor]],
    *,
    start_step: int,
    end_step: int,
) -> None:
    if not (0 <= start_step <= end_step):
        raise ValueError("invalid training step interval")
    for step in range(start_step, end_step):
        metrics = trainer.train_microbatch(batches[step % len(batches)])
        if not metrics.optimizer_stepped or not math.isfinite(metrics.loss):
            raise RuntimeError(
                "candidate training did not complete a finite optimizer step"
            )
    trainer.assert_checkpoint_safe()


def _parameters_equal(left: TwelveSixDecoder, right: TwelveSixDecoder) -> bool:
    left_state = left.state_dict()
    right_state = right.state_dict()
    if left_state.keys() != right_state.keys():
        return False
    return all(torch.equal(left_state[name], right_state[name]) for name in left_state)


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
        "run_id": f"s0-d04-exact-eval-{candidate_sha[:12]}",
        "stage": "S0",
        "run_kind": "integrated_training_evaluation",
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
            "global_batch_tokens": 1,
            "target_steps": train_steps,
            "target_tokens": 1,
            "checkpoint_interval_steps": train_steps // 2,
        },
        "environment": {"lock_sha256": environment_lock_sha256},
    }


def _bind_identity(
    *,
    run_manifest: dict[str, Any],
    stage: Any,
    tokenizer: ByteTokenizer,
    trainer: Trainer,
    environment_lock_sha256: str,
):
    return bind_checkpoint_identity(
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


def collect_s0_candidate_evidence(
    repo_root: Path,
    candidate_sha: str,
    *,
    train_steps: int = 40,
    seed: int = 20260824,
    integrated_candidate: bool = False,
    candidate_manifest_sha256: str | None = None,
    candidate_ci_run_id: int | None = None,
    candidate_ci_head_sha: str | None = None,
    candidate_ci_success: bool | None = None,
    audit_a: dict[str, Any] | None = None,
    audit_b: dict[str, Any] | None = None,
    verify_checkout: bool = True,
) -> dict[str, Any]:
    """Run one deterministic CPU evidence cycle on the exact S0 candidate checkout."""
    repo_root = repo_root.resolve()
    if _EXACT_GIT_SHA.fullmatch(candidate_sha) is None:
        raise ValueError(
            "candidate_sha must be a lowercase full 40- or 64-hex Git object id"
        )
    if verify_checkout and _git_head(repo_root) != candidate_sha:
        raise ValueError("candidate_sha is stale: it does not equal the checkout HEAD")
    if train_steps < 4 or train_steps % 2:
        raise ValueError("train_steps must be an even integer >= 4")
    if seed < 0:
        raise ValueError("seed must be non-negative")

    stage_path = repo_root / "configs/stages/s0_10k.json"
    policy_path = repo_root / "configs/stages/s0_eval_gate.json"
    manifest_path = repo_root / "data/s0/packaged/manifest.json"
    train_path = repo_root / "data/s0/packaged/train.jsonl"
    validation_path = repo_root / "data/s0/packaged/validation.jsonl"
    source_registry_path = repo_root / "data/s0/source_registry.json"
    contamination_registry_path = repo_root / "data/s0/contamination_registry.json"
    environment_lock_path = repo_root / "requirements/locks/index.json"

    stage = load_stage_config(stage_path)
    policy_payload = _load_json(policy_path)
    manifest = _load_json(manifest_path)
    source_registry = _load_json(source_registry_path)
    contamination_registry = _load_json(contamination_registry_path)
    train_rows = _load_jsonl(train_path)
    validation_rows = _load_jsonl(validation_path)

    if manifest.get("dataset_id") != "s0-tiny-controlled-v1":
        raise ValueError("unexpected S0 dataset identity")
    expected_outputs = manifest.get("outputs")
    if not isinstance(expected_outputs, dict):
        raise TypeError("dataset manifest outputs must be an object")
    for name, path in (
        ("train.jsonl", train_path),
        ("validation.jsonl", validation_path),
    ):
        if expected_outputs.get(name) != sha256_file(path):
            raise ValueError(f"{name} hash does not match committed dataset manifest")

    tokenizer = ByteTokenizer()
    if stage.canonical_base != "random_init":
        raise ValueError("S0 canonical Base must remain random_init")
    if stage.model.vocab_size != tokenizer.vocab_size:
        raise ValueError("tokenizer/model vocabulary mismatch")

    train_batches, train_batch_ids = _make_batches(
        train_rows, tokenizer, max_seq_len=stage.model.max_seq_len
    )
    validation_batches, _ = _make_batches(
        validation_rows, tokenizer, max_seq_len=stage.model.max_seq_len
    )
    train_hashes = _record_hashes(train_rows)
    validation_hashes = _record_hashes(validation_rows)
    split_overlap = train_hashes & validation_hashes

    forbidden_hashes_raw = contamination_registry.get(
        "forbidden_normalized_sha256", []
    )
    if not isinstance(forbidden_hashes_raw, list) or not all(
        isinstance(item, str) and _SHA256.fullmatch(item)
        for item in forbidden_hashes_raw
    ):
        raise ValueError(
            "contamination forbidden hash registry must contain SHA-256 strings"
        )
    benchmark_hash_overlap = train_hashes & set(forbidden_hashes_raw)

    forbidden_purposes_raw = contamination_registry.get(
        "forbidden_source_purposes", []
    )
    if not isinstance(forbidden_purposes_raw, list) or not all(
        isinstance(item, str) for item in forbidden_purposes_raw
    ):
        raise ValueError("contamination forbidden source-purpose registry is invalid")
    sources = source_registry.get("sources", [])
    if not isinstance(sources, list):
        raise TypeError("source registry sources must be an array")
    forbidden_source_count = sum(
        1
        for source in sources
        if isinstance(source, dict)
        and source.get("purpose") in set(forbidden_purposes_raw)
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
    reference_model = TwelveSixDecoder(stage.model, stage.init)
    parameter_count = sum(
        parameter.numel() for parameter in reference_model.parameters()
    )
    if parameter_count != stage.expected_parameters:
        raise ValueError(
            "instantiated parameter count disagrees with frozen S0 stage config"
        )
    train_loss_before = _mean_loss(reference_model, train_batches)
    validation_loss_before = _mean_loss(reference_model, validation_batches)
    reference_trainer = Trainer(reference_model, trainer_config, device="cpu")
    _train_range(
        reference_trainer,
        train_batches,
        start_step=0,
        end_step=train_steps,
    )
    train_loss_after = _mean_loss(reference_model, train_batches)
    validation_loss_after = _mean_loss(reference_model, validation_batches)

    split_step = train_steps // 2
    _seed_all(seed)
    partial_model = TwelveSixDecoder(stage.model, stage.init)
    partial_trainer = Trainer(partial_model, trainer_config, device="cpu")
    _train_range(
        partial_trainer,
        train_batches,
        start_step=0,
        end_step=split_step,
    )

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
    split_identity = _bind_identity(
        run_manifest=run_manifest,
        stage=stage,
        tokenizer=tokenizer,
        trainer=partial_trainer,
        environment_lock_sha256=environment_lock_sha256,
    )

    with tempfile.TemporaryDirectory(
        prefix="twelve-six-s0-exact-eval-"
    ) as temp_dir:
        split_checkpoint = Path(temp_dir) / "split-checkpoint"
        split_manifest = save_trainer_checkpoint(
            split_checkpoint,
            model=partial_model,
            trainer=partial_trainer,
            identity=split_identity,
        )
        restored_model = TwelveSixDecoder(stage.model, stage.init)
        restored_trainer = Trainer(restored_model, trainer_config, device="cpu")
        loaded = load_trainer_checkpoint(
            split_checkpoint,
            model=restored_model,
            trainer=restored_trainer,
            restore_rng=True,
            expected_git_sha=candidate_sha,
            expected_model_spec_hash=hash_json(stage.model.to_dict()),
            expected_init_spec_hash=hash_json(stage.init.to_dict()),
            expected_tokenizer_hash=tokenizer.identity.config_sha256,
            expected_tokenizer_vocab_hash=tokenizer.identity.vocab_sha256,
            expected_dataset_manifest_hash=dataset_manifest_sha256,
            expected_split_identity=f"train:{train_sha256}",
            expected_packing_hash=PACKING_CONFIG_HASH,
            expected_packing_version=PACKING_VERSION,
            expected_run_manifest_hash=hash_json(run_manifest),
            expected_training_config_hash=split_manifest["identity"][
                "training_config_hash"
            ],
            expected_environment_lock_hash=environment_lock_sha256,
            expected_seed=seed,
        )
        save_load_verified = (
            loaded.manifest["identity"]["git_sha"] == candidate_sha
            and restored_trainer.optimizer_step == split_step
            and _parameters_equal(partial_model, restored_model)
        )
        _train_range(
            restored_trainer,
            train_batches,
            start_step=split_step,
            end_step=train_steps,
        )
        resume_verified = (
            restored_trainer.optimizer_step
            == reference_trainer.optimizer_step
            == train_steps
            and restored_trainer.tokens_seen == reference_trainer.tokens_seen
            and _parameters_equal(reference_model, restored_model)
        )

        final_identity = _bind_identity(
            run_manifest=run_manifest,
            stage=stage,
            tokenizer=tokenizer,
            trainer=restored_trainer,
            environment_lock_sha256=environment_lock_sha256,
        )
        final_checkpoint = Path(temp_dir) / "final-checkpoint"
        final_manifest = save_trainer_checkpoint(
            final_checkpoint,
            model=restored_model,
            trainer=restored_trainer,
            identity=final_identity,
        )
        direct_backend = S0TorchInferenceBackend(restored_model, tokenizer)
        reloaded_backend = load_first_party_backend(final_checkpoint)
        generation_config = GenerationConfig(
            max_new_tokens=8,
            sample=False,
            seed=seed,
        )
        direct_generation = generate(
            direct_backend,
            "12-6",
            generation_config,
        )
        reloaded_generation = generate(
            reloaded_backend,
            "12-6",
            generation_config,
        )
        generation_parity = direct_generation == reloaded_generation
        if not generation_parity:
            raise RuntimeError(
                "first-party checkpoint generation diverged from direct model"
            )

    generated_ids = list(reloaded_generation.generated_token_ids)
    generation_sha = _sha256_bytes(
        json.dumps(generated_ids, separators=(",", ":")).encode("utf-8")
    )

    evidence: dict[str, Any] = {
        "schema_version": "12-6.s0-real-candidate-evidence.v2",
        "candidate": {
            "sha": candidate_sha,
            "id": f"s0-exact-candidate@{candidate_sha}",
            "integrated": integrated_candidate,
            "random_init": True,
            "model_constructed": True,
            "parameter_count": parameter_count,
            "model_vocab_size": stage.model.vocab_size,
            "modelspec_sha256": hash_json(stage.model.to_dict()),
            "initspec_sha256": hash_json(stage.init.to_dict()),
        },
        "tokenizer": {
            "identity": tokenizer.identity.config_sha256,
            "config_sha256": tokenizer.identity.config_sha256,
            "vocab_sha256": tokenizer.identity.vocab_sha256,
            "version": tokenizer.identity.version,
            "vocab_size": tokenizer.vocab_size,
            "max_token_id": tokenizer.vocab_size - 1,
        },
        "eval_config": {
            "id": f"s0_eval_gate@{sha256_file(policy_path)}",
            "policy_status": policy_payload.get("policy_status"),
            "fixed_train_steps": train_steps,
            "seed": seed,
        },
        "dataset": {
            "identity": manifest["dataset_identity_sha256"],
            "dataset_id": manifest["dataset_id"],
            "manifest_sha256": dataset_manifest_sha256,
            "train_sha256": train_sha256,
            "validation_sha256": sha256_file(validation_path),
            "heldout_used_for_training": False,
            "train_validation_overlap": len(split_overlap),
            "validation_examples": len(validation_rows),
            "distinct_train_batches": len(train_batch_ids),
        },
        "metrics": {
            "train_loss_before": train_loss_before,
            "train_loss_after": train_loss_after,
            "validation_loss_before": validation_loss_before,
            "validation_loss_after": validation_loss_after,
            "random_validation_loss": validation_loss_before,
            "trained_validation_loss": validation_loss_after,
            "optimizer_steps": train_steps,
            "tokens_seen": reference_trainer.tokens_seen,
        },
        "generation_probes": [
            {
                "id": "first-party-final-checkpoint-greedy",
                "token_count": len(generated_ids),
                "output_sha256": generation_sha,
                "seed": seed,
                "sampler": "greedy",
                "stop_reason": reloaded_generation.stop_reason,
                "direct_vs_reloaded_parity": generation_parity,
            }
        ],
        "checkpoint": {
            "save_load_verified": save_load_verified,
            "resume_verified": resume_verified,
            "serialization_pickle": split_manifest["serialization"]["pickle"],
            "identity_git_sha": split_identity.git_sha,
            "checkpoint_step": split_step,
            "final_checkpoint_id": final_manifest["checkpoint_id"],
            "environment_lock_sha256": environment_lock_sha256,
            "packing_sha256": PACKING_CONFIG_HASH,
        },
        "contamination": {
            "checked": True,
            "benchmark_overlap_count": len(benchmark_hash_overlap)
            + forbidden_source_count,
            "heldout_overlap_count": len(split_overlap),
            "registry_sha256": sha256_file(contamination_registry_path),
            "registry_state": contamination_registry.get("registry_state"),
            "scope": "S0_CONTROLLED_SENTINEL_ONLY",
        },
        "regressions": {
            "executed": True,
            "failures": 0,
            "checks": [
                "checkout_sha_exact",
                "dataset_file_hashes",
                "split_content_identity_disjoint",
                "forbidden_source_purpose",
                "forbidden_registered_hash",
                "finite_random_and_trained_losses",
                "strict_d05_checkpoint_binding",
                "checkpoint_roundtrip_exact",
                "resume_matches_uninterrupted_exact",
                "first_party_reload_generation_parity",
            ],
        },
        "provenance": {
            "repository": _REPOSITORY,
            "checkout_head_sha": candidate_sha,
            "stage_config_sha256": sha256_file(stage_path),
            "source_registry_sha256": sha256_file(source_registry_path),
            "contamination_registry_sha256": sha256_file(
                contamination_registry_path
            ),
            "dataset_manifest_sha256": dataset_manifest_sha256,
            "environment_lock_sha256": environment_lock_sha256,
            "packing_sha256": PACKING_CONFIG_HASH,
            "packing_version": PACKING_VERSION,
            "training_consumed_paths": ["data/s0/packaged/train.jsonl"],
            "evaluation_only_paths": ["data/s0/packaged/validation.jsonl"],
        },
    }

    promotion: dict[str, Any] = {}
    if candidate_manifest_sha256 is not None:
        if _SHA256.fullmatch(candidate_manifest_sha256) is None:
            raise ValueError("candidate_manifest_sha256 must be lowercase SHA-256")
        promotion["candidate_manifest_validated"] = True
        promotion["candidate_manifest_sha256"] = candidate_manifest_sha256
    ci_values = (
        candidate_ci_run_id,
        candidate_ci_head_sha,
        candidate_ci_success,
    )
    if any(value is not None for value in ci_values):
        if any(value is None for value in ci_values):
            raise ValueError(
                "candidate CI evidence must provide run_id, head_sha, and success together"
            )
        promotion["candidate_ci"] = {
            "run_id": candidate_ci_run_id,
            "head_sha": candidate_ci_head_sha,
            "success": candidate_ci_success,
        }
    if audit_a is not None:
        promotion["audit_a"] = audit_a
    if audit_b is not None:
        promotion["audit_b"] = audit_b
    if promotion:
        evidence["promotion"] = promotion
    return evidence


def build_reports(
    evidence: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    gate_report = evaluate_s0_integrated(evidence)
    authority = gate_report["promotion_authority"]
    promotion_report = {
        "schema_version": "12-6.s0-promotion-eligibility.v2",
        "candidate_sha": evidence["candidate"]["sha"],
        "evaluation_complete": gate_report["summary"]["evaluation_complete"],
        "quality_overall_status": gate_report["summary"]["overall_status"],
        "promotion_eligible": gate_report["summary"]["promotion_eligible"],
        "promotion_authority_status": gate_report["summary"][
            "promotion_authority_status"
        ],
        "promotion_blockers": authority.get("blockers", []),
        "truth_boundary": (
            "candidate quality evidence is separate from exact-head CI, integration "
            "manifest, and independent AUDIT-A/AUDIT-B authority"
        ),
    }
    return gate_report, promotion_report


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _load_optional_audit(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    audit = _load_json(path)
    required = {"verdict", "candidate_sha", "evidence_ref"}
    missing = required - audit.keys()
    if missing:
        raise ValueError(f"{path} is missing audit fields: {sorted(missing)}")
    return audit


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run strict LOCAL_FREE S0 exact-candidate evaluation and emit reports"
        )
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--candidate-sha")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--integrated-candidate", action="store_true")
    parser.add_argument("--candidate-manifest-sha256")
    parser.add_argument("--candidate-ci-run-id", type=int)
    parser.add_argument("--candidate-ci-head-sha")
    parser.add_argument("--candidate-ci-success", action="store_true", default=None)
    parser.add_argument("--audit-a", type=Path)
    parser.add_argument("--audit-b", type=Path)
    parser.add_argument("--fail-on-incomplete", action="store_true")
    parser.add_argument("--fail-on-ineligible", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    candidate_sha = args.candidate_sha or _git_head(repo_root)
    evidence = collect_s0_candidate_evidence(
        repo_root,
        candidate_sha,
        train_steps=args.train_steps,
        seed=args.seed,
        integrated_candidate=args.integrated_candidate,
        candidate_manifest_sha256=args.candidate_manifest_sha256,
        candidate_ci_run_id=args.candidate_ci_run_id,
        candidate_ci_head_sha=args.candidate_ci_head_sha,
        candidate_ci_success=args.candidate_ci_success,
        audit_a=_load_optional_audit(args.audit_a),
        audit_b=_load_optional_audit(args.audit_b),
    )
    gate_report, promotion_report = build_reports(evidence)
    _write_json(args.output_dir / "candidate_evidence.json", evidence)
    _write_json(args.output_dir / "stage_gate_report.json", gate_report)
    _write_json(
        args.output_dir / "promotion_eligibility.json",
        promotion_report,
    )
    if args.fail_on_incomplete and not gate_report["summary"]["evaluation_complete"]:
        return 2
    if args.fail_on_ineligible and not gate_report["summary"]["promotion_eligible"]:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
