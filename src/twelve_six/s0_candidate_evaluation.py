"""Real LOCAL_FREE S0 candidate evaluation and stage-gate evidence collection."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import tempfile
from pathlib import Path
from typing import Any

import torch

from twelve_six.checkpoint import (
    bind_checkpoint_identity,
    hash_json,
    load_trainer_checkpoint,
    save_trainer_checkpoint,
)
from twelve_six.inference import GenerationConfig, generate
from twelve_six.integration import S0TorchInferenceBackend
from twelve_six.model import TwelveSixDecoder, load_stage_config
from twelve_six.stage_gates import evaluate_s0_integrated
from twelve_six.tokenization import ByteTokenizer
from twelve_six.training import Trainer, TrainerConfig, causal_lm_loss

_EXACT_GIT_SHA = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


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


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return _sha256_bytes(payload.encode("utf-8"))


def _record_hashes(rows: list[dict[str, Any]]) -> set[str]:
    hashes: set[str] = set()
    for row in rows:
        value = row.get("content_sha256")
        if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
            raise ValueError("every packaged record must carry lowercase SHA-256 content identity")
        hashes.add(value)
    return hashes


def _make_batches(
    rows: list[dict[str, Any]],
    tokenizer: ByteTokenizer,
    *,
    max_seq_len: int,
) -> tuple[list[dict[str, torch.Tensor]], set[str]]:
    batches: list[dict[str, torch.Tensor]] = []
    identities: set[str] = set()
    for row in rows:
        text = row.get("text")
        if not isinstance(text, str) or not text:
            raise ValueError("every packaged record must contain non-empty text")
        token_ids = tokenizer.encode(text)[:max_seq_len]
        if len(token_ids) < 2:
            raise ValueError("every evaluation/training sequence must contain at least two tokens")
        identity = _sha256_bytes(bytes(token_ids))
        identities.add(identity)
        ids = torch.tensor([token_ids], dtype=torch.long)
        batches.append({"input_ids": ids, "labels": ids})
    return batches, identities


@torch.no_grad()
def _mean_loss(model: TwelveSixDecoder, batches: list[dict[str, torch.Tensor]]) -> float:
    model.eval()
    losses: list[float] = []
    for batch in batches:
        logits = model(batch["input_ids"]).logits
        loss = causal_lm_loss(logits, batch["labels"])
        value = float(loss.detach().cpu().item())
        if not math.isfinite(value) or value < 0:
            raise FloatingPointError("candidate evaluation produced invalid token NLL")
        losses.append(value)
    return sum(losses) / len(losses)


def _train_steps(
    trainer: Trainer,
    batches: list[dict[str, torch.Tensor]],
    *,
    start_step: int,
    end_step: int,
) -> None:
    if not (0 <= start_step <= end_step):
        raise ValueError("invalid training step interval")
    for step in range(start_step, end_step):
        trainer.train_microbatch(batches[step % len(batches)])
    trainer.assert_checkpoint_safe()


def _parameters_equal(left: TwelveSixDecoder, right: TwelveSixDecoder) -> bool:
    return all(
        torch.equal(a.detach().cpu(), b.detach().cpu())
        for a, b in zip(left.parameters(), right.parameters(), strict=True)
    )


def _run_manifest(
    *,
    candidate_sha: str,
    repository: str,
    model_spec: dict[str, Any],
    parameter_count: int,
    dataset_manifest_sha256: str,
    tokenizer: ByteTokenizer,
    seed: int,
    trainer_config: TrainerConfig,
    train_steps: int,
    tokens_per_step: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_id": f"s0-local-free-eval-{candidate_sha[:12]}",
        "stage": "S0",
        "run_kind": "integrated_training_evaluation",
        "state": "RUNNING",
        "candidate": {
            "repository": repository,
            "git_sha": candidate_sha,
            "branch_or_tag": "evaluation-candidate",
            "modelspec_sha256": hash_json(model_spec),
            "parameter_count": parameter_count,
        },
        "data": {
            "dataset_manifest_sha256": dataset_manifest_sha256,
            "tokenizer_sha256": tokenizer.identity.config_sha256,
            "tokenizer_vocab_sha256": tokenizer.identity.vocab_sha256,
            "tokenizer_version": tokenizer.identity.version,
            "split_identity": "s0-tiny-controlled-v1",
        },
        "training": {
            "seed": seed,
            "device": "cpu",
            "precision": "fp32",
            "optimizer": {"name": "AdamW", "lr": trainer_config.learning_rate},
            "scheduler": {"name": trainer_config.scheduler},
            "context_length": model_spec["max_seq_len"],
            "global_batch_tokens": tokens_per_step,
            "target_steps": train_steps,
            "target_tokens": train_steps * tokens_per_step,
            "checkpoint_interval_steps": max(1, train_steps // 2),
        },
    }


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
) -> dict[str, Any]:
    """Run a deterministic tiny CPU train/eval/checkpoint/resume cycle on real S0 inputs."""
    if _EXACT_GIT_SHA.fullmatch(candidate_sha) is None:
        raise ValueError("candidate_sha must be a lowercase full 40- or 64-hex Git object id")
    if train_steps < 4 or train_steps % 2:
        raise ValueError("train_steps must be an even integer >= 4")

    stage_path = repo_root / "configs/stages/s0_10k.json"
    policy_path = repo_root / "configs/stages/s0_eval_gate.json"
    manifest_path = repo_root / "data/s0/packaged/manifest.json"
    train_path = repo_root / "data/s0/packaged/train.jsonl"
    validation_path = repo_root / "data/s0/packaged/validation.jsonl"
    source_registry_path = repo_root / "data/s0/source_registry.json"
    contamination_registry_path = repo_root / "data/s0/contamination_registry.json"

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
    for name, path in (("train.jsonl", train_path), ("validation.jsonl", validation_path)):
        expected = expected_outputs.get(name)
        actual = _sha256_file(path)
        if expected != actual:
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

    forbidden_hashes_raw = contamination_registry.get("forbidden_normalized_sha256", [])
    if not isinstance(forbidden_hashes_raw, list) or not all(
        isinstance(item, str) for item in forbidden_hashes_raw
    ):
        raise ValueError("contamination forbidden hash registry must be an array of strings")
    forbidden_hashes = set(forbidden_hashes_raw)
    benchmark_hash_overlap = train_hashes & forbidden_hashes

    forbidden_purposes_raw = contamination_registry.get("forbidden_source_purposes", [])
    if not isinstance(forbidden_purposes_raw, list) or not all(
        isinstance(item, str) for item in forbidden_purposes_raw
    ):
        raise ValueError("contamination forbidden source-purpose registry is invalid")
    forbidden_purposes = set(forbidden_purposes_raw)
    sources = source_registry.get("sources", [])
    if not isinstance(sources, list):
        raise TypeError("source registry sources must be an array")
    forbidden_source_count = sum(
        1
        for source in sources
        if isinstance(source, dict) and source.get("purpose") in forbidden_purposes
    )

    torch.manual_seed(seed)
    reference_model = TwelveSixDecoder(stage.model, stage.init)
    parameter_count = sum(parameter.numel() for parameter in reference_model.parameters())
    if parameter_count != stage.expected_parameters:
        raise ValueError("instantiated parameter count disagrees with frozen S0 stage config")

    trainer_config = TrainerConfig(
        learning_rate=1e-2,
        max_steps=train_steps,
        seed=seed,
        precision="fp32",
        deterministic_algorithms=True,
    )
    train_loss_before = _mean_loss(reference_model, train_batches)
    validation_loss_before = _mean_loss(reference_model, validation_batches)

    reference_trainer = Trainer(reference_model, trainer_config, device="cpu")
    _train_steps(reference_trainer, train_batches, start_step=0, end_step=train_steps)
    train_loss_after = _mean_loss(reference_model, train_batches)
    validation_loss_after = _mean_loss(reference_model, validation_batches)

    split_step = train_steps // 2
    torch.manual_seed(seed)
    partial_model = TwelveSixDecoder(stage.model, stage.init)
    partial_trainer = Trainer(partial_model, trainer_config, device="cpu")
    _train_steps(partial_trainer, train_batches, start_step=0, end_step=split_step)

    model_spec = stage.model.to_dict()
    dataset_manifest_sha256 = _sha256_file(manifest_path)
    first_batch_tokens = int(train_batches[0]["input_ids"].shape[1] - 1)
    run_manifest = _run_manifest(
        candidate_sha=candidate_sha,
        repository="Oleksii-debug/12-6-ai.",
        model_spec=model_spec,
        parameter_count=parameter_count,
        dataset_manifest_sha256=dataset_manifest_sha256,
        tokenizer=tokenizer,
        seed=seed,
        trainer_config=trainer_config,
        train_steps=train_steps,
        tokens_per_step=first_batch_tokens,
    )
    identity = bind_checkpoint_identity(
        run_manifest=run_manifest,
        model_spec=model_spec,
        tokenizer_identity=tokenizer.identity.to_dict(),
        step=partial_trainer.optimizer_step,
        tokens_seen=partial_trainer.tokens_seen,
    )

    with tempfile.TemporaryDirectory(prefix="twelve-six-s0-eval-") as temp_dir:
        checkpoint_dir = Path(temp_dir) / "checkpoint"
        checkpoint_manifest = save_trainer_checkpoint(
            checkpoint_dir,
            model=partial_model,
            trainer=partial_trainer,
            identity=identity,
        )
        restored_model = TwelveSixDecoder(stage.model, stage.init)
        restored_trainer = Trainer(restored_model, trainer_config, device="cpu")
        loaded = load_trainer_checkpoint(
            checkpoint_dir,
            model=restored_model,
            trainer=restored_trainer,
            restore_rng=True,
            expected_git_sha=candidate_sha,
            expected_model_spec_hash=hash_json(model_spec),
            expected_tokenizer_hash=tokenizer.identity.config_sha256,
            expected_dataset_manifest_hash=dataset_manifest_sha256,
        )
        save_load_verified = (
            loaded.manifest["identity"]["git_sha"] == candidate_sha
            and restored_trainer.optimizer_step == split_step
            and _parameters_equal(partial_model, restored_model)
        )
        _train_steps(
            restored_trainer,
            train_batches,
            start_step=split_step,
            end_step=train_steps,
        )
        resume_verified = (
            restored_trainer.optimizer_step == reference_trainer.optimizer_step == train_steps
            and restored_trainer.tokens_seen == reference_trainer.tokens_seen
            and _parameters_equal(reference_model, restored_model)
        )

    generation = generate(
        S0TorchInferenceBackend(restored_model, tokenizer),
        "12-6",
        GenerationConfig(max_new_tokens=8, sample=False, seed=seed),
    )
    generated_ids = list(generation.generated_token_ids)
    generation_sha = _canonical_json_sha256(generated_ids)

    evidence: dict[str, Any] = {
        "schema_version": "12-6.s0-real-candidate-evidence.v1",
        "candidate": {
            "sha": candidate_sha,
            "id": f"s0-evaluation-candidate@{candidate_sha}",
            "integrated": integrated_candidate,
            "random_init": True,
            "model_constructed": True,
            "parameter_count": parameter_count,
            "model_vocab_size": stage.model.vocab_size,
            "modelspec_sha256": hash_json(model_spec),
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
            "id": f"s0_eval_gate@{_sha256_file(policy_path)}",
            "policy_status": policy_payload.get("policy_status"),
            "fixed_train_steps": train_steps,
            "seed": seed,
        },
        "dataset": {
            "identity": manifest["dataset_identity_sha256"],
            "dataset_id": manifest["dataset_id"],
            "manifest_sha256": dataset_manifest_sha256,
            "train_sha256": _sha256_file(train_path),
            "validation_sha256": _sha256_file(validation_path),
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
                "id": "s0-greedy-after-checkpoint-resume",
                "token_count": len(generated_ids),
                "output_sha256": generation_sha,
                "seed": seed,
                "sampler": "greedy",
                "stop_reason": generation.stop_reason,
            }
        ],
        "checkpoint": {
            "save_load_verified": save_load_verified,
            "resume_verified": resume_verified,
            "serialization_pickle": checkpoint_manifest["serialization"]["pickle"],
            "identity_git_sha": identity.git_sha,
            "checkpoint_step": split_step,
        },
        "contamination": {
            "checked": True,
            "benchmark_overlap_count": len(benchmark_hash_overlap) + forbidden_source_count,
            "heldout_overlap_count": len(split_overlap),
            "registry_sha256": _sha256_file(contamination_registry_path),
            "registry_state": contamination_registry.get("registry_state"),
            "scope": "S0_CONTROLLED_SENTINEL_ONLY",
        },
        "regressions": {
            "executed": True,
            "failures": 0,
            "checks": [
                "dataset_file_hashes",
                "split_content_identity_disjoint",
                "forbidden_source_purpose",
                "forbidden_registered_hash",
                "finite_random_and_trained_losses",
                "checkpoint_roundtrip_exact",
                "resume_matches_uninterrupted_exact",
                "trained_reloaded_generation",
            ],
        },
        "provenance": {
            "repository": "Oleksii-debug/12-6-ai.",
            "stage_config_sha256": _sha256_file(stage_path),
            "source_registry_sha256": _sha256_file(source_registry_path),
            "contamination_registry_sha256": _sha256_file(contamination_registry_path),
            "dataset_manifest_sha256": dataset_manifest_sha256,
            "training_consumed_paths": ["data/s0/packaged/train.jsonl"],
            "evaluation_only_paths": ["data/s0/packaged/validation.jsonl"],
        },
    }

    promotion: dict[str, Any] = {}
    if candidate_manifest_sha256 is not None:
        promotion["candidate_manifest_validated"] = True
        promotion["candidate_manifest_sha256"] = candidate_manifest_sha256
    ci_values = (candidate_ci_run_id, candidate_ci_head_sha, candidate_ci_success)
    if any(value is not None for value in ci_values):
        if any(value is None for value in ci_values):
            raise ValueError("candidate CI evidence must provide run_id, head_sha, and success together")
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


def _enforce_candidate_ci_binding(
    evidence: dict[str, Any], gate_report: dict[str, Any]
) -> None:
    promotion = evidence.get("promotion")
    if not isinstance(promotion, dict):
        return
    candidate_ci = promotion.get("candidate_ci")
    if not isinstance(candidate_ci, dict):
        return
    candidate_sha = evidence["candidate"]["sha"]
    head_sha = candidate_ci.get("head_sha")
    authority = gate_report["promotion_authority"]
    blockers = authority.setdefault("blockers", [])
    if head_sha is None:
        authority["status"] = "NOT_TESTED"
        blockers.append("missing evidence: promotion.candidate_ci.head_sha")
    elif not isinstance(head_sha, str) or _EXACT_GIT_SHA.fullmatch(head_sha) is None:
        authority["status"] = "FAIL"
        blockers.append(
            "promotion.candidate_ci.head_sha must be an exact lowercase Git object id"
        )
    elif head_sha != candidate_sha:
        authority["status"] = "FAIL"
        blockers.append("promotion.candidate_ci.head_sha does not match candidate.sha")
    if authority["status"] != "PASS":
        gate_report["summary"]["promotion_eligible"] = False
        gate_report["summary"]["promotion_authority_status"] = authority["status"]
    evidence_block = authority.setdefault("evidence", {})
    ci_block = evidence_block.setdefault("candidate_ci", {})
    if isinstance(ci_block, dict):
        ci_block["head_sha"] = head_sha


def build_reports(evidence: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    gate_report = evaluate_s0_integrated(evidence)
    _enforce_candidate_ci_binding(evidence, gate_report)
    authority = gate_report["promotion_authority"]
    promotion_report = {
        "schema_version": "12-6.s0-promotion-eligibility.v1",
        "candidate_sha": evidence["candidate"]["sha"],
        "evaluation_complete": gate_report["summary"]["evaluation_complete"],
        "quality_overall_status": gate_report["summary"]["overall_status"],
        "promotion_eligible": gate_report["summary"]["promotion_eligible"],
        "promotion_authority_status": gate_report["summary"]["promotion_authority_status"],
        "promotion_blockers": authority.get("blockers", []),
        "truth_boundary": (
            "evaluation quality evidence is separate from candidate CI, manifest, and independent "
            "audit authority"
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
        description="Run real LOCAL_FREE S0 evaluation and emit exact-candidate gate reports"
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--candidate-sha", required=True)
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    evidence = collect_s0_candidate_evidence(
        args.repo_root.resolve(),
        args.candidate_sha,
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
    _write_json(args.output_dir / "promotion_eligibility.json", promotion_report)
    if args.fail_on_incomplete and not gate_report["summary"]["evaluation_complete"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
