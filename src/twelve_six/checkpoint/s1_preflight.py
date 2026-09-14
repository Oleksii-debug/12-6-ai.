"""D05 checkpoint-v1 engineering preflight for the non-frozen S1 ModelSpec.

This module deliberately separates low-level serialization compatibility from a
canonical S1 checkpoint claim. S1 does not yet have a selected tokenizer/data
contract, so the strict canonical run binder must reject reuse of the S0 byte
vocabulary as if it were the S1 tokenizer. The low-level D05 serializer can
still be exercised with an explicitly non-canonical controlled fixture.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch

from twelve_six.checkpoint import (
    CheckpointCompatibilityError,
    CheckpointIdentity,
    bind_checkpoint_identity,
    hash_json,
    load_trainer_checkpoint,
    save_trainer_checkpoint,
    sha256_file,
)
from twelve_six.model import TwelveSixDecoder, load_stage_config
from twelve_six.packing import PACKING_CONFIG_HASH, PACKING_VERSION
from twelve_six.tokenization import ByteTokenizer
from twelve_six.training import Trainer, TrainerConfig

SCHEMA = "12-6.s1-checkpoint-preflight.v1"
REPOSITORY = "Oleksii-debug/12-6-ai."
AUTHORITY = "ENGINEERING_CHECKPOINT_PREFLIGHT_ONLY_NOT_STAGE_EVIDENCE"
FIXTURE_SCOPE = "S0_CONTROLLED_FIXTURE_COMPATIBILITY_ONLY_NOT_S1_DATA_OR_TOKENIZER"


def _exact_git_sha(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) in {40, 64}
        and value == value.lower()
        and all(ch in "0123456789abcdef" for ch in value)
    )


def _git_head(repo_root: Path) -> str:
    try:
        value = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("S1 checkpoint preflight requires a Git checkout") from exc
    if not _exact_git_sha(value):
        raise ValueError("git HEAD is not an exact lowercase Git object id")
    return value


def _load_texts(path: Path) -> list[str]:
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
        raise ValueError("controlled fixture contains no training text")
    return texts


def _batch_for_step(
    texts: Sequence[str],
    tokenizer: ByteTokenizer,
    *,
    step: int,
    max_seq_len: int,
) -> dict[str, torch.Tensor]:
    token_ids = tokenizer.encode(texts[step % len(texts)])[:max_seq_len]
    if len(token_ids) < 2:
        raise ValueError("controlled fixture record must encode to at least two tokens")
    ids = torch.tensor([token_ids], dtype=torch.long)
    return {"input_ids": ids, "labels": ids}


def _train_range(
    trainer: Trainer,
    texts: Sequence[str],
    tokenizer: ByteTokenizer,
    *,
    start_step: int,
    end_step: int,
    max_seq_len: int,
) -> None:
    for step in range(start_step, end_step):
        metrics = trainer.train_microbatch(
            _batch_for_step(
                texts,
                tokenizer,
                step=step,
                max_seq_len=max_seq_len,
            )
        )
        if not metrics.optimizer_stepped:
            raise RuntimeError("S1 checkpoint preflight requires committed optimizer steps")
    trainer.assert_checkpoint_safe()


def _nested_equal(left: Any, right: Any) -> bool:
    if is_dataclass(left) and not isinstance(left, type):
        left = asdict(left)
    if is_dataclass(right) and not isinstance(right, type):
        right = asdict(right)
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor) and torch.equal(
            left, right
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            return False
        return left.keys() == right.keys() and all(
            _nested_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if not isinstance(left, (list, tuple)) or not isinstance(right, (list, tuple)):
            return False
        return len(left) == len(right) and all(
            _nested_equal(l_item, r_item)
            for l_item, r_item in zip(left, right, strict=True)
        )
    return left == right


def _models_equal(left: TwelveSixDecoder, right: TwelveSixDecoder) -> bool:
    left_state = left.state_dict()
    right_state = right.state_dict()
    return left_state.keys() == right_state.keys() and all(
        torch.equal(left_state[key], right_state[key]) for key in left_state
    )


def _canonical_binding_rejection(
    *,
    candidate_sha: str,
    stage: Any,
    tokenizer: ByteTokenizer,
    dataset_manifest_sha256: str,
    train_sha256: str,
    environment_lock_sha256: str,
    trainer_config: TrainerConfig,
) -> str:
    """Require the strict canonical binder to reject S0 tokenizer reuse for S1."""

    run_manifest = {
        "schema_version": 1,
        "run_id": f"s1-d05-canonical-binding-negative-{candidate_sha[:12]}",
        "stage": "S1",
        "run_kind": "engineering_checkpoint_preflight",
        "state": "PREPARED_NOT_CANONICAL",
        "candidate": {
            "repository": REPOSITORY,
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
            "split_identity": f"controlled-s0-train:{train_sha256}",
            "packing_sha256": PACKING_CONFIG_HASH,
            "packing_version": PACKING_VERSION,
        },
        "training": {
            "seed": trainer_config.seed,
            "precision": trainer_config.precision,
            "optimizer": {"name": "AdamW", "lr": trainer_config.learning_rate},
            "scheduler": {"name": trainer_config.scheduler},
        },
        "environment": {"lock_sha256": environment_lock_sha256},
    }
    try:
        bind_checkpoint_identity(
            run_manifest=run_manifest,
            model_spec=stage.model.to_dict(),
            init_spec=stage.init.to_dict(),
            tokenizer_identity=tokenizer.identity.to_dict(),
            packing_identity={
                "version": PACKING_VERSION,
                "config_sha256": PACKING_CONFIG_HASH,
            },
            step=0,
            tokens_seen=0,
            environment_lock_hash=environment_lock_sha256,
        )
    except CheckpointCompatibilityError as exc:
        reason = str(exc)
        if "ModelSpec/tokenizer vocab mismatch" not in reason:
            raise RuntimeError(
                "canonical S1 binding failed for an unexpected reason; "
                "expected the unresolved tokenizer-vocabulary boundary"
            ) from exc
        return reason
    raise RuntimeError(
        "canonical binder accepted S0 byte tokenizer as S1 canonical identity; fail closed"
    )


def _preflight_identity(
    *,
    candidate_sha: str,
    stage: Any,
    tokenizer: ByteTokenizer,
    dataset_manifest_sha256: str,
    train_sha256: str,
    environment_lock_sha256: str,
    trainer: Trainer,
) -> CheckpointIdentity:
    init_sha256 = hash_json(stage.init.to_dict())
    preflight_manifest = {
        "schema": SCHEMA,
        "authority": AUTHORITY,
        "repository": REPOSITORY,
        "candidate_sha": candidate_sha,
        "stage": "S1",
        "s1_architecture_status": "ENGINEERING_CANDIDATE_NOT_FROZEN",
        "s1_tokenizer_selected": False,
        "s1_data_selected": False,
        "fixture_scope": FIXTURE_SCOPE,
        "model_spec_sha256": hash_json(stage.model.to_dict()),
        "init_spec_sha256": init_sha256,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "tokenizer_config_sha256": tokenizer.identity.config_sha256,
        "tokenizer_vocab_sha256": tokenizer.identity.vocab_sha256,
        "packing_sha256": PACKING_CONFIG_HASH,
        "environment_lock_sha256": environment_lock_sha256,
        "seed": trainer.config.seed,
        "step": trainer.optimizer_step,
        "tokens_seen": trainer.tokens_seen,
    }
    training_config = {
        "authority": AUTHORITY,
        "stage": "S1",
        "s1_tokenizer_selected": False,
        "s1_data_selected": False,
        "fixture_scope": FIXTURE_SCOPE,
        "init_spec_sha256": init_sha256,
        "training": {
            "seed": trainer.config.seed,
            "precision": trainer.config.precision,
            "max_steps": trainer.config.max_steps,
            "context_length": stage.model.max_seq_len,
        },
        "data": {
            "dataset_manifest_sha256": dataset_manifest_sha256,
            "split_identity": f"controlled-s0-train:{train_sha256}",
            "tokenizer_sha256": tokenizer.identity.config_sha256,
            "tokenizer_vocab_sha256": tokenizer.identity.vocab_sha256,
            "tokenizer_version": tokenizer.identity.version,
            "packing_sha256": PACKING_CONFIG_HASH,
            "packing_version": PACKING_VERSION,
        },
        "environment": {"lock_sha256": environment_lock_sha256},
    }
    return CheckpointIdentity(
        git_sha=candidate_sha,
        model_spec=stage.model.to_dict(),
        parameter_count=stage.expected_parameters,
        tokenizer_hash=tokenizer.identity.config_sha256,
        tokenizer_vocab_hash=tokenizer.identity.vocab_sha256,
        dataset_manifest_hash=dataset_manifest_sha256,
        run_manifest_hash=hash_json(preflight_manifest),
        training_config=training_config,
        seed=trainer.config.seed,
        precision=trainer.config.precision,
        step=trainer.optimizer_step,
        tokens_seen=trainer.tokens_seen,
        optimizer={
            "name": "AdamW",
            "lr": trainer.config.learning_rate,
            "betas": list(trainer.config.betas),
            "eps": trainer.config.eps,
            "weight_decay": trainer.config.weight_decay,
        },
        scheduler={"name": trainer.config.scheduler},
        environment_lock_hash=environment_lock_sha256,
    )


def _evidence_hash(payload: Mapping[str, Any]) -> str:
    material = dict(payload)
    material.pop("evidence_sha256", None)
    return hash_json(material)


def validate_s1_checkpoint_preflight(
    payload: Mapping[str, Any],
    *,
    expected_candidate_sha: str | None = None,
) -> None:
    if payload.get("schema") != SCHEMA:
        raise ValueError("unexpected S1 checkpoint preflight schema")
    if payload.get("authority") != AUTHORITY:
        raise ValueError("S1 checkpoint preflight authority was weakened")
    if payload.get("repository") != REPOSITORY:
        raise ValueError("S1 checkpoint preflight repository mismatch")
    candidate_sha = payload.get("candidate_sha")
    if not _exact_git_sha(candidate_sha):
        raise ValueError("S1 checkpoint preflight candidate SHA is invalid")
    if expected_candidate_sha is not None and candidate_sha != expected_candidate_sha:
        raise ValueError("S1 checkpoint preflight candidate SHA is stale")
    if payload.get("s1_architecture_status") != "ENGINEERING_CANDIDATE_NOT_FROZEN":
        raise ValueError("S1 architecture preflight must remain explicitly non-frozen")
    if payload.get("s1_tokenizer_selected") is not False:
        raise ValueError("S1 checkpoint preflight must not select an S1 tokenizer")
    if payload.get("s1_data_selected") is not False:
        raise ValueError("S1 checkpoint preflight must not select S1 training data")
    if payload.get("fixture_scope") != FIXTURE_SCOPE:
        raise ValueError("controlled fixture scope was weakened")

    canonical = payload.get("canonical_binding")
    checkpoint = payload.get("checkpoint")
    resume = payload.get("resume")
    constraints = payload.get("constraints")
    if not all(isinstance(section, Mapping) for section in (canonical, checkpoint, resume, constraints)):
        raise TypeError("S1 checkpoint preflight sections must be objects")
    if canonical.get("accepted") is not False or canonical.get("rejected_as_expected") is not True:
        raise ValueError("canonical S1 binding did not fail closed")
    if "ModelSpec/tokenizer vocab mismatch" not in str(canonical.get("reason", "")):
        raise ValueError("canonical binding rejection is not tied to S1 tokenizer-vocab gap")
    if checkpoint.get("save_verified") is not True or checkpoint.get("pickle") is not False:
        raise ValueError("checkpoint-v1 low-level S1 preflight did not verify safely")
    if resume.get("model_state_exact") is not True or resume.get("trainer_state_exact") is not True:
        raise ValueError("S1 interrupted/resumed preflight is not exact")
    if constraints.get("paid_compute") is not False:
        raise ValueError("S1 checkpoint preflight must remain LOCAL_FREE")
    if constraints.get("promotion_claimed") is not False:
        raise ValueError("S1 checkpoint preflight cannot grant promotion")
    if constraints.get("s1_quality_claimed") is not False:
        raise ValueError("S1 checkpoint preflight cannot claim S1 quality")
    if payload.get("evidence_sha256") != _evidence_hash(payload):
        raise ValueError("S1 checkpoint preflight evidence self-hash mismatch")


def collect_s1_checkpoint_preflight(
    repo_root: Path,
    candidate_sha: str,
    output_dir: Path,
    *,
    total_steps: int = 4,
    split_step: int = 2,
    seed: int = 20260825,
    verify_checkout: bool = True,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    output_dir = output_dir.resolve()
    if not _exact_git_sha(candidate_sha):
        raise ValueError("candidate_sha must be a full lowercase Git object id")
    if verify_checkout and _git_head(repo_root) != candidate_sha:
        raise ValueError("candidate_sha does not equal checkout HEAD")
    if not isinstance(total_steps, int) or isinstance(total_steps, bool) or total_steps < 2:
        raise ValueError("total_steps must be an integer >= 2")
    if not isinstance(split_step, int) or isinstance(split_step, bool) or not 0 < split_step < total_steps:
        raise ValueError("split_step must be strictly between 0 and total_steps")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    stage = load_stage_config(repo_root / "configs/stages/s1_100k.json")
    tokenizer = ByteTokenizer()
    if stage.stage != "S1" or stage.canonical_base != "random_init":
        raise ValueError("unexpected S1 stage contract")
    if stage.expected_parameters != 107_856:
        raise ValueError("unexpected current S1 engineering parameter count")
    if stage.model.vocab_size != 512 or tokenizer.vocab_size != 256:
        raise ValueError("unexpected current S1/S0 tokenizer compatibility boundary")

    train_path = repo_root / "data/s0/packaged/train.jsonl"
    dataset_manifest_path = repo_root / "data/s0/packaged/manifest.json"
    environment_lock_path = repo_root / "requirements/locks/index.json"
    texts = _load_texts(train_path)
    dataset_manifest_sha256 = sha256_file(dataset_manifest_path)
    train_sha256 = sha256_file(train_path)
    environment_lock_sha256 = sha256_file(environment_lock_path)

    trainer_config = TrainerConfig(
        learning_rate=1e-3,
        weight_decay=0.0,
        max_steps=total_steps,
        scheduler="constant",
        gradient_accumulation_steps=1,
        gradient_clip_norm=1.0,
        precision="fp32",
        seed=seed,
        deterministic_algorithms=True,
    )
    canonical_rejection = _canonical_binding_rejection(
        candidate_sha=candidate_sha,
        stage=stage,
        tokenizer=tokenizer,
        dataset_manifest_sha256=dataset_manifest_sha256,
        train_sha256=train_sha256,
        environment_lock_sha256=environment_lock_sha256,
        trainer_config=trainer_config,
    )

    random.seed(seed)
    torch.manual_seed(seed)
    baseline_model = TwelveSixDecoder(stage.model, stage.init)
    baseline_trainer = Trainer(baseline_model, trainer_config, device="cpu")
    _train_range(
        baseline_trainer,
        texts,
        tokenizer,
        start_step=0,
        end_step=total_steps,
        max_seq_len=stage.model.max_seq_len,
    )

    random.seed(seed)
    torch.manual_seed(seed)
    partial_model = TwelveSixDecoder(stage.model, stage.init)
    partial_trainer = Trainer(partial_model, trainer_config, device="cpu")
    _train_range(
        partial_trainer,
        texts,
        tokenizer,
        start_step=0,
        end_step=split_step,
        max_seq_len=stage.model.max_seq_len,
    )

    identity = _preflight_identity(
        candidate_sha=candidate_sha,
        stage=stage,
        tokenizer=tokenizer,
        dataset_manifest_sha256=dataset_manifest_sha256,
        train_sha256=train_sha256,
        environment_lock_sha256=environment_lock_sha256,
        trainer=partial_trainer,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoint"
    manifest = save_trainer_checkpoint(
        checkpoint_dir,
        model=partial_model,
        trainer=partial_trainer,
        identity=identity,
    )

    restored_model = TwelveSixDecoder(stage.model, stage.init)
    restored_trainer = Trainer(restored_model, trainer_config, device="cpu")
    load_trainer_checkpoint(
        checkpoint_dir,
        model=restored_model,
        trainer=restored_trainer,
        restore_rng=True,
        expected_git_sha=candidate_sha,
        expected_model_spec_hash=hash_json(stage.model.to_dict()),
        expected_init_spec_hash=hash_json(stage.init.to_dict()),
        expected_tokenizer_hash=tokenizer.identity.config_sha256,
        expected_tokenizer_vocab_hash=tokenizer.identity.vocab_sha256,
        expected_dataset_manifest_hash=dataset_manifest_sha256,
        expected_split_identity=f"controlled-s0-train:{train_sha256}",
        expected_packing_hash=PACKING_CONFIG_HASH,
        expected_packing_version=PACKING_VERSION,
        expected_run_manifest_hash=identity.run_manifest_hash,
        expected_training_config_hash=hash_json(identity.training_config),
        expected_environment_lock_hash=environment_lock_sha256,
        expected_seed=seed,
    )
    _train_range(
        restored_trainer,
        texts,
        tokenizer,
        start_step=split_step,
        end_step=total_steps,
        max_seq_len=stage.model.max_seq_len,
    )

    model_state_exact = _models_equal(baseline_model, restored_model)
    trainer_state_exact = _nested_equal(
        baseline_trainer.state_dict(), restored_trainer.state_dict()
    )
    if not model_state_exact or not trainer_state_exact:
        raise RuntimeError("S1 low-level checkpoint resume diverged from uninterrupted control")

    evidence: dict[str, Any] = {
        "schema": SCHEMA,
        "authority": AUTHORITY,
        "repository": REPOSITORY,
        "candidate_sha": candidate_sha,
        "stage": "S1",
        "s1_architecture_status": "ENGINEERING_CANDIDATE_NOT_FROZEN",
        "s1_tokenizer_selected": False,
        "s1_data_selected": False,
        "fixture_scope": FIXTURE_SCOPE,
        "model": {
            "model_spec_sha256": hash_json(stage.model.to_dict()),
            "init_spec_sha256": hash_json(stage.init.to_dict()),
            "parameter_count": stage.expected_parameters,
            "model_vocab_size": stage.model.vocab_size,
            "max_context_tokens": stage.model.max_seq_len,
        },
        "fixture": {
            "tokenizer_version": tokenizer.identity.version,
            "tokenizer_vocab_size": tokenizer.vocab_size,
            "tokenizer_config_sha256": tokenizer.identity.config_sha256,
            "tokenizer_vocab_sha256": tokenizer.identity.vocab_sha256,
            "dataset_manifest_sha256": dataset_manifest_sha256,
            "train_split_sha256": train_sha256,
            "packing_sha256": PACKING_CONFIG_HASH,
            "packing_version": PACKING_VERSION,
            "environment_lock_sha256": environment_lock_sha256,
        },
        "canonical_binding": {
            "accepted": False,
            "rejected_as_expected": True,
            "reason": canonical_rejection,
        },
        "checkpoint": {
            "save_verified": True,
            "format": manifest["format"],
            "format_version": manifest["format_version"],
            "checkpoint_id": manifest["checkpoint_id"],
            "pickle": manifest["serialization"]["pickle"],
            "step": identity.step,
            "tokens_seen": identity.tokens_seen,
            "retained_directory": "checkpoint",
        },
        "resume": {
            "split_step": split_step,
            "final_step": total_steps,
            "model_state_exact": model_state_exact,
            "trainer_state_exact": trainer_state_exact,
            "baseline_tokens_seen": baseline_trainer.tokens_seen,
            "resumed_tokens_seen": restored_trainer.tokens_seen,
        },
        "constraints": {
            "paid_compute": False,
            "promotion_claimed": False,
            "s1_quality_claimed": False,
            "s1_corpus_claimed": False,
            "s1_tokenizer_claimed": False,
            "foreign_pretrained_weights": False,
            "instruction_alignment_training": False,
        },
    }
    evidence["evidence_sha256"] = _evidence_hash(evidence)
    validate_s1_checkpoint_preflight(evidence, expected_candidate_sha=candidate_sha)
    (output_dir / "s1-checkpoint-preflight.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return evidence


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prove checkpoint-v1 S1-shaped save/load/resume mechanics while "
            "failing closed against premature canonical S1 tokenizer/data binding."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--candidate-sha", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--total-steps", type=int, default=4)
    parser.add_argument("--split-step", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260825)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    evidence = collect_s1_checkpoint_preflight(
        args.repo_root,
        args.candidate_sha,
        args.output_dir,
        total_steps=args.total_steps,
        split_step=args.split_step,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "schema": evidence["schema"],
                "authority": evidence["authority"],
                "candidate_sha": evidence["candidate_sha"],
                "checkpoint_id": evidence["checkpoint"]["checkpoint_id"],
                "canonical_binding": evidence["canonical_binding"],
                "resume": evidence["resume"],
                "evidence_sha256": evidence["evidence_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
