"""Cross-architecture checkpoint portability evidence for canonical S0.

The portability layer composes the existing D01/D02/D03/D04/D05/D07 contracts.
It does not define a new checkpoint format and does not claim cross-architecture
training or inference bitwise reproducibility.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from itertools import cycle, islice
from pathlib import Path
from typing import Any

import torch

from twelve_six.checkpoint import (
    bind_checkpoint_identity,
    hash_json,
    load_trainer_checkpoint,
    save_trainer_checkpoint,
    sha256_file,
    verify_checkpoint,
)
from twelve_six.inference.first_party import load_first_party_backend
from twelve_six.model import TwelveSixDecoder, load_stage_config
from twelve_six.packing import PACKING_CONFIG_HASH, PACKING_VERSION
from twelve_six.tokenization import ByteTokenizer
from twelve_six.training import Trainer, TrainerConfig
from twelve_six.training.s0_evidence import (
    DATASET_MANIFEST_SHA256,
    TRAIN_JSONL_SHA256,
    _tensor_batches,
)

PRODUCER_SCHEMA = "12-6.checkpoint-portability-producer.v1"
CONSUMER_SCHEMA = "12-6.checkpoint-portability-consumer.v1"
AUTHORITY = "LOCAL_FREE_CROSS_ARCH_SERIALIZATION_EVIDENCE_NOT_REPRODUCIBILITY"
REPOSITORY = "Oleksii-debug/12-6-ai."
MODEL_SPEC_SHA256 = "86c75b31dff05b7b5db9f6ed068c571a6ead01ba663412fe630f5e52b09d9b6b"
INIT_SPEC_SHA256 = "86483c6df623e80cab2f73aba718863fce18af6fe3b12430c1348414d92b48a5"
TOKENIZER_CONFIG_SHA256 = "b04055c1061dd641dcab7cb9d62a931f09b8d1a070140a926ceb4e91d73ca8e1"
TOKENIZER_VOCAB_SHA256 = "905ed40bb42cc4d550e228ff5f24158d504b38e8ed5974dfa3077bd5867ad571"
PARAMETER_COUNT = 10_140
SEED = 1337
_SHA256_HEX = frozenset("0123456789abcdef")


class CheckpointPortabilityError(ValueError):
    """Raised when checkpoint portability evidence fails closed."""


def _canonical_hash(value: dict[str, Any]) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _validate_source_sha(value: Any) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or value != value.lower()
        or set(value) - _SHA256_HEX
    ):
        raise CheckpointPortabilityError(
            "source SHA must be a full lowercase 40-hex Git SHA"
        )
    return value


def _normalized_architecture(value: str | None = None) -> str:
    machine = (value or platform.machine()).strip().lower().replace("-", "_")
    if machine in {"x86_64", "amd64"}:
        return "x86_64"
    if machine in {"aarch64", "arm64"}:
        return "aarch64"
    return machine


def _git_head(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise CheckpointPortabilityError(
            "checkpoint portability requires a Git checkout"
        ) from exc


def _trainer_config() -> TrainerConfig:
    return TrainerConfig(
        learning_rate=3e-2,
        weight_decay=0.0,
        max_steps=1,
        scheduler="constant",
        gradient_accumulation_steps=1,
        gradient_clip_norm=1.0,
        precision="fp32",
        seed=SEED,
        deterministic_algorithms=True,
        deterministic_warn_only=False,
    )


def _run_manifest(
    *,
    source_sha: str,
    stage: Any,
    tokenizer: ByteTokenizer,
    environment_lock_sha256: str,
) -> dict[str, Any]:
    config = _trainer_config()
    return {
        "schema_version": 1,
        "run_id": f"s0-d05-portability-{source_sha[:12]}",
        "stage": "S0",
        "run_kind": "checkpoint_cross_arch_portability_probe",
        "state": "EVIDENCE_ONLY",
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
            "seed": config.seed,
            "device": "cpu",
            "precision": "fp32",
            "optimizer": {
                "name": "AdamW",
                "lr": config.learning_rate,
                "betas": list(config.betas),
                "eps": config.eps,
                "weight_decay": config.weight_decay,
            },
            "scheduler": {"name": config.scheduler},
            "context_length": stage.model.max_seq_len,
            "target_steps": 1,
        },
        "environment": {"lock_sha256": environment_lock_sha256},
    }


def _checkpoint_file_hashes(checkpoint: Path) -> dict[str, str]:
    return {
        path.name: sha256_file(path)
        for path in sorted(checkpoint.iterdir())
        if path.is_file()
    }


def _validate_canonical_inputs(root: Path) -> tuple[Any, ByteTokenizer, str]:
    stage = load_stage_config(root / "configs/stages/s0_10k.json")
    tokenizer = ByteTokenizer()
    if stage.canonical_base != "random_init":
        raise CheckpointPortabilityError("canonical S0 Base must remain random_init")
    if stage.expected_parameters != PARAMETER_COUNT:
        raise CheckpointPortabilityError("S0 parameter count drift")
    if stage.model.identity_sha256() != MODEL_SPEC_SHA256:
        raise CheckpointPortabilityError("S0 ModelSpec identity drift")
    if stage.init.identity_sha256() != INIT_SPEC_SHA256:
        raise CheckpointPortabilityError("S0 InitSpec identity drift")
    if tokenizer.identity.config_sha256 != TOKENIZER_CONFIG_SHA256:
        raise CheckpointPortabilityError("S0 tokenizer config identity drift")
    if tokenizer.identity.vocab_sha256 != TOKENIZER_VOCAB_SHA256:
        raise CheckpointPortabilityError("S0 tokenizer vocabulary identity drift")
    if sha256_file(root / "data/s0/packaged/manifest.json") != DATASET_MANIFEST_SHA256:
        raise CheckpointPortabilityError("D03 dataset manifest identity drift")
    if sha256_file(root / "data/s0/packaged/train.jsonl") != TRAIN_JSONL_SHA256:
        raise CheckpointPortabilityError("D03 train split identity drift")
    environment_lock_sha256 = sha256_file(root / "requirements/locks/index.json")
    return stage, tokenizer, environment_lock_sha256


def produce_checkpoint_portability_bundle(
    root: str | Path,
    *,
    source_sha: str,
    output_dir: str | Path,
    verify_checkout: bool = True,
    require_architecture: str | None = "x86_64",
) -> dict[str, Any]:
    """Create one real one-step S0 checkpoint and producer portability evidence."""

    source_sha = _validate_source_sha(source_sha)
    root = Path(root).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise CheckpointPortabilityError("output_dir must not already exist")
    if verify_checkout and _git_head(root) != source_sha:
        raise CheckpointPortabilityError("source SHA does not match checkout HEAD")

    architecture = _normalized_architecture()
    if (
        require_architecture is not None
        and architecture != _normalized_architecture(require_architecture)
    ):
        raise CheckpointPortabilityError(
            f"producer architecture mismatch: expected={require_architecture} "
            f"actual={architecture}"
        )

    stage, tokenizer, environment_lock_sha256 = _validate_canonical_inputs(root)
    train_batches, _, _ = _tensor_batches(
        root,
        split="train",
        tokenizer=tokenizer,
        batch_size=3,
    )
    config = _trainer_config()
    torch.manual_seed(SEED)
    model = TwelveSixDecoder(stage.model, stage.init)
    trainer = Trainer(model, config, device="cpu")
    result = trainer.run(islice(cycle(train_batches), 1))
    if result.optimizer_steps_completed != 1 or trainer.optimizer_step != 1:
        raise CheckpointPortabilityError("producer did not commit exactly one step")
    trainer.assert_checkpoint_safe()

    run_manifest = _run_manifest(
        source_sha=source_sha,
        stage=stage,
        tokenizer=tokenizer,
        environment_lock_sha256=environment_lock_sha256,
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

    output_dir.mkdir(parents=True)
    checkpoint = output_dir / "checkpoint"
    manifest = save_trainer_checkpoint(
        checkpoint,
        model=model,
        trainer=trainer,
        identity=identity,
    )
    verified = verify_checkpoint(checkpoint)
    if verified["checkpoint_id"] != manifest["checkpoint_id"]:
        raise CheckpointPortabilityError("producer checkpoint identity changed after save")

    report: dict[str, Any] = {
        "schema_version": PRODUCER_SCHEMA,
        "authority": AUTHORITY,
        "source": {
            "repository": REPOSITORY,
            "source_sha": source_sha,
            "architecture": architecture,
            "byteorder": sys.byteorder,
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "identity": {
            "modelspec_sha256": MODEL_SPEC_SHA256,
            "initspec_sha256": INIT_SPEC_SHA256,
            "parameter_count": PARAMETER_COUNT,
            "tokenizer_config_sha256": TOKENIZER_CONFIG_SHA256,
            "tokenizer_vocab_sha256": TOKENIZER_VOCAB_SHA256,
            "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
            "packing_config_sha256": PACKING_CONFIG_HASH,
            "environment_lock_sha256": environment_lock_sha256,
            "run_manifest_sha256": hash_json(run_manifest),
            "training_config_sha256": verified["identity"]["training_config_hash"],
            "seed": SEED,
            "optimizer_step": trainer.optimizer_step,
            "tokens_seen": trainer.tokens_seen,
        },
        "checkpoint": {
            "checkpoint_id": manifest["checkpoint_id"],
            "relative_path": "checkpoint",
            "artifact_sha256": _checkpoint_file_hashes(checkpoint),
            "verified_on_producer": True,
        },
        "run_manifest": run_manifest,
        "claims": {
            "serialization_portability_only": True,
            "cross_arch_training_bitwise_reproducibility": False,
            "cross_arch_inference_bitwise_reproducibility": False,
            "rng_cross_arch_equivalence": False,
            "foreign_pretrained_weights_used": False,
            "instruction_or_alignment_behavior_added": False,
            "paid_compute_authorized_or_used": False,
            "candidate_or_stable_promotion": False,
            "audit_verdict": False,
        },
    }
    report["report_sha256"] = _canonical_hash(report)
    (output_dir / "producer.json").write_text(
        json.dumps(report, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    validate_producer_report(report)
    return report


def validate_producer_report(report: dict[str, Any]) -> dict[str, Any]:
    if report.get("schema_version") != PRODUCER_SCHEMA:
        raise CheckpointPortabilityError("unexpected producer schema")
    expected_hash = report.get("report_sha256")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise CheckpointPortabilityError("producer report hash is missing")
    unhashed = dict(report)
    del unhashed["report_sha256"]
    if _canonical_hash(unhashed) != expected_hash:
        raise CheckpointPortabilityError("producer report self-hash mismatch")

    source = report.get("source")
    identity = report.get("identity")
    checkpoint = report.get("checkpoint")
    claims = report.get("claims")
    if not isinstance(source, dict) or source.get("repository") != REPOSITORY:
        raise CheckpointPortabilityError("producer repository identity mismatch")
    _validate_source_sha(source.get("source_sha"))
    if not isinstance(identity, dict) or not isinstance(checkpoint, dict):
        raise CheckpointPortabilityError("producer identity/checkpoint evidence is missing")
    required_identity = {
        "modelspec_sha256": MODEL_SPEC_SHA256,
        "initspec_sha256": INIT_SPEC_SHA256,
        "parameter_count": PARAMETER_COUNT,
        "tokenizer_config_sha256": TOKENIZER_CONFIG_SHA256,
        "tokenizer_vocab_sha256": TOKENIZER_VOCAB_SHA256,
        "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
        "packing_config_sha256": PACKING_CONFIG_HASH,
        "seed": SEED,
        "optimizer_step": 1,
    }
    for key, expected in required_identity.items():
        if identity.get(key) != expected:
            raise CheckpointPortabilityError(f"producer identity mismatch for {key}")
    if checkpoint.get("verified_on_producer") is not True:
        raise CheckpointPortabilityError("producer checkpoint was not verified")
    artifact_hashes = checkpoint.get("artifact_sha256")
    if not isinstance(artifact_hashes, dict) or len(artifact_hashes) != 5:
        raise CheckpointPortabilityError("producer checkpoint inventory must have five files")
    if not isinstance(claims, dict):
        raise CheckpointPortabilityError("producer truth boundary is missing")
    if claims.get("serialization_portability_only") is not True:
        raise CheckpointPortabilityError("producer portability scope was weakened")
    for key in (
        "cross_arch_training_bitwise_reproducibility",
        "cross_arch_inference_bitwise_reproducibility",
        "rng_cross_arch_equivalence",
        "foreign_pretrained_weights_used",
        "instruction_or_alignment_behavior_added",
        "paid_compute_authorized_or_used",
        "candidate_or_stable_promotion",
        "audit_verdict",
    ):
        if claims.get(key) is not False:
            raise CheckpointPortabilityError(f"producer claim must remain false: {key}")
    return {
        "status": "PASS",
        "source_sha": source["source_sha"],
        "architecture": source["architecture"],
        "checkpoint_id": checkpoint["checkpoint_id"],
        "report_sha256": expected_hash,
    }


def consume_checkpoint_portability_bundle(
    root: str | Path,
    *,
    source_sha: str,
    bundle_dir: str | Path,
    output: str | Path,
    verify_checkout: bool = True,
    require_architecture: str | None = "aarch64",
) -> dict[str, Any]:
    """Verify and restore the exact producer checkpoint on a second architecture."""

    source_sha = _validate_source_sha(source_sha)
    root = Path(root).resolve()
    bundle_dir = Path(bundle_dir).resolve()
    output = Path(output).resolve()
    if verify_checkout and _git_head(root) != source_sha:
        raise CheckpointPortabilityError("source SHA does not match checkout HEAD")

    producer_path = bundle_dir / "producer.json"
    producer = json.loads(producer_path.read_text(encoding="utf-8"))
    if not isinstance(producer, dict):
        raise CheckpointPortabilityError("producer.json must contain an object")
    validate_producer_report(producer)
    if producer["source"]["source_sha"] != source_sha:
        raise CheckpointPortabilityError("producer source SHA differs from consumer checkout")

    architecture = _normalized_architecture()
    if (
        require_architecture is not None
        and architecture != _normalized_architecture(require_architecture)
    ):
        raise CheckpointPortabilityError(
            f"consumer architecture mismatch: expected={require_architecture} "
            f"actual={architecture}"
        )
    if architecture == producer["source"]["architecture"]:
        raise CheckpointPortabilityError(
            "consumer architecture equals producer; cross-architecture proof is absent"
        )

    stage, tokenizer, environment_lock_sha256 = _validate_canonical_inputs(root)
    checkpoint = bundle_dir / producer["checkpoint"]["relative_path"]
    if not checkpoint.is_dir():
        raise CheckpointPortabilityError("producer checkpoint directory is missing")
    current_hashes = _checkpoint_file_hashes(checkpoint)
    if current_hashes != producer["checkpoint"]["artifact_sha256"]:
        raise CheckpointPortabilityError("checkpoint bytes changed during artifact transfer")
    manifest = verify_checkpoint(checkpoint)
    if manifest["checkpoint_id"] != producer["checkpoint"]["checkpoint_id"]:
        raise CheckpointPortabilityError("consumer checkpoint_id differs from producer")

    torch.manual_seed(SEED + 999)
    model = TwelveSixDecoder(stage.model, stage.init)
    trainer = Trainer(model, _trainer_config(), device="cpu")
    run_manifest = producer.get("run_manifest")
    if not isinstance(run_manifest, dict):
        raise CheckpointPortabilityError("producer run manifest is missing")
    if hash_json(run_manifest) != producer["identity"]["run_manifest_sha256"]:
        raise CheckpointPortabilityError("producer run-manifest identity mismatch")

    load_trainer_checkpoint(
        checkpoint,
        model=model,
        trainer=trainer,
        restore_rng=False,
        expected_git_sha=source_sha,
        expected_model_spec_hash=MODEL_SPEC_SHA256,
        expected_init_spec_hash=INIT_SPEC_SHA256,
        expected_tokenizer_hash=TOKENIZER_CONFIG_SHA256,
        expected_tokenizer_vocab_hash=TOKENIZER_VOCAB_SHA256,
        expected_dataset_manifest_hash=DATASET_MANIFEST_SHA256,
        expected_split_identity=f"train:{TRAIN_JSONL_SHA256}",
        expected_packing_hash=PACKING_CONFIG_HASH,
        expected_packing_version=PACKING_VERSION,
        expected_run_manifest_hash=producer["identity"]["run_manifest_sha256"],
        expected_training_config_hash=producer["identity"]["training_config_sha256"],
        expected_environment_lock_hash=environment_lock_sha256,
        expected_seed=SEED,
    )
    if trainer.optimizer_step != producer["identity"]["optimizer_step"]:
        raise CheckpointPortabilityError("trainer optimizer step changed after cross-arch load")
    if trainer.tokens_seen != producer["identity"]["tokens_seen"]:
        raise CheckpointPortabilityError("trainer token counter changed after cross-arch load")

    backend = load_first_party_backend(checkpoint)
    diagnostics = backend.diagnostics()
    if diagnostics.get("checkpoint_id") != manifest["checkpoint_id"]:
        raise CheckpointPortabilityError("first-party backend lost checkpoint identity")
    if diagnostics.get("git_sha") != source_sha:
        raise CheckpointPortabilityError("first-party backend lost source identity")
    if diagnostics.get("model_spec_sha256") != MODEL_SPEC_SHA256:
        raise CheckpointPortabilityError("first-party backend lost ModelSpec identity")

    report: dict[str, Any] = {
        "schema_version": CONSUMER_SCHEMA,
        "authority": AUTHORITY,
        "source": {
            "repository": REPOSITORY,
            "source_sha": source_sha,
            "architecture": architecture,
            "byteorder": sys.byteorder,
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "producer": {
            "architecture": producer["source"]["architecture"],
            "report_sha256": producer["report_sha256"],
        },
        "checkpoint": {
            "checkpoint_id": manifest["checkpoint_id"],
            "artifact_sha256": _checkpoint_file_hashes(checkpoint),
            "verify_checkpoint_pass": True,
            "trainer_restore_pass": True,
            "first_party_load_pass": True,
            "rng_restored": False,
        },
        "restored_state": {
            "optimizer_step": trainer.optimizer_step,
            "tokens_seen": trainer.tokens_seen,
            "parameter_count": stage.model.parameter_count(),
            "backend_diagnostics": diagnostics,
        },
        "claims": {
            "serialization_portability_proven": True,
            "cross_arch_training_bitwise_reproducibility": False,
            "cross_arch_inference_bitwise_reproducibility": False,
            "rng_cross_arch_equivalence": False,
            "foreign_pretrained_weights_used": False,
            "instruction_or_alignment_behavior_added": False,
            "paid_compute_authorized_or_used": False,
            "candidate_or_stable_promotion": False,
            "audit_verdict": False,
        },
    }
    report["report_sha256"] = _canonical_hash(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    validate_consumer_report(report, producer=producer)
    return report


def validate_consumer_report(
    report: dict[str, Any],
    *,
    producer: dict[str, Any],
) -> dict[str, Any]:
    validate_producer_report(producer)
    if report.get("schema_version") != CONSUMER_SCHEMA:
        raise CheckpointPortabilityError("unexpected consumer schema")
    expected_hash = report.get("report_sha256")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise CheckpointPortabilityError("consumer report hash is missing")
    unhashed = dict(report)
    del unhashed["report_sha256"]
    if _canonical_hash(unhashed) != expected_hash:
        raise CheckpointPortabilityError("consumer report self-hash mismatch")

    source = report.get("source")
    producer_ref = report.get("producer")
    checkpoint = report.get("checkpoint")
    restored = report.get("restored_state")
    claims = report.get("claims")
    if not all(
        isinstance(value, dict)
        for value in (source, producer_ref, checkpoint, restored, claims)
    ):
        raise CheckpointPortabilityError("consumer evidence structure is incomplete")
    if source["source_sha"] != producer["source"]["source_sha"]:
        raise CheckpointPortabilityError("consumer source SHA differs from producer")
    if source["architecture"] == producer["source"]["architecture"]:
        raise CheckpointPortabilityError("consumer architecture must differ from producer")
    if producer_ref.get("report_sha256") != producer["report_sha256"]:
        raise CheckpointPortabilityError("consumer producer-report binding mismatch")
    if checkpoint.get("checkpoint_id") != producer["checkpoint"]["checkpoint_id"]:
        raise CheckpointPortabilityError("consumer checkpoint_id differs from producer")
    if checkpoint.get("artifact_sha256") != producer["checkpoint"]["artifact_sha256"]:
        raise CheckpointPortabilityError("consumer checkpoint bytes differ from producer")
    for key in ("verify_checkpoint_pass", "trainer_restore_pass", "first_party_load_pass"):
        if checkpoint.get(key) is not True:
            raise CheckpointPortabilityError(f"consumer portability gate failed: {key}")
    if checkpoint.get("rng_restored") is not False:
        raise CheckpointPortabilityError("cross-architecture RNG restore must remain disabled")
    if restored.get("optimizer_step") != producer["identity"]["optimizer_step"]:
        raise CheckpointPortabilityError("consumer optimizer step differs from producer")
    if restored.get("tokens_seen") != producer["identity"]["tokens_seen"]:
        raise CheckpointPortabilityError("consumer tokens_seen differs from producer")
    if restored.get("parameter_count") != PARAMETER_COUNT:
        raise CheckpointPortabilityError("consumer parameter count drift")
    diagnostics = restored.get("backend_diagnostics")
    if not isinstance(diagnostics, dict):
        raise CheckpointPortabilityError("consumer backend diagnostics are missing")
    if diagnostics.get("checkpoint_id") != producer["checkpoint"]["checkpoint_id"]:
        raise CheckpointPortabilityError("consumer backend checkpoint identity mismatch")
    if diagnostics.get("git_sha") != producer["source"]["source_sha"]:
        raise CheckpointPortabilityError("consumer backend source identity mismatch")
    if diagnostics.get("model_spec_sha256") != MODEL_SPEC_SHA256:
        raise CheckpointPortabilityError("consumer backend ModelSpec identity mismatch")
    if not isinstance(claims, dict):
        raise CheckpointPortabilityError("consumer truth boundary is missing")
    if claims.get("serialization_portability_proven") is not True:
        raise CheckpointPortabilityError("consumer portability PASS is missing")
    for key in (
        "cross_arch_training_bitwise_reproducibility",
        "cross_arch_inference_bitwise_reproducibility",
        "rng_cross_arch_equivalence",
        "foreign_pretrained_weights_used",
        "instruction_or_alignment_behavior_added",
        "paid_compute_authorized_or_used",
        "candidate_or_stable_promotion",
        "audit_verdict",
    ):
        if claims.get(key) is not False:
            raise CheckpointPortabilityError(f"consumer claim must remain false: {key}")
    return {
        "status": "PASS",
        "source_sha": source["source_sha"],
        "producer_architecture": producer["source"]["architecture"],
        "consumer_architecture": source["architecture"],
        "checkpoint_id": checkpoint["checkpoint_id"],
        "producer_report_sha256": producer["report_sha256"],
        "consumer_report_sha256": expected_hash,
    }
