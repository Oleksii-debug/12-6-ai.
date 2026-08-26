from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from twelve_six.checkpoint import (
    bind_checkpoint_identity,
    capture_rng_state,
    detect_git_sha,
    hash_json,
    load_trainer_checkpoint,
    save_trainer_checkpoint,
    sha256_file,
    verify_checkpoint,
)
from twelve_six.data import build_dataset
from twelve_six.inference import GenerationConfig, generate
from twelve_six.integration import (
    CandidateStatus,
    CIEvidence,
    ComponentDisposition,
    ComponentRef,
    S0TorchInferenceBackend,
    StageCandidateManifest,
)
from twelve_six.model import TwelveSixDecoder, load_stage_config
from twelve_six.packing import PACKING_CONFIG_HASH, PACKING_VERSION
from twelve_six.tokenization import BYTE_TOKENIZER_HASH, ByteTokenizer
from twelve_six.training import Trainer, TrainerConfig

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_PATH = ROOT / "configs/releases/s0_convergence_20260824.experimental.json"
DATASET_MANIFEST_SHA256 = "b085a7ab56510575a11a80824fcff3a95a17f237d46d1be820e59d1289f220c2"
DATASET_IDENTITY_SHA256 = "bab60119d49e93303c972b77900fcb5553817f754cbc5d9a58019228cfa0ca89"
TRAIN_JSONL_SHA256 = "61d24b7138df56527d201cea405d11c9f607684b4a9593dfa20c599cc2ee6998"
ENVIRONMENT_LOCK_SHA256 = "61fa31fbb5da7a4289cccce5abfcebde943664f5318b0ce3d69ae9bb3db852ac"


def _load_first_jsonl(path: Path) -> dict[str, object]:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            return json.loads(line)
    raise AssertionError(f"no records in {path}")


def _assert_nested_exact(actual, expected) -> None:
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        return
    if isinstance(expected, np.ndarray):
        assert isinstance(actual, np.ndarray)
        np.testing.assert_array_equal(actual, expected)
        return
    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_nested_exact(actual[key], expected[key])
        return
    if isinstance(expected, (list, tuple)):
        assert isinstance(actual, type(expected))
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected, strict=True):
            _assert_nested_exact(actual_item, expected_item)
        return
    assert actual == expected


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def test_s0_accepted_contracts_execute_model_data_tokenizer_train_and_inference(
    tmp_path: Path,
) -> None:
    stage = load_stage_config(ROOT / "configs/stages/s0_10k.json")
    tokenizer = ByteTokenizer()

    assert stage.canonical_base == "random_init"
    assert stage.expected_parameters == 10_140
    assert stage.model.vocab_size == tokenizer.vocab_size == 256
    assert tokenizer.identity.config_sha256 == BYTE_TOKENIZER_HASH

    rebuilt_dir = tmp_path / "rebuilt-s0"
    manifest = build_dataset(
        ROOT / "data/s0/source_registry.json",
        ROOT / "data/s0/contamination_registry.json",
        rebuilt_dir,
    )
    assert manifest["dataset_identity_sha256"] == DATASET_IDENTITY_SHA256
    committed_manifest_bytes = (ROOT / "data/s0/packaged/manifest.json").read_bytes()
    assert hashlib.sha256(committed_manifest_bytes).hexdigest() == DATASET_MANIFEST_SHA256
    assert (rebuilt_dir / "manifest.json").read_bytes() == committed_manifest_bytes

    record = _load_first_jsonl(rebuilt_dir / "train.jsonl")
    text = str(record["text"])
    token_ids = tokenizer.encode(text)[: min(stage.model.max_seq_len, 64)]
    assert len(token_ids) >= 2

    torch.manual_seed(20260824)
    model = TwelveSixDecoder(stage.model, stage.init)
    before = model.token_embedding.weight.detach().clone()
    batch_ids = torch.tensor([token_ids], dtype=torch.long)
    trainer = Trainer(
        model,
        TrainerConfig(
            learning_rate=1e-2,
            max_steps=1,
            seed=20260824,
            precision="fp32",
            deterministic_algorithms=True,
        ),
        device="cpu",
    )
    metrics = trainer.train_microbatch({"input_ids": batch_ids, "labels": batch_ids})

    assert metrics.optimizer_stepped is True
    assert math.isfinite(metrics.loss)
    assert metrics.optimizer_step == 1
    assert not torch.equal(before, model.token_embedding.weight.detach())

    backend = S0TorchInferenceBackend(model, tokenizer)
    result = generate(
        backend,
        "12-6",
        GenerationConfig(max_new_tokens=2, sample=False, seed=20260824),
    )
    assert len(result.generated_token_ids) == 2
    assert all(0 <= token_id < tokenizer.vocab_size for token_id in result.generated_token_ids)
    assert result.stop_reason == "max_new_tokens"


def test_s0_interrupted_save_destroy_verify_fresh_trainer_resume_matches_control(
    tmp_path: Path,
) -> None:
    stage = load_stage_config(ROOT / "configs/stages/s0_10k.json")
    tokenizer = ByteTokenizer()
    train_path = ROOT / "data/s0/packaged/train.jsonl"
    assert sha256_file(train_path) == TRAIN_JSONL_SHA256
    assert sha256_file(ROOT / "requirements/locks/index.json") == ENVIRONMENT_LOCK_SHA256
    assert stage.model.identity_sha256() == stage.expected_model_identity_sha256
    assert stage.init.identity_sha256() == stage.expected_init_identity_sha256
    assert PACKING_CONFIG_HASH == "23a695b807f3e3f5c61d19c34968bcd88fafc6a45346dc08673d7a494219f285"

    record = _load_first_jsonl(train_path)
    token_ids = tokenizer.encode(str(record["text"]))[: min(stage.model.max_seq_len, 64)]
    batch_ids = torch.tensor([token_ids], dtype=torch.long)
    batch = {"input_ids": batch_ids, "labels": batch_ids}
    seed = 20260824
    total_steps = 4
    split_step = 2
    trainer_config = TrainerConfig(
        learning_rate=1e-2,
        max_steps=total_steps,
        seed=seed,
        precision="fp32",
        scheduler="linear_warmup",
        warmup_steps=1,
        deterministic_algorithms=True,
    )

    _seed_all(seed)
    control_model = TwelveSixDecoder(stage.model, stage.init)
    control_trainer = Trainer(control_model, trainer_config, device="cpu")
    for _ in range(total_steps):
        control_trainer.train_microbatch(batch)
    control_weights = {
        name: tensor.detach().clone() for name, tensor in control_model.state_dict().items()
    }
    control_state = asdict(control_trainer.state_dict())
    control_rng = capture_rng_state()

    _seed_all(seed)
    interrupted_model = TwelveSixDecoder(stage.model, stage.init)
    interrupted_trainer = Trainer(interrupted_model, trainer_config, device="cpu")
    for _ in range(split_step):
        interrupted_trainer.train_microbatch(batch)

    source_sha = detect_git_sha(ROOT)
    assert source_sha is not None and len(source_sha) in {40, 64}
    model_spec = stage.model.to_dict()
    init_spec = stage.init.to_dict()
    split_identity = f"train:{TRAIN_JSONL_SHA256}"
    run_manifest = {
        "schema_version": 1,
        "run_id": "s0-local-free-real-resume-proof",
        "stage": "S0",
        "run_kind": "integrated_training",
        "state": "RUNNING",
        "candidate": {
            "repository": "Oleksii-debug/12-6-ai.",
            "git_sha": source_sha,
            "branch_or_tag": "d05/integrated-resume-repro-20260824",
            "modelspec_sha256": stage.expected_model_identity_sha256,
            "initspec_sha256": stage.expected_init_identity_sha256,
            "parameter_count": stage.expected_parameters,
        },
        "data": {
            "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
            "tokenizer_sha256": tokenizer.identity.config_sha256,
            "tokenizer_vocab_sha256": tokenizer.identity.vocab_sha256,
            "tokenizer_version": tokenizer.identity.version,
            "split_identity": split_identity,
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
            "scheduler": {
                "name": trainer_config.scheduler,
                "warmup_steps": trainer_config.warmup_steps,
            },
            "trainer_config": asdict(trainer_config),
            "context_length": stage.model.max_seq_len,
            "global_batch_tokens": len(token_ids) - 1,
            "target_steps": total_steps,
            "target_tokens": (len(token_ids) - 1) * total_steps,
            "checkpoint_interval_steps": split_step,
        },
        "environment": {"lock_sha256": ENVIRONMENT_LOCK_SHA256},
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
        step=interrupted_trainer.optimizer_step,
        tokens_seen=interrupted_trainer.tokens_seen,
        environment_lock_hash=ENVIRONMENT_LOCK_SHA256,
    )

    checkpoint_dir = tmp_path / "s0-resume-checkpoint"
    saved_manifest = save_trainer_checkpoint(
        checkpoint_dir,
        model=interrupted_model,
        trainer=interrupted_trainer,
        identity=identity,
    )
    assert saved_manifest["serialization"]["pickle"] is False

    del interrupted_trainer
    del interrupted_model
    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)

    verified = verify_checkpoint(checkpoint_dir)
    assert verified["checkpoint_id"] == saved_manifest["checkpoint_id"]
    assert verified["identity"]["git_sha"] == source_sha
    assert verified["identity"]["run_manifest_hash"] == hash_json(run_manifest)
    assert verified["identity"]["environment_lock_hash"] == ENVIRONMENT_LOCK_SHA256

    restored_model = TwelveSixDecoder(stage.model, stage.init)
    restored_trainer = Trainer(restored_model, trainer_config, device="cpu")
    loaded = load_trainer_checkpoint(
        checkpoint_dir,
        model=restored_model,
        trainer=restored_trainer,
        restore_rng=True,
        expected_git_sha=source_sha,
        expected_model_spec_hash=stage.expected_model_identity_sha256,
        expected_init_spec_hash=stage.expected_init_identity_sha256,
        expected_tokenizer_hash=tokenizer.identity.config_sha256,
        expected_tokenizer_vocab_hash=tokenizer.identity.vocab_sha256,
        expected_dataset_manifest_hash=DATASET_MANIFEST_SHA256,
        expected_split_identity=split_identity,
        expected_packing_hash=PACKING_CONFIG_HASH,
        expected_packing_version=PACKING_VERSION,
        expected_run_manifest_hash=hash_json(run_manifest),
        expected_training_config_hash=saved_manifest["identity"]["training_config_hash"],
        expected_environment_lock_hash=ENVIRONMENT_LOCK_SHA256,
        expected_seed=seed,
    )
    assert loaded.manifest["checkpoint_id"] == saved_manifest["checkpoint_id"]
    assert restored_trainer.optimizer_step == split_step

    for _ in range(split_step, total_steps):
        restored_trainer.train_microbatch(batch)

    for name, actual in restored_model.state_dict().items():
        torch.testing.assert_close(actual, control_weights[name], rtol=0.0, atol=0.0)
    _assert_nested_exact(asdict(restored_trainer.state_dict()), control_state)
    _assert_nested_exact(capture_rng_state(), control_rng)

    control_generation = generate(
        S0TorchInferenceBackend(control_model, tokenizer),
        "12-6",
        GenerationConfig(max_new_tokens=2, sample=False, seed=seed),
    )
    resumed_generation = generate(
        S0TorchInferenceBackend(restored_model, tokenizer),
        "12-6",
        GenerationConfig(max_new_tokens=2, sample=False, seed=seed),
    )
    assert resumed_generation.generated_token_ids == control_generation.generated_token_ids
    assert resumed_generation.stop_reason == control_generation.stop_reason


def test_s0_evidence_accepts_green_checkpoint_lineage_but_holds_red_eval() -> None:
    payload = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))
    components = []
    for row in payload["components"]:
        ci_evidence = CIEvidence(
            run_id=row["ci_run_id"],
            head_sha=row["source_sha"],
            conclusion=row["ci_conclusion"],
            evidence_ref=f"github-actions:{row['ci_run_id']}",
        )
        components.append(
            ComponentRef(
                lane=row["lane"],
                source_sha=row["source_sha"],
                disposition=ComponentDisposition(row["disposition"]),
                component_kind=row["component_kind"],
                pr_number=row["pr_number"],
                ci_evidence=ci_evidence,
                contains_behavioral_weights=row.get("contains_behavioral_weights"),
                contains_foreign_pretrained_weights=row.get(
                    "contains_foreign_pretrained_weights"
                ),
                notes=row.get("hold_reason", ""),
            )
        )

    convergence = StageCandidateManifest.compose(
        stage=payload["stage"],
        integration_anchor_sha=payload["integration_anchor_sha"],
        status=CandidateStatus(payload["status"]),
        base_lineage=payload["base_lineage"],
        components=components,
    )

    assert convergence.accepted_lanes() == frozenset(
        {"D01", "D02", "D03", "D04", "D05", "D07", "D08"}
    )
    assert convergence.missing_required_lanes() == ("D06",)
    assert convergence.ready_for_candidate() is False
    assert payload["accepted_package_reconciliation"]["dependencies"] == [
        "numpy>=1.26",
        "safetensors>=0.5",
        "torch>=2.5",
    ]
    assert payload["audits"] == {
        "AUDIT-A": "CHANGES_REQUIRED",
        "AUDIT-B": "CHANGES_REQUIRED",
    }
