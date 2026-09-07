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
    CheckpointIdentity,
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
ENVIRONMENT_LOCK_SHA256 = "c50841c05f66ae2f3f2bbf08f407de5ac484b0ce83113989bfec19077a0fd268"


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

    _seed_all(seed)
    interrupted_model = TwelveSixDecoder(stage.model, stage.init)
    interrupted_trainer = Trainer(interrupted_model, trainer_config, device="cpu")
    for _ in range(split_step):
        interrupted_trainer.train_microbatch(batch)

    source_git_sha = detect_git_sha(ROOT)
    assert source_git_sha is not None
    training_config = {
        "integration_test": "s0_convergence_resume",
        "init_spec_sha256": stage.expected_init_identity_sha256,
        "trainer": asdict(trainer_config),
        "data": {
            "split_identity": DATASET_IDENTITY_SHA256,
            "packing_sha256": PACKING_CONFIG_HASH,
            "packing_version": PACKING_VERSION,
        },
    }
    identity = CheckpointIdentity(
        git_sha=source_git_sha,
        model_spec=stage.model.to_dict(),
        parameter_count=stage.expected_parameters,
        tokenizer_hash=tokenizer.identity.config_sha256,
        tokenizer_vocab_hash=tokenizer.identity.vocab_sha256,
        dataset_manifest_hash=DATASET_MANIFEST_SHA256,
        run_manifest_hash=hash_json(
            {
                "integration_test": "s0_convergence_resume",
                "dataset_identity_sha256": DATASET_IDENTITY_SHA256,
                "packing_sha256": PACKING_CONFIG_HASH,
                "packing_version": PACKING_VERSION,
            }
        ),
        training_config=training_config,
        seed=seed,
        precision=trainer_config.precision,
        step=interrupted_trainer.optimizer_step,
        tokens_seen=interrupted_trainer.tokens_seen,
        optimizer={"name": interrupted_trainer.optimizer.__class__.__name__},
        scheduler={"name": trainer_config.scheduler},
        environment_lock_hash=ENVIRONMENT_LOCK_SHA256,
    )

    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint = save_trainer_checkpoint(
        checkpoint_dir,
        model=interrupted_model,
        trainer=interrupted_trainer,
        identity=identity,
    )
    verified = verify_checkpoint(checkpoint_dir)
    assert verified["checkpoint_id"] == checkpoint["checkpoint_id"]
    assert checkpoint["identity"]["git_sha"] == source_git_sha
    assert checkpoint["identity"]["model_spec_hash"] == stage.expected_model_identity_sha256
    assert checkpoint["identity"]["training_config_hash"] == hash_json(training_config)

    del interrupted_trainer
    del interrupted_model

    fresh_model = TwelveSixDecoder(stage.model, stage.init)
    fresh_trainer = Trainer(fresh_model, trainer_config, device="cpu")
    loaded = load_trainer_checkpoint(
        checkpoint_dir,
        model=fresh_model,
        trainer=fresh_trainer,
        expected_git_sha=source_git_sha,
        expected_model_spec_hash=stage.expected_model_identity_sha256,
        expected_init_spec_hash=stage.expected_init_identity_sha256,
        expected_tokenizer_hash=tokenizer.identity.config_sha256,
        expected_tokenizer_vocab_hash=tokenizer.identity.vocab_sha256,
        expected_dataset_manifest_hash=DATASET_MANIFEST_SHA256,
        expected_split_identity=DATASET_IDENTITY_SHA256,
        expected_packing_hash=PACKING_CONFIG_HASH,
        expected_packing_version=PACKING_VERSION,
        expected_run_manifest_hash=identity.run_manifest_hash,
        expected_training_config_hash=hash_json(training_config),
        expected_environment_lock_hash=ENVIRONMENT_LOCK_SHA256,
        expected_seed=seed,
    )
    assert loaded.manifest["checkpoint_id"] == checkpoint["checkpoint_id"]

    for _ in range(split_step, total_steps):
        fresh_trainer.train_microbatch(batch)

    for name, tensor in fresh_model.state_dict().items():
        torch.testing.assert_close(tensor, control_weights[name], rtol=0.0, atol=0.0)
    _assert_nested_exact(asdict(fresh_trainer.state_dict()), control_state)


def test_s0_release_candidate_manifest_is_explicitly_experimental() -> None:
    payload = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))
    assert payload["status"] == CandidateStatus.EXPERIMENTAL.value
    assert payload["base_lineage"] is True
    assert payload["model_spec_sha256"] == "86c75b31dff05b7b5db9f6ed068c571a6ead01ba663412fe630f5e52b09d9b6b"
    assert payload["init_spec_sha256"] == "86483c6df623e80cab2f73aba718863fce18af6fe3b12430c1348414d92b48a5"
    assert payload["tokenizer_config_sha256"] == BYTE_TOKENIZER_HASH
    assert payload["dataset_manifest_sha256"] == DATASET_MANIFEST_SHA256

    components = []
    for item in payload["components"]:
        components.append(
            ComponentRef(
                lane=item["lane"],
                source_sha=item["source_sha"],
                disposition=ComponentDisposition(item["disposition"]),
                component_kind=item["component_kind"],
                pr_number=item.get("pr_number"),
                ci_evidence=CIEvidence(
                    run_id=item["ci_run_id"],
                    head_sha=item["source_sha"],
                    conclusion=item["ci_conclusion"],
                    evidence_ref=f"pr://{item.get('pr_number', 'unknown')}#ci-{item['ci_run_id']}",
                ),
                contains_behavioral_weights=item.get("contains_behavioral_weights"),
                contains_foreign_pretrained_weights=item.get(
                    "contains_foreign_pretrained_weights"
                ),
                notes=item.get("hold_reason", ""),
            )
        )

    manifest = StageCandidateManifest.compose(
        stage=payload["stage"],
        integration_anchor_sha=payload["integration_anchor_sha"],
        status=CandidateStatus(payload["status"]),
        base_lineage=payload["base_lineage"],
        components=components,
    )
    d01 = next(component for component in manifest.components if component.lane == "D01")
    d06 = next(component for component in manifest.components if component.lane == "D06")

    assert manifest.status is CandidateStatus.EXPERIMENTAL
    assert manifest.base_lineage is True
    assert d01.component_kind == "random_init_model"
    assert d01.contains_behavioral_weights is False
    assert d01.contains_foreign_pretrained_weights is False
    assert d01.ci_evidence is not None and d01.ci_evidence.passes
    assert d06.disposition is ComponentDisposition.HELD
    assert d06.ci_evidence is not None and not d06.ci_evidence.passes
    assert manifest.missing_required_lanes() == ["D06"]
    assert manifest.ready_for_candidate() is False
    assert manifest.audits_pass() is False
    assert manifest.candidate_sha is None
    assert manifest.release_artifact is None
