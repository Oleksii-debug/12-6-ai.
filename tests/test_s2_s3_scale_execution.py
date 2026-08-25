"""Focused tests for canonical S2/S3 scale execution contracts."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from twelve_six.model import load_stage_config
from twelve_six.training.s2_s3_scale_execution import (
    AUTHORITY,
    RUN_CONFIG_PATH,
    SCHEMA_VERSION,
    ScaleExecutionError,
    load_scale_execution_config,
    validate_scale_execution_evidence,
)


def _hash(payload: dict) -> str:
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _valid_evidence(stage_name: str = "S2") -> dict:
    if stage_name == "S2":
        stage_path = "configs/stages/s2_1m.json"
        model_sha = (
            "2889fdea4d17b5f592686c1a1a2fcd7dd16a9a029219351e95973ccfdef60566"
        )
        parameters = 1_066_112
        vocab = 2_048
        context = 512
        steps = 4
        batch_size = 2
    else:
        stage_path = "configs/stages/s3_10m.json"
        model_sha = (
            "3b6fc1b397e6fea69c2f249ce8ab8eedaad8ca1b13b88b8d2328a6abcf34791a"
        )
        parameters = 10_059_840
        vocab = 8_192
        context = 1_024
        steps = 2
        batch_size = 1
    parameter_bytes = parameters * 4
    optimizer_bytes = parameter_bytes * 2 + 16
    identity = {
        "repository": "Oleksii-debug/12-6-ai.",
        "source_sha": "a" * 40,
        "stage": stage_name,
        "stage_config_path": stage_path,
        "stage_config_file_sha256": "b" * 64,
        "run_config_path": RUN_CONFIG_PATH,
        "run_config_file_sha256": "c" * 64,
        "modelspec_sha256": model_sha,
        "initspec_sha256": (
            "86483c6df623e80cab2f73aba718863fce18af6fe3b12430c1348414d92b48a5"
        ),
        "parameter_count": parameters,
        "model_vocab_size": vocab,
        "max_seq_len": context,
        "environment": {
            "profile_id": "linux-x86_64",
            "python_version": "3.11.16",
            "environment_evidence_sha256": "d" * 64,
        },
        "fixture": {
            "purpose": (
                "CONTROLLED_S0_FIXTURE_COMPATIBILITY_ONLY_NOT_S2_S3_"
                "CORPUS_OR_TOKENIZER"
            ),
            "tokenizer_vocab_size": 256,
            "max_emitted_token_id": 255,
            "unused_model_vocab_rows": vocab - 256,
            "train_record_ids": ["train-a"],
            "validation_record_ids": ["validation-a"],
        },
    }
    evidence = {
        "schema_version": SCHEMA_VERSION,
        "authority": AUTHORITY,
        "identity": identity,
        "identity_sha256": _hash(identity),
        "recipe": {
            "device": "cpu",
            "precision": "fp32",
            "sequence_length": 128,
            "train_batch_limit": 2,
            "validation_batch_limit": 1,
            "batch_size": batch_size,
            "optimizer_steps_requested": steps,
            "learning_rate": 0.001,
            "seed": 1337,
        },
        "training": {
            "status": "PASS",
            "optimizer_steps": steps,
            "microbatches_consumed": steps,
            "optimized_tokens": 100,
            "initial_train_loss": 9.0,
            "final_train_loss": 8.0,
            "initial_validation_loss": 9.1,
            "final_validation_loss": 8.9,
            "gradient_norm_min": 0.1,
            "gradient_norm_max": 1.0,
            "weight_delta": {
                "l2": 0.2,
                "max_abs": 0.01,
                "changed_parameter_elements": 100,
                "trainable_parameter_elements": parameters,
            },
            "validation_optimized_tokens": 0,
            "optimizer_step_before_final_validation": steps,
            "optimizer_step_after_final_validation": steps,
        },
        "resources": {
            "model_parameter_bytes": parameter_bytes,
            "optimizer_tensor_bytes_after_training": optimizer_bytes,
            "measurement_snapshot_bytes": parameter_bytes,
            "observed_tensor_bytes_with_snapshot": (
                parameter_bytes * 2 + optimizer_bytes
            ),
        },
        "runtime": {
            "device": "cpu",
            "precision": "fp32",
            "wall_seconds_training_only": 2.0,
            "process_cpu_seconds_training_only": 2.0,
            "optimized_tokens_per_wall_second": 50.0,
            "python": "3.11.16",
            "torch": "test",
        },
        "claims": {
            "stage_architecture_frozen": False,
            "stage_corpus_or_tokenizer_frozen": False,
            "stage_quality_or_capability_evidence": False,
            "candidate_or_stable_promotion": False,
            "foreign_pretrained_weights_used": False,
            "instruction_or_alignment_training": False,
            "paid_compute_authorized_or_used": False,
            "gpu_or_distributed_execution": False,
            "cross_hardware_bitwise_reproducibility": False,
        },
    }
    evidence["evidence_sha256"] = _hash(evidence)
    return evidence


def test_run_config_matches_current_canonical_stage_specs() -> None:
    root = Path.cwd()
    config = load_scale_execution_config(root)
    assert set(config["stages"]) == {"S2", "S3"}
    for stage_name, expected in (("S2", 1_066_112), ("S3", 10_059_840)):
        path = root / config["stages"][stage_name]["stage_config_path"]
        stage = load_stage_config(path)
        assert stage.stage == stage_name
        assert stage.expected_parameters == expected
        assert stage.model.parameter_count() == expected


@pytest.mark.parametrize("stage_name", ["S2", "S3"])
def test_validator_accepts_strict_mechanics_evidence(stage_name: str) -> None:
    validate_scale_execution_evidence(
        _valid_evidence(stage_name),
        expected_stage=stage_name,
    )


def test_validator_rejects_rehashed_paid_compute_overclaim() -> None:
    evidence = _valid_evidence()
    tampered = copy.deepcopy(evidence)
    tampered["claims"]["paid_compute_authorized_or_used"] = True
    tampered.pop("evidence_sha256")
    tampered["evidence_sha256"] = _hash(tampered)
    with pytest.raises(ScaleExecutionError, match="paid_compute"):
        validate_scale_execution_evidence(tampered)


def test_run_config_is_valid_json() -> None:
    payload = json.loads(Path(RUN_CONFIG_PATH).read_text(encoding="utf-8"))
    assert payload["schema_version"].endswith("config.v1")
