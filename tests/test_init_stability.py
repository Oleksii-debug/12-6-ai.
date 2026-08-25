from __future__ import annotations

import copy
import json

import pytest

from twelve_six.checkpoint.core import hash_json
from twelve_six.checkpoint.run_binding import bind_checkpoint_identity
from twelve_six.model import InitSpec, ModelSpec, StageConfig
from twelve_six.training.init_stability import (
    ProbeSpec,
    run_seed_probe,
    run_stage_matrix,
    validate_report,
)


def _tiny_stage() -> StageConfig:
    model = ModelSpec(
        schema_version=1,
        vocab_size=32,
        max_seq_len=16,
        d_model=16,
        n_layers=2,
        n_heads=2,
        n_kv_heads=2,
        head_dim=8,
        d_ff=32,
        rope_rotary_dim=8,
    )
    init = InitSpec()
    return StageConfig(
        stage="TEST",
        target_parameters=model.parameter_count(),
        expected_parameters=model.parameter_count(),
        canonical_base="random_init",
        expected_model_identity_sha256=model.identity_sha256(),
        expected_init_identity_sha256=init.identity_sha256(),
        model=model,
        init=init,
    )


def test_controls_have_distinct_initspec_identity_and_default_is_preserved() -> None:
    stage = _tiny_stage()
    probe = ProbeSpec(batch_size=1, sequence_length=8, steps=0, seeds=(7,))
    default = run_seed_probe(stage=stage, candidate="stage_default", probe=probe, seed=7)
    unscaled = run_seed_probe(
        stage=stage,
        candidate="unscaled_residual_control",
        probe=probe,
        seed=7,
    )
    width = run_seed_probe(
        stage=stage,
        candidate="s1_width_reference_control",
        probe=probe,
        seed=7,
    )

    assert default["init_identity_sha256"] == stage.init.identity_sha256()
    assert unscaled["init_identity_sha256"] != default["init_identity_sha256"]
    assert width["init_identity_sha256"] != default["init_identity_sha256"]
    assert default["all_finite"] is True
    assert unscaled["all_finite"] is True
    assert width["all_finite"] is True


def test_report_is_hash_bound_and_fail_closed(tmp_path) -> None:
    stage = _tiny_stage()
    payload = {
        "stage": stage.stage,
        "target_parameters": stage.target_parameters,
        "expected_parameters": stage.expected_parameters,
        "canonical_base": stage.canonical_base,
        "expected_model_identity_sha256": stage.expected_model_identity_sha256,
        "expected_init_identity_sha256": stage.expected_init_identity_sha256,
        "model": stage.model.to_dict(),
        "init": stage.init.to_dict(),
    }
    stage_path = tmp_path / "stage.json"
    stage_path.write_text(json.dumps(payload), encoding="utf-8")
    report = run_stage_matrix(
        stage_config_path=stage_path,
        candidate="stage_default",
        probe=ProbeSpec(batch_size=1, sequence_length=8, steps=1, seeds=(3, 5)),
    )
    validate_report(report)

    changed = copy.deepcopy(report)
    changed["truth_boundary"]["stage_promotion_granted"] = True
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        validate_report(changed)


def test_checkpoint_binding_distinguishes_initspec_hash() -> None:
    model_spec = {"vocab_size": 32}
    init_a = InitSpec().to_dict()
    init_b = InitSpec(std=0.01).to_dict()
    tokenizer = {
        "config_sha256": "1" * 64,
        "vocab_sha256": "2" * 64,
        "version": "test-tokenizer",
        "vocab_size": 32,
    }
    packing = {"config_sha256": "4" * 64, "version": "test-packing"}

    def manifest(init_spec: dict[str, object]) -> dict[str, object]:
        return {
            "run_id": "init-regression",
            "stage": "TEST",
            "run_kind": "engineering",
            "candidate": {
                "git_sha": "a" * 40,
                "modelspec_sha256": hash_json(model_spec),
                "initspec_sha256": hash_json(init_spec),
                "parameter_count": 1234,
            },
            "data": {
                "tokenizer_sha256": tokenizer["config_sha256"],
                "tokenizer_vocab_sha256": tokenizer["vocab_sha256"],
                "tokenizer_version": tokenizer["version"],
                "dataset_manifest_sha256": "3" * 64,
                "split_identity": "test-split",
                "packing_sha256": packing["config_sha256"],
                "packing_version": packing["version"],
            },
            "training": {
                "seed": 7,
                "precision": "fp32",
                "optimizer": {"name": "AdamW"},
                "scheduler": {"name": "constant"},
            },
            "environment": {"lock_sha256": "5" * 64},
        }

    identity_a = bind_checkpoint_identity(
        run_manifest=manifest(init_a),
        model_spec=model_spec,
        init_spec=init_a,
        tokenizer_identity=tokenizer,
        packing_identity=packing,
        step=0,
        tokens_seen=0,
    )
    identity_b = bind_checkpoint_identity(
        run_manifest=manifest(init_b),
        model_spec=model_spec,
        init_spec=init_b,
        tokenizer_identity=tokenizer,
        packing_identity=packing,
        step=0,
        tokens_seen=0,
    )

    assert identity_a.training_config["init_spec_sha256"] == hash_json(init_a)
    assert identity_b.training_config["init_spec_sha256"] == hash_json(init_b)
    assert (
        identity_a.training_config["init_spec_sha256"]
        != identity_b.training_config["init_spec_sha256"]
    )
    assert identity_a.run_manifest_hash != identity_b.run_manifest_hash
