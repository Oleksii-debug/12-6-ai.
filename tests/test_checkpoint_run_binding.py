from __future__ import annotations

import copy
import json
from dataclasses import replace

import pytest
from torch import nn

from twelve_six.checkpoint import (
    CheckpointCompatibilityError,
    bind_checkpoint_identity,
    hash_json,
    save_checkpoint,
    verify_checkpoint,
)
from twelve_six.training.config import TrainerConfig, TrainingRecipe

TOKENIZER_SHA = "b04055c1061dd641dcab7cb9d62a931f09b8d1a070140a926ceb4e91d73ca8e1"
VOCAB_SHA = "905ed40bb42cc4d550e228ff5f24158d504b38e8ed5974dfa3077bd5867ad571"
DATASET_SHA = "b085a7ab56510575a11a80824fcff3a95a17f237d46d1be820e59d1289f220c2"
PACKING_SHA = "23a695b807f3e3f5c61d19c34968bcd88fafc6a45346dc08673d7a494219f285"
ENV_LOCK_SHA = "61fa31fbb5da7a4289cccce5abfcebde943664f5318b0ce3d69ae9bb3db852ac"
GIT_SHA = "a" * 40


def model_spec() -> dict[str, object]:
    return {
        "vocab_size": 256,
        "max_seq_len": 128,
        "d_model": 20,
        "n_layers": 1,
        "n_heads": 2,
        "d_ff": 56,
    }


def init_spec() -> dict[str, object]:
    return {
        "schema_version": 1,
        "family": "normal",
        "std": 0.02,
        "residual_branch_scale": "sqrt_2_layers",
    }


def tokenizer_identity() -> dict[str, object]:
    return {
        "version": "s0-byte-v1",
        "config_sha256": TOKENIZER_SHA,
        "vocab_sha256": VOCAB_SHA,
        "vocab_size": 256,
        "normalization": "none",
        "encoding": "utf-8-bytes",
        "special_tokens": {},
    }


def packing_identity() -> dict[str, object]:
    return {
        "version": "s0-byte-pack-v1",
        "config_sha256": PACKING_SHA,
        "sequence_length": 128,
    }


def run_manifest() -> dict[str, object]:
    spec = model_spec()
    initialization = init_spec()
    return {
        "schema_version": 1,
        "run_id": "s0-local-cpu-0001",
        "stage": "S0",
        "run_kind": "integrated_training",
        "state": "RUNNING",
        "candidate": {
            "repository": "Oleksii-debug/12-6-ai.",
            "git_sha": GIT_SHA,
            "branch_or_tag": "candidate/s0",
            "modelspec_sha256": hash_json(spec),
            "initspec_sha256": hash_json(initialization),
            "parameter_count": 10140,
        },
        "data": {
            "dataset_manifest_sha256": DATASET_SHA,
            "tokenizer_sha256": TOKENIZER_SHA,
            "tokenizer_vocab_sha256": VOCAB_SHA,
            "tokenizer_version": "s0-byte-v1",
            "split_identity": "s0-tiny-controlled-v1:train",
            "packing_sha256": PACKING_SHA,
            "packing_version": "s0-byte-pack-v1",
        },
        "training": {
            "seed": 17,
            "device": "cpu",
            "precision": "fp32",
            "optimizer": {"name": "AdamW", "lr": 0.001},
            "scheduler": {"name": "constant"},
            "context_length": 128,
            "global_batch_tokens": 256,
            "target_steps": 20,
            "target_tokens": 5120,
            "checkpoint_interval_steps": 5,
        },
        "environment": {"lock_sha256": ENV_LOCK_SHA},
    }


def bind(manifest: dict[str, object] | None = None):
    return bind_checkpoint_identity(
        run_manifest=manifest or run_manifest(),
        model_spec=model_spec(),
        init_spec=init_spec(),
        tokenizer_identity=tokenizer_identity(),
        packing_identity=packing_identity(),
        step=5,
        tokens_seen=1280,
        environment_lock_hash=ENV_LOCK_SHA,
    )


def test_binds_exact_run_manifest_identity() -> None:
    manifest = run_manifest()
    identity = bind(manifest)

    assert identity.git_sha == GIT_SHA
    assert identity.model_spec == model_spec()
    assert identity.parameter_count == 10140
    assert identity.tokenizer_hash == TOKENIZER_SHA
    assert identity.tokenizer_vocab_hash == VOCAB_SHA
    assert identity.dataset_manifest_hash == DATASET_SHA
    assert identity.run_manifest_hash == hash_json(manifest)
    assert identity.environment_lock_hash == ENV_LOCK_SHA
    assert identity.training_config["run_id"] == "s0-local-cpu-0001"
    assert identity.training_config["run_manifest_sha256"] == hash_json(manifest)
    assert identity.training_config["init_spec_sha256"] == hash_json(init_spec())
    assert identity.training_config["data"]["split_identity"] == "s0-tiny-controlled-v1:train"
    assert identity.training_config["data"]["packing_sha256"] == PACKING_SHA
    assert identity.training_config["data"]["packing_version"] == "s0-byte-pack-v1"
    assert identity.optimizer == {"name": "AdamW", "lr": 0.001}
    assert identity.scheduler == {"name": "constant"}


def frozen_recipe() -> TrainingRecipe:
    return TrainingRecipe(
        trainer=TrainerConfig(
            seed=17, max_steps=20, scheduler="cosine", warmup_steps=2,
            gradient_accumulation_steps=2, weight_decay=0.1,
        ),
        microbatch_size=2,
        sequence_length=128,
        target_unique_loss_positions=2560,
        maximum_total_exposures=5120,
        max_exposures_per_unique_position=2,
    )


def test_recipe_identity_round_trips_and_excludes_provider() -> None:
    recipe = frozen_recipe()
    assert TrainingRecipe.from_dict(json.loads(json.dumps(recipe.to_dict()))) == recipe
    assert recipe.identity_sha256() == hash_json(recipe.to_dict())
    changed_trainer = replace(recipe.trainer, gradient_accumulation_steps=4)
    assert replace(recipe, trainer=changed_trainer).identity_sha256() != recipe.identity_sha256()
    local = run_manifest()
    local["training"].update(recipe.run_training_fields())
    server = copy.deepcopy(local)
    server["training"]["device"] = "cuda:0"
    assert bind(local).run_manifest_hash != bind(server).run_manifest_hash
    assert (
        bind(local).training_config["training"]["recipe_sha256"]
        == bind(server).training_config["training"]["recipe_sha256"]
        == recipe.identity_sha256()
    )


def test_bound_recipe_is_independent_of_mutable_run_input() -> None:
    recipe = frozen_recipe()
    manifest = run_manifest()
    manifest["training"].update(recipe.run_training_fields())
    identity = bind(manifest)
    manifest["training"]["recipe"]["trainer"]["max_steps"] = 1
    assert identity.training_config["training"]["recipe"] == recipe.to_dict()


def test_recipe_digest_survives_checkpoint_write_and_readback(tmp_path) -> None:
    recipe = frozen_recipe()
    manifest = run_manifest()
    manifest["training"].update(recipe.run_training_fields())
    identity = bind(manifest)
    path = tmp_path / "recipe-checkpoint"
    save_checkpoint(path, model=nn.Linear(1, 5070), identity=identity)
    saved = verify_checkpoint(path)["identity"]["training_config"]["training"]
    assert saved["recipe"] == recipe.to_dict()
    assert saved["recipe_sha256"] == recipe.identity_sha256()


@pytest.mark.parametrize(
    "key", ["recipe", "recipe_sha256", "optimizer", "precision", "global_batch_tokens"]
)
def test_missing_or_replaced_recipe_binding_is_rejected(key) -> None:
    manifest = run_manifest()
    manifest["training"].update(frozen_recipe().run_training_fields())
    manifest["training"].pop(key)
    with pytest.raises(CheckpointCompatibilityError, match="recipe"):
        bind(manifest)


def test_recipe_rejects_over_budget_checkpoint_and_packing_drift() -> None:
    recipe = frozen_recipe()
    manifest = run_manifest()
    manifest["training"].update(replace(recipe, sequence_length=64).run_training_fields())
    with pytest.raises(CheckpointCompatibilityError, match="packing sequence length"):
        bind(manifest)
    smaller_budget = replace(
        recipe, target_unique_loss_positions=512, maximum_total_exposures=1024
    )
    manifest["training"].update(smaller_budget.run_training_fields())
    with pytest.raises(CheckpointCompatibilityError, match="recipe budget"):
        bind(manifest)


def test_recipe_context_cannot_exceed_the_model() -> None:
    manifest = run_manifest()
    recipe = replace(frozen_recipe(), sequence_length=256)
    manifest["training"].update(recipe.run_training_fields())
    with pytest.raises(CheckpointCompatibilityError, match="ModelSpec sequence length"):
        bind_checkpoint_identity(
            run_manifest=manifest,
            model_spec=model_spec(),
            init_spec=init_spec(),
            tokenizer_identity=tokenizer_identity(),
            packing_identity={**packing_identity(), "sequence_length": 256},
            step=0,
            tokens_seen=0,
        )


def test_recipe_wire_schema_rejects_provider_fields_and_invalid_budget() -> None:
    recipe = frozen_recipe()
    payload = recipe.to_dict()
    payload["trainer"]["device"] = "cpu"
    with pytest.raises(ValueError, match="TrainerConfig field"):
        TrainingRecipe.from_dict(payload)
    with pytest.raises(ValueError, match="exposure budget"):
        replace(recipe, maximum_total_exposures=5121)


def test_rejects_model_tokenizer_vocab_drift() -> None:
    tokenizer = tokenizer_identity()
    tokenizer["vocab_size"] = 259
    with pytest.raises(CheckpointCompatibilityError, match="vocab mismatch"):
        bind_checkpoint_identity(
            run_manifest=run_manifest(),
            model_spec=model_spec(),
            init_spec=init_spec(),
            tokenizer_identity=tokenizer,
            packing_identity=packing_identity(),
            step=0,
            tokens_seen=0,
            environment_lock_hash=ENV_LOCK_SHA,
        )


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        (("candidate", "modelspec_sha256"), "e" * 64, "ModelSpec hash"),
        (("candidate", "initspec_sha256"), "e" * 64, "InitSpec hash"),
        (("data", "tokenizer_sha256"), "d" * 64, "tokenizer SHA"),
        (("data", "tokenizer_vocab_sha256"), "d" * 64, "vocabulary SHA"),
        (("data", "packing_sha256"), "d" * 64, "packing SHA"),
        (("data", "packing_version"), "wrong-pack", "packing version"),
        (("data", "split_identity"), "UNRESOLVED", "split_identity"),
        (("environment", "lock_sha256"), "d" * 64, "environment lock SHA"),
    ],
)
def test_rejects_canonical_identity_drift(path, value, match) -> None:
    manifest = run_manifest()
    parent, key = path
    manifest[parent][key] = value
    with pytest.raises(CheckpointCompatibilityError, match=match):
        bind(manifest)


def test_rejects_supplied_environment_lock_drift() -> None:
    with pytest.raises(CheckpointCompatibilityError, match="environment lock SHA"):
        bind_checkpoint_identity(
            run_manifest=run_manifest(),
            model_spec=model_spec(),
            init_spec=init_spec(),
            tokenizer_identity=tokenizer_identity(),
            packing_identity=packing_identity(),
            step=0,
            tokens_seen=0,
            environment_lock_hash="d" * 64,
        )


def test_training_config_and_seed_are_transitively_bound_by_run_manifest_hash() -> None:
    original = run_manifest()
    changed = copy.deepcopy(original)
    changed["training"]["seed"] = 18
    changed["training"]["optimizer"]["lr"] = 0.002

    original_identity = bind(original)
    changed_identity = bind(changed)

    assert original_identity.run_manifest_hash != changed_identity.run_manifest_hash
    assert original_identity.training_config != changed_identity.training_config
    assert original_identity.seed == 17
    assert changed_identity.seed == 18


def test_rejects_abbreviated_git_sha_and_unresolved_run() -> None:
    manifest = run_manifest()
    manifest["candidate"]["git_sha"] = "abcdef0"
    with pytest.raises(CheckpointCompatibilityError, match="full lowercase 40- or 64-hex Git SHA"):
        bind(manifest)

    unresolved = copy.deepcopy(run_manifest())
    unresolved["run_id"] = "UNRESOLVED"
    with pytest.raises(CheckpointCompatibilityError, match="run_id.*resolved"):
        bind(unresolved)
