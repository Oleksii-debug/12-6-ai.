from dataclasses import replace
from pathlib import Path

import pytest
import torch

from tools import plan2_s18_physical_release_v1 as physical_release
from tools import plan9_scale500m_launch_preparation_v1 as preparation
from twelve_six.checkpoint import hash_json
from twelve_six.training.config import TrainerConfig, TrainingRecipe

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def verified_boundary(tmp_path, monkeypatch):
    # Unit boundary only. The deployed path always calls the Plan 2 verifier.
    model, _ = preparation._load_profile(ROOT)
    dataset = {"manifest_sha256": "1" * 64, "records": 4}
    split = {
        "manifest_sha256": "2" * 64,
        "s16_admission_manifest_sha256": dataset["manifest_sha256"],
        "roles": {
            "train": {"source_membership_sha256": dataset["manifest_sha256"], "records": 4},
            "validation": {"manifest_sha256": "3" * 64},
            "test": {"manifest_sha256": "4" * 64},
        },
    }
    tokenizer = {
        "tokenizer_identity": {
            "vocab_size": 32768, "config_sha256": "a" * 64, "vocab_sha256": "b" * 64, "version": "test-v1",
        },
        "source_manifest_sha256": "5" * 64,
    }
    packing = {
        "block_size": 32,
        "block_count": 4,
        "schema_version": "test-packing-v1",
        "manifest_sha256": "6" * 64,
        "split": "train",
        "cluster_split_sha256": split["manifest_sha256"],
        "tokenizer_manifest_sha256": tokenizer["source_manifest_sha256"],
        "target_count": 20,
    }
    consumer = {
        "weights_origin": "random_init",
        "model_training_executed": False,
        "plan4_reference_inference_verified": True,
        "ordered_target_count": 20,
        "tokenizer_manifest_sha256": tokenizer["source_manifest_sha256"],
        "split_manifest_sha256": split["manifest_sha256"],
        "packing_manifest_sha256": packing["manifest_sha256"],
        "exposure_manifest_sha256": "7" * 64,
        "manifest_sha256": "8" * 64,
        "profile_compatibility": {
            preparation.PROFILE: {
                "model_spec_sha256": model.model.identity_sha256(),
                "init_spec_sha256": model.init.identity_sha256(),
                "parameters": model.model.parameter_count(),
                "vocab_size": model.model.vocab_size,
            },
        },
    }
    members = {}
    for name, payload in (
        ("plan9-binding.json", {"campaign_authorized": False, "physical_consumer": consumer}),
        ("dataset-manifest.json", dataset),
        ("tokenizer-manifest.json", tokenizer),
        ("split-manifest.json", split),
        ("packing-manifest.json", packing),
    ):
        raw = preparation._canonical(payload)
        (tmp_path / name).write_bytes(raw)
        members[name] = preparation._sha(raw)
    release = preparation._canonical({"members": members})
    (tmp_path / "release-manifest.json").write_bytes(release)
    authority = {
        "plan9_data_release_verified": True,
        "campaign_authorized": False,
        "release_manifest_sha256": preparation._sha(release),
        "authority_sha256": "9" * 64,
        "training_targets": consumer["ordered_target_count"],
    }
    monkeypatch.setattr(preparation, "_open_release", lambda *_: authority)
    return tmp_path, authority


def test_binds_primary_inputs_without_launching(verified_boundary):
    directory, _ = verified_boundary
    result = preparation.build_preparation(ROOT, directory)
    assert result["profile"]["parameters"] == 494_204_160
    assert result["profile"]["initialization"]["ancestor_checkpoint_sha256"] is None
    assert result["data"]["ordered_causal_loss_positions"] == 20
    assert result["recipe_sha256"] is None
    assert result["campaign_authorized"] is result["training_executed"] is False
    assert result == preparation.build_preparation(ROOT, directory)


def test_rejects_descriptor_changed_after_release_verification(verified_boundary):
    directory, _ = verified_boundary
    (directory / "split-manifest.json").write_bytes(b"{}\n")
    with pytest.raises(preparation.LaunchPreparationDenied, match="pinned input changed"):
        preparation.build_preparation(ROOT, directory)


def test_rejects_campaign_authority_in_data_handoff(verified_boundary):
    directory, authority = verified_boundary
    authority["campaign_authorized"] = True
    with pytest.raises(preparation.LaunchPreparationDenied, match="without campaign authority"):
        preparation.build_preparation(ROOT, directory)


@pytest.fixture
def candidate_recipe():
    return TrainingRecipe(
        trainer=TrainerConfig(max_steps=2),
        microbatch_size=2,
        sequence_length=32,
        target_unique_loss_positions=20,
        maximum_total_exposures=20,
        max_exposures_per_unique_position=1,
    )


def test_binds_recipe_identity_without_campaign_acceptance(verified_boundary, candidate_recipe):
    directory, _ = verified_boundary
    result = preparation.build_preparation(ROOT, directory, recipe=candidate_recipe)
    assert result["recipe_sha256"] == candidate_recipe.identity_sha256()
    assert result["candidate_training_fields"] == candidate_recipe.run_training_fields()
    assert result["data"]["packed_block_size"] == candidate_recipe.sequence_length
    assert result["campaign_recipe_accepted"] is result["campaign_authorized"] is False
    assert result["checkpoint_resume_contract"]["d04_plan_and_checkpoint_bound"] is False


def test_rejects_recipe_beyond_physical_width_and_unique_supply(verified_boundary, candidate_recipe):
    directory, _ = verified_boundary
    with pytest.raises(preparation.LaunchPreparationDenied, match="match the released width"):
        preparation.build_preparation(ROOT, directory, recipe=replace(candidate_recipe, sequence_length=2048))
    too_many = replace(candidate_recipe, target_unique_loss_positions=21, maximum_total_exposures=21)
    with pytest.raises(preparation.LaunchPreparationDenied, match="exceeds the released"):
        preparation.build_preparation(ROOT, directory, recipe=too_many)


def test_resume_revalidates_prefix_and_recipe_before_yield(verified_boundary, candidate_recipe, monkeypatch):
    directory, _ = verified_boundary
    batches = [
        {"input_ids": torch.full((2, 32), index), "loss_mask": torch.tensor([[1] * 5 + [0] * 27] * 2)}
        for index in (10, 11)
    ]

    def iter_batches(*_, batch_size):
        assert batch_size == 2
        yield from batches

    monkeypatch.setattr(physical_release, "iter_release_batches", iter_batches)
    cursor = preparation.build_resume_cursor(
        ROOT, directory, recipe=candidate_recipe, next_batch_index=1,
    )
    assert cursor["consumed_blocks"] == 2
    assert cursor["consumed_loss_positions"] == 10

    def resume(record, recipe=candidate_recipe):
        return preparation.iter_resumed_release_batches(
            ROOT, directory, recipe=recipe, cursor=record, expected_cursor_sha256=cursor["cursor_sha256"],
        )

    assert list(resume(cursor)) == batches[1:]
    with pytest.raises(preparation.LaunchPreparationDenied, match="digest mismatch"):
        next(resume({**cursor, "consumed_blocks": 2.0}))
    changed = replace(candidate_recipe, trainer=replace(candidate_recipe.trainer, seed=17))
    with pytest.raises(preparation.LaunchPreparationDenied, match="verified physical prefix"):
        next(resume(cursor, changed))

    terminal = preparation.build_resume_cursor(
        ROOT, directory, recipe=candidate_recipe, next_batch_index=2,
    )
    assert terminal["complete"] is True
    assert terminal["consumed_loss_positions"] == 20
    assert list(preparation.iter_resumed_release_batches(
        ROOT, directory, recipe=candidate_recipe, cursor=terminal, expected_cursor_sha256=terminal["cursor_sha256"],
    )) == []


def test_binds_checkpoint_progress_without_changing_scientific_run_identity(
    verified_boundary, candidate_recipe, monkeypatch,
):
    directory, _ = verified_boundary
    batch = {"loss_mask": torch.tensor([[1] * 5 + [0] * 27] * 2)}
    monkeypatch.setattr(physical_release, "iter_release_batches", lambda *_, **__: iter([batch, batch]))

    def bind(index, step):
        cursor = preparation.build_resume_cursor(
            ROOT, directory, recipe=candidate_recipe, next_batch_index=index,
        )
        identity = preparation.bind_resume_checkpoint_identity(
            ROOT, directory, recipe=candidate_recipe, cursor=cursor,
            expected_cursor_sha256=cursor["cursor_sha256"],
            candidate_git_sha="e" * 40, environment_lock_sha256="f" * 64,
            run_id="test-checkpoint-binding", optimizer_step=step,
        )
        identity.validate()
        assert identity.training_config["data"]["physical_resume_cursor_sha256"] == cursor["cursor_sha256"]
        return identity

    first, second = bind(1, 1), bind(2, 2)
    assert first.run_manifest_hash == second.run_manifest_hash
    assert first.tokens_seen == 10 and second.tokens_seen == 20
    assert hash_json(first.training_config) != hash_json(second.training_config)
    with pytest.raises(preparation.LaunchPreparationDenied, match="optimizer checkpoint boundary"):
        bind(1, 2)
