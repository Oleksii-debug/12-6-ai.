from pathlib import Path

import pytest

from tools import plan9_scale500m_launch_preparation_v1 as preparation

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
    tokenizer = {"tokenizer_identity": {"vocab_size": 32768}, "source_manifest_sha256": "5" * 64}
    consumer = {
        "weights_origin": "random_init",
        "model_training_executed": False,
        "plan4_reference_inference_verified": True,
        "ordered_target_count": 20,
        "tokenizer_manifest_sha256": tokenizer["source_manifest_sha256"],
        "split_manifest_sha256": split["manifest_sha256"],
        "packing_manifest_sha256": "6" * 64,
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
