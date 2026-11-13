import json
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tools import plan2_s18_physical_release_v1 as physical_release
from tools import plan2_s18_release_integrity_v1 as release_integrity
from tools import plan9_scale500m_launch_preparation_v1 as preparation
from twelve_six.checkpoint import hash_json
from twelve_six.checkpoint.transaction import bind_checkpoint_transaction, checkpoint_transaction
from twelve_six.training import runtime_release
from twelve_six.training.config import TrainerConfig, TrainingRecipe
from twelve_six.training.resource_envelope import HostResources

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def verified_boundary(tmp_path, monkeypatch):
    # Unit boundary only. The deployed path always calls the Plan 2 verifier.
    model, _ = preparation._load_profile(ROOT)
    dataset = {"manifest_sha256": "1" * 64, "records": 4, "selected_source_terms": {"test-family": {}}}
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
    notice_path = tmp_path / "notices" / "NOTICE.txt"
    notice_path.parent.mkdir()
    notice_path.write_bytes(b"unit boundary notice\n")
    notice_key = "notices/NOTICE.txt"
    members[notice_key] = preparation._sha(notice_path.read_bytes())
    rights = {
        "dataset_sha256": preparation._sha(preparation._canonical(dataset)),
        "scope": "LOCAL_PLAN9_TRAINING_DATA_HANDOFF",
        "evaluation_payloads_included": False,
        "public_redistribution_authorized": False,
        "selected_source_terms": dataset["selected_source_terms"],
        "notice_sha256": {notice_key: members[notice_key]},
    }
    for name, payload in (
        ("plan9-binding.json", {"campaign_authorized": False, "physical_consumer": consumer}),
        ("dataset-manifest.json", dataset),
        ("tokenizer-manifest.json", tokenizer),
        ("split-manifest.json", split),
        ("packing-manifest.json", packing),
        ("rights-attribution.json", rights),
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
    handoff = {
        "release_manifest_sha256": authority["release_manifest_sha256"],
        "release_authority_sha256": authority["authority_sha256"],
        "plan9_binding_sha256": members["plan9-binding.json"],
        "training_targets": authority["training_targets"],
        "vocab_size": model.model.vocab_size,
        "first_batch_shape": [2, packing["block_size"]],
    }
    monkeypatch.setattr(runtime_release, "verify_runtime_release", lambda *_, **_kw: {
        "qualification": "LOCAL_COMPONENT",
        "full_profile_measurement_status": "OPEN_EXTERNAL",
        "campaign_authorized": False,
        "plan2_handoff": handoff,
    })
    monkeypatch.setattr(runtime_release, "verify_plan2_handoff", lambda *_, **_kw: {
        "runtime_release_sha256": runtime_release.RELEASE_V2_SHA256,
        "plan2_handoff": handoff,
        "optimizer_steps_executed": 0,
        "campaign_authorized": False,
    })
    return tmp_path, authority


def test_binds_primary_inputs_without_launching(verified_boundary):
    directory, _ = verified_boundary
    result = preparation.build_preparation(ROOT, directory)
    assert result["profile"]["parameters"] == 494_204_160
    assert result["profile"]["initialization"]["ancestor_checkpoint_sha256"] is None
    assert result["data"]["ordered_causal_loss_positions"] == 20
    assert result["recipe_sha256"] is None
    assert result["plan3_runtime_release"]["descriptor_sha256"] == runtime_release.RELEASE_V2_SHA256
    assert result["plan3_runtime_release"]["full_profile_measurement_status"] == "OPEN_EXTERNAL"
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


def test_runtime_release_and_live_handoff_must_bind_the_same_plan2_authority(
    verified_boundary, candidate_recipe, monkeypatch,
):
    directory, _ = verified_boundary
    probe = runtime_release.verify_plan2_handoff(ROOT, directory, release_version=2)
    probe["plan2_handoff"] = {**probe["plan2_handoff"], "release_authority_sha256": "e" * 64}
    monkeypatch.setattr(runtime_release, "verify_plan2_handoff", lambda *_, **_kw: probe)
    with pytest.raises(preparation.LaunchPreparationDenied, match="physical Plan 2 handoff differs"):
        preparation.build_candidate_packet(
            ROOT, directory, recipe=candidate_recipe, run_id="inconsistent-runtime",
            environment_lock_sha256="d" * 64,
        )


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


def test_run_controller_cursor_hooks_preserve_exact_checkpoint_next_batch(
    verified_boundary, candidate_recipe, monkeypatch,
):
    directory, _ = verified_boundary
    batches = [
        {"input_ids": torch.full((2, 32), index), "loss_mask": torch.tensor([[1] * 5 + [0] * 27] * 2)}
        for index in (10, 11)
    ]
    monkeypatch.setattr(physical_release, "iter_release_batches", lambda *_, **__: iter(batches))
    lock = "d" * 64
    packet = preparation.build_candidate_packet(
        ROOT, directory, recipe=candidate_recipe, run_id="controller-input", environment_lock_sha256=lock,
    )

    def open_input():
        return preparation.prepare_run_controller_input(
            ROOT, directory, packet=packet, expected_packet_sha256=packet["packet_sha256"],
            environment_lock_sha256=lock,
        )

    identity, initial, source = open_input()
    assert identity.run_manifest_hash == packet["run_manifest_sha256"]
    assert identity.step == identity.tokens_seen == 0
    assert "physical_resume_cursor_sha256" not in identity.training_config["data"]
    hashes = packet["checkpoint_preflight"]
    assert hashes["controller_base_training_config_sha256"] == hash_json(identity.training_config)
    assert hashes["legacy_expected_training_config_hash"] != hashes["controller_base_training_config_sha256"]
    assert packet["preparation"]["runtime_source"]["checkpoint_transaction"]["path"] == (
        "src/twelve_six/checkpoint/transaction.py"
    )
    source.restore_exposure_cursor(initial)
    stream = source.batches(SimpleNamespace(committed_step=0, tokens_seen=0))
    assert next(stream) is batches[0]
    with pytest.raises(preparation.LaunchPreparationDenied, match="trainer progress differs"):
        source.read_exposure_cursor(SimpleNamespace(micro_step=1, tokens_seen=9))
    cursor = source.read_exposure_cursor(SimpleNamespace(micro_step=1, tokens_seen=10))
    txn = bind_checkpoint_transaction(
        replace(identity, step=1, tokens_seen=10), generation=1,
        parent_generation=0, parent_checkpoint_id="f" * 64, exposure_cursor=cursor,
    )
    restored = checkpoint_transaction({"identity": asdict(txn)})
    assert restored is not None
    _, _, fresh = open_input()
    fresh.restore_exposure_cursor(restored.exposure_cursor)
    with pytest.raises(preparation.LaunchPreparationDenied, match="run receipt progress differs"):
        next(fresh.batches(SimpleNamespace(committed_step=1, tokens_seen=9)))
    assert list(fresh.batches(SimpleNamespace(committed_step=1, tokens_seen=10))) == batches[1:]
    with pytest.raises(preparation.LaunchPreparationDenied, match="restore the verified exposure cursor"):
        next(fresh.batches(SimpleNamespace(committed_step=1, tokens_seen=10)))


def test_run_controller_input_refuses_a_batch_before_exposure_budget_is_exceeded(
    verified_boundary, candidate_recipe, monkeypatch,
):
    directory, _ = verified_boundary
    recipe = replace(candidate_recipe, target_unique_loss_positions=15, maximum_total_exposures=15)
    batch = {"input_ids": torch.zeros((2, 32)), "loss_mask": torch.tensor([[1] * 5 + [0] * 27] * 2)}
    monkeypatch.setattr(physical_release, "iter_release_batches", lambda *_, **__: iter([batch, batch]))
    source = preparation.PhysicalRunInput(ROOT, directory, recipe)
    source.restore_exposure_cursor(preparation.build_resume_cursor(ROOT, directory, recipe=recipe, next_batch_index=0))
    batches = source.batches(SimpleNamespace(committed_step=0, tokens_seen=0))
    assert next(batches) is batch
    with pytest.raises(preparation.LaunchPreparationDenied, match="exceed the sealed recipe"):
        next(batches)
    assert source.read_exposure_cursor(SimpleNamespace(micro_step=1, tokens_seen=10))["next_batch_index"] == 1


def test_candidate_packet_rebuild_pins_run_and_rejects_tampering(verified_boundary, candidate_recipe):
    directory, _ = verified_boundary
    inputs = {"environment_lock_sha256": "d" * 64}
    packet = preparation.build_candidate_packet(
        ROOT, directory, recipe=candidate_recipe, run_id="test-candidate", **inputs,
    )
    assert packet["run_manifest_sha256"] == hash_json(packet["run_manifest"])
    assert packet["run_manifest"]["training"]["recipe_sha256"] == candidate_recipe.identity_sha256()
    assert packet["run_manifest"]["candidate"]["git_sha"] == preparation.detect_git_sha(ROOT)
    assert packet["run_manifest"]["plan3_runtime_release"] == packet["preparation"]["plan3_runtime_release"]
    assert packet["plan3_physical_handoff"] == {"aligned_loss_verified": True, "optimizer_steps_executed": 0}
    assert packet["checkpoint_preflight"]["checkpoint_transaction_verified"] is False
    assert packet["checkpoint_preflight"]["checkpoint_artifact_verified"] is False
    assert packet["environment_lock_qualification"] == "PINNED_BYTES_ONLY"
    assert packet["run_manifest"]["campaign_authorized"] is packet["campaign_authorized"] is False

    def verify(record, digest=packet["packet_sha256"], **changes):
        return preparation.verify_candidate_packet(
            ROOT, directory, packet=record, expected_packet_sha256=digest, **{**inputs, **changes},
        )

    assert verify(packet) == packet
    with pytest.raises(preparation.LaunchPreparationDenied, match="trusted identity"):
        verify(packet, "0" * 64)
    with pytest.raises(preparation.LaunchPreparationDenied, match="digest mismatch"):
        verify({**packet, "campaign_authorized": True})
    forged = {**packet, "campaign_authorized": True}
    forged.pop("packet_sha256")
    forged_digest = preparation._sha(preparation._canonical(forged))
    with pytest.raises(preparation.LaunchPreparationDenied, match="verified release"):
        verify({**forged, "packet_sha256": forged_digest}, forged_digest)
    with pytest.raises(preparation.LaunchPreparationDenied, match="verified release"):
        verify(packet, environment_lock_sha256="e" * 64)


def test_cli_candidate_checkout_rejects_runtime_drift(monkeypatch):
    read = preparation._read
    trainer = ROOT / preparation.RUNTIME_PATHS["trainer"]

    def read_changed(path, limit=2_000_000):
        raw = read(path, limit)
        return raw + b"# runtime changed\n" if path == trainer else raw

    monkeypatch.setattr(preparation, "_read", read_changed)
    with pytest.raises(preparation.LaunchPreparationDenied, match="candidate source differs from Git HEAD"):
        preparation._candidate_checkout_sha(ROOT)


def test_candidate_source_rejects_preloaded_foreign_dependency_in_a_fresh_process(tmp_path):
    relative = preparation.RUNTIME_PATHS["dependency_lock_verifier"]
    foreign = tmp_path / "dependency_lock.py"
    foreign.write_bytes((ROOT / relative).read_bytes())
    script = """
import importlib.util
import sys
from pathlib import Path

name = "twelve_six.integration.dependency_lock"
spec = importlib.util.spec_from_file_location(name, sys.argv[1])
module = importlib.util.module_from_spec(spec)
sys.modules[name] = module
spec.loader.exec_module(module)
from tools import plan9_scale500m_launch_preparation_v1 as preparation

assert preparation.dependency_lock is module
try:
    preparation._candidate_checkout_sha(Path.cwd())
except preparation.LaunchPreparationDenied as exc:
    assert "loaded source differs from candidate checkout: " + name == str(exc)
else:
    raise AssertionError("foreign dependency was accepted")
"""
    subprocess.run([sys.executable, "-c", script, str(foreign)], cwd=ROOT, check=True)


def test_candidate_source_checks_transitive_physical_release_helpers(monkeypatch):
    read = preparation._read
    helper = ROOT / "tools/plan2_physical_materialization_v1.py"

    def read_changed(path, limit=2_000_000):
        raw = read(path, limit)
        return raw + b"# transitive source changed\n" if path == helper else raw

    monkeypatch.setattr(preparation, "_read", read_changed)
    with pytest.raises(
        preparation.LaunchPreparationDenied, match=r"candidate source differs.*plan2_physical_materialization",
    ):
        preparation._candidate_checkout_sha(ROOT)


def test_native_lock_verifies_exact_git_bytes_in_crlf_checkout(monkeypatch):
    read = preparation._read

    def read_crlf(path, limit=2_000_000):
        value = read(path, limit)
        return value.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")

    monkeypatch.setattr(preparation, "_read", read_crlf)
    sha = preparation.detect_git_sha(ROOT)
    lock = preparation._native_lock_binding(ROOT, sha)
    assert lock["index_sha256"] == preparation.LOCK_INDEX_SHA256
    assert lock["profile_id"] == preparation.dependency_lock.current_profile_id()

    def changed(path, limit=2_000_000):
        value = read_crlf(path, limit)
        return value + b"# invalid\n" if path.name == "runtime.lock.txt" else value

    monkeypatch.setattr(preparation, "_read", changed)
    with pytest.raises(preparation.LaunchPreparationDenied, match="candidate source differs from Git HEAD"):
        preparation._native_lock_binding(ROOT, sha)


def test_external_gpu_profile_plan_has_independent_specs_and_native_floors(monkeypatch):
    def reject_probe(*_):
        pytest.fail("arithmetic planning must not open a data release or observe a host")

    monkeypatch.setattr(preparation, "observe_host_resources", reject_probe)
    monkeypatch.setattr(preparation, "_open_release", reject_probe)
    plan = preparation.plan_external_gpu_profiles(ROOT)
    assert plan == preparation.plan_external_gpu_profiles(ROOT)
    assert plan["profile_preparation_order"] == ["SCALE-500M", "SCALE-1B", "SCALE-2B"]
    small, medium, large = plan["profiles"]
    assert [profile["profile"] for profile in (small, medium, large)] == plan["profile_preparation_order"]
    assert [profile["parameters"] for profile in (small, medium, large)] == [494_204_160, 1_018_983_168, 1_996_654_592]
    assert plan["resource_evidence"] == "ARITHMETIC_ONLY"
    assert plan["target_profile_measured"] is plan["campaign_authorized"] is plan["training_executed"] is False
    assert small["general_baseline_recipe_sha256"] == preparation.ONE_PASS_RECIPE_SHA256
    assert small["general_baseline_optimizer_updates"] == 2_969
    assert small["resource_floors"]["minimum_unpruned_checkpoint_volume_bytes"] == 18_107_765_944_320
    for profile in (medium, large):
        assert profile["general_baseline_recipe_sha256"] is None
        assert profile["general_baseline_optimizer_updates"] is None
        assert profile["campaign_recipe_sha256"] is profile["campaign_data_release_sha256"] is None
        floors = profile["resource_floors"]
        assert floors["minimum_unpruned_checkpoint_volume_bytes"] is None
        assert floors["minimum_optimizer_checkpoint_payload_bytes"] == (
            floors["minimum_initial_checkpoint_payload_bytes"] + 8 * profile["parameters"]
        )
        assert floors["minimum_device_bytes_at_native_save"] == 28 * profile["parameters"]


def test_external_gpu_packet_uses_pinned_linux_and_one_replica_without_host_probe(monkeypatch):
    recipe = TrainingRecipe.from_dict(
        json.loads((ROOT / preparation.ONE_PASS_RECIPE_FILE).read_bytes()),
    )
    model, _ = preparation._load_profile(ROOT)
    candidate = {
        "model_spec": model.model.to_dict(),
        "run_manifest": {"candidate": {"git_sha": preparation.detect_git_sha(ROOT)}, "run_id": "gpu-baseline"},
        "preparation": {"data": {
            "release_manifest_sha256": "883f0e3949b7a64a07bcc1236bf7aaf86ed5d84deba4c5aa60d2bfcb83c21ad2",
            "release_authority_sha256": "858bf6852fcd418ff50e7726d5e8e6e5ca6efb2075bc86166361494f2c39abda",
            "ordered_causal_loss_positions": 1_985_122,
            "packed_block_count": 71_256,
        }},
    }
    seen = []

    def build_candidate(*_, **kwargs):
        assert kwargs["recipe"] == recipe
        seen.append(kwargs["environment_lock_sha256"])
        return candidate

    def reject_host_probe(*_):
        pytest.fail("external baseline must not observe the current host")

    monkeypatch.setattr(preparation, "build_candidate_packet", build_candidate)
    monkeypatch.setattr(preparation, "observe_host_resources", reject_host_probe)
    monkeypatch.setattr(preparation.dependency_lock, "current_profile_id", reject_host_probe)
    packet = preparation.build_external_gpu_packet(ROOT, ROOT, lock_profile="linux-x86_64", run_id="gpu-baseline")
    execution = packet["execution_requirements"]
    assert execution["gpu_count"] == 1 and execution["optimizer_state_layout"] == "ONE_FULL_REPLICA"
    assert execution["minimum_device_bytes_at_native_save"] == 13_837_716_480
    assert execution["minimum_unpruned_checkpoint_volume_bytes"] == 18_107_765_944_320
    assert execution["environment_lock"]["profile_id"] == "linux-x86_64"
    assert seen == [execution["environment_lock"]["runtime_lock_sha256"]]
    assert packet["profile_preparation_order"] == ["SCALE-500M", "SCALE-1B", "SCALE-2B"]
    assert packet["data"]["purpose"] == "GENERAL_BASELINE"
    assert packet["data"]["cybersecurity_majority_verified"] is packet["campaign_authorized"] is False
    assert packet["target_profile_measured"] is packet["training_executed"] is False

    def verify(record, digest):
        return preparation.verify_external_gpu_packet(ROOT, ROOT, packet=record, expected_packet_sha256=digest)

    assert verify(packet, packet["external_gpu_packet_sha256"]) == packet
    modified = {**packet, "data": {**packet["data"], "cybersecurity_majority_verified": True}}
    modified.pop("external_gpu_packet_sha256")
    digest = preparation._sha(preparation._canonical(modified))
    with pytest.raises(preparation.LaunchPreparationDenied, match="differs from the verified baseline"):
        verify({**modified, "external_gpu_packet_sha256": digest}, digest)
    with pytest.raises(preparation.LaunchPreparationDenied, match="requires a committed Linux lock"):
        preparation.build_external_gpu_packet(
            ROOT, ROOT, lock_profile="windows-x86_64", run_id="gpu-baseline",
        )


def test_native_preflight_binds_initial_packet_and_denies_runtime_and_capacity(
    verified_boundary, candidate_recipe, monkeypatch,
):
    directory, _ = verified_boundary
    lock = preparation._native_lock_binding(ROOT, preparation.detect_git_sha(ROOT))
    monkeypatch.setattr(preparation, "_native_lock_binding", lambda *_: {
        **lock, "observed_python": {"implementation": "cpython", "version": "0.0.0"},
    })
    monkeypatch.setattr(preparation, "observe_host_resources", lambda *args: HostResources("cpu", 1, 1, 1))
    packet = preparation.build_candidate_packet(
        ROOT, directory, recipe=candidate_recipe, run_id="native-preflight",
        environment_lock_sha256=lock["runtime_lock_sha256"],
    )
    report = preparation.inspect_native_launch(
        ROOT, directory, packet=packet, expected_packet_sha256=packet["packet_sha256"],
        checkpoint_root=directory / "not-created", device="cpu",
    )
    assert report["decision"] == "NATIVE_PREFLIGHT_DENIED"
    assert set(report["denials"]) == {
        "DEVICE_MEMORY_BELOW_FP32_ADAMW_FLOOR",
        "DISK_BELOW_UNPRUNED_CHECKPOINT_FLOOR",
        "PYTHON_RUNTIME_LOCK_MISMATCH",
    }
    assert report["resource_envelope"]["parameter_count"] == 494_204_160
    assert report["installed_distribution_provenance_verified"] is False
    assert report["target_profile_measured"] is report["campaign_authorized"] is False
    assert not (directory / "not-created").exists()
    wrong = preparation.build_candidate_packet(
        ROOT, directory, recipe=candidate_recipe, run_id="native-preflight",
        environment_lock_sha256="f" * 64,
    )
    with pytest.raises(preparation.LaunchPreparationDenied, match="verified release, source, lock"):
        preparation.inspect_native_launch(
            ROOT, directory, packet=wrong, expected_packet_sha256=wrong["packet_sha256"],
            checkpoint_root=directory / "not-created", device="cpu",
        )


def test_candidate_packet_rejects_fixture_sealed_under_its_own_manifest(tmp_path, candidate_recipe):
    members = {}
    for name in release_integrity.REQUIRED:
        raw = preparation._canonical({"fixture": True, "document": name})
        (tmp_path / name).write_bytes(raw)
        members[name] = preparation._sha(raw)
    manifest = {
        "schema_version": release_integrity.SCHEMA,
        "producer_commit_sha1": physical_release.S17_ACCEPTED_MAIN,
        "producer_tree_sha1": physical_release.S17_ACCEPTED_TREE,
        "members": members,
    }
    (tmp_path / release_integrity.MANIFEST).write_bytes(preparation._canonical(manifest))
    with pytest.raises(release_integrity.IntegrityDenied, match="trusted manifest SHA-256 mismatch"):
        preparation.build_candidate_packet(
            ROOT, tmp_path, recipe=candidate_recipe, run_id="fixture-cannot-enter",
            environment_lock_sha256="d" * 64,
        )


def test_exact_one_pass_budget_checks_real_mask_counts_and_rights(
    verified_boundary, candidate_recipe, monkeypatch,
):
    directory, _ = verified_boundary
    batch = {"loss_mask": torch.tensor([[1] * 5 + [0] * 27] * 2)}
    monkeypatch.setattr(physical_release, "iter_release_batches", lambda *_, **__: iter([batch, batch]))

    def assess(recipe=candidate_recipe):
        return preparation.check_one_pass_data_budget(ROOT, directory, recipe=recipe)

    result = assess()
    assert result["nominal_optimizer_steps"] == result["planned_microbatches"] == 2
    assert result["exact_one_pass_unique_loss_positions"] == result["exact_one_pass_total_exposures"] == 20
    assert result["notice_count"] == result["source_family_count"] == 1
    assert result["campaign_authorized"] is result["target_environment_qualified"] is False

    early_stop = replace(candidate_recipe, trainer=replace(candidate_recipe.trainer, max_steps=1))
    with pytest.raises(preparation.LaunchPreparationDenied, match="cannot meet the unique target"):
        assess(early_stop)
    cap = replace(candidate_recipe, target_unique_loss_positions=19, maximum_total_exposures=19)
    with pytest.raises(preparation.LaunchPreparationDenied, match="exceeds the candidate recipe exposure"):
        assess(cap)
    (directory / "notices" / "NOTICE.txt").write_bytes(b"changed after release verification\n")
    with pytest.raises(preparation.LaunchPreparationDenied, match="pinned notice changed"):
        assess()
