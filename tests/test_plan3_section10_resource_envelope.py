from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from safetensors.numpy import load_file as load_tensor_arrays
from test_plan3_section9_portable_packet import _batches, _settings
from test_plan3_section9_portable_packet import training_inputs as _portable_training_inputs

from twelve_six.checkpoint import CheckpointIntegrityError, verify_checkpoint
from twelve_six.model import build_canonical_base, load_stage_config
from twelve_six.training import Trainer, portable_packet
from twelve_six.training.portable_packet import PortableTrainingRun
from twelve_six.training.resource_envelope import (
    HostResources,
    ResourceAdmissionError,
    assess_resource_envelope,
    benchmark_optimizer_boundary,
    observe_host_resources,
)
from twelve_six.training.run_control import RunController

_ROOT = Path(__file__).resolve().parents[1]
training_inputs = _portable_training_inputs


@pytest.mark.parametrize(
    ("stage", "count", "disk_denied"),
    [
        ("500m", 494_204_160, False),
        ("750m", 755_842_560, True),
        ("1b", 1_018_983_168, True),
        ("2b", 1_996_654_592, True),
        ("3b", 3_005_966_592, True),
        ("4b", 4_003_207_680, True),
        ("5b", 5_017_022_208, True),
    ],
)
def test_exact_scale_admission_uses_observed_capacity_without_model_allocation(
    stage,
    count,
    disk_denied,
):
    config = load_stage_config(_ROOT / "configs" / "stages" / f"scale_{stage}.json")
    assert config.expected_parameters == config.model.parameter_count() == count
    result = assess_resource_envelope(
        config.model,
        HostResources("cpu", 5 * 2**30, 5 * 2**30, 10 * 2**30),
        remaining_updates=1,
        initial_checkpoint_needed=True,
    )
    assert result["fp32_adamw_device_bytes"] == 16 * count
    assert result["native_checkpoint_save_device_bytes"] == 28 * count
    tied = config.model.vocab_size * config.model.d_model if config.model.tie_embeddings else 0
    assert result["serialized_model_weight_bytes"] == 4 * (count + tied)
    assert result["unpruned_checkpoint_volume_floor_bytes"] == 16 * count + 8 * tied
    assert "DEVICE_MEMORY_BELOW_FP32_ADAMW_FLOOR" in result["denials"]
    assert ("DISK_BELOW_UNPRUNED_CHECKPOINT_FLOOR" in result["denials"]) is disk_denied
    assert result["evidence_class"] == "OBSERVED_CAPACITY_WITH_ARITHMETIC_FLOORS"


def test_native_save_and_cuda_staging_can_rule_out_otherwise_fitting_optimizer(training_inputs):
    packet, _ = training_inputs
    model = packet.model_spec
    count = model.parameter_count()
    tied = model.vocab_size * model.d_model
    weight_bytes = 4 * (count + tied)
    payload_bytes = weight_bytes + 8 * count
    result = assess_resource_envelope(
        model,
        HostResources(
            "cuda:0", payload_bytes - 1, 16 * count, weight_bytes + 2 * payload_bytes - 1
        ),
        remaining_updates=2,
        initial_checkpoint_needed=True,
    )
    assert result["denials"] == [
        "DEVICE_MEMORY_BELOW_NATIVE_SAVE_FLOOR",
        "HOST_MEMORY_BELOW_CHECKPOINT_STAGING_FLOOR",
        "DISK_BELOW_UNPRUNED_CHECKPOINT_FLOOR",
    ]


def test_portable_prepare_denies_impossible_resources_before_weight_allocation(
    training_inputs,
    tmp_path,
    monkeypatch,
):
    packet, refs = training_inputs
    root = tmp_path / "denied"
    run = PortableTrainingRun(
        packet,
        _settings(packet, root),
        expected_packet_sha256=packet.identity_sha256(),
        sources={"local": lambda _, index: _batches(index)},
        **refs,
    )

    def observe(path, device):
        assert path == root / run.run_manifest_sha256 / "checkpoints"
        assert device == "cpu"
        return HostResources("cpu", 1, 1, 1)

    monkeypatch.setattr(portable_packet, "observe_host_resources", observe)

    def unexpected(*_args):
        pytest.fail("impossible resource envelope reached model allocation")

    monkeypatch.setattr(portable_packet, "build_canonical_base", unexpected)
    with pytest.raises(ResourceAdmissionError, match="DEVICE_MEMORY_BELOW_FP32_ADAMW_FLOOR"):
        run.prepare()
    assert not root.exists()


def test_resource_observations_do_not_change_scientific_packet(
    training_inputs,
    tmp_path,
    monkeypatch,
):
    packet, refs = training_inputs
    scientific = packet.identity_sha256()
    manifest = packet.checkpoint_identity().run_manifest_hash
    observed = iter(
        [
            HostResources("cpu", 1 * 2**30, 1 * 2**30, 2 * 2**30),
            HostResources("cpu", 2 * 2**30, 2 * 2**30, 3 * 2**30),
        ]
    )
    monkeypatch.setattr(portable_packet, "observe_host_resources", lambda *_: next(observed))
    receipts = []
    for index, adapter in enumerate(("local", "server")):
        run = PortableTrainingRun(
            packet,
            _settings(packet, tmp_path / str(index), adapter, adapter),
            expected_packet_sha256=scientific,
            sources={adapter: lambda _, start: _batches(start)},
            **refs,
        )
        receipts.append(run.prepare())
        assert run.run_manifest_sha256 == manifest
        assert packet.identity_sha256() == scientific
    assert receipts[0].checkpoint_id == receipts[1].checkpoint_id


def test_cpu_fp16_is_denied_before_native_model_allocation(
    training_inputs,
    tmp_path,
    monkeypatch,
):
    packet, refs = training_inputs
    config = replace(packet.recipe.trainer, precision="fp16")
    packet = replace(packet, recipe=replace(packet.recipe, trainer=config))
    root = tmp_path / "fp16"
    run = PortableTrainingRun(
        packet,
        _settings(packet, root),
        expected_packet_sha256=packet.identity_sha256(),
        sources={"local": lambda _, index: _batches(index)},
        **refs,
    )
    monkeypatch.setattr(
        portable_packet,
        "observe_host_resources",
        lambda *_: HostResources("cpu", 2**30, 2**30, 2**30),
    )

    def unexpected(*_args):
        pytest.fail("unsupported FP16 reached model allocation")

    monkeypatch.setattr(portable_packet, "build_canonical_base", unexpected)
    with pytest.raises(ResourceAdmissionError, match="FP16 training requires a CUDA device"):
        run.prepare()
    assert not root.exists()


def test_changed_ambient_dtype_is_rejected_before_native_model_allocation(
    training_inputs,
    tmp_path,
    monkeypatch,
):
    packet, refs = training_inputs
    root = tmp_path / "float64"
    run = PortableTrainingRun(
        packet,
        _settings(packet, root),
        expected_packet_sha256=packet.identity_sha256(),
        sources={"local": lambda _, index: _batches(index)},
        **refs,
    )

    def unexpected(*_args):
        pytest.fail("unsupported default dtype reached model allocation")

    monkeypatch.setattr(portable_packet, "build_canonical_base", unexpected)
    prior_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        with pytest.raises(ResourceAdmissionError, match="requires default FP32 on CPU"):
            run.prepare()
    finally:
        torch.set_default_dtype(prior_dtype)
    assert not root.exists()


@pytest.mark.parametrize("pending_state", ["verified", "missing", "corrupt"])
def test_pending_recovery_checks_storage_after_authoritative_checkpoint_validation(
    training_inputs,
    tmp_path,
    monkeypatch,
    pending_state,
):
    packet, refs = training_inputs
    opened = []

    def source(_, index):
        opened.append(index)
        return _batches(index)

    run = PortableTrainingRun(
        packet,
        _settings(packet, tmp_path),
        expected_packet_sha256=packet.identity_sha256(),
        sources={"local": source},
        **refs,
    )
    prepared = run.prepare()
    save = RunController._save_checkpoint

    def fail_after_publish(path, trainer, identity):
        save(path, trainer, identity)
        raise OSError("interrupted after publication")

    with monkeypatch.context() as scoped:
        scoped.setattr(RunController, "_save_checkpoint", staticmethod(fail_after_publish))
        with pytest.raises(OSError, match="after publication"):
            run.run(expected_receipt_sha256=prepared.receipt_sha256)
    interrupted = run.status()
    assert (interrupted.committed_step, interrupted.pending_step) == (0, 1)
    checkpoints = tmp_path / run.run_manifest_sha256 / "checkpoints"
    pending = checkpoints / "00000001-00000001"
    if pending_state == "corrupt":
        (pending / "weights.safetensors").write_bytes(b"corrupt")
    elif pending_state == "missing":
        pending.rename(tmp_path / "unavailable")
    names = {path.name for path in checkpoints.iterdir()}
    model = packet.model_spec
    remaining_after_pending = 2 * (
        12 * model.parameter_count() + 4 * model.vocab_size * model.d_model
    )
    monkeypatch.setattr(
        portable_packet,
        "observe_host_resources",
        lambda *_: HostResources("cpu", 2**30, 2**30, remaining_after_pending),
    )

    if pending_state == "verified":
        final = run.run(expected_receipt_sha256=interrupted.receipt_sha256, recover=True)
        assert final.committed_step == 3
        assert opened == [0, 1]
        return

    def unexpected(*_args):
        pytest.fail("unverified pending checkpoint reached publication")

    monkeypatch.setattr(RunController, "_save_checkpoint", staticmethod(unexpected))
    error = CheckpointIntegrityError if pending_state == "corrupt" else ResourceAdmissionError
    with pytest.raises(error):
        run.run(expected_receipt_sha256=interrupted.receipt_sha256, recover=True)
    assert opened == [0]
    assert run.status() == interrupted
    assert {path.name for path in checkpoints.iterdir()} == names


def test_native_optimizer_boundary_reports_actual_cpu_and_checkpoint_measurements(
    training_inputs,
    tmp_path,
):
    packet, _ = training_inputs
    identity = packet.checkpoint_identity()

    def fresh():
        model = build_canonical_base(packet.model_spec, packet.init_spec, packet.initialization)
        return Trainer(model, packet.recipe.trainer)

    facts = observe_host_resources(tmp_path, "cpu")
    assert facts.checkpoint_volume_free_bytes > 0
    path = tmp_path / "checkpoint"
    invalid = fresh()
    with pytest.raises(ValueError, match="differs from the declared ModelSpec"):
        benchmark_optimizer_boundary(
            invalid,
            fresh,
            replace(identity, parameter_count=identity.parameter_count + 1),
            _batches(0),
            path,
            projection_tokens=4,
        )
    assert invalid.optimizer_step == 0
    assert not path.exists()
    measured = benchmark_optimizer_boundary(
        fresh(),
        fresh,
        identity,
        _batches(0),
        path,
        projection_tokens=packet.recipe.target_unique_loss_positions,
    )
    print(
        json.dumps(
            {
                "measurement": measured,
                "available_memory_bytes": facts.host_available_bytes,
                "checkpoint_volume_free_bytes": facts.checkpoint_volume_free_bytes,
            },
            sort_keys=True,
        )
    )
    assert measured["evidence_class"] == "MEASURED_SINGLE_OPTIMIZER_BOUNDARY"
    assert measured["run_manifest_sha256"] == identity.run_manifest_hash
    assert measured["tokens_seen"] == 1
    assert measured["tokens_per_second"] > 0
    assert measured["optimizer_step_ns"] > 0
    assert measured["checkpoint_write_ns"] > 0
    assert measured["checkpoint_restore_ns"] > 0
    assert measured["checkpoint_payload_bytes"] > 0
    assert measured["checkpoint_id"] == verify_checkpoint(path)["checkpoint_id"]
    weights = load_tensor_arrays(path / "weights.safetensors")
    serialized = sum(tensor.nbytes for tensor in weights.values())
    arithmetic = assess_resource_envelope(
        packet.model_spec,
        facts,
        remaining_updates=1,
        initial_checkpoint_needed=False,
    )
    assert packet.model_spec.tie_embeddings
    assert serialized == arithmetic["serialized_model_weight_bytes"]
    assert serialized > 4 * identity.parameter_count
    if os.name == "nt":
        assert measured["process_peak_working_set_bytes"] > 0
