from __future__ import annotations

import json
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from test_ordered_live_exposure_authority import _authority, _live

from twelve_six.checkpoint import CheckpointIntegrityError, hash_json, verify_checkpoint
from twelve_six.checkpoint.transaction import checkpoint_transaction
from twelve_six.data.unique_loss_ledger_v2 import LedgerError
from twelve_six.model import BaseInitManifest, InitSpec, ModelSpec
from twelve_six.training.config import TrainerConfig, TrainingRecipe
from twelve_six.training.portable_packet import (
    PortableTrainingPacket,
    PortableTrainingRun,
    TrainingDataReferences,
    TrainingExecutionSettings,
)
from twelve_six.training.run_control import RunControlError, RunPhase


@pytest.fixture
def training_inputs():
    materialization, ledger, plan, content = _authority()
    model = ModelSpec(
        schema_version=1,
        vocab_size=32,
        max_seq_len=4,
        d_model=8,
        n_layers=1,
        n_heads=1,
        n_kv_heads=1,
        head_dim=8,
        d_ff=16,
        rope_rotary_dim=8,
    )
    init = InitSpec()
    packet = PortableTrainingPacket(
        run_id="portable-fixture",
        stage="fixture",
        source_git_sha="a" * 40,
        environment_lock_sha256="b" * 64,
        model_spec=model,
        init_spec=init,
        initialization=BaseInitManifest(1, model.identity_sha256(), init.identity_sha256(), 713),
        recipe=TrainingRecipe(
            trainer=TrainerConfig(seed=81, max_steps=3, scheduler="cosine"),
            microbatch_size=1,
            sequence_length=2,
            target_unique_loss_positions=4,
            maximum_total_exposures=4,
            max_exposures_per_unique_position=1,
        ),
        data=TrainingDataReferences(
            tokenizer_version="fixture-v1",
            tokenizer_config_sha256=content["tokenizer_identity_sha256"],
            tokenizer_vocab_sha256=hash_json(list(range(model.vocab_size))),
            tokenizer_vocab_size=model.vocab_size,
            dataset_manifest_sha256=hash_json(materialization),
            split_identity="fixture:train",
            packing_version="fixture-v1",
            packing_sha256=ledger["packing_identity_sha256"],
            ledger_sha256=ledger["ledger_identity_sha256"],
            exposure_plan_sha256=plan["plan_identity_sha256"],
            shard_contract_sha256=content["manifest_identity_sha256"],
        ),
    )
    return packet, {"ledger": ledger, "exposure_plan": plan, "shard_contract": content}


def _batches(index):
    for position in range(index, 3):
        inputs, targets, mask = _live(position)
        yield {
            "input_ids": torch.tensor(inputs),
            "target_ids": torch.tensor(targets),
            "loss_mask": torch.tensor(mask),
        }


def _settings(packet, root, adapter="local", provider="workstation"):
    return TrainingExecutionSettings(adapter, provider, "cpu", root, packet.environment_lock_sha256)


def _checkpoint(run, root, receipt):
    path = root / run.run_manifest_sha256 / "checkpoints"
    return verify_checkpoint(path / f"{receipt.checkpoint_epoch:08d}-{receipt.committed_step:08d}")


def test_scientific_packet_and_detached_settings_roundtrip(training_inputs, tmp_path: Path):
    packet, _ = training_inputs
    raw = json.dumps(packet.to_dict()).encode()
    assert PortableTrainingPacket.from_json(raw) == packet
    assert packet.identity_sha256() == hash_json(packet.to_dict())
    assert packet.to_dict()["commands"] == ["prepare", "run", "status"]
    assert (
        packet.recipe.run_training_fields()
        == packet.checkpoint_identity().training_config["training"]
    )
    settings = _settings(packet, tmp_path, "server", "remote-fixture")
    assert (
        TrainingExecutionSettings.from_dict(json.loads(json.dumps(settings.to_dict()))) == settings
    )
    assert all(key not in packet.to_dict() for key in ("device", "provider", "adapter", "run_root"))
    changed = replace(
        packet, recipe=replace(packet.recipe, trainer=replace(packet.recipe.trainer, seed=7))
    )
    assert changed.identity_sha256() != packet.identity_sha256()
    assert (
        changed.checkpoint_identity().run_manifest_hash
        != packet.checkpoint_identity().run_manifest_hash
    )


def test_packet_rejects_mixed_authority_ambiguous_json_and_policy_changes(
    training_inputs, tmp_path: Path
):
    packet, _ = training_inputs
    body = packet.to_dict()
    with pytest.raises(ValueError, match="fields must match"):
        PortableTrainingPacket.from_dict({**body, "device": "cpu"})
    with pytest.raises(TypeError):
        PortableTrainingPacket.from_dict(
            {**body, "model_spec": {**body["model_spec"], "host": "other"}}
        )
    with pytest.raises(ValueError, match="policy, commands or evidence"):
        PortableTrainingPacket.from_dict({**body, "commands": ["arbitrary command"]})
    with pytest.raises(ValueError, match="initialization differs"):
        PortableTrainingPacket.from_dict(
            {
                **body,
                "initialization": {
                    **body["initialization"],
                    "model_spec_sha256": "0" * 64,
                },
            }
        )
    with pytest.raises(CheckpointIntegrityError, match="invalid portable packet JSON"):
        PortableTrainingPacket.from_json(b'{"schema":"extra",' + json.dumps(body).encode()[1:])
    with pytest.raises(ValueError, match="execution fields must match"):
        TrainingExecutionSettings.from_dict({**_settings(packet, tmp_path).to_dict(), "recipe": {}})


def test_portable_open_rejects_unverified_inputs_before_run_root_exists(
    training_inputs, tmp_path: Path
):
    packet, refs = training_inputs
    root = tmp_path / "run"
    settings = _settings(packet, root)
    common = {**refs, "sources": {"local": lambda _, index: _batches(index)}}
    with pytest.raises(ValueError, match="expected SHA-256"):
        PortableTrainingRun(packet, settings, expected_packet_sha256="0" * 64, **common)
    expected = packet.identity_sha256()
    with pytest.raises(ValueError, match="runtime lock"):
        PortableTrainingRun(
            packet,
            replace(settings, environment_lock_sha256="c" * 64),
            expected_packet_sha256=expected,
            **common,
        )
    with pytest.raises(ValueError, match="no batch source"):
        PortableTrainingRun(
            packet, replace(settings, adapter="server"), expected_packet_sha256=expected, **common
        )
    with pytest.raises(ValueError, match="packing contract"):
        PortableTrainingRun(
            packet,
            settings,
            expected_packet_sha256=expected,
            **{
                **common,
                "ledger": {**refs["ledger"], "packing_identity_sha256": "0" * 64},
            },
        )
    tampered_plan = {**refs["exposure_plan"], "num_workers": 4}
    with pytest.raises(LedgerError):
        PortableTrainingRun(
            packet,
            settings,
            expected_packet_sha256=expected,
            **{
                **common,
                "exposure_plan": tampered_plan,
            },
        )
    wrong_shards = PortableTrainingRun(
        packet,
        settings,
        expected_packet_sha256=expected,
        **{
            **common,
            "shard_contract": {**refs["shard_contract"], "manifest_identity_sha256": "0" * 64},
        },
    )
    with pytest.raises(LedgerError):
        wrong_shards.prepare()
    assert not root.exists()


def test_same_packet_resumes_local_free_and_server_without_scientific_drift(
    training_inputs, tmp_path: Path
):
    packet, refs = training_inputs
    packet_sha = packet.identity_sha256()
    payload = json.dumps(packet.to_dict()).encode()
    visited = []
    free_disconnected = False

    def source(adapter):
        def open_batches(settings, index):
            nonlocal free_disconnected
            assert settings.adapter == adapter
            visited.append((settings.provider, index))
            if adapter == "free" and not free_disconnected:
                free_disconnected = True
                raise OSError("fixture worker disconnected")
            return _batches(index)

        return open_batches

    sources = {adapter: source(adapter) for adapter in ("local", "free", "server")}

    def open_run(root, adapter, provider):
        settings = _settings(packet, root, adapter, provider)
        return PortableTrainingRun(
            PortableTrainingPacket.from_json(payload),
            TrainingExecutionSettings.from_dict(settings.to_dict()),
            expected_packet_sha256=packet_sha,
            sources=sources,
            **refs,
        )

    baseline_root = tmp_path / "baseline"
    baseline = open_run(baseline_root, "local", "baseline")
    full = baseline.run(expected_receipt_sha256=baseline.prepare().receipt_sha256)
    assert full.phase == RunPhase.COMPLETED

    root = tmp_path / "migrated"
    local = open_run(root, "local", "workstation")
    initial = local.prepare()
    paused = local.run(
        expected_receipt_sha256=initial.receipt_sha256, control=lambda _: RunPhase.PAUSED
    )
    free = open_run(root, "free", "free-fixture")
    with pytest.raises(OSError, match="disconnected"):
        free.run(expected_receipt_sha256=paused.receipt_sha256)
    failed = free.status()
    assert failed.phase == RunPhase.FAILED
    assert failed.committed_step == paused.committed_step == 1
    assert failed.checkpoint_id == paused.checkpoint_id
    free_paused = free.run(
        expected_receipt_sha256=failed.receipt_sha256,
        recover=True,
        control=lambda _: RunPhase.PAUSED,
    )
    assert (free_paused.phase, free_paused.committed_step) == (RunPhase.PAUSED, 2)

    server = open_run(root, "server", "server-fixture")
    with pytest.raises(RunControlError, match="stale run receipt"):
        server.run(expected_receipt_sha256=paused.receipt_sha256, recover=True)
    resumed = server.run(expected_receipt_sha256=free_paused.receipt_sha256)
    assert (resumed.phase, resumed.committed_step, resumed.tokens_seen) == (
        RunPhase.COMPLETED,
        3,
        4,
    )
    assert baseline.run_manifest_sha256 == local.run_manifest_sha256 == free.run_manifest_sha256
    assert free.run_manifest_sha256 == server.run_manifest_sha256
    assert full.checkpoint_id == resumed.checkpoint_id
    full_manifest = _checkpoint(baseline, baseline_root, full)
    resumed_manifest = _checkpoint(server, root, resumed)
    assert full_manifest["files"] == resumed_manifest["files"]
    assert checkpoint_transaction(full_manifest) == checkpoint_transaction(resumed_manifest)
    assert visited == [
        ("baseline", 0),
        ("workstation", 0),
        ("free-fixture", 1),
        ("free-fixture", 1),
        ("server-fixture", 2),
    ]


def test_adapter_cannot_override_frozen_batch_dimensions(training_inputs, tmp_path: Path):
    packet, refs = training_inputs

    def oversize_source(_, index):
        batch = next(_batches(index))
        yield {key: value.repeat(2, 1) for key, value in batch.items()}

    run = PortableTrainingRun(
        packet,
        _settings(packet, tmp_path),
        expected_packet_sha256=packet.identity_sha256(),
        sources={"local": oversize_source},
        **refs,
    )
    initial = run.prepare()
    with pytest.raises(ValueError, match="batch dimensions"):
        run.run(expected_receipt_sha256=initial.receipt_sha256)
    assert run.status().committed_step == 0


def test_in_process_adapter_conflict_preserves_writer_rng(training_inputs, tmp_path: Path):
    packet, refs = training_inputs
    packet_sha = packet.identity_sha256()
    peer = PortableTrainingRun(
        packet,
        _settings(packet, tmp_path / "peer"),
        expected_packet_sha256=packet_sha,
        sources={"local": lambda _, index: _batches(index)},
        **refs,
    )

    def contested_source(_, index):
        random.random()
        torch.rand(1)
        python_rng, torch_rng = random.getstate(), torch.random.get_rng_state()
        with (
            ThreadPoolExecutor(max_workers=1) as pool,
            pytest.raises(RunControlError, match="operation is active in this process"),
        ):
            pool.submit(peer.prepare).result(timeout=5)
        assert random.getstate() == python_rng
        assert torch.equal(torch.random.get_rng_state(), torch_rng)
        return _batches(index)

    run = PortableTrainingRun(
        packet,
        _settings(packet, tmp_path / "owner"),
        expected_packet_sha256=packet_sha,
        sources={"local": contested_source},
        **refs,
    )
    initial = run.prepare()
    assert run.run(expected_receipt_sha256=initial.receipt_sha256).phase == RunPhase.COMPLETED
    assert not (tmp_path / "peer").exists()
    assert peer.prepare().phase == RunPhase.PREPARED
