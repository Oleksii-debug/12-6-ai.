from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tools import plan9_external_cyber_pilot_v1 as pilot
from twelve_six.training.resource_envelope import DISK_FLOOR_DENIAL, HostResources
from twelve_six.training.run_control import RunPhase, RunReceipt


def test_pilot_cursor_recovers_exact_position_without_opening_training():
    class Stream:
        @staticmethod
        def cursor(position):
            return {
                "next_microbatch_index": position, "optimizer_step": position // 12,
                "consumed_loss_positions": 1303 if position == 24 else 1340,
                "cursor_sha256": "cursor",
            }

        @staticmethod
        def verify_cursor(cursor, digest):
            if cursor != Stream.cursor(cursor["next_microbatch_index"]) or digest != "cursor":
                raise ValueError("bad cursor")

        @staticmethod
        def iter_batches(cursor, digest):
            Stream.verify_cursor(cursor, digest)
            assert cursor["next_microbatch_index"] == 24
            yield {"input_ids": [[2, 3]], "target_ids": [[3, 4]], "loss_mask": [[1, 1]]}

    hooks = pilot.PilotCursor(Stream())
    with pytest.raises(ValueError, match="bad cursor"):
        hooks.restore({**Stream.cursor(24), "consumed_loss_positions": 10})
    hooks.restore(Stream.cursor(24))
    with pytest.raises(ValueError, match="receipt and physical cursor disagree"):
        next(hooks.batches(SimpleNamespace(committed_step=1, tokens_seen=1303)))
    batch = next(hooks.batches(SimpleNamespace(committed_step=2, tokens_seen=1303)))
    assert batch["target_ids"].tolist() == [[3, 4]]
    assert hooks.read(SimpleNamespace(micro_step=25, tokens_seen=1340)) == Stream.cursor(25)


def test_pilot_rejects_unsupported_host_before_training(monkeypatch, tmp_path):
    monkeypatch.setattr(pilot.sys, "platform", "win32")
    monkeypatch.setattr(pilot, "Trainer", lambda *_a, **_kw: pytest.fail("trainer was constructed"))
    args = SimpleNamespace(scratch=tmp_path, action="prepare")
    with pytest.raises(ValueError, match="Linux on an external CUDA host"):
        pilot.execute(Path.cwd(), {"run_manifest_sha256": "a" * 64}, None, args)


def test_pilot_storage_denial_is_deferred_only_for_pending_recovery(monkeypatch, tmp_path):
    monkeypatch.setattr(pilot, "observe_host_resources", lambda *_: HostResources("cuda:0", 1, 1, 0))
    monkeypatch.setattr(pilot, "assess_resource_envelope", lambda *_a, **_kw: {
        "denials": [DISK_FLOOR_DENIAL], "unpruned_checkpoint_volume_floor_bytes": 10,
    })
    monkeypatch.setattr(pilot.native, "restore_memory_floors", lambda *_: (1, 1))
    cfg = SimpleNamespace(model=object())
    with pytest.raises(RuntimeError, match=DISK_FLOOR_DENIAL):
        pilot._admit(cfg, tmp_path, remaining=2, prepare=False)
    pilot._admit(cfg, tmp_path, remaining=2, prepare=False, defer_disk=True)


def test_resume_admission_uses_restore_memory_floor_even_during_pending_recovery(monkeypatch, tmp_path):
    monkeypatch.setattr(pilot, "observe_host_resources", lambda *_: HostResources("cuda:0", 1, 1, 0))
    monkeypatch.setattr(pilot, "assess_resource_envelope", lambda *_a, **_kw: {
        "denials": [DISK_FLOOR_DENIAL], "unpruned_checkpoint_volume_floor_bytes": 10,
    })
    monkeypatch.setattr(pilot.native, "restore_memory_floors", lambda *_: (2, 2))
    cfg = SimpleNamespace(model=object())
    with pytest.raises(RuntimeError, match="PILOT_HOST_MEMORY_BELOW_RESTORE_ENVELOPE.*PILOT_DEVICE"):
        pilot._admit(cfg, tmp_path, remaining=2, prepare=False, defer_disk=True)


def test_benchmark_receipt_binds_runtime_profile_and_checkpoint_restore(monkeypatch, tmp_path):
    commit, tree, model, profile, checkpoint = (char * 64 for char in "abcde")
    lock = {"runtime_release_version": 2}
    candidate = {"git_sha": commit, "modelspec_sha256": model, "initialization": {}, "parameter_count": 42}
    packet = {"profile_sha256": profile, "run": {"candidate": candidate}}
    context = {"boot_id": "this-boot", "device": {"uuid": "gpu"}, "checkpoint_filesystem": {"st_dev": 1, "f_fsid": 2}}
    preflight = {
        "schema": pilot.native.SCHEMA, "source_git_sha": commit, "source_tree_sha": tree,
        "profile": pilot.PROFILE, "profile_config_sha256_lf": profile, "model_spec_sha256": model,
        "initialization": {}, "capacity": {
            "parameter_count": 42, "disposition": "CAPACITY_SCREEN_PASSED", "benchmark_capacity_denials": [],
        },
    }
    measured = {
        "schema": pilot.native.MEASUREMENT_SCHEMA,
        "scope": "SYNTHETIC_SINGLE_OPTIMIZER_BOUNDARY_AND_FRESH_PROCESS_RESTORE",
        "profile": pilot.PROFILE, "device": "cuda:0", "source_git_sha": commit,
        "environment": {**lock, "cublas_workspace_config": ":4096:8"},
        "campaign_authorized": False, "optimizer_steps_executed": 1,
        "capacity_preflight": {**preflight, "receipt_sha256": pilot.hash_json(preflight)},
        "measurement": {"execution_context": context, "boundary": {
            "evidence_class": "MEASURED_SINGLE_OPTIMIZER_BOUNDARY",
            "model_spec_sha256": model, "parameter_count": 42,
            "tokens_seen": 31, "checkpoint_id": checkpoint,
        }},
        "fresh_process_restore": {
            "checkpoint_id": checkpoint, "tokens_seen": 31, "committed_step": 1, "execution_context": context,
        },
    }
    monkeypatch.setattr(pilot.native, "_git", lambda *_: tree.encode())
    receipt = tmp_path / "benchmark.json"

    def save(value):
        raw = pilot.canonical({**value, "receipt_sha256": pilot.hash_json(value)})
        receipt.write_bytes(raw)
        return pilot.sha(raw)

    pin = save(measured)
    assert pilot._verify_benchmark(tmp_path, packet, receipt, pin, lock) == context
    with pytest.raises(ValueError, match="bytes differ"):
        pilot._verify_benchmark(tmp_path, packet, receipt, "f" * 64, lock)
    with pytest.raises(ValueError, match="runtime, model or measured restore"):
        pilot._verify_benchmark(tmp_path, packet, receipt, pin, {"runtime_release_version": 1})
    measured["fresh_process_restore"]["execution_context"] = {**context, "device": {"uuid": "other-gpu"}}
    with pytest.raises(ValueError, match="runtime, model or measured restore"):
        pilot._verify_benchmark(tmp_path, packet, receipt, save(measured), lock)


def test_execute_rejects_changed_benchmark_gpu_or_volume_before_model(monkeypatch, tmp_path):
    monkeypatch.setattr(pilot.sys, "platform", "linux")
    monkeypatch.setattr(pilot.native, "_verify_measurement_inputs", lambda *_: {"profile_id": "linux-x86_64"})
    monkeypatch.setattr(pilot.native, "_cuda_environment", lambda *_: None)
    monkeypatch.setattr(pilot, "Trainer", lambda *_a, **_kw: pytest.fail("trainer was constructed"))
    context = {"boot_id": "same", "device": {"uuid": "gpu"}, "checkpoint_filesystem": {"st_dev": 1, "f_fsid": 2}}
    monkeypatch.setattr(pilot, "_verify_benchmark", lambda *_: context)
    monkeypatch.setattr(pilot.native, "cuda_checkpoint_context", lambda *_: {
        **context, "checkpoint_filesystem": {"st_dev": 1, "f_fsid": 3},
    })
    args = SimpleNamespace(
        scratch=tmp_path, action="prepare", execute=True, expected_packet_sha256="f" * 64,
        install_report=tmp_path / "install.json", install_report_sha256="e" * 64,
        benchmark_receipt=tmp_path / "benchmark.json", benchmark_receipt_sha256="d" * 64,
    )
    packet = {
        "run_manifest_sha256": "a" * 64, "packet_sha256": "f" * 64,
        "run": {"candidate": {"git_sha": "b" * 40},
                "environment": {"lock_sha256": "c" * 64, "profile_id": "linux-x86_64"}},
    }
    with pytest.raises(ValueError, match="GPU or checkpoint filesystem differs"):
        pilot.execute(tmp_path, packet, None, args)


def test_readback_rejects_wrong_cursor_even_with_matching_checkpoint_id(monkeypatch, tmp_path):
    run = "a" * 64
    receipt = RunReceipt(
        run_sha256=run, identity_sha256="b" * 64, generation=6, control_epoch=1, phase=RunPhase.PAUSED,
        control_enabled=True, control_pending=False, committed_step=2, tokens_seen=1303, pending_step=None,
        pending_tokens=None, checkpoint_epoch=1, checkpoint_id="c" * 64, parent_sha256=None,
        receipt_sha256="d" * 64,
    )
    monkeypatch.setattr(pilot.RunController, "status", lambda _: receipt)
    cursor = {"consumed_loss_positions": 1303}
    verify = Mock(return_value={"checkpoint_id": receipt.checkpoint_id, "identity": {
        "run_manifest_hash": run, "step": 2, "tokens_seen": 1303,
    }})
    monkeypatch.setattr(pilot, "verify_checkpoint", verify)
    args = SimpleNamespace(scratch=tmp_path, action="verify-checkpoint", expected_receipt_sha256=receipt.receipt_sha256)
    packet = {"run_manifest_sha256": run, "checkpoints": [None, None, cursor]}
    monkeypatch.setattr(pilot, "checkpoint_transaction", lambda _: SimpleNamespace(exposure_cursor=cursor))
    assert pilot.execute(Path.cwd(), packet, None, args)["checkpoint_verified"]
    verify.assert_called_once_with(tmp_path / run / "checkpoints" / "00000001-00000002")
    monkeypatch.setattr(pilot, "checkpoint_transaction", lambda _: SimpleNamespace(exposure_cursor={}))
    with pytest.raises(ValueError, match="cursor and durable receipt do not agree"):
        pilot.execute(Path.cwd(), packet, None, args)
