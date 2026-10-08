"""Plan7 S11 bounded local fixture: transport, evidence, atomicity, recovery."""
import hashlib
import json
from dataclasses import replace

import pytest

from twelve_six.distributed_control import (
    DistributedDenied, ParallelMesh, begin_run, begin_step,
    commit_step,
)
from twelve_six.distributed_transport import (
    TransportDenied, prepare_transfer, receive, verify_receipts,
    publish_local, readback_local, recover_transport,
)


def fixture():
    m = ParallelMesh(1, "FSDP2", "a" * 64, "b" * 64, "c" * 64,
                     "d" * 64, 0, 2, 1, 1, 1, 1, ("w-00", "w-01"))
    shards = tuple((w, ("value-" + w).encode(),
                    hashlib.sha256(("value-" + w).encode()).hexdigest())
                   for w in m.workers)
    ledger = begin_run(m)
    ticket = begin_step(m, ledger, (0, 1))
    ledger = commit_step(m, ledger, ticket, shards)
    plan = prepare_transfer(m, ledger, shards, "tiny-checkpoint",
                            storage_budget_bytes=1000, network_budget_bytes=1000)
    receipts = tuple(receive(plan, o, data, (idx + 1) * 2)
                     for idx, (o, data, _) in enumerate(shards))
    return m, ledger, shards, plan, receipts


def test_local_publish_exact_readback_and_restart(tmp_path):
    _, _, shards, plan, receipts = fixture()
    summary = verify_receipts(plan, receipts, recovery_ms=10)
    assert summary["status"] == "VERIFIED_NOT_PUBLISHED"
    assert summary["topology_workers"] == 2
    assert summary["checkpoint_duration_ms"] == 4
    assert summary["straggler"] == "w-01"
    assert summary["recovery_time_ms"] == 10
    assert summary["bandwidth_bytes_per_second"] > 0
    assert summary["storage_peak_bytes_estimate"] == 2 * sum(len(s[1]) for s in shards)
    path = publish_local(tmp_path, plan, shards, receipts)
    assert path.name == plan.artifact_id
    assert readback_local(tmp_path, plan)["status"] == "LOCAL_FIXTURE_PUBLISHED_VERIFIED"
    with pytest.raises(TransportDenied):
        publish_local(tmp_path, plan, shards, receipts)


def test_partial_truncated_corrupt_duplicate_or_reordered_denied(tmp_path):
    _, _, shards, plan, receipts = fixture()
    for bad in (receipts[:1], receipts[::-1], receipts[:1] * 2):
        with pytest.raises(TransportDenied):
            verify_receipts(plan, bad)
    with pytest.raises(TransportDenied):
        publish_local(tmp_path, plan, shards[:1], receipts)
    assert not (tmp_path / plan.artifact_id).exists()
    with pytest.raises(TransportDenied):
        receive(plan, "w-00", b"wrong", 2.0)
    with pytest.raises(TransportDenied):
        receive(plan, "w-00", shards[0][1][:-1], 2.0)
    with pytest.raises(TransportDenied):
        receive(plan, "foreign", shards[0][1], 2.0)
    with pytest.raises(TransportDenied):
        publish_local(tmp_path, plan, (shards[1], shards[0]), receipts)
    assert not (tmp_path / plan.artifact_id).exists()


def test_invalid_source_topology_resource_or_uncommitted_denied():
    m, ledger, shards, _, _ = fixture()
    with pytest.raises(DistributedDenied):
        prepare_transfer(m, replace(ledger, checkpoint_sha256=None), shards, "good",
                         storage_budget_bytes=1000, network_budget_bytes=1000)
    with pytest.raises(TransportDenied):
        prepare_transfer(m, ledger, shards, "../escape",
                         storage_budget_bytes=1000, network_budget_bytes=1000)
    with pytest.raises(TransportDenied):
        prepare_transfer(m, ledger, shards, "okay",
                         storage_budget_bytes=1, network_budget_bytes=1000)
    with pytest.raises(TransportDenied):
        prepare_transfer(m, ledger, shards, "okay",
                         storage_budget_bytes=1000, network_budget_bytes=1)
    with pytest.raises(Exception):
        prepare_transfer(replace(m, epoch=2), ledger, shards, "okay",
                         storage_budget_bytes=1000, network_budget_bytes=1000)


def test_forged_plan_metrics_receipts_and_disk_drift(tmp_path):
    _, _, shards, plan, receipts = fixture()
    with pytest.raises(TransportDenied):
        verify_receipts(replace(plan, epoch=10), receipts)
    with pytest.raises(TransportDenied):
        verify_receipts(plan, (replace(receipts[0], elapsed_ms=5.0), receipts[1]))
    with pytest.raises(TransportDenied):
        verify_receipts(plan, receipts, recovery_ms=float("nan"))
    for bad in (0, float("inf"), True):
        with pytest.raises(TransportDenied):
            receive(plan, "w-00", shards[0][1], bad)
    folder = publish_local(tmp_path, plan, shards, receipts)
    (folder / "shard-000.bin").write_bytes(b"tampered")
    with pytest.raises(TransportDenied):
        readback_local(tmp_path, plan)


def test_extra_file_manifest_forgery_and_symlink_denial(tmp_path):
    _, _, shards, plan, receipts = fixture()
    folder = publish_local(tmp_path, plan, shards, receipts)
    (folder / "extra").write_bytes(b"e")
    with pytest.raises(TransportDenied):
        readback_local(tmp_path, plan)
    (folder / "extra").unlink()
    (folder / "manifest.json").write_text(json.dumps({"wrong": True}))
    with pytest.raises(TransportDenied):
        readback_local(tmp_path, plan)
    with pytest.raises(TransportDenied):
        publish_local(tmp_path / "missing", plan, shards, receipts)
    symlink = tmp_path / "alias"
    symlink.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(TransportDenied):
        readback_local(symlink, plan)


def test_same_run_recovery_no_reexposure():
    mesh, ledger, shards, _, _ = fixture()
    recovered = recover_transport(mesh, replace(mesh, epoch=1), ledger, shards, 20.0)
    assert recovered["run_sha256"] == mesh.run_sha256
    assert recovered["next_exposure"] == ledger.next_exposure
    assert recovered["new_epoch"] == 1
    assert recovered["recovery_time_ms"] == 20.0
    with pytest.raises(Exception):
        recover_transport(mesh, replace(mesh, epoch=2), ledger, shards, 20)
    with pytest.raises(TransportDenied):
        recover_transport(mesh, replace(mesh, epoch=1), ledger, shards, -1)
