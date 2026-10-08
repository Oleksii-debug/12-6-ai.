"""Plan7 S10: LOCAL_FREE DP/TP/PP/CP/EP, atomic shards and restart tests."""
import hashlib
from dataclasses import replace

import pytest

from twelve_six.distributed_control import (
    DistributedDenied,
    ParallelMesh,
    ResourceAdmission,
    adapter_packet,
    admit_distributed,
    begin_run,
    begin_step,
    bind_expert_groups,
    commit_step,
    resume_after_loss,
)
from twelve_six.extreme_moe_paths import MoEPolicy


def mesh(*, epoch=0, dp=2, tp=2, ep=2, prefix="w"):
    n = dp * tp
    return ParallelMesh(1, "FSDP2", "a" * 64, "b" * 64, "c" * 64,
                        "d" * 64, epoch, dp, tp, 1, 1, ep,
                        tuple(f"{prefix}-{i:02d}" for i in range(n)))


def shards(mesh_obj, suffix=b"step"):
    return tuple((worker, suffix + worker.encode(),
                  hashlib.sha256(suffix + worker.encode()).hexdigest())
                 for worker in mesh_obj.workers)


def budget(**updates):
    record = {
        "schema_version": 1,
        "free_bytes_per_worker": 4096,
        "required_bytes_per_worker": 2048,
        "checkpoint_free_bytes": 2048,
        "checkpoint_required_bytes": 1024,
        "interconnect_bytes_per_second": 400.0,
        "minimum_interconnect_bytes_per_second": 100.0,
        "estimated_dollars": 0.0,
        "max_dollars": 0.0,
        "paid_compute_requested": False,
    }
    record.update(updates)
    return ResourceAdmission(**record)


def test_mesh_rank_geometry_and_adapters_versioned():
    m = mesh()
    p = adapter_packet(m)
    assert p == adapter_packet(m)
    assert p["world_size"] == 4
    assert [(x["data"], x["tensor"], x["expert_group"]) for x in p["rank_map"]] == [
        (0, 0, 0), (0, 1, 0), (1, 0, 1), (1, 1, 1),
    ]
    assert not p["training_authorized"]
    expanded = ParallelMesh(1, "Megatron", "a" * 64, "b" * 64, "c" * 64,
                            "d" * 64, 0, 2, 2, 2, 2, 2,
                            tuple(f"gpu-{i:02d}" for i in range(16)))
    ranks = adapter_packet(expanded)["rank_map"]
    assert len({tuple(row[k] for k in ("data", "tensor", "pipeline", "context"))
                for row in ranks}) == 16


@pytest.mark.parametrize("kwargs", [
    {"expert_parallel": 3},
    {"data_parallel": True},
    {"tensor_parallel": 0},
    {"workers": ("w-00", "w-01")},
    {"workers": ("w-03", "w-02", "w-01", "w-00")},
    {"workers": ("w-00", "w-00", "w-02", "w-03")},
    {"backend": "UNKNOWN"},
    {"epoch": -1},
    {"adapter_version_sha256": "x" * 64},
])
def test_mesh_bad_topology_denied(kwargs):
    with pytest.raises(ValueError):
        replace(mesh(), **kwargs)


def test_expert_contract_cross_binds_canonical_plan7_s9():
    m = mesh()
    policy = MoEPolicy(1, 4, 2, 1.5, 2, "e" * 64)
    packet = bind_expert_groups(m, policy)
    assert packet["mode"] == "EXPERT_GROUP_FIXTURE_ONLY"
    assert not packet["launch_authorized"]
    with pytest.raises(DistributedDenied):
        bind_expert_groups(m, replace(policy, expert_parallel_groups=1))


@pytest.mark.parametrize("kwargs,reason", [
    ({"paid_compute_requested": True}, "paid_compute_denied"),
    ({"free_bytes_per_worker": 1}, "worker_memory_denied"),
    ({"checkpoint_free_bytes": 1}, "checkpoint_storage_denied"),
    ({"interconnect_bytes_per_second": 1.0}, "interconnect_denied"),
    ({"estimated_dollars": 1.0}, "budget_denied"),
])
def test_admission_fail_closed(kwargs, reason):
    p = admit_distributed(mesh(), budget(**kwargs))
    assert p["status"] == "NO_GO"
    assert reason in p["reasons"]
    assert not p["launch_authorized"]


def test_admission_proxy_pass_still_never_launches():
    p = admit_distributed(mesh(), budget())
    assert p["status"] == "FIXTURE_ADMISSIBLE_NOT_AUTHORIZED"
    assert not p["training_authorized"]
    assert not p["paid_compute_authorized"]


def test_ordered_two_step_commit_and_replay_denial():
    m = mesh()
    ledger0 = begin_run(m)
    ticket0 = begin_step(m, ledger0, (0, 1))
    ledger1 = commit_step(m, ledger0, ticket0, shards(m))
    assert (ledger1.next_step, ledger1.next_exposure) == (1, 2)
    assert ledger1.checkpoint_sha256 and ledger1.ledger_sha256 != ledger0.ledger_sha256
    with pytest.raises(DistributedDenied):
        commit_step(m, ledger1, ticket0, shards(m))
    with pytest.raises(DistributedDenied):
        begin_step(m, ledger1, (1, 2))
    with pytest.raises(DistributedDenied):
        begin_step(m, ledger1, (3,))
    ticket1 = begin_step(m, ledger1, (2, 3))
    ledger2 = commit_step(m, ledger1, ticket1, shards(m, b"step2"))
    assert (ledger2.next_step, ledger2.next_exposure) == (2, 4)
    assert not hasattr(ledger2, "optimizer_updates")


def test_worker_loss_aborts_before_atomic_commit():
    m = mesh()
    ledger = begin_run(m)
    ticket = begin_step(m, ledger, (0, 1))
    all_shards = shards(m)
    for invalid in (
        all_shards[:-1],
        all_shards[::-1],
        (all_shards[0],) * 4,
        (all_shards[0], (all_shards[1][0], b"evil", all_shards[1][2]),
         all_shards[2], all_shards[3]),
    ):
        with pytest.raises(ValueError):
            commit_step(m, ledger, ticket, invalid)
    assert ledger.next_step == 0 and ledger.checkpoint_sha256 is None
    valid = commit_step(m, ledger, ticket, all_shards)
    assert valid.next_step == 1


def test_elastic_failure_resume_preserves_science_and_checkpoint():
    m = mesh()
    first = commit_step(m, begin_run(m), begin_step(m, begin_run(m), (0,)), shards(m))
    recovered_mesh = mesh(epoch=1, dp=4, tp=2, ep=2, prefix="new")
    recovered = resume_after_loss(m, recovered_mesh, first, shards(m))
    assert recovered.next_step == first.next_step == 1
    assert recovered.next_exposure == first.next_exposure == 1
    assert recovered.checkpoint_sha256 == first.checkpoint_sha256
    second = commit_step(recovered_mesh, recovered,
                         begin_step(recovered_mesh, recovered, (1, 2)),
                         shards(recovered_mesh))
    assert second.next_exposure == 3 and second.next_step == 2


@pytest.mark.parametrize("field,value", [
    ("run_sha256", "1" * 64),
    ("recipe_sha256", "2" * 64),
    ("data_order_sha256", "3" * 64),
    ("adapter_version_sha256", "4" * 64),
    ("epoch", 2),
])
def test_recovery_rejects_identity_or_epoch_substitution(field, value):
    old = mesh()
    ledger0 = begin_run(old)
    committed = commit_step(old, ledger0, begin_step(old, ledger0, (0,)),
                            shards(old))
    new = replace(old, epoch=1)
    with pytest.raises(DistributedDenied):
        resume_after_loss(old, replace(new, **{field: value}), committed, shards(old))


def test_resume_rejects_corrupt_or_uncommitted_checkpoint():
    old = mesh()
    with pytest.raises(DistributedDenied):
        resume_after_loss(old, replace(old, epoch=1), begin_run(old), shards(old))
    ledger0 = begin_run(old)
    valid = commit_step(old, ledger0, begin_step(old, ledger0, (0,)), shards(old))
    with pytest.raises(ValueError):
        resume_after_loss(old, replace(old, epoch=1), valid,
                          shards(old, b"altered"))


def test_ticket_tamper_and_ledger_forgery_denied():
    m = mesh()
    old = begin_run(m)
    ticket = begin_step(m, old, (0,))
    tampered = dict(ticket)
    tampered["exposures"] = (99,)
    with pytest.raises(DistributedDenied):
        commit_step(m, old, tampered, shards(m))
    forged = replace(old, next_step=5)
    with pytest.raises(DistributedDenied):
        begin_step(m, forged, (0,))
    with pytest.raises(DistributedDenied):
        commit_step(m, old, replace(old), shards(m))


def test_chunked_checkpoint_receipts_support_over_16_workers():
    large = ParallelMesh(1, "TorchTitan", "a" * 64, "b" * 64, "c" * 64,
                         "d" * 64, 0, 4, 2, 2, 2, 2,
                         tuple(f"rank-{i:03d}" for i in range(32)))
    ledger = begin_run(large)
    committed = commit_step(large, ledger, begin_step(large, ledger, (0,)),
                            shards(large))
    assert committed.next_step == 1
    assert committed.checkpoint_sha256 is not None
