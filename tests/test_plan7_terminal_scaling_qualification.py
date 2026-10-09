"""Plan 7 S12 terminal LOCAL_FREE cross-section qualification, without training authority."""
from __future__ import annotations

import hashlib
import io
from dataclasses import replace

import pytest
import torch

from twelve_six.accelerated_scaling import (
    ProxyBudget,
    ProxyProtocol,
    grow_function_preserving,
)
from twelve_six.distributed_control import (
    DistributedDenied,
    ParallelMesh,
    ResourceAdmission,
    admit_distributed,
    begin_run,
    begin_step,
    commit_step,
    resume_after_loss,
)
from twelve_six.distributed_transport import (
    TransportDenied,
    prepare_transfer,
    publish_local,
    readback_local,
    receive,
    recover_transport,
    verify_receipts,
)
from twelve_six.model import ModelSpec, TwelveSixDecoder
from twelve_six.optional_risk_probes import digest
from twelve_six.product_scale_recipes import (
    CapacityEvidence,
    ScaleLimits,
    ScaleRecipeDenied,
    assess_scale,
    scale_recipe,
)


def _spec() -> ModelSpec:
    return ModelSpec(
        schema_version=1, vocab_size=32, max_seq_len=16,
        d_model=16, n_layers=2, n_heads=2, n_kv_heads=1,
        head_dim=8, d_ff=32, rope_rotary_dim=8,
    )


def _mesh(model_sha: str, workers: tuple[str, ...], epoch: int = 0) -> ParallelMesh:
    return ParallelMesh(
        1, "FSDP2", "a" * 64, model_sha, "b" * 64, "c" * 64,
        epoch, len(workers), 1, 1, 1, 1, workers,
    )


@pytest.fixture(scope="module")
def campaign():
    # A real tiny CPU model is grown and serialized; no optimizer/training run.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(71)
        parent = TwelveSixDecoder(_spec()).cpu().eval()
    target = replace(parent.spec, d_ff=48, n_layers=3)
    protocol = ProxyProtocol(1, "d" * 64, 99, 2, 6, 2)
    budget = ProxyBudget(100_000, 200_000_000, 128_000_000, 1024)
    child, growth = grow_function_preserving(parent, target, protocol, budget, seed=41)
    assert growth["function_preserved_on_fixture"] is True
    assert growth["training_authorized"] is False
    assert growth["paid_compute_authorized"] is False
    assert parent.spec.identity_sha256() != child.spec.identity_sha256()

    # Demonstrate single-rank -> eight-rank topology change with a NEW model run
    # identity; do not silently reuse the parent run for new architecture.
    original = _mesh(parent.spec.identity_sha256(), ("w-00",))
    assert begin_run(original).next_exposure == 0
    workers = tuple(f"w-{i:02d}" for i in range(8))
    mesh = _mesh(child.spec.identity_sha256(), workers)
    blob_io = io.BytesIO()
    torch.save(child.state_dict(), blob_io)
    blob = blob_io.getvalue()
    pieces = tuple(blob[(len(blob) * i) // 8:(len(blob) * (i + 1)) // 8]
                   for i in range(8))
    assert all(0 < len(p) <= 65_536 for p in pieces)
    shards = tuple((w, b, hashlib.sha256(b).hexdigest())
                   for w, b in zip(workers, pieces, strict=True))
    ledger = begin_run(mesh)
    ticket = begin_step(mesh, ledger, (0, 1))
    committed = commit_step(mesh, ledger, ticket, shards)
    plan = prepare_transfer(mesh, committed, shards, "plan7-tiny-growth",
                            storage_budget_bytes=4 * len(blob),
                            network_budget_bytes=2 * len(blob))
    receipts = tuple(receive(plan, w, b, 2.0 + i)
                     for i, (w, b, _) in enumerate(shards))
    return child, growth, mesh, ledger, committed, shards, plan, receipts, blob


def test_tiny_growth_to_multirank_checkpoint_restart_with_real_weights(campaign, tmp_path):
    child, growth, mesh, _, committed, shards, plan, receipts, blob = campaign
    metrics = verify_receipts(plan, receipts, recovery_ms=12.5)
    assert metrics["topology_workers"] == 8
    assert metrics["checkpoint_sha256"] == committed.checkpoint_sha256
    assert metrics["training_authorized"] is False
    assert metrics["paid_compute_authorized"] is False
    publish_local(tmp_path, plan, shards, receipts)
    result = readback_local(tmp_path, plan)
    assert result["status"] == "LOCAL_FIXTURE_PUBLISHED_VERIFIED"

    # Cross-process-equivalent reload of PHYSICAL fixture bytes, not a digest stub.
    restored_blob = b"".join((tmp_path / plan.artifact_id /
                               f"shard-{i:03}.bin").read_bytes() for i in range(8))
    assert restored_blob == blob
    restored_state = torch.load(io.BytesIO(restored_blob), map_location="cpu",
                                weights_only=True)
    recovered = TwelveSixDecoder(child.spec, child.init_spec).cpu().eval()
    recovered.load_state_dict(restored_state, strict=True)
    tokens = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    with torch.no_grad():
        torch.testing.assert_close(recovered(tokens).logits, child(tokens).logits,
                                   atol=0, rtol=0)
    assert growth["descendant_modelspec_sha256"] == mesh.run_sha256


def test_node_loss_same_run_restart_and_no_duplicate_exposure(campaign):
    _, _, mesh, _, committed, shards, _, _, _ = campaign
    new_mesh = replace(mesh, epoch=1)
    recovered = resume_after_loss(mesh, new_mesh, committed, shards)
    assert recovered.run_sha256 == committed.run_sha256
    assert recovered.next_step == committed.next_step
    assert recovered.next_exposure == 2
    assert recover_transport(mesh, new_mesh, committed, shards, 20)["next_exposure"] == 2
    with pytest.raises(DistributedDenied):
        begin_step(new_mesh, recovered, (0, 1))
    next_ticket = begin_step(new_mesh, recovered, (2, 3))
    assert next_ticket["step"] == 1


def test_missing_and_corrupt_checkpoint_never_commit(campaign, tmp_path):
    _, _, mesh, initial, committed, shards, plan, receipts, _ = campaign
    ticket = begin_step(mesh, initial, (0, 1))
    for invalid in (shards[:-1],
                    shards[:-1] + ((shards[-1][0], b"bad", shards[-1][2]),),
                    tuple(reversed(shards))):
        with pytest.raises(DistributedDenied):
            commit_step(mesh, initial, ticket, invalid)
    assert initial.next_exposure == 0
    with pytest.raises(TransportDenied):
        publish_local(tmp_path, plan, shards[:-1], receipts)
    assert not (tmp_path / plan.artifact_id).exists()
    with pytest.raises(TransportDenied):
        verify_receipts(plan, receipts[:-1])
    with pytest.raises(DistributedDenied):
        resume_after_loss(mesh, replace(mesh, epoch=1), committed, shards[:-1])


def test_checkpoint_transport_identity_and_metric_forgery_denied(campaign, tmp_path):
    _, _, mesh, _, committed, shards, plan, receipts, _ = campaign
    with pytest.raises(TransportDenied):
        verify_receipts(replace(plan, epoch=100), receipts)
    with pytest.raises(TransportDenied):
        verify_receipts(plan, (replace(receipts[0], elapsed_ms=True),) + receipts[1:])
    with pytest.raises(TransportDenied):
        prepare_transfer(mesh, committed, shards, "bad",
                         storage_budget_bytes=1, network_budget_bytes=1)
    publish_local(tmp_path, plan, shards, receipts)
    with pytest.raises(TransportDenied):
        publish_local(tmp_path, plan, shards, receipts)
    target = tmp_path / plan.artifact_id / "shard-000.bin"
    target.write_bytes(b"corrupt")
    with pytest.raises(TransportDenied):
        readback_local(tmp_path, plan)


def test_distributed_resource_admission_is_fail_closed(campaign):
    _, _, mesh, _, _, _, _, _, _ = campaign
    good = ResourceAdmission(1, 10_000_000, 1_000, 10_000_000, 1_000,
                             10_000_000.0, 1_000.0, 0.0, 1.0)
    assert admit_distributed(mesh, good)["status"] == "FIXTURE_ADMISSIBLE_NOT_AUTHORIZED"
    assert admit_distributed(mesh, good)["launch_authorized"] is False
    for bad in (replace(good, free_bytes_per_worker=1),
                replace(good, checkpoint_free_bytes=1),
                replace(good, paid_compute_requested=True),
                replace(good, interconnect_bytes_per_second=0.0)):
        report = admit_distributed(mesh, bad)
        assert report["status"] == "NO_GO"
        assert report["training_authorized"] is False
        assert report["paid_compute_authorized"] is False


def _proxy() -> dict:
    inner = {
        "tier": "35M", "full_scale_executed": False,
        "training_authorized": False, "paid_compute_authorized": False,
        "proxy_receipt": {
            "promotion_authorized": False, "training_executed": False,
            "paid_compute_authorized": False,
        },
    }
    out = {
        "schema_version": 1, "tier": "200M",
        "recipe_sha256": scale_recipe("200M")["recipe_sha256"],
        "s4_proxy_receipt_sha256": digest(inner), "proxy": inner,
        "full_scale_executed": False, "launch_authorized": False,
        "training_authorized": False, "paid_compute_authorized": False,
    }
    out["receipt_sha256"] = digest(out)
    return out


def test_product_scale_recipe_requires_real_capacity_and_training_proof():
    proxy = _proxy()
    evidence = CapacityEvidence(1, proxy["receipt_sha256"], 1, 1, 1,
                                0.0, 0.0, 0.0, 1, False, False, 0.0, False)
    limits = ScaleLimits(1, 1, 1, 60.0, 1.0)
    decision = assess_scale("200M", evidence, limits, proxy)
    assert decision["status"] == "NO_GO"
    assert "terminal_20m_proof_missing" in decision["reasons"]
    assert "memory_admission_denied" in decision["reasons"]
    assert decision["launch_authorized"] is False
    assert decision["paid_compute_authorized"] is False
    paid = assess_scale("200M", replace(evidence, paid_compute_requested=True),
                        limits, proxy)
    assert "paid_compute_not_authorized" in paid["reasons"]
    with pytest.raises(ScaleRecipeDenied):
        assess_scale("200M", evidence, limits, {**proxy, "launch_authorized": True})


def test_large_recipe_does_not_silently_promote_bridge_or_external_compute():
    proxy = _proxy()
    proxy["tier"] = "300M"  # A receipt for 200M can never be repurposed for 300M.
    evidence = CapacityEvidence(1, _proxy()["receipt_sha256"], 1, 1, 1,
                                0.0, 0.0, 0.0, 1, False, False, 0.0, False)
    with pytest.raises(ScaleRecipeDenied):
        assess_scale("300M", evidence, ScaleLimits(1, 1, 1, 60, 1), proxy)
