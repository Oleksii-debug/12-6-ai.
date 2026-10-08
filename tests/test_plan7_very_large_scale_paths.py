"""Plan 7 S8: multi-node loss, budgets, sparse denial and shard contracts."""
import hashlib
from dataclasses import asdict, replace

import pytest

from twelve_six.billion_systems_gate import shard_manifest
from twelve_six.large_scale_paths import LargeAdapter, LargeCapacity, LargeLimits, LargePathDenied
from twelve_six.very_large_scale_paths import (
    MultiNodeEvidence, assess_very_large, very_large_recipe, very_large_spec,
)


def fixtures(tier="30B", mode="DENSE"):
    r = very_large_recipe(tier, mode)
    cap = LargeCapacity(1, "a" * 64, 64, 8, 40_000_000_000,
                        500_000_000_000, 2000.0, 10_000_000_000.0,
                        100_000_000_000.0, 5.0, 1_000_000, "b" * 64, "c" * 64)
    limits = LargeLimits(1, 128, 16, 10_000_000_000_000,
                         1_000_000_000_000, 3600.0, 20.0, 60.0,
                         100.0, 100_000_000.0)
    adapter = LargeAdapter(1, "FSDP", "d" * 64, r["modelspec_sha256"],
                           "e" * 64, True, True, True, True, True)
    nodes = tuple(f"node-{i:02d}" for i in range(8))
    placement = tuple(n for n in nodes for _ in range(8))
    topology = MultiNodeEvidence(1, nodes, placement, "f" * 64,
                                 r["recipe_sha256"], "1" * 64,
                                 2.0, 500.0, True, True, True)
    pieces = (b"node-checkpoint-0", b"node-checkpoint-1")
    shards = tuple((f"part-{i:02d}", b, hashlib.sha256(b).hexdigest())
                   for i, b in enumerate(pieces))
    return cap, limits, adapter, topology, shards, shard_manifest(pieces)


@pytest.mark.parametrize("tier", ["30B", "70B", "100B"])
def test_arithmetic_dense_and_sparse_alternatives(tier):
    r = very_large_recipe(tier)
    assert r["dense_baseline_parameters"] == very_large_spec(tier).parameter_count()
    assert abs(r["dense_baseline_parameters"] - r["target_parameters"]) <= r["target_parameters"] // 10
    assert r["dense_train_state_floor_bytes"] == 24 * r["dense_baseline_parameters"]
    assert not r["launch_authorized"]
    sparse = very_large_recipe(tier, "SPARSE_CONTRACT_ONLY")
    assert sparse["sparse_alternative"] == "PLAN7_S9_VERSIONED_MOE_CONTRACT_REQUIRED"
    assert sparse["sparse_parameter_accounting_valid"] is False


@pytest.mark.parametrize("tier", ["30B", "70B", "100B"])
def test_physical_backend_denied_despite_good_proxy_inputs(tier):
    args = fixtures(tier)
    p = assess_very_large(tier, "DENSE", *args, recovered_run_sha256="a" * 64)
    assert p == assess_very_large(tier, "DENSE", *args)
    assert p["status"] == "NO_GO" and "physical_multinode_unverified" in p["reasons"]
    assert p["training_authorized"] is p["launch_authorized"] is False
    assert p["checkpoint_promoted"] is p["production_artifact_transported"] is False


def test_topology_node_loss_reduces_admission_and_preserves_run():
    args = fixtures("100B")
    base = assess_very_large("100B", "DENSE", *args)
    lost = assess_very_large("100B", "DENSE", *args, failed_node="node-00",
                             recovered_run_sha256="a" * 64)
    assert lost["run_sha256"] == base["run_sha256"]
    assert lost["active_workers"] == base["active_workers"] - 8
    assert "memory_denied" in lost["reasons"]
    with pytest.raises(LargePathDenied, match="scientific run identity"):
        assess_very_large("100B", "DENSE", *args, recovered_run_sha256="2" * 64)
    with pytest.raises(LargePathDenied, match="unknown lost node"):
        assess_very_large("100B", "DENSE", *args, failed_node="not-a-node")


@pytest.mark.parametrize("field,value,reason", [
    ("paid_compute_requested", True, "paid_compute_denied"),
    ("free_bytes_per_worker", 1, "memory_denied"),
    ("free_checkpoint_bytes", 1, "checkpoint_storage_denied"),
    ("checkpoint_bytes_per_second", 0, "checkpoint_transport_denied"),
    ("interconnect_bytes_per_second", 0, "interconnect_denied"),
    ("tokens_per_second", 1, "wallclock_denied"),
    ("recovery_seconds", 500, "recovery_slo_denied"),
])
def test_resource_economic_admission_negative(field, value, reason):
    c, l, a, t, shards, manifest = fixtures()
    p = assess_very_large("30B", "DENSE", replace(c, **{field: value}),
                          l, a, t, shards, manifest)
    assert reason in p["reasons"] and p["status"] == "NO_GO"


def test_explicit_economic_budget_and_sparse_fail_closed():
    c, l, a, t, shards, manifest = fixtures()
    p = assess_very_large("30B", "DENSE", c, l, a,
                          replace(t, max_estimated_dollars=0), shards, manifest)
    assert "economic_budget_denied" in p["reasons"]
    for mode in ("SPARSE_CONTRACT_ONLY",):
        cp, lp, ap, tp, sp, mp = fixtures("30B", mode)
        p = assess_very_large("30B", mode, cp, lp, ap, tp, sp, mp)
        assert "sparse_expert_backend_not_qualified" in p["reasons"]
        assert not p["launch_authorized"]


def test_topology_identity_checkpoint_transport_and_adversarial():
    c, l, a, t, shards, manifest = fixtures()
    for bad in (shards[:1], shards[::-1],
                (shards[0], ("part-01", b"tampered", shards[1][2]))):
        with pytest.raises(ValueError):
            assess_very_large("30B", "DENSE", c, l, a, t, bad, manifest)
    with pytest.raises(LargePathDenied, match="scientific recipe"):
        assess_very_large("30B", "DENSE", c, l, a,
                          replace(t, scientific_recipe_sha256="0" * 64),
                          shards, manifest)
    for change in ({"dollars_per_node_hour": float("nan")},
                   {"worker_nodes": t.worker_nodes[::-1]},
                   {"transport_roundtrip_tested": "true"}):
        with pytest.raises(ValueError):
            MultiNodeEvidence(**{**asdict(t), **change})
    object.__setattr__(t, "physical_multinode_executed", "true")
    with pytest.raises(ValueError):
        assess_very_large("30B", "DENSE", c, l, a, t, shards, manifest)


def test_unsupported_input_denied():
    with pytest.raises(LargePathDenied):
        very_large_recipe("1T")
    with pytest.raises(LargePathDenied):
        very_large_recipe("30B", "MOE_ACTUAL")
