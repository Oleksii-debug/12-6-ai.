"""Plan 7 Section 8: bounded topology, economics and fault-injection proof."""
import hashlib
from dataclasses import asdict, replace

import pytest

from twelve_six.billion_systems_gate import shard_manifest
from twelve_six.large_scale_paths import LargeAdapter, LargeCapacity, LargeLimits
from twelve_six.very_large_paths import (
    NodePlacement, VeryLargeDenied, assess_very_large, inspect_transport,
    very_large_recipe, very_large_spec,
)


def fixture(tier="30B", alternative="DENSE"):
    recipe = very_large_recipe(tier, alternative=alternative)
    cap = LargeCapacity(1, "a" * 64, 160, 20, 20_000_000_000,
                        600_000_000_000, 1000.0, 1_000_000_000.0,
                        10_000_000_000.0, 3.0, 1_000_000,
                        "b" * 64, "c" * 64)
    lim = LargeLimits(1, 256, 32, 4_000_000_000_000, 600_000_000_000,
                      2000.0, 100.0, 30.0, 100.0, 100_000_000.0)
    adapter = LargeAdapter(1, "FSDP", "d" * 64, recipe["model_sha256"],
                           "e" * 64, True, True, True, True, True)
    placement = NodePlacement(1, "a" * 64, tuple(f"node-{i}" for i in range(20)),
                              8, "f" * 64, "separate-failure-domains")
    parts = (b"fixture-shard-a", b"fixture-shard-b", b"fixture-shard-c")
    shards = tuple((f"part-{i:02d}", b, hashlib.sha256(b).hexdigest())
                   for i, b in enumerate(parts))
    return cap, lim, adapter, placement, shards, shard_manifest(parts)


def assess(tier="30B", alternative="DENSE", args=None, **kw):
    return assess_very_large(tier, alternative, *(args or fixture(tier, alternative)),
                             hourly_usd=0.0, budget_usd=0.0, **kw)


@pytest.mark.parametrize("tier", ["30B", "70B", "100B"])
def test_dense_profiles_are_concrete_modelspect_and_deterministic(tier):
    recipe = very_large_recipe(tier)
    assert recipe["dense_reference_parameters"] == very_large_spec(tier).parameter_count()
    assert abs(recipe["dense_reference_parameters"] - recipe["target_parameters"]) < (
        recipe["target_parameters"] // 10)
    assert recipe == very_large_recipe(tier)
    assert recipe["training_state_floor_bytes"] == 24 * recipe["dense_reference_parameters"]
    assert not recipe["paid_compute_authorized"] and not recipe["launch_authorized"]
    result = assess(tier)
    assert result["status"] == "NO_GO"
    assert result["reasons"] == ["physical_backend_unverified"]
    assert result == assess(tier)


def test_sparse_alternative_receives_no_unsafe_memory_discount():
    dense, sparse = very_large_recipe("70B"), very_large_recipe("70B", alternative="SPARSE_MOE")
    assert dense["training_state_floor_bytes"] == sparse["training_state_floor_bytes"]
    assert "moe_expert_contract_pending_section9" in assess("70B", "SPARSE_MOE")["reasons"]


def test_same_run_node_loss_preserves_artifact_identity_and_does_not_publish():
    *_, placement, shards, manifest = fixture()
    normal = inspect_transport(placement, shards, manifest)
    lost = inspect_transport(placement, shards, manifest, lost_nodes=("node-0",),
                             recovered_run_sha256="a" * 64)
    assert normal["manifest_sha256"] == lost["manifest_sha256"]
    assert normal["transport_sha256"] != lost["transport_sha256"]
    assert lost["run_sha256"] == normal["run_sha256"]
    assert lost["remote_transport_executed"] is False
    assert lost["canonical_checkpoint_published"] is False
    assert "physical_backend_unverified" in assess(lost_nodes=("node-0",))["reasons"]


def test_partial_corrupt_duplicate_reordered_and_lost_all_replicas_denied():
    *_, p, shards, m = fixture()
    corrupt = (shards[0], ("part-01", b"mutated", shards[1][2]), shards[2])
    for bad in (shards[:1], shards[::-1], shards + shards[-1:], corrupt):
        with pytest.raises(ValueError):
            inspect_transport(p, bad, m)
    with pytest.raises(VeryLargeDenied, match="incomplete checkpoint"):
        inspect_transport(p, shards, m, lost_nodes=("node-0", "node-1"))
    with pytest.raises(VeryLargeDenied, match="identity changed"):
        inspect_transport(p, shards, m, recovered_run_sha256="f" * 64)


@pytest.mark.parametrize("changes,reason", [
    ({"workers": 257}, "worker_admission_denied"),
    ({"free_bytes_per_worker": 1}, "memory_denied"),
    ({"free_checkpoint_bytes": 1}, "checkpoint_denied"),
    ({"tokens_per_second": 0}, "throughput_denied"),
    ({"interconnect_bytes_per_second": 0}, "interconnect_denied"),
    ({"checkpoint_bytes_per_second": 0}, "artifact_transport_denied"),
    ({"recovery_seconds": 40}, "recovery_slo_denied"),
    ({"paid_compute_requested": True}, "paid_compute_denied"),
])
def test_resource_admission_denials(changes, reason):
    c, l, a, p, s, m = fixture()
    c = replace(c, **changes)
    assert reason in assess(args=(c, l, a, p, s, m))["reasons"]


def test_topology_and_economic_fail_closed():
    c, l, a, p, s, m = fixture()
    report = assess(args=(c, l, a, replace(p, workers_per_node=7), s, m))
    assert "topology_mismatch" in report["reasons"]
    report = assess_very_large("30B", "DENSE", c, l, a, p, s, m,
                               hourly_usd=1000.0, budget_usd=1.0)
    assert "economic_budget_denied" in report["reasons"]
    assert report["launch_authorized"] is False


def test_unsupported_and_adversarial_input_types_fail_closed():
    c, l, a, p, s, m = fixture()
    for tier in ("3B", "300B", True):
        with pytest.raises(VeryLargeDenied):
            very_large_recipe(tier)
    for alt in ("MIXTURE", True):
        with pytest.raises(VeryLargeDenied):
            very_large_recipe("30B", alternative=alt)
    for changes in ({"node_ids": ("node-0", "node-0")},
                    {"workers_per_node": True}, {"run_sha256": "BAD"}):
        with pytest.raises(ValueError):
            NodePlacement(**{**asdict(p), **changes})
    for lost in (("node-0", "node-0"), ("unknown",), "node-0"):
        with pytest.raises(VeryLargeDenied):
            inspect_transport(p, s, m, lost_nodes=lost)
    object.__setattr__(p, "workers_per_node", True)
    with pytest.raises(ValueError):
        assess(args=(c, l, a, p, s, m))
    for cost in (float("nan"), float("inf"), True, -1):
        with pytest.raises(ValueError):
            assess_very_large("30B", "DENSE", c, l, a, replace(p, workers_per_node=8),
                              s, m, hourly_usd=cost, budget_usd=0)
