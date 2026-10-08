"""Plan 7 Section 9: 300B/1T dense, MoE, admission, failure and recovery."""
import hashlib
from dataclasses import replace

import pytest

from twelve_six.billion_systems_gate import shard_manifest
from twelve_six.extreme_moe_paths import (
    ExtremeDenied,
    MoEPolicy,
    assess_extreme,
    expert_placement,
    extreme_recipe,
    extreme_spec,
    route_tiny,
)
from twelve_six.large_scale_paths import LargeAdapter, LargeCapacity, LargeLimits
from twelve_six.very_large_paths import NodePlacement, inspect_transport


def policy(**changes):
    return replace(MoEPolicy(1, 8, 2, 2.0, 4, "b" * 64), **changes)


def inputs(tier="300B", p=None):
    recipe = extreme_recipe(tier, p)
    c = LargeCapacity(1, "a" * 64, 2048, 64, 100_000_000_000,
                      100_000_000_000_000, 1000.0, 10_000_000_000.0,
                      1_000_000_000_000.0, 5.0, 100_000,
                      "c" * 64, "d" * 64)
    l = LargeLimits(1, 4096, 128, 300_000_000_000_000, 200_000_000_000_000,
                    1000.0, 100.0, 20.0, 100.0, 1_000_000_000.0)
    a = LargeAdapter(1, "FSDP", "e" * 64, recipe["model_sha256"],
                     "f" * 64, True, True, True, True, True)
    n = NodePlacement(1, "a" * 64, tuple(f"node-{i}" for i in range(64)),
                      32, "9" * 64, "fixture-network")
    parts = (b"tiny-expert-state-a", b"tiny-expert-state-b",
             b"tiny-expert-state-c")
    shards = tuple((f"part-{i:02d}", s, hashlib.sha256(s).hexdigest())
                   for i, s in enumerate(parts))
    return c, l, a, n, shards, shard_manifest(parts)


def assess(tier="300B", p=None, args=None, tokens=None, **kw):
    if p is not None and tokens is None:
        tokens = tuple(range(128))
    return assess_extreme(tier, p, tokens, *(args or inputs(tier, p)),
                          evaluation_protocol_sha256="1" * 64,
                          serving_protocol_sha256="2" * 64,
                          cost_usd_per_hour=0.0, cost_budget_usd=0.0, **kw)


@pytest.mark.parametrize("tier", ("300B", "1T"))
def test_arithmetic_dense_parameter_identity_and_no_launch(tier):
    r = extreme_recipe(tier)
    assert r["dense_reference_parameters"] == extreme_spec(tier).parameter_count()
    assert abs(r["dense_reference_parameters"] - r["target_dense_parameters"]) < (
        r["target_dense_parameters"] // 10)
    assert r["active_fraction"] == 1
    assert r["training_state_floor_bytes"] == 24 * r["total_parameters"]
    assert r == extreme_recipe(tier)
    assert r["training_authorized"] is r["launch_authorized"] is False
    result = assess(tier)
    assert result["status"] == "NO_GO"
    assert result["reasons"] == ["physical_backend_unverified"]


@pytest.mark.parametrize("tier", ("300B", "1T"))
def test_sparse_experts_have_bounded_active_subset_and_full_memory_floor(tier):
    dense, sparse = extreme_recipe(tier), extreme_recipe(tier, policy())
    assert sparse["total_parameters"] > dense["total_parameters"]
    assert sparse["active_parameters"] < sparse["total_parameters"]
    assert sparse["active_fraction"] < 1
    assert 0.9 < sparse["dense_active_ratio"] < 1.1
    assert sparse["training_state_floor_bytes"] == 24 * sparse["total_parameters"]
    assert sparse["checkpoint_dual_copy_floor_bytes"] == 4 * sparse["total_parameters"]
    assert sparse["launch_authorized"] is False
    assert "physical_backend_unverified" in assess(tier, policy())["reasons"]


def test_tiny_router_deterministic_exact_assignments_without_silent_drops():
    r = route_tiny(policy(), tuple(range(128)))
    assert r == route_tiny(policy(), tuple(range(128)))
    assert r["status"] == "SYNTHETIC_ROUTING_PLAUSIBLE"
    assert r["assignments_count"] == 256
    assert r["dropped_assignments"] == 0
    assert r["expert_loads"] and len(r["expert_loads"]) == 8
    assert all(x <= r["capacity_per_expert"] for x in r["expert_loads"])


def test_repeated_tokens_overload_fail_closed_and_preserve_exposure():
    p = policy()
    r = route_tiny(p, (99,) * 128)
    assert r["status"] == "NO_GO" and r["overloaded_experts"]
    assert r["assignments_count"] == 256 and r["dropped_assignments"] == 0
    result = assess(p=p, tokens=(99,) * 128)
    assert result["status"] == "NO_GO"
    assert "expert_capacity_denied" in result["reasons"]


def test_synthetic_expert_failover_same_run_binds_checkpoint():
    p, args = policy(), inputs("300B", policy())
    _, _, _, n, shards, manifest = args
    prior = inspect_transport(n, shards, manifest)
    after = inspect_transport(n, shards, manifest, lost_nodes=("node-0",),
                              recovered_run_sha256=n.run_sha256)
    e = expert_placement(p, n, after, ("node-0",))
    assert prior["run_sha256"] == after["run_sha256"] == e["run_sha256"]
    assert after["transport_sha256"] != prior["transport_sha256"]
    assert e["expert_owners"][0]["survivor"] == "node-1"
    assert e["checkpoint_published"] is False and e["real_failover_executed"] is False
    assert "physical_backend_unverified" in assess(p=p, lost_nodes=("node-0",))["reasons"]


def test_lost_primary_and_replica_denies_checkpoint_and_expert():
    c, l, a, n, shards, manifest = inputs("300B", policy())
    with pytest.raises(ValueError):
        assess(p=policy(), args=(c, l, a, n, shards, manifest),
               lost_nodes=("node-0", "node-1"))


def test_shard_corruption_partial_reorder_duplicate_and_run_identity_denied():
    c, l, a, n, shards, manifest = inputs()
    bad = (shards[0], ("part-01", b"mutated", shards[1][2]), shards[2])
    for invalid in (shards[:1], shards[::-1], shards + shards[-1:], bad):
        with pytest.raises(ValueError):
            assess(args=(c, l, a, n, invalid, manifest))
    with pytest.raises(ValueError):
        assess(args=(replace(c, run_sha256="b" * 64), l, a, n, shards, manifest))


@pytest.mark.parametrize("field,value,reason", (
    ("workers", 5000, "topology_limit_denied"),
    ("free_bytes_per_worker", 1, "memory_denied"),
    ("free_checkpoint_bytes", 1, "checkpoint_denied"),
    ("tokens_per_second", 0, "throughput_denied"),
    ("interconnect_bytes_per_second", 0, "network_denied"),
    ("checkpoint_bytes_per_second", 0, "transport_denied"),
    ("recovery_seconds", 100, "recovery_denied"),
    ("paid_compute_requested", True, "paid_compute_denied"),
))
def test_insufficient_resources_or_unauthorized_paid_compute(field, value, reason):
    c, l, a, n, shards, manifest = inputs()
    report = assess(args=(replace(c, **{field: value}), l, a, n, shards, manifest))
    assert reason in report["reasons"]
    assert not report["launch_authorized"] and not report["paid_compute_authorized"]


def test_economic_budget_denial_and_deterministic_packet():
    c, l, a, n, shards, m = inputs()
    record = assess_extreme("300B", None, None, c, l, a, n, shards, m,
                            evaluation_protocol_sha256="1" * 64,
                            serving_protocol_sha256="2" * 64,
                            cost_usd_per_hour=100.0, cost_budget_usd=1.0)
    assert "economic_budget_denied" in record["reasons"]
    assert record["packet_sha256"] and not record["training_authorized"]
    assert assess() == assess()


@pytest.mark.parametrize("changes", (
    {"expert_count": True}, {"top_k": 8}, {"expert_parallel_groups": 3},
    {"capacity_factor": float("nan")}, {"capacity_factor": 0},
    {"capacity_factor": 100}, {"router_seed_sha256": "BAD"},
))
def test_moe_policy_malformed_denied(changes):
    with pytest.raises(ValueError):
        policy(**changes)


@pytest.mark.parametrize("tokens", (
    (), (True,), (-1,), ("7",), (2**63,), (None,),
))
def test_malformed_tokens_denied(tokens):
    with pytest.raises(ExtremeDenied):
        route_tiny(policy(), tokens)


def test_forged_dataclass_receipts_and_unsupported_family_denied():
    p = policy()
    object.__setattr__(p, "expert_parallel_groups", True)
    with pytest.raises(ValueError):
        route_tiny(p, (1, 2))
    for tier in ("30B", "2T", True):
        with pytest.raises(ExtremeDenied):
            extreme_recipe(tier)
    with pytest.raises(ExtremeDenied):
        extreme_recipe("300B", p)
    with pytest.raises(ExtremeDenied):
        assess(p=None, tokens=(1, 2))
    with pytest.raises(ExtremeDenied):
        assess(p=policy(), tokens=())


def test_tampered_transport_claim_and_node_loss_denied():
    p = policy()
    *_, node, shards, m = inputs("300B", p)
    receipt = inspect_transport(node, shards, m)
    for forged in ({**receipt, "canonical_checkpoint_published": True},
                   {**receipt, "transport_sha256": "X" * 64}):
        with pytest.raises(ValueError):
            expert_placement(p, node, forged, ())
    with pytest.raises(ExtremeDenied):
        expert_placement(p, node, receipt, ("not-a-node",))
