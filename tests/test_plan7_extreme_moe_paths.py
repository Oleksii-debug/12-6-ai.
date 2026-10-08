"""Plan 7 S9: LOCAL_FREE deterministic sparse MoE contracts and failure probes."""
import hashlib
from dataclasses import replace

import pytest

from twelve_six.billion_systems_gate import _digest, shard_manifest
from twelve_six.extreme_moe_paths import (
    ExpertTopology, MoEDenied, assess_moe, moe_recipe, route_fixture,
)


def fixture(tier="300B"):
    recipe = moe_recipe(tier)
    n = recipe["num_experts"]
    topology = ExpertTopology(
        1, ("node-a", "node-b"),
        tuple(("node-a", "node-b") for _ in range(n)),
        2, "a" * 64, "b" * 64, recipe["recipe_sha256"],
        30_000_000_000_000, 30_000_000_000_000,
        5_000_000_000_000, 0.0, 0.0,
    )
    rows = []
    for i in range(2):
        row = [0.0] * n
        row[(2 * i) % n] = 5.0
        row[(2 * i + 1) % n] = 4.0
        rows.append(tuple(row))
    routing = route_fixture(tuple(rows), num_experts=n,
                            top_k=2, capacity_per_expert=1)
    pieces = (b"plan7-moe-shard-0", b"plan7-moe-shard-1")
    shards = tuple((f"part-{i:02d}", piece, hashlib.sha256(piece).hexdigest())
                   for i, piece in enumerate(pieces))
    return topology, routing, shards, shard_manifest(pieces)


@pytest.mark.parametrize("tier,expected,active", [
    ("300B", 308_000_000_000, 56_000_000_000),
    ("1T", 1_000_000_000_000, 100_000_000_000),
])
def test_extreme_arithmetic_is_versioned_and_non_authorizing(tier, expected, active):
    recipe = moe_recipe(tier)
    assert recipe["total_parameters"] == expected
    assert recipe["active_parameters_per_token"] == active < expected
    assert recipe["training_state_floor_bytes"] == 24 * expected
    assert recipe["checkpoint_dual_copy_floor_bytes"] == 4 * expected
    assert recipe["s8_sparse_contract"] == "PLAN7_S9_VERSIONED_MOE_CONTRACT_REQUIRED"
    assert recipe == moe_recipe(tier)
    assert not recipe["training_authorized"] and not recipe["paid_compute_authorized"]


@pytest.mark.parametrize("tier", ["300B", "1T"])
def test_proxy_admission_and_one_node_failover(tier):
    t, routing, shards, manifest = fixture(tier)
    base = assess_moe(tier, t, routing, shards, manifest)
    recovered = assess_moe(tier, t, routing, shards, manifest, lost_node="node-a",
                           resumed_run_sha256=t.run_sha256,
                           resumed_data_order_sha256=t.data_order_sha256)
    assert base == assess_moe(tier, t, routing, shards, manifest)
    assert base["status"] == recovered["status"] == "PROXY_DESIGN_VALID_NOT_AUTHORIZED"
    assert base["effective_expert_nodes"] == ["node-a"] * len(t.expert_replicas)
    assert recovered["effective_expert_nodes"] == ["node-b"] * len(t.expert_replicas)
    assert base["run_sha256"] == recovered["run_sha256"]
    assert not recovered["checkpoint_promoted"]
    assert not recovered["serving_authorized"]


def test_router_tie_order_expert_balancing_and_exact_conservation():
    x = route_fixture(((1.0, 1.0, 0.0, 0.0),
                       (0.0, 0.0, 1.0, 1.0)),
                      num_experts=4, top_k=2, capacity_per_expert=1)
    assert x["assignments"] == [(0, 1), (2, 3)]
    assert x["loads"] == [1, 1, 1, 1]
    assert x["routes"] == 4
    assert x["tokens_dropped"] == 0
    assert x == route_fixture(((1.0, 1.0, 0.0, 0.0),
                               (0.0, 0.0, 1.0, 1.0)),
                              num_experts=4, top_k=2, capacity_per_expert=1)


@pytest.mark.parametrize("scores", [
    ((float("nan"), 0.0),), ((float("inf"), 0.0),),
    ((True, 0.0),), ((0.0,),), ((0.0, 0.0), (0.0, 0.0)),
])
def test_route_fail_closed_no_silent_overflow(scores):
    with pytest.raises(MoEDenied):
        route_fixture(scores, num_experts=2, top_k=2, capacity_per_expert=1)


@pytest.mark.parametrize("kwargs,expected", [
    ({"paid_compute_requested": True}, "paid_compute_denied"),
    ({"free_state_bytes": 1}, "state_capacity_denied"),
    ({"free_checkpoint_bytes": 1}, "checkpoint_capacity_denied"),
    ({"resource_budget_bytes": 1}, "state_capacity_denied"),
    ({"estimated_cost_usd": 1.0}, "cost_limit_denied"),
])
def test_resource_admission_denial(kwargs, expected):
    topology, routing, shards, manifest = fixture()
    packet = assess_moe("300B", replace(topology, **kwargs), routing, shards, manifest)
    assert packet["status"] == "NO_GO"
    assert expected in packet["reasons"]
    assert not packet["launch_authorized"]


def test_identity_and_checkpoint_tamper_fails_closed():
    topology, routing, shards, manifest = fixture()
    for kwargs in ({"resumed_run_sha256": "c" * 64},
                   {"resumed_data_order_sha256": "d" * 64},
                   {"lost_node": "unknown"}):
        with pytest.raises(MoEDenied):
            assess_moe("300B", topology, routing, shards, manifest, **kwargs)
    for invalid in (shards[:1], shards[::-1],
                    (shards[0], ("part-01", b"evil", shards[1][2]))):
        with pytest.raises(ValueError):
            assess_moe("300B", topology, routing, invalid, manifest)


def test_topology_rejects_broken_expert_placement_and_stale_recipe():
    t, r, shards, manifest = fixture()
    with pytest.raises(MoEDenied):
        replace(t, expert_replicas=t.expert_replicas[:-1])
    with pytest.raises(MoEDenied):
        assess_moe("300B", replace(t, recipe_sha256="f" * 64),
                   r, shards, manifest)
    with pytest.raises(MoEDenied):
        replace(t, expert_replicas=(("node-a", "node-a"),) * len(t.expert_replicas))
    with pytest.raises(ValueError):
        replace(t, expert_parallel=True)
    with pytest.raises(ValueError):
        replace(t, estimated_cost_usd=float("nan"))


def test_receipt_forgery_and_frozen_dataclass_revalidation():
    t, r, shards, manifest = fixture()
    forged = dict(r)
    forged["loads"] = [0] * len(r["loads"])
    forged["routing_sha256"] = _digest({k: v for k, v in forged.items()
                                        if k != "routing_sha256"})
    with pytest.raises(MoEDenied):
        assess_moe("300B", t, forged, shards, manifest)
    forged = dict(r)
    forged["tokens_dropped"] = 1
    forged["routing_sha256"] = _digest({k: v for k, v in forged.items()
                                        if k != "routing_sha256"})
    with pytest.raises(MoEDenied):
        assess_moe("300B", t, forged, shards, manifest)
    object.__setattr__(t, "paid_compute_requested", "false")
    with pytest.raises(MoEDenied):
        assess_moe("300B", t, r, shards, manifest)


def test_missing_qualification_never_grants_actual_training():
    with pytest.raises(MoEDenied):
        moe_recipe("100B")
    t, route, shards, manifest = fixture()
    packet = assess_moe("300B", t, route, shards, manifest)
    for key in ("launch_authorized", "training_authorized",
                "paid_compute_authorized", "checkpoint_promoted", "serving_authorized"):
        assert packet[key] is False
    assert packet["evidence"] == "LOCAL_FREE_SIMULATED_EXPERTS_AND_CHECKPOINT_SHARDS"
