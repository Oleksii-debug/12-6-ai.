"""Plan 7 Section 7: 3B/7B/13B resource, shard and recovery gates."""
import hashlib
from dataclasses import replace

import pytest

from twelve_six.billion_systems_gate import shard_manifest
from twelve_six.large_scale_paths import (
    LargeAdapter, LargeCapacity, LargeLimits, LargePathDenied, assess_large,
    large_recipe, large_spec,
)


def inputs(tier="3B"):
    r = large_recipe(tier)
    cap = LargeCapacity(1, "a" * 64, 16, 4, 20_000_000_000,
                        100_000_000_000, 1000.0, 1_000_000_000.0,
                        10_000_000_000.0, 5.0, 1_000_000,
                        "b" * 64, "c" * 64)
    limits = LargeLimits(1, 32, 8, 1_000_000_000_000, 200_000_000_000,
                         2000.0, 300.0, 30.0, 100.0, 100_000_000.0)
    adapter = LargeAdapter(1, "FSDP", "d" * 64, r["model_sha256"],
                           "e" * 64, True, True, True, True, True)
    data = (b"tiny-shard-zero", b"tiny-shard-one")
    shards = tuple((f"part-{i:02d}", v, hashlib.sha256(v).hexdigest())
                   for i, v in enumerate(data))
    return cap, limits, adapter, shards, shard_manifest(data)


@pytest.mark.parametrize("tier", ["3B", "7B", "13B"])
def test_recipe_arithmetic_only(tier):
    r = large_recipe(tier)
    assert r["parameters"] == large_spec(tier).parameter_count()
    assert abs(r["parameters"] - r["target_parameters"]) <= r["target_parameters"] // 10
    assert r["train_state_floor_bytes"] == 24 * r["parameters"]
    assert r["launch_authorized"] is r["training_authorized"] is False
    assert r == large_recipe(tier)


@pytest.mark.parametrize("tier,tp", [("3B", 1), ("7B", 2), ("13B", 4)])
def test_fixture_same_run_and_deterministic_receipt(tier, tp):
    args = inputs(tier)
    a = assess_large(tier, *args, serving_tensor_parallel=tp,
                     recovered_run_sha256="a" * 64)
    assert a == assess_large(tier, *args, serving_tensor_parallel=tp)
    assert a["status"] == "NO_GO" and a["reasons"] == ["physical_backend_unverified"]
    assert not a["canonical_checkpoint_published"]


@pytest.mark.parametrize("field,value,reason", [
    ("workers", 33, "topology_limit_denied"),
    ("free_bytes_per_worker", 1, "memory_denied"),
    ("free_checkpoint_bytes", 1, "checkpoint_storage_denied"),
    ("tokens_per_second", 1, "throughput_denied"),
    ("interconnect_bytes_per_second", 0, "interconnect_denied"),
    ("checkpoint_bytes_per_second", 0, "checkpoint_transport_denied"),
    ("recovery_seconds", 31, "recovery_denied"),
    ("paid_compute_requested", True, "paid_compute_denied"),
])
def test_fail_closed_resource(field, value, reason):
    c, l, a, shards, manifest = inputs()
    report = assess_large("3B", replace(c, **{field: value}), l, a, shards, manifest)
    assert reason in report["reasons"] and report["status"] == "NO_GO"


def test_worker_loss_same_identity_new_capacity_receipt():
    c, l, a, shards, manifest = inputs()
    old = assess_large("3B", c, l, a, shards, manifest)
    restarted = assess_large("3B", replace(c, workers=15), l, a, shards,
                             manifest, recovered_run_sha256="a" * 64)
    assert old["run_sha256"] == restarted["run_sha256"]
    assert old["capacity_sha256"] != restarted["capacity_sha256"]
    with pytest.raises(LargePathDenied, match="scientific run identity"):
        assess_large("3B", c, l, a, shards, manifest,
                     recovered_run_sha256="f" * 64)


def test_shards_reject_partial_corrupt_reordered_duplicate():
    c, l, a, shards, manifest = inputs()
    corrupt = (shards[0], (shards[1][0], b"bad", shards[1][2]))
    for bad in (shards[:1], shards[::-1], shards + shards[-1:], corrupt):
        with pytest.raises(ValueError):
            assess_large("3B", c, l, a, bad, manifest)


def test_adversarial_types_and_adapter_tamper():
    c, l, a, shards, manifest = inputs()
    for changes in ({"workers": True}, {"tokens_per_second": float("nan")},
                    {"paid_compute_requested": "false"}):
        with pytest.raises(ValueError):
            LargeCapacity(**{**c.__dict__, **changes})
    object.__setattr__(a, "same_run_resume_tested", "true")
    with pytest.raises(ValueError):
        assess_large("3B", c, l, a, shards, manifest)
    with pytest.raises(LargePathDenied):
        large_recipe("30B")
