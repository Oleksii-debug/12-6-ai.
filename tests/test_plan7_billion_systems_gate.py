"""Plan 7 S6: bounded 1B backend/admission/shard/recovery fixture tests."""
import hashlib
from dataclasses import replace

import pytest

from twelve_six.billion_systems_gate import (
    ADAPTERS, AdapterEvidence, AdmissionLimits, BillionGateDenied,
    CapacityEvidence, assess_1b, billion_recipe, billion_spec,
    inspect_shards, shard_manifest,
)


def adapter(name="FSDP", **changes):
    return replace(AdapterEvidence(
        1, name, "v0.0-fixture", "a" * 64,
        billion_recipe()["modelspec_sha256"], "b" * 64,
        True, True, True, True, True, False,
    ), **changes)


def capacity(**changes):
    return replace(CapacityEvidence(
        1, "c" * 64, 4, 2, 12_000_000_000, 10_000_000_000,
        1000.0, 500_000_000.0, 500_000_000.0, 15.0,
        1_000_000, "d" * 64, False,
    ), **changes)


def limits(**changes):
    return replace(AdmissionLimits(
        1, 8, 4, 100_000_000_000, 10_000_000_000,
        5000.0, 120.0, 100.0, 100.0, 100_000_000.0,
    ), **changes)


def physical_shards():
    data = (b"fixture-shard-A", b"fixture-shard-B")
    claim = shard_manifest(data)
    shards = tuple((f"part-{i:02d}", part,
                    hashlib.sha256(part).hexdigest())
                   for i, part in enumerate(data))
    return shards, claim


def receipt():
    shards, claim = physical_shards()
    return inspect_shards(shards, claim)


@pytest.mark.parametrize("name", sorted(ADAPTERS))
def test_adapters_are_explicit_contracts_not_real_backend_launch(name):
    result = assess_1b(capacity(), limits(), adapter(name), receipt())
    assert result["status"] == "NO_GO"
    assert result["reasons"] == ["real_backend_unverified"]
    assert result["launch_authorized"] is False
    assert result["training_authorized"] is False
    assert result == assess_1b(capacity(), limits(), adapter(name), receipt())


def test_parameter_accounting_and_parent_contract():
    recipe = billion_recipe()
    assert recipe["parameters"] == billion_spec().parameter_count()
    assert 950_000_000 <= recipe["parameters"] <= 1_050_000_000
    assert recipe["train_state_floor_bytes"] == 24 * recipe["parameters"]
    assert recipe["source_200m_recipe_sha256"] != recipe["recipe_sha256"]
    assert recipe["launch_authorized"] is False


@pytest.mark.parametrize("changes,reason", [
    ({"paid_compute_requested": True}, "paid_compute_denied"),
    ({"workers": 10}, "topology_limit_denied"),
    ({"free_bytes_per_worker": 1}, "memory_budget_denied"),
    ({"free_checkpoint_bytes": 1}, "checkpoint_space_denied"),
    ({"measured_interconnect_bytes_per_second": 0}, "interconnect_unqualified"),
    ({"measured_tokens_per_second": 1}, "throughput_unqualified"),
    ({"measured_checkpoint_bytes_per_second": 0}, "checkpoint_transport_denied"),
    ({"measured_recovery_seconds": 1000}, "recovery_slo_denied"),
])
def test_capacity_denials(changes, reason):
    result = assess_1b(capacity(**changes), limits(), adapter(), receipt())
    assert reason in result["reasons"]
    assert result["status"] == "NO_GO"


def test_worker_loss_and_restart_same_run_identity():
    baseline = assess_1b(capacity(), limits(), adapter(), receipt(),
                         recovered_run_sha256="c" * 64)
    assert baseline["run_sha256"] == "c" * 64
    with pytest.raises(BillionGateDenied, match="scientific run identity"):
        assess_1b(capacity(), limits(), adapter(), receipt(),
                  recovered_run_sha256="e" * 64)
    recovered = assess_1b(capacity(workers=3), limits(), adapter(), receipt(),
                          recovered_run_sha256="c" * 64)
    assert recovered["run_sha256"] == baseline["run_sha256"]
    assert recovered["capacity_sha256"] != baseline["capacity_sha256"]


def test_shards_physically_verified_and_rejected_on_corruption():
    shards, manifest = physical_shards()
    assert inspect_shards(shards, manifest) == inspect_shards(shards, manifest)
    with pytest.raises(BillionGateDenied, match="corrupt shard"):
        inspect_shards((shards[0], ("part-01", b"mutated", shards[1][2])), manifest)
    for invalid in (shards[:1], shards[::-1], shards + shards[-1:]):
        with pytest.raises(BillionGateDenied):
            inspect_shards(invalid, manifest)
    with pytest.raises(BillionGateDenied):
        inspect_shards(shards, "f" * 64)
    assert inspect_shards(shards, manifest)["launch_authorized"] is False


@pytest.mark.parametrize("field,value", [
    ("workers", True), ("nodes", 0), ("paid_compute_requested", "false"),
    ("measured_tokens_per_second", float("nan")),
    ("measured_recovery_seconds", float("inf")),
])
def test_invalid_input_fails_closed(field, value):
    with pytest.raises(BillionGateDenied):
        capacity(**{field: value})


def test_forged_frozen_dataclass_and_adapter_capability_denied():
    a = adapter()
    object.__setattr__(a, "optimizer_semantics_tested", "yes")
    with pytest.raises(BillionGateDenied):
        assess_1b(capacity(), limits(), a, receipt())
    assert "adapter_contract_unqualified" in assess_1b(
        capacity(), limits(), adapter(worker_loss_tested=False), receipt()
    )["reasons"]
    with pytest.raises(BillionGateDenied):
        adapter(name="UnknownBackend")
    assert "modelspec_mismatch" in assess_1b(
        capacity(), limits(), adapter(model_sha256="f" * 64), receipt()
    )["reasons"]


def test_shard_receipt_unknown_and_paid_markers_no_promotion():
    for bad in ({}, {**receipt(), "partial": True},
                {**receipt(), "launch_authorized": True}):
        result = assess_1b(capacity(), limits(), adapter(), bad)
        assert "checkpoint_receipt_unverified" in result["reasons"]
    result = assess_1b(capacity(), limits(),
                       adapter(actual_backend_executed=True), receipt())
    assert result["status"] == "PROXY_SYSTEMS_PLAUSIBLE_NOT_AUTHORIZED"
    assert result["real_backend_promotion_authorized"] is False
    assert result["paid_compute_authorized"] is False
