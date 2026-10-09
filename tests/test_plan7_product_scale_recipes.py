"""Plan 7 S5: deterministic bounded ~200M/bridge admission and failure paths."""
from dataclasses import replace

import pytest

from twelve_six.product_scale_recipes import (
    CapacityEvidence,
    ScaleLimits,
    ScaleRecipeDenied,
    assess_scale,
    migration_packet,
    qualify_proxy,
    scale_recipe,
    scale_spec,
)


def fake_proxy(tier):
    # For decision-unit tests only; integration tests use the actual S4 proxy.
    from twelve_six.optional_risk_probes import digest
    tiers = {"200M": "35M", "300M": "50M", "500M": "100M"}
    inner = {"tier": tiers[tier], "full_scale_executed": False,
             "training_authorized": False, "paid_compute_authorized": False,
             "proxy_receipt": {"promotion_authorized": False,
                               "training_executed": False,
                               "paid_compute_authorized": False}}
    record = {"schema_version": 1, "tier": tier,
              "recipe_sha256": scale_recipe(tier)["recipe_sha256"],
              "s4_proxy_receipt_sha256": digest(inner), "proxy": inner,
              "full_scale_executed": False, "launch_authorized": False,
              "training_authorized": False, "paid_compute_authorized": False}
    record["receipt_sha256"] = digest(record)
    return record


def evidence(tier="200M", **changes):
    return replace(CapacityEvidence(
        1, fake_proxy(tier)["receipt_sha256"],
        20_000_000_000, 10_000_000_000, 1, 1_000.0,
        100_000_000.0, 0.0, 1_000_000,
        True, tier != "200M", 0.3,
    ), **changes)


def limits(**changes):
    return replace(ScaleLimits(30_000_000_000, 10_000_000_000, 4, 5_000.0, 100.0),
                   **changes)


@pytest.mark.parametrize("tier", ("200M", "300M", "500M"))
def test_reproducible_modelspec_and_planning_floors(tier):
    recipe = scale_recipe(tier)
    model = scale_spec(tier)
    assert recipe == scale_recipe(tier)
    assert recipe["modelspec_sha256"] == model.identity_sha256()
    assert recipe["parameters"] == model.parameter_count()
    assert abs(recipe["parameters"] - recipe["target_parameters"]) <= (
        recipe["target_parameters"] // 20)
    assert recipe["train_state_planning_floor_bytes"] == 24 * model.parameter_count()
    assert recipe["launch_authorized"] is False
    assert recipe["paid_compute_authorized"] is False
    assert recipe["mandatory_bridge"] is False


@pytest.mark.parametrize("tier", ("200M", "300M", "500M"))
def test_proxy_evidence_cannot_grant_launch(tier):
    receipt = assess_scale(tier, evidence(tier), limits(), fake_proxy(tier))
    assert receipt["status"] == "PROXY_CAPACITY_PLAUSIBLE_NOT_AUTHORIZED"
    assert receipt == assess_scale(tier, evidence(tier), limits(), fake_proxy(tier))
    assert receipt["launch_authorized"] is False
    assert receipt["training_authorized"] is False
    assert receipt["paid_compute_authorized"] is False


@pytest.mark.parametrize("change,reason", [
    ({"terminal_20m_proven": False}, "terminal_20m_proof_missing"),
    ({"paid_compute_requested": True}, "paid_compute_not_authorized"),
    ({"available_memory_bytes_per_worker": 1}, "memory_admission_denied"),
    ({"available_checkpoint_bytes": 1}, "checkpoint_storage_denied"),
    ({"healthy_workers": 5}, "worker_budget_exceeded"),
    ({"measured_tokens_per_second": 0.0}, "throughput_unqualified"),
    ({"measured_checkpoint_bytes_per_second": 0.0}, "checkpoint_transport_unqualified"),
    ({"measured_tokens_per_second": 100.0, "unique_training_tokens": 1_000_000_000},
     "wallclock_budget_denied"),
    ({"healthy_workers": 2}, "distributed_link_unmeasured"),
])
def test_negative_admission(change, reason):
    packet = assess_scale("200M", evidence(**change), limits(), fake_proxy("200M"))
    assert packet["status"] == "NO_GO"
    assert reason in packet["reasons"]


def test_bridge_not_mandatory_and_rejected_without_need():
    packet = assess_scale("300M", evidence("300M", measured_bridge_risk_reduction=0),
                          limits(), fake_proxy("300M"))
    assert packet["status"] == "NO_GO"
    assert "bridge_has_no_measured_need" in packet["reasons"]
    packet = assess_scale("500M", evidence("500M", terminal_200m_proven=False),
                          limits(), fake_proxy("500M"))
    assert "bridge_requires_200m_proof" in packet["reasons"]


@pytest.mark.parametrize("bad", [True, 1.0, "1", 2])
def test_bad_schema_denied(bad):
    with pytest.raises(ScaleRecipeDenied):
        evidence(schema_version=bad)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0, True])
def test_bad_throughput_denied(bad):
    with pytest.raises(ScaleRecipeDenied):
        evidence(measured_tokens_per_second=bad)


def test_forged_frozen_dataclass_and_proxy_denied():
    packet = evidence()
    object.__setattr__(packet, "paid_compute_requested", "yes")
    with pytest.raises(ScaleRecipeDenied):
        assess_scale("200M", packet, limits(), fake_proxy("200M"))
    forged = {**fake_proxy("200M"), "launch_authorized": True}
    with pytest.raises(ScaleRecipeDenied):
        assess_scale("200M", evidence(), limits(), forged)
    with pytest.raises(ScaleRecipeDenied):
        assess_scale("200M", evidence(), limits(), fake_proxy("300M"))


def test_unknown_scale_refused():
    for tier in ("1B", "35M", "600M", "0M", "", True):
        with pytest.raises(ScaleRecipeDenied):
            scale_recipe(tier)


def test_parent_migration_contract_and_negative_forgery():
    sha1, sha2 = "a" * 64, "b" * 64
    fresh = migration_packet("200M", sha1, sha2, "FRESH_INIT_NEW_RUN")
    assert fresh == migration_packet("200M", sha1, sha2, "FRESH_INIT_NEW_RUN")
    assert fresh["fresh_run_identity_required"] is True
    assert fresh["full_optimizer_migration_proven"] is False
    growth = migration_packet("300M", sha1, sha2, "GROWTH_PROXY_ONLY",
                              {"function_preserved_on_fixture": True,
                               "training_authorized": False})
    assert growth["full_optimizer_migration_proven"] is False
    with pytest.raises(ScaleRecipeDenied):
        migration_packet("200M", "not-a-sha", sha2, "FRESH_INIT_NEW_RUN")
    with pytest.raises(ScaleRecipeDenied):
        migration_packet("200M", sha1, sha2, "GROWTH_PROXY_ONLY", {})
    with pytest.raises(ScaleRecipeDenied):
        migration_packet("200M", sha1, sha2, "FRESH_INIT_NEW_RUN", {})


def test_real_s4_proxy_integration_cpu():
    from twelve_six.accelerated_scaling import ProxyBudget, ProxyProtocol
    protocol = ProxyProtocol(1, "a" * 64, 7, 1, 4, 2)
    budget = ProxyBudget(100_000, 1_000_000_000, 30_000_000, 64)
    for tier in ("200M", "300M", "500M"):
        first = qualify_proxy(tier, protocol, budget)
        second = qualify_proxy(tier, protocol, budget)
        assert first == second
        assert first["full_scale_executed"] is False
        assert first["proxy"]["proxy_receipt"]["promotion_authorized"] is False
        checked = assess_scale(tier, replace(evidence(tier),
                    measured_proxy_receipt_sha256=first["receipt_sha256"]),
                    limits(), first)
        assert checked["training_authorized"] is False



def test_s4_nested_receipt_binding_rejects_rehashed_wrapper():
    from twelve_six.optional_risk_probes import digest

    original = fake_proxy("200M")
    forged_inner = {**original["proxy"], "unverified_worker_receipt": "injected"}
    forged = {**original, "proxy": forged_inner}
    # An untrusted caller can rehash the outer wrapper, but that must not
    # silently replace the previously attested nested S4 receipt.
    forged["receipt_sha256"] = digest({
        key: value for key, value in forged.items() if key != "receipt_sha256"
    })
    matching = replace(
        evidence(), measured_proxy_receipt_sha256=forged["receipt_sha256"]
    )
    with pytest.raises(ScaleRecipeDenied, match="invalid S4 proxy evidence"):
        assess_scale("200M", matching, limits(), forged)


@pytest.mark.parametrize("marker", (
    "training_executed", "paid_compute_authorized", "promotion_authorized",
))
def test_rehashed_nested_proxy_cannot_launder_execution_or_paid_state(marker):
    from twelve_six.optional_risk_probes import digest

    original = fake_proxy("200M")
    original["proxy"]["proxy_receipt"][marker] = True
    # The caller may rehash both untrusted wrappers; a true execution marker
    # still must never appear as an accepted proxy capacity receipt.
    original["s4_proxy_receipt_sha256"] = digest(original["proxy"])
    original["receipt_sha256"] = digest({
        k: v for k, v in original.items() if k != "receipt_sha256"
    })
    claimed = replace(
        evidence(), measured_proxy_receipt_sha256=original["receipt_sha256"],
    )
    with pytest.raises(ScaleRecipeDenied, match="invalid S4 proxy evidence"):
        assess_scale("200M", claimed, limits(), original)
