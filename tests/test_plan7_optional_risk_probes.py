"""Plan 7 S4 bounded optional recipes, negative cases and CPU proxy interoperability."""
from dataclasses import replace
from io import BytesIO

import pytest
import torch

from twelve_six.accelerated_scaling import (
    ProxyBudget,
    ProxyProtocol,
    ScaleAdmissionError,
    grow_function_preserving,
)
from twelve_six.model import InitSpec, TwelveSixDecoder
from twelve_six.optional_risk_probes import (
    ProbeLimits,
    RiskEvidence,
    RiskProbeDenied,
    decide_probe,
    full_recipe,
    model_for_tier,
    tiny_analog,
    verify_recipe,
)


def evidence(**kw):
    raw = RiskEvidence(
        1, "a" * 64, "b" * 64, "fixture-run", 4, 2.0,
        0.5, 3.0, 4_000_000_000, 10.0, 1,
    )
    return replace(raw, **kw)


def limits(**kw):
    return replace(ProbeLimits(5.0, 20.0, 4_000_000_000, 2), **kw)


@pytest.mark.parametrize("tier", ("35M", "50M", "100M"))
def test_real_modelspec_recipe_identity_and_no_automatic_campaign(tier):
    recipe = full_recipe(tier)
    assert recipe["modelspec_sha256"] == model_for_tier(tier).identity_sha256()
    assert recipe["parameters"] == model_for_tier(tier).parameter_count()
    assert abs(recipe["parameters"] - recipe["parameter_target"]) <= (
        recipe["parameter_target"] // 20
    )
    assert recipe == full_recipe(tier)
    assert verify_recipe(recipe)
    assert not verify_recipe({**recipe, "parameters": -1})
    for field in ("mandatory_campaign", "launch_authorized",
                  "training_authorized", "paid_compute_authorized"):
        assert recipe[field] is False


@pytest.mark.parametrize("tier", ("35M", "50M", "100M"))
def test_optional_risk_gating_deterministic_but_never_a_launch_grant(tier):
    packet = decide_probe(tier, evidence(), limits())
    assert packet == decide_probe(tier, evidence(), limits())
    assert packet["status"] == "OPTIONAL_RECIPE_ELIGIBLE_NOT_AUTHORIZED"
    assert packet["launch_authorized"] is False
    assert packet["training_authorized"] is False
    assert packet["paid_compute_authorized"] is False
    assert decide_probe(tier, evidence(paid_compute_requested=True), limits())[
        "status"
    ] == "NO_GO"


@pytest.mark.parametrize("changes,reason", [
    ({"observations": 1}, "insufficient_observations"),
    ({"expected_risk_reduction": 0.0}, "no_material_risk_reduction"),
    ({"expected_probe_cost_units": 6.0}, "cost_budget_exceeded"),
    ({"expected_duration_seconds": 21.0}, "duration_budget_exceeded"),
    ({"healthy_workers": 3}, "worker_budget_exceeded"),
    ({"observed_free_capacity_bytes": 1}, "insufficient_observed_capacity"),
])
def test_resource_failure_negative_proof(changes, reason):
    packet = decide_probe("35M", evidence(**changes), limits())
    assert packet["status"] == "NO_GO"
    assert reason in packet["reasons"]
    assert packet["launch_authorized"] is False


def test_preallocation_budget_denies_a_full_100m_recipe():
    packet = decide_probe(
        "100M", evidence(), limits(max_peak_memory_bytes=1),
    )
    assert "peak_memory_budget_exceeded" in packet["reasons"]


@pytest.mark.parametrize("change", [
    {"observations": True},
    {"observations": -1},
    {"fixture_sha256": "forged"},
    {"healthy_workers": 0},
    {"measured_risk": float("nan")},
    {"expected_risk_reduction": float("inf")},
    {"expected_risk_reduction": 3.0},
])
def test_invalid_evidence_never_admitted(change):
    with pytest.raises(RiskProbeDenied):
        evidence(**change)


@pytest.mark.parametrize("schema_version", (True, 1.0, "1", 2))
def test_schema_version_requires_exact_positive_integer_one(schema_version):
    with pytest.raises(RiskProbeDenied):
        evidence(schema_version=schema_version)


def test_forged_frozen_dataclass_denied_at_trust_boundary():
    record = evidence()
    object.__setattr__(record, "paid_compute_requested", "yes")
    with pytest.raises(RiskProbeDenied):
        decide_probe("35M", record, limits())


def test_unsupported_tier_is_not_silently_extended():
    with pytest.raises(RiskProbeDenied):
        full_recipe("200M")
    assert verify_recipe({"tier": "200M"}) is False


@pytest.mark.parametrize("tier", ("35M", "50M", "100M"))
def test_actual_tiny_cpu_proxy_and_restart_replay(tier):
    protocol = ProxyProtocol(1, "a" * 64, 7, 1, 4, 2)
    budget = ProxyBudget(100_000, 1_000_000_000, 30_000_000, 64)
    first = tiny_analog(tier, protocol, budget)
    second = tiny_analog(tier, protocol, budget)
    assert first == second
    assert first["proxy_receipt"]["resources"] == first["admission"]
    assert first["proxy_receipt"]["promotion_authorized"] is False
    assert first["full_scale_executed"] is False
    assert first["paid_compute_authorized"] is False


def test_proxy_preallocation_denial():
    protocol = ProxyProtocol(1, "a" * 64, 7, 1, 4, 2)
    with pytest.raises(ScaleAdmissionError):
        tiny_analog("100M", protocol, ProxyBudget(1, 1, 1, 1))


def test_incumbent_checkpoint_serving_and_growth_contract_on_tiny_cpu():
    spec = replace(
        model_for_tier("35M"), vocab_size=64, max_seq_len=16,
        d_model=32, n_layers=2, n_heads=4, n_kv_heads=2,
        head_dim=8, d_ff=64, rope_rotary_dim=8,
    )
    parent = TwelveSixDecoder(spec, InitSpec()).eval()
    tokens = torch.tensor([[1, 2, 3, 4]])
    before = parent(tokens).logits.detach()
    stream = BytesIO()
    torch.save(parent.state_dict(), stream)
    stream.seek(0)
    restored = TwelveSixDecoder(spec, InitSpec()).eval()
    restored.load_state_dict(
        torch.load(stream, map_location="cpu", weights_only=True),
    )
    torch.testing.assert_close(before, restored(tokens).logits, rtol=0, atol=0)
    protocol = ProxyProtocol(1, "b" * 64, 9, 1, 4, 2)
    budget = ProxyBudget(100_000, 1_000_000_000, 30_000_000, 64)
    child, receipt = grow_function_preserving(
        restored, replace(spec, d_ff=80, n_layers=3),
        protocol, budget, seed=11,
    )
    assert receipt["function_preserved_on_fixture"] is True
    assert receipt["training_authorized"] is False
    torch.testing.assert_close(
        before, child(tokens).logits, rtol=0, atol=1e-5,
    )
