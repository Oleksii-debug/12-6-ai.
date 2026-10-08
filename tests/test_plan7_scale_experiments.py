"""Plan 7 S1: bounded acceptance, identity, failure and replay tests."""
from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from twelve_six.model import InitSpec, ModelSpec
from twelve_six.accelerated_scaling import (
    ArchitectureHypothesis,
    ProxyBudget,
    ProxyProtocol,
    ScaleAdmissionError,
    admit_proxy,
    compare_proxies,
    estimate_proxy_resources,
    run_proxy,
)


def spec() -> ModelSpec:
    return ModelSpec(
        schema_version=1, vocab_size=32, max_seq_len=16, d_model=16,
        n_layers=2, n_heads=2, n_kv_heads=1, head_dim=8, d_ff=32,
        rope_rotary_dim=8,
    )


def protocol() -> ProxyProtocol:
    return ProxyProtocol(version=1, fixture_sha256="a" * 64, seed=17, batch=2, sequence=6)


def hypothesis(variant="baseline", field="", value=0):
    return ArchitectureHypothesis(1, variant + "-v1", variant, field, value)


def test_identity_stable_and_parent_unmodified() -> None:
    parent = spec()
    identity = parent.identity_sha256()
    assert hypothesis().candidate(parent) is parent
    grown = hypothesis("mlp_width", "d_ff", 48).candidate(parent)
    assert grown.d_ff == 48
    assert grown.identity_sha256() != identity
    assert parent.identity_sha256() == identity
    assert grown.parameter_count() > parent.parameter_count()


@pytest.mark.parametrize(
    ("variant", "field", "value"),
    [
        ("mlp_width", "d_ff", 48),
        ("depth", "n_layers", 3),
        ("gqa", "n_kv_heads", 2),
        ("heads", "n_heads", 4),
        ("context", "max_seq_len", 24),
    ],
)
def test_supported_scale_hypotheses(variant: str, field: str, value: int) -> None:
    parent = spec()
    candidate = hypothesis(variant, field, value).candidate(parent)
    assert getattr(candidate, field) == value
    assert parent.identity_sha256() != candidate.identity_sha256()


def test_exact_replay_across_fresh_proxy_invocations() -> None:
    before = torch.get_rng_state().clone()
    left = run_proxy(spec(), hypothesis(), protocol(), ProxyBudget())
    right = run_proxy(spec(), hypothesis(), protocol(), ProxyBudget())
    assert left == right
    assert left["resources"]["parameters"] == spec().parameter_count()
    assert len(left["output_sha256"]) == 2
    assert left["evidence_class"] == "LOCAL_FREE_SYNTHETIC_PROXY_ONLY"
    assert left["promotion_authorized"] is False
    assert left["training_executed"] is False
    assert torch.equal(before, torch.get_rng_state())


def test_bounded_comparable_candidate_experiments() -> None:
    base = run_proxy(spec(), hypothesis(), protocol(), ProxyBudget())
    gqa = run_proxy(spec(), hypothesis("gqa", "n_kv_heads", 2), protocol(), ProxyBudget())
    comparison = compare_proxies(base, gqa)
    assert comparison["promotion_authorized"] is False
    assert comparison["left_experiment_id"] != comparison["right_experiment_id"]


def test_analytic_accounting_and_quadratic_attention_context() -> None:
    short = estimate_proxy_resources(spec(), protocol())
    wide = estimate_proxy_resources(spec(), replace(protocol(), sequence=12))
    assert short["parameters"] == spec().parameter_count()
    assert short["parameter_bytes"] == short["parameters"] * 4
    assert wide["forward_flops"] > short["forward_flops"]
    assert wide["planning_memory_bytes"] > short["planning_memory_bytes"]
    assert short["score_equivalent_activation_bytes"] > 0
    assert wide["total_proxy_tokens"] == 2 * 12 * protocol().repeats


@pytest.mark.parametrize(
    "budget",
    [
        ProxyBudget(max_parameters=1),
        ProxyBudget(max_flops=1),
        ProxyBudget(max_memory_bytes=1),
        ProxyBudget(max_tokens=1),
    ],
)
def test_fail_closed_before_model_allocation(budget: ProxyBudget, monkeypatch) -> None:
    import twelve_six.accelerated_scaling as target

    def forbidden(*args, **kwargs):
        raise AssertionError("unauthorized model allocation")

    monkeypatch.setattr(target, "TwelveSixDecoder", forbidden)
    with pytest.raises(ScaleAdmissionError):
        run_proxy(spec(), hypothesis(), protocol(), budget)


@pytest.mark.parametrize(
    "bad",
    [
        {"version": 2},
        {"fixture_sha256": "wrong"},
        {"seed": True},
        {"batch": 0},
        {"sequence": 0},
        {"repeats": -1},
        {"dtype": "float16"},
        {"resource_class": "PAID_GPU"},
    ],
)
def test_invalid_protocols_fail_closed(bad: dict) -> None:
    with pytest.raises((ValueError, ScaleAdmissionError)):
        replace(protocol(), **bad)


@pytest.mark.parametrize(
    ("variant", "field", "value"),
    [
        ("mlp_width", "n_layers", 32),
        ("arbitrary", "d_ff", 32),
        ("gqa", "n_kv_heads", 0),
        ("baseline", "", 99),
    ],
)
def test_invalid_variants_rejected(variant: str, field: str, value: int) -> None:
    with pytest.raises(ValueError):
        hypothesis(variant, field, value)


def test_context_and_geometry_fail_closed() -> None:
    with pytest.raises(ScaleAdmissionError):
        admit_proxy(spec(), replace(protocol(), sequence=17), ProxyBudget())
    with pytest.raises(ValueError):
        hypothesis("gqa", "n_kv_heads", 3).candidate(spec())


def test_noncomparable_fixture_and_init_rejected() -> None:
    left = run_proxy(spec(), hypothesis(), protocol(), ProxyBudget())
    changed_protocol = replace(protocol(), fixture_sha256="b" * 64)
    right = run_proxy(spec(), hypothesis(), changed_protocol, ProxyBudget())
    with pytest.raises(ValueError, match="non-comparable"):
        compare_proxies(left, right)
    other_init = run_proxy(spec(), hypothesis(), protocol(), ProxyBudget(), init=InitSpec(std=0.01))
    with pytest.raises(ValueError, match="non-comparable"):
        compare_proxies(left, other_init)


def test_tampered_or_promoted_receipts_fail_closed() -> None:
    left = run_proxy(spec(), hypothesis(), protocol(), ProxyBudget())
    for mutation in (
        {"promotion_authorized": True},
        {"training_executed": True},
        {"evidence_class": "PRODUCT_QUALITY"},
        {"losses": [float("nan")]},
    ):
        bad = {**left, **mutation}
        with pytest.raises(ValueError):
            compare_proxies(left, bad)
    with pytest.raises(ValueError):
        compare_proxies(left, {"schema_version": 1})


def test_restart_after_simulated_worker_failure_is_idempotent() -> None:
    baseline = run_proxy(spec(), hypothesis(), protocol(), ProxyBudget())
    with pytest.raises(ScaleAdmissionError):
        run_proxy(spec(), hypothesis(), protocol(), ProxyBudget(max_flops=1))
    recovered = run_proxy(spec(), hypothesis(), protocol(), ProxyBudget())
    assert baseline == recovered


def test_resource_envelope_rejects_unbounded_candidate() -> None:
    enormous = hypothesis("mlp_width", "d_ff", 1_000_000).candidate(spec())
    with pytest.raises(ScaleAdmissionError, match="parameter budget"):
        admit_proxy(enormous, protocol(), ProxyBudget())
