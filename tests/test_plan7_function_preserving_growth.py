"""Plan 7 S2: function-preserving transforms, resource gates and fault recovery."""
from __future__ import annotations

from dataclasses import replace

import pytest
import torch

import twelve_six.accelerated_scaling as scaling
from twelve_six.accelerated_scaling import (
    GrowthRejected,
    ProxyBudget,
    ProxyProtocol,
    ScaleAdmissionError,
    estimate_proxy_resources,
    fresh_init_scale_up,
    grow_function_preserving,
)
from twelve_six.model import ModelSpec, TwelveSixDecoder


def spec(**changes) -> ModelSpec:
    core = ModelSpec(
        schema_version=1, vocab_size=32, max_seq_len=16,
        d_model=16, n_layers=2, n_heads=2, n_kv_heads=1, head_dim=8,
        d_ff=32, rope_rotary_dim=8,
    )
    return replace(core, **changes)


def fixture() -> ProxyProtocol:
    return ProxyProtocol(
        version=1, fixture_sha256="c" * 64,
        seed=99, batch=2, sequence=6, repeats=2,
    )


def budget() -> ProxyBudget:
    return ProxyBudget(
        max_parameters=100_000, max_flops=200_000_000,
        max_memory_bytes=128_000_000, max_tokens=1024,
    )


def parent_model(**changes) -> TwelveSixDecoder:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(71)
        return TwelveSixDecoder(spec(**changes)).cpu().eval()


@pytest.mark.parametrize(
    ("added_ffn", "added_layers", "bias", "tie"),
    [(16, 0, False, True), (0, 1, False, True),
     (16, 2, True, True), (32, 1, True, False)],
)
def test_exact_functional_parity(
    added_ffn: int, added_layers: int, bias: bool, tie: bool,
) -> None:
    parent = parent_model(attention_bias=bias, mlp_bias=bias, tie_word_embeddings=tie)
    before_state = scaling._growth_state_sha256(parent)
    target = replace(
        parent.spec,
        d_ff=parent.spec.d_ff + added_ffn,
        n_layers=parent.spec.n_layers + added_layers,
    )
    rng = torch.get_rng_state().clone()
    descendant, receipt = grow_function_preserving(parent, target, fixture(), budget(), seed=41)
    assert torch.equal(rng, torch.get_rng_state())
    assert scaling._growth_state_sha256(parent) == before_state
    assert receipt["parent_state_sha256"] == before_state
    assert receipt["added_ffn_width"] == added_ffn
    assert receipt["added_layers"] == added_layers
    assert receipt["function_preserved_on_fixture"] is True
    assert receipt["parity_max_abs_error"] <= receipt["parity_atol"]
    assert receipt["mode"] == "FUNCTION_PRESERVING_GROWTH"
    assert receipt["policy"] == "COPY_PREFIX_ZERO_NEW_CHANNELS_AND_IDENTITY_RESIDUALS"
    assert receipt["optimizer_state_transferred"] is False
    assert receipt["stage_promotion_authorized"] is False
    assert receipt["training_authorized"] is False
    assert receipt["paid_compute_authorized"] is False
    assert receipt["resource_admission"]["parameters"] == target.parameter_count()
    assert receipt["joint_resource_admission"]["joint_planning_memory_bytes"] <= budget().max_memory_bytes
    assert len(receipt["receipt_sha256"]) == 64
    assert all(p.requires_grad for p in descendant.parameters())
    ids = torch.tensor([[1, 2, 3, 4], [3, 2, 1, 0]])
    torch.testing.assert_close(
        descendant(ids).logits, parent(ids).logits, atol=1e-5, rtol=0,
    )
    if tie:
        assert descendant.lm_head.weight is descendant.token_embedding.weight
    else:
        assert descendant.lm_head.weight is not descendant.token_embedding.weight


def test_zero_appended_channels_and_identity_residual_are_explicit() -> None:
    parent = parent_model()
    new_spec = replace(parent.spec, d_ff=64, n_layers=4)
    child, receipt = grow_function_preserving(parent, new_spec, fixture(), budget())
    for old, new in zip(parent.blocks, child.blocks[:2], strict=True):
        assert torch.equal(old.mlp.gate_proj.weight, new.mlp.gate_proj.weight[:32])
        assert torch.equal(old.mlp.up_proj.weight, new.mlp.up_proj.weight[:32])
        assert torch.count_nonzero(new.mlp.gate_proj.weight[32:]) == 0
        assert torch.count_nonzero(new.mlp.up_proj.weight[32:]) == 0
        assert torch.count_nonzero(new.mlp.down_proj.weight[:, 32:]) == 0
    for new in child.blocks[2:]:
        assert torch.count_nonzero(new.attn.out_proj.weight) == 0
        assert torch.count_nonzero(new.mlp.down_proj.weight) == 0
    assert any(row["policy"] == "COPY_COLS_ZERO_APPEND" for row in receipt["tensor_mapping"])
    assert any(row["policy"] == "NEW_RESIDUAL_LAYER" for row in receipt["tensor_mapping"])


def test_fresh_process_checkpoint_reload_of_descendant() -> None:
    parent = parent_model()
    target = replace(parent.spec, d_ff=48, n_layers=3)
    child, receipt = grow_function_preserving(parent, target, fixture(), budget())
    reloaded = TwelveSixDecoder(target, child.init_spec).cpu().eval()
    reloaded.load_state_dict(child.state_dict(), strict=True)
    assert scaling._growth_state_sha256(reloaded) == receipt["descendant_state_sha256"]
    ids = torch.tensor([[1, 2, 3, 4]])
    torch.testing.assert_close(
        child(ids).logits, reloaded(ids).logits, atol=0, rtol=0,
    )


def test_two_simulated_worker_replays_are_byte_identical() -> None:
    parent = parent_model()
    target = replace(parent.spec, d_ff=48, n_layers=3)
    first, first_receipt = grow_function_preserving(parent, target, fixture(), budget(), seed=73)
    second, second_receipt = grow_function_preserving(parent, target, fixture(), budget(), seed=73)
    assert first_receipt == second_receipt
    assert scaling._growth_state_sha256(first) == scaling._growth_state_sha256(second)


def test_transient_transfer_failure_does_not_mutate_parent_and_restart_recovers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = parent_model()
    target = replace(parent.spec, d_ff=48, n_layers=3)
    original_hash = scaling._growth_state_sha256(parent)
    expected, expected_receipt = grow_function_preserving(
        parent, target, fixture(), budget(), seed=5,
    )
    original_copy = scaling._copy_growth_weights

    def interrupted(source: TwelveSixDecoder, child: TwelveSixDecoder):
        original_copy(source, child)
        raise RuntimeError("simulated worker crash after uncommitted copy")

    monkeypatch.setattr(scaling, "_copy_growth_weights", interrupted)
    with pytest.raises(RuntimeError, match="simulated worker crash"):
        grow_function_preserving(parent, target, fixture(), budget(), seed=5)
    assert scaling._growth_state_sha256(parent) == original_hash
    monkeypatch.setattr(scaling, "_copy_growth_weights", original_copy)
    recovered, receipt = grow_function_preserving(
        parent, target, fixture(), budget(), seed=5,
    )
    assert receipt == expected_receipt
    assert scaling._growth_state_sha256(recovered) == scaling._growth_state_sha256(expected)


@pytest.mark.parametrize(
    "changes",
    [
        {"d_ff": 16},
        {"n_layers": 1},
        {"vocab_size": 40},
        {"n_heads": 4},
        {"n_kv_heads": 2},
        {"max_seq_len": 20},
        {"d_ff": 32, "n_layers": 2},
    ],
)
def test_unsupported_or_shrinking_change_rejected_before_allocation(
    changes: dict, monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = parent_model()
    original_init = TwelveSixDecoder.__init__

    def forbidden(*args, **kwargs):
        raise AssertionError("descendant unexpectedly allocated")

    monkeypatch.setattr(TwelveSixDecoder, "__init__", forbidden)
    with pytest.raises(GrowthRejected):
        grow_function_preserving(parent, replace(parent.spec, **changes), fixture(), budget())
    monkeypatch.setattr(TwelveSixDecoder, "__init__", original_init)


def test_resource_and_context_admission_fail_before_alloc(monkeypatch: pytest.MonkeyPatch) -> None:
    parent = parent_model()
    target = replace(parent.spec, d_ff=48)
    def forbidden(*args, **kwargs):
        raise AssertionError("descendant unexpectedly allocated")
    monkeypatch.setattr(TwelveSixDecoder, "__init__", forbidden)
    with pytest.raises(ScaleAdmissionError, match="parameter budget"):
        grow_function_preserving(parent, target, fixture(), ProxyBudget(max_parameters=1))
    with pytest.raises(ScaleAdmissionError, match="sequence exceeds"):
        grow_function_preserving(
            parent, target, replace(fixture(), sequence=17), budget(),
        )


def test_nonfinite_and_training_parent_are_rejected() -> None:
    model = parent_model()
    target = replace(model.spec, n_layers=3)
    model.train()
    with pytest.raises(GrowthRejected, match="eval"):
        grow_function_preserving(model, target, fixture(), budget())
    model.eval()
    with torch.no_grad():
        model.token_embedding.weight[0, 0] = float("nan")
    with pytest.raises(GrowthRejected, match="nonfinite"):
        grow_function_preserving(model, target, fixture(), budget())


@pytest.mark.parametrize("seed", [-1, True, 1.0])
def test_invalid_seed_fails_closed(seed) -> None:
    parent = parent_model()
    with pytest.raises(GrowthRejected):
        grow_function_preserving(
            parent, replace(parent.spec, d_ff=48), fixture(), budget(), seed=seed,
        )


def test_strict_parity_rejects_tampered_descendant(monkeypatch: pytest.MonkeyPatch) -> None:
    parent = parent_model()
    actual_copy = scaling._copy_growth_weights

    def tampered(source, descendant):
        mapping = actual_copy(source, descendant)
        with torch.no_grad():
            descendant.token_embedding.weight.add_(10.0)
        return mapping

    monkeypatch.setattr(scaling, "_copy_growth_weights", tampered)
    with pytest.raises(GrowthRejected, match="function parity failed"):
        grow_function_preserving(
            parent, replace(parent.spec, d_ff=48), fixture(), budget(), parity_atol=0.0,
        )


def test_explicit_fresh_init_fallback_cannot_claim_preservation() -> None:
    parent = parent_model()
    target = replace(parent.spec, d_ff=48, n_layers=3)
    child, receipt = fresh_init_scale_up(parent, target, fixture(), budget(), seed=3)
    assert child.spec.identity_sha256() != parent.spec.identity_sha256()
    assert receipt["policy"] == "NO_WEIGHT_TRANSFER"
    assert receipt["mode"] == "FRESH_INIT_SCALE_UP"
    assert receipt["function_preserved_on_fixture"] is False
    assert receipt["parity_max_abs_error"] is None
    assert receipt["tensor_mapping"] == []
    assert receipt["stage_promotion_authorized"] is False
    assert receipt["checkpoint_resume_authorized"] is False
    assert receipt["paid_compute_authorized"] is False
    with pytest.raises(GrowthRejected, match="must change"):
        fresh_init_scale_up(parent, parent.spec, fixture(), budget())




@pytest.mark.parametrize("dimension", ["memory", "flops", "tokens"])
def test_joint_resource_envelope_rejects_before_allocation(
    dimension: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = parent_model()
    target = replace(parent.spec, d_ff=48, n_layers=3)
    old = estimate_proxy_resources(parent.spec, fixture())
    new = estimate_proxy_resources(target, fixture())
    field, limit = {
        "memory": (
            "max_memory_bytes",
            old["planning_memory_bytes"] + new["planning_memory_bytes"] - 1,
        ),
        "flops": (
            "max_flops",
            old["total_proxy_flops"] + new["total_proxy_flops"] - 1,
        ),
        "tokens": (
            "max_tokens",
            old["total_proxy_tokens"] + new["total_proxy_tokens"] - 1,
        ),
    }[dimension]
    # The individual operation fits. Only the combined envelope is denied.
    assert limit >= {
        "memory": new["planning_memory_bytes"],
        "flops": new["total_proxy_flops"],
        "tokens": new["total_proxy_tokens"],
    }[dimension]
    constrained = replace(budget(), **{field: limit})

    def forbidden(*args, **kwargs):
        raise AssertionError("descendant unexpectedly allocated")

    monkeypatch.setattr(TwelveSixDecoder, "__init__", forbidden)
    with pytest.raises(ScaleAdmissionError, match="joint parent/descendant"):
        grow_function_preserving(parent, target, fixture(), constrained)
    with pytest.raises(ScaleAdmissionError, match="joint parent/descendant"):
        fresh_init_scale_up(parent, target, fixture(), constrained)

def test_costly_gpu_protocol_rejected_even_for_tiny_models() -> None:
    with pytest.raises(ScaleAdmissionError, match="LOCAL_FREE"):
        replace(fixture(), resource_class="PAID_GPU")
