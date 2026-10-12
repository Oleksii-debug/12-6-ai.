"""Plan 3 S2: independent causal numerical qualification on LOCAL_FREE tiny tensors."""
from __future__ import annotations

import json
import subprocess
import sys

import pytest
import torch

from twelve_six.model import BaseInitManifest, InitSpec, ModelSpec, build_canonical_base
from twelve_six.training.loss import causal_lm_loss, causal_pair_loss


def test_shifted_ce_matches_independent_reference_and_gradient() -> None:
    logits = torch.tensor([[[0.2, -0.1, 0.7], [0.3, 1.1, -0.4],
                            [0.8, -0.5, 0.2], [-0.1, 0.4, 0.6]]],
                          dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([[2, 1, -100, 0]])
    observed = causal_lm_loss(logits, labels)
    reference = (-torch.log_softmax(logits[0, 0], dim=-1)[1]
                 - torch.log_softmax(logits[0, 2], dim=-1)[0]) / 2
    assert torch.allclose(observed, reference, rtol=0, atol=1e-12)
    observed_grad = torch.autograd.grad(observed, logits, retain_graph=True)[0]
    reference_grad = torch.autograd.grad(reference, logits)[0]
    assert torch.allclose(observed_grad, reference_grad, rtol=0, atol=1e-12)
    assert torch.count_nonzero(observed_grad[0, 1]) == 0
    assert torch.count_nonzero(observed_grad[0, 3]) == 0


def test_shifted_ce_finite_difference_gradient_on_tiny_reference() -> None:
    base = torch.tensor([[[0.31, -0.47, 0.22], [-0.19, 0.58, 0.04],
                          [0.87, 0.03, -0.15]]], dtype=torch.float64)
    labels = torch.tensor([[2, 0, 1]])
    variable = base.clone().requires_grad_()
    analytic = torch.autograd.grad(causal_lm_loss(variable, labels), variable)[0]
    epsilon = 1e-6
    for time, token in ((0, 0), (0, 2), (1, 1), (2, 2)):
        left, right = base.clone(), base.clone()
        left[0, time, token] -= epsilon
        right[0, time, token] += epsilon
        numeric = (causal_lm_loss(right, labels) - causal_lm_loss(left, labels)) / (2 * epsilon)
        assert torch.allclose(analytic[0, time, token], numeric, rtol=0, atol=1e-9)


def test_shifted_ce_large_finite_logits_remains_finite() -> None:
    logits = torch.tensor([[[10000.0, -10000.0], [-10000.0, 10000.0],
                            [1.0, -1.0]]], dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([[0, 0, 1]])
    loss = causal_lm_loss(logits, labels)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def _tiny_model():
    spec = ModelSpec(
        schema_version=1, vocab_size=16, max_seq_len=8, d_model=16,
        n_layers=2, n_heads=4, n_kv_heads=2, head_dim=4, d_ff=32,
        rope_rotary_dim=4,
    )
    init = InitSpec()
    manifest = BaseInitManifest(1, spec.identity_sha256(), init.identity_sha256(), 77)
    return build_canonical_base(spec, init, manifest)


def test_future_suffix_cannot_change_prefix_logits_or_prefix_loss_gradient() -> None:
    model = _tiny_model().eval()
    first = torch.tensor([[1, 2, 3, 4, 5]])
    changed = torch.tensor([[1, 2, 3, 9, 10]])
    a = model(first).logits
    b = model(changed).logits
    assert torch.allclose(a[:, :3], b[:, :3], rtol=0, atol=1e-6)
    # The first two target positions depend only on the shared prefix.
    la = causal_lm_loss(a[:, :3], first[:, :3])
    lb = causal_lm_loss(b[:, :3], changed[:, :3])
    ga = torch.autograd.grad(la, model.token_embedding.weight, retain_graph=True)[0]
    gb = torch.autograd.grad(lb, model.token_embedding.weight)[0]
    assert torch.allclose(ga, gb, rtol=0, atol=1e-6)


def test_tiny_decoder_parameter_gradient_matches_finite_difference() -> None:
    model = _tiny_model().double().eval()
    tokens = torch.tensor([[1, 2, 3, 4]])
    loss = causal_lm_loss(model(tokens).logits, tokens)
    gradient = torch.autograd.grad(loss, model.token_embedding.weight)[0][1, 0].item()
    weight = model.token_embedding.weight
    original = weight[1, 0].item()
    epsilon = 1e-5
    with torch.no_grad():
        weight[1, 0] = original - epsilon
    left = causal_lm_loss(model(tokens).logits, tokens).item()
    with torch.no_grad():
        weight[1, 0] = original + epsilon
    right = causal_lm_loss(model(tokens).logits, tokens).item()
    with torch.no_grad():
        weight[1, 0] = original
    assert abs(gradient - (right - left) / (2 * epsilon)) < 1e-7


def test_aligned_padded_pairs_match_manual_masked_mean() -> None:
    logits = torch.tensor([[[1.0, 2.0], [3.0, -1.0], [0.4, 0.6]]],
                          requires_grad=True)
    targets = torch.tensor([[1, 0, -100]])
    mask = torch.tensor([[1, 0, 0]])
    observed = causal_pair_loss(logits, targets, loss_mask=mask)
    expected = -torch.log_softmax(logits[0, 0], dim=-1)[1]
    assert torch.allclose(observed, expected)
    observed.backward()
    assert torch.count_nonzero(logits.grad[0, 1:]) == 0


@pytest.mark.parametrize("bad_logits,bad_targets", [
    (torch.zeros(2, 4), torch.zeros(2, 4, dtype=torch.long)),
    (torch.zeros(2, 4, 3), torch.zeros(2, 3, dtype=torch.long)),
    (torch.zeros(2, 1, 3), torch.zeros(2, 1, dtype=torch.long)),
])
def test_shifted_loss_rejects_invalid_rank_shape_or_length(
    bad_logits: torch.Tensor, bad_targets: torch.Tensor,
) -> None:
    with pytest.raises(ValueError):
        causal_lm_loss(bad_logits, bad_targets)


def test_float32_float64_shifted_loss_agree_on_well_conditioned_logits() -> None:
    logits = torch.tensor([[[0.3, 0.8, -0.7], [-0.5, 0.1, 0.6],
                            [0.4, -0.9, 0.2]]], dtype=torch.float64)
    labels = torch.tensor([[0, 1, 2]])
    fp64 = causal_lm_loss(logits, labels)
    fp32 = causal_lm_loss(logits.float(), labels)
    assert abs(fp32.item() - fp64.item()) < 1e-6


def test_aligned_packed_boundary_mask_excludes_cross_document_target() -> None:
    logits = torch.tensor([[[1.1, -0.3, 0.2], [-0.4, 1.7, 0.1],
                            [0.2, -0.1, 1.9], [0.6, 0.4, -0.2]]],
                          dtype=torch.float64, requires_grad=True)
    targets = torch.tensor([[0, 2, 2, 1]])
    # The second token is a document boundary and must never contribute.
    mask = torch.tensor([[1, 0, 1, 0]])
    observed = causal_pair_loss(logits, targets, loss_mask=mask)
    expected = (-torch.log_softmax(logits[0, 0], dim=-1)[0]
                - torch.log_softmax(logits[0, 2], dim=-1)[2]) / 2
    assert torch.allclose(observed, expected, rtol=0, atol=1e-12)
    observed.backward()
    assert torch.count_nonzero(logits.grad[0, 1]) == 0
    assert torch.count_nonzero(logits.grad[0, 3]) == 0


@pytest.mark.parametrize("mask", [torch.tensor([[1, 2]]),
                                      torch.tensor([[1, -1]])])
def test_aligned_loss_rejects_nonbinary_mask(mask: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="binary"):
        causal_pair_loss(torch.zeros(1, 2, 3), torch.tensor([[0, 1]]), loss_mask=mask)


@pytest.mark.parametrize("labels", [torch.tensor([[-100, -100, -100]]),
                                      torch.tensor([[1, -100, -100]])])
def test_shifted_loss_rejects_zero_valid_targets(labels: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="unmasked|valid"):
        causal_lm_loss(torch.zeros(1, 3, 4), labels)


def test_partitioned_loss_reduction_matches_unpartitioned_loss_and_gradients() -> None:
    logits = torch.tensor([[[0.1, 0.3, -0.2], [-0.6, 0.2, 0.7],
                            [0.8, -0.1, -0.4]],
                           [[-0.2, 0.5, 0.7], [0.1, 1.1, -0.3],
                            [0.6, -0.7, 0.2]]], dtype=torch.float64,
                          requires_grad=True)
    labels = torch.tensor([[0, 1, -100], [2, 0, 1]])
    whole = causal_lm_loss(logits, labels)
    count_a = int(labels[:1, 1:].ne(-100).sum())
    count_b = int(labels[1:, 1:].ne(-100).sum())
    reduced = (count_a * causal_lm_loss(logits[:1], labels[:1])
               + count_b * causal_lm_loss(logits[1:], labels[1:])) / (count_a + count_b)
    assert torch.allclose(whole, reduced, rtol=0, atol=1e-12)
    whole_grad = torch.autograd.grad(whole, logits, retain_graph=True)[0]
    reduced_grad = torch.autograd.grad(reduced, logits)[0]
    assert torch.allclose(whole_grad, reduced_grad, rtol=0, atol=1e-12)


def test_serialized_fresh_process_loss_contract_matches_local_reference() -> None:
    """Fixture transport parity; no claim of an actual server or distributed host."""
    values = [[[0.3, -0.1, 0.8], [0.2, 0.9, -0.3], [0.4, -0.2, 0.1]]]
    labels = [[2, 1, 0]]
    expected = causal_lm_loss(torch.tensor(values, dtype=torch.float64),
                              torch.tensor(labels)).item()
    code = (
        "import json,sys,torch\n"
        "from twelve_six.training.loss import causal_lm_loss\n"
        "p=json.load(sys.stdin)\n"
        "v=torch.tensor(p['logits'],dtype=torch.float64)\n"
        "t=torch.tensor(p['labels'])\n"
        "print(json.dumps({'loss':causal_lm_loss(v,t).item()}))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], input=json.dumps({"logits": values, "labels": labels}),
        text=True, capture_output=True, check=True, timeout=30,
    )
    assert abs(json.loads(proc.stdout)["loss"] - expected) < 1e-12
