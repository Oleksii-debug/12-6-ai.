from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import twelve_six.s0_candidate_evaluation as candidate_eval


class _DummyModel:
    def eval(self) -> _DummyModel:
        return self

    def __call__(self, input_ids: torch.Tensor) -> SimpleNamespace:
        logits = torch.zeros(
            (*input_ids.shape, 4),
            dtype=torch.float32,
            device=input_ids.device,
        )
        return SimpleNamespace(logits=logits)


def _batch(targets: list[int]) -> dict[str, torch.Tensor]:
    labels = torch.tensor([[0, *targets]], dtype=torch.long)
    return {"input_ids": labels.clone(), "labels": labels}


def _analytic_mean_loss(
    _logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int = -100,
) -> torch.Tensor:
    shifted = labels[:, 1:]
    valid = shifted.ne(ignore_index)
    if not bool(valid.any().item()):
        return torch.tensor(float("nan"))
    return shifted.masked_select(valid).float().mean()


def test_mean_loss_is_invariant_to_unequal_batch_partition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(candidate_eval, "causal_lm_loss", _analytic_mean_loss)
    model = _DummyModel()

    split = candidate_eval._mean_loss(
        model,
        [_batch([1]), _batch([3, 3, 3])],
    )
    merged = candidate_eval._mean_loss(
        model,
        [_batch([1, 3, 3, 3])],
    )

    assert split == pytest.approx(2.5)
    assert split == pytest.approx(merged)


def test_mean_loss_counts_only_unmasked_shifted_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(candidate_eval, "causal_lm_loss", _analytic_mean_loss)
    model = _DummyModel()

    result = candidate_eval._mean_loss(
        model,
        [_batch([1, -100, -100]), _batch([3, 3, 3])],
    )

    assert result == pytest.approx(2.5)


def test_mean_loss_equal_target_counts_preserve_mean_of_means_control(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(candidate_eval, "causal_lm_loss", _analytic_mean_loss)
    model = _DummyModel()

    result = candidate_eval._mean_loss(
        model,
        [_batch([1, 1]), _batch([3, 3])],
    )

    assert result == pytest.approx(2.0)


def test_mean_loss_rejects_empty_batch_collection() -> None:
    with pytest.raises(ValueError, match="at least one batch"):
        candidate_eval._mean_loss(_DummyModel(), [])


def test_mean_loss_rejects_batch_without_unmasked_causal_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(candidate_eval, "causal_lm_loss", _analytic_mean_loss)

    with pytest.raises(ValueError, match="unmasked causal target"):
        candidate_eval._mean_loss(_DummyModel(), [_batch([-100, -100])])
