from __future__ import annotations

import pytest

from twelve_six.integration.s2_transition_preflight import collect_s2_transition_preflight


def test_s2_transition_preflight_proves_only_bounded_mechanics() -> None:
    report = collect_s2_transition_preflight(sequence_length=8)

    assert report["schema"] == "12-6.s2-transition-preflight.v1"
    assert report["authority"] == "ENGINEERING_S2_MECHANICS_PREFLIGHT_ONLY_NOT_STAGE_EVIDENCE"
    assert report["scope"] == "SYNTHETIC_TOKEN_IDS_ONLY_NOT_S2_DATA_OR_TOKENIZER"
    assert report["stage"] == "S2"
    assert report["target_parameters"] == 1_000_000
    assert report["expected_parameters"] == 1_066_112
    assert report["actual_trainable_parameters"] == 1_066_112
    assert report["logits_shape"] == [1, 8, 2048]
    assert report["all_gradients_finite"] is True
    assert report["any_gradient_nonzero"] is True
    assert report["optimizer_steps"] == 0
    assert report["tokenizer_selected"] is False
    assert report["data_selected"] is False
    assert report["architecture_frozen"] is False
    assert report["quality_claim"] is False
    assert report["promotion_allowed"] is False
    assert report["paid_compute"] is False
    assert report["canonical_base"] == "random_init"


def test_s2_transition_preflight_rejects_invalid_sequence_length() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        collect_s2_transition_preflight(sequence_length=1)
