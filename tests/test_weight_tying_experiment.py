from __future__ import annotations

import json
from pathlib import Path

import pytest

from twelve_six.model import ModelSpec
from twelve_six.weight_tying_experiment import (
    SCHEMA,
    _load_config,
    validate_matched_candidates,
)


CONFIG = Path("configs/experiments/model16_weight_tying_500k.json")


def test_model16_config_is_exact_and_matched() -> None:
    payload = _load_config(CONFIG)
    assert payload["schema"] == SCHEMA
    items = {item["label"]: item for item in payload["candidates"]}
    tied = ModelSpec.from_dict(items["A_tied"]["model"])
    untied = ModelSpec.from_dict(items["B_untied"]["model"])
    assert tied.parameter_count() == items["A_tied"]["expected_parameters"] == 467_808
    assert untied.parameter_count() == items["B_untied"]["expected_parameters"] == 468_192
    assert tied.identity_sha256() == items["A_tied"]["expected_model_identity_sha256"]
    assert untied.identity_sha256() == items["B_untied"]["expected_model_identity_sha256"]
    validate_matched_candidates(
        tied,
        untied,
        max_relative_parameter_delta=payload["max_relative_parameter_delta"],
    )
    relative = abs(tied.parameter_count() - untied.parameter_count()) / (
        (tied.parameter_count() + untied.parameter_count()) / 2
    )
    assert relative < 0.001
    assert tied.d_ff == 256
    assert untied.d_ff == 235


def test_model16_rejects_unrebalanced_untied_candidate() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    items = {item["label"]: item for item in payload["candidates"]}
    tied = ModelSpec.from_dict(items["A_tied"]["model"])
    bad_payload = dict(items["B_untied"]["model"])
    bad_payload["d_ff"] = tied.d_ff
    bad = ModelSpec.from_dict(bad_payload)
    with pytest.raises(ValueError, match="rebalance"):
        validate_matched_candidates(
            tied,
            bad,
            max_relative_parameter_delta=1.0,
        )


def test_tying_changes_model_identity() -> None:
    payload = _load_config(CONFIG)
    items = {item["label"]: item for item in payload["candidates"]}
    tied = ModelSpec.from_dict(items["A_tied"]["model"])
    untied = ModelSpec.from_dict(items["B_untied"]["model"])
    assert tied.tie_word_embeddings is True
    assert untied.tie_word_embeddings is False
    assert tied.identity_sha256() != untied.identity_sha256()
