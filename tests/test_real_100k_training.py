from __future__ import annotations

from collections import Counter
from pathlib import Path

import torch

from twelve_six.model import load_stage_config
from twelve_six.real_100k_training import (
    EXPECTED_CORPUS_SHA256,
    HOLDOUT_INDICES,
    MIXTURE_PATTERN,
    _make_batch,
    _model_spec,
    _split_corpus,
)


def test_experiment_reuses_current_s1_geometry_except_evidence_vocab() -> None:
    canonical = load_stage_config(Path("configs/stages/s1_100k.json"))
    spec, evidence = _model_spec(Path("."), 472)

    assert spec.parameter_count() == 105_936
    assert 95_000 <= spec.parameter_count() <= 110_000
    assert spec.vocab_size == 472
    for field, value in canonical.model.to_dict().items():
        if field != "vocab_size":
            assert spec.to_dict()[field] == value
    assert evidence["canonical_s1_model_identity_sha256"] == canonical.model.identity_sha256()
    assert evidence["only_experimental_geometry_change"] == "vocab_size"


def test_data10_split_is_exact_disjoint_and_one_heldout_per_stratum() -> None:
    split = _split_corpus(Path("."))

    assert split["manifest"]["source_sha256"] == EXPECTED_CORPUS_SHA256
    assert split["manifest"]["holdout_line_indices_zero_based"] == list(HOLDOUT_INDICES)
    assert {name: len(values) for name, values in split["validation_by_stratum"].items()} == {
        "uk": 1,
        "en": 1,
        "code": 1,
    }
    train = set(split["all_train_texts"])
    validation = set(split["all_validation_texts"])
    assert train.isdisjoint(validation)
    assert len(train) == 14
    assert len(validation) == 3


def test_mixture_pattern_is_exact_45_35_20_per_twenty_steps() -> None:
    assert len(MIXTURE_PATTERN) == 20
    assert Counter(MIXTURE_PATTERN) == Counter({"uk": 9, "en": 7, "code": 4})


def test_batch_builder_is_deterministic_cyclic_and_trainer_shaped() -> None:
    stream = [3, 5, 7, 11, 13]
    first = _make_batch(stream, occurrence=2, batch_size=3, sequence_length=4)
    second = _make_batch(stream, occurrence=2, batch_size=3, sequence_length=4)

    assert torch.equal(first, second)
    assert first.shape == (3, 4)
    assert set(first.flatten().tolist()) <= set(stream)
