from __future__ import annotations

import math
from pathlib import Path

import torch

from twelve_six.learn04_1m_entrypoint import bind_data10_normalized_records

bind_data10_normalized_records()

from twelve_six.learn04_1m_experiment import (  # noqa: E402
    DATA10_CORPUS_SHA256,
    DEFAULT_TOKEN_BUDGETS,
    PACKING_VERSION,
    TOKENIZER_EXPECTED_VOCAB,
    TRAIN_RECORDS,
    VALIDATION_RECORDS,
    _EXPECTED_COUNTS_VOCAB472,
    _budget_steps,
    _joined_training_text,
    _make_batch,
    _sha_text,
    _verify_data_boundary,
    controlled_specs,
)
from twelve_six.packing.scale_contracts import MixturePlan, MixtureSource  # noqa: E402


def test_data10_exact_training_bytes_and_split_boundary() -> None:
    evidence = _verify_data_boundary(Path("."))
    assert evidence["corpus_sha256"] == DATA10_CORPUS_SHA256
    assert evidence["representative_corpus"] is False
    assert evidence["external_sources_training_approved"] == 0
    assert evidence["exact_train_validation_overlap"] == []
    assert _sha_text(_joined_training_text()) == DATA10_CORPUS_SHA256
    assert {_sha_text(text) for _, _, text in TRAIN_RECORDS}.isdisjoint(
        {_sha_text(text) for _, _, text in VALIDATION_RECORDS}
    )


def test_bpe_vocab_changes_only_vocab_surface_of_research41_geometries() -> None:
    specs = controlled_specs(TOKENIZER_EXPECTED_VOCAB)
    assert tuple(spec.parameter_count() for spec in specs) == _EXPECTED_COUNTS_VOCAB472
    assert {spec.vocab_size for spec in specs} == {472}
    assert {spec.max_seq_len for spec in specs} == {256}
    assert [(spec.d_model, spec.n_layers, spec.n_heads, spec.n_kv_heads, spec.d_ff) for spec in specs] == [
        (48, 3, 4, 4, 128),
        (72, 4, 6, 6, 192),
        (96, 4, 6, 6, 256),
        (128, 5, 8, 8, 352),
    ]


def test_matched_token_budget_steps_are_deterministic() -> None:
    assert _budget_steps(DEFAULT_TOKEN_BUDGETS, 252) == (17, 66, 261, 1041)


def test_mixture_schedule_and_batch_are_restart_addressable() -> None:
    h = "0" * 64
    plan = MixturePlan(
        plan_id="learn04-test",
        tokenizer_config_sha256=h,
        tokenizer_vocab_sha256="1" * 64,
        packing_config_sha256="2" * 64,
        sources=(
            MixtureSource("uk", "3" * 64, 45),
            MixtureSource("en", "4" * 64, 35),
            MixtureSource("code", "5" * 64, 20),
        ),
        seed=126,
        num_shards=1,
    )
    schedule = []
    counts = {"uk": 0, "en": 0, "code": 0}
    for index in range(32):
        source = plan.source_for_sample(index)
        schedule.append((source, counts[source]))
        counts[source] += 1
    streams = {
        "uk": tuple(range(100, 180)),
        "en": tuple(range(200, 280)),
        "code": tuple(range(300, 380)),
    }
    first = _make_batch(streams, schedule, step=2, batch_size=4, sequence_length=8)
    second = _make_batch(streams, schedule, step=2, batch_size=4, sequence_length=8)
    assert torch.equal(first, second)
    assert tuple(first.shape) == (4, 8)
    assert PACKING_VERSION.endswith("-v1")


def test_matched_family_is_roughly_100k_250k_500k_1m() -> None:
    counts = _EXPECTED_COUNTS_VOCAB472
    targets = (100_000, 250_000, 500_000, 1_000_000)
    relative_errors = [abs(count - target) / target for count, target in zip(counts, targets, strict=True)]
    assert all(error < 0.14 for error in relative_errors)
    assert math.isclose(counts[-1] / 1_000_000, 1.065344)
