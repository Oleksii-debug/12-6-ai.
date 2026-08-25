from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from twelve_six.cloze import conditional_log_likelihood, text_log_likelihood
from twelve_six.en_raw_diagnostic import (
    PHENOMENA,
    RESERVED_INDEX_SHA256,
    RESERVATION_SHA256,
    SUITE_ID,
    SUITE_SHA256,
    benchmark_spec,
    load_suite,
    validate_reservation,
)
from twelve_six.eval_reservations import (
    assert_training_text_not_reserved,
    canonical_json_sha256,
    load_reserved_index,
    training_text_collisions,
)
from twelve_six.evaluation import BenchmarkRegistry
from twelve_six.scaling_experiment import _read_jsonl
from twelve_six.tokenization import ByteTokenizer

ROOT = Path(__file__).resolve().parents[1]


class UniformByteModel:
    def __init__(self) -> None:
        self.spec = SimpleNamespace(vocab_size=256, max_seq_len=64)
        self.training = True

    def eval(self):
        self.training = False
        return self

    def train(self, mode: bool = True):
        self.training = mode
        return self

    def __call__(self, input_ids: torch.Tensor):
        shape = (input_ids.shape[0], input_ids.shape[1], 256)
        return SimpleNamespace(logits=torch.zeros(shape, dtype=torch.float32))


def test_first_party_cloze_uses_raw_conditional_likelihood_and_restores_mode() -> None:
    tokenizer = ByteTokenizer()
    model = UniformByteModel()
    score = conditional_log_likelihood(model, tokenizer, "ab", " cd")
    assert model.training is True
    assert score.target_tokens == 3
    assert score.target_utf8_bytes == 3
    assert score.log_likelihood == pytest.approx(-3.0 * math.log(256.0), abs=1e-6)
    assert score.mean_log_likelihood_per_token == pytest.approx(-math.log(256.0), abs=1e-6)
    assert score.bits_per_byte == pytest.approx(8.0, abs=1e-6)

    text = text_log_likelihood(model, tokenizer, "abcd", require_byte_tokenizer=True)
    assert model.training is True
    assert text.scored_tokens == 3
    assert text.scored_utf8_bytes == 3
    assert text.bits_per_scored_byte == pytest.approx(8.0, abs=1e-6)


def test_suite_is_immutable_balanced_and_length_diagnostic() -> None:
    items = load_suite(ROOT)
    assert len(items) == 32
    assert {item["phenomenon"] for item in items} == set(PHENOMENA)
    counts = {
        phenomenon: sum(item["phenomenon"] == phenomenon for item in items)
        for phenomenon in PHENOMENA
    }
    assert set(counts.values()) == {4}

    length_strata = {"preferred_shorter": 0, "equal_length": 0, "preferred_longer": 0}
    tokenizer = ByteTokenizer()
    for item in items:
        preferred = tokenizer.encode(item["preferred"])
        dispreferred = tokenizer.encode(item["dispreferred"])
        if len(preferred) < len(dispreferred):
            length_strata["preferred_shorter"] += 1
        elif len(preferred) > len(dispreferred):
            length_strata["preferred_longer"] += 1
        else:
            length_strata["equal_length"] += 1
        assert tokenizer.encode(item["context"] + item["preferred"]) == (
            tokenizer.encode(item["context"]) + preferred
        )
        assert len(tokenizer.encode(item["context"] + item["preferred"])) <= 256
        assert len(tokenizer.encode(item["context"] + item["dispreferred"])) <= 256

    assert length_strata == {
        "preferred_shorter": 11,
        "equal_length": 10,
        "preferred_longer": 11,
    }


def test_reserved_registry_binds_every_item_and_d06_training_collision() -> None:
    items = load_suite(ROOT)
    reservation = validate_reservation(ROOT, items)
    assert reservation["reservation_sha256"] == RESERVATION_SHA256
    assert reservation["reserved_index_sha256"] == RESERVED_INDEX_SHA256

    index = load_reserved_index(ROOT)
    assert index["registry_sha256"] == RESERVED_INDEX_SHA256
    unsigned = {key: value for key, value in index.items() if key != "registry_sha256"}
    assert canonical_json_sha256(unsigned) == RESERVED_INDEX_SHA256

    spec = benchmark_spec()
    assert spec.benchmark_id == SUITE_ID
    assert spec.source_id == f"reserved-eval:{SUITE_SHA256}"
    assert spec.held_out is True
    registry = BenchmarkRegistry([spec])
    assert registry.training_collisions([spec.source_id]) == [
        {
            "benchmark_key": f"{SUITE_ID}@{spec.version}",
            "source_id": spec.source_id,
        }
    ]


def test_current_s0_material_does_not_contain_reserved_eval_text() -> None:
    train = _read_jsonl(ROOT / "data/s0/packaged/train.jsonl")
    validation = _read_jsonl(ROOT / "data/s0/packaged/validation.jsonl")
    train_texts = [str(record["text"]) for record in train]
    validation_texts = [str(record["text"]) for record in validation]
    assert_training_text_not_reserved(ROOT, train_texts)
    assert_training_text_not_reserved(ROOT, validation_texts)
    assert training_text_collisions(ROOT, train_texts + validation_texts) == []


def test_reserved_collision_check_fails_closed_on_suite_material() -> None:
    item = load_suite(ROOT)[0]
    leaked = "prefix " + item["context"] + item["preferred"] + " suffix"
    collisions = training_text_collisions(ROOT, [leaked])
    assert collisions and collisions[0]["kind"] == "reserved_substring"


def test_reservation_item_fingerprints_cover_suite_ids() -> None:
    items = load_suite(ROOT)
    reservation_path = (
        ROOT / "data/evaluation/reserved/eval133_en_raw_v1.reservation.json"
    )
    reservation = json.loads(reservation_path.read_text(encoding="utf-8"))
    assert reservation["suite_sha256"] == SUITE_SHA256
    assert reservation["reservation_sha256"] == RESERVATION_SHA256
    assert [entry["id"] for entry in reservation["items"]] == [item["id"] for item in items]
    for entry in reservation["items"]:
        for key, value in entry.items():
            if key.endswith("_sha256"):
                assert isinstance(value, str) and len(value) == 64
