from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from twelve_six.evaluation import BenchmarkRegistry, BenchmarkSpec
from twelve_six.evaluation_ua_v1 import (
    CHANCE_PAIR_ACCURACY,
    DATASET_SHA256,
    D06_REGISTRY_SHA256,
    PHENOMENA,
    RESERVED_REGISTRY_IDENTITY_SHA256,
    SOURCE_ID,
    SOURCE_IDENTITY_SHA256,
    dataset_sha256,
    evaluate_model,
    generate_items,
    load_source,
    rendered_variants,
    reserved_variant_hashes,
    score_completion,
    validate_reserved_registry,
)

ROOT = Path(__file__).resolve().parents[1]


class UniformByteModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.spec = SimpleNamespace(max_seq_len=256)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch, sequence = input_ids.shape
        return self.anchor * torch.ones((batch, sequence, 256), device=input_ids.device)


def test_source_and_dataset_identities_are_frozen() -> None:
    source = load_source()
    items = generate_items()
    assert source["source_id"] == SOURCE_ID
    assert source["source_identity_sha256"] == SOURCE_IDENTITY_SHA256
    assert len(items) == 216
    assert dataset_sha256(items) == DATASET_SHA256
    assert {phenomenon: sum(item["phenomenon"] == phenomenon for item in items) for phenomenon in PHENOMENA} == {
        phenomenon: 24 for phenomenon in PHENOMENA
    }
    assert len({item["item_id"] for item in items}) == 216
    assert all("instruction" not in item and "messages" not in item for item in items)
    assert max(len(text.encode("utf-8")) for text in rendered_variants(items)) <= 256


def test_every_rendered_variant_is_reserved() -> None:
    registry = validate_reserved_registry()
    assert registry["registry_identity_sha256"] == RESERVED_REGISTRY_IDENTITY_SHA256
    entry = next(item for item in registry["sets"] if item["benchmark_id"] == "eval132-ua-raw-base")
    assert entry["variant_count"] == 432
    assert tuple(entry["normalized_sha256"]) == reserved_variant_hashes()


def test_d06_registry_is_evaluation_only_and_identity_bound() -> None:
    persisted = json.loads((ROOT / "data/evaluation/benchmark_registry.json").read_text(encoding="utf-8"))
    entry = persisted["benchmarks"][0]
    spec = BenchmarkSpec(
        benchmark_id=entry["benchmark_id"],
        version=entry["version"],
        source_id=entry["source_id"],
        held_out=entry["held_out"],
        allowed_uses=tuple(entry["allowed_uses"]),
        license_id=entry["license_id"],
        source_url=entry["source_url"],
        notes=entry["notes"],
    )
    rebuilt = BenchmarkRegistry([spec]).manifest()
    assert rebuilt == persisted
    assert rebuilt["manifest_sha256"] == D06_REGISTRY_SHA256
    assert rebuilt["benchmarks"][0]["allowed_uses"] == ["evaluation"]
    assert rebuilt["benchmarks"][0]["held_out"] is True


def test_apostrophe_pair_is_scored_as_exact_raw_text() -> None:
    item = next(item for item in generate_items() if item["item_id"] == "ua-v1-apostrophe_orthography-001")
    assert item["preferred"] == "пам'ять."
    assert item["contrast"] == "память."
    assert item["preferred"].encode("utf-8") != item["contrast"].encode("utf-8")


def test_uniform_model_has_eight_bpb_and_eval_is_non_mutating() -> None:
    model = UniformByteModel()
    model.train()
    score = score_completion(model, "Я ", "тут.")
    assert score["conditional_bpb"] == pytest.approx(8.0)
    assert score["tokens_per_source_byte"] == 1.0
    optimized_tokens = 123
    report = evaluate_model(
        model,
        label="uniform",
        source={"kind": "test"},
        optimized_tokens_getter=lambda: optimized_tokens,
        include_item_rows=False,
    )
    assert model.training is True
    assert report["state_unchanged"] is True
    assert report["optimized_tokens_delta"] == 0
    assert report["overall"]["accuracy"] == 0.0
    assert report["baseline"]["symmetric_pair_choice_chance_accuracy"] == CHANCE_PAIR_ACCURACY
    assert all(report["by_phenomenon"][phenomenon]["n"] == 24 for phenomenon in PHENOMENA)


def test_manifest_forbids_proficiency_claim_and_qualifies_word_order() -> None:
    manifest = json.loads((ROOT / "data/evaluation/ua_raw_base_v1/manifest.json").read_text(encoding="utf-8"))
    assert manifest["interpretation"]["proficiency_claim_authorized"] is False
    assert "unmarked" in manifest["interpretation"]["common_word_order"]
    assert manifest["task"]["instruction_following"] is False
    assert manifest["task"]["chance_pair_accuracy"] == 0.5
    assert manifest["contamination"]["future_training_allowed"] is False
    assert manifest["contamination"]["semantic_universal_cleanliness_claimed"] is False
