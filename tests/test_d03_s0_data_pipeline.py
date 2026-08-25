from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from twelve_six.data import pipeline
from twelve_six.data.ukrainian_normalization import NORMALIZATION_SCHEMA

ROOT = Path(__file__).resolve().parents[1]
SOURCE_REGISTRY = ROOT / "data/s0/source_registry.json"
CONTAMINATION_REGISTRY = ROOT / "data/s0/contamination_registry.json"
PACKAGED = ROOT / "data/s0/packaged"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_current_s0_pipeline_rebuild_is_byte_deterministic(tmp_path: Path) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first = pipeline.build_dataset(SOURCE_REGISTRY, CONTAMINATION_REGISTRY, first_dir)
    second = pipeline.build_dataset(SOURCE_REGISTRY, CONTAMINATION_REGISTRY, second_dir)

    assert first["dataset_identity_sha256"] == second["dataset_identity_sha256"]
    assert first["normalization_schema"] == NORMALIZATION_SCHEMA
    assert first["stats"]["input_documents"] == 12
    assert first["stats"]["train_documents"] == 10
    assert first["stats"]["validation_documents"] == 2
    assert first["stats"]["quality_rejected"] == 0
    assert first["stats"]["contamination_rejected"] == 0
    assert first["stats"]["exact_duplicates_removed"] == 0
    assert first["stats"]["near_duplicates_removed"] == 0
    assert first["stats"]["normalization_changed_documents"] == 0
    assert first["stats"]["normalization_codepoint_delta"] == 0
    assert first["stats"]["normalization_byte_token_delta"] == 0

    for name in ("train.jsonl", "validation.jsonl", "manifest.json"):
        assert (first_dir / name).read_bytes() == (second_dir / name).read_bytes()

    records = [
        json.loads(line)
        for name in ("train.jsonl", "validation.jsonl")
        for line in (first_dir / name).read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 12
    assert all(record["normalization"]["schema"] == NORMALIZATION_SCHEMA for record in records)
    assert all(record["raw_text_sha256"] == record["content_sha256"] for record in records)


def test_legacy_committed_manifest_output_hashes_match_legacy_files() -> None:
    manifest = json.loads((PACKAGED / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["outputs"]["train.jsonl"] == _sha256(PACKAGED / "train.jsonl")
    assert manifest["outputs"]["validation.jsonl"] == _sha256(PACKAGED / "validation.jsonl")
    assert manifest["contamination_state"]["claim"].startswith("controlled S0 sources only")
    assert "normalization_schema" not in manifest


def test_source_registry_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    registry = json.loads(SOURCE_REGISTRY.read_text(encoding="utf-8"))
    registry["sources"][0]["content_sha256"] = "0" * 64
    registry_dir = tmp_path / "data/s0"
    raw_dir = registry_dir / "raw"
    raw_dir.mkdir(parents=True)
    (raw_dir / "project_authored.jsonl").write_bytes(
        (ROOT / "data/s0/raw/project_authored.jsonl").read_bytes()
    )
    bad_registry = registry_dir / "source_registry.json"
    bad_registry.write_text(json.dumps(registry), encoding="utf-8")

    with pytest.raises(pipeline.DataContractError, match="immutable source hash mismatch"):
        pipeline.build_dataset(bad_registry, CONTAMINATION_REGISTRY, tmp_path / "out")


def test_synthetic_source_requires_explicit_kind(tmp_path: Path) -> None:
    registry = json.loads(SOURCE_REGISTRY.read_text(encoding="utf-8"))
    registry["sources"][0]["provenance"].pop("synthetic_kind")
    bad_registry = tmp_path / "source_registry.json"
    bad_registry.write_text(json.dumps(registry), encoding="utf-8")

    with pytest.raises(pipeline.DataContractError, match="synthetic_kind"):
        pipeline.build_dataset(bad_registry, CONTAMINATION_REGISTRY, tmp_path / "out")


def test_benchmark_source_purpose_is_rejected(tmp_path: Path) -> None:
    registry = json.loads(SOURCE_REGISTRY.read_text(encoding="utf-8"))
    registry["sources"][0]["purpose"] = "benchmark"
    bad_registry = tmp_path / "source_registry.json"
    bad_registry.write_text(json.dumps(registry), encoding="utf-8")

    with pytest.raises(pipeline.DataContractError, match="benchmark/evaluation"):
        pipeline.build_dataset(bad_registry, CONTAMINATION_REGISTRY, tmp_path / "out")


def test_contamination_hash_is_filtered(tmp_path: Path) -> None:
    source_dir = tmp_path / "data/s0/raw"
    source_dir.mkdir(parents=True)
    sentinel = "BENCHMARK SENTINEL: this text must never enter S0 pretraining."
    safe = (
        "This sufficiently long safe record exists so the remaining dataset still contains "
        "more than one accepted document for deterministic splitting."
    )
    records = [
        {"document_id": "sentinel", "language_hint": "en", "text": sentinel},
        {"document_id": "safe-1", "language_hint": "en", "text": safe},
        {"document_id": "safe-2", "language_hint": "en", "text": safe + " Another sentence."},
    ]
    raw = "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n" for record in records
    ).encode("utf-8")
    raw_path = source_dir / "fixture.jsonl"
    raw_path.write_bytes(raw)

    registry = json.loads(SOURCE_REGISTRY.read_text(encoding="utf-8"))
    registry["sources"][0]["raw_path"] = "raw/fixture.jsonl"
    registry["sources"][0]["content_sha256"] = hashlib.sha256(raw).hexdigest()
    registry_path = tmp_path / "data/s0/source_registry.json"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps(registry), encoding="utf-8")

    manifest = pipeline.build_dataset(registry_path, CONTAMINATION_REGISTRY, tmp_path / "out")
    assert manifest["stats"]["contamination_rejected"] == 1
    assert manifest["stats"]["accepted_documents"] == 2


def test_pii_email_is_filtered(tmp_path: Path) -> None:
    source_dir = tmp_path / "data/s0/raw"
    source_dir.mkdir(parents=True)
    records = [
        {
            "document_id": "pii",
            "language_hint": "en",
            "text": (
                "This deliberately long record contains contact me@example.org and must be "
                "rejected by the S0 privacy hook before packaging."
            ),
        },
        {
            "document_id": "safe-1",
            "language_hint": "en",
            "text": (
                "A clean training record without contact details remains long enough to pass "
                "the deterministic quality filter."
            ),
        },
        {
            "document_id": "safe-2",
            "language_hint": "en",
            "text": (
                "A second clean training record keeps the tiny dataset splittable after the "
                "privacy filter removes the rejected record."
            ),
        },
    ]
    raw = "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n" for record in records
    ).encode("utf-8")
    raw_path = source_dir / "fixture.jsonl"
    raw_path.write_bytes(raw)

    registry = json.loads(SOURCE_REGISTRY.read_text(encoding="utf-8"))
    registry["sources"][0]["raw_path"] = "raw/fixture.jsonl"
    registry["sources"][0]["content_sha256"] = hashlib.sha256(raw).hexdigest()
    registry_path = tmp_path / "data/s0/source_registry.json"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps(registry), encoding="utf-8")

    manifest = pipeline.build_dataset(registry_path, CONTAMINATION_REGISTRY, tmp_path / "out")
    assert manifest["stats"]["quality_rejected"] == 1
    assert manifest["stats"]["accepted_documents"] == 2


def test_exact_and_near_dedup_stats(tmp_path: Path) -> None:
    source_dir = tmp_path / "data/s0/raw"
    source_dir.mkdir(parents=True)
    base = (
        "Deterministic duplicate handling should remove repeated normalized content while "
        "keeping a distinct training example for the tiny model."
    )
    near = base + " Extra trailing words."
    distinct = (
        "Completely different content describes provenance manifests and deterministic split "
        "assignment so it should remain in the packaged corpus."
    )
    records = [
        {"document_id": "a", "language_hint": "en", "text": base},
        {"document_id": "b", "language_hint": "en", "text": base},
        {"document_id": "c", "language_hint": "en", "text": near},
        {"document_id": "d", "language_hint": "en", "text": distinct},
    ]
    raw = "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n" for record in records
    ).encode("utf-8")
    raw_path = source_dir / "fixture.jsonl"
    raw_path.write_bytes(raw)

    registry = json.loads(SOURCE_REGISTRY.read_text(encoding="utf-8"))
    registry["sources"][0]["raw_path"] = "raw/fixture.jsonl"
    registry["sources"][0]["content_sha256"] = hashlib.sha256(raw).hexdigest()
    registry_path = tmp_path / "data/s0/source_registry.json"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps(registry), encoding="utf-8")

    config = pipeline.PipelineConfig(near_duplicate_threshold=0.70)
    manifest = pipeline.build_dataset(
        registry_path, CONTAMINATION_REGISTRY, tmp_path / "out", config=config
    )
    assert manifest["stats"]["exact_duplicates_removed"] == 1
    assert manifest["stats"]["near_duplicates_removed"] == 1
    assert manifest["stats"]["accepted_documents"] == 2


def test_language_id_covers_s0_languages() -> None:
    assert pipeline.language_id("This English sentence contains enough alphabetic characters.") == "en"
    assert pipeline.language_id(
        "Це українське речення містить достатньо літер для визначення мови."
    ) == "uk"
