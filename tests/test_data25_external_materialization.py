from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from twelve_six.data import corpus_v01
from twelve_six.data.external_sources import (
    ExternalSourceContractError,
    iter_materialized_records,
    validate_external_source_registry,
)

ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "configs" / "data" / "corpus_v01.json"


def _jsonl_bytes(rows: list[dict[str, str]]) -> bytes:
    return b"".join(
        (
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode("utf-8")
        for row in rows
    )


def _source(
    *,
    source_id: str,
    stratum: str,
    path: str,
    digest: str,
    eligible: bool = True,
) -> dict[str, object]:
    return {
        "source_id": source_id,
        "source_version": "2026-08-26",
        "stratum": stratum,
        "training_eligible": eligible,
        "license": {
            "identifier": "CC-BY-4.0",
            "review_status": "approved",
        },
        "provenance": {
            "external_source": True,
            "benchmark_material": False,
            "source_url": f"https://example.invalid/{source_id}",
        },
        "rights": {
            "training_use": "allowed",
            "review_status": "approved",
            "reviewed_by": "test-fixture-reviewer",
        },
        "materialization": {
            "format": "jsonl",
            "path": path,
            "sha256": digest,
            "text_field": "text",
            "document_id_field": "document_id",
        },
    }


def _registry(sources: list[dict[str, object]]) -> dict[str, object]:
    core = {
        "schema_version": corpus_v01.EXTERNAL_SCHEMA,
        "sources": sources,
    }
    return {
        **core,
        "registry_identity_sha256": corpus_v01.sha(corpus_v01.cjson(core)),
    }


def _reserved_registry() -> dict[str, object]:
    core = {
        "schema_version": corpus_v01.RESERVED_SCHEMA,
        "sets": [],
    }
    return {
        **core,
        "registry_identity_sha256": corpus_v01.sha(corpus_v01.cjson(core)),
    }


def _text(stratum: str, n: int) -> str:
    if stratum == "uk":
        return (
            f"Український зовнішній навчальний запис {n} описує мовну модель, корпус, "
            "токенізацію, перевірку якості, походження даних та відтворюваний процес "
            "підготовки навчального матеріалу для контрольованого експерименту."
        )
    if stratum == "en":
        return (
            f"External English training record {n} describes language models, corpus provenance, "
            "tokenization, quality checks, deterministic preparation, and reproducible evidence "
            "for a controlled training experiment."
        )
    return (
        f"def external_record_{n}(values: list[int]) -> list[int]:\n"
        "    result = [value + 1 for value in values]\n"
        "    return result\n"
    )


def _external_repo(tmp_path: Path) -> tuple[Path, list[dict[str, object]]]:
    repo = tmp_path / "repo"
    materialized = repo / "data" / "external" / "materialized"
    materialized.mkdir(parents=True)
    sources: list[dict[str, object]] = []
    for stratum in ("uk", "en", "code"):
        rows = [
            {"document_id": f"{stratum}-{n}", "text": _text(stratum, n)}
            for n in range(6)
        ]
        payload = _jsonl_bytes(rows)
        path = materialized / f"{stratum}.jsonl"
        path.write_bytes(payload)
        sources.append(
            _source(
                source_id=f"fixture-{stratum}",
                stratum=stratum,
                path=f"data/external/materialized/{stratum}.jsonl",
                digest=hashlib.sha256(payload).hexdigest(),
            )
        )
    return repo, sources


def test_registry_requires_explicit_rights_for_training() -> None:
    source = _source(
        source_id="fixture-en",
        stratum="en",
        path="data/external/materialized/en.jsonl",
        digest="0" * 64,
    )
    source["rights"] = {
        "training_use": "unknown",
        "review_status": "approved",
        "reviewed_by": "reviewer",
    }
    parsed = validate_external_source_registry(_registry([source]))[0]
    with pytest.raises(ExternalSourceContractError, match="training_use must be allowed"):
        parsed.assert_training_eligible()


def test_materialized_source_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    repo, sources = _external_repo(tmp_path)
    source = dict(sources[0])
    materialization = dict(source["materialization"])
    materialization["sha256"] = "0" * 64
    source["materialization"] = materialization
    with pytest.raises(ExternalSourceContractError, match="artifact hash mismatch"):
        list(iter_materialized_records(repo, source))


def test_materialized_source_cannot_escape_repository(tmp_path: Path) -> None:
    repo, sources = _external_repo(tmp_path)
    source = dict(sources[0])
    materialization = dict(source["materialization"])
    materialization["path"] = "../outside.jsonl"
    source["materialization"] = materialization
    with pytest.raises(ExternalSourceContractError, match="escapes repository"):
        list(iter_materialized_records(repo, source))


def test_corpus_v01_can_build_from_hash_bound_external_sources(tmp_path: Path) -> None:
    repo, sources = _external_repo(tmp_path)
    external_dir = repo / "data" / "external"
    (external_dir / "external_sources.json").write_text(
        json.dumps(_registry(sources), ensure_ascii=False), encoding="utf-8"
    )
    (external_dir / "reserved_fingerprints.json").write_text(
        json.dumps(_reserved_registry(), ensure_ascii=False), encoding="utf-8"
    )

    config = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    config["target_train_byte_tokens"] = {"uk": 100, "en": 100, "code": 100}
    config["validation_basis_points"] = 1
    config_path = repo / "configs" / "data" / "corpus.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")

    manifest = corpus_v01.build_corpus(config_path, repo / "out")
    assert manifest["external_training_eligible_sources"] == 3
    assert manifest["truth_boundary"]["contains_external_training_data"] is True
    assert manifest["truth_boundary"]["contains_project_authored_data"] is False
    assert manifest["truth_boundary"]["external_source_diversity_representative"] is False
    assert len(manifest["external_source_contracts"]) == 3
    assert set(manifest["by_stratum"]) == {"uk", "en", "code"}
    assert all(item["external"] for item in _read_shards(repo / "out", manifest))


def _read_shards(output: Path, manifest: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for shard in manifest["shards"]:
        path = output / shard["path"]
        for line in path.read_text(encoding="utf-8").splitlines():
            rows.append(json.loads(line))
    return rows
