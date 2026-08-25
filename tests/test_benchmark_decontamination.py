from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from twelve_six.data.benchmark_decontamination import (
    ReferenceRecord,
    assert_fresh_decontamination,
    build_corpus_publication_manifest,
    build_decontamination_report,
    build_reference_bundle,
    exact_matches,
    near_match_records,
    run_datatrove_reference_filter,
)
from twelve_six.data.corpus_foundation import CorpusFoundationError
from twelve_six.data.dedup_scale import DataTroveMinhashExecutionPlan, run_datatrove_reference_index

H = "a" * 64
R = "b" * 64
I = "c" * 64
CURRENT_REGISTRY = "10f7454f77eb2dc3871eeafa5055b1969eab42954eb8e19e61565f217c67df31"


def _empty_d06_registry() -> dict:
    return {
        "schema_version": "12-6.benchmark-registry.v1",
        "benchmarks": [],
        "manifest_sha256": CURRENT_REGISTRY,
    }


def _reference(text: str = "reserved validation text") -> ReferenceRecord:
    return ReferenceRecord(
        source_id="project-authored",
        document_id="validation-1",
        content_sha256=hashlib.sha256(text.encode()).hexdigest(),
        category="heldout_validation",
        evidence_ref="data/s0/source_registry.json",
    )


def test_exact_benchmark_copy_records_both_document_ids_and_rejects() -> None:
    reference = _reference()
    candidate = {
        "id": "train-copy",
        "source_id": "candidate-source",
        "content_sha256": reference.content_sha256,
        "text": "reserved validation text",
    }

    rows = exact_matches([candidate], [reference])

    assert rows == [
        {
            "match_type": "exact_content_sha256",
            "candidate_source_id": "candidate-source",
            "candidate_document_id": "train-copy",
            "candidate_content_sha256": reference.content_sha256,
            "reference_source_id": "project-authored",
            "reference_document_id": "validation-1",
            "reference_content_sha256": reference.content_sha256,
            "reference_category": "heldout_validation",
            "decision": "REJECT_FROM_TRAINING",
        }
    ]


def test_registry_identity_change_requires_fresh_pass() -> None:
    reference = _reference()
    bundle = build_reference_bundle(
        benchmark_registry=_empty_d06_registry(),
        references=[reference],
        rights_evidence_refs=["data/s0/source_registry.json"],
    )
    report = build_decontamination_report(
        benchmark_registry_sha256=CURRENT_REGISTRY,
        reference_bundle_sha256=bundle["reference_bundle_sha256"],
        candidate_manifest_sha256=I,
        exact_match_rows=[],
        near_match_rows=[],
        known_semantic_match_rows=[],
    )

    assert_fresh_decontamination(report, current_benchmark_registry_sha256=CURRENT_REGISTRY)
    with pytest.raises(CorpusFoundationError, match="fresh decontamination pass"):
        assert_fresh_decontamination(
            report,
            current_benchmark_registry_sha256="f" * 64,
        )


def test_publication_manifest_binds_benchmark_registry_identity() -> None:
    reference = _reference()
    bundle = build_reference_bundle(
        benchmark_registry=_empty_d06_registry(),
        references=[reference],
        rights_evidence_refs=["data/s0/source_registry.json"],
    )
    report = build_decontamination_report(
        benchmark_registry_sha256=CURRENT_REGISTRY,
        reference_bundle_sha256=bundle["reference_bundle_sha256"],
        candidate_manifest_sha256=I,
        exact_match_rows=[],
        near_match_rows=[],
        known_semantic_match_rows=[],
    )
    manifest = build_corpus_publication_manifest(
        corpus_manifest_sha256=H,
        decontamination_report=report,
        current_benchmark_registry_sha256=CURRENT_REGISTRY,
        output_files={"train.jsonl": R},
    )

    assert manifest["benchmark_registry_sha256"] == CURRENT_REGISTRY
    assert manifest["decontamination_report_sha256"] == report["decontamination_report_sha256"]
    assert manifest["semantic_universal_cleanliness_claimed"] is False


def test_known_semantic_match_is_audited_and_must_have_reject_decision() -> None:
    reference = _reference()
    bundle = build_reference_bundle(
        benchmark_registry=_empty_d06_registry(),
        references=[reference],
        rights_evidence_refs=["data/s0/source_registry.json"],
    )
    semantic = {
        "match_type": "registered_cross_language_semantic_overlap",
        "candidate_source_id": "candidate-source",
        "candidate_document_id": "train-1",
        "reference_source_id": "project-authored",
        "reference_document_id": "validation-1",
        "decision": "REVIEW_REQUIRED",
    }
    report = build_decontamination_report(
        benchmark_registry_sha256=CURRENT_REGISTRY,
        reference_bundle_sha256=bundle["reference_bundle_sha256"],
        candidate_manifest_sha256=I,
        exact_match_rows=[],
        near_match_rows=[],
        known_semantic_match_rows=[semantic],
    )

    assert report["publication_eligible"] is False
    assert report["semantic_universal_cleanliness_claimed"] is False
    with pytest.raises(CorpusFoundationError, match="does not permit corpus publication"):
        build_corpus_publication_manifest(
            corpus_manifest_sha256=H,
            decontamination_report=report,
            current_benchmark_registry_sha256=CURRENT_REGISTRY,
            output_files={"train.jsonl": R},
        )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _read_jsonl(folder: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(folder.rglob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


@pytest.mark.skipif(importlib.util.find_spec("datatrove") is None, reason="DataTrove integration runtime")
def test_datatrove_near_copy_injection_is_removed_against_reference_only_index(tmp_path: Path) -> None:
    common = [f"token_{index:03d}" for index in range(240)]
    reference_text = " ".join(common)
    near = common.copy()
    near[60] = "changed_alpha"
    near[120] = "changed_beta"
    near[180] = "changed_gamma"
    near_text = " ".join(near)

    reference = {
        "id": "benchmark-reference",
        "source_id": "benchmark-source",
        "text": reference_text,
        "content_sha256": hashlib.sha256(reference_text.encode()).hexdigest(),
    }
    candidate = {
        "id": "candidate-near-copy",
        "source_id": "training-source",
        "text": near_text,
        "content_sha256": hashlib.sha256(near_text.encode()).hexdigest(),
    }
    _write_jsonl(tmp_path / "reference" / "00000.jsonl", [reference])
    _write_jsonl(tmp_path / "candidate" / "00000.jsonl", [candidate])

    plan = DataTroveMinhashExecutionPlan(
        source_registry_sha256=H,
        reserved_registry_sha256=R,
        input_manifest_sha256=I,
        workspace_uri=tmp_path.resolve().as_uri(),
        candidate_shards=1,
        workers=1,
    )
    index = run_datatrove_reference_index(
        plan,
        reference_input=tmp_path / "reference",
        workspace=tmp_path / "dt",
        index_name="data31-test-reference",
    )
    result = run_datatrove_reference_filter(
        plan,
        candidate_input=tmp_path / "candidate",
        workspace=tmp_path / "dt",
        reference_index=Path(index["reference_index"]),
    )
    removed = _read_jsonl(Path(result["removed"]))
    rows = near_match_records(removed, reference_bundle_sha256=R)

    assert [item["candidate_document_id"] for item in rows] == ["candidate-near-copy"]
    assert rows[0]["decision"] == "REJECT_FROM_TRAINING"
    assert rows[0]["reference_document_id"] is None
