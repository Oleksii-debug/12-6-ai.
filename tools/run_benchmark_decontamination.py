"""Execute DATA-31 decontamination on a frozen candidate/reference bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

from twelve_six.data.benchmark_decontamination import (
    ReferenceRecord,
    build_corpus_publication_manifest,
    build_decontamination_report,
    build_reference_bundle,
    exact_matches,
    near_match_records,
    run_datatrove_reference_filter,
)
from twelve_six.data.dedup_scale import (
    DataTroveMinhashExecutionPlan,
    run_datatrove_reference_index,
    validate_datatrove_runtime,
)
from twelve_six.data.external_sources import ReservedSetSpec, build_reserved_fingerprint_registry

RUN_SCHEMA = "12-6.data31-decontamination-run.v1"
CONFIG_SCHEMA = "12-6.data31-decontamination-config.v1"


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    paths = [path] if path.is_file() else sorted(path.rglob("*.jsonl"))
    rows: list[dict[str, Any]] = []
    for current in paths:
        for line in current.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def _require_config(config: Mapping[str, Any]) -> dict[str, Any]:
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("unsupported DATA-31 config schema")
    required = (
        "benchmark_registry_snapshot",
        "candidate",
        "references",
        "rights_evidence",
        "known_semantic_exclusions",
        "minhash",
    )
    for field in required:
        if field not in config:
            raise ValueError(f"missing config field: {field}")
    return dict(config)


def _verify_project_authored_local_check(
    *,
    source_registry: Mapping[str, Any],
    references: list[dict[str, Any]],
) -> None:
    sources = source_registry.get("sources")
    if not isinstance(sources, list):
        raise ValueError("source registry sources must be an array")
    by_id = {item.get("source_id"): item for item in sources if isinstance(item, Mapping)}
    for record in references:
        source_id = record.get("source_id")
        source = by_id.get(source_id)
        if not isinstance(source, Mapping):
            raise ValueError(f"reference source missing from source registry: {source_id}")
        provenance = source.get("provenance")
        if not isinstance(provenance, Mapping):
            raise ValueError(f"reference source lacks provenance: {source_id}")
        if provenance.get("external_source") is not False:
            raise ValueError(f"reference source is not proven project-controlled: {source_id}")
        if provenance.get("origin_type") != "project_authored_fixture":
            raise ValueError(f"reference source is not project-authored fixture: {source_id}")


def _reference_records(records: list[dict[str, Any]], evidence_ref: str) -> list[ReferenceRecord]:
    return [
        ReferenceRecord(
            source_id=str(item["source_id"]),
            document_id=str(item["id"]),
            content_sha256=str(item["content_sha256"]),
            category="heldout_validation",
            evidence_ref=evidence_ref,
        )
        for item in records
    ]


def _semantic_rows(
    exclusions: list[Mapping[str, Any]],
    *,
    candidates: list[dict[str, Any]],
    references: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    candidates_by_id = {str(item["id"]): item for item in candidates}
    references_by_id = {str(item["id"]): item for item in references}
    rows: list[dict[str, Any]] = []
    for exclusion in exclusions:
        candidate_id = str(exclusion["candidate_document_id"])
        reference_id = str(exclusion["reference_document_id"])
        candidate = candidates_by_id.get(candidate_id)
        reference = references_by_id.get(reference_id)
        if candidate is None or reference is None:
            raise ValueError("registered semantic exclusion identity not present in frozen inputs")
        rows.append(
            {
                "match_type": str(exclusion["match_type"]),
                "candidate_source_id": candidate["source_id"],
                "candidate_document_id": candidate_id,
                "candidate_content_sha256": candidate["content_sha256"],
                "reference_source_id": reference["source_id"],
                "reference_document_id": reference_id,
                "reference_content_sha256": reference["content_sha256"],
                "evidence_ref": str(exclusion["evidence_ref"]),
                "decision": "REJECT_FROM_TRAINING",
            }
        )
    return rows


def run(config: Mapping[str, Any], *, repo_root: Path, work_dir: Path) -> dict[str, Any]:
    config = _require_config(config)
    snapshot = config["benchmark_registry_snapshot"]
    d06_manifest = snapshot["manifest"]
    candidate_cfg = config["candidate"]
    reference_cfg = config["references"]
    rights_cfg = config["rights_evidence"]

    candidate_path = repo_root / candidate_cfg["input_jsonl"]
    corpus_manifest_path = repo_root / candidate_cfg["corpus_manifest"]
    source_registry_path = repo_root / rights_cfg["source_registry"]
    reference_path = repo_root / reference_cfg["input_jsonl"]

    candidates = _read_jsonl(candidate_path)
    references = _read_jsonl(reference_path)
    source_registry = json.loads(source_registry_path.read_text(encoding="utf-8"))
    _verify_project_authored_local_check(source_registry=source_registry, references=references)

    refs = _reference_records(references, rights_cfg["source_registry"])
    bundle = build_reference_bundle(
        benchmark_registry=d06_manifest,
        references=refs,
        rights_evidence_refs=[rights_cfg["source_registry"], snapshot["authority_evidence_ref"]],
        unavailable_probe_count=int(reference_cfg["unavailable_probe_count"]),
    )

    reserved = build_reserved_fingerprint_registry(
        [
            ReservedSetSpec(
                set_id="d03:s0-heldout-validation",
                version=str(reference_cfg["version"]),
                source_id=str(reference_cfg["source_id"]),
                purpose="evaluation",
                normalized_sha256=tuple(sorted(item.content_sha256 for item in refs)),
            )
        ]
    )
    corpus_manifest_sha = _file_sha(corpus_manifest_path)
    exact_rows = exact_matches(candidates, refs)
    exact_ids = {item["candidate_document_id"] for item in exact_rows}
    nonexact = [item for item in candidates if str(item["id"]) not in exact_ids]
    nonexact_path = work_dir / "candidate_nonexact" / "00000.jsonl"
    _write_jsonl(nonexact_path, nonexact)

    minhash = config["minhash"]
    plan = DataTroveMinhashExecutionPlan(
        source_registry_sha256=_file_sha(source_registry_path),
        reserved_registry_sha256=reserved["registry_identity_sha256"],
        input_manifest_sha256=corpus_manifest_sha,
        workspace_uri=work_dir.resolve().as_uri(),
        candidate_shards=int(minhash["candidate_shards"]),
        workers=int(minhash["workers"]),
        n_grams=int(minhash["n_grams"]),
        num_buckets=int(minhash["num_buckets"]),
        hashes_per_bucket=int(minhash["hashes_per_bucket"]),
        minhash_seed=int(minhash["seed"]),
        hash_precision=int(minhash["hash_precision"]),
    )
    runtime = validate_datatrove_runtime(plan)
    reference_index = run_datatrove_reference_index(
        plan,
        reference_input=reference_path,
        workspace=work_dir / "datatrove",
        index_name="data31-s0-heldout-validation",
    )
    near_result = run_datatrove_reference_filter(
        plan,
        candidate_input=nonexact_path.parent,
        workspace=work_dir / "datatrove",
        reference_index=Path(reference_index["reference_index"]),
    )
    near_removed = _read_jsonl(Path(near_result["removed"]))
    near_rows = near_match_records(
        near_removed,
        reference_bundle_sha256=bundle["reference_bundle_sha256"],
    )
    near_ids = {item["candidate_document_id"] for item in near_rows}

    semantic_rows = _semantic_rows(
        list(config["known_semantic_exclusions"]),
        candidates=candidates,
        references=references,
    )
    semantic_ids = {item["candidate_document_id"] for item in semantic_rows}
    rejected_ids = exact_ids | near_ids | semantic_ids
    survivors = [item for item in candidates if str(item["id"]) not in rejected_ids]
    final_path = work_dir / "published_candidate" / "train.jsonl"
    _write_jsonl(final_path, survivors)

    report = build_decontamination_report(
        benchmark_registry_sha256=d06_manifest["manifest_sha256"],
        reference_bundle_sha256=bundle["reference_bundle_sha256"],
        candidate_manifest_sha256=corpus_manifest_sha,
        exact_match_rows=exact_rows,
        near_match_rows=near_rows,
        known_semantic_match_rows=semantic_rows,
    )
    publication = build_corpus_publication_manifest(
        corpus_manifest_sha256=corpus_manifest_sha,
        decontamination_report=report,
        current_benchmark_registry_sha256=d06_manifest["manifest_sha256"],
        output_files={"train.jsonl": _file_sha(final_path)},
    )

    core = {
        "schema_version": RUN_SCHEMA,
        "execution_class": "LOCAL_FREE_CURRENT_S0",
        "paid_cost_usd": 0,
        "d06_authority": {
            "registry_schema": d06_manifest["schema_version"],
            "registry_sha256": d06_manifest["manifest_sha256"],
            "snapshot_state": snapshot["snapshot_state"],
            "authority_evidence_ref": snapshot["authority_evidence_ref"],
        },
        "reference_bundle": bundle,
        "datatrove_plan": plan.manifest(),
        "runtime": runtime,
        "decontamination_report": report,
        "publication_manifest": publication,
        "counts": {
            "candidate_input": len(candidates),
            "heldout_references": len(references),
            "exact_matches": len(exact_rows),
            "near_matches": len(near_rows),
            "registered_semantic_matches": len(semantic_rows),
            "rejected_unique_documents": len(rejected_ids),
            "surviving_documents": len(survivors),
        },
        "surviving_document_ids": [item["id"] for item in survivors],
        "truth_boundary": {
            "real_d06_registered_benchmarks_in_snapshot": len(d06_manifest["benchmarks"]),
            "real_hashed_generation_probes_locally_available": 0,
            "supplemental_heldout_validation_references": len(references),
            "semantic_universal_cleanliness_claimed": False,
            "legal_claim": "No external-dataset license conclusion; local references are project-authored fixture material per recorded provenance.",
        },
    }
    return {**core, "run_sha256": _sha(_canonical(core))}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--work-dir", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.work_dir:
        args.work_dir.mkdir(parents=True, exist_ok=True)
        result = run(config, repo_root=args.repo_root, work_dir=args.work_dir)
    else:
        with tempfile.TemporaryDirectory(prefix="data31-decontam-") as temporary:
            result = run(config, repo_root=args.repo_root, work_dir=Path(temporary))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
