"""DATA-110 Corpus V1 release-candidate and bounded ~1M Base learning proof.

This is orchestration only. It composes incumbent source-intake, quality, privacy,
exact/near-dedup, D06 decontamination, split, packing, Trainer, checkpoint,
observability, evaluation, and first-party inference components without replacing them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import torch
import torch.nn.functional as F

from twelve_six.checkpoint import (
    CheckpointIdentity,
    hash_json,
    load_trainer_checkpoint,
    save_trainer_checkpoint,
    sha256_file,
    verify_checkpoint,
)
from twelve_six.data.benchmark_decontamination import (
    ReferenceRecord,
    build_corpus_publication_manifest,
    build_decontamination_report,
    build_reference_bundle,
    exact_matches,
    near_match_records,
    run_datatrove_reference_filter,
)
from twelve_six.data.corpus_foundation import SQLiteExactDedupIndex
from twelve_six.data.corpus_v01 import cjson, verify_rebuild, write_jsonl
from twelve_six.data.dedup_scale import (
    DataTroveMinhashExecutionPlan,
    build_dedup_output_manifest,
    build_training_eligibility_envelope,
    run_datatrove_candidate_dedup,
    run_datatrove_reference_index,
    validate_datatrove_runtime,
)
from twelve_six.data.document_quality import assess_document, default_quality_policy
from twelve_six.data.privacy_filter import (
    assert_no_secret_values_in_manifest,
    build_scan_manifest,
    privacy_policy_manifest,
    scan_record,
)
from twelve_six.evaluation import perplexity_from_nll, relative_loss_improvement
from twelve_six.inference.contracts import GenerationConfig
from twelve_six.inference.first_party import load_first_party_backend
from twelve_six.inference.generation import generate
from twelve_six.model import InitSpec, ModelSpec, TwelveSixDecoder, load_stage_config
from twelve_six.packing import PACKING_VERSION, TextRecord, iter_packed_examples
from twelve_six.split_robustness import (
    SplitFamilySpec,
    SplitRecord,
    build_split_family,
    dedup_relations_identity,
    eligible_corpus_identity,
    verify_split_family_manifest,
)
from twelve_six.tokenization import ByteTokenizer
from twelve_six.training import Trainer, TrainerConfig
from twelve_six.training.observability import TrainingObserver

SCHEMA = "12-6.data110-corpus-v1-rc-learning.v1"
RELEASE_SCHEMA = "12-6.corpus-v1-release-candidate.v1"
AUTHORITY = "LOCAL_FREE_RELEASE_CANDIDATE_NOT_CORPUS_FREEZE_OR_STAGE_PROMOTION"
REPOSITORY = "Oleksii-debug/12-6-ai."
DATA25_CONFIG = Path("configs/data/corpus_v01.json")
DATA25_RETAINED = Path("data/corpus/v0.1/manifest.json")
DATA25_EXPECTED_ID = "422f545da93526ceba2c0ff738e0b9deac65f35bfb31d87d2aab068cda091fc8"
D06_CONFIG = Path("configs/data/decontamination_current_s0_v1.json")
D06_REFERENCE = Path("data/s0/packaged/validation.jsonl")
D06_SOURCE_REGISTRY = Path("data/s0/source_registry.json")
STAGE_CONFIG = Path("configs/stages/s2_1m.json")
RUNTIME_LOCK = Path("requirements/locks/linux-x86_64/runtime.lock.txt")
TOOLCHAIN_LOCK = Path("requirements/locks/linux-x86_64/toolchain.lock.txt")
SEQ = 128
BATCH = 8
MAX_STEPS = 512
RESUME_STEP = 256
CHECKPOINT_STEPS = {0, 128, 256, 384, 512}
SEED = 1337
LR = 3e-4
TARGET_PROJECT_BYTES = {"uk": 900_000, "en": 700_000, "code": 400_000}
SHARD_TARGET_BYTES = 256 * 1024
MIXTURE = (
    "uk", "en", "uk", "code", "en",
    "uk", "en", "uk", "code", "uk",
    "en", "uk", "en", "code", "uk",
    "en", "uk", "code", "en", "uk",
)
PROMPTS = {"uk": "Українська мова ", "en": "The corpus ", "code": "def stable_"}
UPSTREAM = {
    "product_convergence_base": "fb9c6d9b73ce436d637077892d73edf136fcaeac",
    "data21_22_rights_intake": "dcc7dfc39299487bca5bdbfe5e6c70eaa6706278",
    "data31_decontamination": "6b1c7f3418357e0ea1cfc6ab5ceda6a740dc5921",
    "data32_quality_candidate": "b1c9449ca839ed10c872f444010f74fd225acae1",
    "data33_privacy_candidate": "290b82fd0f7d1cc3a1840deae4378b9c500f1c15",
    "data36_split_candidate": "e78104b86a5d3c7978cc281e81410f8bed470ad1",
}
SELECTIVE_BLOBS = {
    "dedup_scale": "9351ae6f3574e58b15149a4c520f7cbc1fe2ee3e",
    "benchmark_decontamination": "9295d4bc425fc604f828332b4caf38fc9144a7d6",
    "document_quality": "4170645b0f2a991adf0217fc5b3231dddb8ef1c6",
    "privacy_filter": "f4ac642967b367f40664caaecb2c13665073e049",
    "split_robustness": "9377beb3f7206bb2e68e5137b7e995e06f51d77b",
}


class Data110Error(RuntimeError):
    pass


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Data110Error(f"{path} must contain a JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    paths = [path] if path.is_file() else sorted(path.rglob("*.jsonl"))
    rows: list[dict[str, Any]] = []
    for current in paths:
        for raw in current.read_text(encoding="utf-8").splitlines():
            if raw.strip():
                value = json.loads(raw)
                if not isinstance(value, dict):
                    raise Data110Error(f"{current}: JSON object required")
                rows.append(value)
    return rows


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_cjson(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = cjson(value)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _append(path: Path, value: object) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def _sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _require_head(repo: Path, source_sha: str) -> None:
    if len(source_sha) != 40 or any(c not in "0123456789abcdef" for c in source_sha):
        raise Data110Error("source_sha must be lowercase full 40-hex")
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    if actual != source_sha:
        raise Data110Error(f"exact-head mismatch: {actual} != {source_sha}")


def _locks(repo: Path) -> dict[str, Any]:
    files = {p.as_posix(): sha256_file(repo / p) for p in (RUNTIME_LOCK, TOOLCHAIN_LOCK)}
    return {"files": files, "combined_sha256": hash_json(files)}


def _model(repo: Path) -> tuple[ModelSpec, InitSpec, dict[str, Any]]:
    stage = load_stage_config(repo / STAGE_CONFIG)
    payload = stage.model.to_dict()
    if payload["vocab_size"] != 2048:
        raise Data110Error("canonical S2 vocab drifted")
    payload["vocab_size"] = 256
    spec = ModelSpec.from_dict(payload)
    if spec.parameter_count() != 836_736:
        raise Data110Error(f"expected 836,736 parameters, got {spec.parameter_count()}")
    init = InitSpec()
    return spec, init, {
        "source_stage_config": STAGE_CONFIG.as_posix(),
        "source_model_spec_sha256": stage.model.identity_sha256(),
        "source_expected_parameters": stage.expected_parameters,
        "only_geometry_change": "vocab_size:2048->256 to bind canonical s0-byte-v1",
    }


def _trainer_config() -> TrainerConfig:
    return TrainerConfig(
        learning_rate=LR,
        weight_decay=0.0,
        betas=(0.9, 0.95),
        eps=1e-8,
        max_steps=MAX_STEPS,
        warmup_steps=0,
        scheduler="constant",
        gradient_accumulation_steps=1,
        gradient_clip_norm=1.0,
        precision="fp32",
        seed=SEED,
        deterministic_algorithms=True,
        deterministic_warn_only=False,
    )


def _data25_rows(repo: Path, build_dir: Path, manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for shard in manifest["shards"]:
        path = build_dir / str(shard["path"])
        if sha256_file(path) != shard["sha256"]:
            raise Data110Error(f"DATA-25 shard hash mismatch: {path}")
        for row in _read_jsonl(path):
            if row.get("split") == "train" and row.get("stratum") in TARGET_PROJECT_BYTES:
                rows.append(row)
    rows.sort(key=lambda r: (str(r["stratum"]), str(r["record_id"])))
    return rows


def _select_project_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    totals = Counter()
    selected: list[dict[str, Any]] = []
    for stratum in ("uk", "en", "code"):
        for row in rows:
            if row["stratum"] != stratum or totals[stratum] >= TARGET_PROJECT_BYTES[stratum]:
                continue
            text = str(row["text"])
            selected.append({
                "id": str(row["record_id"]),
                "source_id": str(row["source_id"]),
                "source_version": str(row["source_version"]),
                "stratum": stratum,
                "modality": "code" if stratum == "code" else "natural",
                "origin": "project_authored",
                "external": False,
                "project_authored": True,
                "content_sha256": _sha_text(text),
                "text": text,
                "upstream_split": "train",
            })
            totals[stratum] += len(text.encode("utf-8"))
        if totals[stratum] < TARGET_PROJECT_BYTES[stratum]:
            raise Data110Error(f"DATA-25 train rows did not reach {stratum} target")
    return selected


def _load_external(intake_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_path = intake_dir / "manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != "12-6.external-source-intake-manifest.v1":
        raise Data110Error("external intake schema mismatch")
    if manifest.get("authority_boundary") != "REAL_BOUNDED_SAMPLE_NOT_CANONICAL_CORPUS_FREEZE_OR_SOURCE_SNAPSHOT_PROMOTION":
        raise Data110Error("external intake authority boundary drift")
    records = manifest.get("records")
    if not isinstance(records, list) or len(records) < 2:
        raise Data110Error("real external intake is missing")
    result: list[dict[str, Any]] = []
    for raw in records:
        if not isinstance(raw, Mapping) or raw.get("status") != "ACCEPTED":
            continue
        if raw.get("allows_model_training") is not True or raw.get("rights_status") != "APPROVED_FOR_TRAINING":
            raise Data110Error("admitted external source lacks explicit training eligibility")
        source_identity = str(raw.get("source_identity_sha256", ""))
        raw_sha = str(raw.get("raw_sha256", ""))
        content_sha = str(raw.get("content_sha256", ""))
        if any(len(x) != 64 for x in (source_identity, raw_sha, content_sha)):
            raise Data110Error("external provenance hash missing")
        text_path = intake_dir / str(raw["text_path"])
        text = text_path.read_text(encoding="utf-8")
        if _sha_text(text) != content_sha:
            raise Data110Error(f"external text hash mismatch: {raw['id']}")
        language = str(raw["language"])
        if language not in {"uk", "en"}:
            raise Data110Error("unexpected external language")
        result.append({
            "id": str(raw["id"]),
            "source_id": str(raw["source_id"]),
            "source_version": str(raw["source_version"]),
            "stratum": language,
            "modality": "natural",
            "origin": "external_real",
            "external": True,
            "project_authored": False,
            "content_sha256": content_sha,
            "raw_sha256": raw_sha,
            "source_identity_sha256": source_identity,
            "license_id": str(raw["license_id"]),
            "rights_status": str(raw["rights_status"]),
            "allows_model_training": True,
            "acquisition_url": str(raw["acquisition_url"]),
            "text": text,
        })
    if {r["stratum"] for r in result} != {"uk", "en"}:
        raise Data110Error("accepted real bytes must include both UK and EN")
    manifest["physical_manifest_sha256"] = sha256_file(manifest_path)
    return sorted(result, key=lambda r: r["id"]), manifest


def _source_registry(external: Sequence[Mapping[str, Any]], project: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    sources: dict[str, dict[str, Any]] = {}
    for row in external:
        sid = str(row["source_id"])
        sources[sid] = {
            "source_id": sid,
            "source_version": str(row["source_version"]),
            "origin": "external_real",
            "allows_model_training": True,
            "rights_status": str(row["rights_status"]),
            "license_id": str(row["license_id"]),
            "source_identity_sha256": str(row["source_identity_sha256"]),
            "immutable_content_evidence": {"raw_sha256": str(row["raw_sha256"]), "normalized_sha256": str(row["content_sha256"])},
        }
    for row in project:
        sid = str(row["source_id"])
        sources.setdefault(sid, {
            "source_id": sid,
            "source_version": str(row["source_version"]),
            "origin": "project_authored",
            "allows_model_training": True,
            "rights_status": "PROJECT_CONTROLLED",
            "license_id": "PROJECT_AUTHORED",
        })
    core = {"schema_version": "12-6.data110-source-registry.v1", "sources": [sources[k] for k in sorted(sources)]}
    return {**core, "registry_identity_sha256": hash_json(core)}


def _policy_gate(rows: Sequence[Mapping[str, Any]], work: Path, source_registry_sha: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    work.mkdir(parents=True, exist_ok=True)
    quality_policy = default_quality_policy()
    quality_manifest = quality_policy.manifest()
    quality_path = work / "quality-decisions.jsonl"
    privacy_results = []
    accepted: list[dict[str, Any]] = []
    exact_rejected = 0
    quality_rejected = 0
    privacy_rejected = 0
    if quality_path.exists():
        quality_path.unlink()
    db = work / "exact-dedup.sqlite3"
    if db.exists():
        db.unlink()
    with SQLiteExactDedupIndex(db) as exact:
        for input_row in sorted(rows, key=lambda r: str(r["id"])):
            row = dict(input_row)
            mode = str(row["stratum"])
            q = assess_document(str(row["id"]), str(row["text"]), mode, policy=quality_policy)
            _append(quality_path, q.manifest())
            if not q.accepted:
                quality_rejected += 1
                continue
            p = scan_record(
                record_id=str(row["id"]),
                source_id=str(row["source_id"]),
                source_version=str(row["source_version"]),
                modality=str(row["modality"]),
                text=str(row["text"]),
            )
            privacy_results.append(p)
            if not p.train_eligible_after_privacy or p.sanitized_text is None:
                privacy_rejected += 1
                continue
            row["text"] = p.sanitized_text
            row["content_sha256"] = _sha_text(p.sanitized_text)
            row["privacy_action"] = p.action
            if exact.seen_or_add(row["content_sha256"]):
                exact_rejected += 1
                continue
            accepted.append(row)
    privacy_manifest = build_scan_manifest(
        privacy_results,
        input_content_sha256=hash_json([str(r["content_sha256"]) for r in sorted(rows, key=lambda r: str(r["id"]))]),
        source_registry_sha256=source_registry_sha,
    )
    assert_no_secret_values_in_manifest(privacy_manifest)
    _write_json(work / "privacy-manifest.json", privacy_manifest)
    evidence = {
        "quality_policy_sha256": quality_manifest["policy_sha256"],
        "quality_decisions_sha256": sha256_file(quality_path),
        "privacy_policy_sha256": privacy_policy_manifest()["policy_sha256"],
        "privacy_manifest_sha256": privacy_manifest["manifest_sha256"],
        "input_records": len(rows),
        "quality_rejected": quality_rejected,
        "privacy_rejected_or_review": privacy_rejected,
        "exact_duplicate_rejected": exact_rejected,
        "records_after_policy_and_exact_dedup": len(accepted),
    }
    return accepted, evidence


def _datatrove_row(row: Mapping[str, Any]) -> dict[str, Any]:
    meta = {k: v for k, v in row.items() if k != "text"}
    return {
        "id": str(row["id"]),
        "text": str(row["text"]),
        "source_id": str(row["source_id"]),
        "content_sha256": str(row["content_sha256"]),
        "metadata": meta,
    }


def _flatten_datatrove(row: Mapping[str, Any]) -> dict[str, Any]:
    meta = row.get("metadata")
    merged: dict[str, Any] = dict(meta) if isinstance(meta, Mapping) else {}
    for key, value in row.items():
        if key != "metadata":
            merged[key] = value
    if "id" not in merged or "text" not in merged:
        raise Data110Error("DataTrove output lost id/text")
    merged.setdefault("content_sha256", _sha_text(str(merged["text"])))
    merged.setdefault("source_id", "UNKNOWN")
    return merged


def _d06(repo: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[ReferenceRecord], dict[str, Any]]:
    config = _read_json(repo / D06_CONFIG)
    snapshot = config["benchmark_registry_snapshot"]
    manifest = dict(snapshot["manifest"])
    if manifest.get("manifest_sha256") != "10f7454f77eb2dc3871eeafa5055b1969eab42954eb8e19e61565f217c67df31":
        raise Data110Error("D06 authority identity drifted")
    refs_raw = _read_jsonl(repo / D06_REFERENCE)
    refs = [
        ReferenceRecord(
            source_id=str(r["source_id"]),
            document_id=str(r["id"]),
            content_sha256=str(r["content_sha256"]),
            category="heldout_validation",
            evidence_ref=D06_SOURCE_REGISTRY.as_posix(),
        )
        for r in refs_raw
    ]
    bundle = build_reference_bundle(
        benchmark_registry=manifest,
        references=refs,
        rights_evidence_refs=[D06_SOURCE_REGISTRY.as_posix(), str(snapshot["authority_evidence_ref"])],
        unavailable_probe_count=int(config["references"]["unavailable_probe_count"]),
    )
    return manifest, refs_raw, refs, bundle


def _near_dedup_and_decontam(
    repo: Path,
    rows: Sequence[Mapping[str, Any]],
    work: Path,
    source_registry_sha: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    d06_manifest, refs_raw, refs, bundle = _d06(repo)
    d06_sha = str(d06_manifest["manifest_sha256"])
    candidate_rows = [_datatrove_row(r) for r in rows]
    pre_core = {
        "records": [
            {"id": r["id"], "source_id": r["source_id"], "content_sha256": r["content_sha256"]}
            for r in candidate_rows
        ]
    }
    input_manifest_sha = hash_json(pre_core)
    plan = DataTroveMinhashExecutionPlan(
        source_registry_sha256=source_registry_sha,
        reserved_registry_sha256=d06_sha,
        input_manifest_sha256=input_manifest_sha,
        workspace_uri="file:///data110/corpus-v1-rc-workspace",
        candidate_shards=1,
        workers=1,
        n_grams=5,
        num_buckets=14,
        hashes_per_bucket=8,
        minhash_seed=1,
        hash_precision=64,
        normalize_numbers=False,
        datatrove_version="0.10.0",
    )
    runtime = validate_datatrove_runtime(plan)
    candidate_dir = work / "candidate_input"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(candidate_dir / "00000.jsonl", candidate_rows)
    reference_dir = work / "reference_input"
    reference_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(reference_dir / "00000.jsonl", refs_raw)
    index = run_datatrove_reference_index(
        plan,
        reference_input=reference_dir,
        workspace=work / "reference_index_work",
        index_name="data110-current-d06-plus-s0-heldout",
    )

    exact_rows = exact_matches(candidate_rows, refs)
    exact_ids = {str(x["candidate_document_id"]) for x in exact_rows}
    nonexact = [r for r in candidate_rows if str(r["id"]) not in exact_ids]
    nonexact_dir = work / "candidate_nonexact"
    nonexact_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(nonexact_dir / "00000.jsonl", nonexact)
    ref_result = run_datatrove_reference_filter(
        plan,
        candidate_input=nonexact_dir,
        workspace=work / "reference_filter_work",
        reference_index=Path(index["reference_index"]),
    )
    ref_removed = _read_jsonl(Path(ref_result["removed"]))
    near_rows = near_match_records(ref_removed, reference_bundle_sha256=bundle["reference_bundle_sha256"])
    near_ids = {str(x["candidate_document_id"]) for x in near_rows}
    reference_clean = [r for r in nonexact if str(r["id"]) not in near_ids]
    reference_clean_dir = work / "reference_clean"
    reference_clean_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(reference_clean_dir / "00000.jsonl", reference_clean)

    internal = run_datatrove_candidate_dedup(
        plan,
        candidate_input=reference_clean_dir,
        workspace=work / "internal_dedup_work",
        reference_index=Path(index["reference_index"]),
        exercise_restart=False,
    )
    survivors_raw = _read_jsonl(Path(internal["output"]))
    internal_removed_raw = _read_jsonl(Path(internal["removed"]))
    survivor_ids = {str(_flatten_datatrove(r)["id"]) for r in survivors_raw}
    original_by_id = {str(r["id"]): _flatten_datatrove(r) for r in reference_clean}
    unknown_ids = survivor_ids - set(original_by_id)
    if unknown_ids:
        raise Data110Error(f"DataTrove emitted unknown survivor ids: {sorted(unknown_ids)[:3]}")
    survivors = [original_by_id[record_id] for record_id in sorted(survivor_ids)]
    ids = [str(r["id"]) for r in survivors]
    if len(ids) != len(set(ids)):
        raise Data110Error("near-dedup output contains duplicate ids")
    if set(ids) & (exact_ids | near_ids):
        raise Data110Error("D06-contaminated record survived")
    for row in survivors:
        if _sha_text(str(row["text"])) != str(row["content_sha256"]):
            raise Data110Error("survivor content hash drift")

    decontam = build_decontamination_report(
        benchmark_registry_sha256=d06_sha,
        reference_bundle_sha256=bundle["reference_bundle_sha256"],
        candidate_manifest_sha256=input_manifest_sha,
        exact_match_rows=exact_rows,
        near_match_rows=near_rows,
        known_semantic_match_rows=[],
    )
    if decontam["publication_eligible"] is not True:
        raise Data110Error("D06 decontamination did not pass")
    final_candidate = work / "near_dedup_survivors.jsonl"
    write_jsonl(final_candidate, [_datatrove_row(r) for r in survivors])
    publication = build_corpus_publication_manifest(
        corpus_manifest_sha256=input_manifest_sha,
        decontamination_report=decontam,
        current_benchmark_registry_sha256=d06_sha,
        output_files={"near_dedup_survivors.jsonl": sha256_file(final_candidate)},
    )
    metrics_sha = hash_json({
        "input": len(rows),
        "exact_d06_removed": len(exact_ids),
        "near_d06_removed": len(near_ids),
        "internal_near_removed_records": len(internal_removed_raw),
        "survivors": len(survivors),
    })
    dedup_manifest = build_dedup_output_manifest(
        plan=plan,
        input_records=len(rows),
        exact_survivors=len(rows),
        final_survivors=len(survivors),
        output_files={"near_dedup_survivors.jsonl": sha256_file(final_candidate)},
        metrics_sha256=metrics_sha,
    )
    eligibility = build_training_eligibility_envelope(
        output_manifest=dedup_manifest,
        source_rights_eligible=True,
        record_policy_eligible=True,
        exact_reserved_overlap_count=0,
        lexical_reserved_overlap_count=0,
        known_semantic_overlap_count=0,
        experiment_acceptance_pass=True,
    )
    if eligibility["training_eligible"] is not True:
        raise Data110Error("training eligibility envelope failed")
    evidence = {
        "datatrove_runtime": runtime,
        "datatrove_plan_sha256": plan.manifest()["plan_sha256"],
        "d06_benchmark_registry_sha256": d06_sha,
        "d06_production_registered_benchmarks": len(d06_manifest["benchmarks"]),
        "supplemental_local_heldout_references": len(refs),
        "reference_bundle_sha256": bundle["reference_bundle_sha256"],
        "exact_d06_matches_removed": len(exact_ids),
        "near_d06_matches_removed": len(near_ids),
        "internal_near_dedup_removed": len(internal_removed_raw),
        "decontamination_report_sha256": decontam["decontamination_report_sha256"],
        "publication_manifest_sha256": publication["publication_manifest_sha256"],
        "dedup_output_manifest_sha256": dedup_manifest["manifest_sha256"],
        "training_eligibility_envelope_sha256": eligibility["envelope_sha256"],
        "semantic_universal_cleanliness_claimed": False,
    }
    return survivors, evidence


def _split(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, str], dict[str, Any]]:
    split_records = [
        SplitRecord(
            id=str(r["id"]),
            text=str(r["text"]),
            source_id=str(r["source_id"]),
            modality=str(r["modality"]),
            content_sha256=str(r["content_sha256"]),
            near_duplicate_cluster_id=f"post-minhash-singleton:{r['content_sha256']}",
            training_eligible=True,
            purpose="training_eligible",
        )
        for r in rows
    ]
    corpus_sha = eligible_corpus_identity(split_records)
    relations_sha = dedup_relations_identity(split_records)
    spec = SplitFamilySpec(
        eligible_corpus_sha256=corpus_sha,
        dedup_relations_sha256=relations_sha,
        variant_seeds=("data110-v1-a", "data110-v1-b"),
        validation_fraction=0.10,
    )
    family = build_split_family(split_records, spec)
    verify_split_family_manifest(split_records, family)
    validation = set(family["validation_union_record_ids"])
    train = set(family["shared_train_record_ids"])
    if train & validation or train | validation != {r.id for r in split_records}:
        raise Data110Error("cluster-safe split coverage mismatch")
    assignments = {record.id: ("train" if record.id in train else "validation") for record in split_records}
    return assignments, family


def _shard(rows: Sequence[Mapping[str, Any]], assignments: Mapping[str, str], out: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    shard_dir = out / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    materialized: list[dict[str, Any]] = []
    by_split = Counter()
    by_stratum = Counter()
    by_origin = Counter()
    by_source = Counter()
    for source in rows:
        row = dict(source)
        row["split"] = assignments[str(row["id"])]
        row["record_id"] = str(row.pop("id"))
        for key in ("raw_sha256", "acquisition_url"):
            row.pop(key, None)
        materialized.append(row)
        n = len(str(row["text"]).encode("utf-8"))
        by_split[row["split"]] += n
        by_stratum[row["stratum"]] += n
        by_origin[row["origin"]] += n
        by_source[row["source_id"]] += n
    materialized.sort(key=lambda r: (str(r["split"]), str(r["stratum"]), str(r["record_id"])))
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_bytes = 0
    for row in materialized:
        size = len(cjson(row))
        if current and current_bytes + size > SHARD_TARGET_BYTES:
            groups.append(current)
            current = []
            current_bytes = 0
        current.append(row)
        current_bytes += size
    if current:
        groups.append(current)
    shards = []
    for index, group in enumerate(groups):
        path = shard_dir / f"part-{index:05d}.jsonl"
        digest, size = write_jsonl(path, group)
        shards.append({
            "path": f"shards/{path.name}",
            "sha256": digest,
            "size_bytes": size,
            "documents": len(group),
            "text_bytes": sum(len(str(r["text"]).encode("utf-8")) for r in group),
        })
    summary = {
        "documents": len(materialized),
        "text_bytes_by_split": dict(sorted(by_split.items())),
        "text_bytes_by_stratum": dict(sorted(by_stratum.items())),
        "text_bytes_by_origin": dict(sorted(by_origin.items())),
        "text_bytes_by_source": dict(sorted(by_source.items())),
        "shards": shards,
    }
    return materialized, summary


def _classification(external: Sequence[Mapping[str, Any]], d06_benchmarks: int) -> dict[str, Any]:
    external_sources = sorted({str(r["source_id"]) for r in external})
    reasons = [
        {
            "code": "EXTERNAL_SOURCE_DIVERSITY_TOO_NARROW",
            "detail": f"Only {len(external_sources)} admitted external source identities are present in this bounded RC.",
        },
        {
            "code": "NO_EXTERNAL_CODE_SOURCE",
            "detail": "Code training data remains explicitly project-authored; no rights-approved external code source is admitted.",
        },
        {
            "code": "D06_PRODUCTION_REGISTRY_SPARSE",
            "detail": f"Current D06 production registry contains {d06_benchmarks} benchmark entries; two local S0 held-out references supplement it.",
        },
        {
            "code": "REPRESENTATIVENESS_NOT_ESTABLISHED",
            "detail": "Passing pipeline mechanics does not establish population/domain representativeness or production readiness.",
        },
        {
            "code": "DATATROVE_EXPERIMENT_RUNTIME_NOT_FULL_TRANSITIVE_HASH_LOCK",
            "detail": "DATA-31 pins the DataTrove 0.10.0 wheel hash, while its auxiliary experiment dependencies are not a complete transitive hash-locked profile.",
        },
    ]
    return {"status": "RETEST_REQUIRED", "machine_readable_reasons": reasons}


def _build_once(repo: Path, external_dir: Path, out: Path) -> dict[str, Any]:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    data25_dir = out / "data25"
    data25 = verify_rebuild(repo / DATA25_CONFIG, data25_dir / "a", data25_dir / "b")
    retained = _read_json(repo / DATA25_RETAINED)
    if data25 != retained or data25["corpus_identity_sha256"] != DATA25_EXPECTED_ID:
        raise Data110Error("DATA-25 incumbent failed exact rebuild/retained identity check")
    project = _select_project_rows(_data25_rows(repo, data25_dir / "a", data25))
    external, intake_manifest = _load_external(external_dir)
    registry = _source_registry(external, project)
    registry_path = out / "source-registry.json"
    _write_cjson(registry_path, registry)
    combined = sorted([*external, *project], key=lambda r: str(r["id"]))
    gated, policy = _policy_gate(combined, out / "policy", registry["registry_identity_sha256"])
    survivors, dedup = _near_dedup_and_decontam(repo, gated, out / "dedup", registry["registry_identity_sha256"])
    if len(survivors) < 12:
        raise Data110Error("too few post-dedup records for a defensible split")
    assignments, split_family = _split(survivors)
    _, shard_summary = _shard(survivors, assignments, out)
    if not shard_summary["text_bytes_by_split"].get("train") or not shard_summary["text_bytes_by_split"].get("validation"):
        raise Data110Error("empty train or validation split")
    split_strata = Counter()
    for row in survivors:
        split_strata[f"{assignments[str(row['id'])]}:{row['stratum']}"] += 1
    for key in ("train:uk", "train:en", "train:code", "validation:uk", "validation:en", "validation:code"):
        if split_strata[key] <= 0:
            raise Data110Error(f"split lacks required stratum: {key}")
    external_admitted = [r for r in survivors if bool(r.get("external"))]
    surviving_external_strata = {str(r["stratum"]) for r in external_admitted}
    if surviving_external_strata != {"uk", "en"}:
        raise Data110Error(f"real external UK/EN did not survive all gates: {sorted(surviving_external_strata)}")
    classification = _classification(external, dedup["d06_production_registered_benchmarks"])
    core = {
        "schema_version": RELEASE_SCHEMA,
        "authority": AUTHORITY,
        "corpus_version": "1.0.0-rc.1",
        "upstream_heads": UPSTREAM,
        "selective_incumbent_blobs": SELECTIVE_BLOBS,
        "data25": {
            "corpus_identity_sha256": data25["corpus_identity_sha256"],
            "rebuild_twice_exact": True,
            "source_train_only": True,
            "project_byte_targets": TARGET_PROJECT_BYTES,
        },
        "external_intake": {
            "manifest_sha256": intake_manifest["manifest_sha256"],
            "physical_manifest_sha256": intake_manifest["physical_manifest_sha256"],
            "candidate_registry_identity_sha256": intake_manifest["candidate_registry_identity_sha256"],
            "admitted_records_before_policy": len(external),
            "admitted_external_source_ids": sorted({r["source_id"] for r in external}),
            "admitted_real_normalized_bytes": sum(len(str(r["text"]).encode("utf-8")) for r in external),
            "all_admitted_external_sources_explicit_training_eligible": all(r["allows_model_training"] is True for r in external),
            "immutable_provenance_bound": True,
        },
        "source_registry_identity_sha256": registry["registry_identity_sha256"],
        "origin_separation": {"external_real": True, "project_authored": True, "field": "origin"},
        "policy": policy,
        "dedup_decontamination": dedup,
        "split": {
            "split_family_identity_sha256": split_family["split_family_identity_sha256"],
            "eligible_corpus_sha256": split_family["eligible_corpus_sha256"],
            "dedup_relations_sha256": split_family["dedup_relations_sha256"],
            "cluster_straddles_across_variants": split_family["cluster_straddles_across_variants"],
            "training_policy": split_family["training_policy"],
            "train_documents": len(split_family["shared_train_record_ids"]),
            "validation_documents": len(split_family["validation_union_record_ids"]),
        },
        "physical": shard_summary,
        "records_by_split_stratum": dict(sorted(split_strata.items())),
        "external_records_surviving_all_gates": len(external_admitted),
        "pipeline": [
            "DATA-21/22 rights-aware real source intake",
            "DATA-25 deterministic normalization/materialization incumbent",
            "DATA-32 document quality incumbent re-executed",
            "DATA-33 privacy/secrets incumbent re-executed",
            "SQLiteExactDedupIndex exact dedup",
            "DATA-31 DataTrove 0.10.0 reference decontamination + internal near dedup",
            "DATA-36 cluster-safe split incumbent re-executed",
            "DATA-25 canonical JSONL physical sharding",
            "Product iter_packed_examples streaming/packing at Trainer runtime",
        ],
        "truth_boundary": {
            "contains_real_external_training_data": len(external_admitted) > 0,
            "contains_project_authored_training_data": True,
            "external_source_diversity_representative": False,
            "production_ready": False,
            "semantic_universal_cleanliness_claimed": False,
            "d06_authority_scope_only": True,
        },
        "classification": classification,
    }
    core["corpus_identity_sha256"] = hash_json(core)
    _write_json(out / "split-family.json", split_family)
    _write_json(out / "manifest.json", core)
    return core


def build_release(repo: Path, source_sha: str, external_dir: Path, out: Path) -> dict[str, Any]:
    _require_head(repo, source_sha)
    out.mkdir(parents=True, exist_ok=True)
    a = _build_once(repo, external_dir, out / "build-a")
    b = _build_once(repo, external_dir, out / "build-b")
    if a["corpus_identity_sha256"] != b["corpus_identity_sha256"]:
        raise Data110Error("two-build corpus identity mismatch")
    if [(x["path"], x["sha256"]) for x in a["physical"]["shards"]] != [(x["path"], x["sha256"]) for x in b["physical"]["shards"]]:
        raise Data110Error("two-build shard identity mismatch")
    release = {
        "schema_version": "12-6.data110-release-manifest.v1",
        "source_sha": source_sha,
        "corpus_identity_sha256": a["corpus_identity_sha256"],
        "build_a_identity_sha256": a["corpus_identity_sha256"],
        "build_b_identity_sha256": b["corpus_identity_sha256"],
        "two_build_deterministic_identity": True,
        "two_build_shards_exact": True,
        "candidate_manifest": a,
        "classification": a["classification"],
    }
    release["release_manifest_sha256"] = hash_json(release)
    _write_json(out / "release-manifest.json", release)
    return release


def _release_rows(root: Path, manifest: Mapping[str, Any], split: str, stratum: str) -> Iterator[dict[str, Any]]:
    for shard in manifest["physical"]["shards"]:
        path = root / str(shard["path"])
        if sha256_file(path) != shard["sha256"]:
            raise Data110Error(f"release shard hash mismatch: {path}")
        for row in _read_jsonl(path):
            if row.get("split") == split and row.get("stratum") == stratum:
                yield row


def _finite_packed(root: Path, manifest: Mapping[str, Any], tok: ByteTokenizer, split: str, stratum: str):
    records = (
        TextRecord(str(row["record_id"]), str(row["text"]), str(row["split"]))
        for row in _release_rows(root, manifest, split, stratum)
    )
    yield from iter_packed_examples(records, tok, expected_split=split, sequence_length=SEQ, cross_document=False)


def _cycling_packed(root: Path, manifest: Mapping[str, Any], tok: ByteTokenizer, stratum: str):
    while True:
        yielded = False
        for example in _finite_packed(root, manifest, tok, "train", stratum):
            yielded = True
            yield example
        if not yielded:
            raise Data110Error(f"no train examples for {stratum}")


def _steps_by_stratum(steps: int) -> dict[str, int]:
    result = {"uk": 0, "en": 0, "code": 0}
    for index in range(steps):
        result[MIXTURE[index % len(MIXTURE)]] += 1
    return result


def _train_iters(root: Path, manifest: Mapping[str, Any], tok: ByteTokenizer, completed_steps: int):
    result = {s: _cycling_packed(root, manifest, tok, s) for s in ("uk", "en", "code")}
    for stratum, steps in _steps_by_stratum(completed_steps).items():
        for _ in range(steps * BATCH):
            next(result[stratum])
    return result


def _batches(iterator):
    while True:
        examples = [next(iterator) for _ in range(BATCH)]
        yield {
            "input_ids": torch.tensor([x.input_ids for x in examples], dtype=torch.long),
            "labels": torch.tensor([x.labels for x in examples], dtype=torch.long),
        }


def _state_hash(model: TwelveSixDecoder) -> str:
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        h.update(name.encode() + b"\0")
        h.update(str(value.dtype).encode() + b"\0")
        h.update(str(tuple(value.shape)).encode() + b"\0")
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def _eval_examples(model: TwelveSixDecoder, examples) -> tuple[float, int]:
    ids = torch.tensor([x.input_ids for x in examples], dtype=torch.long)
    labels = torch.tensor([x.labels for x in examples], dtype=torch.long)
    logits = model(ids).logits[:, :-1, :].contiguous()
    targets = labels[:, 1:].contiguous()
    tokens = int(targets.ne(-100).sum().item())
    nll = F.cross_entropy(logits.reshape(-1, model.spec.vocab_size), targets.reshape(-1), ignore_index=-100, reduction="sum")
    return float(nll.item()), tokens


@torch.no_grad()
def _evaluate(model: TwelveSixDecoder, root: Path, manifest: Mapping[str, Any], tok: ByteTokenizer):
    before = _state_hash(model)
    training = model.training
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    by_stratum = {}
    try:
        for stratum in ("uk", "en", "code"):
            nll_sum = 0.0
            tokens = 0
            pending = []
            for example in _finite_packed(root, manifest, tok, "validation", stratum):
                pending.append(example)
                if len(pending) == 32:
                    n, t = _eval_examples(model, pending)
                    nll_sum += n
                    tokens += t
                    pending = []
            if pending:
                n, t = _eval_examples(model, pending)
                nll_sum += n
                tokens += t
            if tokens <= 0:
                raise Data110Error(f"no held-out target bytes for {stratum}")
            loss = nll_sum / tokens
            by_stratum[stratum] = {
                "loss": loss,
                "bits_per_byte": nll_sum / math.log(2.0) / tokens,
                "perplexity": perplexity_from_nll(loss),
                "predicted_byte_tokens": tokens,
            }
            total_nll += nll_sum
            total_tokens += tokens
    finally:
        model.train(training)
    after = _state_hash(model)
    if after != before:
        raise Data110Error("evaluation mutated model state")
    loss = total_nll / total_tokens
    return {
        "loss": loss,
        "bits_per_byte": total_nll / math.log(2.0) / total_tokens,
        "perplexity": perplexity_from_nll(loss),
        "predicted_byte_tokens": total_tokens,
        "by_stratum": by_stratum,
        "model_state_sha256_before": before,
        "model_state_sha256_after": after,
        "non_mutation_passed": True,
    }


def _run_manifest(source_sha: str, spec: ModelSpec, init: InitSpec, tok: ByteTokenizer, release: Mapping[str, Any], cfg: TrainerConfig, locks: Mapping[str, Any]):
    manifest = release["candidate_manifest"]
    value = {
        "schema": "12-6.data110-learning-run-manifest.v1",
        "source_sha": source_sha,
        "model_spec": spec.to_dict(),
        "model_spec_sha256": spec.identity_sha256(),
        "parameter_count": spec.parameter_count(),
        "init_spec": init.to_dict(),
        "init_spec_sha256": init.identity_sha256(),
        "tokenizer": {
            "version": tok.identity.version,
            "config_sha256": tok.identity.config_sha256,
            "vocab_sha256": tok.identity.vocab_sha256,
            "vocab_size": tok.identity.vocab_size,
        },
        "corpus_identity_sha256": manifest["corpus_identity_sha256"],
        "release_manifest_sha256": release["release_manifest_sha256"],
        "packing": {"version": PACKING_VERSION, "sequence_length": SEQ, "cross_document": False},
        "trainer_config": asdict(cfg),
        "batch_size": BATCH,
        "max_steps": MAX_STEPS,
        "checkpoint_steps": sorted(CHECKPOINT_STEPS),
        "mixture_pattern": list(MIXTURE),
        "environment_lock_sha256": locks["combined_sha256"],
        "foreign_pretrained_weights": False,
        "instruction_tuning": False,
        "paid_compute": False,
    }
    value["identity_sha256"] = hash_json(value)
    return value


def _identity(source_sha, spec, tok, release, run, cfg, trainer, locks):
    manifest = release["candidate_manifest"]
    training_config = {
        "trainer": asdict(cfg),
        "data": {
            "tokenizer_version": tok.identity.version,
            "packing_version": PACKING_VERSION,
            "packing_sequence_length": SEQ,
            "corpus_identity_sha256": manifest["corpus_identity_sha256"],
        },
    }
    return CheckpointIdentity(
        git_sha=source_sha,
        model_spec=spec.to_dict(),
        parameter_count=spec.parameter_count(),
        tokenizer_hash=tok.identity.config_sha256,
        tokenizer_vocab_hash=tok.identity.vocab_sha256,
        dataset_manifest_hash=manifest["corpus_identity_sha256"],
        run_manifest_hash=run["identity_sha256"],
        training_config=training_config,
        seed=cfg.seed,
        precision=cfg.precision,
        step=trainer.optimizer_step,
        tokens_seen=trainer.tokens_seen,
        optimizer={"name": "AdamW", "learning_rate": cfg.learning_rate, "betas": list(cfg.betas), "eps": cfg.eps, "weight_decay": cfg.weight_decay},
        scheduler=None,
        environment_lock_hash=locks["combined_sha256"],
    )


def _save(out, source_sha, spec, tok, release, run, cfg, trainer, locks):
    step = trainer.optimizer_step
    if step not in CHECKPOINT_STEPS:
        raise Data110Error(f"unexpected checkpoint step {step}")
    path = out / f"checkpoint-{step:04d}"
    save_trainer_checkpoint(path, model=trainer.model, trainer=trainer, identity=_identity(source_sha, spec, tok, release, run, cfg, trainer, locks), overwrite=True)
    checked = verify_checkpoint(path)
    return {"step": step, "tokens_seen": trainer.tokens_seen, "checkpoint_id": checked["checkpoint_id"]}


def _generation(checkpoint: Path):
    backend = load_first_party_backend(checkpoint)
    cfg = GenerationConfig(max_new_tokens=48, sample=False)
    outputs = {}
    for name, prompt in PROMPTS.items():
        result = generate(backend, prompt, cfg)
        outputs[name] = {"prompt": prompt, "generated_token_ids": list(result.generated_token_ids), "text": result.text, "stop_reason": result.stop_reason}
    return {"backend_diagnostics": backend.diagnostics(), "decoding": "greedy", "outputs": outputs}


def _machine(source_sha, locks):
    distributions = {}
    try:
        import importlib.metadata as metadata
        for name in ("datatrove", "orjson", "regex", "xxhash", "tokenizers"):
            try:
                distributions[name] = metadata.version(name)
            except metadata.PackageNotFoundError:
                distributions[name] = "NOT_INSTALLED"
    except Exception:
        distributions = {"status": "UNAVAILABLE"}
    return {
        "schema": "12-6.data110-machine-manifest.v1",
        "source_sha": source_sha,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "cuda_available": torch.cuda.is_available(),
        "device": "cpu",
        "pid": os.getpid(),
        "github_actions": os.environ.get("GITHUB_ACTIONS") == "true",
        "runner_os": os.environ.get("RUNNER_OS"),
        "runner_arch": os.environ.get("RUNNER_ARCH"),
        "environment_locks": locks,
        "data_experiment_distributions": distributions,
        "paid_compute": False,
    }


def _common(repo: Path, source_sha: str, out: Path, first: bool):
    _require_head(repo, source_sha)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    release = _read_json(out / "release-manifest.json")
    if release["classification"]["status"] != "RETEST_REQUIRED":
        raise Data110Error("release classification unexpectedly changed")
    tok = ByteTokenizer()
    spec, init, geometry = _model(repo)
    cfg = _trainer_config()
    locks = _locks(repo)
    run = _run_manifest(source_sha, spec, init, tok, release, cfg, locks)
    if first:
        _write_json(out / "run-manifest.json", run)
        _write_json(out / "machine-manifest-phase1.json", _machine(source_sha, locks))
    else:
        if _read_json(out / "run-manifest.json") != run:
            raise Data110Error("run manifest changed between processes")
        _write_json(out / "machine-manifest-resume.json", _machine(source_sha, locks))
    return release, tok, spec, init, geometry, cfg, locks, run


def phase1(repo: Path, source_sha: str, out: Path):
    release, tok, spec, init, geometry, cfg, locks, run = _common(repo, source_sha, out, True)
    corpus_root = out / "build-a"
    manifest = release["candidate_manifest"]
    torch.manual_seed(SEED)
    model = TwelveSixDecoder(spec, init)
    runtime_params = sum(p.numel() for p in model.parameters())
    if runtime_params != spec.parameter_count():
        raise Data110Error("runtime parameter count mismatch")
    random_hash = _state_hash(model)
    trainer = Trainer(model, cfg, device="cpu")
    observer = TrainingObserver(run, device="cpu", max_step_samples=1024)
    initial = observer.measure_region("evaluation", "heldout-init", lambda: _evaluate(model, corpus_root, manifest, tok), optimizer_step=0, tokens_seen=0)
    events = [_save(out, source_sha, spec, tok, release, run, cfg, trainer, locks)]
    generation0 = _generation(out / "checkpoint-0000")
    its = _train_iters(corpus_root, manifest, tok, 0)
    batches = {s: _batches(it) for s, it in its.items()}
    curve = out / "train-curve.jsonl"
    if curve.exists():
        curve.unlink()
    for index in range(RESUME_STEP):
        stratum = MIXTURE[index % len(MIXTURE)]
        batch, wait = observer.measure_next(batches[stratum])
        metrics = observer.train_microbatch(trainer, batch, data_wait_seconds=wait)
        _append(curve, {
            "optimizer_step": metrics.optimizer_step,
            "tokens_seen": trainer.tokens_seen,
            "stratum": stratum,
            "tokens": metrics.tokens,
            "loss": metrics.update_loss if metrics.update_loss is not None else metrics.loss,
            "grad_norm": metrics.grad_norm,
            "learning_rate": metrics.learning_rate,
        })
        if metrics.optimizer_step in (128, 256):
            events.append(observer.measure_region("checkpoint", f"save-{metrics.optimizer_step}", lambda: _save(out, source_sha, spec, tok, release, run, cfg, trainer, locks), optimizer_step=trainer.optimizer_step, tokens_seen=trainer.tokens_seen))
    if trainer.optimizer_step != RESUME_STEP:
        raise Data110Error("phase1 optimizer step mismatch")
    result = {
        "schema": "12-6.data110-learning-phase1.v1",
        "source_sha": source_sha,
        "process": {"pid": os.getpid(), "python_executable": sys.executable},
        "model": {
            "spec": spec.to_dict(),
            "spec_sha256": spec.identity_sha256(),
            "parameter_count": spec.parameter_count(),
            "runtime_parameter_count": runtime_params,
            "init_spec": init.to_dict(),
            "init_spec_sha256": init.identity_sha256(),
            "random_initialization": True,
            "random_init_state_sha256": random_hash,
            "geometry_provenance": geometry,
        },
        "tokenizer": {
            "version": tok.identity.version,
            "config_sha256": tok.identity.config_sha256,
            "vocab_sha256": tok.identity.vocab_sha256,
            "experimental_bpe_status": "REJECTED_FOR_FINAL_VERTICAL",
            "experimental_bpe_reason": "Repeatable TOK-37 ByteLevel BPE remains incompatible with current byte-bound first-party inference/checkpoint vertical and was selected on nonrepresentative DATA-10 evidence.",
        },
        "initial_heldout": initial,
        "initial_generation": generation0,
        "checkpoints": events,
        "observer": observer.summary(),
        "optimizer_step": trainer.optimizer_step,
        "tokens_seen": trainer.tokens_seen,
    }
    result["identity_sha256"] = hash_json(result)
    _write_json(out / "phase1.json", result)
    return result


def resume(repo: Path, source_sha: str, out: Path):
    release, tok, spec, init, _, cfg, locks, run = _common(repo, source_sha, out, False)
    corpus_root = out / "build-a"
    manifest = release["candidate_manifest"]
    p1 = _read_json(out / "phase1.json")
    torch.manual_seed(SEED)
    model = TwelveSixDecoder(spec, init)
    trainer = Trainer(model, cfg, device="cpu")
    loaded = load_trainer_checkpoint(
        out / "checkpoint-0256",
        model=model,
        trainer=trainer,
        strict_model=True,
        restore_rng=True,
        expected_git_sha=source_sha,
        expected_model_spec_hash=spec.identity_sha256(),
        expected_tokenizer_hash=tok.identity.config_sha256,
        expected_dataset_manifest_hash=manifest["corpus_identity_sha256"],
    )
    if loaded.manifest["identity"]["run_manifest_hash"] != run["identity_sha256"] or trainer.optimizer_step != RESUME_STEP:
        raise Data110Error("fresh-process checkpoint identity/step mismatch")
    observer = TrainingObserver(run, device="cpu", max_step_samples=1024)
    its = _train_iters(corpus_root, manifest, tok, RESUME_STEP)
    batches = {s: _batches(it) for s, it in its.items()}
    curve_path = out / "train-curve.jsonl"
    events = []
    first_resumed = None
    for index in range(RESUME_STEP, MAX_STEPS):
        stratum = MIXTURE[index % len(MIXTURE)]
        batch, wait = observer.measure_next(batches[stratum])
        metrics = observer.train_microbatch(trainer, batch, data_wait_seconds=wait)
        first_resumed = first_resumed or metrics.optimizer_step
        _append(curve_path, {
            "optimizer_step": metrics.optimizer_step,
            "tokens_seen": trainer.tokens_seen,
            "stratum": stratum,
            "tokens": metrics.tokens,
            "loss": metrics.update_loss if metrics.update_loss is not None else metrics.loss,
            "grad_norm": metrics.grad_norm,
            "learning_rate": metrics.learning_rate,
        })
        if metrics.optimizer_step in (384, 512):
            events.append(observer.measure_region("checkpoint", f"save-{metrics.optimizer_step}", lambda: _save(out, source_sha, spec, tok, release, run, cfg, trainer, locks), optimizer_step=trainer.optimizer_step, tokens_seen=trainer.tokens_seen))
    if first_resumed != 257 or trainer.optimizer_step != MAX_STEPS:
        raise Data110Error(f"fresh resume transition invalid: first={first_resumed}, final={trainer.optimizer_step}")
    final_eval = observer.measure_region("evaluation", "heldout-final", lambda: _evaluate(model, corpus_root, manifest, tok), optimizer_step=trainer.optimizer_step, tokens_seen=trainer.tokens_seen)
    final_generation = _generation(out / "checkpoint-0512")
    curve = [json.loads(x) for x in curve_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if len(curve) != MAX_STEPS:
        raise Data110Error(f"expected {MAX_STEPS} train points, got {len(curve)}")
    first64 = sum(float(x["loss"]) for x in curve[:64]) / 64
    last64 = sum(float(x["loss"]) for x in curve[-64:]) / 64
    bpb0 = float(p1["initial_heldout"]["bits_per_byte"])
    bpb1 = float(final_eval["bits_per_byte"])
    rel = relative_loss_improvement(float(p1["initial_heldout"]["loss"]), float(final_eval["loss"]))
    if not last64 < first64:
        raise Data110Error("train loss did not decrease")
    if not bpb1 < bpb0 or not rel > 0:
        raise Data110Error("held-out BPB/loss did not improve")
    token_by_stratum = {s: sum(int(x["tokens"]) for x in curve if x["stratum"] == s) for s in ("uk", "en", "code")}
    checkpoint_manifests = {str(step): verify_checkpoint(out / f"checkpoint-{step:04d}") for step in sorted(CHECKPOINT_STEPS)}
    final_checkpoint = checkpoint_manifests["512"]
    report = {
        "schema": SCHEMA,
        "authority": AUTHORITY,
        "source": {"repository": REPOSITORY, "git_sha": source_sha, "branch": "data110/corpus-v1-rc-20260826"},
        "release": release,
        "runtime": {
            "phase1_machine": _read_json(out / "machine-manifest-phase1.json"),
            "resume_machine": _read_json(out / "machine-manifest-resume.json"),
            "fresh_process_resume": {
                "phase1_pid": p1["process"]["pid"],
                "resume_pid": os.getpid(),
                "separate_cli_invocations_required": True,
                "checkpoint_loaded_step": loaded.manifest["identity"]["step"],
                "first_resumed_optimizer_step": first_resumed,
                "passed": p1["process"]["pid"] != os.getpid() and first_resumed == 257,
            },
        },
        "truth_boundary": {
            "genuinely_optimized_from_random_initialization": True,
            "foreign_pretrained_weights": False,
            "instruction_tuning": False,
            "paid_compute": False,
            "broad_intelligence_claim": False,
            "real_external_training_data_present": manifest["truth_boundary"]["contains_real_external_training_data"],
            "representative_corpus_claim": False,
            "production_ready_claim": False,
            "corpus_freeze_authority": False,
        },
        "tokenizer": p1["tokenizer"],
        "packing": {"version": PACKING_VERSION, "sequence_length": SEQ, "cross_document": False, "incumbent_reused_without_edit": True},
        "model": p1["model"],
        "training": {
            "trainer_config": asdict(cfg),
            "optimizer": "AdamW",
            "max_steps": MAX_STEPS,
            "batch_size": BATCH,
            "optimized_tokens": trainer.tokens_seen,
            "optimized_tokens_by_stratum": token_by_stratum,
            "optimized_token_mixture_fraction": {s: token_by_stratum[s] / trainer.tokens_seen for s in token_by_stratum},
            "first64_mean_loss": first64,
            "last64_mean_loss": last64,
            "train_loss_decreased": True,
            "phase1_observability": p1["observer"],
            "resume_observability": observer.summary(),
            "train_curve_path": "train-curve.jsonl",
            "sampling_note": "Streaming iterator may perform multiple explicit epochs over unique corpus records; no duplicate documents are added to the corpus to fake diversity.",
        },
        "evaluation": {
            "initial": p1["initial_heldout"],
            "final": final_eval,
            "initial_bits_per_byte": bpb0,
            "final_bits_per_byte": bpb1,
            "heldout_bits_per_byte_decreased": True,
            "relative_heldout_loss_improvement": rel,
            "evaluation_non_mutation": p1["initial_heldout"]["non_mutation_passed"] and final_eval["non_mutation_passed"],
        },
        "generation": {"before_training": p1["initial_generation"], "after_training": final_generation, "first_party_runtime": True},
        "checkpoints": {
            "steps": sorted(CHECKPOINT_STEPS),
            "phase1_events": p1["checkpoints"],
            "resume_events": events,
            "fresh_process_resume_from": "checkpoint-0256",
            "retained_exact_checkpoint": "checkpoint-0512",
            "retained_checkpoint_id": final_checkpoint["checkpoint_id"],
            "retained_checkpoint_identity": final_checkpoint["identity"],
        },
        "component_selection": {
            "model": "Product ModelSpec/TwelveSixDecoder; S2 geometry with vocab-only 2048->256 adaptation",
            "experimental_tokenizer": "TOK-37 repeatable ByteLevel BPE considered and rejected for final vertical",
            "selected_tokenizer": "canonical versioned s0-byte-v1",
            "corpus": "DATA-110 RC composed from DATA-21/22 real rights-approved UA/EN plus explicitly separated DATA-25 project-authored UK/EN/code",
            "quality": "DATA-32 incumbent byte-for-byte, re-executed on exact composed head",
            "privacy": "DATA-33 incumbent byte-for-byte, re-executed on exact composed head",
            "dedup": "SQLiteExactDedupIndex + DATA-31 DataTrove 0.10.0 MinHash",
            "decontamination": "DATA-31 D06-bound exact/MinHash reference gate",
            "split": "DATA-36 cluster-safe split incumbent byte-for-byte, re-executed",
            "streaming_packing": "DATA-25 JSONL sharding helpers + Product iter_packed_examples",
            "trainer_optimizer": "Product Trainer + AdamW",
            "observability": "TRAIN-29 TrainingObserver",
            "checkpoint_resume": "D05 save/load_trainer_checkpoint",
            "heldout_evaluation": "D06 metric primitives + byte-exact held-out NLL/BPB",
            "first_party_inference": "D07 load_first_party_backend + generate",
        },
        "reproduction": {
            "intake_command": "PYTHONPATH=src python tools/run_external_source_intake.py --output data110-external-intake --max-download-bytes 2000000 --max-normalized-chars 50000",
            "build_command": "PYTHONPATH=src python -m twelve_six.data110_release_candidate build --repo-root . --source-sha \"$(git rev-parse HEAD)\" --external-intake data110-external-intake --output-dir data110-evidence",
            "phase1_command": "PYTHONPATH=src python -m twelve_six.data110_release_candidate phase1 --repo-root . --source-sha \"$(git rev-parse HEAD)\" --output-dir data110-evidence",
            "resume_command": "PYTHONPATH=src python -m twelve_six.data110_release_candidate resume --repo-root . --source-sha \"$(git rev-parse HEAD)\" --output-dir data110-evidence",
        },
        "classification": release["classification"],
        "success": {
            "genuinely_learned_base_artifact": True,
            "all_runtime_training_checkpoint_eval_generation_gates": True,
            "corpus_pipeline_integrity_pass": True,
            "release_classification": release["classification"]["status"],
            "corpus_frozen": False,
        },
    }
    report["report_sha256"] = hash_json(report)
    _write_json(out / "report.json", report)
    return report


def validate(path: Path, expected_source_sha: str | None = None):
    report = _read_json(path)
    supplied = report["report_sha256"]
    unsigned = dict(report)
    unsigned.pop("report_sha256")
    if supplied != hash_json(unsigned):
        raise Data110Error("report self-hash mismatch")
    if report["schema"] != SCHEMA or report["authority"] != AUTHORITY:
        raise Data110Error("report schema/authority mismatch")
    if expected_source_sha and report["source"]["git_sha"] != expected_source_sha:
        raise Data110Error("report source mismatch")
    if report["model"]["parameter_count"] != 836_736:
        raise Data110Error("parameter gate failed")
    if report["release"]["two_build_deterministic_identity"] is not True:
        raise Data110Error("two-build identity gate failed")
    if report["release"]["candidate_manifest"]["dedup_decontamination"]["training_eligibility_envelope_sha256"] is None:
        raise Data110Error("training eligibility evidence missing")
    if report["release"]["candidate_manifest"]["split"]["cluster_straddles_across_variants"] != 0:
        raise Data110Error("cluster split leakage gate failed")
    if report["training"]["optimized_tokens"] < 400_000:
        raise Data110Error("bounded learning token budget too small")
    if not report["training"]["train_loss_decreased"] or not report["evaluation"]["heldout_bits_per_byte_decreased"]:
        raise Data110Error("learning improvement gate failed")
    if not report["evaluation"]["evaluation_non_mutation"]:
        raise Data110Error("evaluation mutation gate failed")
    if not report["runtime"]["fresh_process_resume"]["passed"]:
        raise Data110Error("fresh-process resume gate failed")
    if report["classification"]["status"] != "RETEST_REQUIRED" or report["success"]["corpus_frozen"] is not False:
        raise Data110Error("release truth boundary weakened")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    build = sub.add_parser("build")
    build.add_argument("--repo-root", type=Path, default=Path("."))
    build.add_argument("--source-sha", required=True)
    build.add_argument("--external-intake", type=Path, required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    for name in ("phase1", "resume"):
        q = sub.add_parser(name)
        q.add_argument("--repo-root", type=Path, default=Path("."))
        q.add_argument("--source-sha", required=True)
        q.add_argument("--output-dir", type=Path, required=True)
    q = sub.add_parser("validate")
    q.add_argument("report", type=Path)
    q.add_argument("--expected-source-sha")
    args = parser.parse_args(argv)
    if args.cmd == "build":
        value = build_release(args.repo_root.resolve(), args.source_sha, args.external_intake.resolve(), args.output_dir.resolve())
        print(json.dumps({"corpus_identity_sha256": value["corpus_identity_sha256"], "classification": value["classification"]["status"], "two_build": value["two_build_deterministic_identity"]}, indent=2))
    elif args.cmd == "phase1":
        value = phase1(args.repo_root.resolve(), args.source_sha, args.output_dir.resolve())
        print(json.dumps({"phase": "phase1", "step": value["optimizer_step"], "tokens": value["tokens_seen"], "initial_bpb": value["initial_heldout"]["bits_per_byte"]}, indent=2))
    elif args.cmd == "resume":
        value = resume(args.repo_root.resolve(), args.source_sha, args.output_dir.resolve())
        print(json.dumps({"phase": "resume", "parameters": value["model"]["parameter_count"], "tokens": value["training"]["optimized_tokens"], "initial_bpb": value["evaluation"]["initial_bits_per_byte"], "final_bpb": value["evaluation"]["final_bits_per_byte"], "classification": value["classification"]["status"], "checkpoint_id": value["checkpoints"]["retained_checkpoint_id"]}, indent=2))
    else:
        validate(args.report, args.expected_source_sha)
        print("DATA-110 release-candidate report validation: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
