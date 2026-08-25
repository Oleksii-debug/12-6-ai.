"""DATA-101 deterministic real-source-backed corpus V0.2 release orchestration.

This module composes existing D03/DATA incumbents. It does not implement new
rights, normalization, quality, privacy, exact/near-dedup, decontamination, or
tokenizer engines.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from twelve_six.data.benchmark_decontamination import (
    ReferenceRecord,
    build_decontamination_report,
    build_reference_bundle,
    exact_matches,
    near_match_records,
    run_datatrove_reference_filter,
)
from twelve_six.data.corpus_v01 import authored, cjson, norm, sha, split_for
from twelve_six.data.dedup_scale import (
    DataTroveMinhashExecutionPlan,
    run_datatrove_reference_index,
)
from twelve_six.data.document_quality import assess_document, run_quality_filter
from twelve_six.data.multilingual_pretraining import SQLiteExactDedupIndex
from twelve_six.data.near_dedup import NearDedupPolicy, run_datatrove_policy
from twelve_six.data.privacy_filter import build_scan_manifest, scan_record
from twelve_six.data.source_intake import load_candidate_registry, run_bounded_intake
from twelve_six.tokenization.experiments import (
    CorpusFileIdentity,
    TokenizerArtifactIdentity,
    TokenizerTrainingManifest,
    train_hf_tokenizer,
)

CONFIG_SCHEMA = "12-6.corpus-build-config.v2"
MANIFEST_SCHEMA = "12-6.corpus-manifest.v2"
BUILD_REPORT_SCHEMA = "12-6.corpus-v02-build-report.v1"
ORIGIN_REAL = "REAL_EXTERNAL"
ORIGIN_PROJECT = "PROJECT_AUTHORED"
STRATA = ("uk", "en", "code")


class CorpusV02Error(ValueError):
    """Fail-closed V0.2 release error."""


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _hash(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CorpusV02Error(f"{path}: JSON object required")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical(dict(value)))


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> tuple[str, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = b"".join(_canonical(dict(row)) for row in rows)
    path.write_bytes(payload)
    return _hash(payload), len(payload)


def _read_jsonl_tree(path: Path) -> list[dict[str, Any]]:
    files = [path] if path.is_file() else sorted(path.rglob("*.jsonl"))
    rows: list[dict[str, Any]] = []
    for file in files:
        for line in file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise CorpusV02Error(f"{file}: JSONL row must be object")
                rows.append(value)
    return rows


def _require_config(config: Mapping[str, Any]) -> dict[str, Any]:
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise CorpusV02Error("unsupported corpus V0.2 config schema")
    target = config.get("target_pre_filter_byte_tokens")
    if not isinstance(target, Mapping) or set(target) != set(STRATA):
        raise CorpusV02Error("target_pre_filter_byte_tokens must define uk/en/code")
    if any(isinstance(value, bool) or int(value) <= 0 for value in target.values()):
        raise CorpusV02Error("pre-filter byte targets must be positive")
    if config.get("tokenizer", {}).get("algorithm") != "bpe":
        raise CorpusV02Error("V0.2 experiment currently admits incumbent BPE only")
    return dict(config)


def _external_rows(repo_root: Path, config: Mapping[str, Any], work_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    registry_path = repo_root / str(config["external_candidate_registry"])
    registry, _sources = load_candidate_registry(registry_path)
    intake_dir = work_dir / "external-intake"
    intake = run_bounded_intake(registry, intake_dir)
    rows: list[dict[str, Any]] = []
    for record in intake["records"]:
        if record.get("status") != "ACCEPTED":
            continue
        text_path = intake_dir / str(record["text_path"])
        text = text_path.read_text(encoding="utf-8")
        if text.endswith("\n"):
            text = text[:-1]
        digest = _hash(text.encode("utf-8"))
        if digest != record["content_sha256"]:
            raise CorpusV02Error("external accepted-text content hash drift")
        language = str(record["language"])
        if language not in {"uk", "en"}:
            raise CorpusV02Error(f"unsupported accepted external language: {language}")
        rows.append(
            {
                "record_id": str(record["id"]),
                "source_id": str(record["source_id"]),
                "source_version": str(record["source_version"]),
                "stratum": language,
                "modality": "natural",
                "origin_class": ORIGIN_REAL,
                "external": True,
                "project_authored": False,
                "raw_identity": str(record["raw_sha256"]),
                "rights_status": str(record["rights_status"]),
                "license_id": str(record["license_id"]),
                "text": text,
            }
        )
    if not rows:
        raise CorpusV02Error("DATA-101 requires at least one rights-approved real record")
    return rows, intake


def _project_rows(config: Mapping[str, Any], existing: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    target = {key: int(value) for key, value in config["target_pre_filter_byte_tokens"].items()}
    supplied = Counter()
    for row in existing:
        supplied[str(row["stratum"])] += len(str(row["text"]).encode("utf-8"))

    rows: list[dict[str, Any]] = []
    for record in authored(config):
        stratum = str(record["stratum"])
        if supplied[stratum] >= target[stratum]:
            if all(supplied[key] >= target[key] for key in STRATA):
                break
            continue
        text = norm(str(record["raw_text"]), stratum == "code")
        rows.append(
            {
                "record_id": str(record["record_id"]),
                "source_id": str(record["source_id"]),
                "source_version": str(record["source_version"]),
                "stratum": stratum,
                "modality": "code" if stratum == "code" else "natural",
                "origin_class": ORIGIN_PROJECT,
                "external": False,
                "project_authored": True,
                "raw_identity": _hash(str(record["raw_text"]).encode("utf-8")),
                "rights_status": "PROJECT_CONTROLLED",
                "license_id": "PROJECT_AUTHORED",
                "text": text,
            }
        )
        supplied[stratum] += len(text.encode("utf-8"))
    if any(supplied[key] < target[key] for key in STRATA):
        raise CorpusV02Error(f"project-authored candidate supply exhausted: {dict(supplied)}")
    return rows


def _quality_filter(rows: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    input_identity = _hash(
        _canonical(
            [
                {
                    "id": row["record_id"],
                    "content_sha256": _hash(row["text"].encode("utf-8")),
                    "stratum": row["stratum"],
                }
                for row in sorted(rows, key=lambda item: item["record_id"])
            ]
        )
    )
    report = run_quality_filter(
        [
            {"id": row["record_id"], "text": row["text"], "mode": row["stratum"]}
            for row in rows
        ],
        input_manifest_sha256=input_identity,
    )
    accepted: list[dict[str, Any]] = []
    for row in rows:
        decision = assess_document(row["record_id"], row["text"], row["stratum"])
        if decision.accepted:
            accepted.append(row)
    if len(accepted) != int(report["accepted_documents"]):
        raise CorpusV02Error("DATA-32 quality report/filter decision drift")
    return accepted, report


def _privacy_filter(
    rows: Sequence[dict[str, Any]], *, source_registry_sha256: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    input_identity = _hash(
        _canonical(
            [
                {"id": row["record_id"], "sha256": _hash(row["text"].encode("utf-8"))}
                for row in sorted(rows, key=lambda item: item["record_id"])
            ]
        )
    )
    results = []
    accepted: list[dict[str, Any]] = []
    for row in rows:
        result = scan_record(
            record_id=row["record_id"],
            source_id=row["source_id"],
            source_version=row["source_version"],
            modality=row["modality"],
            text=row["text"],
        )
        results.append(result)
        if not result.train_eligible_after_privacy:
            continue
        updated = dict(row)
        updated["text"] = result.sanitized_text
        updated["privacy_action"] = result.action
        updated["privacy_redactions"] = result.redaction_count
        accepted.append(updated)
    report = build_scan_manifest(
        results,
        input_content_sha256=input_identity,
        source_registry_sha256=source_registry_sha256,
    )
    if len(accepted) != int(report["records_train_eligible_after_privacy"]):
        raise CorpusV02Error("DATA-33 privacy manifest/filter decision drift")
    return accepted, report


def _exact_dedup(rows: Sequence[dict[str, Any]], database: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    survivors: list[dict[str, Any]] = []
    duplicates: list[str] = []
    with SQLiteExactDedupIndex(database) as index:
        for row in sorted(rows, key=lambda item: item["record_id"]):
            digest = _hash(row["text"].encode("utf-8"))
            if index.seen_or_add(digest):
                duplicates.append(row["record_id"])
                continue
            updated = dict(row)
            updated["content_sha256"] = digest
            updated["utf8_bytes"] = len(row["text"].encode("utf-8"))
            survivors.append(updated)
        index.commit()
    core = {
        "engine": "incumbent_SQLiteExactDedupIndex",
        "input_documents": len(rows),
        "surviving_documents": len(survivors),
        "removed_documents": len(duplicates),
        "removed_record_ids": duplicates,
    }
    return survivors, {**core, "report_sha256": _hash(_canonical(core))}


def _policy(config: Mapping[str, Any], modality: str) -> NearDedupPolicy:
    raw = config["near_dedup"][modality]
    return NearDedupPolicy(
        name=str(raw["name"]),
        modality=modality,
        n_grams=int(raw["n_grams"]),
        num_buckets=int(raw["num_buckets"]),
        hashes_per_bucket=int(raw["hashes_per_bucket"]),
        seed=int(config["near_dedup"]["seed"]),
        hash_precision=int(config["near_dedup"]["hash_precision"]),
    )


def _near_dedup(
    rows: Sequence[dict[str, Any]], config: Mapping[str, Any], work_dir: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    survivor_ids: set[str] = set()
    reports: dict[str, Any] = {}
    for modality in ("natural", "code"):
        selected = [row for row in rows if row["modality"] == modality]
        if not selected:
            reports[modality] = {"status": "NOT_RUN_EMPTY_MODALITY"}
            continue
        datatrove_rows = [
            {
                "id": row["record_id"],
                "text": row["text"],
                "metadata": {
                    "source_id": row["source_id"],
                    "raw_identity": row["raw_identity"],
                },
            }
            for row in selected
        ]
        report = run_datatrove_policy(
            datatrove_rows,
            policy=_policy(config, modality),
            workspace=work_dir / f"near-{modality}",
            exercise_skip_completed=True,
        )
        reports[modality] = report
        survivor_ids.update(str(value) for value in report["survivor_ids"])
    survivors = [row for row in rows if row["record_id"] in survivor_ids]
    core = {
        "input_documents": len(rows),
        "surviving_documents": len(survivors),
        "removed_documents": len(rows) - len(survivors),
        "natural_policy_sha256": reports["natural"].get("policy", {}).get("policy_sha256"),
        "code_policy_sha256": reports["code"].get("policy", {}).get("policy_sha256"),
        "natural_survivor_ids": reports["natural"].get("survivor_ids", []),
        "code_survivor_ids": reports["code"].get("survivor_ids", []),
        "natural_removed_ids": reports["natural"].get("removed_ids", []),
        "code_removed_ids": reports["code"].get("removed_ids", []),
        "restart_verified": all(
            report.get("status") == "NOT_RUN_EMPTY_MODALITY"
            or report.get("restart", {}).get("signature_rerun_byte_identical") is True
            for report in reports.values()
        ),
        "engine": "DataTrove MinHash 0.10.0",
    }
    return survivors, {
        "summary": {**core, "report_sha256": _hash(_canonical(core))},
        "executions": reports,
    }


def _datatrove_record(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "id": row["record_id"],
        "text": row["text"],
        "source_id": row["source_id"],
        "content_sha256": row["content_sha256"],
        "metadata": {
            "source_id": row["source_id"],
            "content_sha256": row["content_sha256"],
        },
    }


def _decontaminate_train_against_validation(
    rows: Sequence[dict[str, Any]],
    *,
    config: Mapping[str, Any],
    source_registry_sha256: str,
    work_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    train = [row for row in rows if row["split"] == "train"]
    validation = [row for row in rows if row["split"] == "validation"]
    if not train or not validation:
        raise CorpusV02Error("stable split must contain train and validation records")

    refs = [
        ReferenceRecord(
            source_id=row["source_id"],
            document_id=row["record_id"],
            content_sha256=row["content_sha256"],
            category="corpus_v02_heldout_validation",
            evidence_ref="DATA-101 stable held-out validation split",
        )
        for row in validation
    ]
    benchmark_registry = config["decontamination"]["benchmark_registry"]
    bundle = build_reference_bundle(
        benchmark_registry=benchmark_registry,
        references=refs,
        rights_evidence_refs=[
            f"external-candidate-registry:{source_registry_sha256}",
            "project-authored-origin-class:corpus-v01-material",
        ],
    )
    exact_rows = exact_matches([_datatrove_record(row) for row in train], refs)
    exact_ids = {str(item["candidate_document_id"]) for item in exact_rows}
    nonexact = [row for row in train if row["record_id"] not in exact_ids]

    near_rows: list[dict[str, Any]] = []
    if nonexact:
        candidate_dir = work_dir / "decontam" / "candidate"
        reference_dir = work_dir / "decontam" / "reference"
        _write_jsonl(candidate_dir / "00000.jsonl", [_datatrove_record(row) for row in nonexact])
        _write_jsonl(reference_dir / "00000.jsonl", [_datatrove_record(row) for row in validation])
        split_identity = _hash(
            _canonical(
                [
                    {
                        "id": row["record_id"],
                        "split": row["split"],
                        "sha256": row["content_sha256"],
                    }
                    for row in sorted(rows, key=lambda item: item["record_id"])
                ]
            )
        )
        raw = config["decontamination"]
        plan = DataTroveMinhashExecutionPlan(
            source_registry_sha256=source_registry_sha256,
            reserved_registry_sha256=bundle["reference_bundle_sha256"],
            input_manifest_sha256=split_identity,
            workspace_uri=str((work_dir / "decontam").resolve()),
            candidate_shards=1,
            workers=1,
            n_grams=int(raw["n_grams"]),
            num_buckets=int(raw["num_buckets"]),
            hashes_per_bucket=int(raw["hashes_per_bucket"]),
            minhash_seed=int(raw["seed"]),
            hash_precision=int(raw["hash_precision"]),
        )
        reference_index = run_datatrove_reference_index(
            plan,
            reference_input=reference_dir,
            workspace=work_dir / "decontam" / "datatrove",
            index_name="data101-corpus-v02-validation",
        )
        result = run_datatrove_reference_filter(
            plan,
            candidate_input=candidate_dir,
            workspace=work_dir / "decontam" / "datatrove",
            reference_index=Path(reference_index["reference_index"]),
        )
        near_rows = near_match_records(
            _read_jsonl_tree(Path(result["removed"])),
            reference_bundle_sha256=bundle["reference_bundle_sha256"],
        )
    near_ids = {str(item["candidate_document_id"]) for item in near_rows}
    rejected = exact_ids | near_ids
    clean_train = [row for row in train if row["record_id"] not in rejected]
    final_rows = clean_train + validation
    report = build_decontamination_report(
        benchmark_registry_sha256=benchmark_registry["manifest_sha256"],
        reference_bundle_sha256=bundle["reference_bundle_sha256"],
        candidate_manifest_sha256=_hash(
            _canonical(
                [
                    {"id": row["record_id"], "split": row["split"], "sha256": row["content_sha256"]}
                    for row in sorted(rows, key=lambda item: item["record_id"])
                ]
            )
        ),
        exact_match_rows=exact_rows,
        near_match_rows=near_rows,
        known_semantic_match_rows=[],
    )
    if report["publication_eligible"] is not True:
        raise CorpusV02Error("DATA-31 decontamination report blocks publication")
    return final_rows, {
        "reference_bundle": bundle,
        "report": report,
        "rejected_train_record_ids": sorted(rejected),
    }


def _train_tokenizer(
    rows: Sequence[dict[str, Any]],
    *,
    config: Mapping[str, Any],
    output_dir: Path,
    split_identity: str,
) -> tuple[Any, dict[str, Any]]:
    train = sorted(
        (row for row in rows if row["split"] == "train"),
        key=lambda item: item["record_id"],
    )
    if not train:
        raise CorpusV02Error("tokenizer training split is empty")
    tokenizer_dir = output_dir / "tokenizer"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    training_rows = [
        {"id": row["record_id"], "text": row["text"]}
        for row in train
    ]
    train_path = tokenizer_dir / "train.jsonl"
    train_sha, train_bytes = _write_jsonl(train_path, training_rows)
    raw = config["tokenizer"]
    manifest = TokenizerTrainingManifest(
        experiment_id="data101-corpus-v02-bpe-v1",
        algorithm="bpe",
        tokenizers_version=str(raw["version"]),
        dataset_id="corpus-v02-final-train-split",
        dataset_manifest_sha256=split_identity,
        corpus_files=(CorpusFileIdentity("tokenizer/train.jsonl", train_sha, train_bytes),),
        vocab_size=int(raw["vocab_size"]),
        min_frequency=int(raw["min_frequency"]),
    )
    texts = [row["text"] for row in train]
    first = train_hf_tokenizer(manifest, texts)
    second = train_hf_tokenizer(manifest, texts)
    if first.artifact_identity != second.artifact_identity:
        raise CorpusV02Error("experimental BPE artifact is not repeatable")
    tokenizer_json = first._tokenizer.to_str()  # incumbent adapter owns the runtime
    tokenizer_path = tokenizer_dir / "tokenizer.json"
    tokenizer_path.write_text(tokenizer_json, encoding="utf-8", newline="")
    if _file_hash(tokenizer_path) != first.artifact_identity.tokenizer_json_sha256:
        raise CorpusV02Error("persisted tokenizer.json does not match artifact identity")
    artifact = {
        **asdict(first.artifact_identity),
        "config_sha256": first.artifact_identity.config_sha256,
        "identity": first.identity.as_dict(),
        "training_manifest": manifest.to_dict(),
        "training_manifest_sha256": manifest.sha256,
        "tokenizer_json_path": "tokenizer/tokenizer.json",
        "tokenizer_json_file_sha256": _file_hash(tokenizer_path),
        "repeat_build_identity_equal": True,
    }
    _write_json(tokenizer_dir / "artifact.json", artifact)
    return first, artifact


def _aggregate(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, dict[str, int]]:
    output: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        key = str(row[field])
        output[key]["documents"] += 1
        output[key]["bytes"] += int(row["utf8_bytes"])
        output[key]["tokenizer_tokens"] += int(row["tokenizer_tokens"])
    return {key: dict(value) for key, value in sorted(output.items())}


def build_corpus(config_path: str | Path, output_dir: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    config = _require_config(_load_json(config_path))
    repo_root = config_path.parents[2]
    output = Path(output_dir) if output_dir is not None else repo_root / str(config["output_dir"])
    output = output.resolve()
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    work = output / "work"
    work.mkdir()

    external, intake = _external_rows(repo_root, config, work)
    source_registry_sha = str(intake["candidate_registry_identity_sha256"])
    candidates = external + _project_rows(config, external)
    candidate_identity = _hash(
        _canonical(
            [
                {
                    "id": row["record_id"],
                    "origin": row["origin_class"],
                    "sha256": _hash(row["text"].encode("utf-8")),
                }
                for row in sorted(candidates, key=lambda item: item["record_id"])
            ]
        )
    )

    quality_rows, quality_report = _quality_filter(candidates)
    private_rows, privacy_report = _privacy_filter(
        quality_rows,
        source_registry_sha256=source_registry_sha,
    )
    exact_rows, exact_report = _exact_dedup(private_rows, work / "exact-dedup.sqlite3")
    near_rows, near_report = _near_dedup(exact_rows, config, work)

    split_rows = []
    for row in near_rows:
        updated = dict(row)
        updated["split"] = split_for(
            row["record_id"],
            str(config["split_salt"]),
            int(config["validation_basis_points"]),
        )
        split_rows.append(updated)
    decontaminated, decontamination = _decontaminate_train_against_validation(
        split_rows,
        config=config,
        source_registry_sha256=source_registry_sha,
        work_dir=work,
    )
    decontaminated.sort(key=lambda row: (row["split"], row["stratum"], row["record_id"]))
    train_hashes = {row["content_sha256"] for row in decontaminated if row["split"] == "train"}
    validation_hashes = {
        row["content_sha256"] for row in decontaminated if row["split"] == "validation"
    }
    overlap = train_hashes & validation_hashes
    if overlap:
        raise CorpusV02Error("normalized-content train/validation overlap is non-zero")

    split_identity = _hash(
        _canonical(
            [
                {
                    "id": row["record_id"],
                    "split": row["split"],
                    "content_sha256": row["content_sha256"],
                }
                for row in decontaminated
            ]
        )
    )
    tokenizer, tokenizer_artifact = _train_tokenizer(
        decontaminated,
        config=config,
        output_dir=output,
        split_identity=split_identity,
    )
    final_rows: list[dict[str, Any]] = []
    for row in decontaminated:
        updated = dict(row)
        updated["utf8_bytes"] = len(row["text"].encode("utf-8"))
        updated["tokenizer_tokens"] = len(tokenizer.encode(row["text"]))
        final_rows.append(updated)

    shard_target = int(config["shard_target_bytes"])
    shard_dir = output / "shards"
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_bytes = 0
    for row in final_rows:
        row_bytes = len(_canonical(row))
        if current and current_bytes + row_bytes > shard_target:
            groups.append(current)
            current = []
            current_bytes = 0
        current.append(row)
        current_bytes += row_bytes
    if current:
        groups.append(current)
    shards = []
    for index, group in enumerate(groups):
        path = shard_dir / f"part-{index:05d}.jsonl"
        digest, size = _write_jsonl(path, group)
        shards.append(
            {
                "path": f"shards/{path.name}",
                "sha256": digest,
                "size_bytes": size,
                "documents": len(group),
                "bytes": sum(int(row["utf8_bytes"]) for row in group),
                "tokenizer_tokens": sum(int(row["tokenizer_tokens"]) for row in group),
            }
        )

    train = [row for row in final_rows if row["split"] == "train"]
    real_train_tokens = sum(
        int(row["tokenizer_tokens"])
        for row in train
        if row["origin_class"] == ORIGIN_REAL
    )
    all_train_tokens = sum(int(row["tokenizer_tokens"]) for row in train)
    real_share = real_train_tokens / all_train_tokens if all_train_tokens else 0.0
    real_sources = sorted(
        {
            (row["source_id"], row["source_version"])
            for row in final_rows
            if row["origin_class"] == ORIGIN_REAL
        }
    )
    real_code_documents = sum(
        row["origin_class"] == ORIGIN_REAL and row["stratum"] == "code"
        for row in final_rows
    )

    core = {
        "schema_version": MANIFEST_SCHEMA,
        "corpus_version": str(config["corpus_version"]),
        "builder_sha256": _file_hash(Path(__file__)),
        "config_sha256": _file_hash(config_path),
        "candidate_identity_sha256": candidate_identity,
        "external_intake_manifest_sha256": intake["manifest_sha256"],
        "external_candidate_registry_identity_sha256": source_registry_sha,
        "split_identity_sha256": split_identity,
        "tokenizer": tokenizer_artifact,
        "pipeline": [
            "DATA-21/22 rights_resolver_and_bounded_acquisition",
            "incumbent_normalization",
            "DATA-32_document_quality",
            "DATA-33_privacy_and_secret_filter",
            "incumbent_SQLite_exact_dedup",
            "DATA-30_DataTrove_Minhash_near_dedup",
            "DATA-25_stable_train_validation_split",
            "DATA-31_heldout_decontamination",
            "D04_experimental_BPE_training_on_train_only",
            "physical_shards",
            "immutable_manifest",
        ],
        "filter_evidence": {
            "quality_run_sha256": quality_report["run_sha256"],
            "privacy_manifest_sha256": privacy_report["manifest_sha256"],
            "exact_dedup_report_sha256": exact_report["report_sha256"],
            "near_dedup_report_sha256": near_report["summary"]["report_sha256"],
            "decontamination_report_sha256": decontamination["report"][
                "decontamination_report_sha256"
            ],
        },
        "counts": {
            "candidate_documents": len(candidates),
            "candidate_real_external_documents": len(external),
            "candidate_project_authored_documents": len(candidates) - len(external),
            "quality_survivors": len(quality_rows),
            "privacy_survivors": len(private_rows),
            "exact_dedup_survivors": len(exact_rows),
            "near_dedup_survivors": len(near_rows),
            "final_documents": len(final_rows),
            "train_documents": len(train),
            "validation_documents": len(final_rows) - len(train),
            "real_external_source_identities": len(real_sources),
            "real_external_code_documents": real_code_documents,
        },
        "by_source": _aggregate(final_rows, "source_id"),
        "by_stratum": _aggregate(final_rows, "stratum"),
        "by_modality": _aggregate(final_rows, "modality"),
        "by_origin_class": _aggregate(final_rows, "origin_class"),
        "by_split": _aggregate(final_rows, "split"),
        "optimized_token_supply": {
            "definition": "one deterministic pass over final train documents under the versioned BPE tokenizer",
            "train_tokenizer_tokens": all_train_tokens,
            "real_external_train_tokenizer_tokens": real_train_tokens,
            "real_external_share": real_share,
        },
        "train_validation_normalized_content_overlap": len(overlap),
        "real_source_identities": [
            {"source_id": source_id, "source_version": source_version}
            for source_id, source_version in real_sources
        ],
        "shards": shards,
        "truth_boundary": {
            "contains_real_external_training_data": any(
                row["origin_class"] == ORIGIN_REAL and row["split"] == "train"
                for row in final_rows
            ),
            "contains_project_authored_training_data": any(
                row["origin_class"] == ORIGIN_PROJECT and row["split"] == "train"
                for row in final_rows
            ),
            "real_external_code_available": real_code_documents > 0,
            "broad_real_world_representativeness_claimed": False,
            "real_source_pool_limitation": (
                "Only the currently explicit rights-approved bounded UA/EN source pool is admitted; "
                "no rights-approved real code source exists on this repository state."
            ),
            "project_authored_never_relabeled_external": all(
                not row["external"] and row["project_authored"]
                for row in final_rows
                if row["origin_class"] == ORIGIN_PROJECT
            ),
            "stage_promotion": "NOT_PERFORMED",
        },
        "v01_to_v02_delta": {
            "v01_identity_sha256": "422f545da93526ceba2c0ff738e0b9deac65f35bfb31d87d2aab068cda091fc8",
            "real_external_training_data": "0 -> rights-approved bounded UA/EN records",
            "origin_accounting": "explicit REAL_EXTERNAL versus PROJECT_AUTHORED",
            "quality": "DATA-32 incumbent applied",
            "privacy": "DATA-33 incumbent applied",
            "near_dedup": "DATA-30 DataTrove MinHash applied",
            "decontamination": "DATA-31 held-out train-vs-validation pass applied",
            "tokenizer_accounting": "versioned BPE-512 tokens in addition to UTF-8 bytes",
        },
    }
    manifest = {**core, "corpus_identity_sha256": _hash(_canonical(core))}
    _write_json(output / "manifest.json", manifest)
    _write_json(output / "quality-report.json", quality_report)
    _write_json(output / "privacy-report.json", privacy_report)
    _write_json(output / "exact-dedup-report.json", exact_report)
    _write_json(output / "near-dedup-report.json", near_report)
    _write_json(output / "decontamination-report.json", decontamination)
    shutil.rmtree(work)
    return manifest


def verify_rebuild(
    config_path: str | Path,
    first_output: str | Path,
    second_output: str | Path,
) -> dict[str, Any]:
    first = build_corpus(config_path, first_output)
    second = build_corpus(config_path, second_output)
    if first["corpus_identity_sha256"] != second["corpus_identity_sha256"]:
        raise CorpusV02Error("clean V0.2 rebuild changed corpus identity")
    first_shards = [(row["path"], row["sha256"]) for row in first["shards"]]
    second_shards = [(row["path"], row["sha256"]) for row in second["shards"]]
    if first_shards != second_shards:
        raise CorpusV02Error("clean V0.2 rebuild changed shard hashes")
    if first["tokenizer"]["tokenizer_json_sha256"] != second["tokenizer"]["tokenizer_json_sha256"]:
        raise CorpusV02Error("clean V0.2 rebuild changed tokenizer identity")
    return first


def build_report(manifest: Mapping[str, Any]) -> dict[str, Any]:
    core = {
        "schema_version": BUILD_REPORT_SCHEMA,
        "status": "PASS",
        "corpus_identity_sha256": manifest["corpus_identity_sha256"],
        "train_validation_normalized_content_overlap": manifest[
            "train_validation_normalized_content_overlap"
        ],
        "real_external_share": manifest["optimized_token_supply"]["real_external_share"],
        "real_external_source_identities": manifest["counts"]["real_external_source_identities"],
        "real_external_code_documents": manifest["counts"]["real_external_code_documents"],
        "tokenizer_config_sha256": manifest["tokenizer"]["config_sha256"],
        "tokenizer_vocab_sha256": manifest["tokenizer"]["vocab_sha256"],
        "shard_hashes": {row["path"]: row["sha256"] for row in manifest["shards"]},
        "truth_boundary": manifest["truth_boundary"],
    }
    return {**core, "report_sha256": _hash(_canonical(core))}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--verify-rebuild", action="store_true")
    args = parser.parse_args()
    if args.verify_rebuild:
        manifest = verify_rebuild(
            args.config,
            args.output_dir / "rebuild-a",
            args.output_dir / "rebuild-b",
        )
    else:
        manifest = build_corpus(args.config, args.output_dir)
    report = build_report(manifest)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
