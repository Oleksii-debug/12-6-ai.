#!/usr/bin/env python3
"""Materialize the MILESTONE-100 bounded real UK/EN/code corpus fail-closed."""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from twelve_six.data.document_quality import assess_document
from twelve_six.data.external_sources import (
    EligibilityResolver,
    build_external_source_registry,
    external_source_from_mapping,
)
from twelve_six.data.privacy_filter import scan_record
from twelve_six.data.source_intake import run_bounded_intake

POLICY_REF = "policy://12-6/data/explicit-model-training-evidence-v1"
CAPTURED_AT = "2026-08-26T00:00:00+03:00"
DATA21_UK_BYTES = 88_565


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _canonical_hash(value: Any) -> str:
    payload = (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    return _sha256_bytes(payload)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected JSON object")
    return value


def _object_source_id(parent_source_id: str, raw_sha256: str) -> str:
    return f"{parent_source_id}.object-{raw_sha256[:16]}"


def _build_v2_registry(repo_root: Path, pins: dict[str, Any]) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    specs = []
    by_url: dict[str, dict[str, Any]] = {}
    for source in pins["sources"]:
        evidence_path = repo_root / source["rights_evidence_uri"].removeprefix("file:")
        evidence_sha = _sha256_file(evidence_path)
        if evidence_sha != source["rights_evidence_sha256"]:
            raise RuntimeError(f"rights evidence drift: {evidence_path}")
        for obj in source["objects"]:
            if obj["url"] in by_url:
                raise RuntimeError(f"duplicate pinned acquisition URL: {obj['url']}")
            object_source_id = _object_source_id(source["source_id"], obj["raw_sha256"])
            object_version = f"{source['source_version']}/object:{obj['raw_sha256'][:16]}"
            evidence_refs = [
                {
                    "evidence_id": f"{object_source_id}:rights",
                    "evidence_kind": "explicit_permission",
                    "uri": source["rights_evidence_uri"],
                    "sha256": evidence_sha,
                    "captured_at": CAPTURED_AT,
                    "source_id": object_source_id,
                    "source_version": object_version,
                },
                {
                    "evidence_id": f"{object_source_id}:policy",
                    "evidence_kind": "policy_decision",
                    "uri": source["rights_evidence_uri"],
                    "sha256": evidence_sha,
                    "captured_at": CAPTURED_AT,
                    "source_id": object_source_id,
                    "source_version": object_version,
                },
            ]
            spec = external_source_from_mapping(
                {
                    "source_id": object_source_id,
                    "source_version": object_version,
                    "provider": source["provider"],
                    "source_url": obj["url"],
                    "source_kind": source["source_kind"],
                    "purpose": "pretraining",
                    "synthetic": False,
                    "benchmark_material": False,
                    "held_out": False,
                    "snapshot": {
                        "uri": f"file:artifacts/data21-22/raw/{obj['raw_sha256']}.bin",
                        "sha256": obj["raw_sha256"],
                        "size_bytes": obj["raw_bytes"],
                        "retrieved_at": "2026-08-25",
                        "upstream_version": source["source_version"],
                        "retrieval_method": "DATA21_22_BOUNDED_HTTP_ARTIFACT",
                    },
                    "rights": {
                        "status": "APPROVED_FOR_TRAINING",
                        "license_id": source["license_id"],
                        "terms_url": source["terms_url"],
                        "allows_model_training": True,
                        "allows_derivatives": True,
                        "allows_redistribution": True,
                        "policy_ref": POLICY_REF,
                        "reviewed_at": "2026-08-26",
                        "reviewer_ref": "SWARM_WORKER_ID:DATA-102-UA-BREADTH",
                        "uses": {
                            "acquisition": "ALLOWED",
                            "storage": "ALLOWED",
                            "analysis": "ALLOWED",
                            "model_training": "ALLOWED",
                            "redistribution": "ALLOWED",
                        },
                        "evidence_refs": evidence_refs,
                    },
                }
            )
            specs.append(spec)
            by_url[obj["url"]] = {
                **obj,
                "parent_source_id": source["source_id"],
                "parent_source_version": source["source_version"],
                "object_source_id": object_source_id,
                "object_source_version": object_version,
                "rights_evidence_sha256": evidence_sha,
            }
    registry = build_external_source_registry(specs)
    resolver = EligibilityResolver(registry)
    for spec in specs:
        resolver.assert_model_training_eligible(
            spec.source_id,
            spec.source_version,
            spec.source_manifest_sha256,
        )
    return registry, by_url


def _script_counts(text: str) -> dict[str, int]:
    counts = Counter()
    for char in text:
        if not char.isalpha():
            continue
        name = unicodedata.name(char, "")
        if "CYRILLIC" in name:
            counts["cyrillic_letters"] += 1
        elif "LATIN" in name:
            counts["latin_letters"] += 1
        else:
            counts["other_script_letters"] += 1
    return dict(counts)


def _record_hash_record(record: dict[str, Any]) -> dict[str, Any]:
    text = record["text"]
    return {
        "id": record["id"],
        "source_id": record["source_id"],
        "source_version": record["source_version"],
        "language": record["language"],
        "mode": record["mode"],
        "family": record["family"],
        "text_sha256": _sha256_bytes(text.encode("utf-8")),
        "utf8_bytes": len(text.encode("utf-8")),
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def materialize(repo_root: Path) -> dict[str, Any]:
    pins = _load_json(repo_root / "configs/data/milestone100_real_source_pins_v1.json")
    registry, pins_by_url = _build_v2_registry(repo_root, pins)
    resolver = EligibilityResolver(registry)

    evidence_dir = repo_root / "evidence/milestone100"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    (evidence_dir / "data24_eligibility_inventory.json").write_text(
        json.dumps(resolver.inventory(), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    intake_dir = repo_root / "artifacts/milestone100/source_intake"
    if intake_dir.exists():
        shutil.rmtree(intake_dir)
    candidate_registry = _load_json(repo_root / "configs/data/external_source_candidates_ua_en_v1.json")
    intake_manifest = run_bounded_intake(
        candidate_registry,
        intake_dir,
        max_download_bytes=2_000_000,
        max_normalized_chars=100_000,
    )
    if intake_manifest["record_counts"]["accepted"] != 3:
        raise RuntimeError(f"expected all 3 DATA-21/22 objects, got {intake_manifest['record_counts']}")

    external_records: list[dict[str, Any]] = []
    filter_evidence: list[dict[str, Any]] = []
    seen_text_hashes: set[str] = set()
    for item in intake_manifest["records"]:
        if item.get("status") != "ACCEPTED":
            continue
        url = item["acquisition_url"]
        pin = pins_by_url.get(url)
        if pin is None:
            raise RuntimeError(f"acquired object did not pass DATA-24 preflight: {url}")
        if item["raw_sha256"] != pin["raw_sha256"] or int(item["raw_bytes"]) != int(pin["raw_bytes"]):
            raise RuntimeError(f"immutable source drift after preflight: {url}")
        if item["content_sha256"] != pin["normalized_sha256_data21"]:
            raise RuntimeError(f"normalization drift versus exact DATA-21/22 artifact: {url}")
        text_path = intake_dir / item["text_path"]
        text = text_path.read_text(encoding="utf-8").rstrip("\n")
        privacy = scan_record(
            record_id=item["id"],
            source_id=pin["object_source_id"],
            source_version=pin["object_source_version"],
            modality="natural",
            text=text,
        )
        if not privacy.train_eligible_after_privacy:
            raise RuntimeError(f"external record failed privacy gate: {item['id']} {privacy.action}")
        sanitized = privacy.sanitized_text
        assert sanitized is not None
        quality = assess_document(item["id"], sanitized, item["language"])
        if not quality.accepted:
            raise RuntimeError(f"external record failed quality gate: {item['id']} {quality.reasons}")
        content_hash = _sha256_bytes(sanitized.encode("utf-8"))
        if content_hash in seen_text_hashes:
            raise RuntimeError(f"cross-source exact duplicate after privacy filter: {item['id']}")
        seen_text_hashes.add(content_hash)
        external_records.append(
            {
                "id": item["id"],
                "text": sanitized,
                "source_id": pin["object_source_id"],
                "source_version": pin["object_source_version"],
                "upstream_source_id": pin["parent_source_id"],
                "upstream_source_version": pin["parent_source_version"],
                "raw_sha256": item["raw_sha256"],
                "normalized_sha256": item["content_sha256"],
                "post_privacy_sha256": content_hash,
                "language": item["language"],
                "mode": item["language"],
                "family": pin["family"],
                "synthetic": False,
                "training_eligible": True,
            }
        )
        filter_evidence.append(
            {
                "id": item["id"],
                "privacy": privacy.evidence_record(),
                "quality": quality.manifest(),
            }
        )

    code_records: list[dict[str, Any]] = []
    code_excluded: list[dict[str, Any]] = []
    source_sha = __import__("subprocess").check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip()
    for path in sorted((repo_root / "src/twelve_six").rglob("*.py")):
        rel = path.relative_to(repo_root).as_posix()
        text = path.read_text(encoding="utf-8")
        record_id = "code-" + hashlib.sha256(rel.encode("utf-8")).hexdigest()[:24]
        privacy = scan_record(
            record_id=record_id,
            source_id="project-authored.12-6-ai.source-code",
            source_version=f"git:{source_sha}",
            modality="code",
            text=text,
        )
        if not privacy.train_eligible_after_privacy:
            code_excluded.append({"id": record_id, "path": rel, "reason": f"privacy:{privacy.action}"})
            continue
        sanitized = privacy.sanitized_text
        assert sanitized is not None
        quality = assess_document(record_id, sanitized, "code")
        if not quality.accepted:
            code_excluded.append({"id": record_id, "path": rel, "reason": f"quality:{','.join(quality.reasons)}"})
            continue
        content_hash = _sha256_bytes(sanitized.encode("utf-8"))
        if content_hash in seen_text_hashes:
            code_excluded.append({"id": record_id, "path": rel, "reason": "exact_duplicate"})
            continue
        seen_text_hashes.add(content_hash)
        code_records.append(
            {
                "id": record_id,
                "text": sanitized,
                "source_id": "project-authored.12-6-ai.source-code",
                "source_version": f"git:{source_sha}",
                "source_path": rel,
                "language": "code",
                "mode": "code",
                "family": "project_authored_repository_code",
                "synthetic": False,
                "training_eligible": True,
            }
        )
        filter_evidence.append(
            {
                "id": record_id,
                "privacy": privacy.evidence_record(),
                "quality": quality.manifest(),
            }
        )

    if not code_records:
        raise RuntimeError("no project-authored code survived quality/privacy gates")

    metadata_val = next(
        record for record in external_records if record["raw_sha256"] == "7ac53dfb4bf6f73f178560e09f33160d0250c69fb679802f3254dc0eb4c9f509"
    )
    train_external = [record for record in external_records if record["id"] != metadata_val["id"]]
    train_records = train_external + code_records
    validation_records = [metadata_val]
    train_ids = {row["id"] for row in train_records}
    val_ids = {row["id"] for row in validation_records}
    if train_ids & val_ids:
        raise RuntimeError("train/validation identity overlap")

    packaged = repo_root / "data/s0/packaged"
    _write_jsonl(packaged / "train.jsonl", train_records)
    _write_jsonl(packaged / "validation.jsonl", validation_records)

    all_records = train_records + validation_records
    hashed_records = [_record_hash_record(row) for row in all_records]
    family_bytes: dict[str, int] = defaultdict(int)
    language_bytes: dict[str, int] = defaultdict(int)
    script_counts: Counter[str] = Counter()
    for row in all_records:
        byte_count = len(row["text"].encode("utf-8"))
        family_bytes[row["family"]] += byte_count
        language_bytes[row["language"]] += byte_count
        script_counts.update(_script_counts(row["text"]))

    manifest_core = {
        "schema_version": "12-6.milestone100-real-corpus.v1",
        "dataset_id": "m100-real-uk-en-code-v1",
        "source_git_sha": source_sha,
        "train_record_count": len(train_records),
        "validation_record_count": len(validation_records),
        "train_validation_record_overlap": [],
        "record_hashes": hashed_records,
        "source_families": dict(sorted(Counter(row["family"] for row in all_records).items())),
        "bytes_by_family": dict(sorted(family_bytes.items())),
        "byte_token_contribution_by_family": dict(sorted(family_bytes.items())),
        "bytes_by_language_or_mode": dict(sorted(language_bytes.items())),
        "script_distribution": dict(sorted(script_counts.items())),
        "average_document_utf8_bytes": sum(item["utf8_bytes"] for item in hashed_records) / len(hashed_records),
        "exact_duplicate_rate_after_filters": 0.0,
        "near_duplicate_status": "NOT_RUN_BOUNDED_SMALL_CORPUS",
        "external_training_eligible_records": len(external_records),
        "external_training_eligible_utf8_bytes": sum(len(row["text"].encode("utf-8")) for row in external_records),
        "project_authored_code_records": len(code_records),
        "contains_foreign_pretrained_weights": False,
        "contains_instruction_tuning": False,
        "synthetic_training_records": 0,
        "validation_never_used_for_training": True,
        "representativeness": {
            "intended_modalities_present": ["uk", "en", "code"],
            "bounded_small_vertical_representative": True,
            "broad_external_corpus_representative": False,
            "truth": "Real bounded UK/EN public/reference text plus real project-authored code; suitable for a small learning milestone, not a population-representative pretraining corpus."
        }
    }
    dataset_identity = _canonical_hash(manifest_core)
    manifest = {**manifest_core, "dataset_identity_sha256": dataset_identity}
    (packaged / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    uk_bytes = sum(len(row["text"].encode("utf-8")) for row in external_records if row["language"] == "uk")
    data102_comparison = {
        "schema_version": "12-6.data102-ua-breadth-comparison.v1",
        "previous_data21_uk_normalized_bytes": DATA21_UK_BYTES,
        "current_eligible_external_uk_bytes": uk_bytes,
        "newly_eligible_external_uk_bytes": max(0, uk_bytes - DATA21_UK_BYTES),
        "new_candidate_inventory_path": "configs/data/data102_ua_candidate_inventory_v1.json",
        "new_candidate_families_researched": 10,
        "new_external_training_comparison_status": (
            "NOT_RUN_NO_NEW_ELIGIBLE_UA_BYTES" if uk_bytes <= DATA21_UK_BYTES else "READY"
        ),
        "truth_boundary": "DATA-102 did not weaken DATA-24 to force breadth. Rights-compatible but not content-addressed candidates remain blocked."
    }
    (evidence_dir / "data102_ua_breadth_comparison.json").write_text(
        json.dumps(data102_comparison, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    (evidence_dir / "corpus_filter_evidence.json").write_text(
        json.dumps({"records": filter_evidence, "code_excluded": code_excluded}, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    corpus_report = {
        "dataset_identity_sha256": dataset_identity,
        "manifest_sha256": _sha256_file(packaged / "manifest.json"),
        "train_jsonl_sha256": _sha256_file(packaged / "train.jsonl"),
        "validation_jsonl_sha256": _sha256_file(packaged / "validation.jsonl"),
        "data24_registry_identity_sha256": registry["registry_identity_sha256"],
        "data21_intake_manifest_sha256": intake_manifest["manifest_sha256"],
        "data102": data102_comparison,
        "manifest": manifest,
    }
    (evidence_dir / "corpus_report.json").write_text(
        json.dumps(corpus_report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return corpus_report


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    report = materialize(repo_root)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
