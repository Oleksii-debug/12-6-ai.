#!/usr/bin/env python3
"""Full text-only streaming scan for the exact Rada_Trees secondary archive.

This successor reuses the existing archive lister, member streamer, and classifier.
It never extracts the full archive and never grants training/corpus capacity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import classify_d03_rada_trees_members as classify
import inventory_d03_rada_trees_archive as inventory
import probe_d03_rada_trees_secondary_role as probe

SCHEMA = "12-6.d03-rada-trees-secondary-full-txt-scan.v1"
EVIDENCE_SCHEMA = "12-6.d03-rada-trees-secondary-terminal-role.v1"
DATASET = "uacorpus/Rada_Trees"
REVISION = "1b994a5804dcda122721e8d33a03fd172cf8d867"
ARCHIVE = "rada_xtag_texts.7z"
ARCHIVE_BYTES = 697_768_591
ARCHIVE_SHA256 = "737fa5df061c55555a8c761023559a206cbcabf0299ae52d1ccae387f685803e"
GIT_BLOB_OID = "2ddd106a0140b6980c9a6152406f40f1f9bd5c30"
XET_HASH = "46c56a953551f3d33800c8b50fbd355104197abec45c366f245103e830b17d30"
EXPECTED_MEMBER_COUNT = 8_782
EXPECTED_UNCOMPRESSED_BYTES = 19_711_802_635
EXPECTED_TXT_MEMBER_COUNT = 4_391
EXPECTED_LISTING_IDENTITY = "9e5bad28b2455a0b1682b7c91d0f81f4113a928223898570c9800cf8af7ad15a"
EXPECTED_PARENT_EVIDENCE_ID = "3c4a9446c9a737bf9b4653d703296505f797d54195b5b466f5339c5e7e84ca8e"
PARENT_RUN = 34_158_830_442
PARENT_JOB = 101_856_189_735
PARENT_ARTIFACT = 10_032_220_704
PARENT_ARTIFACT_DIGEST = (
    "sha256:b9e67cd6ebcc8ab7e40a2eed6e19c2dda9bc5523d6ab3c617d2ce689dd9d196a"
)
PARENT_SOURCE_HEAD = "66ad574e1fe8183efd88a592323e5fccce00bde0"


class FullTextScanError(RuntimeError):
    """Fail-closed full text scan error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise FullTextScanError(message)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_obj(value: Any) -> str:
    return sha256_bytes(canonical_bytes(value))


def require_sha256(value: Any, label: str) -> str:
    require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{label} must be lowercase SHA-256",
    )
    return value


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FullTextScanError(f"cannot read JSON: {path}") from exc
    require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def validate_terminal_evidence(evidence: Mapping[str, Any]) -> None:
    require(evidence.get("schema_version") == EVIDENCE_SCHEMA, "terminal evidence schema drift")
    claimed_identity = require_sha256(
        evidence.get("evidence_identity_sha256"),
        "terminal evidence identity",
    )
    require(claimed_identity == EXPECTED_PARENT_EVIDENCE_ID, "terminal evidence identity drift")
    evidence_core = dict(evidence)
    evidence_core.pop("evidence_identity_sha256", None)
    require(
        sha256_obj(evidence_core) == claimed_identity,
        "terminal evidence self-hash mismatch",
    )
    execution = evidence.get("execution")
    require(isinstance(execution, Mapping), "terminal execution evidence missing")
    require(execution.get("conclusion") == "success", "terminal execution was not successful")
    require(execution.get("head_sha") == PARENT_SOURCE_HEAD, "terminal source head drift")
    require(execution.get("workflow_run") == PARENT_RUN, "terminal workflow run drift")
    require(execution.get("workflow_job") == PARENT_JOB, "terminal workflow job drift")
    require(execution.get("artifact_id") == PARENT_ARTIFACT, "terminal artifact id drift")
    require(
        execution.get("artifact_digest") == PARENT_ARTIFACT_DIGEST,
        "terminal artifact digest drift",
    )
    source = evidence.get("source")
    require(isinstance(source, Mapping), "terminal source evidence missing")
    expected_source = {
        "dataset": DATASET,
        "dataset_revision": REVISION,
        "archive_path": ARCHIVE,
        "expected_size_bytes": ARCHIVE_BYTES,
        "git_blob_oid": GIT_BLOB_OID,
        "xet_hash": XET_HASH,
        "content_sha256": ARCHIVE_SHA256,
    }
    for key, expected in expected_source.items():
        require(source.get(key) == expected, f"terminal source drift: {key}")
    observed = evidence.get("inventory")
    require(isinstance(observed, Mapping), "terminal inventory evidence missing")
    require(observed.get("member_count") == EXPECTED_MEMBER_COUNT, "member count drift")
    require(
        observed.get("uncompressed_bytes") == EXPECTED_UNCOMPRESSED_BYTES,
        "uncompressed bytes drift",
    )
    require(
        observed.get("listing_identity_sha256") == EXPECTED_LISTING_IDENTITY,
        "listing identity drift",
    )
    classification = evidence.get("classification")
    require(isinstance(classification, Mapping), "terminal classification evidence missing")
    require(
        classification.get("plain_text_suffix_member_count") == EXPECTED_TXT_MEMBER_COUNT,
        "plain-text suffix count drift",
    )
    boundary = evidence.get("claim_boundary")
    require(isinstance(boundary, Mapping), "terminal claim boundary missing")
    require(boundary.get("training_authorized_bytes") == 0, "parent training credit drift")
    require(
        boundary.get("unique_causal_loss_positions_authorized") == 0,
        "parent loss credit drift",
    )
    for key in (
        "tokenizer_fit_authorized",
        "model_training_executed",
        "final_test_payload_accessed",
        "paid_compute_used",
    ):
        require(boundary.get(key) is False, f"parent claim boundary weakened: {key}")


def verify_archive(archive: Path) -> str:
    require(archive.name == ARCHIVE, f"archive filename must be {ARCHIVE}")
    require(archive.is_file() and not archive.is_symlink(), "archive must be a regular file")
    stat = archive.stat()
    require(stat.st_size == ARCHIVE_BYTES, "archive byte size drift")
    digest = hashlib.sha256()
    with archive.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    observed = digest.hexdigest()
    require(observed == ARCHIVE_SHA256, "archive content SHA-256 drift")
    return observed


def listing_identity(listing: Sequence[Mapping[str, Any]]) -> str:
    members = [
        {"path": str(row["path"]), "size_bytes": int(row["size_bytes"])}
        for row in listing
    ]
    payload = {
        "dataset_head_sha": REVISION,
        "archive_path": ARCHIVE,
        "upstream_object_identity": XET_HASH,
        "archive_sha256": ARCHIVE_SHA256,
        "archive_size_bytes": ARCHIVE_BYTES,
        "members": members,
    }
    return inventory.sha256_bytes(inventory.canonical_json(payload))


def validate_listing(listing: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    require(len(listing) == EXPECTED_MEMBER_COUNT, "listing member count drift")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    total = 0
    for index, raw in enumerate(listing):
        require(isinstance(raw, Mapping), f"listing[{index}] must be an object")
        path = raw.get("path")
        size = raw.get("size_bytes")
        require(isinstance(path, str) and path, f"listing[{index}].path invalid")
        try:
            canonical = inventory.canonical_member_path(path)
        except ValueError as exc:
            raise FullTextScanError(f"noncanonical member path: {path}") from exc
        require(canonical == path, f"noncanonical member path: {path}")
        require(path not in seen, f"duplicate member path: {path}")
        require(
            isinstance(size, int) and not isinstance(size, bool) and 0 <= size <= 50_000_000,
            f"invalid member size: {path}",
        )
        seen.add(path)
        total += size
        rows.append({"path": path, "size_bytes": size})
    require(total == EXPECTED_UNCOMPRESSED_BYTES, "listing uncompressed byte total drift")
    require(
        listing_identity(rows) == EXPECTED_LISTING_IDENTITY,
        "listing identity does not match terminal authority",
    )
    txt = [row for row in rows if Path(row["path"]).suffix.lower() == ".txt"]
    require(len(txt) == EXPECTED_TXT_MEMBER_COUNT, "listing .txt count drift")
    return sorted(txt, key=lambda row: row["path"])


def _duplicate_groups(candidate_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_hash: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        by_hash[str(row["sha256"])].append(row)
    groups: list[dict[str, Any]] = []
    for digest, rows in by_hash.items():
        if len(rows) < 2:
            continue
        sorted_rows = sorted(rows, key=lambda item: str(item["path"]))
        groups.append(
            {
                "sha256": digest,
                "member_count": len(sorted_rows),
                "member_paths": [str(item["path"]) for item in sorted_rows],
                "raw_bytes_per_member": int(sorted_rows[0]["size_bytes"]),
            }
        )
    groups.sort(key=lambda item: (item["sha256"], item["member_paths"]))
    return groups


def scan_plain_text_members(
    listing: Sequence[Mapping[str, Any]],
    *,
    stream_member: Callable[[Mapping[str, Any]], bytes],
    classifier_policy: Mapping[str, Any],
) -> dict[str, Any]:
    txt_members = validate_listing(listing)
    classes: Counter[str] = Counter()
    class_bytes: Counter[str] = Counter()
    encodings: Counter[str] = Counter()
    candidate_rows: list[dict[str, Any]] = []
    scanned_bytes = 0

    for item in txt_members:
        payload = stream_member(item)
        require(isinstance(payload, bytes), f"stream result is not bytes: {item['path']}")
        require(len(payload) == int(item["size_bytes"]), f"streamed size drift: {item['path']}")
        decision = classify.classify_content(
            str(item["path"]),
            payload,
            dict(classifier_policy),
        )
        label = decision.get("class")
        require(isinstance(label, str) and label, f"classifier result invalid: {item['path']}")
        classes[label] += 1
        class_bytes[label] += len(payload)
        scanned_bytes += len(payload)
        encoding = decision.get("encoding")
        if isinstance(encoding, str) and encoding:
            encodings[encoding] += 1
        if label == "PLAIN_TEXT_CANDIDATE":
            candidate_rows.append(
                {
                    "path": str(item["path"]),
                    "size_bytes": len(payload),
                    "sha256": sha256_bytes(payload),
                    "decoded_encoding": encoding,
                    "text_metrics": decision.get("metrics"),
                    "year_hints": classify.year_hints(str(item["path"])),
                }
            )

    candidate_rows.sort(key=lambda row: row["path"])
    duplicate_groups = _duplicate_groups(candidate_rows)
    by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        by_hash[row["sha256"]].append(row)
    unique_candidates = [
        min(rows, key=lambda row: row["path"]) for rows in by_hash.values()
    ]
    unique_candidates.sort(key=lambda row: row["path"])

    candidate_raw_bytes = sum(int(row["size_bytes"]) for row in candidate_rows)
    candidate_unique_bytes = sum(int(row["size_bytes"]) for row in unique_candidates)
    duplicate_discount = candidate_raw_bytes - candidate_unique_bytes

    candidate_projection = [
        {
            "path": row["path"],
            "size_bytes": row["size_bytes"],
            "sha256": row["sha256"],
            "decoded_encoding": row["decoded_encoding"],
            "year_hints": row["year_hints"],
        }
        for row in candidate_rows
    ]
    unique_projection = [
        {
            "path": row["path"],
            "size_bytes": row["size_bytes"],
            "sha256": row["sha256"],
            "decoded_encoding": row["decoded_encoding"],
            "year_hints": row["year_hints"],
        }
        for row in unique_candidates
    ]

    return {
        "txt_member_count_scanned": len(txt_members),
        "txt_member_bytes_scanned": scanned_bytes,
        "classification_counts": dict(sorted(classes.items())),
        "classification_bytes": dict(sorted(class_bytes.items())),
        "encoding_counts": dict(sorted(encodings.items())),
        "plain_text_candidate_member_count": len(candidate_rows),
        "plain_text_candidate_raw_bytes": candidate_raw_bytes,
        "plain_text_candidate_unique_content_count": len(unique_candidates),
        "plain_text_candidate_raw_bytes_after_exact_duplicate_collapse": candidate_unique_bytes,
        "exact_duplicate_discount_bytes": duplicate_discount,
        "exact_duplicate_member_count": len(candidate_rows) - len(unique_candidates),
        "exact_duplicate_groups": duplicate_groups,
        "candidate_projection": candidate_projection,
        "candidate_projection_sha256": sha256_obj(candidate_projection),
        "unique_candidate_projection": unique_projection,
        "unique_candidate_projection_sha256": sha256_obj(unique_projection),
    }


def build_report(
    evidence: Mapping[str, Any],
    *,
    archive_sha256: str,
    listing: Sequence[Mapping[str, Any]],
    scan: Mapping[str, Any],
) -> dict[str, Any]:
    validate_terminal_evidence(evidence)
    require(archive_sha256 == ARCHIVE_SHA256, "archive SHA detached from terminal authority")
    txt_members = validate_listing(listing)
    require(
        scan.get("txt_member_count_scanned") == len(txt_members),
        "scan did not cover exact text member set",
    )
    require(
        scan.get("plain_text_candidate_member_count", 0)
        <= scan.get("txt_member_count_scanned", 0),
        "candidate count exceeds scanned members",
    )
    core: dict[str, Any] = {
        "schema_version": SCHEMA,
        "execution_profile": "LOCAL_FREE",
        "parent_terminal_evidence_identity_sha256": EXPECTED_PARENT_EVIDENCE_ID,
        "parent_execution": {
            "head_sha": PARENT_SOURCE_HEAD,
            "workflow_run": PARENT_RUN,
            "workflow_job": PARENT_JOB,
            "artifact_id": PARENT_ARTIFACT,
            "artifact_digest": PARENT_ARTIFACT_DIGEST,
        },
        "source": {
            "dataset": DATASET,
            "dataset_revision": REVISION,
            "archive_path": ARCHIVE,
            "archive_size_bytes": ARCHIVE_BYTES,
            "archive_sha256": ARCHIVE_SHA256,
            "git_blob_oid": GIT_BLOB_OID,
            "xet_hash": XET_HASH,
            "listing_identity_sha256": EXPECTED_LISTING_IDENTITY,
            "member_count": EXPECTED_MEMBER_COUNT,
            "uncompressed_bytes": EXPECTED_UNCOMPRESSED_BYTES,
            "plain_text_suffix_member_count": EXPECTED_TXT_MEMBER_COUNT,
        },
        "scan": dict(scan),
        "claim_boundary": {
            "source_role_scan_complete": True,
            "exact_raw_duplicate_collapse_complete": True,
            "period_session_provenance_complete": False,
            "rights_scope_revalidation_complete": False,
            "language_quality_privacy_complete": False,
            "global_lineage_dedup_complete": False,
            "evaluation_decontamination_complete": False,
            "family_mix_gate_complete": False,
            "training_authorized_bytes": 0,
            "unique_causal_loss_positions_authorized": 0,
            "tokenizer_fit_authorized": False,
            "model_training_executed": False,
            "optimizer_updates": 0,
            "final_test_payload_accessed": False,
            "paid_compute_used": False,
            "research_corpus_v1_released": False,
        },
        "next_required": [
            "PERIOD_SESSION_PROVENANCE",
            "RIGHTS_SCOPE_REVALIDATION",
            "LANGUAGE_QUALITY_PRIVACY",
            "GLOBAL_EXACT_NEAR_FRAGMENT_LINEAGE_DEDUP",
            "RESERVED_EVALUATION_DECONTAMINATION",
            "BALANCE_AND_FAMILY_CAP_RETEST",
        ],
        "raw_member_text_emitted": False,
    }
    core["report_identity_sha256"] = sha256_obj(core)
    return core


def verify_report(report: Mapping[str, Any]) -> None:
    require(report.get("schema_version") == SCHEMA, "report schema drift")
    identity = require_sha256(report.get("report_identity_sha256"), "report identity")
    core = dict(report)
    core.pop("report_identity_sha256", None)
    require(sha256_obj(core) == identity, "report self-hash mismatch")
    require(report.get("execution_profile") == "LOCAL_FREE", "report profile drift")
    require(report.get("raw_member_text_emitted") is False, "raw text emission boundary drift")
    source = report.get("source")
    require(isinstance(source, Mapping), "report source missing")
    require(source.get("archive_sha256") == ARCHIVE_SHA256, "report archive SHA drift")
    require(
        source.get("plain_text_suffix_member_count") == EXPECTED_TXT_MEMBER_COUNT,
        "report text member count drift",
    )
    scan = report.get("scan")
    require(isinstance(scan, Mapping), "report scan missing")
    require(
        scan.get("txt_member_count_scanned") == EXPECTED_TXT_MEMBER_COUNT,
        "report scan incomplete",
    )
    boundary = report.get("claim_boundary")
    require(isinstance(boundary, Mapping), "report claim boundary missing")
    require(boundary.get("training_authorized_bytes") == 0, "report training credit drift")
    require(
        boundary.get("unique_causal_loss_positions_authorized") == 0,
        "report loss credit drift",
    )
    for key in (
        "tokenizer_fit_authorized",
        "model_training_executed",
        "final_test_payload_accessed",
        "paid_compute_used",
        "research_corpus_v1_released",
    ):
        require(boundary.get(key) is False, f"report boundary weakened: {key}")


def run_scan(archive: Path, terminal_evidence: Path, output: Path) -> dict[str, Any]:
    evidence = load_json(terminal_evidence)
    validate_terminal_evidence(evidence)
    archive_sha = verify_archive(archive)
    config = probe.load_config()
    extractor = inventory.find_extractor(
        None,
        list(config["inventory_policy"]["accepted_extractors"]),
    )
    listing = inventory.list_archive(extractor, archive)
    classifier_config = json.loads(probe.CLASSIFIER_CONFIG.read_text(encoding="utf-8"))
    classify.validate_config(classifier_config)
    policy = classifier_config["classification_policy"]

    def stream(item: Mapping[str, Any]) -> bytes:
        return probe._stream_member(extractor, archive, dict(item))

    scan = scan_plain_text_members(
        listing,
        stream_member=stream,
        classifier_policy=policy,
    )
    report = build_report(
        evidence,
        archive_sha256=archive_sha,
        listing=listing,
        scan=scan,
    )
    verify_report(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--terminal-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-report", type=Path)
    args = parser.parse_args()
    try:
        if args.verify_report is not None:
            report = load_json(args.verify_report)
            verify_report(report)
            print("PASS_RADA_SECONDARY_FULL_TXT_SCAN_REPORT")
            return 0
        report = run_scan(args.archive, args.terminal_evidence, args.output)
    except (
        FullTextScanError,
        classify.ClassificationError,
        OSError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        print(f"BLOCKED: {exc}")
        return 2
    print("D03_RADA_SECONDARY_FULL_TXT_SCAN=COMPLETE_ZERO_CREDIT")
    print("REPORT_SHA256=" + report["report_identity_sha256"])
    print(
        "PLAIN_TEXT_CANDIDATE_MEMBERS="
        + str(report["scan"]["plain_text_candidate_member_count"])
    )
    print(
        "PLAIN_TEXT_UNIQUE_RAW_BYTES="
        + str(
            report["scan"][
                "plain_text_candidate_raw_bytes_after_exact_duplicate_collapse"
            ]
        )
    )
    print("TRAINING_AUTHORIZED_BYTES=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
