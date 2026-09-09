#!/usr/bin/env python3
"""Full selective scan of Rada_Trees secondary plain-text members, zero credit.

Consumes the exact immutable `rada_xtag_texts.7z` object already terminally role-probed
by D03. The whole 19.7 GB logical archive is never extracted. Instead, the complete
7z listing is verified against terminal evidence, exactly the 4,391 `texts/*.txt`
members (879,031,855 listed bytes) are selected under a separate 2 GB envelope, and
only that exact list is extracted. Every selected payload is hash-verified, classified
with the incumbent member classifier, and exact duplicates are collapsed for candidate
accounting. No member text is persisted and no training/corpus authority is granted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import classify_d03_rada_trees_members as classify
import inventory_d03_rada_trees_archive as inventory
import probe_d03_rada_trees_secondary_role as probe

ROOT = Path(__file__).resolve().parents[1]
TERMINAL_EVIDENCE = ROOT / "evidence/d03-rada-trees/secondary-archive-terminal-role-v1.json"
SCHEMA = "12-6.d03-rada-trees-secondary-plaintext-full-scan.v1"
EXPECTED_TERMINAL_EVIDENCE_ID = "3c4a9446c9a737bf9b4653d703296505f797d54195b5b466f5339c5e7e84ca8e"
EXPECTED_PARENT_REPORT_SHA256 = "4c24e90faabbcbeeae2cd7cc431c1bff0108d67c376d78b5838888424586e14c"
EXPECTED_CONTENT_SHA256 = "737fa5df061c55555a8c761023559a206cbcabf0299ae52d1ccae387f685803e"
EXPECTED_RUNTIME_IDENTITY = "336daf7d18be20b3c2bdf157d35d4dc41df161dd46966f0770e1d42d2402aa54"
EXPECTED_ARTIFACT_DIGEST = "sha256:b9e67cd6ebcc8ab7e40a2eed6e19c2dda9bc5523d6ab3c617d2ce689dd9d196a"
EXPECTED_LISTING_IDENTITY = "9e5bad28b2455a0b1682b7c91d0f81f4113a928223898570c9800cf8af7ad15a"
EXPECTED_MEMBER_COUNT = 8_782
EXPECTED_UNCOMPRESSED_BYTES = 19_711_802_635
EXPECTED_TEXT_MEMBER_COUNT = 4_391
EXPECTED_TEXT_LISTED_BYTES = 879_031_855
SELECTIVE_MAX_BYTES = 2_000_000_000
TEXT_PREFIX = "texts/"
TEXT_SUFFIX = ".txt"
EXPECTED_EXTRACTOR_VERSION = (
    "7-Zip 23.01 (x64) : Copyright (c) 1999-2023 Igor Pavlov : 2023-06-20"
)


class PlaintextScanError(RuntimeError):
    """Fail-closed selective-plaintext scan error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PlaintextScanError(message)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, chunk: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            data = handle.read(chunk)
            if not data:
                break
            digest.update(data)
    return digest.hexdigest()


def load_terminal_evidence(path: Path = TERMINAL_EVIDENCE) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), "terminal evidence root must be an object")
    require(
        value.get("schema_version") == "12-6.d03-rada-trees-secondary-terminal-role.v1",
        "terminal evidence schema drift",
    )
    require(value.get("evidence_identity_sha256") == EXPECTED_TERMINAL_EVIDENCE_ID, "terminal evidence identity drift")
    execution = value.get("execution")
    require(isinstance(execution, dict) and execution.get("conclusion") == "success", "terminal role execution is not success")
    require(execution.get("report_sha256") == EXPECTED_PARENT_REPORT_SHA256, "terminal role report identity drift")
    require(execution.get("artifact_digest") == EXPECTED_ARTIFACT_DIGEST, "terminal role artifact digest drift")
    source = value.get("source")
    require(isinstance(source, dict), "terminal source binding missing")
    require(source.get("dataset") == probe.DATASET, "terminal dataset drift")
    require(source.get("dataset_revision") == probe.REVISION, "terminal revision drift")
    require(source.get("archive_path") == probe.ARCHIVE, "terminal archive path drift")
    require(source.get("expected_size_bytes") == 697_768_591, "terminal archive size drift")
    require(source.get("content_sha256") == EXPECTED_CONTENT_SHA256, "terminal content SHA drift")
    inventory_block = value.get("inventory")
    require(isinstance(inventory_block, dict), "terminal inventory binding missing")
    require(inventory_block.get("listing_identity_sha256") == EXPECTED_LISTING_IDENTITY, "terminal listing identity drift")
    require(inventory_block.get("member_count") == EXPECTED_MEMBER_COUNT, "terminal member count drift")
    require(inventory_block.get("uncompressed_bytes") == EXPECTED_UNCOMPRESSED_BYTES, "terminal uncompressed byte drift")
    runtime = value.get("runtime")
    require(isinstance(runtime, dict), "terminal runtime binding missing")
    require(runtime.get("runtime_identity_sha256") == EXPECTED_RUNTIME_IDENTITY, "terminal runtime identity drift")
    require(runtime.get("python_version") == "3.11.16", "terminal Python runtime drift")
    require(runtime.get("extractor_command") == "7z", "terminal extractor command drift")
    require(runtime.get("extractor_version") == EXPECTED_EXTRACTOR_VERSION, "terminal extractor version drift")
    classification = value.get("classification")
    require(isinstance(classification, dict), "terminal classification binding missing")
    require(classification.get("plain_text_suffix_member_count") == EXPECTED_TEXT_MEMBER_COUNT, "terminal text-member count drift")
    boundary = value.get("claim_boundary")
    require(isinstance(boundary, dict), "terminal claim boundary missing")
    require(boundary.get("training_authorized_bytes") == 0, "terminal training credit drift")
    require(boundary.get("unique_causal_loss_positions_authorized") == 0, "terminal loss credit drift")
    for key in ("tokenizer_fit_authorized", "model_training_executed", "final_test_payload_accessed", "paid_compute_used"):
        require(boundary.get(key) is False, f"terminal boundary weakened: {key}")
    return value


def listing_identity(config: Mapping[str, Any], listing: list[dict[str, Any]]) -> str:
    source = config["source"]
    payload = {
        "dataset_head_sha": probe.REVISION,
        "archive_path": probe.ARCHIVE,
        "upstream_object_identity": source["xet_hash"],
        "archive_sha256": EXPECTED_CONTENT_SHA256,
        "archive_size_bytes": source["expected_size_bytes"],
        "members": listing,
    }
    return inventory.sha256_bytes(inventory.canonical_json(payload))


def select_plaintext_listing(listing: list[dict[str, Any]]) -> list[dict[str, Any]]:
    require(len(listing) == EXPECTED_MEMBER_COUNT, "full listing member count drift")
    total = sum(int(item["size_bytes"]) for item in listing)
    require(total == EXPECTED_UNCOMPRESSED_BYTES, "full listing byte total drift")
    selected: list[dict[str, Any]] = []
    for raw in listing:
        path = inventory.canonical_member_path(str(raw["path"]))
        if Path(path).suffix.lower() != TEXT_SUFFIX:
            continue
        require(path.startswith(TEXT_PREFIX), f"plain-text suffix outside {TEXT_PREFIX}: {path}")
        require("\n" not in path and "\r" not in path, f"newline forbidden in selected path: {path!r}")
        size = int(raw["size_bytes"])
        require(0 <= size <= 50_000_000, f"selected member size outside bound: {path}")
        selected.append({"path": path, "size_bytes": size})
    selected.sort(key=lambda item: item["path"])
    require(len(selected) == EXPECTED_TEXT_MEMBER_COUNT, "plain-text member count drift")
    selected_bytes = sum(item["size_bytes"] for item in selected)
    require(selected_bytes == EXPECTED_TEXT_LISTED_BYTES, "plain-text listed-byte total drift")
    require(selected_bytes <= SELECTIVE_MAX_BYTES, "plain-text subset exceeds selective extraction envelope")
    return selected


def extract_exact_selected(
    executable: str,
    archive: Path,
    destination: Path,
    expected: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    resolved = shutil.which(executable)
    require(resolved is not None, f"7z extractor unavailable: {executable}")
    destination.mkdir(parents=True, exist_ok=False)
    listfile = destination.parent / "rada-secondary-plaintext-members.txt"
    listfile.write_text("".join(item["path"] + "\n" for item in expected), encoding="utf-8")
    try:
        proc = subprocess.run(
            [
                resolved,
                "x",
                "-y",
                "-bd",
                f"-o{destination}",
                "-scsUTF-8",
                str(archive),
                f"@{listfile}",
            ],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=3600,
        )
        if proc.returncode != 0:
            raise PlaintextScanError(
                f"selective 7z extraction failed rc={proc.returncode}: {proc.stderr[-1000:]}"
            )
    finally:
        listfile.unlink(missing_ok=True)
    try:
        return inventory.inventory_extracted_tree(
            destination,
            expected,
            max_member=50_000_000,
            max_total=SELECTIVE_MAX_BYTES,
        )
    except ValueError as exc:
        raise PlaintextScanError(str(exc)) from exc


def summarize_classified(classified: list[dict[str, Any]]) -> dict[str, Any]:
    require(len(classified) == EXPECTED_TEXT_MEMBER_COUNT, "classified member count drift")
    class_counts = Counter(str(item["classification"]) for item in classified)
    class_bytes: Counter[str] = Counter()
    for item in classified:
        class_bytes[str(item["classification"])] += int(item["size_bytes"])
    candidates = [item for item in classified if item["classification"] == "PLAIN_TEXT_CANDIDATE"]
    by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in candidates:
        by_hash[str(item["sha256"])].append(item)
    duplicate_groups = [group for group in by_hash.values() if len(group) > 1]
    unique_bytes = sum(int(group[0]["size_bytes"]) for group in by_hash.values())
    before = sum(int(item["size_bytes"]) for item in candidates)
    duplicate_discount = before - unique_bytes
    return {
        "full_plaintext_member_classification_complete": True,
        "class_counts": dict(sorted(class_counts.items())),
        "class_bytes": dict(sorted(class_bytes.items())),
        "plain_text_candidate_members": len(candidates),
        "plain_text_candidate_bytes_before_exact_duplicate_collapse": before,
        "plain_text_candidate_exact_unique_payload_count": len(by_hash),
        "plain_text_candidate_bytes_after_exact_duplicate_collapse": unique_bytes,
        "plain_text_exact_duplicate_group_count": len(duplicate_groups),
        "plain_text_exact_duplicate_member_discount": sum(len(group) - 1 for group in duplicate_groups),
        "plain_text_exact_duplicate_byte_discount": duplicate_discount,
        "raw_member_text_emitted": False,
    }


def build_report(archive: Path) -> dict[str, Any]:
    terminal = load_terminal_evidence()
    config = probe.load_config()
    require(archive.is_file() and not archive.is_symlink(), "archive must be a regular file")
    require(archive.stat().st_size == int(config["source"]["expected_size_bytes"]), "archive size drift")
    observed_content_sha = sha256_file(archive)
    require(observed_content_sha == EXPECTED_CONTENT_SHA256, "archive content SHA does not reproduce terminal observation")

    extractor = inventory.find_extractor(None, list(config["inventory_policy"]["accepted_extractors"]))
    version = inventory.extractor_version(extractor)
    require(version == EXPECTED_EXTRACTOR_VERSION, "extractor runtime drift")
    listing = inventory.list_archive(extractor, archive)
    probe._listing_bounds(listing, config["inventory_policy"])
    require(listing_identity(config, listing) == EXPECTED_LISTING_IDENTITY, "complete archive listing identity drift")
    selected = select_plaintext_listing(listing)

    classifier_config = json.loads(probe.CLASSIFIER_CONFIG.read_text(encoding="utf-8"))
    classify.validate_config(classifier_config)
    policy = classifier_config["classification_policy"]
    with tempfile.TemporaryDirectory(prefix="rada-secondary-plaintext-") as tmp:
        extracted_root = Path(tmp) / "selected"
        members = extract_exact_selected(extractor, archive, extracted_root, selected)
        classified = probe.classify_extracted(extracted_root, members, policy)

    summary = summarize_classified(classified)
    member_metadata = sorted(classified, key=lambda item: item["path"])
    metadata_identity = sha256_bytes(canonical_bytes(member_metadata))
    role_state = (
        "FULL_SELECTIVE_PLAINTEXT_CLASSIFICATION_ALL_CANDIDATE_ZERO_CREDIT"
        if summary["plain_text_candidate_members"] == EXPECTED_TEXT_MEMBER_COUNT
        else "FULL_SELECTIVE_PLAINTEXT_CLASSIFICATION_MIXED_ZERO_CREDIT"
    )
    core: dict[str, Any] = {
        "schema_version": SCHEMA,
        "worker_id": "D03-RADA-TREES-SECONDARY-PLAINTEXT-FULL-SCAN-20260907",
        "execution_profile": "LOCAL_FREE",
        "parent_terminal_role": {
            "evidence_identity_sha256": EXPECTED_TERMINAL_EVIDENCE_ID,
            "report_sha256": EXPECTED_PARENT_REPORT_SHA256,
            "listing_identity_sha256": EXPECTED_LISTING_IDENTITY,
            "runtime_identity_sha256": terminal["runtime"]["runtime_identity_sha256"],
        },
        "source": {
            "dataset": probe.DATASET,
            "dataset_revision": probe.REVISION,
            "archive_path": probe.ARCHIVE,
            "xet_hash": config["source"]["xet_hash"],
            "expected_size_bytes": config["source"]["expected_size_bytes"],
            "content_sha256": observed_content_sha,
            "content_sha256_reproduced_against_terminal_observation": True,
        },
        "selective_extraction": {
            "selected_prefix": TEXT_PREFIX,
            "selected_suffix": TEXT_SUFFIX,
            "full_archive_member_count": len(listing),
            "full_archive_uncompressed_bytes": sum(int(item["size_bytes"]) for item in listing),
            "selected_member_count": len(selected),
            "selected_listed_bytes": sum(int(item["size_bytes"]) for item in selected),
            "selective_max_bytes": SELECTIVE_MAX_BYTES,
            "xtag_extracted": False,
            "listing_extraction_exact_match": True,
            "extractor_command": extractor,
            "extractor_version": version,
        },
        "classification": {
            **summary,
            "member_metadata_identity_sha256": metadata_identity,
            "member_metadata": member_metadata,
        },
        "decision": {
            "role_state": role_state,
            "content_identity_reproduced": True,
            "full_plaintext_role_scan_complete": True,
            "training_admission_claimed": False,
            "next_required": [
                "PERIOD_SESSION_PROVENANCE",
                "RIGHTS_SCOPE_REVALIDATION",
                "LANGUAGE_QUALITY_PRIVACY",
                "GLOBAL_EXACT_NEAR_LINEAGE_DEDUP",
                "RESERVED_EVALUATION_DECONTAMINATION",
                "FAMILY_CAP_MIX_RECOMPUTE",
            ],
        },
        "claim_boundary": {
            "candidate_bytes_are_training_credit": False,
            "training_authorized_bytes": 0,
            "unique_causal_loss_positions_authorized": 0,
            "tokenizer_fit_authorized": False,
            "optimizer_updates": 0,
            "model_training_executed": False,
            "final_test_payload_accessed": False,
            "paid_compute_used": False,
            "research_corpus_v1_released": False,
        },
    }
    return {**core, "report_sha256": sha256_bytes(canonical_bytes(core))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=1200.0)
    args = parser.parse_args()
    config = probe.load_config()
    try:
        if not args.archive.exists():
            observed = probe.acquire(config, args.archive, args.timeout)
            require(observed == EXPECTED_CONTENT_SHA256, "downloaded content SHA drift")
        report = build_report(args.archive)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        print("D03_RADA_TREES_SECONDARY_PLAINTEXT_SCAN=PASS_ZERO_CREDIT")
        print("REPORT_SHA256=" + report["report_sha256"])
        print("PLAINTEXT_CANDIDATE_MEMBERS=" + str(report["classification"]["plain_text_candidate_members"]))
        print("PLAINTEXT_CANDIDATE_UNIQUE_BYTES=" + str(report["classification"]["plain_text_candidate_bytes_after_exact_duplicate_collapse"]))
        return 0
    except (PlaintextScanError, probe.SecondaryProbeError, classify.ClassificationError, ValueError, RuntimeError, OSError) as exc:
        print(f"BLOCKED: {exc}")
        return 2
    finally:
        # Workflow may intentionally reuse the exact downloaded archive for a second
        # independent processing pass. Do not delete a user-supplied/existing file.
        args.archive.with_suffix(args.archive.suffix + ".partial").unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
