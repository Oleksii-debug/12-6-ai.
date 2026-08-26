#!/usr/bin/env python3
"""Fail-closed Rada_Trees plain-text member classifier.

Consumes a complete hashed-member report from D03/PR #708, re-verifies the
archive and every extracted regular file, and labels original plain-text
*candidates* separately from annotation/metadata/unknown members. This layer
never grants corpus capacity, tokenizer authority, or model-training authority.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import materialize_d03_rada_trees_archive as intake

CONFIG_SCHEMA = "12-6.d03-rada-trees-member-classification.v1"
REPORT_SCHEMA = "12-6.d03-rada-trees-member-classification-report.v1"
PARENT_REPORT_SCHEMA = "12-6.d03-rada-trees-archive-intake.v2"
DATASET = "uacorpus/Rada_Trees"
DATASET_HEAD = "1b994a5804dcda122721e8d33a03fd172cf8d867"
PARENT_HEAD = "ff50eb1e3b9b264ac713e248d01e2342a9784156"
PRIMARY_ARCHIVE = "Rada_Trees.7z"
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_YEAR = re.compile(r"(?<!\d)(19[9]\d|20(?:0\d|1\d|2[0-4]))(?!\d)")


class ClassificationError(RuntimeError):
    """Fail-closed classification/provenance error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ClassificationError(message)


def canonical_sha256(value: dict[str, Any]) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ClassificationError(f"cannot read JSON object: {path}") from exc
    require(isinstance(value, dict), f"{path}: JSON root must be an object")
    return value


def validate_config(config: dict[str, Any]) -> None:
    require(config.get("schema_version") == CONFIG_SCHEMA, "config schema drift")
    require(
        config.get("worker_id") == "D03-RADA-TREES-MEMBER-CLASSIFICATION-20260826",
        "worker id drift",
    )
    require(config.get("execution_profile") == "LOCAL_FREE", "execution profile weakened")

    parent = config.get("parent")
    require(isinstance(parent, dict), "parent binding missing")
    require(parent.get("pr") == 708, "parent PR drift")
    require(parent.get("head_sha") == PARENT_HEAD, "parent head drift")
    require(parent.get("report_schema") == PARENT_REPORT_SCHEMA, "parent schema drift")
    require(parent.get("dataset") == DATASET, "dataset drift")
    require(parent.get("dataset_head") == DATASET_HEAD, "dataset head drift")
    require(parent.get("archive_filename") == PRIMARY_ARCHIVE, "archive filename drift")
    require(parent.get("member_content_hashes_required") is True, "member hash gate weakened")
    require(parent.get("hf_object_identity_required") is True, "HF object identity gate weakened")

    policy = config.get("classification_policy")
    require(isinstance(policy, dict), "classification policy missing")
    require(policy.get("plain_text_candidate_suffixes") == [".txt"], "plain suffix policy drift")
    require(
        policy.get("ud_derivative_suffixes") == [".conllu", ".conll", ".cupt"],
        "UD suffix policy drift",
    )
    require(
        policy.get("annotation_derivative_suffixes")
        == [".xml", ".json", ".jsonl", ".tsv", ".csv"],
        "annotation suffix policy drift",
    )
    require(policy.get("strict_decode_order") == ["utf-8-sig", "windows-1251"], "decode policy drift")
    require(policy.get("reject_nul") is True, "NUL gate weakened")
    require(policy.get("conllu_min_noncomment_rows") == 3, "CoNLL-U threshold drift")
    require(policy.get("conllu_required_columns") == 10, "CoNLL-U column policy drift")
    require(policy.get("plain_text_requires_nonempty") is True, "empty-text gate weakened")
    require(policy.get("plain_text_max_tab_fraction") == 0.10, "tabular threshold drift")
    require(policy.get("emit_member_text") is False, "member text emission forbidden")
    require(policy.get("emit_member_content_preview") is False, "content preview forbidden")
    require(
        policy.get("exact_content_duplicates_collapsed_for_candidate_accounting") is True,
        "exact duplicate collapse weakened",
    )

    rights = config.get("rights_and_lineage")
    require(isinstance(rights, dict), "rights/lineage policy missing")
    require(rights.get("dataset_card_license") == "CC-BY-4.0", "license discovery binding drift")
    for key in (
        "attribution_required",
        "plain_text_original_transcripts_only_candidate",
        "parlamint_grac_overlap_requires_lineage_dedup",
        "period_provenance_stratification_required",
        "member_classification_is_not_rights_admission",
        "member_classification_is_not_family_independence_authority",
    ):
        require(rights.get(key) is True, f"rights/lineage gate weakened: {key}")
    require(rights.get("ud_annotation_default") == "HOLD_ZERO_CREDIT", "UD hold weakened")
    require(rights.get("nlp_uk_annotation_default") == "HOLD_ZERO_CREDIT", "nlp_uk hold weakened")

    expected_downstream = [
        "BIND_MEMBER_LEVEL_ATTRIBUTION_AND_SOURCE_PROVENANCE",
        "STRATIFY_PERIOD_PROVENANCE_AND_TRANSCRIPT_GENERATION_REGIME",
        "RUN_UKRAINIAN_LANGUAGE_QUALITY_PRIVACY_FILTERS",
        "RUN_EXACT_AND_NEAR_LINEAGE_DEDUP_AGAINST_RADA_LAWS_PARLAMINT_GRAC_AND_LIVE_CORPUS",
        "RUN_EVALUATION_DECONTAMINATION",
        "RECOMPUTE_FAMILY_CAP_AND_UA_MIXTURE_FEASIBILITY",
        "ONLY_THEN_PROPOSE_NONZERO_SOURCE_CAPACITY_CREDIT",
    ]
    require(config.get("downstream_required") == expected_downstream, "downstream order drift")

    boundary = config.get("claim_boundary")
    require(isinstance(boundary, dict), "claim boundary missing")
    for key in (
        "plain_text_member_classification_complete",
        "period_provenance_stratification_complete",
        "member_rights_terminal",
        "member_provenance_terminal",
        "language_quality_privacy_complete",
        "global_lineage_dedup_complete",
        "evaluation_decontamination_complete",
        "family_independence_terminal",
        "tokenizer_fit_authorized",
        "model_training_executed",
        "paid_compute_used",
        "research_corpus_v1_released",
    ):
        require(boundary.get(key) is False, f"claim boundary weakened: {key}")
    require(boundary.get("training_authorized_bytes") == 0, "training bytes must remain zero")
    require(
        boundary.get("unique_causal_loss_positions_authorized") == 0,
        "unique causal-loss positions must remain zero",
    )
    require(boundary.get("optimizer_updates") == 0, "optimizer updates must remain zero")


def verify_parent_report(report: dict[str, Any]) -> list[dict[str, Any]]:
    require(report.get("schema_version") == PARENT_REPORT_SCHEMA, "parent report schema mismatch")
    require(report.get("dataset") == DATASET, "parent dataset mismatch")
    require(report.get("dataset_head") == DATASET_HEAD, "parent dataset head mismatch")

    identity = report.get("report_sha256")
    require(isinstance(identity, str) and _HEX64.fullmatch(identity) is not None, "parent report identity invalid")
    body = dict(report)
    del body["report_sha256"]
    require(canonical_sha256(body) == identity, "parent report self-hash mismatch")

    archive = report.get("archive")
    require(isinstance(archive, dict), "parent archive block missing")
    require(archive.get("filename") == PRIMARY_ARCHIVE, "parent archive filename mismatch")
    require(archive.get("content_sha256_verified") is True, "parent content SHA-256 not verified")
    require(archive.get("object_snapshot_size_match") is True, "parent object-size binding missing")
    require(isinstance(archive.get("compressed_bytes"), int) and archive["compressed_bytes"] > 0, "parent archive size invalid")
    require(isinstance(archive.get("content_sha256"), str) and _HEX64.fullmatch(archive["content_sha256"]), "parent archive SHA-256 invalid")
    require(isinstance(archive.get("git_blob_oid"), str) and _HEX40.fullmatch(archive["git_blob_oid"]), "parent Git blob identity invalid")
    require(isinstance(archive.get("xet_hash"), str) and _HEX64.fullmatch(archive["xet_hash"]), "parent Xet identity invalid")
    require(
        isinstance(report.get("object_snapshot_identity_sha256"), str)
        and _HEX64.fullmatch(report["object_snapshot_identity_sha256"]),
        "parent object snapshot identity invalid",
    )

    boundary = report.get("claim_boundary")
    require(isinstance(boundary, dict), "parent claim boundary missing")
    for key in (
        "hf_object_identity_verified",
        "archive_content_identity_verified",
        "safe_member_inventory_verified",
        "member_content_hashes_verified",
    ):
        require(boundary.get(key) is True, f"parent prerequisite not terminal: {key}")
    require(boundary.get("plain_text_classification_complete") is False, "parent already claims classification")
    require(boundary.get("training_authorized_bytes") == 0, "parent training credit must remain zero")
    require(boundary.get("tokenizer_fit_authorized") is False, "parent tokenizer boundary weakened")
    require(boundary.get("model_training_authorized") is False, "parent model-training boundary weakened")

    inventory = report.get("inventory")
    require(isinstance(inventory, dict), "parent inventory missing")
    require(inventory.get("member_hashes_complete") is True, "parent member hashes incomplete")
    members = inventory.get("members")
    require(isinstance(members, list) and members, "parent member vector missing")

    files: list[dict[str, Any]] = []
    seen: set[str] = set()
    total = 0
    for raw in members:
        require(isinstance(raw, dict), "parent member entry invalid")
        path = raw.get("path")
        require(isinstance(path, str), "parent member path missing")
        require(intake.normalize_member_path(path) == path, f"noncanonical parent path: {path}")
        require(path not in seen, f"duplicate parent path: {path}")
        seen.add(path)
        is_directory = raw.get("is_directory")
        require(isinstance(is_directory, bool), f"invalid directory flag: {path}")
        size = raw.get("size")
        require(isinstance(size, int) and not isinstance(size, bool) and size >= 0, f"invalid member size: {path}")
        if is_directory:
            require(raw.get("sha256") is None, f"directory carries content hash: {path}")
            continue
        digest = raw.get("sha256")
        require(isinstance(digest, str) and _HEX64.fullmatch(digest), f"invalid member hash: {path}")
        files.append({"path": path, "size": size, "sha256": digest})
        total += size

    require(len(files) == inventory.get("file_count"), "parent file count mismatch")
    require(total == inventory.get("total_uncompressed_file_bytes"), "parent byte total mismatch")
    return sorted(files, key=lambda item: item["path"])


def decode_text(data: bytes, order: list[str]) -> tuple[str | None, str | None]:
    for encoding in order:
        try:
            return data.decode(encoding, errors="strict"), encoding
        except UnicodeDecodeError:
            pass
    return None, None


def looks_like_conllu(text: str, min_rows: int, required_columns: int) -> bool:
    matched = 0
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if len(line.split("\t")) == required_columns:
            matched += 1
            if matched >= min_rows:
                return True
    return False


def year_hints(path: str) -> list[int]:
    return sorted({int(match.group(1)) for match in _YEAR.finditer(path)})


def classify_content(path: str, data: bytes, policy: dict[str, Any]) -> dict[str, Any]:
    name = Path(path).name.lower()
    suffix = Path(path).suffix.lower()
    metadata = {str(item).lower() for item in policy["metadata_basenames"]}
    if name in metadata:
        return {"class": "METADATA_HOLD", "encoding": None, "metrics": None}
    if policy["reject_nul"] and b"\x00" in data:
        return {"class": "BINARY_OR_NUL_HOLD", "encoding": None, "metrics": None}
    if suffix in set(policy["ud_derivative_suffixes"]):
        return {"class": "DERIVED_UD_HOLD", "encoding": None, "metrics": None}
    if suffix in set(policy["annotation_derivative_suffixes"]):
        return {"class": "DERIVED_ANNOTATION_HOLD", "encoding": None, "metrics": None}
    if suffix not in set(policy["plain_text_candidate_suffixes"]):
        return {"class": "UNKNOWN_FORMAT_HOLD", "encoding": None, "metrics": None}

    text, encoding = decode_text(data, list(policy["strict_decode_order"]))
    if text is None or encoding is None:
        return {"class": "UNDECODABLE_TEXT_HOLD", "encoding": None, "metrics": None}
    stripped = text.strip()
    if not stripped and policy["plain_text_requires_nonempty"]:
        return {"class": "EMPTY_TEXT_HOLD", "encoding": encoding, "metrics": {"characters": len(text)}}

    prefix = stripped[:256].lower()
    if any(prefix.startswith(item) for item in policy["markup_prefixes"]):
        return {"class": "MARKUP_ANNOTATION_HOLD", "encoding": encoding, "metrics": None}
    if looks_like_conllu(
        text,
        int(policy["conllu_min_noncomment_rows"]),
        int(policy["conllu_required_columns"]),
    ):
        return {"class": "DERIVED_UD_HOLD", "encoding": encoding, "metrics": None}

    lines = text.splitlines()
    nonempty = [line for line in lines if line.strip()]
    tab_fraction = (sum("\t" in line for line in nonempty) / len(nonempty)) if nonempty else 0.0
    letters = [char for char in text if char.isalpha()]
    cyrillic = sum("\u0400" <= char <= "\u04ff" for char in letters)
    ua_specific = sum(char.lower() in "іїєґ" for char in letters)
    metrics = {
        "characters": len(text),
        "lines": len(lines),
        "nonempty_lines": len(nonempty),
        "tab_fraction": tab_fraction,
        "letters": len(letters),
        "cyrillic_letter_fraction": (cyrillic / len(letters)) if letters else 0.0,
        "ukrainian_specific_letter_count": ua_specific,
    }
    if tab_fraction > float(policy["plain_text_max_tab_fraction"]):
        return {"class": "TABULAR_ANNOTATION_HOLD", "encoding": encoding, "metrics": metrics}
    return {"class": "PLAIN_TEXT_CANDIDATE", "encoding": encoding, "metrics": metrics}


def extract_verify_classify(
    archive: Path,
    files: list[dict[str, Any]],
    policy: dict[str, Any],
    executable: str,
) -> list[dict[str, Any]]:
    resolved = shutil.which(executable)
    if resolved is None:
        raise ClassificationError(f"required 7z executable not found: {executable}")
    expected = {item["path"]: item for item in files}

    with tempfile.TemporaryDirectory(prefix="rada-trees-classify-") as tmp:
        root = Path(tmp)
        result = subprocess.run(
            [resolved, "x", "-y", "-bd", f"-o{root}", "--", str(archive)],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=1800,
        )
        if result.returncode != 0:
            raise ClassificationError(
                f"7z extraction failed with code {result.returncode}: {result.stderr[-500:]}"
            )

        actual: dict[str, Path] = {}
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            base = Path(dirpath)
            for name in dirnames:
                candidate = base / name
                if candidate.is_symlink():
                    raise ClassificationError(f"extracted directory symlink rejected: {candidate}")
            for name in filenames:
                candidate = base / name
                if candidate.is_symlink():
                    raise ClassificationError(f"extracted file symlink rejected: {candidate}")
                rel = intake.normalize_member_path(candidate.relative_to(root).as_posix())
                require(rel not in actual, f"duplicate extracted path: {rel}")
                actual[rel] = candidate

        require(set(actual) == set(expected), "extracted files differ from hashed parent inventory")
        classified: list[dict[str, Any]] = []
        for path in sorted(expected):
            source = expected[path]
            disk = actual[path]
            require(disk.is_file(), f"extracted member is not regular: {path}")
            require(disk.stat().st_size == source["size"], f"member size drift: {path}")
            data = disk.read_bytes()
            require(sha256_bytes(data) == source["sha256"], f"member SHA-256 drift: {path}")
            decision = classify_content(path, data, policy)
            classified.append(
                {
                    "path": path,
                    "bytes": len(data),
                    "sha256": source["sha256"],
                    "classification": decision["class"],
                    "decoded_encoding": decision["encoding"],
                    "text_metrics": decision["metrics"],
                    "path_year_hints": year_hints(path),
                    "text_emitted": False,
                }
            )
        return classified


def build_report(
    archive: Path,
    parent_report: dict[str, Any],
    config: dict[str, Any],
    executable: str = "7z",
) -> dict[str, Any]:
    validate_config(config)
    files = verify_parent_report(parent_report)
    require(archive.name == PRIMARY_ARCHIVE and archive.is_file(), "expected existing Rada_Trees.7z")

    archive_block = parent_report["archive"]
    require(archive.stat().st_size == archive_block["compressed_bytes"], "archive byte-count drift")
    require(sha256_file(archive) == archive_block["content_sha256"], "archive SHA-256 drift")

    members = extract_verify_classify(
        archive,
        files,
        config["classification_policy"],
        executable,
    )
    class_counts = Counter(item["classification"] for item in members)
    class_bytes: Counter[str] = Counter()
    for item in members:
        class_bytes[item["classification"]] += item["bytes"]

    candidates = [item for item in members if item["classification"] == "PLAIN_TEXT_CANDIDATE"]
    by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in candidates:
        by_hash[item["sha256"]].append(item)
    duplicate_groups = [
        {
            "sha256": digest,
            "paths": sorted(item["path"] for item in group),
            "member_count": len(group),
            "bytes_per_member": group[0]["bytes"],
        }
        for digest, group in sorted(by_hash.items())
        if len(group) > 1
    ]
    unique_exact_bytes = sum(group[0]["bytes"] for group in by_hash.values())
    year_counts = Counter(
        year
        for item in candidates
        for year in item["path_year_hints"]
    )

    core: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA,
        "worker_id": config["worker_id"],
        "execution_profile": "LOCAL_FREE",
        "dataset": DATASET,
        "dataset_head": DATASET_HEAD,
        "parent": {
            "pr": 708,
            "head_sha": PARENT_HEAD,
            "report_sha256": parent_report["report_sha256"],
            "object_snapshot_identity_sha256": parent_report[
                "object_snapshot_identity_sha256"
            ],
            "archive_content_sha256": archive_block["content_sha256"],
            "git_blob_oid": archive_block["git_blob_oid"],
            "xet_hash": archive_block["xet_hash"],
        },
        "config_sha256": canonical_sha256(config),
        "classification": {
            "file_count": len(members),
            "class_counts": dict(sorted(class_counts.items())),
            "class_bytes": dict(sorted(class_bytes.items())),
            "plain_text_candidate_member_count": len(candidates),
            "plain_text_candidate_bytes_before_exact_duplicate_collapse": sum(
                item["bytes"] for item in candidates
            ),
            "plain_text_candidate_bytes_after_exact_duplicate_collapse": unique_exact_bytes,
            "exact_duplicate_group_count": len(duplicate_groups),
            "exact_duplicate_groups": duplicate_groups,
            "candidate_path_year_hint_counts": {
                str(year): count for year, count in sorted(year_counts.items())
            },
            "members": members,
        },
        "interpretation": {
            "plain_text_candidate_is_training_admission": False,
            "candidate_bytes_are_training_capacity": False,
            "path_year_hints_are_terminal_period_provenance": False,
            "exact_duplicate_collapse_is_global_lineage_dedup": False,
            "parlamint_grac_overlap_still_requires_cross_source_lineage_dedup": True,
        },
        "claim_boundary": {
            "plain_text_member_classification_complete": True,
            "period_provenance_stratification_complete": False,
            "member_rights_terminal": False,
            "member_provenance_terminal": False,
            "language_quality_privacy_complete": False,
            "global_lineage_dedup_complete": False,
            "evaluation_decontamination_complete": False,
            "family_independence_terminal": False,
            "training_authorized_bytes": 0,
            "unique_causal_loss_positions_authorized": 0,
            "tokenizer_fit_authorized": False,
            "model_training_executed": False,
            "optimizer_updates": 0,
            "paid_compute_used": False,
            "research_corpus_v1_released": False,
            "safe_result": (
                "MEMBERS_CLASSIFIED_PLAIN_TEXT_CANDIDATES_REQUIRE_"
                "PROVENANCE_QUALITY_AND_LINEAGE_DEDUP"
            ),
        },
        "raw_member_text_emitted": False,
    }
    return {**core, "report_sha256": canonical_sha256(core)}


def verify_report(report: dict[str, Any]) -> None:
    require(report.get("schema_version") == REPORT_SCHEMA, "report schema mismatch")
    identity = report.get("report_sha256")
    require(isinstance(identity, str) and _HEX64.fullmatch(identity), "report self-hash invalid")
    body = dict(report)
    del body["report_sha256"]
    require(canonical_sha256(body) == identity, "report self-hash mismatch")
    require(report.get("execution_profile") == "LOCAL_FREE", "report execution profile drift")
    require(report.get("dataset") == DATASET, "report dataset drift")
    require(report.get("dataset_head") == DATASET_HEAD, "report dataset head drift")
    require(report.get("raw_member_text_emitted") is False, "raw text emission forbidden")

    section = report.get("classification")
    require(isinstance(section, dict), "classification section missing")
    members = section.get("members")
    require(isinstance(members, list), "classified member vector missing")
    require(section.get("file_count") == len(members), "classified file count mismatch")
    require(
        section.get("plain_text_candidate_bytes_after_exact_duplicate_collapse", 0)
        <= section.get("plain_text_candidate_bytes_before_exact_duplicate_collapse", -1),
        "candidate duplicate accounting invalid",
    )
    for item in members:
        require(isinstance(item, dict), "classified member entry invalid")
        require(item.get("text_emitted") is False, "classified member text emission")
        require("text" not in item and "preview" not in item, "forbidden member text field")

    boundary = report.get("claim_boundary")
    require(isinstance(boundary, dict), "report claim boundary missing")
    require(boundary.get("plain_text_member_classification_complete") is True, "classification not terminal")
    for key in (
        "period_provenance_stratification_complete",
        "member_rights_terminal",
        "member_provenance_terminal",
        "language_quality_privacy_complete",
        "global_lineage_dedup_complete",
        "evaluation_decontamination_complete",
        "family_independence_terminal",
        "tokenizer_fit_authorized",
        "model_training_executed",
        "paid_compute_used",
        "research_corpus_v1_released",
    ):
        require(boundary.get(key) is False, f"report truth boundary weakened: {key}")
    require(boundary.get("training_authorized_bytes") == 0, "report granted training bytes")
    require(
        boundary.get("unique_causal_loss_positions_authorized") == 0,
        "report granted causal-loss positions",
    )
    require(boundary.get("optimizer_updates") == 0, "report claimed optimizer updates")


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("archive", type=Path)
    run.add_argument("--parent-report", type=Path, required=True)
    run.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data/d03_rada_trees_member_classification_v1.json"),
    )
    run.add_argument("--seven-zip", default="7z")
    run.add_argument("--output", type=Path, required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    try:
        if args.command == "verify":
            verify_report(load_object(args.report))
            print("D03 RADA_TREES MEMBER CLASSIFICATION PASS")
            return 0
        report = build_report(
            args.archive,
            load_object(args.parent_report),
            load_object(args.config),
            args.seven_zip,
        )
        verify_report(report)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(report["claim_boundary"]["safe_result"])
        print("REPORT_SHA256=" + report["report_sha256"])
        print("TRAINING_AUTHORIZED_BYTES=0")
        return 0
    except (ClassificationError, intake.IntakeError) as exc:
        print(f"BLOCKED: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
