#!/usr/bin/env python3
"""Classify hashed Rada_Trees archive members without granting training credit.

The classifier consumes the exact hashed-member report produced by the stacked
D03 archive-intake authority, re-verifies the archive and every extracted member,
and separates original plain-text *candidates* from annotation/metadata/unknown
members. Classification is intentionally not rights, quality, dedup, evaluation,
or training authorization.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import materialize_d03_rada_trees_archive as intake

CONFIG_SCHEMA = "12-6.d03-rada-trees-member-classification.v1"
REPORT_SCHEMA = "12-6.d03-rada-trees-member-classification-report.v1"
PARENT_REPORT_SCHEMA = "12-6.d03-rada-trees-archive-intake.v1"
DATASET = "uacorpus/Rada_Trees"
DATASET_HEAD = "1b994a5804dcda122721e8d33a03fd172cf8d867"
PARENT_HEAD = "e74afbd4a9883dab348c8698a748dc9003b79192"
PRIMARY_ARCHIVE = "Rada_Trees.7z"


class ClassificationError(RuntimeError):
    """Fail-closed member-classification error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ClassificationError(message)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"{path}: JSON root must be an object")
    return value


def validate_config(config: dict[str, Any]) -> None:
    require(config.get("schema_version") == CONFIG_SCHEMA, "classification config schema drift")
    require(
        config.get("worker_id") == "D03-RADA-TREES-MEMBER-CLASSIFICATION-20260826",
        "worker id drift",
    )
    require(config.get("execution_profile") == "LOCAL_FREE", "execution profile weakened")

    parent = config.get("parent")
    require(isinstance(parent, dict), "parent config missing")
    require(parent.get("pr") == 708, "parent PR drift")
    require(parent.get("head_sha") == PARENT_HEAD, "parent head drift")
    require(parent.get("report_schema") == PARENT_REPORT_SCHEMA, "parent report schema drift")
    require(parent.get("dataset") == DATASET, "dataset drift")
    require(parent.get("dataset_head") == DATASET_HEAD, "dataset head drift")
    require(parent.get("archive_filename") == PRIMARY_ARCHIVE, "archive filename drift")
    require(parent.get("member_content_hashes_required") is True, "member hash requirement weakened")

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
    require(policy.get("reject_nul") is True, "NUL rejection weakened")
    require(policy.get("conllu_min_noncomment_rows") == 3, "CoNLL-U row threshold drift")
    require(policy.get("conllu_required_columns") == 10, "CoNLL-U column contract drift")
    require(policy.get("plain_text_requires_nonempty") is True, "empty-text policy weakened")
    require(policy.get("plain_text_max_tab_fraction") == 0.10, "tabular threshold drift")
    require(policy.get("emit_member_text") is False, "member text emission forbidden")
    require(policy.get("emit_member_content_preview") is False, "content preview emission forbidden")
    require(
        policy.get("exact_content_duplicates_collapsed_for_candidate_accounting") is True,
        "exact duplicate collapse weakened",
    )

    rights = config.get("rights_and_lineage")
    require(isinstance(rights, dict), "rights/lineage policy missing")
    require(rights.get("dataset_card_license") == "CC-BY-4.0", "license discovery binding drift")
    require(rights.get("attribution_required") is True, "attribution requirement weakened")
    require(rights.get("plain_text_original_transcripts_only_candidate") is True, "plain-text scope weakened")
    require(rights.get("ud_annotation_default") == "HOLD_ZERO_CREDIT", "UD hold boundary weakened")
    require(rights.get("nlp_uk_annotation_default") == "HOLD_ZERO_CREDIT", "nlp_uk hold boundary weakened")
    require(rights.get("parlamint_grac_overlap_requires_lineage_dedup") is True, "lineage gate weakened")
    require(rights.get("member_classification_is_not_rights_admission") is True, "rights boundary weakened")
    require(
        rights.get("member_classification_is_not_family_independence_authority") is True,
        "family boundary weakened",
    )

    expected_downstream = [
        "BIND_MEMBER_LEVEL_ATTRIBUTION_AND_SOURCE_PROVENANCE",
        "RUN_UKRAINIAN_LANGUAGE_QUALITY_PRIVACY_FILTERS",
        "RUN_EXACT_AND_NEAR_LINEAGE_DEDUP_AGAINST_RADA_LAWS_PARLAMINT_GRAC_AND_LIVE_CORPUS",
        "RUN_EVALUATION_DECONTAMINATION",
        "RECOMPUTE_FAMILY_CAP_AND_UA_MIXTURE_FEASIBILITY",
        "ONLY_THEN_PROPOSE_NONZERO_SOURCE_CAPACITY_CREDIT",
    ]
    require(config.get("downstream_required") == expected_downstream, "downstream gate order drift")

    boundary = config.get("claim_boundary")
    require(isinstance(boundary, dict), "claim boundary missing")
    false_keys = [
        "plain_text_member_classification_complete",
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
    ]
    for key in false_keys:
        require(boundary.get(key) is False, f"claim boundary weakened: {key}")
    require(boundary.get("training_authorized_bytes") == 0, "training byte authority must remain zero")
    require(
        boundary.get("unique_causal_loss_positions_authorized") == 0,
        "unique-loss authority must remain zero",
    )
    require(boundary.get("optimizer_updates") == 0, "optimizer updates must remain zero")


def verify_parent_report(report: dict[str, Any]) -> list[dict[str, Any]]:
    require(report.get("schema_version") == PARENT_REPORT_SCHEMA, "parent report schema mismatch")
    require(report.get("dataset") == DATASET, "parent dataset mismatch")
    require(report.get("dataset_head") == DATASET_HEAD, "parent dataset head mismatch")

    expected_hash = report.get("report_sha256")
    core = dict(report)
    core.pop("report_sha256", None)
    require(
        isinstance(expected_hash, str) and expected_hash == sha256_bytes(canonical_json_bytes(core)),
        "parent report self-hash mismatch",
    )

    archive = report.get("archive")
    require(isinstance(archive, dict), "parent archive block missing")
    require(archive.get("filename") == PRIMARY_ARCHIVE, "parent archive filename mismatch")
    require(archive.get("sha256_verified") is True, "parent archive identity not verified")
    archive_hash = archive.get("sha256")
    require(
        isinstance(archive_hash, str)
        and len(archive_hash) == 64
        and all(ch in "0123456789abcdef" for ch in archive_hash),
        "parent archive SHA-256 invalid",
    )

    inventory = report.get("inventory")
    require(isinstance(inventory, dict), "parent inventory missing")
    require(inventory.get("member_hashes_complete") is True, "parent member hashes are not complete")
    members = inventory.get("members")
    require(isinstance(members, list) and members, "parent inventory has no members")
    files: list[dict[str, Any]] = []
    seen: set[str] = set()
    total = 0
    for raw in members:
        require(isinstance(raw, dict), "parent member entry must be an object")
        path = raw.get("path")
        require(isinstance(path, str), "parent member path missing")
        normalized = intake.normalize_member_path(path)
        require(normalized == path, f"parent member path is not canonical: {path}")
        require(path not in seen, f"duplicate parent member path: {path}")
        seen.add(path)
        is_directory = raw.get("is_directory")
        require(isinstance(is_directory, bool), f"parent directory flag invalid: {path}")
        size = raw.get("size")
        require(isinstance(size, int) and not isinstance(size, bool) and size >= 0, f"parent member size invalid: {path}")
        if is_directory:
            require(raw.get("sha256") is None, f"directory must not carry content hash: {path}")
            continue
        member_hash = raw.get("sha256")
        require(
            isinstance(member_hash, str)
            and len(member_hash) == 64
            and all(ch in "0123456789abcdef" for ch in member_hash),
            f"parent member SHA-256 invalid: {path}",
        )
        files.append({"path": path, "size": size, "sha256": member_hash})
        total += size
    require(len(files) == inventory.get("file_count"), "parent file-count mismatch")
    require(total == inventory.get("total_uncompressed_file_bytes"), "parent byte-total mismatch")
    return sorted(files, key=lambda item: item["path"])


def decode_text(data: bytes, order: list[str]) -> tuple[str | None, str | None]:
    for encoding in order:
        try:
            return data.decode(encoding, errors="strict"), encoding
        except UnicodeDecodeError:
            continue
    return None, None


def looks_like_conllu(text: str, *, min_rows: int, required_columns: int) -> bool:
    rows = 0
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if len(line.split("\t")) == required_columns:
            rows += 1
            if rows >= min_rows:
                return True
    return False


def classify_content(path: str, data: bytes, policy: dict[str, Any]) -> dict[str, Any]:
    lower_name = Path(path).name.lower()
    suffix = Path(path).suffix.lower()
    metadata_names = {str(item).lower() for item in policy["metadata_basenames"]}
    if lower_name in metadata_names:
        return {"class": "METADATA_HOLD", "encoding": None, "text_metrics": None}
    if b"\x00" in data and policy["reject_nul"]:
        return {"class": "BINARY_OR_NUL_HOLD", "encoding": None, "text_metrics": None}
    if suffix in set(policy["ud_derivative_suffixes"]):
        return {"class": "DERIVED_UD_HOLD", "encoding": None, "text_metrics": None}
    if suffix in set(policy["annotation_derivative_suffixes"]):
        return {"class": "DERIVED_ANNOTATION_HOLD", "encoding": None, "text_metrics": None}
    if suffix not in set(policy["plain_text_candidate_suffixes"]):
        return {"class": "UNKNOWN_FORMAT_HOLD", "encoding": None, "text_metrics": None}

    text, encoding = decode_text(data, list(policy["strict_decode_order"]))
    if text is None or encoding is None:
        return {"class": "UNDECODABLE_TEXT_HOLD", "encoding": None, "text_metrics": None}
    stripped = text.strip()
    if not stripped and policy["plain_text_requires_nonempty"]:
        return {
            "class": "EMPTY_TEXT_HOLD",
            "encoding": encoding,
            "text_metrics": {"characters": len(text), "lines": len(text.splitlines()), "tab_fraction": 0.0},
        }
    lower = stripped[:256].lower()
    if any(lower.startswith(prefix) for prefix in policy["markup_prefixes"]):
        return {"class": "MARKUP_ANNOTATION_HOLD", "encoding": encoding, "text_metrics": None}
    if looks_like_conllu(
        text,
        min_rows=int(policy["conllu_min_noncomment_rows"]),
        required_columns=int(policy["conllu_required_columns"]),
    ):
        return {"class": "DERIVED_UD_HOLD", "encoding": encoding, "text_metrics": None}

    lines = text.splitlines()
    nonempty = [line for line in lines if line.strip()]
    tabbed = sum("\t" in line for line in nonempty)
    tab_fraction = (tabbed / len(nonempty)) if nonempty else 0.0
    letters = [ch for ch in text if ch.isalpha()]
    cyrillic = sum("\u0400" <= ch <= "\u04ff" for ch in letters)
    ukrainian_specific = sum(ch.lower() in "іїєґ" for ch in letters)
    metrics = {
        "characters": len(text),
        "lines": len(lines),
        "nonempty_lines": len(nonempty),
        "tab_fraction": tab_fraction,
        "letters": len(letters),
        "cyrillic_letter_fraction": (cyrillic / len(letters)) if letters else 0.0,
        "ukrainian_specific_letter_count": ukrainian_specific,
    }
    if tab_fraction > float(policy["plain_text_max_tab_fraction"]):
        return {"class": "TABULAR_ANNOTATION_HOLD", "encoding": encoding, "text_metrics": metrics}
    return {"class": "PLAIN_TEXT_CANDIDATE", "encoding": encoding, "text_metrics": metrics}


def extract_verify_and_classify(
    archive: Path,
    files: list[dict[str, Any]],
    policy: dict[str, Any],
    *,
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
        require(set(actual) == set(expected), "extracted file set differs from hashed parent inventory")

        classified: list[dict[str, Any]] = []
        for path in sorted(expected):
            record = expected[path]
            disk = actual[path]
            require(disk.is_file(), f"extracted member is not a regular file: {path}")
            require(disk.stat().st_size == record["size"], f"extracted member size drift: {path}")
            data = disk.read_bytes()
            require(sha256_bytes(data) == record["sha256"], f"extracted member hash drift: {path}")
            decision = classify_content(path, data, policy)
            classified.append(
                {
                    "path": path,
                    "bytes": len(data),
                    "sha256": record["sha256"],
                    "classification": decision["class"],
                    "decoded_encoding": decision["encoding"],
                    "text_metrics": decision["text_metrics"],
                    "text_emitted": False,
                }
            )
        return classified


def build_report(
    archive: Path,
    parent_report: dict[str, Any],
    config: dict[str, Any],
    *,
    executable: str = "7z",
) -> dict[str, Any]:
    validate_config(config)
    files = verify_parent_report(parent_report)
    require(archive.name == PRIMARY_ARCHIVE and archive.is_file(), "expected existing Rada_Trees.7z")
    archive_block = parent_report["archive"]
    require(archive.stat().st_size == archive_block["compressed_bytes"], "archive byte-count drift")
    require(sha256_file(archive) == archive_block["sha256"], "archive SHA-256 drift")

    classified = extract_verify_and_classify(
        archive,
        files,
        config["classification_policy"],
        executable=executable,
    )
    class_counts = Counter(item["classification"] for item in classified)
    class_bytes: Counter[str] = Counter()
    for item in classified:
        class_bytes[item["classification"]] += int(item["bytes"])

    candidates = [item for item in classified if item["classification"] == "PLAIN_TEXT_CANDIDATE"]
    by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in candidates:
        by_hash[item["sha256"]].append(item)
    exact_duplicate_groups = [
        {
            "sha256": digest,
            "paths": sorted(item["path"] for item in group),
            "member_count": len(group),
            "bytes_per_member": group[0]["bytes"],
        }
        for digest, group in sorted(by_hash.items())
        if len(group) > 1
    ]
    unique_candidate_bytes = sum(group[0]["bytes"] for group in by_hash.values())

    config_sha = sha256_bytes(canonical_json_bytes(config))
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
            "archive_sha256": archive_block["sha256"],
        },
        "config_sha256": config_sha,
        "classification": {
            "file_count": len(classified),
            "class_counts": dict(sorted(class_counts.items())),
            "class_bytes": dict(sorted(class_bytes.items())),
            "plain_text_candidate_member_count": len(candidates),
            "plain_text_candidate_bytes_before_exact_duplicate_collapse": sum(
                int(item["bytes"]) for item in candidates
            ),
            "plain_text_candidate_bytes_after_exact_duplicate_collapse": unique_candidate_bytes,
            "exact_duplicate_group_count": len(exact_duplicate_groups),
            "exact_duplicate_groups": exact_duplicate_groups,
            "members": classified,
        },
        "interpretation": {
            "plain_text_candidate_is_training_admission": False,
            "candidate_bytes_are_training_capacity": False,
            "exact_duplicate_collapse_is_global_lineage_dedup": False,
            "dataset_card_reports_original_plain_text_plus_ud_plus_nlp_uk": True,
            "parlamint_grac_overlap_still_requires_cross_source_lineage_dedup": True,
        },
        "claim_boundary": {
            "plain_text_member_classification_complete": True,
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
            "safe_result": "MEMBERS_CLASSIFIED_PLAIN_TEXT_CANDIDATES_REQUIRE_PROVENANCE_QUALITY_DEDUP",
        },
        "raw_member_text_emitted": False,
    }
    return {**core, "report_sha256": sha256_bytes(canonical_json_bytes(core))}


def verify_report(report: dict[str, Any]) -> None:
    require(report.get("schema_version") == REPORT_SCHEMA, "classification report schema mismatch")
    expected_hash = report.get("report_sha256")
    core = dict(report)
    core.pop("report_sha256", None)
    require(
        isinstance(expected_hash, str) and expected_hash == sha256_bytes(canonical_json_bytes(core)),
        "classification report self-hash mismatch",
    )
    require(report.get("execution_profile") == "LOCAL_FREE", "classification report execution drift")
    require(report.get("dataset") == DATASET and report.get("dataset_head") == DATASET_HEAD, "classification report source drift")
    require(report.get("raw_member_text_emitted") is False, "raw member text emission forbidden")
    section = report.get("classification")
    require(isinstance(section, dict), "classification section missing")
    members = section.get("members")
    require(isinstance(members, list), "classified member vector missing")
    require(section.get("file_count") == len(members), "classified file-count mismatch")
    require(
        section.get("plain_text_candidate_bytes_after_exact_duplicate_collapse", 0)
        <= section.get("plain_text_candidate_bytes_before_exact_duplicate_collapse", -1),
        "exact-duplicate candidate accounting invalid",
    )
    for item in members:
        require(isinstance(item, dict), "classified member entry invalid")
        require(item.get("text_emitted") is False, "classified member leaked text")
        require("text" not in item and "preview" not in item, "classified member contains forbidden text field")
    boundary = report.get("claim_boundary")
    require(isinstance(boundary, dict), "classification claim boundary missing")
    require(boundary.get("plain_text_member_classification_complete") is True, "classification completion missing")
    for key in (
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
        require(boundary.get(key) is False, f"classification truth boundary weakened: {key}")
    require(boundary.get("training_authorized_bytes") == 0, "classification granted training bytes")
    require(boundary.get("unique_causal_loss_positions_authorized") == 0, "classification granted loss positions")
    require(boundary.get("optimizer_updates") == 0, "classification claimed optimizer updates")


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
        parent_report = load_object(args.parent_report)
        config = load_object(args.config)
        report = build_report(args.archive, parent_report, config, executable=args.seven_zip)
        verify_report(report)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(report["claim_boundary"]["safe_result"])
        print(report["report_sha256"])
        return 0
    except (ClassificationError, intake.IntakeError) as exc:
        print(f"BLOCKED: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
