#!/usr/bin/env python3
"""Fail-closed Rada_Trees plain-text member classifier for canonical PR #697."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import inventory_d03_rada_trees_archive as inventory

CONFIG_SCHEMA = "12-6.d03-rada-trees-member-classification.v1"
REPORT_SCHEMA = "12-6.d03-rada-trees-member-classification-report.v1"
PARENT_REPORT_SCHEMA = "12-6.d03-rada-trees-archive-inventory-report.v1"
DATASET = "uacorpus/Rada_Trees"
DATASET_HEAD = "1b994a5804dcda122721e8d33a03fd172cf8d867"
PARENT_HEAD = "a42bcfa2f87c91e3d302cdf97694b5f324bc793f"
PRIMARY_ARCHIVE = "Rada_Trees.7z"
PINNED_SHA256 = "5e53939cd255276c58190569aebfaa6c90fb085fb10063e3e5f661747749719d"
PINNED_XET = "a31d24710d417246fb7e48028baaf6b9efb9a199983d78264f7997c69e42a801"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
YEAR = re.compile(r"(?<!\d)(199\d|20(?:0\d|1\d|2[0-4]))(?!\d)")


class ClassificationError(RuntimeError):
    """Fail-closed classification/provenance error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ClassificationError(message)


def canonical_sha256(value: dict[str, Any]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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
        config.get("worker_id") == "D03-RADA-TREES-MEMBER-CLASSIFICATION-20260907",
        "worker id drift",
    )
    require(config.get("execution_profile") == "LOCAL_FREE", "execution profile weakened")

    parent = config.get("parent")
    require(isinstance(parent, dict), "parent binding missing")
    require(parent.get("pr") == 697, "parent PR drift")
    require(parent.get("inventory_implementation_head_sha") == PARENT_HEAD, "parent head drift")
    require(parent.get("report_schema") == PARENT_REPORT_SCHEMA, "parent schema drift")
    require(parent.get("dataset") == DATASET, "dataset drift")
    require(parent.get("dataset_head_sha") == DATASET_HEAD, "dataset head drift")
    require(parent.get("archive_filename") == PRIMARY_ARCHIVE, "archive filename drift")
    require(parent.get("archive_sha256") == PINNED_SHA256, "archive SHA-256 drift")
    require(parent.get("archive_xet_hash") == PINNED_XET, "archive Xet drift")
    require(parent.get("member_content_hashes_required") is True, "member hash gate weakened")

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
    require(
        policy.get("strict_decode_order") == ["utf-8-sig", "windows-1251"],
        "decode policy drift",
    )
    require(policy.get("reject_nul") is True, "NUL gate weakened")
    require(policy.get("plain_text_requires_nonempty") is True, "empty-text gate weakened")
    require(policy.get("plain_text_max_tab_fraction") == 0.10, "tab threshold drift")
    require(policy.get("emit_member_text") is False, "member text emission forbidden")
    require(policy.get("emit_member_content_preview") is False, "preview emission forbidden")
    require(
        policy.get("exact_content_duplicates_collapsed_for_candidate_accounting") is True,
        "exact duplicate policy weakened",
    )

    rights = config.get("rights_and_lineage")
    require(isinstance(rights, dict), "rights/lineage policy missing")
    require(rights.get("dataset_card_license") == "CC-BY-4.0", "license binding drift")
    for key in (
        "attribution_required",
        "plain_text_original_transcripts_only_candidate",
        "parlamint_grac_overlap_requires_lineage_dedup",
        "period_provenance_stratification_required",
        "member_classification_is_not_rights_admission",
        "member_classification_is_not_family_independence_authority",
    ):
        require(rights.get(key) is True, f"rights/lineage gate weakened: {key}")

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
        "causal-loss positions must remain zero",
    )
    require(boundary.get("optimizer_updates") == 0, "optimizer updates must remain zero")


def verify_parent_report(report: dict[str, Any]) -> list[dict[str, Any]]:
    require(report.get("schema_version") == PARENT_REPORT_SCHEMA, "parent report schema mismatch")
    require(report.get("dataset_head_sha") == DATASET_HEAD, "parent dataset head mismatch")
    require(
        report.get("parent_probe_head_sha") == "92c1fd05d4399b0f0c4a35f0689160383f963c9c",
        "probe head drift",
    )
    require(
        report.get("state")
        == "EXACT_ARCHIVE_AND_MEMBER_INVENTORY_MATERIALIZED_CLASSIFICATION_NOT_RUN",
        "parent state drift",
    )

    archive = report.get("archive")
    require(isinstance(archive, dict), "parent archive block missing")
    require(archive.get("path") == PRIMARY_ARCHIVE, "parent archive filename mismatch")
    require(archive.get("sha256") == PINNED_SHA256, "parent archive SHA-256 mismatch")
    require(archive.get("upstream_object_identity") == PINNED_XET, "parent Xet mismatch")
    require(
        isinstance(archive.get("size_bytes"), int) and archive["size_bytes"] > 0,
        "parent archive size invalid",
    )

    require(report.get("training_authorized_bytes") == 0, "parent training credit must remain zero")
    require(
        report.get("training_exposure_authorized") is False,
        "parent exposure boundary weakened",
    )
    require(report.get("tokenizer_fit_authorized") is False, "parent tokenizer boundary weakened")
    require(report.get("model_training_executed") is False, "parent model boundary weakened")
    require(report.get("optimizer_updates") == 0, "parent optimizer boundary weakened")
    require(report.get("paid_compute_used") is False, "parent compute boundary weakened")
    require(
        report.get("member_payload_classification") == "NOT_RUN_SUCCESSOR_REQUIRED",
        "parent already classified",
    )

    members = report.get("members")
    require(isinstance(members, list) and members, "parent member vector missing")
    require(report.get("member_count") == len(members), "parent member count mismatch")
    files: list[dict[str, Any]] = []
    total = 0
    seen: set[str] = set()
    for raw in members:
        require(isinstance(raw, dict), "parent member entry invalid")
        path = raw.get("path")
        require(isinstance(path, str), "parent member path missing")
        require(inventory.canonical_member_path(path) == path, f"noncanonical parent path: {path}")
        require(path not in seen, f"duplicate parent path: {path}")
        seen.add(path)
        size = raw.get("size_bytes")
        digest = raw.get("sha256")
        require(
            isinstance(size, int) and not isinstance(size, bool) and size >= 0,
            f"invalid member size: {path}",
        )
        require(
            isinstance(digest, str) and HEX64.fullmatch(digest),
            f"invalid member hash: {path}",
        )
        files.append({"path": path, "size_bytes": size, "sha256": digest})
        total += size
    require(report.get("uncompressed_bytes_observed") == total, "parent byte total mismatch")

    payload = {
        "dataset_head_sha": DATASET_HEAD,
        "archive_path": PRIMARY_ARCHIVE,
        "upstream_object_identity": PINNED_XET,
        "archive_sha256": PINNED_SHA256,
        "archive_size_bytes": archive["size_bytes"],
        "members": files,
    }
    expected_inventory_id = inventory.sha256_bytes(inventory.canonical_json(payload))
    require(
        report.get("inventory_identity_sha256") == expected_inventory_id,
        "parent inventory identity mismatch",
    )

    stable = dict(report)
    extractor = report.get("extractor")
    require(
        isinstance(extractor, dict) and isinstance(extractor.get("name"), str),
        "parent extractor missing",
    )
    stable["extractor"] = {"name": extractor["name"]}
    expected_report_id = inventory.sha256_bytes(inventory.canonical_json(stable))
    claimed = report.get("report_identity_sha256")
    stable.pop("report_identity_sha256", None)
    expected_report_id = inventory.sha256_bytes(inventory.canonical_json(stable))
    require(claimed == expected_report_id, "parent report identity mismatch")
    return sorted(files, key=lambda item: item["path"])


def decode_text(data: bytes, order: list[str]) -> tuple[str | None, str | None]:
    for encoding in order:
        try:
            return data.decode(encoding, errors="strict"), encoding
        except UnicodeDecodeError:
            continue
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
    return sorted({int(match.group(1)) for match in YEAR.finditer(path)})


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
        return {
            "class": "EMPTY_TEXT_HOLD",
            "encoding": encoding,
            "metrics": {"characters": len(text)},
        }
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
        proc = subprocess.run(
            [resolved, "x", "-y", "-bd", f"-o{root}", "--", str(archive)],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=3600,
        )
        if proc.returncode != 0:
            raise ClassificationError(
                f"7z extraction failed rc={proc.returncode}: {proc.stderr[-500:]}"
            )

        actual: dict[str, Path] = {}
        for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
            rel = inventory.canonical_member_path(path.relative_to(root).as_posix())
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode):
                raise ClassificationError(f"extracted symlink rejected: {rel}")
            if stat.S_ISDIR(mode):
                continue
            if not stat.S_ISREG(mode):
                raise ClassificationError(f"extracted special file rejected: {rel}")
            require(rel not in actual, f"duplicate extracted path: {rel}")
            actual[rel] = path
        require(set(actual) == set(expected), "extracted files differ from parent inventory")

        classified: list[dict[str, Any]] = []
        for path in sorted(expected):
            source = expected[path]
            disk = actual[path]
            data = disk.read_bytes()
            require(len(data) == source["size_bytes"], f"member size drift: {path}")
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
    require(archive.name == PRIMARY_ARCHIVE, "archive filename mismatch")
    require(
        not archive.is_symlink() and archive.is_file(),
        "archive must be a regular non-symlink file",
    )
    require(
        archive.stat().st_size == parent_report["archive"]["size_bytes"],
        "archive byte-count drift",
    )
    require(inventory.sha256_file(archive) == PINNED_SHA256, "archive SHA-256 drift")

    members = extract_verify_classify(archive, files, config["classification_policy"], executable)
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
    year_counts = Counter(year for item in candidates for year in item["path_year_hints"])

    core: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA,
        "worker_id": config["worker_id"],
        "execution_profile": "LOCAL_FREE",
        "dataset": DATASET,
        "dataset_head_sha": DATASET_HEAD,
        "parent": {
            "pr": 697,
            "inventory_implementation_head_sha": PARENT_HEAD,
            "inventory_identity_sha256": parent_report["inventory_identity_sha256"],
            "report_identity_sha256": parent_report["report_identity_sha256"],
            "archive_sha256": PINNED_SHA256,
            "archive_xet_hash": PINNED_XET,
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
                "MEMBERS_CLASSIFIED_PLAIN_TEXT_CANDIDATES_REQUIRE_PROVENANCE_QUALITY_AND_LINEAGE_DEDUP"
            ),
        },
        "raw_member_text_emitted": False,
    }
    return {**core, "report_sha256": canonical_sha256(core)}


def verify_report(report: dict[str, Any]) -> None:
    require(report.get("schema_version") == REPORT_SCHEMA, "report schema mismatch")
    identity = report.get("report_sha256")
    require(isinstance(identity, str) and HEX64.fullmatch(identity), "report self-hash invalid")
    body = dict(report)
    del body["report_sha256"]
    require(canonical_sha256(body) == identity, "report self-hash mismatch")
    require(report.get("execution_profile") == "LOCAL_FREE", "report execution profile drift")
    require(report.get("dataset") == DATASET, "report dataset drift")
    require(report.get("dataset_head_sha") == DATASET_HEAD, "report dataset head drift")
    require(report.get("raw_member_text_emitted") is False, "raw text emission forbidden")
    section = report.get("classification")
    require(isinstance(section, dict), "classification section missing")
    members = section.get("members")
    require(isinstance(members, list), "classified member vector missing")
    require(section.get("file_count") == len(members), "classified file count mismatch")
    for item in members:
        require(item.get("text_emitted") is False, "classified member text emission")
        require("text" not in item and "preview" not in item, "forbidden member text field")
    boundary = report.get("claim_boundary")
    require(isinstance(boundary, dict), "report claim boundary missing")
    require(
        boundary.get("plain_text_member_classification_complete") is True,
        "classification not terminal",
    )
    require(boundary.get("training_authorized_bytes") == 0, "report granted training bytes")
    require(
        boundary.get("unique_causal_loss_positions_authorized") == 0,
        "report granted loss positions",
    )
    require(boundary.get("tokenizer_fit_authorized") is False, "report granted tokenizer fit")
    require(boundary.get("model_training_executed") is False, "report claimed training")
    require(boundary.get("optimizer_updates") == 0, "report claimed optimizer updates")
    require(boundary.get("paid_compute_used") is False, "report claimed paid compute")


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
    except (ClassificationError, ValueError, RuntimeError) as exc:
        print(f"BLOCKED: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
