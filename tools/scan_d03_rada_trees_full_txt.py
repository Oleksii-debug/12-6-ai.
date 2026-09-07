#!/usr/bin/env python3
"""Classify every .txt member of the pinned Rada_Trees secondary archive.

The scanner consumes the exact immutable archive identity established by PR #820,
revalidates the complete archive listing, extracts only the exact .txt member set
into an ephemeral workspace, classifies every selected member with the incumbent
Rada_Trees classifier, and emits text-free exact-duplicate-collapsed authority.

This is a source-capacity measurement only. It never grants training exposure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

import classify_d03_rada_trees_members as classify
import inventory_d03_rada_trees_archive as inventory
import probe_d03_rada_trees_secondary_role as parent_probe

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/data/d03_rada_trees_full_txt_scan_v1.json"
CLASSIFIER_CONFIG = ROOT / "configs/data/d03_rada_trees_member_classification_v1.json"
SCHEMA = "12-6.d03-rada-trees-full-txt-scan-report.v1"
SOURCE_SHA256 = "737fa5df061c55555a8c761023559a206cbcabf0299ae52d1ccae387f685803e"
LISTING_SHA256 = "9e5bad28b2455a0b1682b7c91d0f81f4113a928223898570c9800cf8af7ad15a"
ARCHIVE_SIZE = 697_768_591
ARCHIVE_MEMBER_COUNT = 8_782
ARCHIVE_UNCOMPRESSED_BYTES = 19_711_802_635
TXT_MEMBER_COUNT = 4_391


class FullTxtScanError(RuntimeError):
    """Fail-closed full .txt scan error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise FullTxtScanError(message)


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_config(path: Path = CONFIG) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(
        value.get("schema_version") == "12-6.d03-rada-trees-full-txt-scan.v1",
        "config schema drift",
    )
    require(value.get("execution_profile") == "LOCAL_FREE", "execution profile drift")

    parent = value.get("parent_scientific_authority")
    require(isinstance(parent, dict), "parent authority missing")
    require(parent.get("pr") == 820, "parent PR drift")
    require(
        parent.get("source_head_sha") == "66ad574e1fe8183efd88a592323e5fccce00bde0",
        "parent scientific head drift",
    )
    require(parent.get("workflow_run") == 34158830442, "parent run drift")
    require(parent.get("workflow_job") == 101856189735, "parent job drift")
    require(parent.get("conclusion") == "success", "parent scientific run is not success")
    require(
        parent.get("terminal_role_evidence_identity_sha256")
        == "3c4a9446c9a737bf9b4653d703296505f797d54195b5b466f5339c5e7e84ca8e",
        "parent evidence identity drift",
    )

    source = value.get("source")
    require(isinstance(source, dict), "source binding missing")
    require(source.get("dataset") == "uacorpus/Rada_Trees", "dataset drift")
    require(
        source.get("dataset_revision") == "1b994a5804dcda122721e8d33a03fd172cf8d867",
        "dataset revision drift",
    )
    require(source.get("archive_path") == "rada_xtag_texts.7z", "archive path drift")
    require(source.get("expected_size_bytes") == ARCHIVE_SIZE, "archive size drift")
    require(source.get("content_sha256") == SOURCE_SHA256, "archive SHA drift")
    require(
        source.get("git_blob_oid") == "2ddd106a0140b6980c9a6152406f40f1f9bd5c30",
        "Git blob drift",
    )
    require(
        source.get("xet_hash")
        == "46c56a953551f3d33800c8b50fbd355104197abec45c366f245103e830b17d30",
        "Xet hash drift",
    )

    terminal = value.get("terminal_listing")
    require(isinstance(terminal, dict), "terminal listing binding missing")
    require(terminal.get("listing_identity_sha256") == LISTING_SHA256, "listing SHA drift")
    require(terminal.get("member_count") == ARCHIVE_MEMBER_COUNT, "member count drift")
    require(
        terminal.get("uncompressed_bytes") == ARCHIVE_UNCOMPRESSED_BYTES,
        "uncompressed byte count drift",
    )
    require(terminal.get("plain_text_suffix") == ".txt", "plain-text suffix drift")
    require(
        terminal.get("plain_text_suffix_member_count") == TXT_MEMBER_COUNT,
        "plain-text member count drift",
    )

    policy = value.get("scan_policy")
    require(isinstance(policy, dict), "scan policy missing")
    require(policy.get("accepted_extractors") == ["7zz", "7z"], "extractor policy drift")
    require(policy.get("max_single_member_bytes") == 50_000_000, "member bound drift")
    require(
        policy.get("max_selected_total_uncompressed_bytes") == 5_000_000_000,
        "selected-byte envelope drift",
    )
    require(policy.get("require_exact_listing_identity") is True, "listing gate weakened")
    require(policy.get("require_exact_selected_member_set") is True, "selection gate weakened")
    require(policy.get("reject_symlinks") is True, "symlink gate weakened")
    require(policy.get("reject_special_files") is True, "special-file gate weakened")
    require(policy.get("reject_duplicate_normalized_paths") is True, "path gate weakened")
    require(policy.get("exact_duplicate_key") == "sha256", "duplicate key drift")
    require(
        policy.get("exact_duplicate_survivor") == "LEXICOGRAPHIC_PATH_ASC_V1",
        "duplicate survivor policy drift",
    )
    require(policy.get("raw_text_persisted") is False, "raw-text boundary weakened")

    boundary = value.get("claim_boundary")
    require(isinstance(boundary, dict), "claim boundary missing")
    for key in (
        "training_authorized_bytes",
        "unique_causal_loss_positions_authorized",
        "optimizer_updates",
    ):
        require(boundary.get(key) == 0, f"claim boundary weakened: {key}")
    for key in (
        "tokenizer_fit_authorized",
        "model_training_executed",
        "final_test_payload_accessed",
        "paid_compute_used",
        "training_admission_claimed",
    ):
        require(boundary.get(key) is False, f"claim boundary weakened: {key}")
    return value


def _listing_identity(config: dict[str, Any], listing: list[dict[str, Any]]) -> str:
    source = config["source"]
    payload = {
        "dataset_head_sha": source["dataset_revision"],
        "archive_path": source["archive_path"],
        "upstream_object_identity": source["xet_hash"],
        "archive_sha256": source["content_sha256"],
        "archive_size_bytes": source["expected_size_bytes"],
        "members": listing,
    }
    return inventory.sha256_bytes(inventory.canonical_json(payload))


def validate_listing(
    config: dict[str, Any], listing: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    terminal = config["terminal_listing"]
    policy = config["scan_policy"]
    require(len(listing) == int(terminal["member_count"]), "archive member count drift")
    observed_total = sum(int(item["size_bytes"]) for item in listing)
    require(
        observed_total == int(terminal["uncompressed_bytes"]),
        "archive uncompressed byte count drift",
    )
    require(
        _listing_identity(config, listing) == terminal["listing_identity_sha256"],
        "archive listing identity drift",
    )

    suffix = str(terminal["plain_text_suffix"])
    selected = [item for item in listing if Path(item["path"]).suffix.lower() == suffix]
    require(
        len(selected) == int(terminal["plain_text_suffix_member_count"]),
        "exact .txt member set count drift",
    )
    selected_total = inventory.validate_bounds(
        selected,
        int(policy["max_single_member_bytes"]),
        int(policy["max_selected_total_uncompressed_bytes"]),
    )
    require(selected_total > 0, "selected .txt member set is empty")
    return selected


def _verify_archive(config: dict[str, Any], archive: Path) -> None:
    source = config["source"]
    require(archive.is_file(), "archive input is not a regular file")
    require(archive.stat().st_size == source["expected_size_bytes"], "archive byte size drift")
    require(inventory.sha256_file(archive) == source["content_sha256"], "archive content SHA drift")


def _write_member_list(path: Path, selected: list[dict[str, Any]]) -> None:
    names: list[str] = []
    for item in selected:
        name = inventory.canonical_member_path(str(item["path"]))
        require(name.endswith(".txt"), f"non-.txt path reached extractor: {name}")
        require("\n" not in name and "\r" not in name, "newline in member path")
        names.append(name)
    require(len(names) == len(set(names)), "duplicate selected member path")
    path.write_text("\n".join(names) + "\n", encoding="utf-8")


def extract_selected_txt(
    extractor: str,
    archive: Path,
    destination: Path,
    selected: list[dict[str, Any]],
    policy: dict[str, Any],
    *,
    timeout_seconds: int = 7200,
) -> list[dict[str, Any]]:
    """Extract exactly the selected .txt paths in one 7z invocation."""
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix="rada-txt-members-",
        suffix=".list",
        delete=False,
    ) as handle:
        list_path = Path(handle.name)
    try:
        _write_member_list(list_path, selected)
        proc = subprocess.run(
            [
                extractor,
                "x",
                "-y",
                "-bd",
                "-scsUTF-8",
                f"-o{destination}",
                str(archive),
                f"@{list_path}",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        if proc.returncode != 0:
            raise FullTxtScanError(
                "selective .txt extraction failed "
                f"rc={proc.returncode}: {(proc.stderr or proc.stdout)[-1200:]}"
            )
    finally:
        list_path.unlink(missing_ok=True)

    return inventory.inventory_extracted_tree(
        destination,
        selected,
        int(policy["max_single_member_bytes"]),
        int(policy["max_selected_total_uncompressed_bytes"]),
    )


def _text_free_member_metadata(classified: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for item in classified:
        result.append(
            {
                "path": item["path"],
                "size_bytes": int(item["size_bytes"]),
                "sha256": item["sha256"],
                "classification": item["classification"],
                "decoded_encoding": item["decoded_encoding"],
                "text_metrics": item["text_metrics"],
                "path_year_hints": item["path_year_hints"],
            }
        )
    result.sort(key=lambda item: str(item["path"]))
    return result


def summarize_classification(classified: list[dict[str, Any]]) -> dict[str, Any]:
    require(bool(classified), "classification result is empty")
    member_metadata = _text_free_member_metadata(classified)
    require(len(member_metadata) == len(classified), "member metadata coverage drift")
    require(
        len({str(item["path"]) for item in member_metadata}) == len(member_metadata),
        "classified member path is not unique",
    )

    class_counts = Counter(str(item["classification"]) for item in member_metadata)
    class_bytes: Counter[str] = Counter()
    for item in member_metadata:
        class_bytes[str(item["classification"])] += int(item["size_bytes"])

    candidates = [
        item for item in member_metadata if item["classification"] == "PLAIN_TEXT_CANDIDATE"
    ]
    by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in candidates:
        by_hash[str(item["sha256"])].append(item)

    survivors: list[dict[str, Any]] = []
    duplicate_groups: list[dict[str, Any]] = []
    for payload_sha256 in sorted(by_hash):
        group = sorted(by_hash[payload_sha256], key=lambda item: str(item["path"]))
        survivor = group[0]
        survivors.append(
            {
                "path": survivor["path"],
                "size_bytes": int(survivor["size_bytes"]),
                "sha256": payload_sha256,
                "decoded_encoding": survivor["decoded_encoding"],
                "text_metrics": survivor["text_metrics"],
                "path_year_hints": survivor["path_year_hints"],
            }
        )
        if len(group) > 1:
            duplicate_groups.append(
                {
                    "sha256": payload_sha256,
                    "size_bytes": int(survivor["size_bytes"]),
                    "survivor_path": survivor["path"],
                    "duplicate_paths": [item["path"] for item in group[1:]],
                    "member_count": len(group),
                }
            )

    before = sum(int(item["size_bytes"]) for item in candidates)
    after = sum(int(item["size_bytes"]) for item in survivors)
    require(before >= after, "exact duplicate collapse increased candidate capacity")
    return {
        "full_txt_member_count": len(member_metadata),
        "full_txt_bytes": sum(int(item["size_bytes"]) for item in member_metadata),
        "full_txt_member_inventory_sha256": canonical_sha256(member_metadata),
        "full_txt_member_metadata": member_metadata,
        "class_counts": dict(sorted(class_counts.items())),
        "class_bytes": dict(sorted(class_bytes.items())),
        "plain_text_candidate_members_before_exact_duplicate_collapse": len(candidates),
        "plain_text_candidate_bytes_before_exact_duplicate_collapse": before,
        "plain_text_candidate_members_after_exact_duplicate_collapse": len(survivors),
        "plain_text_candidate_bytes_after_exact_duplicate_collapse": after,
        "exact_duplicate_discount_bytes": before - after,
        "exact_duplicate_group_count": len(duplicate_groups),
        "candidate_survivors": survivors,
        "exact_duplicate_groups": duplicate_groups,
        "raw_member_text_emitted": False,
    }


def build_report(config: dict[str, Any], archive: Path) -> dict[str, Any]:
    _verify_archive(config, archive)
    policy = config["scan_policy"]
    extractor = inventory.find_extractor(None, list(policy["accepted_extractors"]))
    listing = inventory.list_archive(extractor, archive)
    selected = validate_listing(config, listing)

    classifier_config = json.loads(CLASSIFIER_CONFIG.read_text(encoding="utf-8"))
    classify.validate_config(classifier_config)
    classifier_policy = classifier_config["classification_policy"]

    with tempfile.TemporaryDirectory(prefix="rada-full-txt-") as tmp:
        extracted_root = Path(tmp)
        extracted = extract_selected_txt(
            extractor,
            archive,
            extracted_root,
            selected,
            policy,
        )
        classified = parent_probe.classify_extracted(
            extracted_root,
            extracted,
            classifier_policy,
        )
    require(len(classified) == len(selected), "classified .txt coverage drift")
    classification = summarize_classification(classified)

    runtime = {
        "python_implementation": sys.implementation.name,
        "python_version": ".".join(str(part) for part in sys.version_info[:3]),
        "extractor_command": extractor,
        "extractor_version": inventory.extractor_version(extractor),
    }
    runtime["runtime_identity_sha256"] = canonical_sha256(runtime)

    core: dict[str, Any] = {
        "schema_version": SCHEMA,
        "worker_id": config["worker_id"],
        "execution_profile": "LOCAL_FREE",
        "parent_scientific_authority": config["parent_scientific_authority"],
        "source": config["source"],
        "terminal_listing": config["terminal_listing"],
        "runtime": runtime,
        "classification": classification,
        "decision": {
            "status": "FULL_TXT_CLASSIFICATION_COMPLETE_ZERO_CREDIT",
            "exact_duplicate_collapse_complete": True,
            "training_admission_claimed": False,
            "next_required": [
                "period_session_provenance",
                "rights_scope_revalidation",
                "language_quality_privacy",
                "global_exact_near_lineage_dedup",
                "reserved_evaluation_decontamination",
                "family_cap_mix_recompute",
            ],
        },
        "claim_boundary": config["claim_boundary"],
    }
    return {**core, "report_sha256": canonical_sha256(core)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()
    config = load_config()
    try:
        observed = parent_probe.acquire(config, args.archive_output, args.timeout)
        require(observed == config["source"]["content_sha256"], "downloaded archive SHA drift")
        report = build_report(config, args.archive_output)
        args.report_output.parent.mkdir(parents=True, exist_ok=True)
        args.report_output.write_text(
            json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        print("D03_RADA_TREES_FULL_TXT_SCAN=PASS_ZERO_CREDIT")
        print(
            "CANDIDATE_BYTES_AFTER_EXACT_DUP="
            + str(
                report["classification"][
                    "plain_text_candidate_bytes_after_exact_duplicate_collapse"
                ]
            )
        )
        print("REPORT_SHA256=" + report["report_sha256"])
        return 0
    except (
        FullTxtScanError,
        parent_probe.SecondaryProbeError,
        ValueError,
        RuntimeError,
        OSError,
    ) as exc:
        print(f"BLOCKED: {exc}")
        return 2
    finally:
        args.archive_output.unlink(missing_ok=True)
        args.archive_output.with_suffix(args.archive_output.suffix + ".partial").unlink(
            missing_ok=True
        )


if __name__ == "__main__":
    raise SystemExit(main())
