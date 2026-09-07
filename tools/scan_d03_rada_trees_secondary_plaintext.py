#!/usr/bin/env python3
"""Selective full scan of the exact Rada_Trees secondary plain-text layer.

The terminal parent listing proves the archive contains a bounded .txt subset
inside a much larger derived XTAG layer. This successor extracts only that exact
.txt member vector, verifies every member, classifies every payload with the
incumbent classifier, and emits text-free evidence. It never grants training
or family credit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
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

import classify_d03_rada_trees_members as classify
import inventory_d03_rada_trees_archive as inventory
import probe_d03_rada_trees_secondary_role as parent_probe

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/data/d03_rada_trees_secondary_plaintext_full_scan_v1.json"
CLASSIFIER_CONFIG = ROOT / "configs/data/d03_rada_trees_member_classification_v1.json"
SCHEMA = "12-6.d03-rada-trees-secondary-plaintext-full-scan-report.v1"
SOURCE_SHA256 = "737fa5df061c55555a8c761023559a206cbcabf0299ae52d1ccae387f685803e"
LISTING_SHA256 = "9e5bad28b2455a0b1682b7c91d0f81f4113a928223898570c9800cf8af7ad15a"
ARCHIVE_BYTES = 697_768_591
MEMBER_COUNT = 8_782
UNCOMPRESSED_BYTES = 19_711_802_635
TXT_COUNT = 4_391
TXT_BYTES = 879_031_855
XTAG_COUNT = 4_391
XTAG_BYTES = 18_832_770_780


class PlaintextScanError(RuntimeError):
    """Fail-closed selective scan error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PlaintextScanError(message)


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path = CONFIG) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PlaintextScanError(f"cannot load scan config: {path}") from exc
    require(isinstance(value, dict), "config root must be an object")
    require(
        value.get("schema_version")
        == "12-6.d03-rada-trees-secondary-plaintext-full-scan.v1",
        "config schema drift",
    )
    require(value.get("execution_profile") == "LOCAL_FREE", "execution profile drift")

    parent = value.get("parent_authority")
    require(isinstance(parent, dict), "parent authority missing")
    expected_parent = {
        "pr": 820,
        "source_head_sha": "66ad574e1fe8183efd88a592323e5fccce00bde0",
        "scientific_run_id": 34158830442,
        "scientific_job_id": 101856189735,
        "artifact_id": 10032220704,
        "artifact_sha256": (
            "b9e67cd6ebcc8ab7e40a2eed6e19c2dda9bc5523d6ab3c617d2ce689dd9d196a"
        ),
        "report_sha256": (
            "4c24e90faabbcbeeae2cd7cc431c1bff0108d67c376d78b5838888424586e14c"
        ),
        "listing_identity_sha256": LISTING_SHA256,
    }
    require(parent == expected_parent, "parent authority drift")

    source = value.get("source")
    require(isinstance(source, dict), "source binding missing")
    require(source.get("dataset") == "uacorpus/Rada_Trees", "dataset drift")
    require(
        source.get("dataset_revision") == parent_probe.REVISION,
        "dataset revision drift",
    )
    require(source.get("archive_path") == parent_probe.ARCHIVE, "archive path drift")
    require(source.get("expected_size_bytes") == ARCHIVE_BYTES, "archive size drift")
    require(source.get("content_sha256") == SOURCE_SHA256, "archive SHA-256 drift")
    require(
        source.get("xet_hash")
        == "46c56a953551f3d33800c8b50fbd355104197abec45c366f245103e830b17d30",
        "archive Xet drift",
    )

    terminal = value.get("terminal_listing")
    require(isinstance(terminal, dict), "terminal listing binding missing")
    expected_terminal = {
        "member_count": MEMBER_COUNT,
        "uncompressed_bytes": UNCOMPRESSED_BYTES,
        "plain_text_suffix": ".txt",
        "plain_text_member_count": TXT_COUNT,
        "plain_text_listing_bytes": TXT_BYTES,
        "other_suffix": ".xtag",
        "other_member_count": XTAG_COUNT,
        "other_listing_bytes": XTAG_BYTES,
    }
    require(terminal == expected_terminal, "terminal listing facts drift")

    policy = value.get("selective_extract_policy")
    require(isinstance(policy, dict), "selective extraction policy missing")
    require(policy.get("required_path_prefix") == "texts/", "path prefix drift")
    require(policy.get("required_plain_suffix") == ".txt", "plain suffix drift")
    require(policy.get("max_single_member_bytes") == 50_000_000, "member bound drift")
    require(
        policy.get("max_selected_uncompressed_bytes") == 2_000_000_000,
        "selected extraction envelope drift",
    )
    for field in (
        "require_exact_parent_listing_identity",
        "require_exact_selected_member_count_and_bytes",
        "reject_non_regular_extracted_files",
        "hash_every_selected_member",
        "classify_every_selected_member",
        "collapse_exact_content_duplicates_for_candidate_accounting",
    ):
        require(policy.get(field) is True, f"policy weakened: {field}")
    require(policy.get("emit_member_text") is False, "raw member text emission forbidden")
    require(
        policy.get("emit_member_content_preview") is False,
        "member preview emission forbidden",
    )

    boundary = value.get("claim_boundary")
    require(isinstance(boundary, dict), "claim boundary missing")
    for field in (
        "training_authorized_bytes",
        "unique_causal_loss_positions_authorized",
        "family_credit_added",
        "optimizer_updates",
    ):
        require(boundary.get(field) == 0, f"zero boundary drift: {field}")
    for field in (
        "tokenizer_fit_authorized",
        "model_training_executed",
        "final_test_payload_accessed",
        "paid_compute_used",
        "research_corpus_v1_released",
    ):
        require(boundary.get(field) is False, f"false boundary drift: {field}")
    return value


def verify_archive(path: Path) -> str:
    require(not path.is_symlink() and path.is_file(), "archive must be a regular file")
    require(path.stat().st_size == ARCHIVE_BYTES, "archive byte count mismatch")
    digest = sha256_file(path)
    require(digest == SOURCE_SHA256, "archive SHA-256 mismatch")
    return digest


def listing_identity(listing: list[dict[str, Any]], config: dict[str, Any]) -> str:
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


def select_plaintext_listing(
    listing: list[dict[str, Any]], config: dict[str, Any]
) -> list[dict[str, Any]]:
    require(len(listing) == MEMBER_COUNT, "terminal member count mismatch")
    total = sum(int(item["size_bytes"]) for item in listing)
    require(total == UNCOMPRESSED_BYTES, "terminal uncompressed byte total mismatch")
    require(listing_identity(listing, config) == LISTING_SHA256, "terminal listing identity mismatch")

    suffix_counts: Counter[str] = Counter()
    suffix_bytes: Counter[str] = Counter()
    selected: list[dict[str, Any]] = []
    policy = config["selective_extract_policy"]
    for item in listing:
        path = item.get("path")
        size = item.get("size_bytes")
        require(isinstance(path, str), "listing path missing")
        require(inventory.canonical_member_path(path) == path, "listing path not canonical")
        require(
            isinstance(size, int) and not isinstance(size, bool) and size >= 0,
            "listing member size invalid",
        )
        suffix = Path(path).suffix.lower()
        suffix_counts[suffix] += 1
        suffix_bytes[suffix] += size
        if suffix == ".txt":
            require(path.startswith(policy["required_path_prefix"]), "plain path prefix mismatch")
            require(size <= policy["max_single_member_bytes"], "plain member exceeds size bound")
            selected.append({"path": path, "size_bytes": size})

    require(set(suffix_counts) == {".txt", ".xtag"}, "unexpected suffix in terminal archive")
    require(suffix_counts[".txt"] == TXT_COUNT, "plain member count mismatch")
    require(suffix_bytes[".txt"] == TXT_BYTES, "plain listing bytes mismatch")
    require(suffix_counts[".xtag"] == XTAG_COUNT, "XTAG member count mismatch")
    require(suffix_bytes[".xtag"] == XTAG_BYTES, "XTAG listing bytes mismatch")
    require(
        sum(item["size_bytes"] for item in selected)
        <= policy["max_selected_uncompressed_bytes"],
        "selected plain layer exceeds extraction envelope",
    )
    return selected


def extract_selected(
    extractor: str,
    archive: Path,
    selected: list[dict[str, Any]],
    destination: Path,
) -> None:
    resolved = shutil.which(extractor)
    if resolved is None:
        raise PlaintextScanError(f"required extractor unavailable: {extractor}")
    destination.mkdir(parents=True, exist_ok=False)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        prefix="rada-plaintext-members-",
        suffix=".txt",
        delete=False,
    ) as handle:
        list_path = Path(handle.name)
        for item in selected:
            handle.write(item["path"] + "\n")
    try:
        proc = subprocess.run(
            [
                resolved,
                "x",
                "-y",
                "-bd",
                "-scsUTF-8",
                f"-o{destination}",
                "--",
                str(archive),
                f"@{list_path}",
            ],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=3600,
        )
    finally:
        list_path.unlink(missing_ok=True)
    if proc.returncode != 0:
        raise PlaintextScanError(
            f"selective 7z extraction failed rc={proc.returncode}: {proc.stderr[-1000:]}"
        )


def classify_selected_tree(
    root: Path,
    selected: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    policy = config["selective_extract_policy"]
    members = inventory.inventory_extracted_tree(
        root,
        selected,
        int(policy["max_single_member_bytes"]),
        int(policy["max_selected_uncompressed_bytes"]),
    )
    classifier_config = json.loads(CLASSIFIER_CONFIG.read_text(encoding="utf-8"))
    classify.validate_config(classifier_config)
    classifier_policy = classifier_config["classification_policy"]

    result: list[dict[str, Any]] = []
    for item in members:
        disk = root / Path(item["path"])
        mode = disk.lstat().st_mode
        require(stat.S_ISREG(mode), f"non-regular extracted plain member: {item['path']}")
        payload = disk.read_bytes()
        require(len(payload) == item["size_bytes"], f"member size drift: {item['path']}")
        digest = hashlib.sha256(payload).hexdigest()
        require(digest == item["sha256"], f"member hash drift: {item['path']}")
        decision = classify.classify_content(item["path"], payload, classifier_policy)
        result.append(
            {
                "path": item["path"],
                "size_bytes": item["size_bytes"],
                "sha256": digest,
                "classification": decision["class"],
                "decoded_encoding": decision["encoding"],
                "text_metrics": decision["metrics"],
                "path_year_hints": classify.year_hints(item["path"]),
                "text_emitted": False,
            }
        )
    return result


def build_report(
    config: dict[str, Any],
    archive: Path,
    extractor: str,
) -> dict[str, Any]:
    content_sha256 = verify_archive(archive)
    listing = inventory.list_archive(extractor, archive)
    selected = select_plaintext_listing(listing, config)

    with tempfile.TemporaryDirectory(prefix="rada-secondary-plaintext-") as tmp:
        root = Path(tmp) / "selected"
        extract_selected(extractor, archive, selected, root)
        classified = classify_selected_tree(root, selected, config)

    class_counts = Counter(item["classification"] for item in classified)
    class_bytes: Counter[str] = Counter()
    for item in classified:
        class_bytes[item["classification"]] += int(item["size_bytes"])

    candidates = [
        item for item in classified if item["classification"] == "PLAIN_TEXT_CANDIDATE"
    ]
    by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in candidates:
        by_hash[item["sha256"]].append(item)
    unique_candidate_bytes = sum(group[0]["size_bytes"] for group in by_hash.values())
    duplicate_observations = sum(max(0, len(group) - 1) for group in by_hash.values())

    report: dict[str, Any] = {
        "schema_version": SCHEMA,
        "state": "FULL_PLAINTEXT_LAYER_CLASSIFIED_ZERO_CREDIT",
        "parent_report_sha256": config["parent_authority"]["report_sha256"],
        "parent_listing_identity_sha256": LISTING_SHA256,
        "source": {
            "dataset": config["source"]["dataset"],
            "dataset_revision": config["source"]["dataset_revision"],
            "archive_path": config["source"]["archive_path"],
            "archive_bytes": ARCHIVE_BYTES,
            "archive_sha256": content_sha256,
            "xet_hash": config["source"]["xet_hash"],
        },
        "extractor": {
            "command": extractor,
            "version": inventory.extractor_version(extractor),
        },
        "terminal_listing": {
            "member_count": len(listing),
            "uncompressed_bytes": UNCOMPRESSED_BYTES,
            "listing_identity_sha256": listing_identity(listing, config),
        },
        "selected_plaintext": {
            "member_count": len(selected),
            "listing_bytes": sum(item["size_bytes"] for item in selected),
            "classification_complete": len(classified) == len(selected),
            "class_counts": dict(sorted(class_counts.items())),
            "class_bytes": dict(sorted(class_bytes.items())),
            "plain_text_candidate_members_before_exact_dedup": len(candidates),
            "plain_text_candidate_bytes_before_exact_dedup": sum(
                int(item["size_bytes"]) for item in candidates
            ),
            "plain_text_candidate_unique_sha256_groups": len(by_hash),
            "plain_text_candidate_duplicate_observations": duplicate_observations,
            "plain_text_candidate_bytes_after_exact_dedup": unique_candidate_bytes,
            "members": classified,
            "raw_member_text_emitted": False,
        },
        "gates": {
            "exact_archive_identity": "PASS",
            "exact_parent_listing_identity": "PASS",
            "selective_plaintext_extraction": "PASS",
            "full_plaintext_member_classification": "PASS",
            "exact_content_duplicate_collapse": "PASS",
            "rights_and_member_provenance": "NOT_RUN",
            "language_quality_privacy": "NOT_RUN",
            "global_lineage_dedup": "NOT_RUN",
            "evaluation_decontamination": "NOT_RUN",
            "family_independence": "NOT_RUN",
        },
        "claim_boundary": dict(config["claim_boundary"]),
    }
    report["report_sha256"] = hashlib.sha256(canonical_bytes(report)).hexdigest()
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--extractor")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config()
    if args.download:
        args.archive.parent.mkdir(parents=True, exist_ok=True)
        args.archive.unlink(missing_ok=True)
        observed = parent_probe.acquire(parent_probe.load_config(), args.archive, args.timeout)
        require(observed == SOURCE_SHA256, "downloaded source SHA-256 drift")
    extractor = inventory.find_extractor(
        args.extractor,
        list(parent_probe.load_config()["inventory_policy"]["accepted_extractors"]),
    )
    report = build_report(config, args.archive, extractor)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    tmp.write_bytes(canonical_bytes(report))
    os.replace(tmp, args.output)
    print(
        json.dumps(
            {
                "state": report["state"],
                "report_sha256": report["report_sha256"],
                "plain_text_candidates": report["selected_plaintext"][
                    "plain_text_candidate_members_before_exact_dedup"
                ],
                "unique_candidate_bytes": report["selected_plaintext"][
                    "plain_text_candidate_bytes_after_exact_dedup"
                ],
                "training_authorized_bytes": 0,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
