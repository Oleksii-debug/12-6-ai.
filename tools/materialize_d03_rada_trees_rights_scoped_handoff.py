#!/usr/bin/env python3
"""Materialize rights-scoped Rada_Trees text records for downstream gates.

This is a source-to-record handoff only. It replays the exact immutable secondary
archive, reproduces the terminal selective plaintext classification and provenance /
rights-scoped inventory from #820, and emits local JSONL records for the existing
language/quality/privacy -> dedup -> decontamination pipeline. The JSONL payload is an
operator artifact; this tool never grants corpus or training credit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Mapping

TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import classify_d03_rada_trees_members as classify
import derive_d03_rada_trees_provenance_rights as rights
import probe_d03_rada_trees_secondary_role as probe
import scan_d03_rada_trees_secondary_plaintext as scan

ROOT = Path(__file__).resolve().parents[1]
RIGHTS_CONFIG = ROOT / "configs/data/d03_rada_trees_provenance_rights_v1.json"
RIGHTS_EVIDENCE = ROOT / "evidence/d03-rada-trees/secondary-plaintext-provenance-rights-v1.json"
SCHEMA = "12-6.d03-rada-trees-rights-scoped-handoff-report.v1"
RIGHTS_REPORT_SHA256 = "7eea6d0b79353ef565910738dce9de93008a961059cab503b6f05686c0f27a7d"
RIGHTS_CONFIG_SHA256 = "34da44a047c5e0d562ee6a86987cb66e3ed266e1c1f09af95e63c38c219fe1a3"
FULL_SCAN_EVIDENCE_ID = "b25ff7e7948e2fa215ee8d059ea5594025693e105389b7095f5250808ecc3a9d"
SOURCE_DATASET = "uacorpus/Rada_Trees"
SOURCE_REVISION = "1b994a5804dcda122721e8d33a03fd172cf8d867"
SOURCE_ARCHIVE = "rada_xtag_texts.7z"
SOURCE_SHA256 = "737fa5df061c55555a8c761023559a206cbcabf0299ae52d1ccae387f685803e"
SOURCE_BYTES = 697_768_591
SOURCE_FAMILY = "ua.rada.open-data.plenary-transcripts"
EXPECTED_RECORDS = 4_384
EXPECTED_SOURCE_BYTES = 877_899_128
EXPECTED_ACCEPTED_INVENTORY = "7b93056f38fbc87c11e14df9069066e21380db1f1b8d8016314330f841bfa6fc"
EXPECTED_HELD_INVENTORY = "566760e10157cd835ed0879abb37f052b57d31cff6af358a81ff717f4f7f59d9"
REQUIRED_DOWNSTREAM = (
    "LANGUAGE_QUALITY_PRIVACY",
    "CURRENT_GLOBAL_EXACT_NEAR_LINEAGE_DEDUP",
    "FRESH_RESERVED_EVALUATION_DECONTAMINATION",
    "BALANCE_AND_FAMILY_CAPS",
    "CLUSTER_SAFE_SPLIT",
    "DETERMINISTIC_PACKING_TWO_CLEAN_BUILDS",
    "POSITIVE_UNIQUE_LOSS_LEDGER",
)


class HandoffError(RuntimeError):
    """Fail-closed handoff error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise HandoffError(message)


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HandoffError(f"cannot load JSON authority: {path}") from exc
    require(isinstance(value, dict), f"authority root must be object: {path}")
    return value


def load_parent_authority() -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = load_json(RIGHTS_CONFIG)
    evidence = load_json(RIGHTS_EVIDENCE)
    require(
        cfg.get("schema_version") == "12-6.d03-rada-trees-provenance-rights.v1",
        "rights config schema drift",
    )
    require(cfg.get("execution_profile") == "LOCAL_FREE", "LOCAL_FREE boundary drift")
    require(rights.canonical_sha256(cfg) == RIGHTS_CONFIG_SHA256, "rights config hash drift")
    require(
        evidence.get("schema_version")
        == "12-6.d03-rada-trees-provenance-rights-report.v1",
        "rights evidence schema drift",
    )
    claimed = evidence.get("report_sha256")
    core = dict(evidence)
    core.pop("report_sha256", None)
    require(rights.canonical_sha256(core) == claimed, "rights evidence self-hash invalid")
    require(claimed == RIGHTS_REPORT_SHA256, "rights evidence report drift")
    require(evidence.get("config_sha256") == RIGHTS_CONFIG_SHA256, "rights config binding drift")
    parent = evidence.get("parent_full_scan")
    require(isinstance(parent, Mapping), "full-scan parent authority missing")
    require(
        parent.get("terminal_evidence_identity_sha256") == FULL_SCAN_EVIDENCE_ID,
        "full-scan evidence identity drift",
    )
    source = evidence.get("source")
    require(isinstance(source, Mapping), "rights source binding missing")
    require(source.get("dataset") == SOURCE_DATASET, "dataset drift")
    require(source.get("dataset_revision") == SOURCE_REVISION, "dataset revision drift")
    require(source.get("archive_path") == SOURCE_ARCHIVE, "archive path drift")
    require(source.get("content_sha256") == SOURCE_SHA256, "archive SHA drift")
    provenance = evidence.get("provenance")
    require(isinstance(provenance, Mapping), "rights provenance missing")
    require(
        provenance.get("accepted_exact_unique_members") == EXPECTED_RECORDS,
        "accepted record count drift",
    )
    require(
        provenance.get("accepted_exact_unique_bytes") == EXPECTED_SOURCE_BYTES,
        "accepted source byte total drift",
    )
    require(
        provenance.get("accepted_path_inventory_sha256") == EXPECTED_ACCEPTED_INVENTORY,
        "accepted inventory identity drift",
    )
    require(
        provenance.get("held_path_inventory_sha256") == EXPECTED_HELD_INVENTORY,
        "held inventory identity drift",
    )
    boundary = evidence.get("claim_boundary")
    require(isinstance(boundary, Mapping), "rights claim boundary missing")
    require(boundary.get("training_authorized_bytes") == 0, "parent training credit drift")
    require(
        boundary.get("unique_causal_loss_positions_authorized") == 0,
        "parent loss-position credit drift",
    )
    for key in (
        "tokenizer_fit_authorized",
        "model_training_executed",
        "final_test_payload_accessed",
        "paid_compute_used",
    ):
        require(boundary.get(key) is False, f"parent boundary weakened: {key}")
    return cfg, evidence


def rights_scoped_survivors(
    classified: list[dict[str, Any]],
    cfg: Mapping[str, Any],
    *,
    verify_expected: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates = [
        row for row in classified if row.get("classification") == "PLAIN_TEXT_CANDIDATE"
    ]
    by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_hash[str(row["sha256"])].append(row)
    survivors = sorted(
        (min(group, key=lambda row: str(row["path"])) for group in by_hash.values()),
        key=lambda row: str(row["path"]),
    )
    policy = cfg["path_provenance_policy"]
    pattern = re.compile(str(policy["dated_plenary_path_regex"]))
    holds_by_path = {str(row["path"]): row for row in policy["explicit_holds"]}
    accepted: list[dict[str, Any]] = []
    held: list[dict[str, Any]] = []
    for row in survivors:
        path = str(row["path"])
        compact: dict[str, Any] = {
            "path": path,
            "size_bytes": int(row["size_bytes"]),
            "sha256": str(row["sha256"]),
        }
        match = pattern.fullmatch(path)
        if match:
            parsed = date.fromisoformat(match.group("date"))
            require(
                int(policy["minimum_year"]) <= parsed.year <= int(policy["maximum_year"]),
                f"dated path outside approved period: {path}",
            )
            compact["date"] = parsed.isoformat()
            accepted.append(compact)
            continue
        hold = holds_by_path.get(path)
        require(hold is not None, f"unrecognized exact-unique path is not held: {path}")
        require(str(row["sha256"]) == hold["sha256"], f"held SHA drift: {path}")
        require(int(row["size_bytes"]) == int(hold["size_bytes"]), f"held size drift: {path}")
        held.append(compact)

    if verify_expected:
        expected = cfg["expected_result"]
        require(
            len(accepted) == int(expected["accepted_exact_unique_members"]),
            "accepted count drift",
        )
        require(
            sum(row["size_bytes"] for row in accepted)
            == int(expected["accepted_exact_unique_bytes"]),
            "accepted byte total drift",
        )
        require(len(held) == int(expected["held_exact_unique_members"]), "held count drift")
        require(
            sum(row["size_bytes"] for row in held) == int(expected["held_exact_unique_bytes"]),
            "held byte total drift",
        )
        require(
            rights.inventory_identity(accepted) == expected["accepted_path_inventory_sha256"],
            "accepted inventory drift",
        )
        require(
            rights.inventory_identity(held) == expected["held_path_inventory_sha256"],
            "held inventory drift",
        )
    return accepted, held


def _candidate_record(
    row: Mapping[str, Any],
    payload: bytes,
    *,
    decoded_encoding: str,
    decode_order: list[str],
) -> dict[str, Any]:
    require(len(payload) == int(row["size_bytes"]), f"payload size drift: {row['path']}")
    require(sha256_bytes(payload) == row["sha256"], f"payload SHA drift: {row['path']}")
    text, observed_encoding = classify.decode_text(payload, decode_order)
    require(text is not None and observed_encoding is not None, f"payload decode failed: {row['path']}")
    require(observed_encoding == decoded_encoding, f"decoded encoding drift: {row['path']}")
    text_bytes = text.encode("utf-8")
    return {
        "record_id": f"rada-trees:{row['sha256']}:{row['path']}",
        "source_family": SOURCE_FAMILY,
        "source_dataset": SOURCE_DATASET,
        "source_revision": SOURCE_REVISION,
        "source_archive": SOURCE_ARCHIVE,
        "source_path": row["path"],
        "session_date": row["date"],
        "source_payload_sha256": row["sha256"],
        "source_payload_bytes": row["size_bytes"],
        "decoded_encoding": observed_encoding,
        "decoded_text_utf8_sha256": sha256_bytes(text_bytes),
        "decoded_text_utf8_bytes": len(text_bytes),
        "rights_scope_status": "RIGHTS_SCOPE_SUPPORTED_DATED_PARLIAMENT_TRANSCRIPT_CANDIDATE",
        "attribution_required": True,
        "language_quality_privacy_complete": False,
        "global_dedup_complete": False,
        "reserved_evaluation_decontamination_complete": False,
        "training_eligible": False,
        "evaluation_eligible": False,
        "text": text,
    }


def emit_candidate_jsonl(
    extracted_root: Path,
    classified: list[dict[str, Any]],
    accepted: list[dict[str, Any]],
    output: Path,
    decode_order: list[str],
) -> dict[str, Any]:
    by_path = {str(row["path"]): row for row in classified}
    require(len(by_path) == len(classified), "classified path vector contains duplicates")
    partial = output.with_suffix(output.suffix + ".partial")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial.unlink(missing_ok=True)
    digest = hashlib.sha256()
    output_bytes = 0
    text_utf8_bytes = 0
    source_bytes = 0
    encoding_counts: Counter[str] = Counter()
    year_counts: Counter[str] = Counter()
    try:
        with partial.open("wb") as handle:
            for row in accepted:
                path = str(row["path"])
                classified_row = by_path.get(path)
                require(classified_row is not None, f"accepted path not classified: {path}")
                require(
                    classified_row.get("classification") == "PLAIN_TEXT_CANDIDATE",
                    f"accepted path not plaintext candidate: {path}",
                )
                encoding = classified_row.get("decoded_encoding")
                require(isinstance(encoding, str) and encoding, f"accepted path encoding missing: {path}")
                disk = extracted_root / Path(path)
                require(disk.is_file() and not disk.is_symlink(), f"accepted payload missing: {path}")
                record = _candidate_record(
                    row,
                    disk.read_bytes(),
                    decoded_encoding=encoding,
                    decode_order=decode_order,
                )
                line = canonical_bytes(record)
                handle.write(line)
                digest.update(line)
                output_bytes += len(line)
                text_utf8_bytes += int(record["decoded_text_utf8_bytes"])
                source_bytes += int(record["source_payload_bytes"])
                encoding_counts[encoding] += 1
                year_counts[str(record["session_date"])[:4]] += 1
        require(len(accepted) > 0, "candidate handoff emitted zero records")
        partial.replace(output)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    return {
        "candidate_jsonl_sha256": digest.hexdigest(),
        "candidate_jsonl_bytes": output_bytes,
        "candidate_records": len(accepted),
        "accepted_source_payload_bytes": source_bytes,
        "decoded_text_utf8_bytes": text_utf8_bytes,
        "decoded_encoding_counts": dict(sorted(encoding_counts.items())),
        "year_record_counts": dict(sorted(year_counts.items())),
    }


def materialize(archive: Path, candidate_jsonl: Path, report_path: Path) -> dict[str, Any]:
    cfg, evidence = load_parent_authority()
    source_cfg = probe.load_config()
    require(archive.is_file() and not archive.is_symlink(), "archive must be a regular file")
    require(archive.stat().st_size == SOURCE_BYTES, "archive byte-size drift")
    require(scan.sha256_file(archive) == SOURCE_SHA256, "archive content SHA drift")
    extractor = probe.inventory.find_extractor(
        None, list(source_cfg["inventory_policy"]["accepted_extractors"])
    )
    require(scan.inventory.extractor_version(extractor) == scan.EXPECTED_EXTRACTOR_VERSION, "extractor runtime drift")
    listing = scan.inventory.list_archive(extractor, archive)
    probe._listing_bounds(listing, source_cfg["inventory_policy"])
    require(scan.listing_identity(source_cfg, listing) == scan.EXPECTED_LISTING_IDENTITY, "archive listing identity drift")
    selected = scan.select_plaintext_listing(listing)

    classifier_cfg = json.loads(probe.CLASSIFIER_CONFIG.read_text(encoding="utf-8"))
    classify.validate_config(classifier_cfg)
    policy = classifier_cfg["classification_policy"]
    with tempfile.TemporaryDirectory(prefix="rada-rights-handoff-") as tmp:
        extracted_root = Path(tmp) / "selected"
        members = scan.extract_exact_selected(extractor, archive, extracted_root, selected)
        classified = probe.classify_extracted(extracted_root, members, policy)
        require(
            all(row.get("classification") == "PLAIN_TEXT_CANDIDATE" for row in classified),
            "terminal all-plaintext classification no longer reproduces",
        )
        accepted, held = rights_scoped_survivors(classified, cfg)
        output = emit_candidate_jsonl(
            extracted_root,
            classified,
            accepted,
            candidate_jsonl,
            list(policy["strict_decode_order"]),
        )

    require(output["candidate_records"] == EXPECTED_RECORDS, "handoff record count drift")
    require(
        output["accepted_source_payload_bytes"] == EXPECTED_SOURCE_BYTES,
        "handoff source byte total drift",
    )
    core: dict[str, Any] = {
        "schema_version": SCHEMA,
        "execution_profile": "LOCAL_FREE",
        "source": {
            "dataset": SOURCE_DATASET,
            "revision": SOURCE_REVISION,
            "archive": SOURCE_ARCHIVE,
            "archive_bytes": SOURCE_BYTES,
            "archive_sha256": SOURCE_SHA256,
            "family": SOURCE_FAMILY,
        },
        "parent_authority": {
            "full_scan_evidence_identity_sha256": FULL_SCAN_EVIDENCE_ID,
            "rights_config_sha256": RIGHTS_CONFIG_SHA256,
            "rights_report_sha256": RIGHTS_REPORT_SHA256,
            "accepted_path_inventory_sha256": EXPECTED_ACCEPTED_INVENTORY,
            "held_path_inventory_sha256": EXPECTED_HELD_INVENTORY,
        },
        "materialization": {
            **output,
            "held_records_not_emitted": len(held),
            "held_source_payload_bytes_not_emitted": sum(row["size_bytes"] for row in held),
            "source_native_document_boundaries_preserved": True,
            "record_text_persisted_in_report": False,
        },
        "rights": {
            "attribution_required": True,
            "project_rights_scope_status": evidence["decision"]["status"],
        },
        "claim_boundary": {
            "candidate_jsonl_is_canonical_corpus": False,
            "language_quality_privacy_complete": False,
            "global_dedup_complete": False,
            "reserved_evaluation_decontamination_complete": False,
            "family_caps_complete": False,
            "training_authorized_bytes": 0,
            "unique_causal_loss_positions_authorized": 0,
            "tokenizer_fit_authorized": False,
            "optimizer_updates": 0,
            "model_training_executed": False,
            "final_test_payload_accessed": False,
            "paid_compute_used": False,
        },
        "required_downstream_gates": list(REQUIRED_DOWNSTREAM),
        "safe_result": "RADA_TREES_RIGHTS_SCOPED_HANDOFF_ZERO_CREDIT",
    }
    report = {**core, "report_sha256": sha256_bytes(canonical_bytes(core))}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_bytes(canonical_bytes(report))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--candidate-jsonl", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = materialize(args.archive, args.candidate_jsonl, args.report)
        print("D03_RADA_TREES_RIGHTS_SCOPED_HANDOFF=PASS_ZERO_CREDIT")
        print("CANDIDATE_RECORDS=" + str(report["materialization"]["candidate_records"]))
        print(
            "CANDIDATE_SOURCE_BYTES="
            + str(report["materialization"]["accepted_source_payload_bytes"])
        )
        print("REPORT_SHA256=" + report["report_sha256"])
        return 0
    except (
        HandoffError,
        scan.PlaintextScanError,
        probe.SecondaryProbeError,
        classify.ClassificationError,
        rights.ProvenanceRightsError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
    ) as exc:
        print(f"BLOCKED: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
