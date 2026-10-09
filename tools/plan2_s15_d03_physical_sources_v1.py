"""Plan 2 S15: exact D03 Ukrainian physical provenance (NONRELEASE).

The historical NEXT100-028/NEXT100-030 normalization formulas are repeated
verbatim as pure deterministic adapters. G06 and DATA232 remain the only
privacy and decontamination engines. These source seals do not admit S3-S9.
"""
from __future__ import annotations

import argparse
import html
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from tools import plan2_public_domain_books_v1 as books
from tools import plan2_reserved_eval_firewall_v1 as firewall
from tools.plan2_physical_materialization_v1 import _atomic_write, _read_destination
from twelve_six.data import decontamination_authority_v2 as data232
from twelve_six.data import privacy_execution_authority as g06

SCHEMA = "12-6.plan2-s15-d03-physical-source-candidate.v1"
CONFIG = "configs/data/plan2_s15_d03_physical_source_candidates_v1.json"
SEALS = "configs/data/d03_terminal_ua_source_seals_v1.json"
SEALS_BLOB = "e80e09e9b5de6eade9b579b64503e0dfab2a03c8"
OUTPUT = "d03-physical-source-candidate.json"
FAMILIES = {
    "php": "php.manual.documentation",
    "rust": "rust-book.documentation.uk-translation",
}
PHP_REV = "c165db75cc6f81cfdabf754656e73a68940de46c"
RUST_REV = "ca2d2e4f4434c661836926017af23bdd40ad4e3d"
PREFIX = "data/external/snapshots/plan2-s15-d03-ua-v1/"


class D03SourceDenied(ValueError):
    """Invalid source snapshot, rights, privacy or qualification authority."""


def need(value: bool, reason: str) -> None:
    if not value:
        raise D03SourceDenied(reason)


def normalize_php(raw: bytes) -> bytes:
    """Pure historical NEXT100-028 v2 XML normalization."""
    text = raw.decode("utf-8", "strict").replace("\r\n", "\n").replace("\r", "\n")
    text = unicodedata.normalize("NFC", text)
    text = re.compile(r"<!--.*?-->", re.S).sub("\n", text)
    text = text.replace("<![CDATA[", "\n").replace("]]>", "\n")
    text = re.compile(r"<[^>]+>", re.S).sub("\n", text)
    text = html.unescape(text)
    text = re.compile(r"&[A-Za-z_][A-Za-z0-9_.:-]*;").sub(" ", text)
    lines = [" ".join(line.split()) for line in text.splitlines()]
    lines = [line for line in lines if line]
    need(bool(lines), "PHP physical normalization empty")
    return ("\n".join(lines) + "\n").encode("utf-8")


def normalize_rust(raw: bytes) -> bytes:
    """Pure historical NEXT100-030 Markdown normalization."""
    text = raw.decode("utf-8", "strict").replace("\r\n", "\n").replace("\r", "\n")
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"<!--.*?-->", "\n", text, flags=re.DOTALL)
    text = re.sub(r"\x60\x60\x60.*?\x60\x60\x60", "\n",
                  text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"^\s*\[[^\]]+\]:\s+\S+\s*$", "", text,
                  flags=re.MULTILINE)
    text = re.sub(r"\x60[^\x60]*\x60", " ", text)
    lines: list[str] = []
    for line in text.splitlines():
        line = re.sub(r"^\s{0,3}(?:#{1,6}|>|[-*+]\s+|\d+\.\s+)", "", line)
        line = line.replace("**", "").replace("__", "").replace("*", "").replace("_", "")
        collapsed = " ".join(line.split())
        if collapsed:
            lines.append(collapsed)
    need(bool(lines), "Rust physical normalization empty")
    return ("\n".join(lines) + "\n").encode("utf-8")


def inspect(root: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    config_raw = books.read_checked(root, CONFIG)
    config = json.loads(config_raw.decode("utf-8", "strict"))
    need(config["schema_version"] == "12-6.plan2-s15-d03-physical-source-candidates.v1"
         and config["source_candidate_only"] is True
         and config["training_corpus_authorized"] is False
         and config["production_release_authorized"] is False
         and config["tokenizer_fit_authorized"] is False
         and config["terminal_done"] is False
         and config["physical_source_family_count"] == 2
         and config["physical_document_count"] == 12
         and config["physical_normalized_expected_bytes"] == 48675,
         "D03 candidate flags or counts unexpectedly elevated")
    seal_raw = books.read_checked(root, SEALS)
    need(books.git_blob(seal_raw) == SEALS_BLOB
         and config["d03_source_seals_git_blob_sha1"] == SEALS_BLOB,
         "D03 source authority Git identity drift")
    seals = json.loads(seal_raw.decode("utf-8", "strict"))
    need(seals["authority_state"] == "TERMINAL_SOURCE_AUTHORITIES_ZERO_CANONICAL_CREDIT"
         and len(seals["sources"]) == 2,
         "source seals cannot grant Plan2 S3-S9 training")
    approved = {
        "php": next(x for x in seals["sources"]
                    if x["source_id"] == "next100-028-php-doc-uk"),
        "rust": next(x for x in seals["sources"]
                     if x["source_id"] == "next100-030-rustbook-ua-oer"),
    }
    families = {x["id"]: x for x in config["source_families"]}
    need(len(families) == len(config["source_families"]) == 2
         and set(families) == set(FAMILIES.values())
         and all(x["model_training_rights"] == "ALLOWED"
                 and x["terminal_verdict"] == "ADMIT"
                 and x["evaluation"] == "NOT_SEPARATELY_ADMITTED"
                 for x in approved.values()),
         "source family membership or historical training rights missing")
    for alias, sealed in approved.items():
        family = families[FAMILIES[alias]]
        need(family["revision"] == sealed["source_commit"]
             and family["revision"] == (PHP_REV if alias == "php" else RUST_REV)
             and family["historical_artifact_id"] == sealed["terminal_artifact_id"]
             and family["historical_normalized_bundle_sha256"] ==
                 sealed["normalized_bundle_sha256"]
             and family["training_rights"] ==
                 "ALLOWED_SOURCE_LEVEL_NOT_PLAN2_CORPUS"
             and family["source_rights"] == (
                 "CC-BY-3.0-or-later" if alias == "php"
                 else "MIT_AND_APACHE-2.0_WITH_NOTICES"
             )
             and family["evaluation_admitted"] is False,
             "historical rights or exact revisions changed")
    licenses = config["source_license_files"]
    need(len(licenses) == 3 and len({x["path"] for x in licenses}) == 3,
         "source licenses missing or duplicated")
    for item in licenses:
        license_text = books.read_checked(root, item["path"])
        need(books.git_blob(license_text) == item["git_blob_sha1"]
             and 1000 < len(license_text) < 30000,
             "pinned source license drift")
    note = books.read_checked(root, PREFIX + "ATTRIBUTION.txt")
    need(b"CC BY 3.0" in note and b"LICENSE-MIT" in note
         and b"LICENSE-APACHE" in note, "license attribution is missing")
    records: list[dict[str, Any]] = []
    rows: list[dict[str, str]] = []
    paths: set[str] = set()
    total = 0
    for item in config["objects"]:
        alias = item["family"]
        need(alias in FAMILIES, "unregistered source family")
        path = item["physical_path"]
        prefix = PREFIX + "raw/" + alias + "/"
        need(path.startswith(prefix) and path not in paths
             and item["raw_repo_path"].endswith(path.removeprefix(prefix)),
             "physical source path or identity drift")
        paths.add(path)
        raw = books.read_checked(root, path)
        need(books.git_blob(raw) == item["raw_repo_git_blob_sha1"]
             and 100 < len(raw) <= 65536,
             "pinned raw source Git blob mismatch")
        payload = normalize_php(raw) if alias == "php" else normalize_rust(raw)
        digest = books.sha(payload)
        need(digest == item["normalized_expected_sha256"]
             and len(payload) == item["normalized_expected_bytes"],
             "actual physical normalized source differs from D03 seal")
        sid = approved[alias]["source_id"]
        rid = sid + f":r{len(records):08d}"
        records.append({
            "record_id": rid, "source_id": sid,
            "source_family": FAMILIES[alias], "source_path": path,
            "raw_git_blob_sha1": item["raw_repo_git_blob_sha1"],
            "normalized_sha256": digest, "normalized_bytes": len(payload),
        })
        rows.append({"id": rid, "text": payload.decode("utf-8"), "mode": "uk"})
        total += len(payload)
    need(len(records) == 12 and total == 48675
         and len({r["record_id"] for r in records}) == 12
         and len({r["normalized_sha256"] for r in records}) == 12
         and {r["source_family"] for r in records} == set(FAMILIES.values()),
         "D03 physical source member/family/byte counts drifted")
    inventory = [{
        "record_id": r["record_id"], "source_id": r["source_id"],
        "family": r["source_family"], "modality": "uk",
        "payload_sha256": r["normalized_sha256"],
        "payload_bytes": r["normalized_bytes"],
    } for r in records]
    input_sha = g06.input_rows_sha256_from_text_free_inventory(inventory)
    privacy = g06.build_privacy_execution_authority(
        rows, expected_input_rows_sha256=input_sha)
    g06.verify_privacy_execution_authority(
        privacy, rows, expected_input_rows_sha256=input_sha,
        expected_execution_identity_sha256=privacy["execution_identity_sha256"])
    rejected = sorted(x["record_id"] for x in privacy["records"]
                      if x["action"] != "ALLOW")
    reserved = books.read_checked(root, firewall.RESERVE_PATH)
    _fixture, eval_rows, authorities = firewall._reserve(reserved)
    by_id = {r["id"]: r["text"] for r in rows}
    training = [{
        "record_id": r["record_id"], "source_id": r["source_id"],
        "source_family": r["source_family"],
        "lineage_family": r["source_path"],
        "modality": "uk", "text": by_id[r["record_id"]],
    } for r in records if r["record_id"] not in rejected]
    need(bool(training), "no privacy-eligible D03 candidate")
    selection_sha = books.sha(books.canonical([
        row for row in eval_rows
        if row["record_id"].startswith("reserved.selection.")
    ]))
    final_sha = books.sha(books.canonical([
        row for row in eval_rows
        if row["record_id"].startswith("reserved.final.")
    ]))
    report = data232.build_report(
        training, eval_rows,
        training_corpus_identity=books.sha(books.canonical(records)),
        selection_validation_identity=selection_sha,
        final_test_identity=final_sha, authorities=authorities,
        quarantine_cross_source_families=True)
    data232.verify_report(report)
    core = {
        "schema_version": SCHEMA,
        "decision": "D03_PHYSICAL_UK_SOURCE_CANDIDATE_NOT_S3_S9_RELEASE",
        "historical_seals_sha256": books.sha(seal_raw),
        "candidate_config_sha256": books.sha(config_raw),
        "physical_source_families": sorted(FAMILIES.values()),
        "physical_family_count": 2,
        "physical_record_count": 12,
        "normalized_source_bytes": total,
        "record_inventory_sha256": books.sha(books.canonical(records)),
        "records": sorted(records, key=lambda item: item["record_id"]),
        "g06_input_root_sha256": input_sha,
        "g06_execution_identity_sha256": privacy["execution_identity_sha256"],
        "g06_rejected_record_ids": rejected,
        "data232_report_sha256": report["report_sha256"],
        "data232_excluded_record_count":
            report["counts"]["excluded_training_records"],
        "data232_quarantined_source_family_count":
            report["counts"]["quarantined_source_families"],
        "data232_fixture_eval_only": True,
        "final_test_outcomes_read": False,
        "training_corpus_authorized": False,
        "tokenizer_fit_authorized": False,
        "production_release_authorized": False,
        "terminal_done": False,
    }
    return {**core, "manifest_sha256": books.sha(books.canonical(core))}


def stage(root: Path, destination: Path) -> dict[str, Any]:
    need(not any(p.is_symlink() for p in (destination, *destination.parents)),
         "symlink publication")
    proof = inspect(root)
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / OUTPUT
    raw = books.canonical(proof)
    if target.exists() or target.is_symlink():
        need(target.is_file() and not target.is_symlink()
             and _read_destination(target) == raw,
             "immutable D03 source proof changed")
    else:
        _atomic_write(destination, target, raw)
    need(_read_destination(target) == raw, "D03 source readback mismatch")
    return proof


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    receipt = stage(args.root, args.out_dir)
    print(json.dumps({
        "decision": receipt["decision"],
        "manifest_sha256": receipt["manifest_sha256"],
        "production_release_authorized": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
