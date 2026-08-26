#!/usr/bin/env python3
"""Deterministic fail-closed probe for bounded Rada laws-texts expansion.

No output of this tool grants training capacity. PROBE produces only hashes,
sizes and bounded quality/privacy metadata. A successor must lock exact archive
bytes and pass global dedup plus evaluation decontamination.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
import urllib.request
import zipfile
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/data/next100_105_rada_bulk_expansion_v1.json"
EMAIL_RE = re.compile(r"(?i)(?<![\w.+-])[\w.+-]+@[\w.-]+\.[a-z]{2,}(?![\w.-])")
UA_INTL_PHONE_RE = re.compile(
    r"(?<!\d)\+380[\s()\-]*\d{2}[\s()\-]*\d{3}[\s()\-]*\d{2}[\s()\-]*\d{2}(?!\d)"
)
PRIVATE_KEY_MARKERS = (
    "-----BEGIN PRIVATE KEY-----",
    "-----BEGIN RSA PRIVATE KEY-----",
    "-----BEGIN OPENSSH PRIVATE KEY-----",
    "-----BEGIN EC PRIVATE KEY-----",
)
META_CHARSET_RE = re.compile(br"(?i)charset\s*=\s*[\"']?\s*([a-z0-9._-]+)")
INVISIBLE = {ord(ch): None for ch in "\u200b\u200c\u200d\u2060\ufeff"}


class ProbeError(RuntimeError):
    pass


class VisibleHTML(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        tag = tag.lower()
        if tag in {"script", "style", "noscript", "template"}:
            self._skip_depth += 1
        elif not self._skip_depth and tag in {
            "p", "br", "div", "li", "tr", "td", "th", "h1", "h2", "h3", "h4", "h5", "h6", "title"
        }:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"script", "style", "noscript", "template"} and self._skip_depth:
            self._skip_depth -= 1
        elif not self._skip_depth and tag in {"p", "div", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6"}:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            self._parts.append(data)

    def text(self) -> str:
        return "".join(self._parts)


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def identity(value: dict[str, Any], field: str) -> str:
    payload = dict(value)
    payload.pop(field, None)
    return sha256(canonical_json(payload))


def load_config(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    validate_config(data)
    return data


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != "12-6.next100-105-rada-bulk-expansion.v1":
        raise ProbeError("unexpected schema_version")
    if config.get("local_free_only") is not True:
        raise ProbeError("LOCAL_FREE must remain true")
    if any(
        config.get(key)
        for key in (
            "training_executed",
            "tokenizer_fit_executed",
            "paid_compute_used",
            "final_test_payload_read",
        )
    ):
        raise ProbeError("execution/firewall flags must remain false")
    if config.get("mode") not in {"PROBE", "LOCKED"}:
        raise ProbeError("mode must be PROBE or LOCKED")

    parent = config["parent_registry"]
    if (
        parent["registry_identity_sha256"]
        != "77fb69c558df8c59fdae00583c955c62ad088cda98fd16b335eedb26fb2d7526"
    ):
        raise ProbeError("parent registry identity drift")
    source = config["source"]
    if (
        source["dataset_id"] != "laws-texts"
        or source["family_id"] != "ua.rada.open-data.laws-texts"
    ):
        raise ProbeError("Rada source lineage drift")
    authority = source["existing_training_authority"]
    if authority["model_training"] != "ALLOWED":
        raise ProbeError("existing training authority is not ALLOWED")
    if authority["evaluation"] != "NOT_SEPARATELY_ADMITTED":
        raise ProbeError("evaluation permission must remain separate")

    balance = config["frozen_balance_policy"]
    target = int(balance["research_corpus_target_normalized_bytes"])
    total_cap = int(target * float(balance["family_max_total_fraction"]))
    stratum_cap = int(
        int(balance["uk_target_normalized_bytes"])
        * float(balance["family_max_stratum_fraction"])
    )
    effective = min(total_cap, stratum_cap)
    if effective != balance["effective_family_cap_normalized_bytes"]:
        raise ProbeError("effective family cap arithmetic drift")
    additional = effective - int(balance["existing_rada_normalized_bytes"])
    if additional != balance["maximum_additional_pre_dedup_normalized_bytes"]:
        raise ProbeError("additional Rada cap arithmetic drift")
    if config["selection_policy"]["candidate_byte_cap"] != additional:
        raise ProbeError("selection byte cap must equal remaining family cap")
    if balance["new_family_credit"] != 0:
        raise ProbeError("same lineage may not create new family credit")

    firewall = config["evaluation_firewall"]
    if firewall["selection_reserved_records_may_enter_training"] is not False:
        raise ProbeError("selection reservation firewall weakened")
    if firewall["final_test_records_may_enter_training"] is not False:
        raise ProbeError("final-test firewall weakened")
    if firewall["final_test_payload_may_be_read_by_this_worker"] is not False:
        raise ProbeError("final-test payload access must remain prohibited")
    if (
        firewall[
            "hash_only_exact_near_fragment_decontamination_required_before_training_credit"
        ]
        is not True
    ):
        raise ProbeError("decontamination requirement weakened")

    if config["mode"] == "PROBE" and config.get("lock") is not None:
        raise ProbeError("PROBE mode must not contain a lock")
    if config["mode"] == "LOCKED":
        lock = config.get("lock")
        if not isinstance(lock, dict):
            raise ProbeError("LOCKED mode requires lock object")
        for key in (
            "archive_sha256",
            "archive_bytes",
            "selected_manifest_identity_sha256",
            "selected_pre_dedup_normalized_bytes",
        ):
            if key not in lock:
                raise ProbeError(f"LOCKED mode missing {key}")


def fetch_archive(config: dict[str, Any], output: Path) -> dict[str, Any]:
    max_bytes = int(config["selection_policy"]["max_archive_bytes"])
    request = urllib.request.Request(
        config["source"]["archive_url"],
        headers={"User-Agent": "12-6-ai-next100-105-rada-probe/1"},
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    total = 0
    sha = hashlib.sha256()
    md5 = hashlib.md5(usedforsecurity=False)
    with urllib.request.urlopen(request, timeout=60) as response, tmp.open("wb") as handle:  # noqa: S310
        while chunk := response.read(1024 * 1024):
            total += len(chunk)
            if total > max_bytes:
                raise ProbeError(f"archive exceeds max_archive_bytes={max_bytes}")
            sha.update(chunk)
            md5.update(chunk)
            handle.write(chunk)
    tmp.replace(output)
    return {
        "archive_bytes": total,
        "archive_sha256": sha.hexdigest(),
        "archive_md5": md5.hexdigest(),
    }


def decode_html(raw: bytes) -> tuple[str, str]:
    if b"\x00" in raw:
        raise ProbeError("NUL byte in HTML")
    match = META_CHARSET_RE.search(raw[:4096])
    encodings: list[str] = []
    if match:
        declared = match.group(1).decode("ascii", errors="strict").lower()
        aliases = {"windows-1251": "cp1251", "win-1251": "cp1251", "utf8": "utf-8"}
        encodings.append(aliases.get(declared, declared))
    for fallback in ("utf-8-sig", "cp1251"):
        if fallback not in encodings:
            encodings.append(fallback)
    for encoding in encodings:
        try:
            text = raw.decode(encoding, errors="strict")
        except (LookupError, UnicodeDecodeError):
            continue
        if "\ufffd" not in text:
            return text, encoding
    raise ProbeError("strict HTML decode failed")


def visible_normalize(raw: bytes) -> tuple[bytes, str]:
    text, encoding = decode_html(raw)
    parser = VisibleHTML()
    try:
        parser.feed(text)
        parser.close()
    except Exception as exc:
        raise ProbeError(f"HTML parse failed: {type(exc).__name__}") from exc
    visible = unicodedata.normalize("NFKC", parser.text()).translate(INVISIBLE)
    lines = []
    for line in visible.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        compact = " ".join(line.split())
        if compact:
            lines.append(compact)
    normalized = ("\n".join(lines).strip() + "\n").encode("utf-8") if lines else b""
    return normalized, encoding


def cyrillic_ratio(text: str) -> float:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    count = sum("CYRILLIC" in unicodedata.name(ch, "") for ch in letters)
    return count / len(letters)


def risk_reason(text: str, config: dict[str, Any]) -> str | None:
    policy = config["privacy_quality_policy"]
    if policy["reject_private_key_markers"] and any(
        marker in text for marker in PRIVATE_KEY_MARKERS
    ):
        return "private_key_marker"
    if policy["reject_live_email_addresses"] and EMAIL_RE.search(text):
        return "email"
    if policy["reject_ua_international_phone_patterns"] and UA_INTL_PHONE_RE.search(text):
        return "ua_international_phone"
    return None


def probe_archive(config: dict[str, Any], archive: Path) -> dict[str, Any]:
    validate_config(config)
    raw_archive = archive.read_bytes()
    if len(raw_archive) > int(config["selection_policy"]["max_archive_bytes"]):
        raise ProbeError("archive larger than configured maximum")
    archive_sha256 = sha256(raw_archive)
    archive_md5 = hashlib.md5(raw_archive, usedforsecurity=False).hexdigest()

    policy = config["selection_policy"]
    member_re = re.compile(policy["member_name_regex"])
    excluded = set(policy["exclude_member_names"])
    max_member = int(policy["max_member_uncompressed_bytes"])
    max_total = int(policy["max_total_uncompressed_bytes"])
    min_bytes = int(policy["min_normalized_bytes_per_record"])
    min_cyrillic = float(policy["min_cyrillic_alpha_ratio"])
    byte_cap = int(policy["candidate_byte_cap"])
    records: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    seen_hashes: set[str] = set()
    total_uncompressed = 0

    try:
        zf = zipfile.ZipFile(archive)
    except (zipfile.BadZipFile, OSError) as exc:
        raise ProbeError("invalid zip archive") from exc

    with zf:
        infos = zf.infolist()
        for info in infos:
            total_uncompressed += int(info.file_size)
            if total_uncompressed > max_total:
                raise ProbeError("zip uncompressed-size ceiling exceeded")
            name = info.filename.replace("\\", "/")
            if name.startswith("/") or ".." in name.split("/"):
                raise ProbeError(f"unsafe zip path: {name}")
            if info.is_dir():
                continue
            base = name.rsplit("/", 1)[-1]
            if base in excluded:
                rejected["explicit_existing_source_exclusion"] += 1
                continue
            match = member_re.fullmatch(base)
            if not match:
                rejected["noncanonical_member"] += 1
                continue
            if int(info.file_size) > max_member:
                rejected["member_too_large"] += 1
                continue
            try:
                raw = zf.read(info)
                normalized, encoding = visible_normalize(raw)
            except (ProbeError, RuntimeError, ValueError, OSError) as exc:
                rejected[f"decode_or_parse:{type(exc).__name__}"] += 1
                continue
            if len(normalized) < min_bytes:
                rejected["too_short"] += 1
                continue
            text = normalized.decode("utf-8")
            ratio = cyrillic_ratio(text)
            if ratio < min_cyrillic:
                rejected["language_ratio"] += 1
                continue
            risk = risk_reason(text, config)
            if risk:
                rejected[risk] += 1
                continue
            normalized_sha = sha256(normalized)
            if normalized_sha in seen_hashes:
                rejected["exact_duplicate"] += 1
                continue
            seen_hashes.add(normalized_sha)
            records.append(
                {
                    "member": base,
                    "document_id": int(match.group(1)),
                    "raw_bytes": len(raw),
                    "raw_sha256": sha256(raw),
                    "normalized_bytes": len(normalized),
                    "normalized_sha256": normalized_sha,
                    "source_encoding": encoding,
                    "cyrillic_alpha_ratio_ppm": int(round(ratio * 1_000_000)),
                    "zip_crc32": f"{info.CRC:08x}",
                }
            )

    records.sort(key=lambda item: (item["document_id"], item["member"]))
    selected: list[dict[str, Any]] = []
    selected_bytes = 0
    for record in records:
        size = int(record["normalized_bytes"])
        if selected_bytes + size > byte_cap:
            rejected["quota_overflow"] += 1
            continue
        selected.append(record)
        selected_bytes += size

    manifest = {
        "schema_version": "12-6.next100-105-rada-selected-hash-inventory.v1",
        "archive_sha256": archive_sha256,
        "family_id": config["source"]["family_id"],
        "normalization": policy["normalization"],
        "candidate_byte_cap": byte_cap,
        "selected_pre_dedup_normalized_bytes": selected_bytes,
        "selected_record_count": len(selected),
        "records": selected,
    }
    manifest_identity = sha256(canonical_json(manifest))

    if config["mode"] == "LOCKED":
        lock = config["lock"]
        checks = {
            "archive_sha256": archive_sha256,
            "archive_bytes": len(raw_archive),
            "selected_manifest_identity_sha256": manifest_identity,
            "selected_pre_dedup_normalized_bytes": selected_bytes,
        }
        for key, actual in checks.items():
            if lock[key] != actual:
                raise ProbeError(f"locked {key} mismatch")

    decision = (
        "PROBE_LOCK_REQUIRED"
        if config["mode"] == "PROBE"
        else "LOCKED_SOURCE_CANDIDATE_REQUIRES_GLOBAL_DEDUP_DECONTAMINATION"
    )
    report: dict[str, Any] = {
        "schema_version": "12-6.next100-105-rada-bulk-probe-report.v1",
        "decision": decision,
        "mode": config["mode"],
        "archive": {
            "bytes": len(raw_archive),
            "sha256": archive_sha256,
            "md5": archive_md5,
            "zip_member_count": len(infos),
            "total_uncompressed_bytes": total_uncompressed,
        },
        "portal_observation": config["source"]["portal_observation"],
        "family": {
            "id": config["source"]["family_id"],
            "existing_normalized_bytes": config["frozen_balance_policy"][
                "existing_rada_normalized_bytes"
            ],
            "additional_candidate_cap": byte_cap,
            "family_credit_added": 0,
        },
        "candidate": {
            "eligible_before_near_dedup_count": len(records),
            "selected_record_count": len(selected),
            "selected_pre_dedup_normalized_bytes": selected_bytes,
            "selected_manifest_identity_sha256": manifest_identity,
            "rejection_counts": dict(sorted(rejected.items())),
            "records": selected,
        },
        "gates": {
            "global_cross_source_near_dedup": "REQUIRED",
            "evaluation_decontamination": "REQUIRED_HASH_ONLY_BEFORE_TRAINING_CREDIT",
            "quality_privacy_composed_rerun": "REQUIRED",
            "balance_diversity_retest": "REQUIRED",
            "training_capacity_credit": 0,
            "training_eligible": False,
        },
        "execution": {
            "local_free_only": True,
            "training_executed": False,
            "tokenizer_fit_executed": False,
            "paid_compute_used": False,
            "final_test_payload_read": False,
        },
    }
    report["report_identity_sha256"] = identity(report, "report_identity_sha256")
    return report


def verify_report(config: dict[str, Any], report: dict[str, Any]) -> None:
    validate_config(config)
    if report.get("report_identity_sha256") != identity(report, "report_identity_sha256"):
        raise ProbeError("report self-hash mismatch")
    if report["family"]["family_credit_added"] != 0:
        raise ProbeError("same-family expansion cannot add family credit")
    if (
        report["candidate"]["selected_pre_dedup_normalized_bytes"]
        > config["selection_policy"]["candidate_byte_cap"]
    ):
        raise ProbeError("candidate exceeds Rada family byte cap")
    if (
        report["gates"]["training_capacity_credit"] != 0
        or report["gates"]["training_eligible"] is not False
    ):
        raise ProbeError("candidate cannot grant training credit")
    expected_execution = {
        "final_test_payload_read": False,
        "local_free_only": True,
        "paid_compute_used": False,
        "tokenizer_fit_executed": False,
        "training_executed": False,
    }
    if report["execution"] != expected_execution:
        raise ProbeError("execution truth boundary drift")
    expected_decision = (
        "PROBE_LOCK_REQUIRED"
        if config["mode"] == "PROBE"
        else "LOCKED_SOURCE_CANDIDATE_REQUIRES_GLOBAL_DEDUP_DECONTAMINATION"
    )
    if report["decision"] != expected_decision:
        raise ProbeError("decision inconsistent with config mode")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(value))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate-config")
    fetch = sub.add_parser("fetch")
    fetch.add_argument("--output", type=Path, required=True)
    fetch.add_argument("--metadata-output", type=Path)
    probe = sub.add_parser("probe")
    probe.add_argument("--archive", type=Path, required=True)
    probe.add_argument("--report", type=Path, required=True)
    verify = sub.add_parser("verify-report")
    verify.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
        if args.command == "validate-config":
            print("NEXT100-105 CONFIG PASS")
            return 0
        if args.command == "fetch":
            metadata = fetch_archive(config, args.output)
            if args.metadata_output:
                write_json(args.metadata_output, metadata)
            print(json.dumps(metadata, sort_keys=True))
            return 0
        if args.command == "probe":
            report = probe_archive(config, args.archive)
            write_json(args.report, report)
            verify_report(config, report)
            print(
                f"NEXT100-105 {report['decision']} archive={report['archive']['sha256']} "
                f"records={report['candidate']['selected_record_count']} "
                f"bytes={report['candidate']['selected_pre_dedup_normalized_bytes']}"
            )
            return 0
        if args.command == "verify-report":
            report = json.loads(args.report.read_text(encoding="utf-8"))
            verify_report(config, report)
            print(f"NEXT100-105 REPORT PASS {report['report_identity_sha256']}")
            return 0
    except (ProbeError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"NEXT100-105 FAIL: {exc}", file=sys.stderr)
        return 2
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
