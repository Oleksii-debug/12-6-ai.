#!/usr/bin/env python3
"""Deterministically materialize visible-text records from a probed Rada ZIP.

This is a normalization boundary only. It verifies the exact D03 probe inventory,
then emits normalized JSONL plus a text-free manifest. It deliberately grants
zero training capacity until quality/privacy/dedup/decontamination and later
corpus gates are terminal.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import unicodedata
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Mapping

CONFIG_SCHEMA = "12-6.d03-rada-bulk-normalization.v1"
PROBE_SCHEMA = "12-6.d03-rada-bulk-source-probe-report.v1"
MANIFEST_SCHEMA = "12-6.d03-rada-bulk-normalization-manifest.v1"
DEFAULT_CONFIG = Path("configs/data/d03_rada_bulk_normalization_v1.json")
CANONICAL_BASENAME = re.compile(r"d[0-9]+\.htm")


class NormalizationError(RuntimeError):
    """Fail-closed materialization error."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json_bytes(data: Mapping[str, Any]) -> bytes:
    return json.dumps(
        data,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _load_json(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        raw_bytes = path.read_bytes()
        value = json.loads(raw_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NormalizationError(f"cannot load JSON: {path}") from exc
    if not isinstance(value, dict):
        raise NormalizationError(f"JSON root must be an object: {path}")
    return value, raw_bytes


def _validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise NormalizationError("unsupported normalization config schema")
    if config.get("local_free_only") is not True:
        raise NormalizationError("normalization must remain LOCAL_FREE")
    boundary = config.get("claim_boundary")
    if not isinstance(boundary, Mapping):
        raise NormalizationError("claim_boundary missing")
    required_false = (
        "bulk_source_admitted",
        "normalized_capacity_credited",
        "tokenizer_fit_authorized",
        "model_training_executed",
        "paid_compute_used",
        "research_corpus_v1_released",
        "learned_20m_claimed",
    )
    for key in required_false:
        if boundary.get(key) is not False:
            raise NormalizationError(f"truth boundary weakened: {key}")
    if boundary.get("training_authorized_bytes") != 0:
        raise NormalizationError("normalization must authorize zero training bytes")


class _VisibleTextParser(HTMLParser):
    def __init__(self, *, hidden_tags: set[str], block_tags: set[str]) -> None:
        super().__init__(convert_charrefs=True)
        self.hidden_tags = hidden_tags
        self.block_tags = block_tags
        self.hidden_stack: list[str] = []
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        tag = tag.lower()
        if self.hidden_stack:
            if tag in self.hidden_tags:
                self.hidden_stack.append(tag)
            return
        if tag in self.hidden_tags:
            self.hidden_stack.append(tag)
            return
        if tag in self.block_tags:
            self.parts.append("\n")

    def handle_startendtag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        self.handle_starttag(tag, attrs)
        if not self.hidden_stack and tag.lower() in self.block_tags:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if self.hidden_stack:
            if tag == self.hidden_stack[-1]:
                self.hidden_stack.pop()
            return
        if tag in self.block_tags:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.hidden_stack:
            self.parts.append(data)


def normalize_html_bytes(raw: bytes, config: Mapping[str, Any]) -> str:
    """Extract deterministic visible text from one strict-UTF-8 HTML object."""
    normalization = config.get("normalization")
    if not isinstance(normalization, Mapping):
        raise NormalizationError("normalization policy missing")
    try:
        decoded = raw.decode("utf-8-sig", errors="strict")
    except UnicodeDecodeError as exc:
        raise NormalizationError("Rada HTML is not strict UTF-8") from exc

    hidden_tags = {str(tag).lower() for tag in normalization.get("hidden_tags", [])}
    block_tags = {str(tag).lower() for tag in normalization.get("block_tags", [])}
    if not hidden_tags or not block_tags:
        raise NormalizationError("normalization tag policy must be non-empty")

    parser = _VisibleTextParser(hidden_tags=hidden_tags, block_tags=block_tags)
    try:
        parser.feed(decoded)
        parser.close()
    except Exception as exc:  # HTMLParser may expose malformed-input errors.
        raise NormalizationError("HTML parsing failed") from exc

    text = unicodedata.normalize("NFKC", "".join(parser.parts))
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    for char in text:
        category = unicodedata.category(char)
        if category == "Cc" and char not in "\n\t":
            raise NormalizationError(
                f"visible text contains unsupported control U+{ord(char):04X}"
            )

    lines: list[str] = []
    previous_blank = True
    for raw_line in text.split("\n"):
        line = re.sub(r"[^\S\n]+", " ", raw_line).strip()
        if line:
            lines.append(line)
            previous_blank = False
        elif not previous_blank:
            lines.append("")
            previous_blank = True
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def _validate_probe(
    probe: Mapping[str, Any],
    config: Mapping[str, Any],
    archive: bytes,
) -> dict[str, Mapping[str, Any]]:
    if probe.get("schema_version") != PROBE_SCHEMA:
        raise NormalizationError("unsupported probe report schema")
    parent = config["parent_probe"]
    if probe.get("source_family") != parent["source_family"]:
        raise NormalizationError("probe source-family drift")
    if probe.get("training_authorized_bytes") != 0:
        raise NormalizationError("probe unexpectedly grants training capacity")
    if probe.get("corpus_admitted") is not False:
        raise NormalizationError("probe unexpectedly claims corpus admission")

    gates = probe.get("gates")
    if not isinstance(gates, Mapping):
        raise NormalizationError("probe gates missing")
    if gates.get("exact_archive_identity") != "PASS":
        raise NormalizationError("probe archive identity is not PASS")
    if gates.get("safe_zip_inventory") != "PASS":
        raise NormalizationError("probe ZIP inventory is not PASS")
    if gates.get("canonical_normalization") != "NOT_RUN":
        raise NormalizationError("probe normalization state is not the expected handoff")

    archive_meta = probe.get("archive")
    if not isinstance(archive_meta, Mapping):
        raise NormalizationError("probe archive metadata missing")
    if archive_meta.get("bytes") != len(archive):
        raise NormalizationError("archive byte count does not match probe")
    if archive_meta.get("sha256") != _sha256(archive):
        raise NormalizationError("archive SHA-256 does not match probe")

    inventory = probe.get("inventory")
    if not isinstance(inventory, Mapping):
        raise NormalizationError("probe inventory missing")
    entries = inventory.get("entries")
    if not isinstance(entries, list) or not entries:
        raise NormalizationError("probe inventory entries missing")

    by_basename: dict[str, Mapping[str, Any]] = {}
    for raw_entry in entries:
        if not isinstance(raw_entry, Mapping):
            raise NormalizationError("probe entry must be an object")
        basename = raw_entry.get("basename")
        if not isinstance(basename, str) or not CANONICAL_BASENAME.fullmatch(basename):
            raise NormalizationError("probe contains a noncanonical basename")
        if basename in by_basename:
            raise NormalizationError(f"duplicate probe basename: {basename}")
        by_basename[basename] = raw_entry
    if inventory.get("canonical_entry_count") != len(by_basename):
        raise NormalizationError("probe canonical-entry count drift")
    return by_basename


def materialize_normalized_records(
    archive: bytes,
    probe: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    probe_report_sha256: str,
) -> tuple[bytes, dict[str, Any]]:
    """Verify the exact probe inventory and return deterministic JSONL + manifest."""
    _validate_config(config)
    expected = _validate_probe(probe, config, archive)
    normalizer = config["normalization"]
    prefix = str(normalizer["record_id_prefix"])

    try:
        archive_file = zipfile.ZipFile(io.BytesIO(archive))
    except zipfile.BadZipFile as exc:
        raise NormalizationError("invalid ZIP archive") from exc

    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    observed_basenames: set[str] = set()
    with archive_file as zf:
        for info in zf.infolist():
            normalized_path = info.filename.replace("\\", "/")
            basename = Path(normalized_path).name
            if not CANONICAL_BASENAME.fullmatch(basename):
                continue
            if basename in observed_basenames:
                raise NormalizationError(f"duplicate canonical basename: {basename}")
            observed_basenames.add(basename)
            expected_entry = expected.get(basename)
            if expected_entry is None:
                raise NormalizationError(f"canonical archive entry absent from probe: {basename}")
            if expected_entry.get("path") != normalized_path:
                raise NormalizationError(f"archive path drift for {basename}")
            try:
                raw = zf.read(info)
            except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
                raise NormalizationError(f"cannot read ZIP entry: {basename}") from exc
            if expected_entry.get("raw_bytes") != len(raw):
                raise NormalizationError(f"raw byte count drift for {basename}")
            raw_sha256 = _sha256(raw)
            if expected_entry.get("raw_sha256") != raw_sha256:
                raise NormalizationError(f"raw SHA-256 drift for {basename}")

            text = normalize_html_bytes(raw, config)
            normalized = text.encode("utf-8")
            record_id = prefix + Path(basename).stem
            if record_id in seen_ids:
                raise NormalizationError(f"duplicate record id: {record_id}")
            seen_ids.add(record_id)
            records.append(
                {
                    "record_id": record_id,
                    "source_path": normalized_path,
                    "raw_bytes": len(raw),
                    "raw_sha256": raw_sha256,
                    "normalized_bytes": len(normalized),
                    "normalized_sha256": _sha256(normalized),
                    "text": text,
                }
            )

    if observed_basenames != set(expected):
        missing = sorted(set(expected) - observed_basenames)
        raise NormalizationError(f"probe entries missing from archive: {missing[:3]}")

    records.sort(key=lambda row: Path(row["source_path"]).name)
    jsonl = b"".join(
        _canonical_json_bytes(record) + b"\n"
        for record in records
    )
    record_manifest = [
        {key: value for key, value in record.items() if key != "text"}
        for record in records
    ]
    normalized_bytes = sum(int(record["normalized_bytes"]) for record in records)
    nonempty_records = sum(int(record["normalized_bytes"]) > 0 for record in records)

    inventory_hasher = hashlib.sha256()
    for record in record_manifest:
        inventory_hasher.update(_canonical_json_bytes(record))
        inventory_hasher.update(b"\n")

    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA,
        "worker_id": config["worker_id"],
        "local_free_only": True,
        "parent_probe": {
            "pr": config["parent_probe"]["pr"],
            "head_sha": config["parent_probe"]["head_sha"],
            "probe_report_sha256": probe_report_sha256,
            "archive_sha256": probe["archive"]["sha256"],
            "entry_identity_sha256": probe["inventory"]["entry_identity_sha256"],
        },
        "normalization": {
            "name": normalizer["name"],
            "record_count": len(records),
            "nonempty_record_count": nonempty_records,
            "raw_bytes": sum(int(record["raw_bytes"]) for record in records),
            "normalized_bytes_observed_not_credited": normalized_bytes,
            "normalized_record_inventory_sha256": inventory_hasher.hexdigest(),
            "jsonl_sha256": _sha256(jsonl),
        },
        "records": record_manifest,
        "gates": {
            "exact_probe_inventory": "PASS",
            "canonical_normalization": "PASS",
            "quality": "NOT_RUN",
            "privacy": "NOT_RUN",
            "global_cross_source_dedup": "NOT_RUN",
            "evaluation_decontamination": "NOT_RUN",
            "balance_diversity": "NOT_RUN",
            "corpus_materialization": "NOT_RUN",
            "unique_loss_ledger": "NOT_RUN",
        },
        "training_authorized_bytes": 0,
        "normalized_capacity_credited": 0,
        "tokenizer_fit_authorized": False,
        "model_training_executed": False,
        "paid_compute_used": False,
        "research_corpus_v1_released": False,
        "safe_result": "NORMALIZED_RECORD_MATERIALIZATION_ONLY_DOWNSTREAM_GATES_REQUIRED",
    }
    manifest["manifest_identity_sha256"] = _sha256(_canonical_json_bytes(manifest))
    return jsonl, manifest


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--probe-report", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config, _ = _load_json(args.config)
    probe, probe_bytes = _load_json(args.probe_report)
    try:
        archive = args.archive.read_bytes()
    except OSError as exc:
        raise NormalizationError(f"cannot read archive: {args.archive}") from exc

    jsonl, manifest = materialize_normalized_records(
        archive,
        probe,
        config,
        probe_report_sha256=_sha256(probe_bytes),
    )
    manifest_bytes = json.dumps(
        manifest,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    ).encode("utf-8") + b"\n"
    _atomic_write(args.output_jsonl, jsonl)
    _atomic_write(args.output_manifest, manifest_bytes)
    print(
        json.dumps(
            {
                "status": "PASS_NORMALIZATION_ONLY",
                "record_count": manifest["normalization"]["record_count"],
                "normalized_bytes_observed_not_credited": manifest["normalization"][
                    "normalized_bytes_observed_not_credited"
                ],
                "manifest_identity_sha256": manifest["manifest_identity_sha256"],
                "training_authorized_bytes": 0,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
