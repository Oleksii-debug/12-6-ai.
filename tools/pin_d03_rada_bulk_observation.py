#!/usr/bin/env python3
"""Pin one exact zero-credit Rada bulk observation for normalization only.

The upstream URL is mutable. This tool never promotes the observation's failed
discovery threshold or grants training/corpus authority. It binds the exact
current-main observation authority, then independently verifies the report,
archive, inventory, and every canonical member before emitting a normalization-
only content-addressed successor pin.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

CONFIG_SCHEMA = "12-6.d03-rada-bulk-observation-pin.v1"
PROBE_SCHEMA = "12-6.d03-rada-bulk-source-probe-report.v1"
SHA256_RE = re.compile(r"[0-9a-f]{64}")
CANONICAL_BASENAME = re.compile(r"d[0-9]+\.htm")
DERIVED_ARCHIVE_GATE = "PASS_PINNED_DISCOVERY_REVALIDATED"
DERIVED_SAFE_RESULT = "PINNED_BULK_ARCHIVE_INVENTORIED_DOWNSTREAM_GATES_REQUIRED"
EXPECTED_PARENT = {
    "pr": 830,
    "head_sha": "077f45fea310ecbc50b87801ff916502aa7f11f9",
    "workflow_run_id": 34159688896,
    "artifact_id": 10032567466,
    "artifact_digest": "sha256:5d95cffacd815f2ca2afbd657d2ad04145a4e51303b63513b863eb897fa1f4dd",
    "probe_report_sha256": "1c33b31844b11822479adb1382a683a430cee470bbdf23f259084f8b5ebf0458",
    "probe_config_identity_sha256": "c2f198120cae00ba247c4eaad36d2a357770a47c7fa9a7608cc5ec182971b82b",
}


class ObservationPinError(RuntimeError):
    """Raised when the observation cannot be pinned without weakening gates."""


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_json(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ObservationPinError(f"cannot load JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ObservationPinError(f"JSON root must be an object: {path}")
    return value, raw


def _require_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise ObservationPinError(f"{field} must be a lowercase SHA-256")
    return value


def _validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise ObservationPinError("unsupported observation-pin config schema")
    if config.get("local_free_only") is not True:
        raise ObservationPinError("observation pin must remain LOCAL_FREE")

    parent = config.get("parent_probe")
    snapshot = config.get("exact_snapshot")
    observation = config.get("expected_observation")
    policy = config.get("pin_policy")
    boundary = config.get("claim_boundary")
    for name, value in (
        ("parent_probe", parent),
        ("exact_snapshot", snapshot),
        ("expected_observation", observation),
        ("pin_policy", policy),
        ("claim_boundary", boundary),
    ):
        if not isinstance(value, Mapping):
            raise ObservationPinError(f"{name} must be an object")

    for field, expected in EXPECTED_PARENT.items():
        if parent.get(field) != expected:
            raise ObservationPinError(f"parent_probe.{field} drifted from terminal #830 authority")

    if (
        isinstance(snapshot.get("archive_bytes"), bool)
        or not isinstance(snapshot.get("archive_bytes"), int)
        or snapshot["archive_bytes"] <= 0
    ):
        raise ObservationPinError("exact_snapshot.archive_bytes must be positive")
    archive_md5 = snapshot.get("archive_md5")
    if not isinstance(archive_md5, str) or not re.fullmatch(
        r"[0-9a-f]{32}", archive_md5
    ):
        raise ObservationPinError("exact_snapshot.archive_md5 must be lowercase MD5")
    _require_sha256(snapshot.get("archive_sha256"), "exact_snapshot.archive_sha256")
    for field in ("canonical_entry_count", "canonical_raw_bytes"):
        value = snapshot.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ObservationPinError(f"exact_snapshot.{field} must be positive")
    _require_sha256(
        snapshot.get("entry_identity_sha256"),
        "exact_snapshot.entry_identity_sha256",
    )

    expected_pairs: dict[str, object] = {
        "source_family": "ua.rada.open-data.laws-texts",
        "exact_archive_identity": "OBSERVED_UNPINNED",
        "safe_zip_inventory": "PASS",
        "discovery_capacity_threshold": "FAIL_BELOW_MINIMUM",
        "safe_result": (
            "CURRENT_UPSTREAM_OBSERVED_BELOW_DISCOVERY_MINIMUM_"
            "SUCCESSOR_TRIAGE_REQUIRED"
        ),
        "discovery_observation_revalidated": False,
    }
    for field, expected in expected_pairs.items():
        if observation.get(field) != expected:
            raise ObservationPinError(
                f"expected_observation.{field} weakened or drifted"
            )

    required_policy = {
        "purpose": "NORMALIZATION_INPUT_ONLY",
        "recompute_entry_identity": True,
        "verify_every_canonical_entry_against_archive": True,
        "preserve_failed_capacity_threshold": True,
        "derived_exact_archive_identity": "PASS_SUCCESSOR_EXACT_ARTIFACT_PIN",
        "derived_safe_result": (
            "SUCCESSOR_PINNED_EXACT_ARCHIVE_NORMALIZATION_ONLY_"
            "CAPACITY_STILL_UNCREDITED"
        ),
    }
    for field, expected in required_policy.items():
        if policy.get(field) != expected:
            raise ObservationPinError(f"pin_policy.{field} weakened or drifted")

    required_false = (
        "corpus_admitted",
        "normalized_capacity_credited",
        "tokenizer_fit_authorized",
        "model_training_executed",
        "paid_compute_used",
        "research_corpus_v1_released",
        "learned_20m_claimed",
    )
    if boundary.get("training_authorized_bytes") != 0:
        raise ObservationPinError("pin must authorize zero training bytes")
    for field in required_false:
        if boundary.get(field) is not False:
            raise ObservationPinError(f"truth boundary weakened: {field}")


def _inventory_identity(
    entries: list[Mapping[str, Any]],
) -> tuple[str, int, dict[str, Mapping[str, Any]]]:
    by_basename: dict[str, Mapping[str, Any]] = {}
    for entry in entries:
        basename = entry.get("basename")
        if not isinstance(basename, str) or not CANONICAL_BASENAME.fullmatch(basename):
            raise ObservationPinError("probe contains noncanonical basename")
        if basename in by_basename:
            raise ObservationPinError(f"duplicate probe basename: {basename}")
        raw_bytes = entry.get("raw_bytes")
        raw_sha = entry.get("raw_sha256")
        path = entry.get("path")
        if isinstance(raw_bytes, bool) or not isinstance(raw_bytes, int) or raw_bytes < 0:
            raise ObservationPinError(f"invalid raw byte count for {basename}")
        _require_sha256(raw_sha, f"inventory.{basename}.raw_sha256")
        if not isinstance(path, str) or Path(path.replace("\\", "/")).name != basename:
            raise ObservationPinError(f"invalid inventory path for {basename}")
        by_basename[basename] = entry

    digest = hashlib.sha256()
    raw_total = 0
    for basename in sorted(by_basename):
        entry = by_basename[basename]
        raw_bytes = int(entry["raw_bytes"])
        digest.update(basename.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(raw_bytes).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(entry["raw_sha256"]).encode("ascii"))
        digest.update(b"\n")
        raw_total += raw_bytes
    return digest.hexdigest(), raw_total, by_basename


def pin_observation(
    archive: bytes,
    probe: Mapping[str, Any],
    probe_bytes: bytes,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify and convert one zero-credit observation into an exact normalization pin."""
    _validate_config(config)
    parent = config["parent_probe"]
    snapshot = config["exact_snapshot"]
    expected = config["expected_observation"]

    if probe.get("schema_version") != PROBE_SCHEMA:
        raise ObservationPinError("probe report schema drift")
    if _sha256(probe_bytes) != parent["probe_report_sha256"]:
        raise ObservationPinError("probe report SHA-256 mismatch")
    if probe.get("config_identity_sha256") != parent["probe_config_identity_sha256"]:
        raise ObservationPinError("probe config identity drift")
    if probe.get("source_family") != expected["source_family"]:
        raise ObservationPinError("source family drift")
    if probe.get("discovery_observation_revalidated") is not False:
        raise ObservationPinError("observation must be the zero-credit mutable-upstream run")
    if probe.get("safe_result") != expected["safe_result"]:
        raise ObservationPinError("observation safe result drift")

    gates = probe.get("gates")
    if not isinstance(gates, Mapping):
        raise ObservationPinError("probe gates missing")
    for field in (
        "exact_archive_identity",
        "safe_zip_inventory",
        "discovery_capacity_threshold",
    ):
        if gates.get(field) != expected[field]:
            raise ObservationPinError(f"probe gate drift: {field}")
    for field in (
        "canonical_normalization",
        "quality",
        "privacy",
        "global_cross_source_dedup",
        "evaluation_decontamination",
        "balance_diversity",
        "corpus_materialization",
        "unique_loss_ledger",
    ):
        if gates.get(field) != "NOT_RUN":
            raise ObservationPinError(f"unexpected downstream gate state: {field}")

    if probe.get("training_authorized_bytes") != 0:
        raise ObservationPinError("probe unexpectedly grants training bytes")
    for field in (
        "corpus_admitted",
        "tokenizer_fit_authorized",
        "model_training_executed",
        "paid_compute_used",
    ):
        if probe.get(field) is not False:
            raise ObservationPinError(f"probe truth boundary weakened: {field}")

    archive_meta = probe.get("archive")
    inventory = probe.get("inventory")
    if not isinstance(archive_meta, Mapping) or not isinstance(inventory, Mapping):
        raise ObservationPinError("probe archive/inventory metadata missing")
    if len(archive) != snapshot["archive_bytes"] or archive_meta.get("bytes") != len(
        archive
    ):
        raise ObservationPinError("archive byte identity mismatch")
    if hashlib.md5(archive, usedforsecurity=False).hexdigest() != snapshot["archive_md5"]:
        raise ObservationPinError("archive MD5 mismatch")
    if (
        _sha256(archive) != snapshot["archive_sha256"]
        or archive_meta.get("sha256") != snapshot["archive_sha256"]
    ):
        raise ObservationPinError("archive SHA-256 mismatch")
    if archive_meta.get("md5") != snapshot["archive_md5"]:
        raise ObservationPinError("probe archive MD5 drift")

    raw_entries = inventory.get("entries")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ObservationPinError("probe inventory entries missing")
    entries: list[Mapping[str, Any]] = []
    for entry in raw_entries:
        if not isinstance(entry, Mapping):
            raise ObservationPinError("probe inventory entry must be an object")
        entries.append(entry)
    identity, raw_total, by_basename = _inventory_identity(entries)
    if len(by_basename) != snapshot["canonical_entry_count"]:
        raise ObservationPinError("canonical entry count mismatch")
    if inventory.get("canonical_entry_count") != len(by_basename):
        raise ObservationPinError("probe canonical entry count drift")
    if (
        raw_total != snapshot["canonical_raw_bytes"]
        or inventory.get("canonical_raw_bytes") != raw_total
    ):
        raise ObservationPinError("canonical raw byte total mismatch")
    if (
        identity != snapshot["entry_identity_sha256"]
        or inventory.get("entry_identity_sha256") != identity
    ):
        raise ObservationPinError("entry inventory identity mismatch")

    try:
        archive_file = zipfile.ZipFile(io.BytesIO(archive))
    except zipfile.BadZipFile as exc:
        raise ObservationPinError("archive is not a valid ZIP") from exc
    observed: set[str] = set()
    with archive_file as zf:
        for info in zf.infolist():
            basename = Path(info.filename.replace("\\", "/")).name
            if not CANONICAL_BASENAME.fullmatch(basename):
                continue
            if basename in observed:
                raise ObservationPinError(
                    f"duplicate archive canonical basename: {basename}"
                )
            observed.add(basename)
            expected_entry = by_basename.get(basename)
            if expected_entry is None:
                raise ObservationPinError(
                    f"archive has unreported canonical entry: {basename}"
                )
            if expected_entry.get("path") != info.filename.replace("\\", "/"):
                raise ObservationPinError(f"archive path mismatch for {basename}")
            try:
                raw = zf.read(info)
            except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
                raise ObservationPinError(f"cannot read archive member: {basename}") from exc
            if len(raw) != expected_entry.get("raw_bytes"):
                raise ObservationPinError(
                    f"archive member byte mismatch for {basename}"
                )
            if _sha256(raw) != expected_entry.get("raw_sha256"):
                raise ObservationPinError(
                    f"archive member SHA-256 mismatch for {basename}"
                )
    if observed != set(by_basename):
        raise ObservationPinError("archive is missing one or more probed canonical entries")

    derived = json.loads(json.dumps(probe, ensure_ascii=False))
    derived["gates"]["exact_archive_identity"] = DERIVED_ARCHIVE_GATE
    # Exact pinning does not convert the old discovery-count heuristic into PASS.
    derived["safe_result"] = DERIVED_SAFE_RESULT
    derived["successor_pin"] = {
        "schema_version": CONFIG_SCHEMA,
        "purpose": "NORMALIZATION_INPUT_ONLY",
        "parent_probe_pr": parent["pr"],
        "parent_probe_head_sha": parent["head_sha"],
        "workflow_run_id": parent["workflow_run_id"],
        "artifact_id": parent["artifact_id"],
        "artifact_digest": parent["artifact_digest"],
        "source_probe_report_sha256": parent["probe_report_sha256"],
        "pin_config_identity_sha256": _sha256(_canonical_json_bytes(config)),
        "source_observation_exact_archive_identity": expected["exact_archive_identity"],
        "source_observation_safe_result": expected["safe_result"],
        "source_observation_revalidated": expected[
            "discovery_observation_revalidated"
        ],
        "preserved_discovery_capacity_threshold": expected[
            "discovery_capacity_threshold"
        ],
        "derived_pin_status": config["pin_policy"]["derived_exact_archive_identity"],
        "derived_pin_safe_result": config["pin_policy"]["derived_safe_result"],
        "training_authorized_bytes": 0,
        "normalized_capacity_credited": 0,
    }
    return derived


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--probe-report", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config, _ = _load_json(args.config)
    probe, probe_bytes = _load_json(args.probe_report)
    try:
        archive = args.archive.read_bytes()
    except OSError as exc:
        raise ObservationPinError(f"cannot read archive: {args.archive}") from exc
    derived = pin_observation(archive, probe, probe_bytes, config)
    output_bytes = json.dumps(
        derived, ensure_ascii=False, sort_keys=True, indent=2
    ).encode("utf-8") + b"\n"
    _atomic_write(args.output, output_bytes)
    print(
        json.dumps(
            {
                "status": "PASS_NORMALIZATION_INPUT_PIN_ONLY",
                "archive_sha256": derived["archive"]["sha256"],
                "canonical_entry_count": derived["inventory"][
                    "canonical_entry_count"
                ],
                "entry_identity_sha256": derived["inventory"][
                    "entry_identity_sha256"
                ],
                "discovery_capacity_threshold": derived["gates"][
                    "discovery_capacity_threshold"
                ],
                "pinned_report_sha256": _sha256(output_bytes),
                "training_authorized_bytes": 0,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
