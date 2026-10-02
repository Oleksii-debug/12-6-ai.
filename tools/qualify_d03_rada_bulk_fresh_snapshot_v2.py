#!/usr/bin/env python3
"""Qualify two fresh Rada source probes without granting corpus/training credit."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

CONFIG_SCHEMA = "12-6.d03-rada-bulk-fresh-snapshot-capture.v2"
PROBE_SCHEMA = "12-6.d03-rada-bulk-source-probe-report.v1"
OUTPUT_SCHEMA = "12-6.d03-rada-bulk-fresh-snapshot-qualification.v2"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
MD5_RE = re.compile(r"^[0-9a-f]{32}$")
CRC32_RE = re.compile(r"^[0-9a-f]{8}$")
NAME_RE = re.compile(r"^d[0-9]+\.htm$")
NOT_RUN_GATES = (
    "canonical_normalization",
    "quality",
    "privacy",
    "global_cross_source_dedup",
    "evaluation_decontamination",
    "balance_diversity",
    "corpus_materialization",
    "unique_loss_ledger",
)


class FreshSnapshotQualificationError(RuntimeError):
    """Fail-closed fresh-snapshot qualification error."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FreshSnapshotQualificationError(message)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        _require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _float(raw: str) -> float:
    value = float(raw)
    _require(math.isfinite(value), "non-finite JSON number")
    return value


def _constant(raw: str) -> None:
    raise FreshSnapshotQualificationError(f"non-standard JSON constant: {raw}")


def _strict_json(raw: bytes, *, label: str) -> dict[str, Any]:
    _require(type(raw) is bytes and bool(raw), f"{label} is empty")
    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_pairs,
            parse_float=_float,
            parse_constant=_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise FreshSnapshotQualificationError(f"{label} is not strict JSON") from exc
    _require(type(value) is dict, f"{label} root must be exact object")
    return value


_strict_json_bytes = _strict_json


def _read(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise FreshSnapshotQualificationError(f"cannot read {label}: {path}") from exc
    return _strict_json(raw, label=label), raw


def _int(value: Any, *, label: str, minimum: int = 0) -> int:
    _require(type(value) is int and value >= minimum, f"{label} invalid")
    return value


def _hex(value: Any, regex: re.Pattern[str], *, label: str) -> str:
    _require(type(value) is str and regex.fullmatch(value) is not None, f"{label} malformed")
    return value


def _validate_config(config: Mapping[str, Any]) -> None:
    _require(config.get("schema_version") == CONFIG_SCHEMA, "capture config schema drift")
    source = config.get("source")
    old = config.get("predecessor_authority")
    current = config.get("expected_current_observation")
    probe = config.get("probe_authority")
    truth = config.get("truth_boundary")
    for label, value in (("source", source), ("predecessor", old), ("current", current),
                         ("probe", probe), ("truth", truth)):
        _require(type(value) is dict, f"{label} config missing")
    _require(source.get("family_id") == "ua.rada.open-data.laws-texts", "source family drift")
    _require(source.get("dataset_id") == "laws-texts", "source dataset drift")
    _require(
        source.get("archive_url") == "https://data.rada.gov.ua/ogd/zak/perv/text/texts.zip",
        "source URL drift",
    )
    _require(old.get("historical_qp_pr") == 1787, "historical Q/P predecessor drift")
    _require(old.get("historical_execution_carrier_pr") == 2230, "carrier predecessor drift")
    _int(old.get("archive_bytes"), label="historical archive bytes", minimum=1)
    old_sha = _hex(old.get("archive_sha256"), SHA256_RE, label="historical archive SHA")
    _int(current.get("archive_bytes"), label="current archive bytes", minimum=1)
    new_sha = _hex(current.get("archive_sha256"), SHA256_RE, label="current archive SHA")
    _require(new_sha != old_sha, "fresh observation must not rewrite historical snapshot")
    _require(current.get("observed_in_issue") == 2019, "observation issue drift")
    _require(current.get("observation_comment_id") == 5938332653, "observation comment drift")
    _require(probe.get("schema_version") == PROBE_SCHEMA, "probe schema authority drift")
    _require(probe.get("worker_id") == "D03-RADA-BULK-SOURCE-PROBE-20260826", "probe worker drift")
    _hex(probe.get("config_identity_sha256"), SHA256_RE, label="probe config identity")
    _hex(probe.get("parent_head_sha"), re.compile(r"^[0-9a-f]{40}$"), label="probe parent")
    _hex(probe.get("parent_registry_identity_sha256"), SHA256_RE, label="probe registry")
    required = {
        "local_free_only": True,
        "canonical_capacity_credited": 0,
        "training_authorized_bytes": 0,
        "authorized_optimized_target_exposure": 0,
        "tokenizer_fit_authorized": False,
        "optimizer_updates_executed_on_real_targets": 0,
        "training_executed": False,
        "learned_weights_created": False,
        "final_test_outcomes_read": False,
        "paid_compute_used": False,
        "foreign_pretrained_weights": False,
    }
    _require(truth == required, "capture truth boundary drift")


def _inventory(report: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    inv = report.get("inventory")
    _require(type(inv) is dict, f"{label} inventory missing")
    entries = inv.get("entries")
    _require(type(entries) is list and bool(entries), f"{label} entries missing")
    digest = hashlib.sha256()
    total = 0
    names: list[str] = []
    for index, entry in enumerate(entries):
        _require(type(entry) is dict, f"{label} entry {index} malformed")
        name = entry.get("basename")
        path = entry.get("path")
        _require(type(name) is str and NAME_RE.fullmatch(name) is not None, f"{label} name drift")
        _require(
            type(path) is str and Path(path.replace("\\", "/")).name == name,
            f"{label} path drift",
        )
        size = _int(entry.get("raw_bytes"), label=f"{label} entry bytes")
        sha = _hex(entry.get("raw_sha256"), SHA256_RE, label=f"{label} entry SHA")
        _hex(entry.get("crc32"), CRC32_RE, label=f"{label} CRC32")
        names.append(name)
        digest.update(name.encode("utf-8") + b"\0" + str(size).encode("ascii") + b"\0")
        digest.update(sha.encode("ascii") + b"\n")
        total += size
    _require(
        names == sorted(names) and len(set(names)) == len(names),
        f"{label} order/duplicate drift",
    )
    _require(inv.get("canonical_entry_count") == len(entries), f"{label} entry count drift")
    _require(inv.get("canonical_raw_bytes") == total, f"{label} raw-byte total drift")
    _require(
        inv.get("entry_identity_sha256") == digest.hexdigest(),
        f"{label} entry identity mismatch",
    )
    _int(inv.get("ignored_file_count"), label=f"{label} ignored count")
    total_zip = _int(inv.get("total_zip_uncompressed_bytes"), label=f"{label} ZIP bytes")
    _require(total_zip >= total, f"{label} ZIP total underflow")
    return dict(inv)


def _validate_report(
    report: Mapping[str, Any], config: Mapping[str, Any], *, label: str
) -> dict[str, Any]:
    expected_keys = {
        "schema_version", "worker_id", "config_identity_sha256",
        "parent_authority", "source_family", "source_dataset_id",
        "archive", "inventory", "gates", "training_authorized_bytes",
        "corpus_admitted", "tokenizer_fit_authorized",
        "model_training_executed", "paid_compute_used", "safe_result",
        "http_response", "discovery_observation_revalidated",
    }
    _require(set(report) == expected_keys, f"{label} schema drift")
    probe = config["probe_authority"]
    source = config["source"]
    current = config["expected_current_observation"]
    _require(report.get("schema_version") == PROBE_SCHEMA, f"{label} schema drift")
    _require(report.get("worker_id") == probe["worker_id"], f"{label} worker drift")
    _require(
        report.get("config_identity_sha256") == probe["config_identity_sha256"],
        f"{label} config drift",
    )
    _require(report.get("source_family") == source["family_id"], f"{label} family drift")
    _require(report.get("source_dataset_id") == source["dataset_id"], f"{label} dataset drift")
    parent = report.get("parent_authority")
    _require(
        type(parent) is dict and parent.get("head_sha") == probe["parent_head_sha"],
        f"{label} parent drift",
    )
    _require(
        parent.get("registry_identity_sha256") == probe["parent_registry_identity_sha256"],
        f"{label} registry drift",
    )
    archive = report.get("archive")
    _require(
        type(archive) is dict and archive.get("url") == source["archive_url"],
        f"{label} archive drift",
    )
    _require(
        type(archive.get("bytes")) is int
        and archive["bytes"] == current["archive_bytes"],
        f"{label} archive byte identity drift",
    )
    _require(archive.get("sha256") == current["archive_sha256"], f"{label} archive SHA-256 drift")
    _hex(archive.get("md5"), MD5_RE, label=f"{label} archive MD5")
    _inventory(report, label=label)
    gates = report.get("gates")
    _require(type(gates) is dict, f"{label} gates missing")
    _require(gates.get("exact_archive_identity") == "OBSERVED_UNPINNED", f"{label} not fresh")
    _require(gates.get("safe_zip_inventory") == "PASS", f"{label} ZIP safety not PASS")
    capacity = gates.get("discovery_capacity_threshold")
    _require(capacity in {"PASS", "FAIL_BELOW_MINIMUM"}, f"{label} capacity gate drift")
    for gate in NOT_RUN_GATES:
        _require(gates.get(gate) == "NOT_RUN", f"{label} downstream gate ran: {gate}")
    expected_safe = (
        "CURRENT_UPSTREAM_OBSERVED_SUCCESSOR_PIN_REQUIRED"
        if capacity == "PASS"
        else "CURRENT_UPSTREAM_OBSERVED_BELOW_DISCOVERY_MINIMUM_SUCCESSOR_TRIAGE_REQUIRED"
    )
    _require(report.get("safe_result") == expected_safe, f"{label} safe result drift")
    _require(
        report.get("discovery_observation_revalidated") is False,
        f"{label} old snapshot claimed",
    )
    _require(
        type(report.get("training_authorized_bytes")) is int
        and report["training_authorized_bytes"] == 0,
        f"{label} training bytes widened",
    )
    for field in (
        "corpus_admitted",
        "tokenizer_fit_authorized",
        "model_training_executed",
        "paid_compute_used",
    ):
        _require(report.get(field) is False, f"{label} truth widened: {field}")
    _require(type(report.get("http_response")) is dict, f"{label} HTTP evidence missing")
    return {key: report[key] for key in report if key != "http_response"}


def qualify_two_clean_probes(
    config: Mapping[str, Any],
    probe_a: Mapping[str, Any],
    probe_b: Mapping[str, Any],
    *,
    probe_a_bytes: bytes,
    probe_b_bytes: bytes,
) -> dict[str, Any]:
    _validate_config(config)
    stable_a = _validate_report(probe_a, config, label="probe A")
    stable_b = _validate_report(probe_b, config, label="probe B")
    _require(stable_a == stable_b, "two fresh probes did not converge semantically")
    archive = stable_a["archive"]
    inv = stable_a["inventory"]
    truth = config["truth_boundary"]
    core = {
        "schema_version": OUTPUT_SCHEMA,
        "status": "QUALIFIED_EXACT_MUTABLE_SOURCE_SNAPSHOT_ZERO_CREDIT",
        "execution_profile": "LOCAL_FREE",
        "source_family": config["source"]["family_id"],
        "archive_bytes": archive["bytes"],
        "archive_md5": archive["md5"],
        "archive_sha256": archive["sha256"],
        "canonical_entry_count": inv["canonical_entry_count"],
        "canonical_raw_bytes": inv["canonical_raw_bytes"],
        "entry_identity_sha256": inv["entry_identity_sha256"],
        "discovery_capacity_threshold": stable_a["gates"]["discovery_capacity_threshold"],
        "stable_probe_projection_sha256": _sha256(_canonical(stable_a)),
        "probe_a_file_sha256": _sha256(probe_a_bytes),
        "probe_b_file_sha256": _sha256(probe_b_bytes),
        "two_fresh_probes_semantically_identical": True,
        "historical_qp_authority_preserved": True,
        "historical_archive_sha256": config["predecessor_authority"]["archive_sha256"],
        "historical_qp_pr": 1787,
        "historical_execution_carrier_pr": 2230,
        "source_snapshot_repin_required": True,
        "rights_recheck_required": True,
        "normalization_reexecution_required": True,
        "quality_privacy_reexecution_required": True,
        "qp_reprojection_required": True,
        "global_dedup_reexecution_required": True,
        "canonical_capacity_credited": truth["canonical_capacity_credited"],
        "training_authorized_bytes": truth["training_authorized_bytes"],
        "authorized_optimized_target_exposure": truth["authorized_optimized_target_exposure"],
        "tokenizer_fit_authorized": truth["tokenizer_fit_authorized"],
        "optimizer_updates_executed_on_real_targets": truth[
            "optimizer_updates_executed_on_real_targets"
        ],
        "training_executed": truth["training_executed"],
        "learned_weights_created": truth["learned_weights_created"],
        "final_test_outcomes_read": truth["final_test_outcomes_read"],
        "paid_compute_used": truth["paid_compute_used"],
        "foreign_pretrained_weights": truth["foreign_pretrained_weights"],
    }
    return {**core, "evidence_identity_sha256": _sha256(_canonical(core))}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--probe-a", type=Path, required=True)
    parser.add_argument("--probe-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config, _ = _read(args.config, label="capture config")
    probe_a, raw_a = _read(args.probe_a, label="probe A")
    probe_b, raw_b = _read(args.probe_b, label="probe B")
    evidence = qualify_two_clean_probes(
        config, probe_a, probe_b, probe_a_bytes=raw_a, probe_b_bytes=raw_b
    )
    output = json.dumps(evidence, sort_keys=True, indent=2).encode() + b"\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.output.open("xb") as stream:
            stream.write(output)
    except FileExistsError as exc:
        raise FreshSnapshotQualificationError("qualification output already exists") from exc
    summary = {
        "status": evidence["status"],
        "archive_sha256": evidence["archive_sha256"],
        "entry_identity_sha256": evidence["entry_identity_sha256"],
        "evidence_identity_sha256": evidence["evidence_identity_sha256"],
        "training_authorized_bytes": 0,
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
