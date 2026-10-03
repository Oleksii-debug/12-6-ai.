#!/usr/bin/env python3
"""Qualify two fresh Rada source probes and attributed snapshot retention."""
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
RIGHTS_SCHEMA = "12-6.d03-rada-bulk-fresh-snapshot-rights.v2"
RIGHTS_AUTHORITY = "D03-RADA-BULK-FRESH-SNAPSHOT-RIGHTS-V2"
RIGHTS_STATUS = "SOURCE_POLICY_QUALIFIED_FOR_ATTRIBUTED_SNAPSHOT_RETENTION_ZERO_CREDIT"
RIGHTS_IDENTITY = "47e43afc87e798a52be1745d4313f353a349bc957e809126369a3923ffb68d0f"
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
TRUTH = {
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
    expected_keys = {
        "schema_version",
        "source",
        "predecessor_authority",
        "prior_mutable_observation",
        "capture_policy",
        "probe_authority",
        "truth_boundary",
    }
    _require(set(config) == expected_keys, "capture config root schema drift")
    source = config.get("source")
    old = config.get("predecessor_authority")
    prior = config.get("prior_mutable_observation")
    policy = config.get("capture_policy")
    probe = config.get("probe_authority")
    truth = config.get("truth_boundary")
    for label, value in (
        ("source", source),
        ("predecessor", old),
        ("prior", prior),
        ("capture policy", policy),
        ("probe", probe),
        ("truth", truth),
    ):
        _require(type(value) is dict, f"{label} config missing")
    _require(source.get("family_id") == "ua.rada.open-data.laws-texts", "source family drift")
    _require(source.get("dataset_id") == "laws-texts", "source dataset drift")
    _require(
        source.get("archive_url") == "https://data.rada.gov.ua/ogd/zak/perv/text/texts.zip",
        "source URL drift",
    )
    _require(
        source.get("upstream_mutability")
        == "FREQUENTLY_UPDATED_REQUIRES_TWO_CLEAN_LIVE_READS_AND_EXACT_RETENTION",
        "source mutability policy drift",
    )
    _require(old.get("historical_qp_pr") == 1787, "historical Q/P predecessor drift")
    _require(old.get("historical_execution_carrier_pr") == 2230, "carrier predecessor drift")
    _int(old.get("archive_bytes"), label="historical archive bytes", minimum=1)
    old_sha = _hex(old.get("archive_sha256"), SHA256_RE, label="historical archive SHA")
    _int(prior.get("archive_bytes"), label="prior archive bytes", minimum=1)
    prior_sha = _hex(prior.get("archive_sha256"), SHA256_RE, label="prior archive SHA")
    _require(prior_sha != old_sha, "prior mutable observation rewrites historical authority")
    _require(prior.get("observed_in_issue") == 2019, "prior observation issue drift")
    _require(prior.get("observation_comment_id") == 5938332653, "prior observation comment drift")
    _require(
        prior.get("status") == "SUPERSEDED_MUTABLE_OBSERVATION_DO_NOT_PIN_AS_CURRENT",
        "prior observation status drift",
    )
    expected_policy = {
        "establish_exact_identity_from_two_clean_live_reads": True,
        "require_byte_identical_archives": True,
        "require_semantically_identical_probes": True,
        "retained_artifact_requires_attribution": True,
        "rights_policy_path": "configs/data/d03_rada_bulk_fresh_snapshot_rights_v2.json",
        "historical_qp_authority_is_immutable": True,
    }
    _require(policy == expected_policy, "capture policy drift")
    _require(probe.get("schema_version") == PROBE_SCHEMA, "probe schema authority drift")
    _require(
        probe.get("worker_id") == "D03-RADA-BULK-SOURCE-PROBE-20260826",
        "probe worker drift",
    )
    _hex(probe.get("config_identity_sha256"), SHA256_RE, label="probe config identity")
    _hex(probe.get("parent_head_sha"), re.compile(r"^[0-9a-f]{40}$"), label="probe parent")
    _hex(probe.get("parent_registry_identity_sha256"), SHA256_RE, label="probe registry")
    expected_truth = {"local_free_only": True, **TRUTH}
    _require(truth == expected_truth, "capture truth boundary drift")


def _validate_rights(rights: Mapping[str, Any]) -> str:
    expected_keys = {
        "schema_version",
        "authority_id",
        "status",
        "execution_profile",
        "source",
        "primary_evidence",
        "project_decision",
        "attribution",
        "truth_boundary",
        "policy_identity_sha256",
    }
    _require(set(rights) == expected_keys, "rights policy root schema drift")
    _require(rights.get("schema_version") == RIGHTS_SCHEMA, "rights schema drift")
    _require(rights.get("authority_id") == RIGHTS_AUTHORITY, "rights authority drift")
    _require(rights.get("status") == RIGHTS_STATUS, "rights status drift")
    _require(rights.get("execution_profile") == "LOCAL_FREE", "rights execution profile drift")
    source = rights.get("source")
    evidence = rights.get("primary_evidence")
    decision = rights.get("project_decision")
    attribution = rights.get("attribution")
    truth = rights.get("truth_boundary")
    for label, value in (
        ("rights source", source),
        ("rights evidence", evidence),
        ("rights decision", decision),
        ("attribution", attribution),
        ("rights truth", truth),
    ):
        expected_type = list if label == "rights evidence" else dict
        _require(type(value) is expected_type, f"{label} malformed")
    assert isinstance(source, dict)
    assert isinstance(evidence, list)
    assert isinstance(decision, dict)
    assert isinstance(attribution, dict)
    assert isinstance(truth, dict)
    _require(
        source
        == {
            "dataset_id": "laws-texts",
            "title": 'Тексти первинних законів бази даних "Законодавство України"',
            "publisher": "Апарат Верховної Ради України",
            "dataset_page_url": (
                "https://data.rada.gov.ua/open/data/en/laws-texts/page?lang=en"
            ),
            "archive_url": "https://data.rada.gov.ua/ogd/zak/perv/text/texts.zip",
            "source_family": "ua.rada.open-data.laws-texts",
        },
        "rights source drift",
    )
    _require(len(evidence) == 1 and type(evidence[0]) is dict, "rights evidence count drift")
    _require(
        evidence[0]
        == {
            "authority": "VERKHOVNA_RADA_OPEN_DATA_DATASET_PAGE",
            "url": "https://data.rada.gov.ua/open/data/en/laws-texts/page?lang=en",
            "observed_at_utc_date": "2026-10-02",
            "reuse_scope": "FREE_USE_REUSE_REDISTRIBUTION_INCLUDING_COMMERCIAL",
            "condition": "MANDATORY_SOURCE_ATTRIBUTION",
            "license": "CC-BY-4.0_UNLESS_OTHERWISE_SPECIFIED",
            "observation_binding": "CURRENT_OFFICIAL_PRIMARY_PORTAL_OBSERVATION_NON_IMMUTABLE",
        },
        "rights primary evidence drift",
    )
    _require(
        decision
        == {
            "scope": "ARTIFACT_RETENTION_AND_REPRODUCIBILITY_ONLY",
            "rights_recheck_status": "PASS_ATTRIBUTED_RETENTION_ONLY",
            "legal_conclusion_claimed": False,
            "raw_snapshot_retention_requires_attribution": True,
            "bulk_corpus_admission_granted": False,
            "training_authority_granted": False,
            "downstream_rights_and_provenance_recheck_required": True,
        },
        "rights project decision drift",
    )
    _require(
        attribution
        == {
            "creator": "Апарат Верховної Ради України",
            "source_page": "https://data.rada.gov.ua/open/data/en/laws-texts/page?lang=en",
            "source_archive": "https://data.rada.gov.ua/ogd/zak/perv/text/texts.zip",
            "license": (
                "Creative Commons Attribution 4.0 International unless otherwise specified"
            ),
            "license_uri": "https://creativecommons.org/licenses/by/4.0/",
            "notice": "Source: Verkhovna Rada of Ukraine Open Data Portal",
        },
        "attribution drift",
    )
    _require(truth == TRUTH, "rights truth boundary drift")
    identity = rights.get("policy_identity_sha256")
    _hex(identity, SHA256_RE, label="rights policy identity")
    core = dict(rights)
    core.pop("policy_identity_sha256")
    _require(_sha256(_canonical(core)) == identity, "rights policy identity mismatch")
    _require(identity == RIGHTS_IDENTITY, "rights policy identity constant drift")
    return str(identity)


def attribution_text(rights: Mapping[str, Any], archive: Mapping[str, Any]) -> str:
    _validate_rights(rights)
    attr = rights["attribution"]
    source = rights["source"]
    return (
        f"{source['title']}\n"
        f"Creator/attribution: {attr['creator']}\n"
        f"Source page: {attr['source_page']}\n"
        f"Source archive: {attr['source_archive']}\n"
        f"Captured archive bytes: {archive['bytes']}\n"
        f"Captured archive SHA-256: {archive['sha256']}\n"
        f"License: {attr['license']}\n"
        f"License URI: {attr['license_uri']}\n"
        f"{attr['notice']}\n"
        "Project scope: retained exact source snapshot for reproducibility only; "
        "this notice grants no corpus or training credit.\n"
    )


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
        path_value = entry.get("path")
        _require(
            type(name) is str and NAME_RE.fullmatch(name) is not None,
            f"{label} name drift",
        )
        _require(
            type(path_value) is str and Path(path_value.replace("\\", "/")).name == name,
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
        "schema_version",
        "worker_id",
        "config_identity_sha256",
        "parent_authority",
        "source_family",
        "source_dataset_id",
        "archive",
        "inventory",
        "gates",
        "training_authorized_bytes",
        "corpus_admitted",
        "tokenizer_fit_authorized",
        "model_training_executed",
        "paid_compute_used",
        "safe_result",
        "http_response",
        "discovery_observation_revalidated",
    }
    _require(set(report) == expected_keys, f"{label} schema drift")
    probe = config["probe_authority"]
    source = config["source"]
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
    _int(archive.get("bytes"), label=f"{label} archive bytes", minimum=1)
    _hex(archive.get("sha256"), SHA256_RE, label=f"{label} archive SHA-256")
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
        f"{label} historical snapshot claimed",
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
    rights: Mapping[str, Any],
    probe_a: Mapping[str, Any],
    probe_b: Mapping[str, Any],
    *,
    probe_a_bytes: bytes,
    probe_b_bytes: bytes,
) -> dict[str, Any]:
    _validate_config(config)
    rights_identity = _validate_rights(rights)
    _require(
        config["capture_policy"]["rights_policy_path"]
        == "configs/data/d03_rada_bulk_fresh_snapshot_rights_v2.json",
        "rights policy binding drift",
    )
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
        "exact_identity_established_by_two_clean_live_reads": True,
        "prior_mutable_observation_treated_as_noncurrent": True,
        "prior_mutable_observation_sha256": config["prior_mutable_observation"][
            "archive_sha256"
        ],
        "historical_qp_authority_preserved": True,
        "historical_archive_sha256": config["predecessor_authority"]["archive_sha256"],
        "historical_qp_pr": 1787,
        "historical_execution_carrier_pr": 2230,
        "artifact_retention_rights_recheck_status": "PASS_ATTRIBUTED_RETENTION_ONLY",
        "rights_policy_identity_sha256": rights_identity,
        "attribution_required": True,
        "corpus_training_rights_recheck_required": True,
        "source_snapshot_repin_required": True,
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


def _write_new(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(data)
    except FileExistsError as exc:
        raise FreshSnapshotQualificationError(f"refusing to overwrite output: {path}") from exc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--rights-policy", type=Path, required=True)
    parser.add_argument("--probe-a", type=Path, required=True)
    parser.add_argument("--probe-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attribution-output", type=Path, required=True)
    args = parser.parse_args()
    config, _ = _read(args.config, label="capture config")
    rights, _ = _read(args.rights_policy, label="rights policy")
    probe_a, raw_a = _read(args.probe_a, label="probe A")
    probe_b, raw_b = _read(args.probe_b, label="probe B")
    evidence = qualify_two_clean_probes(
        config,
        rights,
        probe_a,
        probe_b,
        probe_a_bytes=raw_a,
        probe_b_bytes=raw_b,
    )
    output = json.dumps(evidence, sort_keys=True, indent=2).encode() + b"\n"
    _write_new(args.output, output)
    _write_new(
        args.attribution_output,
        attribution_text(rights, probe_a["archive"]).encode("utf-8"),
    )
    summary = {
        "status": evidence["status"],
        "archive_bytes": evidence["archive_bytes"],
        "archive_sha256": evidence["archive_sha256"],
        "entry_identity_sha256": evidence["entry_identity_sha256"],
        "rights_policy_identity_sha256": evidence["rights_policy_identity_sha256"],
        "evidence_identity_sha256": evidence["evidence_identity_sha256"],
        "training_authorized_bytes": 0,
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
