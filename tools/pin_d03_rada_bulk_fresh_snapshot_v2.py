#!/usr/bin/env python3
"""Bind one two-clean Rada capture to an exact attributed normalization-only pin."""
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

from qualify_d03_rada_bulk_fresh_snapshot_v2 import (
    FreshSnapshotQualificationError,
    _canonical,
    _strict_json,
    _validate_config,
    _validate_report,
    _validate_rights,
    attribution_text,
    qualify_two_clean_probes,
)

SCHEMA = "12-6.d03-rada-bulk-successor-pin.v2"
STATUS = "PASS_SUCCESSOR_EXACT_ARTIFACT_PIN_ATTRIBUTED_NORMALIZATION_ONLY"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SHA40_RE = re.compile(r"^[0-9a-f]{40}$")


class FreshSnapshotPinError(RuntimeError):
    """Fail-closed exact-source pinning error."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FreshSnapshotPinError(message)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _load_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = _strict_json(raw, label=label)
    except (OSError, FreshSnapshotQualificationError) as exc:
        raise FreshSnapshotPinError(f"cannot load {label}") from exc
    _require(type(value) is dict, f"{label} root must be object")
    return value, raw


def _verify_qualification(
    qualification: Mapping[str, Any],
    config: Mapping[str, Any],
    rights: Mapping[str, Any],
    probe: Mapping[str, Any],
) -> None:
    identity = qualification.get("evidence_identity_sha256")
    _require(
        type(identity) is str and SHA256_RE.fullmatch(identity) is not None,
        "qualification identity malformed",
    )
    core = dict(qualification)
    core.pop("evidence_identity_sha256")
    _require(_sha256(_canonical(core)) == identity, "qualification identity mismatch")
    _require(
        qualification.get("status")
        == "QUALIFIED_EXACT_MUTABLE_SOURCE_SNAPSHOT_ZERO_CREDIT",
        "qualification status drift",
    )
    archive = probe["archive"]
    inventory = probe["inventory"]
    expected = {
        "source_family": config["source"]["family_id"],
        "archive_bytes": archive["bytes"],
        "archive_md5": archive["md5"],
        "archive_sha256": archive["sha256"],
        "canonical_entry_count": inventory["canonical_entry_count"],
        "canonical_raw_bytes": inventory["canonical_raw_bytes"],
        "entry_identity_sha256": inventory["entry_identity_sha256"],
        "rights_policy_identity_sha256": rights["policy_identity_sha256"],
        "two_fresh_probes_semantically_identical": True,
        "exact_identity_established_by_two_clean_live_reads": True,
        "historical_qp_authority_preserved": True,
        "artifact_retention_rights_recheck_status": "PASS_ATTRIBUTED_RETENTION_ONLY",
        "attribution_required": True,
        "corpus_training_rights_recheck_required": True,
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
    for field, wanted in expected.items():
        actual = qualification.get(field)
        if type(wanted) is bool:
            matches = actual is wanted
        elif type(wanted) is int:
            matches = type(actual) is int and actual == wanted
        else:
            matches = type(actual) is type(wanted) and actual == wanted
        _require(matches, f"qualification field drift: {field}")


def _verify_archive_against_probe(archive: bytes, probe: Mapping[str, Any]) -> None:
    meta = probe.get("archive")
    inventory = probe.get("inventory")
    _require(type(meta) is dict and type(inventory) is dict, "probe archive/inventory missing")
    _require(type(meta.get("bytes")) is int and meta["bytes"] == len(archive), "archive bytes drift")
    _require(meta.get("sha256") == _sha256(archive), "archive SHA-256 drift")
    try:
        zf = zipfile.ZipFile(io.BytesIO(archive))
    except zipfile.BadZipFile as exc:
        raise FreshSnapshotPinError("retained archive is not ZIP") from exc

    entries = inventory.get("entries")
    _require(type(entries) is list and bool(entries), "probe entries missing")
    expected: dict[str, Mapping[str, Any]] = {}
    for entry in entries:
        _require(type(entry) is dict, "probe entry malformed")
        name = entry.get("basename")
        _require(type(name) is str and name not in expected, "duplicate probe basename")
        expected[name] = entry

    observed: set[str] = set()
    with zf:
        for info in zf.infolist():
            normalized_path = info.filename.replace("\\", "/")
            basename = Path(normalized_path).name
            entry = expected.get(basename)
            if entry is None:
                continue
            _require(basename not in observed, f"duplicate retained basename: {basename}")
            observed.add(basename)
            _require(entry.get("path") == normalized_path, f"retained path drift: {basename}")
            raw = zf.read(info)
            _require(entry.get("raw_bytes") == len(raw), f"retained byte drift: {basename}")
            _require(
                entry.get("raw_sha256") == _sha256(raw),
                f"retained SHA drift: {basename}",
            )
    _require(observed == set(expected), "retained archive/probe inventory mismatch")


def pin_capture(
    archive: bytes,
    probe: Mapping[str, Any],
    qualification: Mapping[str, Any],
    attribution: bytes,
    config: Mapping[str, Any],
    rights: Mapping[str, Any],
    *,
    execution_head_sha: str,
    probe_b: Mapping[str, Any],
    probe_a_bytes: bytes,
    probe_b_bytes: bytes,
) -> dict[str, Any]:
    try:
        _validate_config(config)
        _validate_rights(rights)
        _require(
            _strict_json(probe_a_bytes, label="probe A") == probe,
            "probe A raw bytes/object mismatch",
        )
        _require(
            _strict_json(probe_b_bytes, label="probe B") == probe_b,
            "probe B raw bytes/object mismatch",
        )
        expected_qualification = qualify_two_clean_probes(
            config,
            rights,
            probe,
            probe_b,
            probe_a_bytes=probe_a_bytes,
            probe_b_bytes=probe_b_bytes,
        )
    except FreshSnapshotQualificationError as exc:
        raise FreshSnapshotPinError(str(exc)) from exc
    _require(SHA40_RE.fullmatch(execution_head_sha) is not None, "execution head malformed")
    _verify_qualification(qualification, config, rights, probe)
    _require(
        qualification == expected_qualification,
        "qualification does not match two original raw probe reports",
    )
    _verify_archive_against_probe(archive, probe)

    wanted_attribution = attribution_text(rights, probe["archive"]).encode("utf-8")
    _require(attribution == wanted_attribution, "attribution bytes drift")

    inv = probe["inventory"]
    core = {
        "schema_version": SCHEMA,
        "status": STATUS,
        "execution_profile": "LOCAL_FREE",
        "execution_head_sha": execution_head_sha,
        "source_family": config["source"]["family_id"],
        "archive_bytes": len(archive),
        "archive_md5": probe["archive"]["md5"],
        "archive_sha256": _sha256(archive),
        "canonical_entry_count": inv["canonical_entry_count"],
        "canonical_raw_bytes": inv["canonical_raw_bytes"],
        "entry_identity_sha256": inv["entry_identity_sha256"],
        "source_qualification_identity_sha256": qualification["evidence_identity_sha256"],
        "rights_policy_identity_sha256": rights["policy_identity_sha256"],
        "attribution_sha256": _sha256(attribution),
        "purpose": "NORMALIZATION_INPUT_ONLY",
        "historical_qp_authority_preserved": True,
        "historical_archive_sha256": config["predecessor_authority"]["archive_sha256"],
        "prior_mutable_observation_is_noncurrent": True,
        "source_snapshot_repin_fulfilled": True,
        "normalization_reexecution_required": True,
        "quality_privacy_reexecution_required": True,
        "qp_reprojection_required": True,
        "global_dedup_reexecution_required": True,
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
    return {**core, "pin_identity_sha256": _sha256(_canonical(core))}


def _write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(payload)
    except FileExistsError as exc:
        raise FreshSnapshotPinError(f"refusing to overwrite pin output: {path}") from exc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--probe-report", type=Path, required=True)
    parser.add_argument("--probe-b", type=Path, required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--attribution", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--rights-policy", type=Path, required=True)
    parser.add_argument("--execution-head-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    try:
        archive = args.archive.read_bytes()
        attribution = args.attribution.read_bytes()
    except OSError as exc:
        raise FreshSnapshotPinError("cannot read retained capture input") from exc
    probe, raw_a = _load_json(args.probe_report, "probe A")
    probe_b, raw_b = _load_json(args.probe_b, "probe B")
    qualification, _ = _load_json(args.qualification, "qualification")
    config, _ = _load_json(args.config, "capture config")
    rights, _ = _load_json(args.rights_policy, "rights policy")
    result = pin_capture(
        archive,
        probe,
        qualification,
        attribution,
        config,
        rights,
        execution_head_sha=args.execution_head_sha,
        probe_b=probe_b,
        probe_a_bytes=raw_a,
        probe_b_bytes=raw_b,
    )
    encoded = json.dumps(result, sort_keys=True, indent=2).encode() + b"\n"
    _write_new(args.output, encoded)
    print(
        json.dumps(
            {
                "status": result["status"],
                "archive_sha256": result["archive_sha256"],
                "pin_identity_sha256": result["pin_identity_sha256"],
                "training_authorized_bytes": 0,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
