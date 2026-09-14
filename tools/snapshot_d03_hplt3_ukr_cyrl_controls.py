#!/usr/bin/env python3
"""Materialize a zero-credit HPLT3 Ukrainian map/MD5 control snapshot.

This tool snapshots only the tiny upstream control objects. It does not download
corpus shards, inspect model evaluation payloads, authorize tokenizer fitting,
or grant any training capacity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Mapping

CONFIG = Path("configs/data/d03_hplt3_ukr_cyrl_control_snapshot_v1.json")
SCHEMA = "12-6.d03-hplt3-ukr-cyrl-control-snapshot-report.v1"
CONFIG_SCHEMA = "12-6.d03-hplt3-ukr-cyrl-control-snapshot.v1"
EXPECTED_WORKER = "D03-HPLT3-UKR-CYRL-CONTROL-SNAPSHOT-20260907"
EXPECTED_SOURCE = "HPLT-3.0-ukr_Cyrl"
EXPECTED_MAP_URL = "https://data.hplt-project.org/three/sorted/ukr_Cyrl.map"
EXPECTED_MD5_URL = "https://data.hplt-project.org/three/sorted/ukr_Cyrl.md5"
SHA256_RE = re.compile(r"[0-9a-f]{64}")
MD5_RE = re.compile(r"[0-9a-fA-F]{32}")
USER_AGENT = "12-6-ai-hplt3-ukr-control-snapshot/1"


class HpltControlError(RuntimeError):
    """Fail-closed control-snapshot error."""


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise HpltControlError(message)


def load_config(path: Path = CONFIG) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HpltControlError(f"cannot load config: {path}") from exc
    _require(isinstance(value, dict), "config root must be an object")
    _require(value.get("schema_version") == CONFIG_SCHEMA, "config schema drift")
    _require(value.get("worker_id") == EXPECTED_WORKER, "worker identity drift")
    _require(value.get("execution_profile") == "LOCAL_FREE", "execution profile drift")

    parent = value.get("parent")
    _require(isinstance(parent, Mapping), "parent authority missing")
    _require(parent.get("pr") == 760, "parent PR drift")
    _require(parent.get("source_id") == EXPECTED_SOURCE, "source identity drift")
    _require(
        parent.get("source_audit_head_sha") == "45f13628830a00af6ac8539a2dc62c48be43c4ec",
        "source-audit head drift",
    )
    _require(
        parent.get("dataset_card_commit") == "3394d6ba8dae4da834e3b11771daf95028a960b1",
        "dataset-card commit drift",
    )

    controls = value.get("control_objects")
    _require(isinstance(controls, Mapping), "control_objects missing")
    _require(controls.get("map_url") == EXPECTED_MAP_URL, "map URL drift")
    _require(controls.get("md5_url") == EXPECTED_MD5_URL, "MD5 URL drift")
    _require(controls.get("allowed_host") == "data.hplt-project.org", "allowed host drift")
    _require(controls.get("max_each_bytes") == 2_097_152, "control size cap drift")
    _require(controls.get("map_snapshot_sha256") is None, "pre-snapshot map hash must be null")
    _require(controls.get("md5_snapshot_sha256") is None, "pre-snapshot MD5 hash must be null")

    policy = value.get("shard_policy")
    _require(isinstance(policy, Mapping), "shard_policy missing")
    expected_policy = {
        "allowed_url_scheme": "https",
        "allowed_host": "data.hplt-project.org",
        "required_path_prefix": "/three/sorted/ukr_Cyrl/",
        "required_suffix": ".jsonl.zst",
        "filename_pattern": r"^(?:10|[5-9])_[1-9][0-9]*\.jsonl\.zst$",
        "require_unique_urls": True,
        "require_unique_filenames": True,
        "require_md5_for_every_map_entry": True,
        "require_no_orphan_md5_entries": True,
        "project_sha256_required_after_shard_download": True,
    }
    for key, expected in expected_policy.items():
        _require(policy.get(key) == expected, f"shard_policy.{key} drift")

    bounded = value.get("bounded_successor")
    _require(isinstance(bounded, Mapping), "bounded_successor missing")
    _require(
        bounded.get("selection_rule") == "highest_wds_bin_then_lexicographically_first_shard",
        "selection rule drift",
    )
    _require(bounded.get("max_selected_shards") == 1, "selected-shard cap drift")
    _require(bounded.get("bulk_download_forbidden") is True, "bulk-download boundary weakened")
    _require(
        bounded.get("preserve_record_provenance_required") is True,
        "provenance boundary weakened",
    )
    _require(
        bounded.get("source_level_rights_required_before_training_credit") is True,
        "source-rights boundary weakened",
    )

    claim = value.get("claim_boundary")
    _require(isinstance(claim, Mapping), "claim_boundary missing")
    zero_fields = (
        "shards_downloaded",
        "downloaded_shard_bytes",
        "training_authorized_bytes",
        "unique_causal_loss_positions_authorized",
        "optimizer_updates",
    )
    for field in zero_fields:
        _require(claim.get(field) == 0, f"claim boundary requires {field}=0")
    false_fields = (
        "control_snapshot_materialized",
        "immutable_acquisition_identity",
        "source_level_rights_terminal",
        "privacy_complete",
        "global_dedup_complete",
        "evaluation_decontamination_complete",
        "tokenizer_fit_authorized",
        "model_training_executed",
        "final_test_payload_accessed",
        "paid_compute_used",
    )
    for field in false_fields:
        _require(claim.get(field) is False, f"claim boundary requires {field}=false")
    return value


def _decode_lines(payload: bytes, label: str) -> list[str]:
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise HpltControlError(f"{label} is not strict UTF-8") from exc
    if "\x00" in text:
        raise HpltControlError(f"{label} contains NUL")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    _require(lines, f"{label} contains no records")
    return lines


def parse_map(payload: bytes, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    policy = config["shard_policy"]
    filename_re = re.compile(str(policy["filename_pattern"]))
    seen_urls: set[str] = set()
    seen_filenames: set[str] = set()
    rows: list[dict[str, Any]] = []
    for line in _decode_lines(payload, "map"):
        parsed = urllib.parse.urlsplit(line)
        _require(parsed.scheme == policy["allowed_url_scheme"], "map URL scheme rejected")
        _require(parsed.hostname == policy["allowed_host"], "map URL host rejected")
        _require(parsed.username is None and parsed.password is None, "map URL credentials rejected")
        _require(parsed.port in (None, 443), "map URL port rejected")
        _require(not parsed.query and not parsed.fragment, "map URL query/fragment rejected")
        _require(
            parsed.path.startswith(str(policy["required_path_prefix"])),
            "map URL path prefix rejected",
        )
        filename = Path(parsed.path).name
        _require(filename_re.fullmatch(filename) is not None, "map shard filename rejected")
        _require(line not in seen_urls, "duplicate map URL")
        _require(filename not in seen_filenames, "duplicate map filename")
        seen_urls.add(line)
        seen_filenames.add(filename)
        stem = filename.removesuffix(".jsonl.zst")
        bin_text, shard_text = stem.split("_", 1)
        rows.append(
            {
                "url": line,
                "filename": filename,
                "wds_bin": int(bin_text),
                "shard_index": int(shard_text),
            }
        )
    return rows


def _md5_filename(raw: str, filename_re: re.Pattern[str]) -> str:
    value = raw.strip()
    if value.startswith("*"):
        value = value[1:]
    if urllib.parse.urlsplit(value).scheme:
        parsed = urllib.parse.urlsplit(value)
        _require(parsed.scheme == "https", "MD5 URL scheme rejected")
        _require(parsed.hostname == "data.hplt-project.org", "MD5 URL host rejected")
        value = parsed.path
    value = value.replace("\\", "/")
    _require(".." not in value.split("/"), "MD5 path traversal rejected")
    filename = Path(value).name
    _require(filename_re.fullmatch(filename) is not None, "MD5 shard filename rejected")
    return filename


def parse_md5(payload: bytes, config: Mapping[str, Any]) -> dict[str, str]:
    filename_re = re.compile(str(config["shard_policy"]["filename_pattern"]))
    result: dict[str, str] = {}
    for line in _decode_lines(payload, "md5"):
        match = re.fullmatch(r"([0-9a-fA-F]{32})\s+(.+)", line)
        _require(match is not None, "MD5 line format rejected")
        assert match is not None
        digest = match.group(1).lower()
        _require(MD5_RE.fullmatch(digest) is not None, "invalid MD5 digest")
        filename = _md5_filename(match.group(2), filename_re)
        _require(filename not in result, "duplicate MD5 filename")
        result[filename] = digest
    return result


def build_report(
    map_payload: bytes,
    md5_payload: bytes,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    controls = config["control_objects"]
    maximum = int(controls["max_each_bytes"])
    _require(len(map_payload) <= maximum, "map control exceeds size cap")
    _require(len(md5_payload) <= maximum, "MD5 control exceeds size cap")

    map_rows = parse_map(map_payload, config)
    md5_by_filename = parse_md5(md5_payload, config)
    map_filenames = {row["filename"] for row in map_rows}
    md5_filenames = set(md5_by_filename)
    _require(map_filenames == md5_filenames, "map/MD5 shard inventory mismatch")

    inventory = [
        {**row, "upstream_md5": md5_by_filename[row["filename"]]}
        for row in map_rows
    ]
    highest_bin = max(row["wds_bin"] for row in inventory)
    candidates = sorted(
        (row for row in inventory if row["wds_bin"] == highest_bin),
        key=lambda row: row["filename"],
    )
    selected = candidates[0]
    inventory_identity = _sha256(_canonical_json_bytes(sorted(inventory, key=lambda row: row["filename"])))

    core: dict[str, Any] = {
        "schema_version": SCHEMA,
        "worker_id": config["worker_id"],
        "execution_profile": "LOCAL_FREE",
        "parent": dict(config["parent"]),
        "control_snapshot": {
            "map_url": controls["map_url"],
            "map_bytes": len(map_payload),
            "map_sha256": _sha256(map_payload),
            "md5_url": controls["md5_url"],
            "md5_bytes": len(md5_payload),
            "md5_sha256": _sha256(md5_payload),
            "control_snapshot_materialized": True,
        },
        "shard_inventory": {
            "shard_count": len(inventory),
            "inventory_identity_sha256": inventory_identity,
            "entries": inventory,
        },
        "bounded_selection": {
            "selection_rule": config["bounded_successor"]["selection_rule"],
            "max_selected_shards": 1,
            "highest_wds_bin": highest_bin,
            "selected": selected,
            "selection_uses_evaluation_results": False,
            "shard_downloaded": False,
            "project_sha256": None,
        },
        "gates": {
            "control_object_snapshot": "PASS",
            "map_md5_inventory_match": "PASS",
            "bounded_shard_selection": "PASS",
            "selected_shard_sha256": "NOT_RUN",
            "record_provenance": "NOT_RUN",
            "source_level_rights": "NOT_RUN",
            "privacy": "NOT_RUN",
            "global_cross_source_dedup": "NOT_RUN",
            "evaluation_decontamination": "NOT_RUN",
        },
        "claim_boundary": {
            "immutable_control_snapshot": True,
            "immutable_shard_sha256_identity": False,
            "shards_downloaded": 0,
            "downloaded_shard_bytes": 0,
            "training_authorized_bytes": 0,
            "unique_causal_loss_positions_authorized": 0,
            "tokenizer_fit_authorized": False,
            "optimizer_updates": 0,
            "model_training_executed": False,
            "final_test_payload_accessed": False,
            "paid_compute_used": False,
        },
        "safe_result": "CONTROL_OBJECTS_SNAPSHOTTED_BOUNDED_SHARD_SELECTION_READY_ZERO_CREDIT",
    }
    return {**core, "report_sha256": _sha256(_canonical_json_bytes(core))}


def _download_control(url: str, config: Mapping[str, Any]) -> bytes:
    controls = config["control_objects"]
    allowed = {controls["map_url"], controls["md5_url"]}
    _require(url in allowed, "control URL not allowlisted")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "text/plain,*/*;q=0.1"},
    )
    maximum = int(controls["max_each_bytes"])
    with urllib.request.urlopen(request, timeout=60) as response:
        _require(response.geturl() == url, "control object redirected")
        length = response.headers.get("Content-Length")
        if length is not None:
            try:
                declared = int(length)
            except ValueError as exc:
                raise HpltControlError("invalid control Content-Length") from exc
            _require(declared <= maximum, "control Content-Length exceeds cap")
        payload = response.read(maximum + 1)
    _require(len(payload) <= maximum, "control object exceeds size cap")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--map-file", type=Path)
    parser.add_argument("--md5-file", type=Path)
    parser.add_argument("--download-live", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)

    if args.download_live:
        _require(args.map_file is None and args.md5_file is None, "live/local input modes conflict")
        map_payload = _download_control(config["control_objects"]["map_url"], config)
        md5_payload = _download_control(config["control_objects"]["md5_url"], config)
    else:
        _require(args.map_file is not None and args.md5_file is not None, "both local controls required")
        try:
            map_payload = args.map_file.read_bytes()
            md5_payload = args.md5_file.read_bytes()
        except OSError as exc:
            raise HpltControlError("cannot read local control objects") from exc

    report = build_report(map_payload, md5_payload, config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "PASS_ZERO_CREDIT_CONTROL_SNAPSHOT",
                "map_sha256": report["control_snapshot"]["map_sha256"],
                "md5_sha256": report["control_snapshot"]["md5_sha256"],
                "shard_count": report["shard_inventory"]["shard_count"],
                "selected": report["bounded_selection"]["selected"]["filename"],
                "training_authorized_bytes": 0,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
