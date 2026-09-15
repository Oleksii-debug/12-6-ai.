#!/usr/bin/env python3
"""Execute zero-credit G05/G06 replay over the exact clean DATA526 graph."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from twelve_six.data.privacy_execution_authority import (
    build_privacy_execution_authority,
    input_rows_sha256,
    input_rows_sha256_from_text_free_inventory,
    verify_privacy_execution_authority,
    verify_privacy_execution_root,
)
from twelve_six.data.quality_execution_authority import (
    build_quality_execution_authority,
    verify_quality_execution_authority,
)

SCHEMA = "12-6.d03-clean-g05-g06-replay-execution.v1"
PAIR_SCHEMA = "12-6.d03-clean-g05-g06-replay-pair.v1"
EXECUTION_PROFILE = "LOCAL_FREE"

INTEGRATED_SOURCE_MERGE_SHA = "5e8ced70926a2e7f14aad984cf552f2dc646aa59"
SOURCE_RELEASE_HEAD_SHA = "07754c5a1d61669061e608323ea35ddc093bb946"
PHYSICAL_EXECUTION_HEAD_SHA = "3d7dd363f6b1701694c00c76c6353077f80d1492"
PHYSICAL_RUN_ID = 34911721640
PHYSICAL_JOB_ID = 104200523132
PHYSICAL_ARTIFACT_ID = 10374891614
PHYSICAL_ARTIFACT_ZIP_SHA256 = (
    "663eec03d04252b6de574bf97ea0975703e0ba25779ca27340cb3acea941943a"
)
PHYSICAL_SOURCE_REPORT_SHA256 = (
    "db72eb1d4f86cd025741efd0c612c1f2e124ce24dfa133548c375d781331c93c"
)
PHYSICAL_SURVIVOR_AUTHORITY_SHA256 = (
    "e1c94f5eed4afa78a63d577fe33a67e28305c1e084c27cd1061061ad113f8ce5"
)
PHYSICAL_DATA526_EVIDENCE_IDENTITY_SHA256 = (
    "45ac421c9e4a5930330c1516af79e1b67d0535fc90b768ecfe9df54dd0a86c5c"
)

CLEAN_RECORD_COUNT = 274
CLEAN_SOURCE_OBJECT_COUNT = 261
CLEAN_TOTAL_PAYLOAD_BYTES = 6_093_662
CLEAN_RECORD_PAYLOAD_JSONL_SHA256 = (
    "dc22d829921890ea8c5b51cbedaae099688c37c3cb624a597973305c7fa5b2c3"
)
CLEAN_RECORD_INVENTORY_SHA256 = (
    "7d6782e91243505c01b0f2f6d6f85b5bbe3a6abf628d73721b1e0c1c77e4f352"
)
CLEAN_PAYLOAD_INVENTORY_SHA256 = (
    "59f9c5a7b5db9e45fcde2e3bab10a4adc97cb6832f906fc241edc3e35248b576"
)

QUALITY_MODULE_PATH = Path("src/twelve_six/data/quality_execution_authority.py")
QUALITY_MODULE_BLOB_SHA1 = "c8963d2d697f3cd124a5b511d2883a93f8caf34d"
PRIVACY_MODULE_PATH = Path("src/twelve_six/data/privacy_execution_authority.py")
PRIVACY_MODULE_BLOB_SHA1 = "e798adf593b4bd4475acb47f017b4acc36a439a9"
CLEAN_MATERIALIZER_PATH = Path(
    "tools/materialize_d03_nomis_free_data526_successor_v1.py"
)
CLEAN_MATERIALIZER_BLOB_SHA1 = "d183bb83df73ca4ca528d4360048c9ccc031cf33"

NOMIS_RECORD_ID = "ua.verba.nomis1864.bounded24"
NOMIS_FAMILY = "ua.verba.public-domain.nomis1864"
NOMIS_PAYLOAD_SHA256 = (
    "1eb91dbd631898c6a2efe274b700a5be0deaca243c0a9d5d30994ddadcf43598"
)

RECORD_KEYS = {
    "record_id", "source_id", "family", "modality", "normalized_payload"
}
INVENTORY_KEYS = {
    "schema_version", "record_count", "total_payload_bytes",
    "record_inventory_digest_sha256", "payload_inventory_digest_sha256",
    "records",
}
INVENTORY_ROW_KEYS = {
    "record_id", "source_id", "family", "modality", "payload_sha256",
    "payload_bytes",
}
MODES = {"uk", "en", "code"}

TRUTH_BOUNDARY = {
    "clean_g05_g06_replay_executed": True,
    "reserved_evaluation_decontamination_executed": False,
    "current_retained_corpus_launch_authoritative": False,
    "tokenizer_fit_authorized": False,
    "authorized_optimized_target_exposure": 0,
    "optimizer_updates_executed_on_real_targets": 0,
    "training_executed": False,
    "learned_weights_created": False,
    "final_test_outcomes_read": False,
    "paid_compute_used": False,
    "foreign_pretrained_weights": False,
    "raw_payloads_retained_in_output": False,
}


class CleanG05G06ReplayError(ValueError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise CleanG05G06ReplayError(message)


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def cjson(value: Any) -> bytes:
    return canonical(value) + b"\n"


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def git_blob_sha1(raw: bytes) -> str:
    prefix = f"blob {len(raw)}\0".encode("ascii")
    return hashlib.sha1(prefix + raw).hexdigest()  # noqa: S324


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(type(value) is dict, f"JSON root must be object: {path}")
    return value


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def checkout_head(root: Path) -> str:
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    require(len(out) == 40, "checkout HEAD malformed")
    require(all(ch in "0123456789abcdef" for ch in out), "checkout HEAD malformed")
    return out


def validate_runtime_bindings(root: Path) -> dict[str, str]:
    expected = {
        QUALITY_MODULE_PATH: QUALITY_MODULE_BLOB_SHA1,
        PRIVACY_MODULE_PATH: PRIVACY_MODULE_BLOB_SHA1,
        CLEAN_MATERIALIZER_PATH: CLEAN_MATERIALIZER_BLOB_SHA1,
    }
    result = {}
    for path, wanted in expected.items():
        observed = git_blob_sha1((root / path).read_bytes())
        require(observed == wanted, f"bound Git blob drift: {path}")
        result[str(path)] = observed
    return result


def validate_clean_evidence(value: Mapping[str, Any]) -> str:
    require(
        value.get("schema_version")
        == "12-6.d03-nomis-free-data526-successor-evidence.v1",
        "clean DATA526 evidence schema drift",
    )
    require(value.get("execution_profile") == EXECUTION_PROFILE, "profile drift")
    require(
        value.get("source_report_sha256") == PHYSICAL_SOURCE_REPORT_SHA256,
        "clean source report identity drift",
    )
    require(
        value.get("survivor_authority_sha256")
        == PHYSICAL_SURVIVOR_AUTHORITY_SHA256,
        "clean survivor identity drift",
    )
    require(
        value.get("raw_text_emitted_to_durable_evidence") is False,
        "raw durable evidence widened",
    )
    clean = value.get("clean_data526")
    require(isinstance(clean, Mapping), "clean_data526 missing")
    exact = {
        "record_count": CLEAN_RECORD_COUNT,
        "source_object_count": CLEAN_SOURCE_OBJECT_COUNT,
        "total_payload_bytes": CLEAN_TOTAL_PAYLOAD_BYTES,
        "record_payload_jsonl_sha256": CLEAN_RECORD_PAYLOAD_JSONL_SHA256,
        "record_inventory_digest_sha256": CLEAN_RECORD_INVENTORY_SHA256,
        "payload_inventory_digest_sha256": CLEAN_PAYLOAD_INVENTORY_SHA256,
    }
    for key, wanted in exact.items():
        require(clean.get(key) == wanted, f"clean_data526.{key} drift")
    boundary = value.get("truth_boundary")
    require(isinstance(boundary, Mapping), "clean truth boundary missing")
    for key in (
        "tokenizer_fit_authorized", "model_training_executed",
        "learned_weights_created", "final_test_payload_read",
        "paid_compute_used",
    ):
        require(boundary.get(key) is False, f"clean truth boundary widened: {key}")
    require(boundary.get("authorized_training_exposure") == 0, "training exposure widened")
    require(boundary.get("optimizer_updates") == 0, "optimizer updates widened")
    identity = value.get("evidence_identity_sha256")
    require(isinstance(identity, str) and len(identity) == 64, "evidence identity malformed")
    body = dict(value)
    body.pop("evidence_identity_sha256", None)
    require(sha256(canonical(body)) == identity, "clean evidence self-hash mismatch")
    if value.get("execution_head_sha") == PHYSICAL_EXECUTION_HEAD_SHA:
        require(
            identity == PHYSICAL_DATA526_EVIDENCE_IDENTITY_SHA256,
            "integrated #2107 evidence identity drift",
        )
    return identity


def normalize_inventory(value: Mapping[str, Any]) -> list[dict[str, Any]]:
    require(set(value) == INVENTORY_KEYS, "clean inventory schema drift")
    require(
        value.get("schema_version") == "12-6.data526-record-inventory.v1",
        "clean inventory version drift",
    )
    rows = value.get("records")
    require(isinstance(rows, list) and rows, "clean inventory rows missing")
    normalized = []
    seen = set()
    source_ids = set()
    for index, row in enumerate(rows):
        require(
            isinstance(row, Mapping) and set(row) == INVENTORY_ROW_KEYS,
            f"inventory row {index} schema drift",
        )
        record_id = row["record_id"]
        require(
            isinstance(record_id, str) and record_id and record_id not in seen,
            f"inventory row {index} id drift",
        )
        require(row["modality"] in MODES, f"inventory row {index} mode drift")
        require(type(row["payload_bytes"]) is int, f"inventory row {index} bytes drift")
        require(record_id != NOMIS_RECORD_ID, "Nomis record survived clean inventory")
        require(row["family"] != NOMIS_FAMILY, "Nomis family survived clean inventory")
        require(
            row["payload_sha256"] != NOMIS_PAYLOAD_SHA256,
            "Nomis payload survived clean inventory",
        )
        normalized.append(dict(row))
        seen.add(record_id)
        source_ids.add(row["source_id"])
    normalized.sort(key=lambda row: row["record_id"])
    record_root = sha256(canonical(normalized))
    payload_root = sha256(canonical([
        {
            "record_id": row["record_id"],
            "payload_sha256": row["payload_sha256"],
            "payload_bytes": row["payload_bytes"],
        }
        for row in normalized
    ]))
    require(len(normalized) == CLEAN_RECORD_COUNT, "clean record count drift")
    require(len(source_ids) == CLEAN_SOURCE_OBJECT_COUNT, "clean source count drift")
    require(
        sum(row["payload_bytes"] for row in normalized)
        == CLEAN_TOTAL_PAYLOAD_BYTES,
        "clean payload byte drift",
    )
    require(value.get("record_count") == CLEAN_RECORD_COUNT, "declared count drift")
    require(
        value.get("total_payload_bytes") == CLEAN_TOTAL_PAYLOAD_BYTES,
        "declared bytes drift",
    )
    require(
        value.get("record_inventory_digest_sha256")
        == record_root == CLEAN_RECORD_INVENTORY_SHA256,
        "clean record inventory root drift",
    )
    require(
        value.get("payload_inventory_digest_sha256")
        == payload_root == CLEAN_PAYLOAD_INVENTORY_SHA256,
        "clean payload inventory root drift",
    )
    return normalized


def load_records(path: Path, inventory_rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    raw = path.read_bytes()
    require(sha256(raw) == CLEAN_RECORD_PAYLOAD_JSONL_SHA256, "clean JSONL identity drift")
    inventory = {row["record_id"]: row for row in inventory_rows}
    records = []
    seen = set()
    sources = set()
    for number, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line.decode("utf-8"))
        require(type(row) is dict and set(row) == RECORD_KEYS, f"record {number} schema drift")
        record_id = row["record_id"]
        require(record_id not in seen, f"duplicate record: {record_id}")
        require(row["modality"] in MODES, f"record {number} mode drift")
        require(record_id != NOMIS_RECORD_ID, "Nomis record survived clean payload")
        require(row["family"] != NOMIS_FAMILY, "Nomis family survived clean payload")
        payload = row["normalized_payload"].encode("utf-8", errors="strict")
        inv = inventory.get(record_id)
        require(inv is not None, f"record not in inventory: {record_id}")
        require(inv["source_id"] == row["source_id"], "source binding drift")
        require(inv["family"] == row["family"], "family binding drift")
        require(inv["modality"] == row["modality"], "modality binding drift")
        require(inv["payload_sha256"] == sha256(payload), "payload hash drift")
        require(inv["payload_bytes"] == len(payload), "payload bytes drift")
        records.append({"id": record_id, "text": row["normalized_payload"], "mode": row["modality"]})
        seen.add(record_id)
        sources.add(row["source_id"])
    require(len(records) == CLEAN_RECORD_COUNT, "raw record count drift")
    require(len(sources) == CLEAN_SOURCE_OBJECT_COUNT, "raw source count drift")
    require(seen == set(inventory), "raw/inventory record set drift")
    return records


def execute_replay(
    *, run_id: str, records_jsonl: Path, inventory_json: Path,
    evidence_json: Path, output_g05: Path, output_g06: Path,
    output_replay: Path,
) -> dict[str, Any]:
    require(bool(run_id.strip()), "run id must be non-empty")
    root = repo_root()
    head = checkout_head(root)
    bindings = validate_runtime_bindings(root)
    fresh_evidence = validate_clean_evidence(read_json(evidence_json))
    inventory = normalize_inventory(read_json(inventory_json))
    records = load_records(records_jsonl, inventory)
    input_root = input_rows_sha256(records)
    require(
        input_root == input_rows_sha256_from_text_free_inventory(inventory),
        "raw/inventory G05/G06 input root mismatch",
    )
    g05 = build_quality_execution_authority(
        records,
        input_manifest_sha256=PHYSICAL_DATA526_EVIDENCE_IDENTITY_SHA256,
        expected_input_rows_sha256=input_root,
    )
    g05_id = g05["execution_identity_sha256"]
    verify_quality_execution_authority(
        g05, records,
        expected_input_manifest_sha256=PHYSICAL_DATA526_EVIDENCE_IDENTITY_SHA256,
        expected_input_rows_sha256=input_root,
        expected_execution_identity_sha256=g05_id,
    )
    g06 = build_privacy_execution_authority(
        records, expected_input_rows_sha256=input_root
    )
    g06_id = g06["execution_identity_sha256"]
    verify_privacy_execution_root(
        g06, expected_input_rows_sha256=input_root,
        expected_execution_identity_sha256=g06_id,
    )
    verify_privacy_execution_authority(
        g06, records, expected_input_rows_sha256=input_root,
        expected_execution_identity_sha256=g06_id,
    )
    g05_raw = cjson(g05)
    g06_raw = cjson(g06)
    core = {
        "schema_version": SCHEMA,
        "execution_profile": EXECUTION_PROFILE,
        "run_id": run_id,
        "execution_head_sha": head,
        "runner_git_blob_sha1": git_blob_sha1(Path(__file__).read_bytes()),
        "dependencies": {
            "integrated_source_merge_sha": INTEGRATED_SOURCE_MERGE_SHA,
            "source_release_head_sha": SOURCE_RELEASE_HEAD_SHA,
            "physical_execution_head_sha": PHYSICAL_EXECUTION_HEAD_SHA,
            "physical_run_id": PHYSICAL_RUN_ID,
            "physical_job_id": PHYSICAL_JOB_ID,
            "physical_artifact_id": PHYSICAL_ARTIFACT_ID,
            "physical_artifact_zip_sha256": PHYSICAL_ARTIFACT_ZIP_SHA256,
            "physical_source_report_sha256": PHYSICAL_SOURCE_REPORT_SHA256,
            "physical_survivor_authority_sha256": PHYSICAL_SURVIVOR_AUTHORITY_SHA256,
            "physical_data526_evidence_identity_sha256": PHYSICAL_DATA526_EVIDENCE_IDENTITY_SHA256,
            "quality_execution_module_git_blob_sha1": bindings[str(QUALITY_MODULE_PATH)],
            "privacy_execution_module_git_blob_sha1": bindings[str(PRIVACY_MODULE_PATH)],
            "clean_data526_materializer_git_blob_sha1": bindings[str(CLEAN_MATERIALIZER_PATH)],
        },
        "fresh_reconstruction_evidence_identity_sha256": fresh_evidence,
        "clean_record_payload_jsonl_sha256": CLEAN_RECORD_PAYLOAD_JSONL_SHA256,
        "clean_record_inventory_digest_sha256": CLEAN_RECORD_INVENTORY_SHA256,
        "clean_payload_inventory_digest_sha256": CLEAN_PAYLOAD_INVENTORY_SHA256,
        "covered_record_count": CLEAN_RECORD_COUNT,
        "covered_payload_bytes": CLEAN_TOTAL_PAYLOAD_BYTES,
        "covered_source_object_count": CLEAN_SOURCE_OBJECT_COUNT,
        "g05_g06_input_rows_sha256": input_root,
        "g05_execution_identity_sha256": g05_id,
        "g05_authority_sha256": sha256(g05_raw),
        "g06_execution_identity_sha256": g06_id,
        "g06_authority_sha256": sha256(g06_raw),
        "truth_boundary": dict(TRUTH_BOUNDARY),
    }
    replay = {**core, "replay_execution_authority_sha256": sha256(cjson(core))}
    for path, raw in (
        (output_g05, g05_raw), (output_g06, g06_raw),
        (output_replay, cjson(replay)),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    return replay


def validate_receipt(value: Mapping[str, Any]) -> None:
    require(value.get("schema_version") == SCHEMA, "replay schema drift")
    require(value.get("execution_profile") == EXECUTION_PROFILE, "replay profile drift")
    require(value.get("covered_record_count") == CLEAN_RECORD_COUNT, "replay count drift")
    require(value.get("covered_payload_bytes") == CLEAN_TOTAL_PAYLOAD_BYTES, "replay bytes drift")
    require(value.get("covered_source_object_count") == CLEAN_SOURCE_OBJECT_COUNT, "replay sources drift")
    require(value.get("truth_boundary") == TRUTH_BOUNDARY, "replay truth boundary drift")
    require(
        value.get("clean_record_payload_jsonl_sha256")
        == CLEAN_RECORD_PAYLOAD_JSONL_SHA256,
        "replay JSONL root drift",
    )
    require(
        value.get("clean_record_inventory_digest_sha256")
        == CLEAN_RECORD_INVENTORY_SHA256,
        "replay record root drift",
    )
    require(
        value.get("clean_payload_inventory_digest_sha256")
        == CLEAN_PAYLOAD_INVENTORY_SHA256,
        "replay payload root drift",
    )
    body = dict(value)
    claimed = body.pop("replay_execution_authority_sha256", None)
    require(claimed == sha256(cjson(body)), "replay self-hash mismatch")


def verify_pair(
    *, replay_a: Path, replay_b: Path, g05_a: Path, g05_b: Path,
    g06_a: Path, g06_b: Path, output: Path,
) -> dict[str, Any]:
    a = read_json(replay_a)
    b = read_json(replay_b)
    validate_receipt(a)
    validate_receipt(b)
    require(a.get("run_id") != b.get("run_id"), "replay run ids must be distinct")
    require(a.get("execution_head_sha") == b.get("execution_head_sha"), "replay head drift")
    g05_a_raw, g05_b_raw = g05_a.read_bytes(), g05_b.read_bytes()
    g06_a_raw, g06_b_raw = g06_a.read_bytes(), g06_b.read_bytes()
    require(g05_a_raw == g05_b_raw, "independent G05 authorities differ")
    require(g06_a_raw == g06_b_raw, "independent G06 authorities differ")
    require(
        sha256(g05_a_raw) == a.get("g05_authority_sha256")
        == b.get("g05_authority_sha256"),
        "G05 receipt hash mismatch",
    )
    require(
        sha256(g06_a_raw) == a.get("g06_authority_sha256")
        == b.get("g06_authority_sha256"),
        "G06 receipt hash mismatch",
    )
    fields = (
        "g05_g06_input_rows_sha256", "g05_execution_identity_sha256",
        "g05_authority_sha256", "g06_execution_identity_sha256",
        "g06_authority_sha256",
    )
    for field in fields:
        require(a.get(field) == b.get(field), f"replay science drift: {field}")
    core = {
        "schema_version": PAIR_SCHEMA,
        "execution_profile": EXECUTION_PROFILE,
        "execution_head_sha": a["execution_head_sha"],
        "run_ids": [a["run_id"], b["run_id"]],
        "replay_execution_authorities": [
            a["replay_execution_authority_sha256"],
            b["replay_execution_authority_sha256"],
        ],
        "covered_record_count": CLEAN_RECORD_COUNT,
        "covered_payload_bytes": CLEAN_TOTAL_PAYLOAD_BYTES,
        "covered_source_object_count": CLEAN_SOURCE_OBJECT_COUNT,
        "g05_g06_input_rows_sha256": a["g05_g06_input_rows_sha256"],
        "g05_execution_identity_sha256": a["g05_execution_identity_sha256"],
        "g05_authority_sha256": a["g05_authority_sha256"],
        "g06_execution_identity_sha256": a["g06_execution_identity_sha256"],
        "g06_authority_sha256": a["g06_authority_sha256"],
        "two_independent_replays_agree": True,
        "truth_boundary": dict(TRUTH_BOUNDARY),
    }
    result = {**core, "pair_execution_authority_sha256": sha256(cjson(core))}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(cjson(result))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check-bindings")
    check.add_argument("--repo-root", type=Path, default=Path("."))
    run = sub.add_parser("run")
    run.add_argument("--run-id", required=True)
    run.add_argument("--records-jsonl", type=Path, required=True)
    run.add_argument("--inventory-json", type=Path, required=True)
    run.add_argument("--data526-evidence-json", type=Path, required=True)
    run.add_argument("--output-g05", type=Path, required=True)
    run.add_argument("--output-g06", type=Path, required=True)
    run.add_argument("--output-replay", type=Path, required=True)
    pair = sub.add_parser("verify-pair")
    for name in ("replay-a", "replay-b", "g05-a", "g05-b", "g06-a", "g06-b"):
        pair.add_argument(f"--{name}", type=Path, required=True)
    pair.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "check-bindings":
        print(json.dumps(validate_runtime_bindings(args.repo_root.resolve()), sort_keys=True))
        return 0
    if args.command == "run":
        result = execute_replay(
            run_id=args.run_id, records_jsonl=args.records_jsonl,
            inventory_json=args.inventory_json,
            evidence_json=args.data526_evidence_json,
            output_g05=args.output_g05, output_g06=args.output_g06,
            output_replay=args.output_replay,
        )
    else:
        result = verify_pair(
            replay_a=args.replay_a, replay_b=args.replay_b,
            g05_a=args.g05_a, g05_b=args.g05_b,
            g06_a=args.g06_a, g06_b=args.g06_b, output=args.output,
        )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
