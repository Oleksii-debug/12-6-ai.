#!/usr/bin/env python3
"""Execute exact current-Rada global-dedup survivors through DATA-232/G05/G06.

Execution-only carrier. It authenticates terminal #2818 global-dedup evidence and
terminal #2820 bounded training-purpose rights evidence, freshly rematerializes the
exact current-Rada candidate, retains only exact #2818 survivors in ephemeral memory,
and delegates DATA-232, G05 quality, and G06 privacy to the incumbent current-clean
implementation. Durable output is text-free and grants zero corpus/tokenizer/training/
scale authority.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from twelve_six.data import current_clean_execution_v1 as clean
from twelve_six.data import current_reserved_decontamination_v1 as reserved
from twelve_six.data import rada_current_snapshot_dedup_adapter_v1 as rada

SCHEMA = "12-6.d03-rada-current-decontam-g05-g06-execution.v1"

PARENT_EXECUTION_HEAD = "a4663e87b010b190343caf1d42784f5dc7984601"
PARENT_WORKFLOW_RUN_ID = 37410396900
PARENT_ARTIFACT_ID = 11389910019
PARENT_ARTIFACT_ZIP_SHA256 = (
    "63f9e1bf5713989a155429c8862196cacaff3207a7e0fe97586f7dd2c98881f9"
)
PARENT_SURVIVORS_FILE_SHA256 = (
    "ebdb68e689625e2f46e8a44a02b9d38ffafd9e79a04dbdc49bf6f12b5dcfb6c9"
)
PARENT_EVIDENCE_FILE_SHA256 = (
    "43a9621ab11fd82232e8251ab26aae4126cf0b4854878af1e97184d587740caa"
)
PARENT_TWO_CLEAN_FILE_SHA256 = (
    "748adad71a730f7daf18fa52c9e78e3c249ae683e6f3d9cb865b1bbfc9365aa9"
)
PARENT_MATCHER_REPORT_SHA256 = (
    "64e687ae431804862003d5839b9a90c715838e794daad907200f0abfc73b4333"
)
PARENT_SURVIVOR_AUTHORITY_SHA256 = (
    "f103a3f18216519bd9228e586bd673d49f73cace03b3b0278f2dd0a33383bffb"
)
PARENT_EVIDENCE_IDENTITY_SHA256 = (
    "a3f7e396cb13a6b107aaa0eb330edcbdc61bead6fa12eb68542ff5c3a3ed9301"
)
PARENT_TWO_CLEAN_AUTHORITY_SHA256 = (
    "f24f4b2d23bee6cb1273a4680297d942aa59030dab50d5c5de7a4d659ff99a5e"
)
EXPECTED_RADA_OBJECTS = 98_601
EXPECTED_RADA_BYTES = 186_855_914

RIGHTS_EXECUTION_HEAD = "3cec9fa6078ae86d3af6f7efb58f2fc36e111dd5"
RIGHTS_WORKFLOW_RUN_ID = 37398554709
RIGHTS_ARTIFACT_ID = 11383988739
RIGHTS_ARTIFACT_ZIP_SHA256 = (
    "683fcf5966dcc7d338cffa8e0f6f5c227caab351bf11ed5010af6fd50f8d7b76"
)
RIGHTS_A_FILE_SHA256 = (
    "90a9fe5c455584ea1f493b500f182eb101e06cee2f9f7470b4676f75a5f9d955"
)
RIGHTS_B_FILE_SHA256 = (
    "d00b1b49ccb0ab6c2e08d6d6da729d98bdfab4d4e48452cf4ae3ad22b66d347d"
)
RIGHTS_A_EVIDENCE_ID = (
    "7694778f36b3c8ea018802f87a162694ee2663a24aa1056b167d580334f70c41"
)
RIGHTS_B_EVIDENCE_ID = (
    "48445c8effead1034447853567e0d8930a240b0cd93c67de2a423cfd7152ec55"
)
RIGHTS_AUTHORITY_ID = (
    "c95db543b3b04868d6dcd531db130197c160883971c16e237bdf5828234c71d6"
)
RIGHTS_SEMANTIC_ID = (
    "589bffcc268b2359b79c0146f5c6f4ba4d33fdf3d494db972846c4ffb61aa467"
)
RIGHTS_DECISION = "ALLOWED_WITH_SOURCE_ATTRIBUTION_AFTER_REMAINING_DATA_GATES"
SOURCE_FAMILY = "ua.rada.open-data.laws-texts"
SOURCE_ARCHIVE_SHA256 = (
    "9f937b3283dd2d9508a0a92442fc007e7e6d373a17d3d50fa428e324cb2fa290"
)
SOURCE_ARCHIVE_BYTES = 46_774_786

CURRENT_CLEAN_BLOB = "b82ed11626a267dfffa16acd79e45c0cf3e6750b"
CURRENT_RESERVED_BLOB = "e5c555e3cd27844e98d4ae91af0b746e427f36c9"
RADA_ADAPTER_BLOB = "b07bf36c4efd84b524b5f1330d8650179dd0e5a1"
QUALITY_BLOB = "4659a9d4aba49908f372250904a54361c8d8cf46"
PRIVACY_BLOB = "9215287e81c0a82f05ec8405dc4f34c60313c193"
EVAL233_BLOB = "71dd98204c588bd0bb67b7a7b3ddc5ae8aa87c00"
EVAL303_BLOB = "659cb17fdba903d9c50c1dfc5becf046f523e5ee"
EVAL647_BLOB = "5516577a0720150a7ec12c1bf8898972968e6970"
EVAL647_CONFIG_BLOB = "89af932a1b0e2dae30b08f766505c8dba9d3e39f"
EVAL647_EVIDENCE_BLOB = "efa5b77275919fa9417e35efb30c8c9a527bd960"
RADA_AUTHORITY_PATH = (
    ROOT / "evidence/d03_rada_bulk/current_snapshot_github_replay_authority_v1.json"
)
RADA_AUTHORITY_BLOB = "9953940072fdd973cd6c83eb5899314d3484da30"

_SHA40 = re.compile(r"^[0-9a-f]{40}$")
_SHA64 = re.compile(r"^[0-9a-f]{64}$")


class RadaCleanExecutionError(RuntimeError):
    """Fail-closed current-Rada post-dedup/current-clean mismatch."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RadaCleanExecutionError(message)


def canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )
    if check and result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or str(result.returncode)
        raise RadaCleanExecutionError(
            "git failed: " + " ".join(args) + ": " + detail
        )
    return result


def strict_json(path: Path, label: str, expected_sha256: str) -> dict[str, Any]:
    raw = path.read_bytes()
    require(sha256(raw) == expected_sha256, f"{label}: raw SHA-256 drift")
    seen_duplicate = False

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        nonlocal seen_duplicate
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                seen_duplicate = True
            value[key] = item
        return value

    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=lambda item: (_ for _ in ()).throw(
                RadaCleanExecutionError(
                    f"{label}: non-finite constant {item}"
                )
            ),
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RadaCleanExecutionError(f"{label}: invalid strict JSON") from exc
    require(not seen_duplicate, f"{label}: duplicate JSON member")
    require(type(value) is dict, f"{label}: root must be exact object")
    return value


def self_hash(
    value: Mapping[str, Any],
    field: str,
    expected: str,
    label: str,
) -> None:
    require(type(value) is dict, f"{label}: root must be exact object")
    body = copy.deepcopy(dict(value))
    claimed = body.pop(field, None)
    require(
        type(claimed) is str and _SHA64.fullmatch(claimed) is not None,
        f"{label}: identity malformed",
    )
    require(claimed == expected, f"{label}: identity drift")
    require(sha256(canonical(body)) == claimed, f"{label}: self-hash mismatch")


def verify_local_authority(expected_execution_head: str) -> dict[str, str]:
    require(
        type(expected_execution_head) is str
        and _SHA40.fullmatch(expected_execution_head) is not None,
        "expected execution head must be exact lowercase 40-hex SHA",
    )
    require(
        git("rev-parse", "HEAD").stdout.strip() == expected_execution_head,
        "HEAD drift",
    )
    ancestor = git(
        "merge-base",
        "--is-ancestor",
        PARENT_EXECUTION_HEAD,
        expected_execution_head,
        check=False,
    )
    require(
        ancestor.returncode == 0,
        "#2818 global-dedup parent is not ancestor",
    )

    expected = {
        "src/twelve_six/data/current_clean_execution_v1.py": CURRENT_CLEAN_BLOB,
        "src/twelve_six/data/current_reserved_decontamination_v1.py": (
            CURRENT_RESERVED_BLOB
        ),
        "src/twelve_six/data/rada_current_snapshot_dedup_adapter_v1.py": (
            RADA_ADAPTER_BLOB
        ),
        "src/twelve_six/data/quality_execution_authority.py": QUALITY_BLOB,
        "src/twelve_six/data/privacy_execution_authority.py": PRIVACY_BLOB,
        "src/twelve_six/data/eval233_final_test_resolver_v1.py": EVAL233_BLOB,
        "src/twelve_six/data/eval303_selection_payload_resolver_v1.py": (
            EVAL303_BLOB
        ),
        "src/twelve_six/data/eval647_future_training_exclusion_v1.py": (
            EVAL647_BLOB
        ),
        "configs/evaluation/eval_code_reserve_v1.json": EVAL647_CONFIG_BLOB,
        "evidence/eval647/code_selection_source_materialization_v1.json": (
            EVAL647_EVIDENCE_BLOB
        ),
        str(RADA_AUTHORITY_PATH.relative_to(ROOT)): RADA_AUTHORITY_BLOB,
    }
    for path, blob in expected.items():
        require(
            git("rev-parse", f"HEAD:{path}").stdout.strip() == blob,
            f"Git blob drift: {path}",
        )
        require(
            git("hash-object", str(ROOT / path)).stdout.strip() == blob,
            f"worktree blob drift: {path}",
        )

    carrier = "tools/run_d03_rada_current_decontam_qp_v1.py"
    carrier_blob = git("rev-parse", f"HEAD:{carrier}").stdout.strip()
    require(
        git("hash-object", str(ROOT / carrier)).stdout.strip() == carrier_blob,
        "executing carrier worktree drift",
    )
    return {
        **expected,
        **{
            f"current_clean:{key}": value
            for key, value in clean.verify_dependency_blobs().items()
        },
        carrier: carrier_blob,
    }


def verify_parent_artifact(
    survivors_path: Path,
    evidence_path: Path,
    two_clean_path: Path,
) -> dict[str, dict[str, Any]]:
    survivors = strict_json(
        survivors_path,
        "Rada parent survivor authority",
        PARENT_SURVIVORS_FILE_SHA256,
    )
    evidence = strict_json(
        evidence_path,
        "Rada parent execution evidence",
        PARENT_EVIDENCE_FILE_SHA256,
    )
    two_clean = strict_json(
        two_clean_path,
        "Rada parent two-clean authority",
        PARENT_TWO_CLEAN_FILE_SHA256,
    )
    self_hash(
        survivors,
        "survivor_authority_sha256",
        PARENT_SURVIVOR_AUTHORITY_SHA256,
        "Rada parent survivor authority",
    )
    self_hash(
        evidence,
        "evidence_identity_sha256",
        PARENT_EVIDENCE_IDENTITY_SHA256,
        "Rada parent execution evidence",
    )
    self_hash(
        two_clean,
        "two_clean_identity_sha256",
        PARENT_TWO_CLEAN_AUTHORITY_SHA256,
        "Rada parent two-clean authority",
    )

    require(
        survivors.get("schema_version")
        == "12-6.d03-rada-current-global-dedup-survivors.v1",
        "parent survivor schema drift",
    )
    require(
        survivors.get("matcher_report_sha256") == PARENT_MATCHER_REPORT_SHA256,
        "parent matcher report drift",
    )
    require(
        survivors.get("current_rada_survivor_source_object_count")
        == EXPECTED_RADA_OBJECTS,
        "parent Rada survivor count drift",
    )
    require(
        survivors.get("current_rada_survivor_declared_capacity_bytes")
        == EXPECTED_RADA_BYTES,
        "parent Rada survivor byte drift",
    )
    rows = survivors.get("current_rada_survivor_inventory")
    require(
        type(rows) is list and len(rows) == EXPECTED_RADA_OBJECTS,
        "parent survivor inventory cardinality drift",
    )
    by_id: dict[str, dict[str, Any]] = {}
    declared_total = 0
    for row in rows:
        require(type(row) is dict, "parent survivor row must be exact object")
        source_id = row.get("source_id")
        declared = row.get("declared_capacity_bytes")
        raw_sha = row.get("verified_raw_sha256")
        require(
            type(source_id) is str
            and source_id.startswith("rada-laws-qp:")
            and source_id not in by_id,
            "parent survivor source ID invalid or duplicate",
        )
        require(
            type(declared) is int and declared > 0,
            f"parent survivor bytes invalid: {source_id}",
        )
        require(
            type(raw_sha) is str and _SHA64.fullmatch(raw_sha) is not None,
            f"parent survivor raw SHA invalid: {source_id}",
        )
        by_id[source_id] = dict(row)
        declared_total += declared
    require(
        declared_total == EXPECTED_RADA_BYTES,
        "parent survivor inventory byte sum drift",
    )

    discounted = survivors.get("current_rada_discounted_source_ids")
    require(
        type(discounted) is list
        and len(discounted) == 3_132
        and len(set(discounted)) == len(discounted),
        "parent discounted Rada set drift",
    )
    require(
        not set(by_id).intersection(discounted),
        "parent survivor/discounted sets overlap",
    )
    require(
        len(by_id) + len(discounted) == 101_733,
        "parent Rada partition arithmetic drift",
    )

    require(
        evidence.get("schema_version")
        == "12-6.d03-rada-current-global-dedup-execution.v1",
        "parent evidence schema drift",
    )
    require(
        evidence.get("execution_head_sha") == PARENT_EXECUTION_HEAD,
        "parent execution head drift",
    )
    require(
        evidence.get("matcher_report_sha256") == PARENT_MATCHER_REPORT_SHA256,
        "parent evidence matcher root drift",
    )
    require(
        evidence.get("survivor_authority_sha256")
        == PARENT_SURVIVOR_AUTHORITY_SHA256,
        "parent evidence/survivor root drift",
    )
    require(
        evidence.get("current_rada_survivor_source_object_count")
        == EXPECTED_RADA_OBJECTS
        and evidence.get("current_rada_survivor_declared_capacity_bytes")
        == EXPECTED_RADA_BYTES,
        "parent evidence Rada arithmetic drift",
    )

    require(
        two_clean.get("schema_version")
        == "12-6.d03-rada-current-global-dedup-two-clean.v1",
        "parent two-clean schema drift",
    )
    require(
        two_clean.get("execution_head_sha") == PARENT_EXECUTION_HEAD
        and two_clean.get("fresh_process_count") == 2
        and two_clean.get("byte_identical_outputs") is True,
        "parent two-clean execution boundary drift",
    )
    require(
        two_clean.get("matcher_report_sha256") == PARENT_MATCHER_REPORT_SHA256
        and two_clean.get("survivor_authority_sha256")
        == PARENT_SURVIVOR_AUTHORITY_SHA256,
        "parent two-clean roots drift",
    )
    require(
        two_clean.get("current_rada_survivor_source_object_count")
        == EXPECTED_RADA_OBJECTS
        and two_clean.get("current_rada_survivor_declared_capacity_bytes")
        == EXPECTED_RADA_BYTES,
        "parent two-clean Rada arithmetic drift",
    )

    for truth in (survivors, evidence, two_clean):
        require(
            truth.get("canonical_capacity_credited") == 0
            and truth.get("training_authorized_bytes") == 0
            and truth.get("tokenizer_fit_authorized") is False,
            "parent scientific truth boundary widened",
        )
    require(
        survivors.get("training_executed") is False
        and survivors.get("learned_weights_created") is False
        and survivors.get("final_test_outcomes_read") is False
        and survivors.get("paid_compute_used") is False,
        "parent survivor truth boundary widened",
    )
    require(
        evidence.get("training_executed") is False
        and evidence.get("learned_weights_created") is False
        and evidence.get("final_test_outcomes_read") is False
        and evidence.get("paid_compute_used") is False,
        "parent execution truth boundary widened",
    )
    return by_id


def verify_rights_artifact(
    rights_a_path: Path,
    rights_b_path: Path,
) -> dict[str, Any]:
    rights_a = strict_json(
        rights_a_path,
        "Rada rights execution A",
        RIGHTS_A_FILE_SHA256,
    )
    rights_b = strict_json(
        rights_b_path,
        "Rada rights execution B",
        RIGHTS_B_FILE_SHA256,
    )
    self_hash(
        rights_a,
        "evidence_identity_sha256",
        RIGHTS_A_EVIDENCE_ID,
        "Rada rights execution A",
    )
    self_hash(
        rights_b,
        "evidence_identity_sha256",
        RIGHTS_B_EVIDENCE_ID,
        "Rada rights execution B",
    )
    for label, evidence in (("A", rights_a), ("B", rights_b)):
        require(
            evidence.get("schema_version")
            == "12-6.d03-rada-current-training-rights-evidence.v1",
            f"rights {label} schema drift",
        )
        require(
            evidence.get("authority_identity_sha256") == RIGHTS_AUTHORITY_ID,
            f"rights {label} authority identity drift",
        )
        snapshot = evidence.get("exact_snapshot")
        require(
            isinstance(snapshot, Mapping)
            and snapshot.get("source_family") == SOURCE_FAMILY
            and snapshot.get("archive_bytes") == SOURCE_ARCHIVE_BYTES
            and snapshot.get("archive_sha256") == SOURCE_ARCHIVE_SHA256,
            f"rights {label} exact snapshot drift",
        )
        portal = evidence.get("official_portal_semantic_evidence")
        require(
            isinstance(portal, Mapping)
            and portal.get("semantic_identity_sha256") == RIGHTS_SEMANTIC_ID,
            f"rights {label} semantic identity drift",
        )
        projection = portal.get("semantic_projection")
        require(
            isinstance(projection, Mapping)
            and sha256(canonical(dict(projection))) == RIGHTS_SEMANTIC_ID,
            f"rights {label} semantic projection self-hash drift",
        )
        decision = evidence.get("project_decision")
        require(
            isinstance(decision, Mapping)
            and decision.get("model_training_purpose") == RIGHTS_DECISION
            and decision.get("purpose_rights_recheck_complete_for_exact_snapshot")
            is True
            and decision.get("attribution_required") is True
            and decision.get("evaluation") == "NOT_SEPARATELY_ADMITTED"
            and decision.get("final_test") == "PROHIBITED"
            and decision.get("canonical_capacity_credited") == 0
            and decision.get("training_authorized_bytes") == 0,
            f"rights {label} bounded project decision drift",
        )
        truth = evidence.get("truth_boundary")
        require(
            isinstance(truth, Mapping)
            and truth.get("canonical_capacity_credited") == 0
            and truth.get("training_authorized_bytes") == 0
            and truth.get("authorized_unique_loss_positions") == 0
            and truth.get("authorized_optimized_target_exposure") == 0
            and truth.get("tokenizer_fit_authorized") is False
            and truth.get("training_executed") is False
            and truth.get("learned_weights_created") is False
            and truth.get("final_test_outcomes_read") is False
            and truth.get("paid_compute_used") is False
            and truth.get("scale_promotion_authorized") is False,
            f"rights {label} truth boundary widened",
        )

    semantic_a = rights_a["official_portal_semantic_evidence"]["semantic_projection"]
    semantic_b = rights_b["official_portal_semantic_evidence"]["semantic_projection"]
    require(semantic_a == semantic_b, "two-clean rights semantics diverged")
    require(
        rights_a["project_decision"] == rights_b["project_decision"],
        "two-clean rights project decision diverged",
    )
    return {
        "execution_head_sha": RIGHTS_EXECUTION_HEAD,
        "workflow_run_id": RIGHTS_WORKFLOW_RUN_ID,
        "artifact_id": RIGHTS_ARTIFACT_ID,
        "artifact_zip_sha256": RIGHTS_ARTIFACT_ZIP_SHA256,
        "authority_identity_sha256": RIGHTS_AUTHORITY_ID,
        "semantic_identity_sha256": RIGHTS_SEMANTIC_ID,
        "evidence_a_identity_sha256": RIGHTS_A_EVIDENCE_ID,
        "evidence_b_identity_sha256": RIGHTS_B_EVIDENCE_ID,
        "model_training_purpose": RIGHTS_DECISION,
        "attribution_required": True,
        "evaluation": "NOT_SEPARATELY_ADMITTED",
        "final_test": "PROHIBITED",
        "remaining_data_gates_required": True,
    }


def acquire_rada_survivors(
    *,
    candidate_jsonl: Path,
    parent_inventory: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, bytes], dict[str, Any]]:
    projection = rada.validate_and_project_current_rada_candidate(
        candidate_jsonl,
        RADA_AUTHORITY_PATH,
        repository_root=ROOT,
        retain_payloads=True,
    )
    require(
        projection.sources is not None and projection.payloads is not None,
        "Rada projection payloads missing",
    )
    rows_by_id = {str(row["source_id"]): dict(row) for row in projection.sources}
    parent_ids = set(parent_inventory)
    require(
        parent_ids.issubset(rows_by_id),
        "fresh Rada projection lacks parent survivor IDs",
    )
    require(
        parent_ids.issubset(projection.payloads),
        "fresh Rada payload map lacks parent survivor IDs",
    )
    require(
        len(rows_by_id) == 101_733 and len(projection.payloads) == 101_733,
        "fresh full Rada projection cardinality drift",
    )

    rows: list[dict[str, Any]] = []
    payloads: dict[str, bytes] = {}
    for source_id in sorted(parent_ids):
        row = rows_by_id[source_id]
        raw = projection.payloads[source_id]
        parent_row = parent_inventory[source_id]
        require(
            row.get("source_family") == SOURCE_FAMILY
            and row.get("modality") == "uk",
            f"fresh Rada source semantics drift: {source_id}",
        )
        require(
            row.get("declared_capacity_bytes")
            == parent_row.get("declared_capacity_bytes")
            == len(raw),
            f"fresh Rada survivor byte drift: {source_id}",
        )
        require(
            sha256(raw) == parent_row.get("verified_raw_sha256"),
            f"fresh Rada survivor SHA drift: {source_id}",
        )
        stable_origin = row.get("stable_origin_id")
        stable_object = row.get("stable_object_id")
        require(
            type(stable_origin) is str
            and sha256(stable_origin.encode("utf-8"))
            == parent_row.get("stable_origin_id_sha256"),
            f"fresh Rada stable origin drift: {source_id}",
        )
        require(
            type(stable_object) is str
            and sha256(stable_object.encode("utf-8"))
            == parent_row.get("stable_object_id_sha256"),
            f"fresh Rada stable object drift: {source_id}",
        )
        rows.append(row)
        payloads[source_id] = raw

    require(
        len(rows) == EXPECTED_RADA_OBJECTS,
        "fresh Rada survivor count drift",
    )
    require(
        sum(len(raw) for raw in payloads.values()) == EXPECTED_RADA_BYTES,
        "fresh Rada survivor byte sum drift",
    )
    source_authority = {
        "candidate_sha256": projection.receipt["candidate_jsonl_sha256"],
        "projection_receipt_identity_sha256": projection.receipt[
            "receipt_identity_sha256"
        ],
        "post_global_dedup_survivor_ids_sha256": sha256(
            canonical(sorted(parent_ids))
        ),
        "post_global_dedup_survivor_objects": len(rows),
        "post_global_dedup_survivor_bytes": sum(
            len(raw) for raw in payloads.values()
        ),
        "parent_survivor_authority_sha256": PARENT_SURVIVOR_AUTHORITY_SHA256,
    }
    return rows, payloads, source_authority


def build_training_authorities(
    source_rows: list[dict[str, Any]],
    payloads: Mapping[str, bytes],
) -> tuple[
    list[dict[str, str]],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    str,
    str,
]:
    records: list[dict[str, str]] = []
    inventory_rows: list[dict[str, Any]] = []
    for row in source_rows:
        source_id = str(row["source_id"])
        raw = payloads[source_id]
        text = raw.decode("utf-8", errors="strict")
        require(
            text.encode("utf-8") == raw,
            f"{source_id}: UTF-8 round-trip drift",
        )
        family = str(row["source_family"])
        modality = str(row["modality"])
        records.append(
            {
                "record_id": source_id,
                "source_id": source_id,
                "source_family": family,
                "modality": modality,
                "text": text,
            }
        )
        inventory_rows.append(
            {
                "record_id": source_id,
                "source_id": source_id,
                "source_family": family,
                "modality": modality,
                "payload_sha256": sha256(raw),
                "payload_bytes": len(raw),
            }
        )
    records.sort(key=lambda row: row["record_id"])
    inventory_rows.sort(key=lambda row: row["record_id"])
    require(
        len(records) == EXPECTED_RADA_OBJECTS,
        "Rada training record count drift",
    )
    require(
        sum(row["payload_bytes"] for row in inventory_rows)
        == EXPECTED_RADA_BYTES,
        "Rada training byte drift",
    )

    inventory_core = {
        "schema_version": "12-6.d03-rada-current-postdedup-inventory.v1",
        "parent_survivor_authority_sha256": PARENT_SURVIVOR_AUTHORITY_SHA256,
        "record_count": EXPECTED_RADA_OBJECTS,
        "total_payload_bytes": EXPECTED_RADA_BYTES,
        "records": inventory_rows,
    }
    inventory = {
        **inventory_core,
        "inventory_identity_sha256": sha256(canonical(inventory_core)),
    }
    subset_core = {
        "schema_version": "12-6.d03-rada-current-postdedup-subset.v1",
        "parent_execution_head_sha": PARENT_EXECUTION_HEAD,
        "parent_matcher_report_sha256": PARENT_MATCHER_REPORT_SHA256,
        "parent_survivor_authority_sha256": PARENT_SURVIVOR_AUTHORITY_SHA256,
        "parent_two_clean_authority_sha256": (
            PARENT_TWO_CLEAN_AUTHORITY_SHA256
        ),
        "inventory_identity_sha256": inventory["inventory_identity_sha256"],
        "source_object_count": EXPECTED_RADA_OBJECTS,
        "declared_capacity_bytes": EXPECTED_RADA_BYTES,
        "source_ids_sha256": sha256(canonical(sorted(payloads))),
    }
    subset = {
        **subset_core,
        "subset_authority_sha256": sha256(canonical(subset_core)),
    }
    projection = reserved._record_projection(records)
    handoff_core = {
        "schema_version": reserved.TRAINING_HANDOFF_SCHEMA,
        "postdedup_inventory_identity_sha256": inventory[
            "inventory_identity_sha256"
        ],
        "input_survivor_authority_sha256": subset[
            "subset_authority_sha256"
        ],
        "retained_source_count": EXPECTED_RADA_OBJECTS,
        "matcher_input_projection": projection,
        "matcher_input_projection_sha256": sha256(canonical(projection)),
        "raw_text_persisted_in_evidence": False,
        "final_test_payload_accessed": False,
        "final_test_outcomes_accessed": False,
        "authorized_training_exposure": 0,
    }
    handoff = {
        **handoff_core,
        "handoff_identity_sha256": sha256(canonical(handoff_core)),
    }
    training_records_sha = clean._sha256(
        b"".join(clean._cjson(row) for row in records)
    )
    handoff_raw_sha = clean._sha256(clean._cjson(handoff))
    return (
        records,
        inventory,
        subset,
        handoff,
        training_records_sha,
        handoff_raw_sha,
    )


def patch_clean_release(
    *,
    training_records_sha: str,
    handoff_raw_sha: str,
    inventory_identity: str,
    subset_authority: str,
) -> dict[str, Any]:
    names = (
        "PRODUCTION_INPUT_RECORD_COUNT",
        "PRODUCTION_TRAINING_RECORDS_SHA256",
        "PRODUCTION_TRAINING_HANDOFF_SHA256",
        "PRODUCTION_POST_G05_G06_MATERIALIZATION_IDENTITY_SHA256",
        "PRODUCTION_COMPOSITION_PREFLIGHT_IDENTITY_SHA256",
    )
    original = {name: getattr(clean, name) for name in names}
    clean.PRODUCTION_INPUT_RECORD_COUNT = EXPECTED_RADA_OBJECTS
    clean.PRODUCTION_TRAINING_RECORDS_SHA256 = training_records_sha
    clean.PRODUCTION_TRAINING_HANDOFF_SHA256 = handoff_raw_sha
    clean.PRODUCTION_POST_G05_G06_MATERIALIZATION_IDENTITY_SHA256 = (
        inventory_identity
    )
    clean.PRODUCTION_COMPOSITION_PREFLIGHT_IDENTITY_SHA256 = subset_authority
    return original


def restore_clean_release(original: Mapping[str, Any]) -> None:
    for key, value in original.items():
        setattr(clean, key, value)


def load_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number, raw in enumerate(path.read_bytes().splitlines(), start=1):
        require(bool(raw.strip()), f"{label}: blank line {number}")
        try:
            value = json.loads(raw.decode("utf-8", errors="strict"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise RadaCleanExecutionError(
                f"{label}: invalid row {number}"
            ) from exc
        require(
            type(value) is dict,
            f"{label}: row {number} must be exact object",
        )
        rows.append(value)
    require(bool(rows), f"{label}: no rows")
    return rows


def load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RadaCleanExecutionError(f"{label}: invalid JSON") from exc
    require(type(value) is dict, f"{label}: root must be exact object")
    return value


def execute(
    *,
    expected_execution_head: str,
    parent_survivors_json: Path,
    parent_evidence_json: Path,
    parent_two_clean_json: Path,
    rights_a_json: Path,
    rights_b_json: Path,
    candidate_jsonl: Path,
    evaluation_records_jsonl: Path,
    base_reserved_binding_json: Path,
    eval647_manifest_json: Path,
    eval647_materialization_evidence_json: Path,
    expected_base_reserved_binding_identity_sha256: str,
    expected_composed_reserved_binding_identity_sha256: str,
    expected_eval647_materialization_evidence_identity_sha256: str,
    expected_eval647_object_set_identity_sha256: str,
    expected_selection_validation_identity_sha256: str,
    expected_final_test_identity_sha256: str,
    output_evidence: Path,
) -> dict[str, Any]:
    authority_blobs = verify_local_authority(expected_execution_head)
    parent_inventory = verify_parent_artifact(
        parent_survivors_json,
        parent_evidence_json,
        parent_two_clean_json,
    )
    rights_authority = verify_rights_artifact(
        rights_a_json,
        rights_b_json,
    )
    source_rows, payloads, source_authority = acquire_rada_survivors(
        candidate_jsonl=candidate_jsonl,
        parent_inventory=parent_inventory,
    )
    (
        records,
        inventory,
        subset,
        handoff,
        records_sha,
        handoff_raw_sha,
    ) = build_training_authorities(source_rows, payloads)

    evaluation_records = load_jsonl(
        evaluation_records_jsonl,
        "reserved evaluation records",
    )
    base_reserved_binding = load_json(
        base_reserved_binding_json,
        "base reserved binding",
    )
    eval647_manifest = load_json(
        eval647_manifest_json,
        "EVAL-647 manifest",
    )
    eval647_evidence = load_json(
        eval647_materialization_evidence_json,
        "EVAL-647 materialization evidence",
    )

    original = patch_clean_release(
        training_records_sha=records_sha,
        handoff_raw_sha=handoff_raw_sha,
        inventory_identity=inventory["inventory_identity_sha256"],
        subset_authority=subset["subset_authority_sha256"],
    )
    try:
        (
            receipt,
            _data232,
            _decontam,
            _eval647_receipt,
            _quality,
            _privacy,
            final_survivors,
            survivor_inventory,
        ) = clean.execute_current_clean_composition(
            records,
            evaluation_records,
            training_handoff_evidence=handoff,
            base_reserved_binding=base_reserved_binding,
            eval647_manifest=eval647_manifest,
            eval647_materialization_evidence=eval647_evidence,
            expected_base_reserved_binding_identity_sha256=(
                expected_base_reserved_binding_identity_sha256
            ),
            expected_composed_reserved_binding_identity_sha256=(
                expected_composed_reserved_binding_identity_sha256
            ),
            expected_eval647_materialization_evidence_identity_sha256=(
                expected_eval647_materialization_evidence_identity_sha256
            ),
            expected_eval647_object_set_identity_sha256=(
                expected_eval647_object_set_identity_sha256
            ),
            expected_inventory_identity_sha256=inventory[
                "inventory_identity_sha256"
            ],
            expected_survivor_authority_sha256=subset[
                "subset_authority_sha256"
            ],
            expected_training_handoff_identity_sha256=handoff[
                "handoff_identity_sha256"
            ],
            expected_selection_validation_identity_sha256=(
                expected_selection_validation_identity_sha256
            ),
            expected_final_test_identity_sha256=(
                expected_final_test_identity_sha256
            ),
        )
        clean.verify_current_clean_composition_receipt(
            receipt,
            expected_receipt_identity_sha256=receipt[
                "receipt_identity_sha256"
            ],
            expected_data232_report_sha256=receipt[
                "data232_report_sha256"
            ],
            expected_decontamination_execution_identity_sha256=receipt[
                "decontamination_execution_identity_sha256"
            ],
            expected_eval647_execution_receipt_identity_sha256=receipt[
                "eval647_execution_receipt_identity_sha256"
            ],
            expected_quality_execution_identity_sha256=receipt[
                "quality_execution_identity_sha256"
            ],
            expected_privacy_execution_identity_sha256=receipt[
                "privacy_execution_identity_sha256"
            ],
            expected_post_decontamination_input_rows_sha256=receipt[
                "post_decontamination_input_rows_sha256"
            ],
            expected_post_quality_input_rows_sha256=receipt[
                "post_quality_input_rows_sha256"
            ],
            expected_survivor_jsonl_sha256=receipt[
                "survivor_jsonl_sha256"
            ],
            expected_survivor_record_inventory_digest_sha256=receipt[
                "survivor_record_inventory_digest_sha256"
            ],
            expected_survivor_payload_inventory_digest_sha256=receipt[
                "survivor_payload_inventory_digest_sha256"
            ],
        )
    finally:
        restore_clean_release(original)

    final_bytes = int(receipt["survivor_payload_bytes"])
    require(
        0 <= final_bytes <= EXPECTED_RADA_BYTES,
        "Rada post-QP byte widening",
    )
    require(
        survivor_inventory.get("total_payload_bytes") == final_bytes,
        "Rada survivor inventory/receipt byte drift",
    )
    require(
        len(final_survivors) == receipt["survivor_records"],
        "Rada survivor count drift",
    )

    evidence_core = {
        "schema_version": SCHEMA,
        "execution_profile": "LOCAL_FREE",
        "execution_head_sha": expected_execution_head,
        "parent": {
            "execution_head_sha": PARENT_EXECUTION_HEAD,
            "workflow_run_id": PARENT_WORKFLOW_RUN_ID,
            "artifact_id": PARENT_ARTIFACT_ID,
            "artifact_zip_sha256": PARENT_ARTIFACT_ZIP_SHA256,
            "matcher_report_sha256": PARENT_MATCHER_REPORT_SHA256,
            "survivor_authority_sha256": PARENT_SURVIVOR_AUTHORITY_SHA256,
            "two_clean_authority_sha256": (
                PARENT_TWO_CLEAN_AUTHORITY_SHA256
            ),
            "source_objects": EXPECTED_RADA_OBJECTS,
            "payload_bytes": EXPECTED_RADA_BYTES,
        },
        "rights_authority": rights_authority,
        "rada_authority": {
            "source_family": SOURCE_FAMILY,
            "source_object_count": EXPECTED_RADA_OBJECTS,
            "pre_gate_payload_bytes": EXPECTED_RADA_BYTES,
            "inventory_identity_sha256": inventory[
                "inventory_identity_sha256"
            ],
            "subset_authority_sha256": subset[
                "subset_authority_sha256"
            ],
            "training_records_sha256": records_sha,
            "training_handoff_raw_sha256": handoff_raw_sha,
            "training_handoff_identity_sha256": handoff[
                "handoff_identity_sha256"
            ],
            "source_authority": source_authority,
        },
        "gate_execution": {
            "data232_report_sha256": receipt["data232_report_sha256"],
            "decontamination_execution_identity_sha256": receipt[
                "decontamination_execution_identity_sha256"
            ],
            "eval647_execution_receipt_identity_sha256": receipt[
                "eval647_execution_receipt_identity_sha256"
            ],
            "quality_execution_identity_sha256": receipt[
                "quality_execution_identity_sha256"
            ],
            "privacy_execution_identity_sha256": receipt[
                "privacy_execution_identity_sha256"
            ],
            "composition_receipt_identity_sha256": receipt[
                "receipt_identity_sha256"
            ],
            "input_records": receipt["input_training_records"],
            "post_decontamination_records": receipt[
                "post_decontamination_records"
            ],
            "post_quality_records": receipt["post_quality_records"],
            "survivor_records": receipt["survivor_records"],
            "survivor_source_objects": receipt[
                "survivor_source_objects"
            ],
            "survivor_payload_bytes": final_bytes,
            "survivor_jsonl_sha256": receipt[
                "survivor_jsonl_sha256"
            ],
            "survivor_record_inventory_digest_sha256": receipt[
                "survivor_record_inventory_digest_sha256"
            ],
            "survivor_payload_inventory_digest_sha256": receipt[
                "survivor_payload_inventory_digest_sha256"
            ],
            "rejection_counts": receipt["rejection_counts"],
            "privacy_detector_counts": receipt[
                "privacy_detector_counts"
            ],
            "later_gate_loss_bytes": EXPECTED_RADA_BYTES - final_bytes,
        },
        "survivor_inventory": survivor_inventory,
        "authority_blobs": authority_blobs,
        "content_boundary": {
            "raw_training_text_persisted": False,
            "raw_evaluation_text_persisted": False,
            "raw_survivor_text_persisted": False,
            "durable_output_text_free": True,
        },
        "truth_boundary": {
            "training_purpose_rights_complete_for_exact_rada_snapshot": True,
            "global_cross_source_dedup_complete_for_rada_parent": True,
            "reserved_evaluation_decontamination_complete_for_current_rada": True,
            "canonical_quality_privacy_complete_for_current_rada": True,
            "cross_lineage_rededup_after_post_qp_complete": False,
            "balance_diversity_retest_complete": False,
            "family_caps_complete": False,
            "cluster_safe_split_complete": False,
            "deterministic_pack_two_clean_complete": False,
            "positive_exact_unique_loss_ledger": False,
            "canonical_capacity_credited": 0,
            "training_authorized_bytes": 0,
            "authorized_unique_loss_positions": 0,
            "authorized_optimized_target_exposure": 0,
            "tokenizer_fit_authorized": False,
            "optimizer_updates_executed_on_real_targets": 0,
            "training_executed": False,
            "learned_weights_created": False,
            "final_test_outcomes_read": False,
            "paid_compute_used": False,
            "foreign_pretrained_weights_used": False,
            "scale_promotion_authorized": False,
        },
    }
    evidence = {
        **evidence_core,
        "evidence_identity_sha256": sha256(canonical(evidence_core)),
    }
    output_evidence.parent.mkdir(parents=True, exist_ok=True)
    output_evidence.write_bytes(canonical(evidence) + b"\n")
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--expected-execution-head", required=True)
    parser.add_argument("--parent-survivors-json", type=Path, required=True)
    parser.add_argument("--parent-evidence-json", type=Path, required=True)
    parser.add_argument("--parent-two-clean-json", type=Path, required=True)
    parser.add_argument("--rights-a-json", type=Path, required=True)
    parser.add_argument("--rights-b-json", type=Path, required=True)
    parser.add_argument("--candidate-jsonl", type=Path, required=True)
    parser.add_argument("--evaluation-records-jsonl", type=Path, required=True)
    parser.add_argument("--base-reserved-binding-json", type=Path, required=True)
    parser.add_argument("--eval647-manifest-json", type=Path, required=True)
    parser.add_argument(
        "--eval647-materialization-evidence-json",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--expected-base-reserved-binding-identity-sha256",
        required=True,
    )
    parser.add_argument(
        "--expected-composed-reserved-binding-identity-sha256",
        required=True,
    )
    parser.add_argument(
        "--expected-eval647-materialization-evidence-identity-sha256",
        required=True,
    )
    parser.add_argument(
        "--expected-eval647-object-set-identity-sha256",
        required=True,
    )
    parser.add_argument(
        "--expected-selection-validation-identity-sha256",
        required=True,
    )
    parser.add_argument("--expected-final-test-identity-sha256", required=True)
    parser.add_argument("--output-evidence", type=Path, required=True)
    args = parser.parse_args()
    try:
        evidence = execute(
            expected_execution_head=args.expected_execution_head,
            parent_survivors_json=args.parent_survivors_json,
            parent_evidence_json=args.parent_evidence_json,
            parent_two_clean_json=args.parent_two_clean_json,
            rights_a_json=args.rights_a_json,
            rights_b_json=args.rights_b_json,
            candidate_jsonl=args.candidate_jsonl,
            evaluation_records_jsonl=args.evaluation_records_jsonl,
            base_reserved_binding_json=args.base_reserved_binding_json,
            eval647_manifest_json=args.eval647_manifest_json,
            eval647_materialization_evidence_json=(
                args.eval647_materialization_evidence_json
            ),
            expected_base_reserved_binding_identity_sha256=(
                args.expected_base_reserved_binding_identity_sha256
            ),
            expected_composed_reserved_binding_identity_sha256=(
                args.expected_composed_reserved_binding_identity_sha256
            ),
            expected_eval647_materialization_evidence_identity_sha256=(
                args.expected_eval647_materialization_evidence_identity_sha256
            ),
            expected_eval647_object_set_identity_sha256=(
                args.expected_eval647_object_set_identity_sha256
            ),
            expected_selection_validation_identity_sha256=(
                args.expected_selection_validation_identity_sha256
            ),
            expected_final_test_identity_sha256=(
                args.expected_final_test_identity_sha256
            ),
            output_evidence=args.output_evidence,
        )
    except (
        RadaCleanExecutionError,
        rada.RadaCurrentSnapshotDedupAdapterError,
        clean.CurrentCleanExecutionError,
        reserved.CurrentDecontaminationExecutionError,
        OSError,
        UnicodeError,
        ValueError,
    ) as exc:
        detail = " ".join(str(exc).split())[:320]
        print(f"D03_RADA_CURRENT_POST_QP=BLOCKED: {detail}")
        return 2

    print("D03_RADA_CURRENT_POST_QP=PASS_ZERO_CREDIT")
    print(
        "EVIDENCE_IDENTITY_SHA256="
        f"{evidence['evidence_identity_sha256']}"
    )
    print(
        "POST_QP_RADA_BYTES="
        f"{evidence['gate_execution']['survivor_payload_bytes']}"
    )
    print(
        "POST_QP_RADA_OBJECTS="
        f"{evidence['gate_execution']['survivor_records']}"
    )
    print("EXACT_INCREMENTAL_UNIQUE_CAPACITY_CREDITED_BYTES=0")
    print("CROSS_LINEAGE_REDEDUP_REQUIRED=true")
    print("AUTHORIZED_OPTIMIZED_TARGET_EXPOSURE=0")
    print("TRAINING_EXECUTED=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
