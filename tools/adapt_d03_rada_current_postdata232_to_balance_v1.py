#!/usr/bin/env python3
"""Bind current-Rada post-DATA232 G05/G06 survivors to the merged balance vector.

This execution-only adapter authenticates the exact text-free G05/G06 receipt and
survivor inventory produced by its pinned parent, derives the already-merged
postmaterialization family-vector schema, and validates that vector with the merged
#945 verifier.

It defines no balance policy, quality/privacy rule, family identity, tokenizer rule,
training authority, optimizer behavior, or scale policy. Raw survivor text is never
loaded or persisted here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

STACK_BASE_HEAD = "f3adefadf5a7061e498969a23e3535024a62a275"
PARENT_SCHEMA = "12-6.d03-rada-current-postdata232-g05-g06-execution.v1"
PARENT_DATA232_HEAD = "b66ac95e68f83641ff8134baabd55a29fd62b99f"
PRODUCT_DATA232_HEAD = "3bd56b62b318e1ecf5b1b5f16298a85350d31efb"
GLOBAL_DEDUP_HEAD = "a4663e87b010b190343caf1d42784f5dc7984601"
EXPECTED_FULL_SELECTION_SHA256 = (
    "601398c39769dabb930997f25506eaaf71bd8960aa82d8af9c697d1cc4075e22"
)
EXPECTED_RADA_SLICE_SHA256 = (
    "f103a3f18216519bd9228e586bd673d49f73cace03b3b0278f2dd0a33383bffb"
)
FAMILY_VECTOR_SCHEMA = "12-6.d03-postmaterialization-family-vector.v1"
INVENTORY_SCHEMA = "12-6.data526-record-inventory.v1"

EXPECTED_CANONICAL_BLOBS = {
    "src/twelve_six/__init__.py":
        "5433166c507bc845bd12d8d5c4145f1fbedda204",
    "src/twelve_six/data/trusted_family_authority_v1.py":
        "9382332d2d0d5c09aa5582e6f949b1630c459cbb",
    "src/twelve_six/data/postdecontam_balance_projection_v1.py":
        "8512a7363cb3953c46d7072cb4b3c09c86751236",
    "src/twelve_six/data/postmaterialization_balance_projection_v1.py":
        "62a6640ad5911e21f406ef12c1b0d7e5e5b1afef",
}

_SHA40 = re.compile(r"^[0-9a-f]{40}$")
_SHA64 = re.compile(r"^[0-9a-f]{64}$")


class CurrentRadaBalanceAdapterError(RuntimeError):
    """Raised when current-Rada balance projection cannot be trusted."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CurrentRadaBalanceAdapterError(message)


def canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_line(value: Any) -> bytes:
    return canonical(value) + b"\n"


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def require_sha256(value: Any, label: str) -> str:
    require(
        isinstance(value, str) and _SHA64.fullmatch(value) is not None,
        f"{label} must be 64 lowercase hex",
    )
    return value


def require_git_sha(value: Any, label: str) -> str:
    require(
        isinstance(value, str) and _SHA40.fullmatch(value) is not None,
        f"{label} must be 40 lowercase hex",
    )
    return value


def require_positive_int(value: Any, label: str) -> int:
    require(
        type(value) is int and value > 0,
        f"{label} must be a positive exact integer",
    )
    return value


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant rejected: {value}")


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key rejected: {key}")
        result[key] = value
    return result


def load_json_bytes(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise CurrentRadaBalanceAdapterError(
            f"{label}: strict JSON decode failed"
        ) from exc
    require(isinstance(value, dict), f"{label}: root must be an object")
    return value


def _git(*args: str) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    require(
        proc.returncode == 0,
        "git command failed: " + " ".join(args),
    )
    return proc.stdout.strip()


def _git_is_ancestor(ancestor: str, descendant: str) -> bool:
    proc = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return proc.returncode == 0


def verify_parent_execution_ancestry(expected_parent_head: str) -> str:
    expected = require_git_sha(
        expected_parent_head,
        "expected parent G05/G06 execution head",
    )
    current = require_git_sha(_git("rev-parse", "HEAD"), "adapter checkout head")
    require(
        _git_is_ancestor(STACK_BASE_HEAD, expected),
        "terminal parent head is outside the stacked #2861 lineage",
    )
    require(
        _git_is_ancestor(expected, current),
        "adapter checkout does not contain terminal parent head",
    )
    return expected


def verify_canonical_dependency_blobs() -> dict[str, str]:
    observed: dict[str, str] = {}
    for relative, expected in EXPECTED_CANONICAL_BLOBS.items():
        committed = _git("rev-parse", f"HEAD:{relative}")
        working = _git("hash-object", relative)
        require(
            committed == expected,
            f"canonical committed blob drift: {relative}",
        )
        require(
            working == expected,
            f"canonical working-tree blob drift: {relative}",
        )
        observed[relative] = expected
    return dict(sorted(observed.items()))


def load_canonical_bridge() -> tuple[Any, Any, Any]:
    from twelve_six.data.postmaterialization_balance_projection_v1 import (
        verify_postmaterialization_family_vector,
    )
    from twelve_six.data.trusted_family_authority_v1 import (
        TRUSTED_FAMILY_SEMANTICS,
        trusted_family_authority_root_sha256,
    )

    return (
        verify_postmaterialization_family_vector,
        TRUSTED_FAMILY_SEMANTICS,
        trusted_family_authority_root_sha256,
    )


EXPECTED_CONTENT_BOUNDARY = {
    "raw_training_text_persisted": False,
    "raw_evaluation_text_persisted": False,
    "raw_survivor_text_persisted": False,
    "durable_output_text_free": True,
}

EXPECTED_TRUTH_BOUNDARY = {
    "current_rada_data232_parent_two_clean_complete": True,
    "canonical_quality_privacy_executed": True,
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
}


def verify_parent_receipt(
    evidence: Mapping[str, Any],
    *,
    expected_parent_execution_head: str,
    expected_evidence_identity_sha256: str,
    expected_inventory_file_sha256: str,
    expected_record_payload_jsonl_sha256: str,
    expected_record_count: int,
    expected_total_payload_bytes: int,
    expected_source_object_count: int,
    expected_record_inventory_digest_sha256: str,
    expected_payload_inventory_digest_sha256: str,
) -> None:
    expected_identity = require_sha256(
        expected_evidence_identity_sha256,
        "expected evidence identity",
    )
    require(
        set(evidence)
        == {
            "schema_version",
            "execution_profile",
            "execution_head_sha",
            "parent",
            "g05",
            "g06",
            "durable_artifacts",
            "survivor_inventory",
            "counts",
            "authority_blobs",
            "content_boundary",
            "truth_boundary",
            "next_gate",
            "evidence_identity_sha256",
        },
        "parent evidence top-level fields drift",
    )
    require(evidence.get("schema_version") == PARENT_SCHEMA, "parent schema drift")
    require(evidence.get("execution_profile") == "LOCAL_FREE", "parent profile drift")
    expected_execution_head = require_git_sha(
        expected_parent_execution_head,
        "expected parent G05/G06 execution head",
    )
    require(
        evidence.get("execution_head_sha") == expected_execution_head,
        "parent G05/G06 execution head drift",
    )
    claimed = require_sha256(
        evidence.get("evidence_identity_sha256"),
        "parent evidence identity",
    )
    core = dict(evidence)
    del core["evidence_identity_sha256"]
    require(
        claimed == sha256(canonical(core)),
        "parent evidence self-hash mismatch",
    )
    require(claimed == expected_identity, "parent evidence identity is not expected")

    parent = evidence.get("parent")
    require(
        isinstance(parent, Mapping)
        and set(parent)
        == {
            "execution_head_sha",
            "product_data232_head_sha",
            "artifact_id",
            "artifact_zip_sha256",
            "inventory_identity_sha256",
            "training_handoff_identity_sha256",
            "data232_report_sha256",
            "data232_execution_identity_sha256",
            "data232_result_identity_sha256",
            "data232_two_clean_proof_identity_sha256",
            "full_selection_projection_sha256",
            "current_rada_slice_authority_sha256",
            "two_fresh_data232_processes_byte_identical",
        },
        "parent lineage fields drift",
    )
    require(
        parent.get("execution_head_sha") == PARENT_DATA232_HEAD
        and parent.get("product_data232_head_sha") == PRODUCT_DATA232_HEAD
        and parent.get("full_selection_projection_sha256")
        == EXPECTED_FULL_SELECTION_SHA256
        and parent.get("current_rada_slice_authority_sha256")
        == EXPECTED_RADA_SLICE_SHA256
        and parent.get("two_fresh_data232_processes_byte_identical") is True,
        "parent DATA232/global-dedup lineage drift",
    )
    require_positive_int(parent.get("artifact_id"), "parent.artifact_id")
    require_sha256(parent.get("artifact_zip_sha256"), "parent.artifact_zip_sha256")
    for field in (
        "inventory_identity_sha256",
        "training_handoff_identity_sha256",
        "data232_report_sha256",
        "data232_execution_identity_sha256",
        "data232_result_identity_sha256",
        "data232_two_clean_proof_identity_sha256",
    ):
        require_sha256(parent.get(field), f"parent.{field}")

    g05 = evidence.get("g05")
    g06 = evidence.get("g06")
    require(isinstance(g05, Mapping), "G05 receipt missing")
    require(isinstance(g06, Mapping), "G06 receipt missing")
    require_sha256(g05.get("input_rows_sha256"), "G05 input root")
    require_sha256(g05.get("execution_identity_sha256"), "G05 execution identity")
    require_sha256(g06.get("input_rows_sha256"), "G06 input root")
    require_sha256(g06.get("execution_identity_sha256"), "G06 execution identity")

    authority_blobs = evidence.get("authority_blobs")
    require(
        isinstance(authority_blobs, Mapping) and bool(authority_blobs),
        "parent authority blob map missing",
    )
    for relative, blob in authority_blobs.items():
        require(
            isinstance(relative, str) and bool(relative),
            "parent authority path missing",
        )
        require_git_sha(blob, f"parent authority blob {relative}")

    artifacts = evidence.get("durable_artifacts")
    require(
        isinstance(artifacts, Mapping)
        and set(artifacts)
        == {
            "g05_authority_file_sha256",
            "g06_authority_file_sha256",
            "survivor_inventory_file_sha256",
            "completion_marker",
        },
        "durable artifact receipt fields drift",
    )
    require(
        artifacts.get("completion_marker")
        == "POST_G05_G06_EVIDENCE_WRITTEN_LAST",
        "parent completion marker missing",
    )
    require(
        artifacts.get("survivor_inventory_file_sha256")
        == require_sha256(
            expected_inventory_file_sha256,
            "expected inventory file SHA-256",
        ),
        "parent inventory file SHA-256 drift",
    )
    require_sha256(
        artifacts.get("g05_authority_file_sha256"),
        "G05 authority file SHA-256",
    )
    require_sha256(
        artifacts.get("g06_authority_file_sha256"),
        "G06 authority file SHA-256",
    )

    survivor = evidence.get("survivor_inventory")
    require(
        isinstance(survivor, Mapping)
        and set(survivor)
        == {
            "record_payload_jsonl_sha256",
            "record_count",
            "total_payload_bytes",
            "record_inventory_digest_sha256",
            "payload_inventory_digest_sha256",
        },
        "parent survivor receipt fields drift",
    )
    require(
        survivor.get("record_payload_jsonl_sha256")
        == require_sha256(
            expected_record_payload_jsonl_sha256,
            "expected record-payload JSONL SHA-256",
        ),
        "record-payload JSONL root drift",
    )
    require(
        survivor.get("record_count") == expected_record_count,
        "receipt record count drift",
    )
    require(
        survivor.get("total_payload_bytes") == expected_total_payload_bytes,
        "receipt payload-byte count drift",
    )
    require(
        survivor.get("record_inventory_digest_sha256")
        == require_sha256(
            expected_record_inventory_digest_sha256,
            "expected record inventory digest",
        ),
        "receipt record inventory root drift",
    )
    require(
        survivor.get("payload_inventory_digest_sha256")
        == require_sha256(
            expected_payload_inventory_digest_sha256,
            "expected payload inventory digest",
        ),
        "receipt payload inventory root drift",
    )

    counts = evidence.get("counts")
    require(isinstance(counts, Mapping), "parent counts missing")
    require(
        counts.get("post_g06_records") == expected_record_count
        and counts.get("post_g06_source_objects") == expected_source_object_count
        and counts.get("post_g06_payload_bytes") == expected_total_payload_bytes,
        "parent count projection drift",
    )
    require(
        evidence.get("content_boundary") == EXPECTED_CONTENT_BOUNDARY,
        "parent content boundary drift",
    )
    require(
        evidence.get("truth_boundary") == EXPECTED_TRUTH_BOUNDARY,
        "parent zero-credit truth boundary drift",
    )
    require(
        evidence.get("next_gate")
        == "CURRENT_RADA_BALANCE_DIVERSITY_FAMILY_CAP_RETEST",
        "parent next gate drift",
    )


def verify_inventory(
    inventory: Mapping[str, Any],
    *,
    expected_record_count: int,
    expected_total_payload_bytes: int,
    expected_source_object_count: int,
    expected_record_inventory_digest_sha256: str,
    expected_payload_inventory_digest_sha256: str,
) -> list[dict[str, Any]]:
    require(
        set(inventory)
        == {
            "schema_version",
            "record_count",
            "total_payload_bytes",
            "record_inventory_digest_sha256",
            "payload_inventory_digest_sha256",
            "records",
        },
        "inventory top-level fields drift",
    )
    require(
        inventory.get("schema_version") == INVENTORY_SCHEMA,
        "inventory schema drift",
    )
    rows = inventory.get("records")
    require(isinstance(rows, list) and bool(rows), "inventory records missing")

    expected_row_fields = {
        "record_id",
        "source_id",
        "family",
        "modality",
        "payload_sha256",
        "payload_bytes",
    }
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(rows):
        require(
            isinstance(raw, Mapping) and set(raw) == expected_row_fields,
            f"inventory.records[{index}] fields drift",
        )
        record_id = raw.get("record_id")
        source_id = raw.get("source_id")
        family = raw.get("family")
        modality = raw.get("modality")
        require(
            isinstance(record_id, str) and bool(record_id),
            f"inventory.records[{index}].record_id missing",
        )
        require(record_id not in seen, f"duplicate record_id: {record_id}")
        seen.add(record_id)
        for value, label in (
            (source_id, "source_id"),
            (family, "family"),
            (modality, "modality"),
        ):
            require(
                isinstance(value, str) and bool(value),
                f"inventory.records[{index}].{label} missing",
            )
        payload_sha = require_sha256(
            raw.get("payload_sha256"),
            f"inventory.records[{index}].payload_sha256",
        )
        payload_bytes = require_positive_int(
            raw.get("payload_bytes"),
            f"inventory.records[{index}].payload_bytes",
        )
        normalized.append(
            {
                "record_id": record_id,
                "source_id": source_id,
                "family": family,
                "modality": modality,
                "payload_sha256": payload_sha,
                "payload_bytes": payload_bytes,
            }
        )

    require(
        normalized == sorted(normalized, key=lambda row: row["record_id"]),
        "inventory record order drift",
    )
    record_count = len(normalized)
    total_bytes = sum(row["payload_bytes"] for row in normalized)
    source_count = len({row["source_id"] for row in normalized})
    record_root = sha256(canonical(normalized))
    payload_projection = [
        {
            "record_id": row["record_id"],
            "payload_sha256": row["payload_sha256"],
            "payload_bytes": row["payload_bytes"],
        }
        for row in normalized
    ]
    payload_root = sha256(canonical(payload_projection))

    require(record_count == expected_record_count, "inventory record count drift")
    require(total_bytes == expected_total_payload_bytes, "inventory byte count drift")
    require(source_count == expected_source_object_count, "inventory source count drift")
    require(
        record_root
        == require_sha256(
            expected_record_inventory_digest_sha256,
            "expected record inventory digest",
        ),
        "inventory record root drift",
    )
    require(
        payload_root
        == require_sha256(
            expected_payload_inventory_digest_sha256,
            "expected payload inventory digest",
        ),
        "inventory payload root drift",
    )
    require(
        inventory.get("record_count") == record_count
        and inventory.get("total_payload_bytes") == total_bytes
        and inventory.get("record_inventory_digest_sha256") == record_root
        and inventory.get("payload_inventory_digest_sha256") == payload_root,
        "inventory declared summary drift",
    )
    return normalized


def _trusted_stratum(
    row: Mapping[str, Any],
    trusted: Mapping[str, Mapping[str, Any]],
) -> str:
    family = str(row["family"])
    authority = trusted.get(family)
    require(authority is not None, f"untrusted survivor family: {family}")
    stratum = str(authority["stratum"])
    modality = str(row["modality"])
    if stratum == "code":
        require(modality == "code", f"code family modality drift: {family}")
    elif stratum == "en":
        require(
            modality in {"en", "text"},
            f"English family modality drift: {family}",
        )
    elif stratum == "uk":
        require(
            modality in {"uk", "ua", "text"},
            f"Ukrainian family modality drift: {family}",
        )
    else:
        raise CurrentRadaBalanceAdapterError(
            f"unsupported trusted family stratum: {family}"
        )
    return stratum


def build_family_vector(
    *,
    evidence: Mapping[str, Any],
    inventory_rows: list[dict[str, Any]],
    trusted: Mapping[str, Mapping[str, Any]],
    trusted_root_fn: Any,
    materialization_execution_head: str,
) -> dict[str, Any]:
    by_family_bytes: dict[tuple[str, str], int] = defaultdict(int)
    by_family_records: dict[tuple[str, str], int] = defaultdict(int)
    for row in inventory_rows:
        stratum = _trusted_stratum(row, trusted)
        key = (stratum, row["family"])
        by_family_bytes[key] += row["payload_bytes"]
        by_family_records[key] += 1

    families = [
        {
            "stratum": stratum,
            "family": family,
            "record_count": by_family_records[(stratum, family)],
            "capacity_bytes": by_family_bytes[(stratum, family)],
        }
        for stratum, family in sorted(by_family_bytes)
    ]
    family_names = {row["family"] for row in families}
    record_membership = [
        {
            "record_id": row["record_id"],
            "source_id": row["source_id"],
            "family": row["family"],
            "modality": row["modality"],
        }
        for row in inventory_rows
    ]
    survivor = evidence["survivor_inventory"]
    counts = evidence["counts"]
    vector: dict[str, Any] = {
        "schema": FAMILY_VECTOR_SCHEMA,
        "status": "PASS",
        "source_git_sha": materialization_execution_head,
        "materialization_identity_sha256": evidence[
            "evidence_identity_sha256"
        ],
        "materialization_execution_head_sha": materialization_execution_head,
        "record_payload_jsonl_sha256": survivor[
            "record_payload_jsonl_sha256"
        ],
        "record_inventory_digest_sha256": survivor[
            "record_inventory_digest_sha256"
        ],
        "payload_inventory_digest_sha256": survivor[
            "payload_inventory_digest_sha256"
        ],
        "record_count": survivor["record_count"],
        "total_payload_bytes": survivor["total_payload_bytes"],
        "source_object_count": counts["post_g06_source_objects"],
        "record_membership_sha256": sha256(canonical(record_membership)),
        "trusted_family_authority_root_sha256": trusted_root_fn(family_names),
        "families": families,
        "stratum_capacity_bytes": {
            stratum: sum(
                row["capacity_bytes"]
                for row in families
                if row["stratum"] == stratum
            )
            for stratum in ("code", "en", "uk")
        },
        "stratum_family_counts": {
            stratum: sum(
                1 for row in families if row["stratum"] == stratum
            )
            for stratum in ("code", "en", "uk")
        },
        "next_gate": "NEXT100-106_BALANCE_FAMILY_CAP",
        "training_eligible": False,
        "evaluation_eligible": False,
        "tokenizer_fit_authorized": False,
        "model_training_authorized": False,
        "authorized_optimized_target_exposure": 0,
        "training_authorized_by_this_report": False,
    }
    vector["family_vector_identity_sha256"] = sha256(canonical(vector))
    return vector


def write_immutable(path: Path, payload: bytes) -> None:
    require(not path.is_symlink(), "output must not be a symlink")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.is_file(), "output exists but is not a regular file")
        require(path.read_bytes() == payload, "refusing divergent output overwrite")
        return
    temp = path.with_name(path.name + ".tmp")
    require(not temp.exists() and not temp.is_symlink(), "output temp already exists")
    created_temp = False
    try:
        with temp.open("xb") as handle:
            created_temp = True
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temp.replace(path)
    except OSError:
        if created_temp:
            try:
                temp.unlink(missing_ok=True)
            except OSError:
                pass
        raise


def execute(args: argparse.Namespace) -> dict[str, Any]:
    canonical_blobs = verify_canonical_dependency_blobs()
    (
        canonical_verify,
        trusted,
        trusted_root_fn,
    ) = load_canonical_bridge()

    evidence_raw = args.evidence.read_bytes()
    inventory_raw = args.inventory.read_bytes()
    require(
        sha256(evidence_raw)
        == require_sha256(
            args.expected_evidence_file_sha256,
            "expected evidence file SHA-256",
        ),
        "parent evidence file SHA-256 drift",
    )
    require(
        sha256(inventory_raw)
        == require_sha256(
            args.expected_inventory_file_sha256,
            "expected inventory file SHA-256",
        ),
        "parent inventory file SHA-256 drift",
    )
    evidence = load_json_bytes(evidence_raw, "parent evidence")
    inventory = load_json_bytes(inventory_raw, "survivor inventory")

    expected_record_count = require_positive_int(
        args.expected_record_count,
        "expected record count",
    )
    expected_total_payload_bytes = require_positive_int(
        args.expected_total_payload_bytes,
        "expected total payload bytes",
    )
    expected_source_object_count = require_positive_int(
        args.expected_source_object_count,
        "expected source object count",
    )

    expected_parent_execution_head = verify_parent_execution_ancestry(
        args.expected_parent_execution_head
    )
    verify_parent_receipt(
        evidence,
        expected_parent_execution_head=expected_parent_execution_head,
        expected_evidence_identity_sha256=(
            args.expected_evidence_identity_sha256
        ),
        expected_inventory_file_sha256=args.expected_inventory_file_sha256,
        expected_record_payload_jsonl_sha256=(
            args.expected_record_payload_jsonl_sha256
        ),
        expected_record_count=expected_record_count,
        expected_total_payload_bytes=expected_total_payload_bytes,
        expected_source_object_count=expected_source_object_count,
        expected_record_inventory_digest_sha256=(
            args.expected_record_inventory_digest_sha256
        ),
        expected_payload_inventory_digest_sha256=(
            args.expected_payload_inventory_digest_sha256
        ),
    )
    rows = verify_inventory(
        inventory,
        expected_record_count=expected_record_count,
        expected_total_payload_bytes=expected_total_payload_bytes,
        expected_source_object_count=expected_source_object_count,
        expected_record_inventory_digest_sha256=(
            args.expected_record_inventory_digest_sha256
        ),
        expected_payload_inventory_digest_sha256=(
            args.expected_payload_inventory_digest_sha256
        ),
    )

    vector = build_family_vector(
        evidence=evidence,
        inventory_rows=rows,
        trusted=trusted,
        trusted_root_fn=trusted_root_fn,
        materialization_execution_head=expected_parent_execution_head,
    )
    identity = vector["family_vector_identity_sha256"]
    canonical_verify(vector, expected_identity_sha256=identity)
    require(
        vector["authorized_optimized_target_exposure"] == 0
        and vector["tokenizer_fit_authorized"] is False
        and vector["model_training_authorized"] is False,
        "canonical vector widened scientific boundary",
    )
    require(
        all(
            "normalized_payload" not in row
            and "text" not in row
            for row in inventory_rows
        ),
        "raw survivor payload leaked into durable adapter state",
    )
    write_immutable(args.output, canonical_line(vector))
    return {
        "family_vector_identity_sha256": identity,
        "canonical_dependency_blobs": canonical_blobs,
        "record_count": vector["record_count"],
        "total_payload_bytes": vector["total_payload_bytes"],
        "source_object_count": vector["source_object_count"],
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    result.add_argument("--evidence", type=Path, required=True)
    result.add_argument("--expected-parent-execution-head", required=True)
    result.add_argument("--inventory", type=Path, required=True)
    result.add_argument("--expected-evidence-file-sha256", required=True)
    result.add_argument("--expected-evidence-identity-sha256", required=True)
    result.add_argument("--expected-inventory-file-sha256", required=True)
    result.add_argument("--expected-record-payload-jsonl-sha256", required=True)
    result.add_argument("--expected-record-inventory-digest-sha256", required=True)
    result.add_argument("--expected-payload-inventory-digest-sha256", required=True)
    result.add_argument("--expected-record-count", type=int, required=True)
    result.add_argument("--expected-total-payload-bytes", type=int, required=True)
    result.add_argument("--expected-source-object-count", type=int, required=True)
    result.add_argument("--output", type=Path, required=True)
    return result


def main() -> int:
    try:
        result = execute(parser().parse_args())
    except (CurrentRadaBalanceAdapterError, OSError, RuntimeError, ValueError) as exc:
        detail = " ".join(str(exc).split())[:500]
        print(f"D03_RADA_CURRENT_BALANCE_ADAPTER=BLOCKED: {detail}")
        return 2
    print("D03_RADA_CURRENT_BALANCE_ADAPTER=PASS_ZERO_CREDIT")
    print(
        "FAMILY_VECTOR_IDENTITY_SHA256="
        + result["family_vector_identity_sha256"]
    )
    print("AUTHORIZED_OPTIMIZED_TARGET_EXPOSURE=0")
    print("TOKENIZER_FIT_AUTHORIZED=false")
    print("TRAINING_EXECUTED=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
