"""Fail-closed independence-cluster guard for terminal corpus split manifests.

D04 does not decide decontamination truth and does not create a split here. The guard
consumes one externally identified, text-free terminal decontamination handoff plus one
complete split manifest. It proves that every independence/origin cluster is atomic
across train/selection/final_test and preserves the evaluation-reservation firewall.

This package authorizes zero training exposure. It exists specifically because global
dedup may retain multiple capacity-bearing sibling objects from one upstream origin;
those objects may contribute capacity but must not be scattered across splits.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

HANDOFF_SCHEMA = "12-6.d04-decontaminated-split-handoff.v1"
SPLIT_MANIFEST_SCHEMA = "12-6.d04-cluster-safe-split-manifest.v1"
PROOF_SCHEMA = "12-6.d04-cluster-safe-split-proof.v1"
SPLIT_POLICY = "INDEPENDENCE_CLUSTER_ATOMIC_V1"
ALLOWED_SPLITS = frozenset({"train", "selection", "final_test"})

_RECORD_KEYS = frozenset(
    {
        "record_id",
        "source_id",
        "family",
        "modality",
        "payload_sha256",
        "payload_bytes",
        "independence_cluster_identity_sha256",
        "evaluation_reserved",
    }
)
_ASSIGNMENT_KEYS = frozenset({"record_id", "split"})


class ClusterSafeSplitError(RuntimeError):
    """Raised when split authority is incomplete, stale, or leakage-prone."""


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_obj_without(value: Mapping[str, Any], identity_key: str) -> str:
    core = deepcopy(dict(value))
    core.pop(identity_key, None)
    return _sha256_bytes(_canonical_bytes(core))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ClusterSafeSplitError(message)


def _require_sha256(value: Any, label: str) -> str:
    _require(
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value),
        f"{label} must be lowercase 64-hex SHA-256",
    )
    return value


def _require_text(value: Any, label: str) -> str:
    _require(isinstance(value, str) and bool(value), f"{label} must be non-empty text")
    return value


def _require_nonnegative_int(value: Any, label: str) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0,
        f"{label} must be a non-negative integer",
    )
    return int(value)


def _validate_handoff(
    handoff: Mapping[str, Any],
    *,
    expected_decontamination_authority_sha256: str,
    expected_handoff_identity_sha256: str,
) -> list[dict[str, Any]]:
    _require(isinstance(handoff, Mapping), "decontamination handoff must be an object")
    _require(handoff.get("schema_version") == HANDOFF_SCHEMA, "handoff schema drift")
    _require(handoff.get("decontamination_terminal") is True, "decontamination is nonterminal")
    _require(
        handoff.get("reserved_evaluation_decontamination_complete") is True,
        "reserved-evaluation decontamination is incomplete",
    )
    _require(handoff.get("raw_text_emitted") is False, "handoff must remain text-free")
    _require(handoff.get("final_test_payload_read") is False, "final-test payload boundary weakened")
    _require(
        handoff.get("authorized_training_exposure") == 0,
        "split handoff must not pre-authorize training exposure",
    )

    expected_decontam = _require_sha256(
        expected_decontamination_authority_sha256,
        "expected_decontamination_authority_sha256",
    )
    _require(
        handoff.get("decontamination_authority_identity_sha256") == expected_decontam,
        "handoff does not match expected terminal decontamination authority",
    )
    observed_identity = _require_sha256(
        handoff.get("handoff_identity_sha256"),
        "handoff_identity_sha256",
    )
    expected_identity = _require_sha256(
        expected_handoff_identity_sha256,
        "expected_handoff_identity_sha256",
    )
    _require(observed_identity == expected_identity, "handoff does not match expected identity")
    _require(
        _sha256_obj_without(handoff, "handoff_identity_sha256") == observed_identity,
        "handoff self-hash mismatch",
    )

    raw_records = handoff.get("records")
    _require(
        isinstance(raw_records, Sequence) and not isinstance(raw_records, (str, bytes)),
        "handoff records must be a sequence",
    )
    records: list[dict[str, Any]] = []
    seen_records: set[str] = set()
    source_to_cluster: dict[str, str] = {}
    for index, raw in enumerate(raw_records):
        _require(
            isinstance(raw, Mapping) and set(raw) == _RECORD_KEYS,
            f"records[{index}] fields do not match schema",
        )
        record_id = _require_text(raw["record_id"], f"records[{index}].record_id")
        _require(record_id not in seen_records, "duplicate record_id in split handoff")
        seen_records.add(record_id)
        source_id = _require_text(raw["source_id"], f"records[{index}].source_id")
        cluster = _require_sha256(
            raw["independence_cluster_identity_sha256"],
            f"records[{index}].independence_cluster_identity_sha256",
        )
        previous_cluster = source_to_cluster.setdefault(source_id, cluster)
        _require(
            previous_cluster == cluster,
            "one source_id maps to multiple independence clusters",
        )
        _require(
            isinstance(raw["evaluation_reserved"], bool),
            f"records[{index}].evaluation_reserved must be boolean",
        )
        records.append(
            {
                "record_id": record_id,
                "source_id": source_id,
                "family": _require_text(raw["family"], f"records[{index}].family"),
                "modality": _require_text(raw["modality"], f"records[{index}].modality"),
                "payload_sha256": _require_sha256(
                    raw["payload_sha256"], f"records[{index}].payload_sha256"
                ),
                "payload_bytes": _require_nonnegative_int(
                    raw["payload_bytes"], f"records[{index}].payload_bytes"
                ),
                "independence_cluster_identity_sha256": cluster,
                "evaluation_reserved": raw["evaluation_reserved"],
            }
        )
    records.sort(key=lambda row: row["record_id"])
    _require(
        list(raw_records) == records,
        "handoff records must use canonical record_id order",
    )
    _require(
        handoff.get("record_count") == len(records),
        "handoff record_count mismatch",
    )
    _require(bool(records), "terminal decontamination handoff must retain at least one record")
    return records


def _validate_manifest(
    manifest: Mapping[str, Any],
    *,
    expected_handoff_identity_sha256: str,
    expected_split_manifest_identity_sha256: str,
) -> dict[str, str]:
    _require(isinstance(manifest, Mapping), "split manifest must be an object")
    _require(manifest.get("schema_version") == SPLIT_MANIFEST_SCHEMA, "split manifest schema drift")
    _require(manifest.get("split_policy") == SPLIT_POLICY, "split policy drift")
    _require(
        manifest.get("input_handoff_identity_sha256") == expected_handoff_identity_sha256,
        "split manifest is bound to a different decontamination handoff",
    )
    observed_identity = _require_sha256(
        manifest.get("split_manifest_identity_sha256"),
        "split_manifest_identity_sha256",
    )
    expected_identity = _require_sha256(
        expected_split_manifest_identity_sha256,
        "expected_split_manifest_identity_sha256",
    )
    _require(observed_identity == expected_identity, "split manifest does not match expected identity")
    _require(
        _sha256_obj_without(manifest, "split_manifest_identity_sha256") == observed_identity,
        "split manifest self-hash mismatch",
    )

    raw_assignments = manifest.get("assignments")
    _require(
        isinstance(raw_assignments, Sequence)
        and not isinstance(raw_assignments, (str, bytes)),
        "split assignments must be a sequence",
    )
    assignments: list[dict[str, str]] = []
    by_record: dict[str, str] = {}
    for index, raw in enumerate(raw_assignments):
        _require(
            isinstance(raw, Mapping) and set(raw) == _ASSIGNMENT_KEYS,
            f"assignments[{index}] fields do not match schema",
        )
        record_id = _require_text(raw["record_id"], f"assignments[{index}].record_id")
        split = _require_text(raw["split"], f"assignments[{index}].split")
        _require(split in ALLOWED_SPLITS, f"unsupported split: {split}")
        _require(record_id not in by_record, "duplicate split assignment")
        by_record[record_id] = split
        assignments.append({"record_id": record_id, "split": split})
    assignments.sort(key=lambda row: row["record_id"])
    _require(
        list(raw_assignments) == assignments,
        "split assignments must use canonical record_id order",
    )
    return by_record


def verify_cluster_safe_split(
    handoff: Mapping[str, Any],
    split_manifest: Mapping[str, Any],
    *,
    expected_decontamination_authority_sha256: str,
    expected_handoff_identity_sha256: str,
    expected_split_manifest_identity_sha256: str,
) -> dict[str, Any]:
    """Verify complete, reservation-safe, independence-cluster-atomic splitting."""
    records = _validate_handoff(
        handoff,
        expected_decontamination_authority_sha256=expected_decontamination_authority_sha256,
        expected_handoff_identity_sha256=expected_handoff_identity_sha256,
    )
    assignments = _validate_manifest(
        split_manifest,
        expected_handoff_identity_sha256=expected_handoff_identity_sha256,
        expected_split_manifest_identity_sha256=expected_split_manifest_identity_sha256,
    )
    expected_ids = {row["record_id"] for row in records}
    _require(set(assignments) == expected_ids, "split assignment coverage must equal retained records")

    cluster_to_split: dict[str, str] = {}
    split_record_counts = {split: 0 for split in sorted(ALLOWED_SPLITS)}
    split_payload_bytes = {split: 0 for split in sorted(ALLOWED_SPLITS)}
    for record in records:
        split = assignments[record["record_id"]]
        cluster = record["independence_cluster_identity_sha256"]
        previous = cluster_to_split.setdefault(cluster, split)
        _require(
            previous == split,
            "independence cluster crosses split boundary",
        )
        if record["evaluation_reserved"]:
            _require(split != "train", "evaluation-reserved record assigned to train")
        else:
            _require(split == "train", "non-reserved record assigned to held-out split")
        split_record_counts[split] += 1
        split_payload_bytes[split] += record["payload_bytes"]

    cluster_projection = [
        {"independence_cluster_identity_sha256": cluster, "split": split}
        for cluster, split in sorted(cluster_to_split.items())
    ]
    proof_core: dict[str, Any] = {
        "schema_version": PROOF_SCHEMA,
        "decontamination_authority_identity_sha256": _require_sha256(
            expected_decontamination_authority_sha256,
            "expected_decontamination_authority_sha256",
        ),
        "input_handoff_identity_sha256": _require_sha256(
            expected_handoff_identity_sha256,
            "expected_handoff_identity_sha256",
        ),
        "split_manifest_identity_sha256": _require_sha256(
            expected_split_manifest_identity_sha256,
            "expected_split_manifest_identity_sha256",
        ),
        "split_policy": SPLIT_POLICY,
        "record_count": len(records),
        "independence_cluster_count": len(cluster_to_split),
        "split_record_counts": split_record_counts,
        "split_payload_bytes": split_payload_bytes,
        "cluster_assignment_identity_sha256": _sha256_bytes(
            _canonical_bytes(cluster_projection)
        ),
        "complete_record_coverage": True,
        "independence_clusters_cross_splits": False,
        "evaluation_reserved_records_in_train": False,
        "raw_text_read": False,
        "final_test_payload_read": False,
        "authorized_training_exposure": 0,
        "tokenizer_fit_authorized": False,
        "model_training_executed": False,
        "paid_compute_used": False,
    }
    proof_core["proof_identity_sha256"] = _sha256_bytes(_canonical_bytes(proof_core))
    return proof_core
