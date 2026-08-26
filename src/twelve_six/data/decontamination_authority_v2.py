"""Public DATA-232 decontamination authority report API."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from twelve_six.data._data232_decontamination_matching import (
    ALGORITHM,
    CODE_SKELETON,
    DEFAULT_THRESHOLDS,
    NORMALIZATION,
    SCHEMA,
    DecontaminationError,
    _blocked_pairs,
    _fingerprint,
    _pair,
    _thresholds,
    _train_pairs,
    authority_composite_identity,
    sha256_bytes,
    stable_identity,
    validate_authority_metadata,
)


def _hashed_record(fp: dict[str, Any]) -> dict[str, Any]:
    r = fp["record"]
    return {
        "record_id_sha256": sha256_bytes(r["record_id"].encode()),
        "source_id_sha256": sha256_bytes(r["source_id"].encode()),
        "source_family_sha256": sha256_bytes(r["source_family"].encode()),
        "lineage_family_sha256": sha256_bytes(r["lineage_family"].encode()) if r.get("lineage_family") else None,
        "modality": r["modality"],
        "raw_sha256": fp["raw"],
        "normalized_sha256": fp["normalized"],
    }


def build_report(
    training_records: Sequence[Mapping[str, Any]],
    evaluation_records: Sequence[Mapping[str, Any]],
    *,
    training_corpus_identity: str,
    selection_validation_identity: str,
    final_test_identity: str,
    authorities: Mapping[str, Any],
    thresholds: Mapping[str, Any] | None = None,
    quarantine_cross_source_families: bool = True,
) -> dict[str, Any]:
    for name, identity in (
        ("training_corpus_identity", training_corpus_identity),
        ("selection_validation_identity", selection_validation_identity),
        ("final_test_identity", final_test_identity),
    ):
        if not isinstance(identity, str) or not re.fullmatch(r"[0-9a-f]{64}", identity):
            raise DecontaminationError(f"{name} must be a lowercase SHA-256 identity")
    validate_authority_metadata(authorities)
    t = _thresholds(thresholds)
    train = [_fingerprint(row, t) for row in training_records]
    evaluation = [_fingerprint(row, t) for row in evaluation_records]
    if not train or not evaluation:
        raise DecontaminationError("training and evaluation records must both be non-empty")
    train_ids = [row["record"]["record_id"] for row in train]
    eval_ids = [row["record"]["record_id"] for row in evaluation]
    if len(set(train_ids)) != len(train_ids) or len(set(eval_ids)) != len(eval_ids):
        raise DecontaminationError("record_id values must be unique within each partition")

    parent = {"t:" + x: "t:" + x for x in train_ids} | {"e:" + x: "e:" + x for x in eval_ids}

    def find(x: str) -> str:
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    matches = []
    for i, j in sorted(_blocked_pairs(train, evaluation)):
        evidence = _pair(train[i], evaluation[j], "eval", t)
        if evidence:
            union("t:" + train_ids[i], "e:" + eval_ids[j])
            matches.extend(evidence)
    for i, j in sorted(_train_pairs(train)):
        evidence = _pair(train[i], train[j], "peer", t)
        if evidence:
            union("t:" + train_ids[i], "t:" + train_ids[j])
            for item in evidence:
                item["match_scope"] = "training_cross_source_mirror" if item["cross_source_family"] else "training_within_family"
            matches.extend(evidence)

    eval_roots = {find("e:" + record_id) for record_id in eval_ids}
    excluded = {record_id for record_id in train_ids if find("t:" + record_id) in eval_roots}
    families = set()
    if quarantine_cross_source_families:
        by_id = {fp["record"]["record_id"]: fp for fp in train}
        for item in matches:
            if item.get("cross_source_family") and item["train_record_id"] in excluded:
                families.add(by_id[item["train_record_id"]]["record"]["source_family"])
        excluded |= {fp["record"]["record_id"] for fp in train if fp["record"]["source_family"] in families}

    by_id = {fp["record"]["record_id"]: fp for fp in train}
    family_hashes = {sha256_bytes(name.encode()) for name in families}
    exclusions = []
    for record_id in sorted(excluded):
        row = _hashed_record(by_id[record_id])
        row["reason"] = "source_family_quarantine" if row["source_family_sha256"] in family_hashes else "evaluation_connected_component"
        exclusions.append(row)

    public_matches = []
    for item in sorted(matches, key=lambda x: (x["train_record_id"], x.get("eval_record_id", x.get("peer_record_id", "")), x["match_type"])):
        public = {k: v for k, v in item.items() if not k.endswith("_record_id")}
        public["train_record_id_sha256"] = sha256_bytes(item["train_record_id"].encode())
        for kind in ("eval", "peer"):
            key = kind + "_record_id"
            if key in item:
                public[kind + "_record_id_sha256"] = sha256_bytes(item[key].encode())
        public_matches.append(public)

    clusters = []
    for eval_id in eval_ids:
        root = find("e:" + eval_id)
        training_members = [x for x in train_ids if find("t:" + x) == root]
        evaluation_members = [x for x in eval_ids if find("e:" + x) == root]
        cluster = {
            "cluster_id_sha256": sha256_bytes(root.encode()),
            "training_record_id_sha256": sorted(sha256_bytes(x.encode()) for x in training_members),
            "evaluation_record_id_sha256": sorted(sha256_bytes(x.encode()) for x in evaluation_members),
        }
        if cluster not in clusters:
            clusters.append(cluster)

    report = {
        "schema": SCHEMA,
        "worker_id": "DATA-232-DECONTAMINATION-AUTHORITY-V2",
        "status": "PASS_WITH_EXCLUSIONS" if exclusions else "PASS_CLEAN",
        "training_corpus_identity": training_corpus_identity,
        "selection_validation_identity": selection_validation_identity,
        "final_test_identity": final_test_identity,
        "all_reserved_authorities_identity": authority_composite_identity(authorities, {x["role"] for x in authorities["authorities"]}),
        "matching": {
            "algorithm": ALGORITHM,
            "normalization": NORMALIZATION,
            "code_skeleton": CODE_SKELETON,
            "thresholds": t,
            "cluster_policy": "exclude every training node in an evaluation-connected component",
            "family_policy": "quarantine contaminated cross-source training family" if quarantine_cross_source_families else "cluster_only",
        },
        "counts": {
            "training_records": len(train),
            "evaluation_records": len(evaluation),
            "excluded_training_records": len(exclusions),
            "quarantined_source_families": len(families),
            "match_evidence_records": len(public_matches),
        },
        "quarantined_source_family_sha256": sorted(family_hashes),
        "excluded_records": exclusions,
        "contaminated_clusters": sorted(clusters, key=lambda x: x["cluster_id_sha256"]),
        "match_evidence": public_matches,
        "evaluation_authorities": [
            {key: row[key] for key in ("authority_id", "identity_sha256", "role", "source_sha")}
            for row in sorted(authorities["authorities"], key=lambda x: x["authority_id"])
        ],
        "hash_only_evidence": True,
        "final_test_outcomes_read": False,
        "model_architecture_or_hyperparameters_selected": False,
        "training_executed": False,
        "local_free_only": True,
    }
    report["report_sha256"] = stable_identity("data232-report-v2", report)
    return report


def build_blocker_report(authorities: Mapping[str, Any], *, reason: str) -> dict[str, Any]:
    validate_authority_metadata(authorities)
    report = {
        "schema": SCHEMA,
        "worker_id": "DATA-232-DECONTAMINATION-AUTHORITY-V2",
        "status": "BLOCKED_MISSING_DATA230",
        "blocking_reason": reason,
        "training_corpus_identity": None,
        "selection_validation_identity": None,
        "final_test_identity": authority_composite_identity(authorities, {"final_test"}),
        "all_reserved_authorities_identity": authority_composite_identity(authorities, {x["role"] for x in authorities["authorities"]}),
        "matching": {"algorithm": ALGORITHM, "normalization": NORMALIZATION, "code_skeleton": CODE_SKELETON, "thresholds": dict(DEFAULT_THRESHOLDS)},
        "excluded_records": [],
        "hash_only_evidence": True,
        "final_test_outcomes_read": False,
        "model_architecture_or_hyperparameters_selected": False,
        "training_executed": False,
        "local_free_only": True,
        "unblock_requires": [
            "terminal DATA-230 corpus identity",
            "exact DATA-230 training record inventory",
            "immutable selection-validation identity from the DATA-230 lineage",
        ],
    }
    report["report_sha256"] = stable_identity("data232-report-v2", report)
    return report


def verify_report(report: Mapping[str, Any]) -> None:
    if report.get("schema") != SCHEMA:
        raise DecontaminationError("report schema mismatch")
    body = dict(report)
    claimed = body.pop("report_sha256", None)
    if claimed != stable_identity("data232-report-v2", body):
        raise DecontaminationError("report SHA-256 mismatch")
    if report.get("hash_only_evidence") is not True or report.get("final_test_outcomes_read") is not False:
        raise DecontaminationError("report safety invariant failed")
    forbidden = {"text", "source_text", "content", "prefix", "continuation", "canary_text"}

    def walk(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                if str(key).lower() in forbidden:
                    raise DecontaminationError(f"raw-text field leaked into report: {key}")
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(report)


def write_immutable_report(path: Path, report: Mapping[str, Any]) -> None:
    verify_report(report)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != payload:
        raise DecontaminationError(f"refusing to overwrite immutable report: {path}")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")
