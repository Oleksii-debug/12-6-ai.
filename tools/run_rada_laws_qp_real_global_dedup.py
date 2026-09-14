#!/usr/bin/env python3
"""Two-clean-run carrier for the real Rada-laws Q/P global-dedup execution.

This tool intentionally contains no matching science. It authenticates already-produced
source/base evidence, loads the exact incumbent matcher closure plus the separately
qualified indexed executor, composes the full source graph, and delegates matching.

The durable outputs remain zero-credit until the downstream decontamination, G05/G06,
balance, split, packing, and unique-loss gates independently pass.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import importlib.util
import json
import re
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

try:
    import resource
except ImportError:  # pragma: no cover - Windows operator path
    resource = None  # type: ignore[assignment]

MANIFEST_SCHEMA = "12-6.d03-rada-laws-qp-real-global-dedup-launch.v1"
BASE_AUTHORITY_SCHEMA = "12-6.d03-clean-base-global-dedup-input-authority.v1"
RUN_SCHEMA = "12-6.d03-rada-laws-qp-real-global-dedup-run.v1"
SURVIVOR_SCHEMA = "12-6.d03-rada-laws-qp-global-dedup-survivors.v1"
TWO_RUN_SCHEMA = "12-6.d03-rada-laws-qp-two-clean-global-dedup.v1"
BLOCKER_SCHEMA = "12-6.d03-rada-laws-qp-real-global-dedup-blocker.v1"
SELECTION_RULE = "largest_declared_capacity_then_lexicographically_smallest_source_id"

EXPECTED_RADA_ADAPTER_HEAD = "d0f3cb8036f36c49e1cdfe69f07458bd56f79717"
EXPECTED_RADA_ADAPTER_BLOB = "441dc06791c838f05cc439f149979b71ae2f48f9"
EXPECTED_MATCHER_HEAD = "d3333ec1b4a508df232a5aefccd6686adda745fb"
EXPECTED_MATCHER_BLOBS = {
    "src/twelve_six/data/cross_source_capacity_audit_v3.py":
        "11490b1803e0aa2266d8ac0053676efcfb0f91ba",
    "src/twelve_six/data/cross_source_capacity_audit.py":
        "84cdf00b2d468d2709a542ac3ee2ea372aae5716",
    "src/twelve_six/data/_data232_decontamination_matching.py":
        "dab5da98dfc43133aa8f3c2e3c78c809252b741b",
}
EXPECTED_RADA_SOURCE_FAMILY = "ua.rada.open-data.laws-texts"
PROHIBITED_NOMIS_SOURCE_ID = "ua.verba.nomis1864.bounded24"
PROHIBITED_NOMIS_RAW_SHA256 = (
    "1eb91dbd631898c6a2efe274b700a5be0deaca243c0a9d5d30994ddadcf43598"
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


class RadaRealDedupError(RuntimeError):
    """Fail-closed real-execution carrier error."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RadaRealDedupError(message)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _git_blob_sha1(raw: bytes) -> str:
    prefix = b"blob " + str(len(raw)).encode("ascii") + b"\0"
    return hashlib.sha1(prefix + raw).hexdigest()


def _self_hash(value: Mapping[str, Any], field: str) -> str:
    core = dict(value)
    core.pop(field, None)
    return _sha256(_canonical_bytes(core))


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8", errors="strict"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RadaRealDedupError(f"cannot read {label}: {path}") from exc
    _require(type(value) is dict, f"{label} root must be an object")
    return value


def _require_sha256(value: Any, label: str) -> str:
    _require(isinstance(value, str) and SHA256_RE.fullmatch(value) is not None, label)
    return value


def _require_git_sha(value: Any, label: str) -> str:
    _require(isinstance(value, str) and GIT_SHA_RE.fullmatch(value) is not None, label)
    return value


def _verify_blob(path: Path, expected: str, label: str) -> None:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise RadaRealDedupError(f"cannot read {label}: {path}") from exc
    _require(_git_blob_sha1(raw) == expected, f"{label} Git blob identity drift")


def _validate_truth_boundary(boundary: Any, *, label: str) -> None:
    _require(isinstance(boundary, Mapping), f"{label} truth boundary missing")
    _require(boundary.get("authorized_optimized_target_exposure") == 0, f"{label} exposure")
    _require(boundary.get("tokenizer_fit_authorized") is False, f"{label} tokenizer")
    _require(boundary.get("optimizer_updates_executed_on_real_targets") == 0, f"{label} optimizer")
    _require(boundary.get("training_executed") is False, f"{label} training")
    _require(boundary.get("learned_weights_created") is False, f"{label} weights")
    _require(boundary.get("final_test_outcomes_read") is False, f"{label} final-test")
    _require(boundary.get("paid_compute_used") is False, f"{label} paid-compute")
    _require(boundary.get("foreign_pretrained_weights") is False, f"{label} foreign weights")


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    _require(manifest.get("schema_version") == MANIFEST_SCHEMA, "launch manifest schema drift")
    _require(manifest.get("local_free_only") is True, "launch must be LOCAL_FREE")
    _require(manifest.get("execution_authorized") is True, "real execution is not authorized")
    _require(manifest.get("execution_authorized_by") == "#2019", "execution authority drift")
    _require_git_sha(manifest.get("main_sha_at_launch"), "launch main SHA malformed")
    _require(manifest.get("rada_adapter_terminal") is True, "Rada adapter is not terminal")
    rada_audit = manifest.get("rada_adapter_terminal_audit_ref")
    _require(isinstance(rada_audit, str) and rada_audit, "Rada terminal audit reference missing")
    rada_integration = manifest.get("rada_adapter_integration_ref")
    _require(
        isinstance(rada_integration, str) and rada_integration,
        "Rada adapter integration reference missing",
    )
    _require(manifest.get("indexed_executor_terminal") is True, "indexed executor is not terminal")
    _require(manifest.get("clean_base_terminal") is True, "clean base is not terminal")
    clean_audit = manifest.get("clean_base_terminal_audit_ref")
    _require(isinstance(clean_audit, str) and clean_audit, "clean-base terminal audit missing")
    _require(manifest.get("rada_adapter_head") == EXPECTED_RADA_ADAPTER_HEAD, "Rada head drift")
    _require(
        manifest.get("rada_adapter_git_blob_sha1") == EXPECTED_RADA_ADAPTER_BLOB,
        "Rada adapter blob drift",
    )
    _require(
        manifest.get("matcher_head") == EXPECTED_MATCHER_HEAD,
        "historical matcher head drift",
    )
    _require(
        manifest.get("matcher_git_blobs") == EXPECTED_MATCHER_BLOBS,
        "historical matcher blob set drift",
    )
    _require(manifest.get("indexed_executor_pr") == 1459, "indexed executor PR binding drift")
    _require_git_sha(manifest.get("indexed_executor_head"), "indexed executor head malformed")
    _require_git_sha(
        manifest.get("indexed_executor_git_blob_sha1"),
        "indexed executor blob identity malformed",
    )
    audit_ref = manifest.get("indexed_executor_terminal_audit_ref")
    _require(isinstance(audit_ref, str) and audit_ref, "indexed executor terminal audit missing")
    _require_git_sha(manifest.get("carrier_head"), "carrier head malformed")
    _require_git_sha(
        manifest.get("carrier_git_blob_sha1"),
        "carrier blob identity malformed",
    )
    _verify_blob(Path(__file__), manifest["carrier_git_blob_sha1"], "execution carrier")
    _require_sha256(
        manifest.get("clean_base_authority_sha256"),
        "clean-base authority identity malformed",
    )
    limits = manifest.get("work_limits")
    _require(isinstance(limits, Mapping), "work limits missing")
    for key in ("max_candidate_pairs", "max_index_postings", "max_pair_expansions"):
        value = limits.get(key)
        _require(type(value) is int and value > 0, f"invalid work limit: {key}")
    _validate_truth_boundary(manifest.get("truth_boundary"), label="launch")
    _require(
        manifest.get("manifest_identity_sha256")
        == _self_hash(manifest, "manifest_identity_sha256"),
        "launch manifest self-hash mismatch",
    )


def _payload_inventory(
    inventory: Mapping[str, Any],
    payload_map: Mapping[str, Any],
    *,
    map_root: Path,
) -> tuple[dict[str, bytes], list[dict[str, Any]]]:
    rows = inventory.get("sources")
    _require(isinstance(rows, list) and rows, "base inventory sources missing")
    source_ids: list[str] = []
    for row in rows:
        _require(isinstance(row, Mapping), "base source row must be object")
        source_id = row.get("source_id")
        _require(isinstance(source_id, str) and source_id, "base source_id malformed")
        _require(source_id not in source_ids, "duplicate base source_id")
        source_ids.append(source_id)
    _require(set(payload_map) == set(source_ids), "base payload-map key set drift")

    payloads: dict[str, bytes] = {}
    projection: list[dict[str, Any]] = []
    for source_id in sorted(source_ids):
        rel = payload_map[source_id]
        _require(isinstance(rel, str) and rel, f"payload path malformed: {source_id}")
        path = Path(rel)
        if not path.is_absolute():
            path = map_root / path
        _require(path.is_file() and not path.is_symlink(), f"payload file invalid: {source_id}")
        raw = path.read_bytes()
        payloads[source_id] = raw
        projection.append(
            {"source_id": source_id, "bytes": len(raw), "sha256": _sha256(raw)}
        )
    return payloads, projection


def validate_clean_base(
    *,
    inventory_path: Path,
    payload_map_path: Path,
    authority_path: Path,
    expected_authority_sha256: str,
) -> tuple[dict[str, Any], dict[str, bytes], dict[str, Any]]:
    authority = _read_json(authority_path, "clean-base authority")
    _require(authority.get("schema_version") == BASE_AUTHORITY_SCHEMA, "base authority schema")
    _require(authority.get("authority_status") == "TERMINAL", "base authority nonterminal")
    upstream_ref = authority.get("upstream_clean_successor_ref")
    _require(isinstance(upstream_ref, str) and upstream_ref, "clean-successor reference missing")
    audit_ref = authority.get("terminal_audit_ref")
    _require(
        isinstance(audit_ref, str) and audit_ref,
        "clean-base terminal audit reference missing",
    )
    _require(
        authority.get("base_kind") == "TERMINAL_CLEAN_POSTDEDUP_SURVIVOR_GRAPH",
        "clean-base authority is not a terminal survivor graph",
    )
    _require_sha256(
        authority.get("survivor_authority_sha256"),
        "clean-base survivor authority identity malformed",
    )
    _require(authority.get("local_free_only") is True, "base authority not LOCAL_FREE")
    _require(
        authority.get("external_llm_or_api_used_for_data_or_intelligence") is False,
        "clean base still carries external-LLM provenance",
    )
    _require(
        authority.get("blocked_nomis_source_id") == PROHIBITED_NOMIS_SOURCE_ID,
        "Nomis blocked source binding drift",
    )
    _require(
        authority.get("blocked_nomis_raw_sha256") == PROHIBITED_NOMIS_RAW_SHA256,
        "Nomis blocked payload binding drift",
    )
    _require(authority.get("blocked_nomis_present") is False, "Nomis still present in clean base")
    _validate_truth_boundary(authority.get("truth_boundary"), label="clean-base")
    observed_authority = _self_hash(authority, "authority_identity_sha256")
    _require(
        authority.get("authority_identity_sha256") == observed_authority,
        "clean-base authority self-hash mismatch",
    )
    _require(
        observed_authority == expected_authority_sha256,
        "clean-base authority does not match launch manifest",
    )

    inventory_raw = inventory_path.read_bytes()
    inventory = _read_json(inventory_path, "clean-base inventory")
    _require(
        _sha256(inventory_raw) == authority.get("inventory_file_sha256"),
        "clean-base inventory file identity drift",
    )
    _require(
        inventory.get("schema_version") == "12-6.next100-065-cross-source-dedup.v3",
        "clean-base inventory schema drift",
    )
    _require(inventory.get("local_free_only") is True, "clean-base inventory not LOCAL_FREE")
    _require(inventory.get("model_training_executed") is False, "base inventory training drift")

    payload_map = _read_json(payload_map_path, "clean-base payload map")
    payloads, projection = _payload_inventory(
        inventory, payload_map, map_root=payload_map_path.parent
    )
    _require(
        len(projection) == authority.get("source_object_count"),
        "clean-base source-object count drift",
    )
    _require(
        sum(item["bytes"] for item in projection) == authority.get("payload_bytes"),
        "clean-base payload byte total drift",
    )
    _require(
        _sha256(_canonical_bytes(projection)) == authority.get("payload_inventory_sha256"),
        "clean-base payload inventory root drift",
    )
    _require(PROHIBITED_NOMIS_SOURCE_ID not in payloads, "blocked Nomis source present")
    for row in inventory["sources"]:
        _require(
            row.get("source_family") != "ua.verba.public-domain.nomis1864",
            "blocked Nomis family present",
        )
        _require(
            row.get("expected_raw_sha256") != PROHIBITED_NOMIS_RAW_SHA256,
            "blocked Nomis payload present under alias",
        )
    return inventory, payloads, authority


def _load_file_module(name: str, path: Path, expected_blob: str) -> ModuleType:
    _verify_blob(path, expected_blob, name)
    spec = importlib.util.spec_from_file_location(name, path)
    _require(spec is not None and spec.loader is not None, f"cannot load module: {name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(name, None)
        raise RadaRealDedupError(f"cannot execute module: {name}: {exc}") from exc
    return module


def _load_historical_matcher(root: Path) -> ModuleType:
    for rel, expected in EXPECTED_MATCHER_BLOBS.items():
        _verify_blob(root / rel, expected, rel)
    source_root = root / "src"
    _require(source_root.is_dir(), "historical matcher src root missing")

    # This standalone carrier imports no twelve_six module before this point.
    # Refuse a polluted process rather than silently mixing current-tree modules.
    polluted = [
        name
        for name in sys.modules
        if name == "twelve_six" or name.startswith("twelve_six.")
    ]
    _require(not polluted, f"matcher import namespace already populated: {polluted[:3]}")
    sys.path.insert(0, str(source_root.resolve()))
    try:
        module = importlib.import_module("twelve_six.data.cross_source_capacity_audit_v3")
    except Exception as exc:
        raise RadaRealDedupError(f"cannot import historical V3 matcher: {exc}") from exc
    return module


def _family_counts(report: Mapping[str, Any]) -> dict[str, int]:
    rows = report.get("sources")
    _require(isinstance(rows, list), "dedup report sources missing")
    result: dict[str, int] = {}
    for modality in ("uk", "en", "code"):
        result[modality] = len(
            {
                row["source_family"]
                for row in rows
                if isinstance(row, Mapping) and row.get("modality") == modality
            }
        )
    return result


def derive_survivor_authority(report: Mapping[str, Any]) -> dict[str, Any]:
    report_sha = _require_sha256(report.get("report_sha256"), "dedup report identity malformed")
    rows = report.get("sources")
    _require(isinstance(rows, list) and rows, "dedup report source rows missing")
    by_id: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        _require(isinstance(row, Mapping), "dedup source row malformed")
        source_id = row.get("source_id")
        capacity = row.get("declared_capacity_bytes")
        _require(isinstance(source_id, str) and source_id and source_id not in by_id, "source id")
        _require(type(capacity) is int and capacity > 0, f"capacity malformed: {source_id}")
        by_id[source_id] = row

    terminal = report.get("terminal_candidates")
    _require(isinstance(terminal, Mapping), "terminal candidate summary missing")
    clusters = terminal.get("duplicate_clusters")
    _require(isinstance(clusters, list), "duplicate clusters missing")
    used: set[str] = set()
    dropped: set[str] = set()
    normalized_clusters: list[dict[str, Any]] = []
    for raw_cluster in clusters:
        _require(
            isinstance(raw_cluster, Sequence) and not isinstance(raw_cluster, (str, bytes)),
            "duplicate cluster malformed",
        )
        cluster = sorted(str(value) for value in raw_cluster)
        _require(len(cluster) >= 2 and len(cluster) == len(set(cluster)), "cluster uniqueness")
        _require(all(source_id in by_id for source_id in cluster), "cluster unknown source")
        _require(not (used & set(cluster)), "duplicate clusters overlap")
        used.update(cluster)
        maximum = max(int(by_id[source_id]["declared_capacity_bytes"]) for source_id in cluster)
        survivor = min(
            source_id
            for source_id in cluster
            if int(by_id[source_id]["declared_capacity_bytes"]) == maximum
        )
        dropped.update(source_id for source_id in cluster if source_id != survivor)
        normalized_clusters.append(
            {
                "member_source_ids": cluster,
                "selected_source_id": survivor,
                "selected_declared_capacity_bytes": maximum,
            }
        )
    normalized_clusters.sort(key=lambda item: tuple(item["member_source_ids"]))

    survivor_ids = sorted(set(by_id) - dropped)
    survivor_rows: list[dict[str, Any]] = []
    for source_id in survivor_ids:
        row = by_id[source_id]
        survivor_rows.append(
            {
                "source_id": source_id,
                "source_family": row["source_family"],
                "modality": row["modality"],
                "declared_capacity_bytes": row["declared_capacity_bytes"],
                "verified_raw_sha256": row["verified_raw_sha256"],
                "normalized_sha256": row["normalized_sha256"],
                "stable_origin_id_sha256": row["stable_origin_id_sha256"],
                "stable_object_id_sha256": row["stable_object_id_sha256"],
            }
        )
    survivor_bytes = sum(int(row["declared_capacity_bytes"]) for row in survivor_rows)
    before = terminal.get("declared_capacity_bytes_before")
    after = terminal.get("conservative_unique_capacity_bytes_after")
    discount = terminal.get("duplicate_discount_bytes")
    _require(
        type(before) is int and type(after) is int and type(discount) is int,
        "capacity summary",
    )
    _require(survivor_bytes == after, "survivor bytes do not reproduce V3 capacity")
    _require(before - survivor_bytes == discount, "survivor discount mismatch")
    _require(len(normalized_clusters) == terminal.get("duplicate_cluster_count"), "cluster count")

    core: dict[str, Any] = {
        "schema_version": SURVIVOR_SCHEMA,
        "selection_rule": SELECTION_RULE,
        "dedup_report_sha256": report_sha,
        "pre_dedup_source_object_count": len(rows),
        "post_dedup_survivor_source_object_count": len(survivor_rows),
        "pre_dedup_declared_capacity_bytes": before,
        "post_dedup_declared_capacity_bytes": survivor_bytes,
        "duplicate_discount_bytes": discount,
        "duplicate_cluster_count": len(normalized_clusters),
        "duplicate_clusters": normalized_clusters,
        "survivors": survivor_rows,
        "by_modality": {
            modality: {
                "source_object_count": sum(
                    1 for row in survivor_rows if row["modality"] == modality
                ),
                "declared_capacity_bytes": sum(
                    int(row["declared_capacity_bytes"])
                    for row in survivor_rows
                    if row["modality"] == modality
                ),
            }
            for modality in ("uk", "en", "code")
        },
        "truth_boundary": {
            "source_object_authority_only": True,
            "authorized_optimized_target_exposure": 0,
            "tokenizer_fit_authorized": False,
            "optimizer_updates_executed_on_real_targets": 0,
            "training_executed": False,
            "learned_weights_created": False,
            "final_test_outcomes_read": False,
            "paid_compute_used": False,
            "foreign_pretrained_weights": False,
        },
    }
    core["survivor_authority_sha256"] = _sha256(_canonical_bytes(core))
    return core


def _compose_inventory(
    base: Mapping[str, Any],
    rada_sources: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    combined = copy.deepcopy(dict(base))
    base_rows = combined.get("sources")
    _require(isinstance(base_rows, list), "base source rows missing")
    existing = {row.get("source_id") for row in base_rows if isinstance(row, Mapping)}
    projected = [copy.deepcopy(dict(row)) for row in rada_sources]
    projected_ids = {row.get("source_id") for row in projected}
    _require(len(projected_ids) == len(projected), "Rada projection source IDs not unique")
    _require(not (existing & projected_ids), "Rada source IDs collide with clean base")
    _require(
        all(row.get("source_family") == EXPECTED_RADA_SOURCE_FAMILY for row in projected),
        "Rada projection family drift",
    )
    combined["sources"] = [*base_rows, *projected]
    combined["final_refresh_required"] = False
    combined["terminal_refresh_rule"] = (
        "Terminal clean-base source graph plus exact PR1787 Rada-laws Q/P projection; "
        "all outputs remain zero-credit pending downstream corpus gates."
    )
    return combined


def _runtime_metrics(started: float) -> dict[str, Any]:
    metrics: dict[str, Any] = {"elapsed_seconds": round(time.perf_counter() - started, 6)}
    if resource is not None:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        metrics["max_rss_kib"] = int(usage.ru_maxrss)
    else:
        metrics["max_rss_kib"] = None
    return metrics


def execute_once(
    *,
    manifest: Mapping[str, Any],
    base_inventory: Mapping[str, Any],
    base_payloads: Mapping[str, bytes],
    base_authority: Mapping[str, Any],
    candidate_jsonl: Path,
    quality_report: Path,
    execution_evidence: Path,
    rada_adapter_path: Path,
    indexed_executor_path: Path,
    matcher_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    validate_manifest(manifest)
    started = time.perf_counter()
    rada = _load_file_module(
        "_rada_laws_qp_adapter_exact",
        rada_adapter_path,
        EXPECTED_RADA_ADAPTER_BLOB,
    )
    projection = rada.validate_and_project_rada_laws_qp(
        candidate_jsonl,
        quality_report,
        execution_evidence,
        upstream_head=rada.UPSTREAM_FINAL_EVIDENCE_COMMIT,
        retain_payloads=True,
    )
    _require(projection.sources is not None and projection.payloads is not None, "Rada projection")
    _require(len(projection.sources) == rada.ACCEPTED_CHUNK_COUNT, "Rada projected count drift")
    _require(
        sum(len(raw) for raw in projection.payloads.values()) == rada.ACCEPTED_TEXT_UTF8_BYTES,
        "Rada projected byte total drift",
    )

    v3 = _load_historical_matcher(matcher_root)
    indexed = _load_file_module(
        "_incumbent_dedup_indexed_execution_exact",
        indexed_executor_path,
        str(manifest["indexed_executor_git_blob_sha1"]),
    )
    indexed.attest_incumbent_runtime(v3)

    inventory = _compose_inventory(base_inventory, projection.sources)
    payloads = dict(base_payloads)
    _require(not (set(payloads) & set(projection.payloads)), "Rada payload IDs collide with base")
    payloads.update(projection.payloads)
    limits = manifest["work_limits"]
    report = indexed.audit_payloads_indexed(
        v3,
        inventory,
        payloads,
        max_candidate_pairs=int(limits["max_candidate_pairs"]),
        max_index_postings=int(limits["max_index_postings"]),
        max_pair_expansions=int(limits["max_pair_expansions"]),
    )
    v3.verify_report(report)
    survivor = derive_survivor_authority(report)

    core: dict[str, Any] = {
        "schema_version": RUN_SCHEMA,
        "execution_profile": "LOCAL_FREE",
        "launch_manifest_identity_sha256": manifest["manifest_identity_sha256"],
        "clean_base_authority_sha256": base_authority["authority_identity_sha256"],
        "clean_base_survivor_authority_sha256": base_authority["survivor_authority_sha256"],
        "clean_base_terminal_audit_ref": manifest["clean_base_terminal_audit_ref"],
        "rada_adapter_head": EXPECTED_RADA_ADAPTER_HEAD,
        "rada_adapter_terminal_audit_ref": manifest["rada_adapter_terminal_audit_ref"],
        "rada_adapter_integration_ref": manifest["rada_adapter_integration_ref"],
        "rada_adapter_git_blob_sha1": EXPECTED_RADA_ADAPTER_BLOB,
        "indexed_executor_head": manifest["indexed_executor_head"],
        "indexed_executor_git_blob_sha1": manifest["indexed_executor_git_blob_sha1"],
        "indexed_executor_terminal_audit_ref": manifest["indexed_executor_terminal_audit_ref"],
        "carrier_head": manifest["carrier_head"],
        "carrier_git_blob_sha1": manifest["carrier_git_blob_sha1"],
        "matcher_head": EXPECTED_MATCHER_HEAD,
        "matcher_git_blobs": EXPECTED_MATCHER_BLOBS,
        "rada_intake_receipt_sha256": projection.receipt["receipt_identity_sha256"],
        "dedup_report_sha256": report["report_sha256"],
        "survivor_authority_sha256": survivor["survivor_authority_sha256"],
        "source_count": report["source_count"],
        "source_family_counts": _family_counts(report),
        "incumbent_all_pair_dispatches": (
            int(report["source_count"]) * (int(report["source_count"]) - 1) // 2
        ),
        "total_match_records": sum(int(value) for value in report["match_counts"].values()),
        "match_counts": copy.deepcopy(report["match_counts"]),
        "capacity_collapsing_match_counts": copy.deepcopy(
            report["capacity_collapsing_match_counts"]
        ),
        "work_limits": copy.deepcopy(limits),
        "pre_dedup_declared_capacity_bytes":
            report["terminal_candidates"]["declared_capacity_bytes_before"],
        "post_dedup_declared_capacity_bytes":
            survivor["post_dedup_declared_capacity_bytes"],
        "duplicate_discount_bytes": survivor["duplicate_discount_bytes"],
        "duplicate_cluster_count": survivor["duplicate_cluster_count"],
        "raw_text_emitted": False,
        "truth_boundary": copy.deepcopy(manifest["truth_boundary"]),
    }
    core["deterministic_identity_sha256"] = _sha256(_canonical_bytes(core))
    runtime = _runtime_metrics(started)
    elapsed = float(runtime["elapsed_seconds"])
    if elapsed > 0.0:
        runtime["source_objects_per_second"] = round(int(report["source_count"]) / elapsed, 6)
        runtime["payload_mib_per_second"] = round(
            (int(core["pre_dedup_declared_capacity_bytes"]) / (1024 * 1024)) / elapsed,
            6,
        )
    core["runtime"] = runtime
    return report, survivor, core


def build_blocker_receipt(
    *,
    manifest: Mapping[str, Any],
    base_authority: Mapping[str, Any],
    started: float,
    failure: Exception,
    run_label: str,
) -> dict[str, Any]:
    message = str(failure).replace("\n", " ")[:2000]
    lowered = message.lower()
    resource_signal = (
        isinstance(failure, MemoryError)
        or "budget exceeded" in lowered
        or "out of memory" in lowered
        or "memoryerror" in lowered
    )
    core: dict[str, Any] = {
        "schema_version": BLOCKER_SCHEMA,
        "execution_profile": "LOCAL_FREE",
        "run_label": run_label,
        "launch_manifest_identity_sha256": manifest["manifest_identity_sha256"],
        "clean_base_authority_sha256": base_authority["authority_identity_sha256"],
        "clean_base_survivor_authority_sha256": base_authority[
            "survivor_authority_sha256"
        ],
        "rada_adapter_head": EXPECTED_RADA_ADAPTER_HEAD,
        "rada_adapter_git_blob_sha1": EXPECTED_RADA_ADAPTER_BLOB,
        "indexed_executor_head": manifest["indexed_executor_head"],
        "indexed_executor_git_blob_sha1": manifest["indexed_executor_git_blob_sha1"],
        "matcher_head": EXPECTED_MATCHER_HEAD,
        "matcher_git_blobs": EXPECTED_MATCHER_BLOBS,
        "work_limits": copy.deepcopy(manifest["work_limits"]),
        "failure_type": type(failure).__name__,
        "failure_message": message,
        "resource_or_work_budget_signal": resource_signal,
        "candidate_sampling_used": False,
        "partial_capacity_credit": 0,
        "partition_safe_continuation": {
            "sampling_as_terminal_proof_allowed": False,
            "partial_survivor_authority_allowed": False,
            "partitioning_currently_authorized": False,
            "required_semantics": (
                "complete-graph equivalence to the exact incumbent matcher; any future "
                "partitioning must cover and reconcile the complete candidate relation"
            ),
            "next_action": (
                "repair/checkpoint the same executor lineage or revise an explicit bounded "
                "work limit from measured evidence; never substitute sampling"
            ),
        },
        "runtime": _runtime_metrics(started),
        "raw_text_emitted": False,
        "truth_boundary": copy.deepcopy(manifest["truth_boundary"]),
    }
    core["blocker_identity_sha256"] = _sha256(_canonical_bytes(core))
    return core


def build_subprocess_blocker(
    *,
    manifest: Mapping[str, Any],
    started: float,
    returncode: int,
    stderr: str,
    run_label: str,
) -> dict[str, Any]:
    message = stderr.replace("\n", " ")[-2000:]
    core: dict[str, Any] = {
        "schema_version": BLOCKER_SCHEMA,
        "execution_profile": "LOCAL_FREE",
        "run_label": run_label,
        "launch_manifest_identity_sha256": manifest["manifest_identity_sha256"],
        "clean_base_authority_sha256": manifest["clean_base_authority_sha256"],
        "rada_adapter_head": EXPECTED_RADA_ADAPTER_HEAD,
        "indexed_executor_head": manifest["indexed_executor_head"],
        "matcher_head": EXPECTED_MATCHER_HEAD,
        "work_limits": copy.deepcopy(manifest["work_limits"]),
        "failure_type": "SUBPROCESS_TERMINATED",
        "failure_returncode": returncode,
        "failure_stderr_tail": message,
        "candidate_sampling_used": False,
        "partial_capacity_credit": 0,
        "partition_safe_continuation": {
            "sampling_as_terminal_proof_allowed": False,
            "partial_survivor_authority_allowed": False,
            "partitioning_currently_authorized": False,
            "required_semantics": (
                "complete-graph equivalence to the exact incumbent matcher; any future "
                "partitioning must cover and reconcile the complete candidate relation"
            ),
        },
        "runtime": {
            "elapsed_seconds": round(time.perf_counter() - started, 6),
            "child_max_rss_kib": None,
        },
        "raw_text_emitted": False,
        "truth_boundary": copy.deepcopy(manifest["truth_boundary"]),
    }
    core["blocker_identity_sha256"] = _sha256(_canonical_bytes(core))
    return core


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_bytes(value) + b"\n")


def compare_runs(run_a: Mapping[str, Any], run_b: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "launch_manifest_identity_sha256",
        "clean_base_authority_sha256",
        "clean_base_survivor_authority_sha256",
        "rada_intake_receipt_sha256",
        "dedup_report_sha256",
        "survivor_authority_sha256",
        "source_count",
        "source_family_counts",
        "incumbent_all_pair_dispatches",
        "total_match_records",
        "match_counts",
        "capacity_collapsing_match_counts",
        "work_limits",
        "pre_dedup_declared_capacity_bytes",
        "post_dedup_declared_capacity_bytes",
        "duplicate_discount_bytes",
        "duplicate_cluster_count",
        "deterministic_identity_sha256",
        "truth_boundary",
    )
    for key in keys:
        _require(run_a.get(key) == run_b.get(key), f"two-clean-run drift: {key}")
    core: dict[str, Any] = {
        "schema_version": TWO_RUN_SCHEMA,
        "clean_base_authority_sha256": run_a["clean_base_authority_sha256"],
        "clean_base_survivor_authority_sha256": run_a[
            "clean_base_survivor_authority_sha256"
        ],
        "deterministic_identity_sha256": run_a["deterministic_identity_sha256"],
        "dedup_report_sha256": run_a["dedup_report_sha256"],
        "survivor_authority_sha256": run_a["survivor_authority_sha256"],
        "incumbent_all_pair_dispatches": run_a["incumbent_all_pair_dispatches"],
        "total_match_records": run_a["total_match_records"],
        "match_counts": copy.deepcopy(run_a["match_counts"]),
        "capacity_collapsing_match_counts": copy.deepcopy(
            run_a["capacity_collapsing_match_counts"]
        ),
        "post_dedup_declared_capacity_bytes": run_a["post_dedup_declared_capacity_bytes"],
        "run_a_runtime": copy.deepcopy(run_a.get("runtime")),
        "run_b_runtime": copy.deepcopy(run_b.get("runtime")),
        "fresh_processes_required": True,
        "raw_text_emitted": False,
        "truth_boundary": copy.deepcopy(run_a["truth_boundary"]),
    }
    core["two_run_identity_sha256"] = _sha256(_canonical_bytes(core))
    return core


def _once_command(args: argparse.Namespace) -> int:
    manifest = _read_json(args.manifest, "launch manifest")
    validate_manifest(manifest)
    base_inventory, base_payloads, base_authority = validate_clean_base(
        inventory_path=args.base_inventory,
        payload_map_path=args.base_payload_map,
        authority_path=args.base_authority,
        expected_authority_sha256=manifest["clean_base_authority_sha256"],
    )
    started = time.perf_counter()
    try:
        report, survivor, run = execute_once(
            manifest=manifest,
            base_inventory=base_inventory,
            base_payloads=base_payloads,
            base_authority=base_authority,
            candidate_jsonl=args.rada_candidate,
            quality_report=args.rada_quality_report,
            execution_evidence=args.rada_execution_evidence,
            rada_adapter_path=args.rada_adapter_root
            / "src/twelve_six/data/rada_laws_qp_dedup_adapter.py",
            indexed_executor_path=args.indexed_executor_root
            / "src/twelve_six/data/incumbent_dedup_indexed_execution.py",
            matcher_root=args.matcher_root,
        )
    except Exception as exc:
        blocker = build_blocker_receipt(
            manifest=manifest,
            base_authority=base_authority,
            started=started,
            failure=exc,
            run_label="single",
        )
        _write_json(args.output_dir / "blocker.json", blocker)
        print(f"blocker_identity_sha256={blocker['blocker_identity_sha256']}")
        raise
    _write_json(args.output_dir / "dedup-report.json", report)
    _write_json(args.output_dir / "survivor-authority.json", survivor)
    _write_json(args.output_dir / "run-receipt.json", run)
    print(f"dedup_report_sha256={run['dedup_report_sha256']}")
    print(f"survivor_authority_sha256={run['survivor_authority_sha256']}")
    print(f"total_match_records={run['total_match_records']}")
    return 0


def _twice_command(args: argparse.Namespace) -> int:
    manifest = _read_json(args.manifest, "launch manifest")
    validate_manifest(manifest)
    _require(not args.output_dir.exists(), "two-run output directory already exists")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix="rada-dedup-run-a-") as first:
        with tempfile.TemporaryDirectory(prefix="rada-dedup-run-b-") as second:
            runs = []
            for label, output in (("a", Path(first)), ("b", Path(second))):
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "run-once",
                    "--manifest",
                    str(args.manifest.resolve()),
                    "--base-inventory",
                    str(args.base_inventory.resolve()),
                    "--base-payload-map",
                    str(args.base_payload_map.resolve()),
                    "--base-authority",
                    str(args.base_authority.resolve()),
                    "--rada-candidate",
                    str(args.rada_candidate.resolve()),
                    "--rada-quality-report",
                    str(args.rada_quality_report.resolve()),
                    "--rada-execution-evidence",
                    str(args.rada_execution_evidence.resolve()),
                    "--rada-adapter-root",
                    str(args.rada_adapter_root.resolve()),
                    "--indexed-executor-root",
                    str(args.indexed_executor_root.resolve()),
                    "--matcher-root",
                    str(args.matcher_root.resolve()),
                    "--output-dir",
                    str(output),
                ]
                run_started = time.perf_counter()
                completed = subprocess.run(command, check=False, text=True, capture_output=True)
                if completed.returncode != 0:
                    inner_blocker = output / "blocker.json"
                    if inner_blocker.is_file():
                        target = args.output_dir / f"run-{label}-blocker.json"
                        target.write_bytes(inner_blocker.read_bytes())
                    else:
                        outer_blocker = build_subprocess_blocker(
                            manifest=manifest,
                            started=run_started,
                            returncode=completed.returncode,
                            stderr=completed.stderr,
                            run_label=label,
                        )
                        _write_json(
                            args.output_dir / f"run-{label}-subprocess-blocker.json",
                            outer_blocker,
                        )
                    raise RadaRealDedupError(
                        f"clean run {label} failed rc={completed.returncode}: "
                        f"{completed.stderr[-4000:]}"
                    )
                run = _read_json(output / "run-receipt.json", f"clean run {label} receipt")
                runs.append(run)
                # Only text-free outputs are copied into the durable two-run directory.
                for name in ("dedup-report.json", "survivor-authority.json", "run-receipt.json"):
                    target = args.output_dir / f"run-{label}-{name}"
                    target.write_bytes((output / name).read_bytes())
            proof = compare_runs(runs[0], runs[1])
            _write_json(args.output_dir / "two-clean-run-proof.json", proof)
    print(f"two_run_identity_sha256={proof['two_run_identity_sha256']}")
    print(f"survivor_authority_sha256={proof['survivor_authority_sha256']}")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check-manifest")
    check.add_argument("--manifest", type=Path, required=True)

    def common(target: argparse.ArgumentParser) -> None:
        target.add_argument("--manifest", type=Path, required=True)
        target.add_argument("--base-inventory", type=Path, required=True)
        target.add_argument("--base-payload-map", type=Path, required=True)
        target.add_argument("--base-authority", type=Path, required=True)
        target.add_argument("--rada-candidate", type=Path, required=True)
        target.add_argument("--rada-quality-report", type=Path, required=True)
        target.add_argument("--rada-execution-evidence", type=Path, required=True)
        target.add_argument("--rada-adapter-root", type=Path, required=True)
        target.add_argument("--indexed-executor-root", type=Path, required=True)
        target.add_argument("--matcher-root", type=Path, required=True)
        target.add_argument("--output-dir", type=Path, required=True)

    common(sub.add_parser("run-once"))
    common(sub.add_parser("run-twice"))
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "check-manifest":
        manifest = _read_json(args.manifest, "launch manifest")
        validate_manifest(manifest)
        print("PASS_RADA_REAL_DEDUP_LAUNCH_MANIFEST")
        return 0
    if args.command == "run-once":
        return _once_command(args)
    return _twice_command(args)


if __name__ == "__main__":
    raise SystemExit(main())
