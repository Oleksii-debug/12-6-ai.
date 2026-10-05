#!/usr/bin/env python3
"""Execute exact authenticated Ubuntu IRC candidate through incumbent global dedup.

Execution glue only. Reuses the independently qualified indexed executor and
historical V8 reconstruction. Durable outputs are text-free and grant zero
corpus or training authority.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for location in (str(ROOT / "tools"), str(ROOT / "src")):
    if location not in sys.path:
        sys.path.insert(0, location)

import run_d03_franko1901_global_dedup_execution_v1 as incumbent
import run_next100_065f_global_dedup_v8 as v8
from twelve_six.data import expanded_global_dedup_v9 as v9_semantics
from twelve_six.data import incumbent_dedup_indexed_execution as indexed
from twelve_six.data import ubuntu_irc_v9_intake as ubuntu

SCHEMA = "12-6.d03-ubuntu-irc-global-dedup-execution.v1"
SURVIVOR_SCHEMA = "12-6.d03-ubuntu-irc-global-dedup-survivors.v1"
PARENT_CONVERGENCE_HEAD = "e0bbcbce07a480651be0d16f3be8c1782fc699ca"
CARRIER_PATH = "tools/run_d03_ubuntu_irc_global_dedup_execution_v1.py"
EXPECTED_BASE_OBJECTS = incumbent.EXPECTED_BASE_OBJECTS
EXPECTED_BASE_BYTES = incumbent.EXPECTED_BASE_BYTES
EXPECTED_UBUNTU_OBJECTS = ubuntu.CANDIDATE_RECORDS
EXPECTED_UBUNTU_BYTES = ubuntu.CANDIDATE_NORMALIZED_BYTES
EXPECTED_COMBINED_OBJECTS = EXPECTED_BASE_OBJECTS + EXPECTED_UBUNTU_OBJECTS
EXPECTED_COMBINED_BYTES = EXPECTED_BASE_BYTES + EXPECTED_UBUNTU_BYTES

AUTHORITY_PATHS = (
    "src/twelve_six/data/ubuntu_irc_v9_intake.py",
    "tools/materialize_d03_common_pile_ubuntu_irc_v1.py",
    "configs/data/d03_common_pile_ubuntu_irc_bounded_v1.json",
    "configs/data/d03_ubuntu_irc_repaired_execution_rights_crossbind_v1.json",
    "configs/data/d03_common_pile_ubuntu_irc_source_rights_v1.json",
    "configs/data/common_pile_source_rights_v1.json",
    "evidence/d03_common_pile_ubuntu_irc_real_execution_v1.json",
    "src/twelve_six/data/expanded_global_dedup_v9.py",
    "src/twelve_six/data/_expanded_global_dedup_v9_impl.py",
    "src/twelve_six/data/incumbent_dedup_indexed_execution.py",
    "src/twelve_six/data/_incumbent_dedup_indexed_execution_core.py",
    "tools/run_d03_franko1901_global_dedup_execution_v1.py",
    "tools/run_next100_065f_global_dedup_v8.py",
)


class UbuntuGlobalDedupError(RuntimeError):
    """Fail-closed physical execution or authority mismatch."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise UbuntuGlobalDedupError(message)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _verify_authority_paths() -> dict[str, str]:
    ancestor = incumbent._git(
        "merge-base", "--is-ancestor", PARENT_CONVERGENCE_HEAD, "HEAD", check=False
    )
    _require(ancestor.returncode == 0, "current-main Ubuntu parent is not an ancestor")
    blobs: dict[str, str] = {}
    for path in AUTHORITY_PATHS:
        parent_blob = incumbent._git(
            "rev-parse", f"{PARENT_CONVERGENCE_HEAD}:{path}"
        ).stdout.strip()
        head_blob = incumbent._git("rev-parse", f"HEAD:{path}").stdout.strip()
        worktree_blob = incumbent._git("hash-object", str(ROOT / path)).stdout.strip()
        _require(
            bool(parent_blob) and head_blob == parent_blob and worktree_blob == head_blob,
            f"authority path drift: {path}",
        )
        blobs[path] = head_blob
    incumbent.verify_repository_authority()
    return blobs


def _verify_carrier() -> str:
    expected = incumbent._git("rev-parse", f"HEAD:{CARRIER_PATH}").stdout.strip()
    observed = incumbent._git("hash-object", str(ROOT / CARRIER_PATH)).stdout.strip()
    _require(len(expected) == 40 and expected == observed, "execution carrier drift")
    return observed


def _read_candidate(path: Path) -> bytes:
    _require(not path.is_symlink() and path.is_file(), "candidate must be regular file")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise UbuntuGlobalDedupError("candidate is unreadable") from exc


def _compose_graph(
    base_inventory: Mapping[str, Any],
    base_payloads: Mapping[str, bytes],
    extension_sources: list[dict[str, Any]],
    extension_payloads: Mapping[str, bytes],
) -> tuple[dict[str, Any], dict[str, bytes]]:
    rows = base_inventory.get("sources")
    _require(type(rows) is list and bool(rows), "base source rows missing")
    base_ids = {row.get("source_id") for row in rows if type(row) is dict}
    extension_ids = {row.get("source_id") for row in extension_sources}
    _require(
        len(base_ids) == len(rows) and None not in base_ids,
        "base source identities invalid",
    )
    _require(base_ids == set(base_payloads), "base inventory/payload coverage mismatch")
    _require(
        len(extension_ids) == len(extension_sources) and None not in extension_ids,
        "Ubuntu source identities invalid",
    )
    _require(
        extension_ids == set(extension_payloads),
        "Ubuntu inventory/payload coverage mismatch",
    )
    _require(not (base_ids & extension_ids), "Ubuntu source-id collision with incumbent graph")
    inventory = copy.deepcopy(dict(base_inventory))
    inventory["sources"] = [*copy.deepcopy(rows), *copy.deepcopy(extension_sources)]
    inventory["final_refresh_required"] = False
    inventory["terminal_refresh_rule"] = (
        "Exact reconstructed Nomis-free V8 graph plus exact authenticated Ubuntu "
        "IRC rows; pair decisions delegate to terminal PR #824/V3 semantics through "
        "the independently qualified indexed executor."
    )
    payloads = dict(base_payloads)
    payloads.update(extension_payloads)
    return inventory, payloads


def _terminal(
    report: Mapping[str, Any], expected_objects: int, expected_bytes: int
) -> Mapping[str, Any]:
    _require(report.get("source_count") == expected_objects, "dedup source count drift")
    terminal = report.get("terminal_candidates")
    _require(isinstance(terminal, Mapping), "terminal dedup summary missing")
    before = terminal.get("declared_capacity_bytes_before")
    after = terminal.get("conservative_unique_capacity_bytes_after")
    discount = terminal.get("duplicate_discount_bytes")
    _require(type(before) is int and before == expected_bytes, "pre-dedup byte drift")
    _require(type(after) is int and 0 < after <= before, "post-dedup byte value invalid")
    _require(
        type(discount) is int and discount >= 0 and before - after == discount,
        "dedup byte arithmetic drift",
    )
    return terminal


def _validate_projection(
    report: Mapping[str, Any],
    projection: Mapping[str, Any],
    expected_objects: int,
    expected_bytes: int,
) -> None:
    terminal = _terminal(report, expected_objects, expected_bytes)
    _require(
        projection.get("schema_version") == v9_semantics.SURVIVOR_SCHEMA,
        "survivor projection schema drift",
    )
    core = dict(projection)
    claimed = core.pop("survivor_authority_sha256", None)
    _require(
        type(claimed) is str and claimed == _sha256(_canonical(core)),
        "projection hash drift",
    )
    _require(
        projection.get("matcher_report_sha256") == report.get("report_sha256"),
        "projection/report identity drift",
    )
    ids = projection.get("survivor_source_ids")
    _require(
        type(ids) is list
        and all(type(item) is str and item for item in ids)
        and len(ids) == len(set(ids)),
        "survivor ids invalid",
    )
    _require(
        projection.get("pre_dedup_source_object_count") == expected_objects,
        "projection pre-dedup object count drift",
    )
    _require(
        projection.get("pre_dedup_declared_capacity_bytes") == expected_bytes,
        "projection pre-dedup byte count drift",
    )
    _require(
        projection.get("post_dedup_declared_capacity_bytes")
        == terminal.get("conservative_unique_capacity_bytes_after"),
        "projection post-dedup byte count drift",
    )
    _require(
        projection.get("post_dedup_survivor_source_object_count") == len(ids),
        "projection survivor count drift",
    )


def _ubuntu_survivor_authority(
    inventory: Mapping[str, Any],
    combined_report: Mapping[str, Any],
    projection: Mapping[str, Any],
    *,
    marginal_unique_bytes: int,
) -> dict[str, Any]:
    rows = inventory.get("sources")
    _require(type(rows) is list, "combined inventory rows missing")
    by_id = {
        row["source_id"]: row
        for row in rows
        if type(row) is dict and type(row.get("source_id")) is str
    }
    survivor_ids = projection.get("survivor_source_ids")
    _require(type(survivor_ids) is list, "combined survivor ids missing")
    ubuntu_rows: list[dict[str, Any]] = []
    for source_id in survivor_ids:
        row = by_id.get(source_id)
        _require(type(row) is dict, "survivor missing from combined inventory")
        if row.get("source_family") != ubuntu.SOURCE_FAMILY:
            continue
        ubuntu_rows.append(
            {
                "source_id": source_id,
                "source_family": row.get("source_family"),
                "stable_origin_id": row.get("stable_origin_id"),
                "stable_object_id": row.get("stable_object_id"),
                "declared_capacity_bytes": row.get("declared_capacity_bytes"),
                "expected_raw_sha256": row.get("expected_raw_sha256"),
                "origin_key": row.get("origin_key"),
            }
        )
    _require(
        all(
            type(row["declared_capacity_bytes"]) is int
            and row["declared_capacity_bytes"] > 0
            and type(row["expected_raw_sha256"]) is str
            and len(row["expected_raw_sha256"]) == 64
            for row in ubuntu_rows
        ),
        "Ubuntu survivor metadata invalid",
    )
    survivor_bytes = sum(row["declared_capacity_bytes"] for row in ubuntu_rows)
    core = {
        "schema_version": SURVIVOR_SCHEMA,
        "matcher_report_sha256": combined_report.get("report_sha256"),
        "selection_projection_sha256": projection.get("survivor_authority_sha256"),
        "ubuntu_source_family": ubuntu.SOURCE_FAMILY,
        "pre_dedup_source_object_count": EXPECTED_UBUNTU_OBJECTS,
        "pre_dedup_declared_capacity_bytes": EXPECTED_UBUNTU_BYTES,
        "post_dedup_survivor_source_object_count": len(ubuntu_rows),
        "post_dedup_survivor_declared_capacity_bytes": survivor_bytes,
        "marginal_global_unique_capacity_bytes": marginal_unique_bytes,
        "survivors": ubuntu_rows,
        "truth_boundary": {
            "global_dedup_execution_complete": True,
            "reserved_evaluation_decontamination_complete": False,
            "canonical_quality_privacy_complete": False,
            "family_caps_complete": False,
            "cluster_safe_split_complete": False,
            "packing_complete": False,
            "canonical_capacity_credited": 0,
            "training_authorized_bytes": 0,
            "authorized_optimized_target_exposure": 0,
            "tokenizer_fit_authorized": False,
            "optimizer_updates": 0,
            "training_executed": False,
            "learned_weights_created": False,
            "final_test_outcomes_read": False,
            "paid_compute_used": False,
            "whole_corpus_external_llm_cleanliness_claimed": False,
        },
    }
    return {**core, "survivor_authority_sha256": _sha256(_canonical(core))}


def execute(
    *,
    v7_root: Path,
    bulk_workspace: Path,
    candidate_jsonl: Path,
    output_base_report: Path,
    output_combined_report: Path,
    output_survivors: Path,
    output_evidence: Path,
    expected_execution_head: str,
    max_candidate_pairs: int,
    max_index_postings: int,
    max_pair_expansions: int,
) -> dict[str, Any]:
    execution_head = incumbent._bind_execution_head(expected_execution_head)
    authority_blobs = _verify_authority_paths()
    carrier_blob = _verify_carrier()
    bulk_workspace = incumbent._prepare_empty_bulk_workspace(bulk_workspace)
    verified_v7_head = incumbent._verify_v7_worktree(v7_root)
    config = v8.load_config(ROOT / "configs/data/next100_065f_global_dedup_v8.json")
    matcher, base_inventory, base_payloads, removal = (
        incumbent._reconstruct_v8_with_historical_namespace(
            v7_root=v7_root,
            bulk_workspace=bulk_workspace,
            config=config,
        )
    )
    _require(
        incumbent._verify_v7_worktree(v7_root) == verified_v7_head,
        "V7 worktree drifted during reconstruction",
    )
    _require(len(base_payloads) == EXPECTED_BASE_OBJECTS, "base object-count drift")
    _require(
        incumbent._declared_capacity_bytes(
            base_inventory, base_payloads, label="Ubuntu execution base"
        )
        == EXPECTED_BASE_BYTES,
        "base declared-capacity drift",
    )

    candidate_bytes = _read_candidate(candidate_jsonl)
    facade_bytes = (ROOT / "src/twelve_six/data/expanded_global_dedup_v9.py").read_bytes()
    crossbind_bytes = (
        ROOT / "configs/data/d03_ubuntu_irc_repaired_execution_rights_crossbind_v1.json"
    ).read_bytes()
    rights_bytes = (
        ROOT / "configs/data/d03_common_pile_ubuntu_irc_source_rights_v1.json"
    ).read_bytes()
    parent_bytes = (ROOT / "configs/data/common_pile_source_rights_v1.json").read_bytes()
    evidence_bytes = (
        ROOT / "evidence/d03_common_pile_ubuntu_irc_real_execution_v1.json"
    ).read_bytes()
    extension_sources, extension_payloads, intake_receipt = (
        ubuntu.prepare_ubuntu_v9_intake(
            incumbent_v9_product_head=ubuntu.INCUMBENT_V9_PRODUCT_HEAD,
            incumbent_v9_facade_bytes=facade_bytes,
            crossbind_bytes=crossbind_bytes,
            rights_authority_bytes=rights_bytes,
            parent_registry_bytes=parent_bytes,
            execution_evidence_bytes=evidence_bytes,
            candidate_bytes=candidate_bytes,
        )
    )
    _require(len(extension_sources) == EXPECTED_UBUNTU_OBJECTS, "Ubuntu object-count drift")
    _require(
        incumbent._declared_capacity_bytes(
            {"sources": extension_sources},
            extension_payloads,
            label="Ubuntu projection",
        )
        == EXPECTED_UBUNTU_BYTES,
        "Ubuntu declared-capacity drift",
    )
    inventory, payloads = _compose_graph(
        base_inventory,
        base_payloads,
        extension_sources,
        extension_payloads,
    )
    _require(len(payloads) == EXPECTED_COMBINED_OBJECTS, "combined object-count drift")
    _require(
        incumbent._declared_capacity_bytes(inventory, payloads, label="combined graph")
        == EXPECTED_COMBINED_BYTES,
        "combined declared-capacity drift",
    )

    indexed.attest_incumbent_runtime(matcher)
    base_report = indexed.audit_payloads_indexed(
        matcher,
        base_inventory,
        base_payloads,
        max_candidate_pairs=max_candidate_pairs,
        max_index_postings=max_index_postings,
        max_pair_expansions=max_pair_expansions,
    )
    matcher.verify_report(base_report)
    base_terminal = _terminal(base_report, EXPECTED_BASE_OBJECTS, EXPECTED_BASE_BYTES)
    # V3 match rows retain the exact literal score=1.0 from _lineage_matches.
    # CPython marshal v4 reference encoding is refcount-sensitive, so keeping the
    # verified base report alive can make the unchanged strict attester report a
    # false callable-code drift. Preserve only the durable facts needed below,
    # then release the match-bearing report before the next unchanged attestation.
    base_report_sha256 = base_report.get("report_sha256")
    _require(
        type(base_report_sha256) is str and len(base_report_sha256) == 64,
        "base report identity invalid",
    )
    base_report_bytes = incumbent._canonical(dict(base_report))
    del base_report

    indexed.attest_incumbent_runtime(matcher)
    combined_report = indexed.audit_payloads_indexed(
        matcher,
        inventory,
        payloads,
        max_candidate_pairs=max_candidate_pairs,
        max_index_postings=max_index_postings,
        max_pair_expansions=max_pair_expansions,
    )
    matcher.verify_report(combined_report)
    combined_terminal = _terminal(
        combined_report,
        EXPECTED_COMBINED_OBJECTS,
        EXPECTED_COMBINED_BYTES,
    )
    projection = v9_semantics._derive_survivors(combined_report)
    _validate_projection(
        combined_report,
        projection,
        EXPECTED_COMBINED_OBJECTS,
        EXPECTED_COMBINED_BYTES,
    )
    marginal = (
        combined_terminal["conservative_unique_capacity_bytes_after"]
        - base_terminal["conservative_unique_capacity_bytes_after"]
    )
    _require(0 <= marginal <= EXPECTED_UBUNTU_BYTES, "marginal unique capacity invalid")
    survivors = _ubuntu_survivor_authority(
        inventory,
        combined_report,
        projection,
        marginal_unique_bytes=marginal,
    )

    evidence_core = {
        "schema_version": SCHEMA,
        "execution_profile": "LOCAL_FREE",
        "execution_head_sha": execution_head,
        "parent_convergence_head_sha": PARENT_CONVERGENCE_HEAD,
        "execution_carrier_git_blob_sha1": carrier_blob,
        "authority_path_blobs": authority_blobs,
        "v7_head_sha": verified_v7_head,
        "base": {
            "source_object_count": EXPECTED_BASE_OBJECTS,
            "declared_capacity_bytes": EXPECTED_BASE_BYTES,
            "report_sha256": base_report_sha256,
            "post_dedup_unique_bytes": base_terminal.get(
                "conservative_unique_capacity_bytes_after"
            ),
            "nomis1864_deauthorization": removal,
        },
        "ubuntu_irc": {
            "candidate_sha256": ubuntu.CANDIDATE_SHA256,
            "candidate_source_object_count": EXPECTED_UBUNTU_OBJECTS,
            "candidate_normalized_utf8_bytes": EXPECTED_UBUNTU_BYTES,
            "matcher_input_sha256": intake_receipt["matcher_input_sha256"],
            "post_dedup_survivor_source_object_count": survivors[
                "post_dedup_survivor_source_object_count"
            ],
            "post_dedup_survivor_declared_capacity_bytes": survivors[
                "post_dedup_survivor_declared_capacity_bytes"
            ],
            "marginal_global_unique_capacity_bytes": marginal,
            "global_dedup_loss_vs_raw_candidate_bytes": EXPECTED_UBUNTU_BYTES - marginal,
        },
        "combined": {
            "source_object_count": EXPECTED_COMBINED_OBJECTS,
            "declared_capacity_bytes": EXPECTED_COMBINED_BYTES,
            "report_sha256": combined_report.get("report_sha256"),
            "post_dedup_unique_bytes": combined_terminal.get(
                "conservative_unique_capacity_bytes_after"
            ),
        },
        "survivor_authority_sha256": survivors["survivor_authority_sha256"],
        "content_boundary": {
            "candidate_payload_uploaded": False,
            "durable_source_text_emitted": False,
            "reports_text_free": True,
            "survivor_authority_text_free": True,
        },
        "truth_boundary": {
            "canonical_capacity_credited": 0,
            "training_authorized_bytes": 0,
            "authorized_unique_loss_positions": 0,
            "authorized_optimized_target_exposure": 0,
            "tokenizer_fit_authorized": False,
            "model_training_authorized": False,
            "optimizer_updates": 0,
            "training_executed": False,
            "learned_weights_created": False,
            "final_test_outcomes_read": False,
            "paid_compute_used": False,
        },
    }
    evidence = {
        **evidence_core,
        "evidence_identity_sha256": _sha256(_canonical(evidence_core)),
    }

    # Reconstruct publication data only after the final strict runtime attestation.
    # JSON decoding creates a value-equivalent report without retaining references
    # to V3 code-object literal constants. Re-canonicalization must reproduce the
    # exact verified bytes frozen above before the atomic publisher can consume it.
    try:
        published_base_report = json.loads(base_report_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise UbuntuGlobalDedupError("frozen base report is not strict UTF-8 JSON") from exc
    _require(type(published_base_report) is dict, "frozen base report root invalid")
    _require(
        incumbent._canonical(dict(published_base_report)) == base_report_bytes,
        "frozen base report canonical bytes drift",
    )
    _require(
        published_base_report.get("report_sha256") == base_report_sha256,
        "frozen base report identity drift",
    )
    incumbent._publish_json_outputs(
        (
            (output_base_report, published_base_report),
            (output_combined_report, combined_report),
            (output_survivors, survivors),
            (output_evidence, evidence),
        )
    )
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v7-root", type=Path, required=True)
    parser.add_argument("--bulk-workspace", type=Path, required=True)
    parser.add_argument("--candidate-jsonl", type=Path, required=True)
    parser.add_argument("--output-base-report", type=Path, required=True)
    parser.add_argument("--output-combined-report", type=Path, required=True)
    parser.add_argument("--output-survivors", type=Path, required=True)
    parser.add_argument("--output-evidence", type=Path, required=True)
    parser.add_argument("--expected-execution-head", required=True)
    parser.add_argument("--max-candidate-pairs", type=int, default=5_000_000)
    parser.add_argument(
        "--max-index-postings",
        type=int,
        default=indexed.DEFAULT_MAX_INDEX_POSTINGS,
    )
    parser.add_argument(
        "--max-pair-expansions",
        type=int,
        default=indexed.DEFAULT_MAX_PAIR_EXPANSIONS,
    )
    args = parser.parse_args()
    evidence = execute(
        v7_root=args.v7_root,
        bulk_workspace=args.bulk_workspace,
        candidate_jsonl=args.candidate_jsonl,
        output_base_report=args.output_base_report,
        output_combined_report=args.output_combined_report,
        output_survivors=args.output_survivors,
        output_evidence=args.output_evidence,
        expected_execution_head=args.expected_execution_head,
        max_candidate_pairs=args.max_candidate_pairs,
        max_index_postings=args.max_index_postings,
        max_pair_expansions=args.max_pair_expansions,
    )
    print("D03_UBUNTU_IRC_GLOBAL_DEDUP_EXECUTION=PASS_ZERO_CREDIT")
    print("EVIDENCE_IDENTITY_SHA256=" + evidence["evidence_identity_sha256"])
    print(
        "MARGINAL_GLOBAL_UNIQUE_CAPACITY_BYTES="
        + str(evidence["ubuntu_irc"]["marginal_global_unique_capacity_bytes"])
    )
    print("SURVIVOR_AUTHORITY_SHA256=" + evidence["survivor_authority_sha256"])
    print("AUTHORIZED_OPTIMIZED_TARGET_EXPOSURE=0")
    print("TRAINING_EXECUTED=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
