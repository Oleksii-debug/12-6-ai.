#!/usr/bin/env python3
"""Extend the exact code4+Flask execution with physically reacquired attrs 26.1.0.

Execution-only carrier. It reuses the incumbent matcher and the already-qualified
code4+Flask runner, reacquires the exact attrs authority into ephemeral storage,
then measures the complete six-family extension under one global dedup graph.
"""
from __future__ import annotations

import argparse
import json
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
SRC = ROOT / "src"

import sys

for location in (str(TOOLS), str(SRC)):
    if location not in sys.path:
        sys.path.insert(0, location)

import run_d03_code4_flask_global_dedup_v1 as bundle
import validate_next100_053_attrs_code_source as attrs_source

SCHEMA = "12-6.d03-code4-flask-attrs-global-dedup-execution.v1"
PARENT_CODE4_FLASK_HEAD = "245f7afc4860d10da677bedb5e84f9fe50e0ba49"
EXPECTED_CODE4_FLASK_RUNNER_BLOB = "e14aea662fd5c5bb17341e0f9cefe0c9d5ac1485"
EXPECTED_ATTRS_CONFIG_BLOB = "7a9d3b2fc0f2ec5de2257d256be3b0c4fb66144d"
EXPECTED_ATTRS_VALIDATOR_BLOB = "30d57525bd19a60fed84273c3e018b4d17b05049"

ATTRS_PRODUCT_HEAD = "cda0232d5574ef91eae0d7e0b7fa5efddcbe218b"
ATTRS_PHYSICAL_RUN_ID = 33006080831
ATTRS_HISTORICAL_AUTHORITY_SHA256 = (
    "151e593c3b67ae4c7686323983e6c45306a870b732573ee4820c0c017b65a7d4"
)
ATTRS_ARTIFACT_ID = 9621650719
ATTRS_ARTIFACT_DIGEST = (
    "sha256:a8176b50a2254fcb50a6f80ca82b63459ba8e9cfddba904b16e5ac79f9c55ff2"
)

ATTRS_OBJECTS = 4
ATTRS_BYTES = 170_435
ATTRS_FAMILIES = 1

CODE4_BYTES = 336_947
FLASK_BYTES = 183_088
BASE_EXTENSION_OBJECTS = 16
BASE_EXTENSION_BYTES = 520_035
BASE_EXTENSION_FAMILIES = 5

EXTENSION_OBJECTS = BASE_EXTENSION_OBJECTS + ATTRS_OBJECTS
EXTENSION_BYTES = BASE_EXTENSION_BYTES + ATTRS_BYTES
EXTENSION_FAMILIES = BASE_EXTENSION_FAMILIES + ATTRS_FAMILIES
COMBINED_OBJECTS = bundle.code4.EXPECTED_BASE_OBJECTS + EXTENSION_OBJECTS
COMBINED_BYTES = bundle.code4.EXPECTED_BASE_BYTES + EXTENSION_BYTES
_SHA40 = re.compile(r"^[0-9a-f]{40}$")


class Code4FlaskAttrsGlobalDedupError(RuntimeError):
    """Fail-closed exact-authority or bundle execution mismatch."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Code4FlaskAttrsGlobalDedupError(message)


def verify_local_authority(expected_execution_head: str) -> dict[str, str]:
    require(
        isinstance(expected_execution_head, str)
        and _SHA40.fullmatch(expected_execution_head) is not None,
        "expected execution head must be exact lowercase 40-hex SHA",
    )
    require(
        bundle.code4.git("rev-parse", "HEAD") == expected_execution_head,
        "execution HEAD drift",
    )
    ancestor = bundle.code4.incumbent._git(
        "merge-base",
        "--is-ancestor",
        PARENT_CODE4_FLASK_HEAD,
        expected_execution_head,
        check=False,
    )
    require(
        ancestor.returncode == 0,
        "terminal code4+Flask execution head is not an ancestor",
    )

    pinned = {
        "tools/run_d03_code4_flask_global_dedup_v1.py": (
            EXPECTED_CODE4_FLASK_RUNNER_BLOB
        ),
        "configs/data/next100_053_attrs_code_source_v1.json": (
            EXPECTED_ATTRS_CONFIG_BLOB
        ),
        "tools/validate_next100_053_attrs_code_source.py": (
            EXPECTED_ATTRS_VALIDATOR_BLOB
        ),
    }
    for path, expected_blob in pinned.items():
        require(
            bundle.code4.git("rev-parse", f"HEAD:{path}") == expected_blob,
            f"authority blob drift: {path}",
        )
        require(
            bundle.code4.git("hash-object", str(ROOT / path)) == expected_blob,
            f"authority worktree drift: {path}",
        )
    return pinned


def capture_attrs_sources() -> tuple[
    list[dict[str, Any]],
    dict[str, bytes],
    list[dict[str, Any]],
    dict[str, Any],
]:
    cfg = attrs_source.load_config(ROOT)
    require(
        cfg["upstream_commit"] == "7bfc49e9b22d5ba25b6e429524c3d49fee27cb36",
        "attrs commit authority drift",
    )
    require(
        cfg["source_family"] == "github:python-attrs/attrs",
        "attrs source-family authority drift",
    )

    rows: list[dict[str, Any]] = []
    payloads: dict[str, bytes] = {}
    edges: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="d03-attrs-physical-") as tmp:
        output = Path(tmp) / "attrs-evidence"
        report = attrs_source.build_report(ROOT, output)
        attrs_source.validate_report(report)
        require(
            report["authority_sha256"] == ATTRS_HISTORICAL_AUTHORITY_SHA256,
            "attrs exact terminal authority replay drift",
        )
        require(report["object_count"] == ATTRS_OBJECTS, "attrs object count drift")
        require(report["capacity_bytes"] == ATTRS_BYTES, "attrs capacity drift")
        require(
            report["training_purpose_decision"] == "ALLOWED",
            "attrs training-purpose decision drift",
        )
        require(
            report["evaluation_reservation"][
                "selected_objects_are_training_only_not_evaluation_authority"
            ]
            is True,
            "attrs evaluation boundary drift",
        )

        by_path = {item["path"]: item for item in report["objects"]}
        for index, item in enumerate(cfg["files"]):
            path = str(item["path"])
            observed = by_path.get(path)
            require(isinstance(observed, Mapping), f"attrs report missing {path}")
            raw = (
                output
                / "snapshots"
                / f"{item['git_blob_sha1']}.py"
            ).read_bytes()
            require(len(raw) == int(item["size_bytes"]), f"{path}: attrs byte drift")
            require(
                attrs_source.git_blob_sha1(raw) == str(item["git_blob_sha1"]),
                f"{path}: attrs blob drift",
            )
            require(
                bundle.code4.sha256(raw) == observed["raw_sha256"],
                f"{path}: attrs SHA-256 drift",
            )

            leaf = re.sub(r"[^A-Za-z0-9]+", "_", path).strip("_").lower()
            source_id = f"code.attrs.26_1_0.{index:02d}.{leaf}"
            url = attrs_source.upstream_raw_url(cfg, path)
            row = bundle.code4._source_row(
                repository="python-attrs/attrs",
                commit=str(cfg["upstream_commit"]),
                family=str(cfg["source_family"]),
                source_id=source_id,
                path=path,
                url=url,
                expected_bytes=int(item["size_bytes"]),
                expected_blob=str(item["git_blob_sha1"]),
                raw=raw,
            )
            rows.append(row)
            payloads[source_id] = raw

    require(len(rows) == ATTRS_OBJECTS, "attrs physical object count drift")
    require(
        sum(int(row["declared_capacity_bytes"]) for row in rows) == ATTRS_BYTES,
        "attrs physical declared-capacity drift",
    )
    anchor = rows[0]["source_id"]
    edges = [
        {
            "left_source_id": anchor,
            "right_source_id": row["source_id"],
            "relation": "sibling_same_origin",
            "capacity_collapsing": False,
            "independence_collapsing": True,
            "evidence": (
                "Exact sibling objects from one pinned upstream repository revision: "
                "python-attrs/attrs@7bfc49e9b22d5ba25b6e429524c3d49fee27cb36"
            ),
        }
        for row in rows[1:]
    ]
    require(len(edges) == ATTRS_OBJECTS - 1, "attrs sibling edge count drift")

    authority = {
        "product_head_sha": ATTRS_PRODUCT_HEAD,
        "historical_physical_run_id": ATTRS_PHYSICAL_RUN_ID,
        "historical_authority_sha256": ATTRS_HISTORICAL_AUTHORITY_SHA256,
        "historical_artifact_id": ATTRS_ARTIFACT_ID,
        "historical_artifact_digest": ATTRS_ARTIFACT_DIGEST,
        "fresh_replay_authority_sha256": ATTRS_HISTORICAL_AUTHORITY_SHA256,
        "source_family": "github:python-attrs/attrs",
        "source_family_count": ATTRS_FAMILIES,
        "object_count": ATTRS_OBJECTS,
        "candidate_raw_bytes": ATTRS_BYTES,
        "upstream_commit": "7bfc49e9b22d5ba25b6e429524c3d49fee27cb36",
        "license_spdx": "MIT",
        "training_purpose_decision": "ALLOWED",
        "evaluation_authorized": False,
    }
    return rows, payloads, edges, authority


def execute(
    *,
    v7_root: Path,
    bulk_workspace: Path,
    expected_execution_head: str,
    output_base_report: Path,
    output_combined_report: Path,
    output_survivors: Path,
    output_evidence: Path,
    max_candidate_pairs: int,
    max_index_postings: int,
    max_pair_expansions: int,
) -> dict[str, Any]:
    local_authority = verify_local_authority(expected_execution_head)

    original_capture = bundle.capture_flask_sources
    original_schema = bundle.SCHEMA
    original_extension_objects = bundle.EXTENSION_OBJECTS
    original_extension_bytes = bundle.EXTENSION_BYTES
    original_extension_families = bundle.EXTENSION_FAMILIES
    original_combined_objects = bundle.COMBINED_OBJECTS
    original_combined_bytes = bundle.COMBINED_BYTES

    def capture_flask_and_attrs() -> tuple[
        list[dict[str, Any]],
        dict[str, bytes],
        list[dict[str, Any]],
        dict[str, Any],
    ]:
        flask_rows, flask_payloads, flask_edges, flask_authority = original_capture()
        attrs_rows, attrs_payloads, attrs_edges, attrs_authority = (
            capture_attrs_sources()
        )
        overlap = set(flask_payloads) & set(attrs_payloads)
        require(not overlap, f"Flask/attrs source-id collision: {sorted(overlap)}")
        return (
            [*flask_rows, *attrs_rows],
            {**flask_payloads, **attrs_payloads},
            [*flask_edges, *attrs_edges],
            {"flask": flask_authority, "attrs": attrs_authority},
        )

    inner_evidence = output_evidence.with_name(output_evidence.name + ".inner")
    try:
        bundle.SCHEMA = SCHEMA
        bundle.EXTENSION_OBJECTS = EXTENSION_OBJECTS
        bundle.EXTENSION_BYTES = EXTENSION_BYTES
        bundle.EXTENSION_FAMILIES = EXTENSION_FAMILIES
        bundle.COMBINED_OBJECTS = COMBINED_OBJECTS
        bundle.COMBINED_BYTES = COMBINED_BYTES
        bundle.capture_flask_sources = capture_flask_and_attrs
        evidence = bundle.execute(
            v7_root=v7_root,
            bulk_workspace=bulk_workspace,
            expected_execution_head=expected_execution_head,
            output_base_report=output_base_report,
            output_combined_report=output_combined_report,
            output_survivors=output_survivors,
            output_evidence=inner_evidence,
            max_candidate_pairs=max_candidate_pairs,
            max_index_postings=max_index_postings,
            max_pair_expansions=max_pair_expansions,
        )
    finally:
        bundle.capture_flask_sources = original_capture
        bundle.SCHEMA = original_schema
        bundle.EXTENSION_OBJECTS = original_extension_objects
        bundle.EXTENSION_BYTES = original_extension_bytes
        bundle.EXTENSION_FAMILIES = original_extension_families
        bundle.COMBINED_OBJECTS = original_combined_objects
        bundle.COMBINED_BYTES = original_combined_bytes

    nested = evidence.pop("flask_physical_source_authority", None)
    require(
        isinstance(nested, Mapping)
        and isinstance(nested.get("flask"), Mapping)
        and isinstance(nested.get("attrs"), Mapping),
        "nested Flask/attrs authority missing",
    )
    delta = evidence.pop("code4_plus_flask_delta")
    inner_identity = str(evidence.pop("evidence_identity_sha256"))

    previous_wrapper_authority = evidence.pop("wrapper_authority_path_blobs", None)
    evidence["schema_version"] = SCHEMA
    evidence["parent_code4_flask_execution_head_sha"] = PARENT_CODE4_FLASK_HEAD
    evidence["code4_flask_wrapper_authority_path_blobs"] = previous_wrapper_authority
    evidence["wrapper_authority_path_blobs"] = local_authority
    evidence["flask_physical_source_authority"] = dict(nested["flask"])
    evidence["attrs_physical_source_authority"] = dict(nested["attrs"])
    evidence["candidate_bundle"] = {
        "source_family_count": EXTENSION_FAMILIES,
        "source_object_count": EXTENSION_OBJECTS,
        "candidate_raw_bytes": EXTENSION_BYTES,
        "components": {
            "code4_raw_bytes": CODE4_BYTES,
            "flask_raw_bytes": FLASK_BYTES,
            "attrs_raw_bytes": ATTRS_BYTES,
        },
    }
    delta["pre_dedup_candidate_bytes"] = EXTENSION_BYTES
    delta["global_dedup_loss_vs_raw_candidate_bytes"] = (
        EXTENSION_BYTES - int(delta["marginal_global_unique_capacity_bytes"])
    )
    evidence["code4_plus_flask_attrs_delta"] = delta
    evidence["inner_code4_flask_execution_evidence_identity_sha256"] = inner_identity

    evidence["evidence_identity_sha256"] = bundle.code4.sha256(
        bundle.code4.canonical(evidence)
    )
    bundle.code4.incumbent._publish_json_outputs(((output_evidence, evidence),))
    try:
        inner_evidence.unlink()
    except FileNotFoundError:
        pass
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--v7-root", type=Path, required=True)
    parser.add_argument("--bulk-workspace", type=Path, required=True)
    parser.add_argument("--expected-execution-head", required=True)
    parser.add_argument("--output-base-report", type=Path, required=True)
    parser.add_argument("--output-combined-report", type=Path, required=True)
    parser.add_argument("--output-survivors", type=Path, required=True)
    parser.add_argument("--output-evidence", type=Path, required=True)
    parser.add_argument("--max-candidate-pairs", type=int, default=5_000_000)
    parser.add_argument(
        "--max-index-postings",
        type=int,
        default=bundle.code4.incumbent.indexed.DEFAULT_MAX_INDEX_POSTINGS,
    )
    parser.add_argument(
        "--max-pair-expansions",
        type=int,
        default=bundle.code4.incumbent.indexed.DEFAULT_MAX_PAIR_EXPANSIONS,
    )
    args = parser.parse_args()

    try:
        evidence = execute(
            v7_root=args.v7_root,
            bulk_workspace=args.bulk_workspace,
            expected_execution_head=args.expected_execution_head,
            output_base_report=args.output_base_report,
            output_combined_report=args.output_combined_report,
            output_survivors=args.output_survivors,
            output_evidence=args.output_evidence,
            max_candidate_pairs=args.max_candidate_pairs,
            max_index_postings=args.max_index_postings,
            max_pair_expansions=args.max_pair_expansions,
        )
    except (
        Code4FlaskAttrsGlobalDedupError,
        bundle.Code4FlaskGlobalDedupError,
        bundle.code4.FourCodeGlobalDedupError,
        bundle.code4.incumbent.Franko1901GlobalDedupError,
        bundle.code4.incumbent.indexed.IndexedExecutionError,
        attrs_source.AdmissionError,
        OSError,
        ValueError,
    ) as exc:
        rendered = " ".join(str(exc).split())[:240]
        print(
            json.dumps(
                {
                    "status": "BLOCKED_CODE4_FLASK_ATTRS_GLOBAL_DEDUP",
                    "error_type": type(exc).__name__,
                    "error_detail": rendered,
                    "canonical_capacity_credited": 0,
                    "authorized_optimized_target_exposure": 0,
                    "training_executed": False,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 2

    delta = evidence["code4_plus_flask_attrs_delta"]
    print("D03_CODE4_FLASK_ATTRS_GLOBAL_DEDUP=PASS_ZERO_CREDIT")
    print(
        "MARGINAL_GLOBAL_UNIQUE_CAPACITY_BYTES="
        + str(delta["marginal_global_unique_capacity_bytes"])
    )
    print(
        "LATER_GATE_LOSS_BUDGET_IF_TARGET_STILL_POSSIBLE="
        + str(delta["later_gate_loss_budget_if_target_still_possible"])
    )
    print("EVIDENCE_IDENTITY_SHA256=" + evidence["evidence_identity_sha256"])
    print("AUTHORIZED_OPTIMIZED_TARGET_EXPOSURE=0")
    print("TRAINING_EXECUTED=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
