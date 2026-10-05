#!/usr/bin/env python3
"""Run the physically qualified code4 + Flask bundle through incumbent global dedup.

Execution-only carrier. Raw source bytes remain ephemeral. Durable outputs are
text-free incumbent matcher reports, survivor authority, and zero-credit evidence.
"""
from __future__ import annotations

import argparse
import ast
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

import run_d03_four_code_sources_global_dedup_v1 as code4
from twelve_six import next100_043_flask_admission as flask

SCHEMA = "12-6.d03-code4-flask-global-dedup-execution.v1"
PARENT_CODE4_HEAD = "e33b441a767ee48a182f930d202f1419e3849480"
EXPECTED_CODE4_RUNNER_BLOB = "eb89d36fb8bf7c97d4860d242e960940a0fc0ed3"
EXPECTED_FLASK_MODULE_BLOB = "632e7e61d58060283bf8c287300e12eaa8fa2b84"
EXPECTED_FLASK_CONFIG_BLOB = "aafb3cc9171440ec408baf654e317fad9fc6eefd"

FLASK_PRODUCT_HEAD = "116383789dfbf5af8b1ef663e28f74f499b5187f"
FLASK_PHYSICAL_HEAD = "914485d5a75987447a4008588b180a4e826582bd"
FLASK_PHYSICAL_RUN_ID = 37277236307
FLASK_REPORT_SHA256 = (
    "061003f0d7922332ae8a20f210d9b3cf5cc6714f7fc0522719e4bfafecd1e0c7"
)
FLASK_SNAPSHOT_MANIFEST_SHA256 = (
    "fa25f2e561f321452ee8e6c60574a3bac501c88d643a806a0d4f187ddfa133ee"
)

CODE4_OBJECTS = 8
CODE4_BYTES = 336_947
CODE4_FAMILIES = 4
FLASK_OBJECTS = 8
FLASK_BYTES = 183_088
FLASK_FAMILIES = 1
EXTENSION_OBJECTS = CODE4_OBJECTS + FLASK_OBJECTS
EXTENSION_BYTES = CODE4_BYTES + FLASK_BYTES
EXTENSION_FAMILIES = CODE4_FAMILIES + FLASK_FAMILIES
COMBINED_OBJECTS = code4.EXPECTED_BASE_OBJECTS + EXTENSION_OBJECTS
COMBINED_BYTES = code4.EXPECTED_BASE_BYTES + EXTENSION_BYTES
_SHA40 = re.compile(r"^[0-9a-f]{40}$")


class Code4FlaskGlobalDedupError(RuntimeError):
    """Fail-closed bundle execution or authority mismatch."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Code4FlaskGlobalDedupError(message)


def verify_local_authority(expected_execution_head: str) -> dict[str, str]:
    require(
        isinstance(expected_execution_head, str)
        and _SHA40.fullmatch(expected_execution_head) is not None,
        "expected execution head must be exact lowercase 40-hex SHA",
    )
    require(
        code4.git("rev-parse", "HEAD") == expected_execution_head,
        "execution HEAD drift",
    )
    ancestor = code4.incumbent._git(
        "merge-base",
        "--is-ancestor",
        PARENT_CODE4_HEAD,
        expected_execution_head,
        check=False,
    )
    require(ancestor.returncode == 0, "code4 physical execution parent is not ancestor")

    pinned = {
        "tools/run_d03_four_code_sources_global_dedup_v1.py": (
            EXPECTED_CODE4_RUNNER_BLOB
        ),
        "src/twelve_six/next100_043_flask_admission.py": (
            EXPECTED_FLASK_MODULE_BLOB
        ),
        "configs/data/next100_043_flask_code_rights_v1.json": (
            EXPECTED_FLASK_CONFIG_BLOB
        ),
    }
    for path, expected_blob in pinned.items():
        require(
            code4.git("rev-parse", f"HEAD:{path}") == expected_blob,
            f"authority blob drift: {path}",
        )
        require(
            code4.git("hash-object", str(ROOT / path)) == expected_blob,
            f"authority worktree drift: {path}",
        )
    return pinned


def capture_flask_sources() -> tuple[
    list[dict[str, Any]],
    dict[str, bytes],
    list[dict[str, Any]],
    dict[str, Any],
]:
    manifest = flask.load_manifest(ROOT)
    flask.validate_manifest(manifest)
    upstream = manifest["upstream"]
    require(
        upstream["commit"] == "22d924701a6ae2e4cd01e9a15bbaf3946094af65",
        "Flask commit authority drift",
    )
    require(
        upstream["source_family"] == "github:pallets/flask",
        "Flask family authority drift",
    )
    identity = flask._verify_upstream_identity(manifest)

    with tempfile.TemporaryDirectory(prefix="d03-flask-rights-") as tmp:
        license_evidence, _ = flask._verify_license(manifest, Path(tmp))
    require(
        license_evidence.get("training_purpose_decision") == "ALLOWED",
        "Flask training-purpose rights drift",
    )
    require(
        license_evidence.get("redistribution_decision") == "ALLOWED",
        "Flask redistribution rights drift",
    )

    rows: list[dict[str, Any]] = []
    payloads: dict[str, bytes] = {}
    snapshot_rows: list[dict[str, Any]] = []
    repository = "pallets/flask"
    commit = str(upstream["commit"])
    family = str(upstream["source_family"])

    for index, item in enumerate(manifest["selected_files"]):
        path = str(item["path"])
        flask._assert_path_allowed(path)
        url = f"https://raw.githubusercontent.com/{repository}/{commit}/{path}"
        raw = flask._download(url)
        require(len(raw) == int(item["size_bytes"]), f"{path}: Flask byte drift")
        require(
            flask._git_blob_sha1(raw) == str(item["blob_sha1"]),
            f"{path}: Flask Git blob drift",
        )
        require(not flask._scan_secrets(raw), f"{path}: Flask secret scan rejected")
        require(not flask._scan_privacy(raw), f"{path}: Flask privacy scan rejected")
        require(b"\x00" not in raw, f"{path}: Flask NUL byte")
        text = raw.decode("utf-8")
        require(text.encode("utf-8") == raw, f"{path}: Flask UTF-8 identity drift")
        ast.parse(text, filename=path)

        leaf = re.sub(r"[^A-Za-z0-9]+", "_", path).strip("_").lower()
        source_id = f"code.flask.3_1_3.{index:02d}.{leaf}"
        row = code4._source_row(
            repository=repository,
            commit=commit,
            family=family,
            source_id=source_id,
            path=path,
            url=url,
            expected_bytes=int(item["size_bytes"]),
            expected_blob=str(item["blob_sha1"]),
            raw=raw,
        )
        rows.append(row)
        payloads[source_id] = raw
        snapshot_rows.append(
            {
                "path": path,
                "blob_sha1": str(item["blob_sha1"]),
                "raw_sha256": code4.sha256(raw),
                "size_bytes": len(raw),
            }
        )

    require(len(rows) == FLASK_OBJECTS, "Flask object count drift")
    require(
        sum(int(row["declared_capacity_bytes"]) for row in rows) == FLASK_BYTES,
        "Flask candidate byte drift",
    )
    snapshot_sha = code4.sha256(flask._cjson(snapshot_rows))
    require(
        snapshot_sha == FLASK_SNAPSHOT_MANIFEST_SHA256,
        "Flask physical snapshot identity drift",
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
                f"{repository}@{commit}"
            ),
        }
        for row in rows[1:]
    ]
    require(len(edges) == FLASK_OBJECTS - 1, "Flask sibling edge count drift")

    authority = {
        "product_head_sha": FLASK_PRODUCT_HEAD,
        "physical_execution_head_sha": FLASK_PHYSICAL_HEAD,
        "dedicated_run_id": FLASK_PHYSICAL_RUN_ID,
        "report_sha256": FLASK_REPORT_SHA256,
        "snapshot_manifest_sha256": snapshot_sha,
        "source_family": family,
        "source_family_count": FLASK_FAMILIES,
        "object_count": FLASK_OBJECTS,
        "candidate_raw_bytes": FLASK_BYTES,
        "upstream_identity": identity,
        "license_id": license_evidence["license_id"],
        "training_purpose_decision": license_evidence["training_purpose_decision"],
        "redistribution_decision": license_evidence["redistribution_decision"],
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

    original_build = code4.build_extension_graph
    original_compose = code4.compose_graph
    original_schema = code4.SCHEMA
    original_extension_objects = code4.EXPECTED_EXTENSION_OBJECTS
    original_extension_bytes = code4.EXPECTED_EXTENSION_BYTES
    original_combined_objects = code4.EXPECTED_COMBINED_OBJECTS
    original_combined_bytes = code4.EXPECTED_COMBINED_BYTES
    flask_authority: dict[str, Any] = {}

    def build_bundle(
        captured: Mapping[str, bytes],
    ) -> tuple[list[dict[str, Any]], dict[str, bytes], list[dict[str, Any]]]:
        nonlocal flask_authority
        code4.EXPECTED_EXTENSION_OBJECTS = original_extension_objects
        code4.EXPECTED_EXTENSION_BYTES = original_extension_bytes
        code4.EXPECTED_COMBINED_OBJECTS = original_combined_objects
        code4.EXPECTED_COMBINED_BYTES = original_combined_bytes
        rows, payloads, edges = original_build(captured)
        require(len(rows) == CODE4_OBJECTS, "code4 inherited object count drift")
        require(
            sum(int(row["declared_capacity_bytes"]) for row in rows) == CODE4_BYTES,
            "code4 inherited candidate byte drift",
        )

        flask_rows, flask_payloads, flask_edges, flask_authority = (
            capture_flask_sources()
        )
        rows = [*rows, *flask_rows]
        payloads = {**payloads, **flask_payloads}
        edges = [*edges, *flask_edges]

        code4.EXPECTED_EXTENSION_OBJECTS = EXTENSION_OBJECTS
        code4.EXPECTED_EXTENSION_BYTES = EXTENSION_BYTES
        code4.EXPECTED_COMBINED_OBJECTS = COMBINED_OBJECTS
        code4.EXPECTED_COMBINED_BYTES = COMBINED_BYTES
        return rows, payloads, edges

    def compose_bundle(
        base_inventory: Mapping[str, Any],
        base_payloads: Mapping[str, bytes],
        extension_rows: list[dict[str, Any]],
        extension_payloads: Mapping[str, bytes],
        extension_edges: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], dict[str, bytes]]:
        inventory, payloads = original_compose(
            base_inventory,
            base_payloads,
            extension_rows,
            extension_payloads,
            extension_edges,
        )
        inventory["terminal_refresh_rule"] = (
            "Exact reconstructed quarantined-source-free incumbent V8 graph plus "
            "sixteen physically qualified Pydantic/SciPy/Pandas/Typer/Flask source "
            "objects. Pair decisions delegate to terminal PR #824 V3 semantics "
            "through the independently qualified PR #1459 indexed executor."
        )
        quarantine_authority = json.loads(
            (
                ROOT / code4.incumbent.clean_successor.QUARANTINE_CONFIG_PATH
            ).read_text(encoding="utf-8")
        )
        code4.incumbent._verify_clean_payload_graph(
            inventory,
            payloads,
            quarantine_authority,
        )
        return inventory, payloads

    inner_evidence = output_evidence.with_name(output_evidence.name + ".inner")
    try:
        code4.SCHEMA = SCHEMA
        code4.build_extension_graph = build_bundle
        code4.compose_graph = compose_bundle
        evidence = code4.execute(
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
        code4.SCHEMA = original_schema
        code4.build_extension_graph = original_build
        code4.compose_graph = original_compose
        code4.EXPECTED_EXTENSION_OBJECTS = original_extension_objects
        code4.EXPECTED_EXTENSION_BYTES = original_extension_bytes
        code4.EXPECTED_COMBINED_OBJECTS = original_combined_objects
        code4.EXPECTED_COMBINED_BYTES = original_combined_bytes

    require(bool(flask_authority), "Flask authority was not captured")
    inner_identity = str(evidence["evidence_identity_sha256"])
    delta = evidence.pop("four_source_delta")
    evidence.pop("physical_source_authority", None)
    evidence.pop("evidence_identity_sha256", None)
    evidence["schema_version"] = SCHEMA
    evidence["parent_code4_execution_head_sha"] = PARENT_CODE4_HEAD
    evidence["wrapper_authority_path_blobs"] = local_authority
    evidence["code4_physical_source_authority"] = {
        "dedicated_run_id": 37269452704,
        "source_family_count": CODE4_FAMILIES,
        "object_count": CODE4_OBJECTS,
        "candidate_raw_bytes": CODE4_BYTES,
        "summary_sha256": code4.EXPECTED_PHYSICAL_SUMMARY_SHA256,
        "object_set_sha256": code4.EXPECTED_OBJECT_SET_SHA256,
        "license_set_sha256": code4.EXPECTED_LICENSE_SET_SHA256,
    }
    evidence["flask_physical_source_authority"] = flask_authority
    evidence["candidate_bundle"] = {
        "source_family_count": EXTENSION_FAMILIES,
        "source_object_count": EXTENSION_OBJECTS,
        "candidate_raw_bytes": EXTENSION_BYTES,
        "components": {
            "code4_raw_bytes": CODE4_BYTES,
            "flask_raw_bytes": FLASK_BYTES,
        },
    }
    delta["pre_dedup_candidate_bytes"] = EXTENSION_BYTES
    delta["global_dedup_loss_vs_raw_candidate_bytes"] = (
        EXTENSION_BYTES - int(delta["marginal_global_unique_capacity_bytes"])
    )
    evidence["code4_plus_flask_delta"] = delta
    evidence["inner_code4_execution_evidence_identity_sha256"] = inner_identity

    evidence["evidence_identity_sha256"] = code4.sha256(code4.canonical(evidence))
    code4.incumbent._publish_json_outputs(((output_evidence, evidence),))
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
        default=code4.incumbent.indexed.DEFAULT_MAX_INDEX_POSTINGS,
    )
    parser.add_argument(
        "--max-pair-expansions",
        type=int,
        default=code4.incumbent.indexed.DEFAULT_MAX_PAIR_EXPANSIONS,
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
        Code4FlaskGlobalDedupError,
        code4.FourCodeGlobalDedupError,
        code4.incumbent.Franko1901GlobalDedupError,
        code4.incumbent.indexed.IndexedExecutionError,
        flask.AdmissionError,
        OSError,
        ValueError,
    ) as exc:
        rendered = " ".join(str(exc).split())[:240]
        print(
            json.dumps(
                {
                    "status": "BLOCKED_CODE4_FLASK_GLOBAL_DEDUP",
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

    delta = evidence["code4_plus_flask_delta"]
    print("D03_CODE4_FLASK_GLOBAL_DEDUP=PASS_ZERO_CREDIT")
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
