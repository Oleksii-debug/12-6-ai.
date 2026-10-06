"""Execute exact current-Rada replacement through incumbent indexed global dedup.

Execution-only carrier. It reconstructs the incumbent Nomis-free V8 graph, removes
the exact predecessor Rada family object, validates the exact current-Rada replay
candidate through the Product adapter, then delegates matching to the already
qualified incumbent indexed matcher. Outputs contain hashes/metadata only.

This module grants no corpus, tokenizer-fit, training, final-test, paid-compute,
or scale-promotion authority.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TOOLS = ROOT / "tools"
for location in (str(SRC), str(TOOLS)):
    if location not in sys.path:
        sys.path.insert(0, location)

from twelve_six.data import rada_current_snapshot_dedup_adapter_v1 as rada

PARENT_HEAD = "e0194d1fd76673e492d33ca3f674ca794a04926a"
PARENT_AUTHORITY_ID = "543f9cdd5a9aacaf2cc00b5d4057ad8b142aa685870545cff8bd518f8085055e"
PARENT_AUTHORITY_BLOB = "9953940072fdd973cd6c83eb5899314d3484da30"
PARENT_ADAPTER_BLOB = "86a481be4fdae8fd4aa75e348a4839ba7a8b6ce8"
PARENT_QP_AUTHORITY_BLOB = "6b0c7e173460f23d3d3b702448b228253135c6cc"
PARENT_RIGHTS_BLOB = "4a6cb0bd6b009ef36c9d2fb712967a4ae1cbfe0b"

NBU_HELPER_HEAD = "b5235cfd83852854e345dea6c2e89cfbcd1cef79"
NBU_HELPER_BLOB = "dc0ed88924610fc7ac19ba4fc7d960d8679c9742"
V7_HEAD = "d3333ec1b4a508df232a5aefccd6686adda745fb"
NBU_HELPER_PATH = "tools/run_d03_nbu_current40_global_dedup_v1.py"

EXPECTED_BASE_OBJECTS = 263
EXPECTED_BASE_DECLARED_BYTES = 6_093_965
REPLACED_SOURCE_ID = "ua.rada.open-data.laws-texts.d23314"
REPLACED_DECLARED_BYTES = 88_565
REPLACED_RAW_BYTES = 332_400
REPLACED_RAW_SHA256 = "36eae31c3b0676ea7c02236fa05bd695c240c9a8eade5febc00457b8103ee1a4"
EXPECTED_REPLACED_OBJECTS = 1
EXPECTED_POST_REPLACEMENT_OBJECTS = 262
EXPECTED_POST_REPLACEMENT_DECLARED_BYTES = 6_005_400

EXPECTED_CURRENT_OBJECTS = 101_733
EXPECTED_CURRENT_PAYLOAD_BYTES = 192_393_157
EXPECTED_CURRENT_JSONL_BYTES = 224_897_989
EXPECTED_CURRENT_JSONL_SHA256 = (
    "8d1343708b3ce1747d32c3b551d6fb8ab7123be5b37f74fff3c457b6439159a7"
)
EXPECTED_CURRENT_INVENTORY_SHA256 = (
    "468a3132819527e1ce2b3aa375a55a15d55b7af53e66bf06ae1e5a69daf73e7a"
)
EXPECTED_CURRENT_DUPLICATE_HASHES = 765

EXPECTED_COMBINED_OBJECTS = 101_995
EXPECTED_COMBINED_DECLARED_BYTES = 198_398_557

AUTHORITY_PATH = (
    ROOT / "evidence/d03_rada_bulk/current_snapshot_github_replay_authority_v1.json"
)
ADAPTER_PATH = "src/twelve_six/data/rada_current_snapshot_dedup_adapter_v1.py"
QP_AUTHORITY_PATH = "src/twelve_six/data/rada_current_snapshot_qp_authority_v1.py"
RIGHTS_PATH = "configs/data/d03_rada_bulk_fresh_snapshot_rights_v2.json"

SURVIVOR_SCHEMA = "12-6.d03-rada-current-global-dedup-survivors.v1"
EVIDENCE_SCHEMA = "12-6.d03-rada-current-global-dedup-execution.v1"
TWO_CLEAN_SCHEMA = "12-6.d03-rada-current-global-dedup-two-clean.v1"


class RadaCurrentGlobalDedupError(RuntimeError):
    """Fail-closed current-Rada global-dedup execution error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RadaCurrentGlobalDedupError(message)


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


def is_hex(value: object, length: int) -> bool:
    return (
        type(value) is str
        and len(value) == length
        and all(char in "0123456789abcdef" for char in value)
    )


def git(*args: str, cwd: Path = ROOT) -> str:
    try:
        proc = subprocess.run(
            ["git", "-C", str(cwd), *args],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RadaCurrentGlobalDedupError(f"git execution failed: {exc}") from exc
    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip() or str(proc.returncode)
        raise RadaCurrentGlobalDedupError(
            f"git {' '.join(args)} failed: {detail}"
        )
    return proc.stdout.strip()


def git_blob(path: str, *, cwd: Path = ROOT) -> str:
    return git("hash-object", str(cwd / path), cwd=cwd)


def verify_product_parent(expected_execution_head: str) -> None:
    require(is_hex(expected_execution_head, 40), "execution head must be lowercase 40-hex")
    require(git("rev-parse", "HEAD") == expected_execution_head, "execution HEAD drift")
    require(
        git("merge-base", PARENT_HEAD, "HEAD") == PARENT_HEAD,
        "exact #2808 parent is not merge-base",
    )
    expected_blobs = {
        str(AUTHORITY_PATH.relative_to(ROOT)): PARENT_AUTHORITY_BLOB,
        ADAPTER_PATH: PARENT_ADAPTER_BLOB,
        QP_AUTHORITY_PATH: PARENT_QP_AUTHORITY_BLOB,
        RIGHTS_PATH: PARENT_RIGHTS_BLOB,
    }
    for path, expected in expected_blobs.items():
        observed = git("rev-parse", f"{PARENT_HEAD}:{path}")
        require(observed == expected, f"#2808 parent blob drift: {path}")
        require(git("rev-parse", f"HEAD:{path}") == expected, f"child changed parent path: {path}")
        require(git_blob(path) == expected, f"worktree parent path drift: {path}")


def load_helper(nbu_root: Path) -> ModuleType:
    require(nbu_root.is_dir(), "NBU helper worktree missing")
    require(git("rev-parse", "HEAD", cwd=nbu_root) == NBU_HELPER_HEAD, "NBU helper HEAD drift")
    helper_path = nbu_root / NBU_HELPER_PATH
    require(helper_path.is_file(), "NBU helper runner missing")
    require(git_blob(NBU_HELPER_PATH, cwd=nbu_root) == NBU_HELPER_BLOB, "NBU helper blob drift")

    import twelve_six
    import twelve_six.data

    nbu_pkg = str((nbu_root / "src/twelve_six").resolve(strict=True))
    nbu_data = str((nbu_root / "src/twelve_six/data").resolve(strict=True))
    if nbu_pkg not in twelve_six.__path__:
        twelve_six.__path__.append(nbu_pkg)
    if nbu_data not in twelve_six.data.__path__:
        twelve_six.data.__path__.append(nbu_data)

    for location in (
        str((nbu_root / "tools").resolve(strict=True)),
        str((nbu_root / "src").resolve(strict=True)),
    ):
        if location not in sys.path:
            sys.path.append(location)

    spec = importlib.util.spec_from_file_location("exact_nbu_global_dedup_helper", helper_path)
    require(spec is not None and spec.loader is not None, "cannot load exact NBU helper")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise RadaCurrentGlobalDedupError("cannot import exact NBU helper") from exc
    require(module.EXPECTED_MAIN == "019944d5fe12334791f05f1232d13de4a12e37d3", "helper main drift")
    return module


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise RadaCurrentGlobalDedupError(f"cannot load JSON: {path}") from exc
    require(type(value) is dict, f"JSON root must be exact object: {path}")
    return value


def declared_capacity(
    inventory: Mapping[str, Any],
    payloads: Mapping[str, bytes],
    *,
    label: str,
) -> int:
    rows = inventory.get("sources")
    require(type(rows) is list, f"{label}: source vector missing")
    ids: set[str] = set()
    total = 0
    for row in rows:
        require(type(row) is dict, f"{label}: source row invalid")
        source_id = row.get("source_id")
        declared = row.get("declared_capacity_bytes")
        require(
            type(source_id) is str and source_id and source_id not in ids,
            f"{label}: source id invalid or duplicate",
        )
        require(type(declared) is int and declared >= 0, f"{label}: declared bytes invalid")
        ids.add(source_id)
        total += declared
    require(ids == set(payloads), f"{label}: inventory/payload coverage mismatch")
    require(
        all(type(key) is str and type(value) is bytes for key, value in payloads.items()),
        f"{label}: payload map type drift",
    )
    return total


def remove_incumbent_rada_family(
    inventory: Mapping[str, Any],
    payloads: Mapping[str, bytes],
) -> tuple[dict[str, Any], dict[str, bytes], dict[str, Any]]:
    require(type(inventory) is dict, "base inventory must be exact object")
    rows = inventory.get("sources")
    require(type(rows) is list, "base sources missing")
    require(len(rows) == EXPECTED_BASE_OBJECTS, "base object count drift")
    require(
        declared_capacity(inventory, payloads, label="base")
        == EXPECTED_BASE_DECLARED_BYTES,
        "base declared-capacity drift",
    )

    removed = [
        row
        for row in rows
        if type(row) is dict and row.get("source_family") == rada.SOURCE_FAMILY
    ]
    require(len(removed) == EXPECTED_REPLACED_OBJECTS, "incumbent Rada family cardinality drift")
    row = removed[0]
    require(row.get("source_id") == REPLACED_SOURCE_ID, "incumbent Rada source id drift")
    require(
        row.get("declared_capacity_bytes") == REPLACED_DECLARED_BYTES,
        "incumbent Rada declared bytes drift",
    )
    raw = payloads.get(REPLACED_SOURCE_ID)
    require(type(raw) is bytes, "incumbent Rada raw payload missing")
    require(len(raw) == REPLACED_RAW_BYTES, "incumbent Rada raw byte count drift")
    require(sha256(raw) == REPLACED_RAW_SHA256, "incumbent Rada raw SHA drift")

    replacement_inventory = copy.deepcopy(dict(inventory))
    replacement_inventory["sources"] = [
        copy.deepcopy(source)
        for source in rows
        if source.get("source_id") != REPLACED_SOURCE_ID
    ]
    replacement_payloads = dict(payloads)
    del replacement_payloads[REPLACED_SOURCE_ID]
    require(
        len(replacement_inventory["sources"]) == EXPECTED_POST_REPLACEMENT_OBJECTS,
        "post-replacement base object count drift",
    )
    require(
        declared_capacity(
            replacement_inventory,
            replacement_payloads,
            label="post-replacement base",
        )
        == EXPECTED_POST_REPLACEMENT_DECLARED_BYTES,
        "post-replacement base declared bytes drift",
    )
    proof = {
        "source_family": rada.SOURCE_FAMILY,
        "replaced_source_ids": [REPLACED_SOURCE_ID],
        "replaced_source_object_count": EXPECTED_REPLACED_OBJECTS,
        "replaced_declared_capacity_bytes": REPLACED_DECLARED_BYTES,
        "replaced_raw_bytes": REPLACED_RAW_BYTES,
        "replaced_raw_sha256": REPLACED_RAW_SHA256,
        "replacement_required_by_current_adapter": True,
        "append_forbidden_by_current_adapter": True,
    }
    return replacement_inventory, replacement_payloads, proof


def validate_current_projection(
    candidate_jsonl: Path,
) -> tuple[list[dict[str, Any]], dict[str, bytes], dict[str, Any]]:
    projection = rada.validate_and_project_current_rada_candidate(
        candidate_jsonl,
        AUTHORITY_PATH,
        repository_root=ROOT,
        retain_payloads=True,
    )
    require(projection.sources is not None, "current Rada source projection missing")
    require(projection.payloads is not None, "current Rada payload projection missing")
    sources = [dict(row) for row in projection.sources]
    payloads = dict(projection.payloads)
    receipt = projection.receipt
    require(receipt.get("replay_authority_identity_sha256") == PARENT_AUTHORITY_ID, "Rada authority identity drift")
    require(receipt.get("candidate_jsonl_sha256") == EXPECTED_CURRENT_JSONL_SHA256, "Rada candidate identity drift")
    require(receipt.get("candidate_jsonl_file_bytes") == EXPECTED_CURRENT_JSONL_BYTES, "Rada candidate transport bytes drift")
    require(receipt.get("source_object_count") == EXPECTED_CURRENT_OBJECTS, "Rada projected object count drift")
    require(receipt.get("source_payload_utf8_bytes") == EXPECTED_CURRENT_PAYLOAD_BYTES, "Rada projected payload bytes drift")
    require(receipt.get("accepted_inventory_sha256") == EXPECTED_CURRENT_INVENTORY_SHA256, "Rada accepted inventory drift")
    require(receipt.get("exact_duplicate_payload_hashes_preserved") == EXPECTED_CURRENT_DUPLICATE_HASHES, "Rada duplicate observation drift")
    require(receipt.get("requires_replace_existing_source_family") is True, "Rada replacement requirement missing")
    require(receipt.get("must_not_append_to_existing_source_family") is True, "Rada append prohibition missing")
    require(receipt.get("consumer_gate") == "CURRENT_GLOBAL_CROSS_SOURCE_DEDUP_ONLY", "Rada consumer gate drift")
    require(receipt.get("rights_scope") == "ARTIFACT_RETENTION_AND_REPRODUCIBILITY_ONLY", "Rada rights scope drift")
    require(receipt.get("bulk_corpus_admission_granted") is False, "Rada corpus admission widened")
    require(receipt.get("training_authority_granted") is False, "Rada training authority widened")
    require(receipt.get("rights_recheck_for_training_required") is True, "Rada rights recheck boundary lost")
    require(receipt.get("canonical_capacity_credited") == 0, "Rada capacity credit widened")
    require(receipt.get("training_authorized_bytes") == 0, "Rada training bytes widened")
    require(receipt.get("tokenizer_fit_authorized") is False, "Rada tokenizer fit widened")
    require(len(sources) == EXPECTED_CURRENT_OBJECTS, "Rada projected source cardinality drift")
    require(len(payloads) == EXPECTED_CURRENT_OBJECTS, "Rada projected payload cardinality drift")
    require(sum(len(raw) for raw in payloads.values()) == EXPECTED_CURRENT_PAYLOAD_BYTES, "Rada payload sum drift")
    return sources, payloads, receipt


def compose_current_graph(
    helper: ModuleType,
    base_inventory: Mapping[str, Any],
    base_payloads: Mapping[str, bytes],
    current_sources: list[dict[str, Any]],
    current_payloads: Mapping[str, bytes],
) -> tuple[dict[str, Any], dict[str, bytes], dict[str, Any]]:
    replacement_inventory, replacement_payloads, replacement_proof = (
        remove_incumbent_rada_family(base_inventory, base_payloads)
    )
    base_ids = {row["source_id"] for row in replacement_inventory["sources"]}
    current_ids = {row["source_id"] for row in current_sources}
    require(len(current_ids) == len(current_sources), "current Rada source IDs duplicate")
    require(current_ids == set(current_payloads), "current Rada source/payload coverage mismatch")
    require(not (base_ids & current_ids), "current Rada source ID collides after replacement")

    inventory = copy.deepcopy(replacement_inventory)
    inventory["sources"] = [
        *copy.deepcopy(replacement_inventory["sources"]),
        *copy.deepcopy(current_sources),
    ]
    inventory["final_refresh_required"] = False
    inventory["terminal_refresh_rule"] = (
        "Exact reconstructed Nomis-free V8 source graph with the predecessor "
        "ua.rada.open-data.laws-texts family removed, then exact current-Rada "
        "replay projection supplied by #2808; pair decisions delegate only to "
        "terminal V3 semantics through the qualified indexed executor."
    )
    payloads = dict(replacement_payloads)
    payloads.update(current_payloads)
    require(len(payloads) == EXPECTED_COMBINED_OBJECTS, "combined object count drift")
    require(
        declared_capacity(inventory, payloads, label="combined")
        == EXPECTED_COMBINED_DECLARED_BYTES,
        "combined declared-capacity drift",
    )
    quarantine_authority = load_json(
        helper.ROOT / helper.clean_successor.QUARANTINE_CONFIG_PATH
    )
    helper._verify_clean_payload_graph(inventory, payloads, quarantine_authority)
    return inventory, payloads, replacement_proof


def verify_report_current_rada(
    report: Mapping[str, Any],
    current_sources: list[dict[str, Any]],
    current_payloads: Mapping[str, bytes],
) -> dict[str, Mapping[str, Any]]:
    require(report.get("source_count") == EXPECTED_COMBINED_OBJECTS, "report source count drift")
    terminal = report.get("terminal_candidates")
    require(type(terminal) is dict, "terminal matcher summary missing")
    before = terminal.get("declared_capacity_bytes_before")
    after = terminal.get("conservative_unique_capacity_bytes_after")
    discount = terminal.get("duplicate_discount_bytes")
    clusters = terminal.get("duplicate_cluster_count")
    require(before == EXPECTED_COMBINED_DECLARED_BYTES, "report pre-dedup bytes drift")
    require(type(after) is int and 0 < after <= before, "report post-dedup bytes invalid")
    require(type(discount) is int and discount == before - after, "report discount arithmetic drift")
    require(type(clusters) is int and clusters >= 0, "report cluster count invalid")

    rows = report.get("sources")
    require(type(rows) is list and len(rows) == EXPECTED_COMBINED_OBJECTS, "report source vector drift")
    by_id = {
        row.get("source_id"): row
        for row in rows
        if type(row) is dict and type(row.get("source_id")) is str
    }
    require(len(by_id) == len(rows), "report source IDs invalid or duplicate")

    expected_by_id = {row["source_id"]: row for row in current_sources}
    require(set(expected_by_id) == set(current_payloads), "current projection map drift")
    for source_id, expected in expected_by_id.items():
        observed = by_id.get(source_id)
        require(type(observed) is dict, f"current Rada source absent from report: {source_id}")
        raw = current_payloads[source_id]
        raw_sha = sha256(raw)
        for field in ("source_family", "modality", "evidence_status", "declared_capacity_bytes"):
            require(
                type(observed.get(field)) is type(expected.get(field))
                and observed.get(field) == expected.get(field),
                f"current Rada report field drift: {source_id}:{field}",
            )
        stable_origin = expected.get("stable_origin_id")
        stable_object = expected.get("stable_object_id")
        require(type(stable_origin) is str and stable_origin, "Rada stable origin missing")
        require(type(stable_object) is str and stable_object, "Rada stable object missing")
        require(
            observed.get("stable_origin_id_sha256") == sha256(stable_origin.encode("utf-8")),
            f"current Rada stable origin drift: {source_id}",
        )
        require(
            observed.get("stable_object_id_sha256") == sha256(stable_object.encode("utf-8")),
            f"current Rada stable object drift: {source_id}",
        )
        require(observed.get("verified_raw_bytes") == len(raw), f"current Rada raw byte drift: {source_id}")
        require(observed.get("verified_raw_sha256") == raw_sha, f"current Rada raw SHA drift: {source_id}")
        require(observed.get("comparison_policy") == "DATA232_GENERIC_FROM_RAW", f"current Rada comparison policy drift: {source_id}")
        require(observed.get("comparison_payload_bytes") == len(raw), f"current Rada comparison byte drift: {source_id}")
        require(observed.get("comparison_payload_sha256") == raw_sha, f"current Rada comparison SHA drift: {source_id}")
    return by_id


def survivor_authority(
    helper: ModuleType,
    report: Mapping[str, Any],
    by_id: Mapping[str, Mapping[str, Any]],
    current_source_ids: set[str],
    replacement_proof: Mapping[str, Any],
) -> dict[str, Any]:
    selection = helper.v9_semantics._derive_survivors(report)
    selection_core = dict(selection)
    selection_id = selection_core.pop("survivor_authority_sha256", None)
    require(
        is_hex(selection_id, 64) and selection_id == sha256(canonical(selection_core)),
        "selection projection self-hash mismatch",
    )
    require(
        selection.get("pre_dedup_source_object_count") == EXPECTED_COMBINED_OBJECTS,
        "selection pre-dedup object count drift",
    )
    require(
        selection.get("pre_dedup_declared_capacity_bytes")
        == EXPECTED_COMBINED_DECLARED_BYTES,
        "selection pre-dedup bytes drift",
    )
    require(
        selection.get("post_dedup_declared_capacity_bytes")
        == report["terminal_candidates"]["conservative_unique_capacity_bytes_after"],
        "selection post-dedup bytes drift",
    )
    require(
        selection.get("duplicate_discount_bytes")
        == report["terminal_candidates"]["duplicate_discount_bytes"],
        "selection duplicate discount drift",
    )
    survivor_ids = selection.get("survivor_source_ids")
    require(
        type(survivor_ids) is list
        and all(type(value) is str and value for value in survivor_ids)
        and len(survivor_ids) == len(set(survivor_ids)),
        "selection survivor IDs invalid",
    )
    require(all(value in by_id for value in survivor_ids), "selection references unknown source")
    current_survivors = sorted(set(survivor_ids) & current_source_ids)
    current_removed = sorted(current_source_ids - set(current_survivors))
    require(
        len(current_survivors) + len(current_removed) == EXPECTED_CURRENT_OBJECTS,
        "current Rada survivor partition drift",
    )

    inventory: list[dict[str, Any]] = []
    survivor_bytes = 0
    for source_id in current_survivors:
        row = by_id[source_id]
        declared = row.get("declared_capacity_bytes")
        require(type(declared) is int and declared > 0, "Rada survivor declared bytes invalid")
        verified_sha = row.get("verified_raw_sha256")
        require(is_hex(verified_sha, 64), "Rada survivor raw SHA invalid")
        inventory.append(
            {
                "source_id": source_id,
                "declared_capacity_bytes": declared,
                "verified_raw_sha256": verified_sha,
                "stable_origin_id_sha256": row.get("stable_origin_id_sha256"),
                "stable_object_id_sha256": row.get("stable_object_id_sha256"),
            }
        )
        survivor_bytes += declared

    terminal = report["terminal_candidates"]
    core = {
        "schema_version": SURVIVOR_SCHEMA,
        "matcher_report_sha256": report.get("report_sha256"),
        "selection_projection_schema": selection.get("schema_version"),
        "selection_projection_sha256": selection_id,
        "replacement_proof": dict(replacement_proof),
        "pre_dedup_source_object_count": EXPECTED_COMBINED_OBJECTS,
        "pre_dedup_declared_capacity_bytes": EXPECTED_COMBINED_DECLARED_BYTES,
        "post_dedup_survivor_source_object_count": selection.get(
            "post_dedup_survivor_source_object_count"
        ),
        "post_dedup_declared_capacity_bytes": selection.get(
            "post_dedup_declared_capacity_bytes"
        ),
        "duplicate_discount_bytes": selection.get("duplicate_discount_bytes"),
        "duplicate_cluster_count": selection.get("duplicate_cluster_count"),
        "current_rada_input_source_object_count": EXPECTED_CURRENT_OBJECTS,
        "current_rada_input_payload_bytes": EXPECTED_CURRENT_PAYLOAD_BYTES,
        "current_rada_survivor_source_object_count": len(current_survivors),
        "current_rada_survivor_declared_capacity_bytes": survivor_bytes,
        "current_rada_discounted_source_object_count": len(current_removed),
        "current_rada_survivor_inventory": inventory,
        "current_rada_discounted_source_ids": current_removed,
        "terminal_conservative_unique_capacity_bytes_after": terminal.get(
            "conservative_unique_capacity_bytes_after"
        ),
        "canonical_capacity_credited": 0,
        "training_authorized_bytes": 0,
        "authorized_unique_loss_positions": 0,
        "authorized_optimized_target_exposure": 0,
        "tokenizer_fit_authorized": False,
        "training_executed": False,
        "learned_weights_created": False,
        "final_test_outcomes_read": False,
        "paid_compute_used": False,
        "scale_promotion_authorized": False,
        "rights_recheck_for_training_required": True,
        "next_gate": "CURRENT_RADA_RIGHTS_PROVENANCE_RECHECK_FOR_TRAINING",
    }
    return {**core, "survivor_authority_sha256": sha256(canonical(core))}


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_bytes(canonical(dict(value)) + b"\n")


def execute(args: argparse.Namespace) -> None:
    verify_product_parent(args.expected_execution_head)
    helper = load_helper(args.nbu_helper_root)
    require(
        git("rev-parse", "HEAD", cwd=args.v7_root) == V7_HEAD,
        "historical V7 HEAD drift",
    )
    current_sources, current_payloads, projection_receipt = validate_current_projection(
        args.candidate_jsonl
    )
    config = load_json(
        args.nbu_helper_root / "configs/data/next100_065f_global_dedup_v8.json"
    )
    matcher, base_inventory, base_payloads, removal = (
        helper._reconstruct_v8_with_historical_namespace(
            v7_root=args.v7_root,
            bulk_workspace=args.bulk_workspace,
            config=config,
        )
    )
    helper._verify_removal_proof(removal)
    inventory, payloads, replacement_proof = compose_current_graph(
        helper,
        base_inventory,
        base_payloads,
        current_sources,
        current_payloads,
    )
    report, _elapsed = helper._execute_indexed_matcher(
        matcher,
        inventory,
        payloads,
        max_candidate_pairs=args.max_candidate_pairs,
        max_index_postings=args.max_index_postings,
        max_pair_expansions=args.max_pair_expansions,
    )
    by_id = verify_report_current_rada(report, current_sources, current_payloads)
    authority = survivor_authority(
        helper,
        report,
        by_id,
        set(current_payloads),
        replacement_proof,
    )
    evidence_core = {
        "schema_version": EVIDENCE_SCHEMA,
        "execution_head_sha": args.expected_execution_head,
        "parent_product_head_sha": PARENT_HEAD,
        "parent_replay_authority_identity_sha256": PARENT_AUTHORITY_ID,
        "nbu_helper_head_sha": NBU_HELPER_HEAD,
        "nbu_helper_blob_sha1": NBU_HELPER_BLOB,
        "v7_head_sha": V7_HEAD,
        "projection_receipt_identity_sha256": projection_receipt.get(
            "receipt_identity_sha256"
        ),
        "candidate_jsonl_sha256": EXPECTED_CURRENT_JSONL_SHA256,
        "candidate_source_object_count": EXPECTED_CURRENT_OBJECTS,
        "candidate_payload_bytes": EXPECTED_CURRENT_PAYLOAD_BYTES,
        "replacement_proof": replacement_proof,
        "combined_source_object_count": EXPECTED_COMBINED_OBJECTS,
        "combined_declared_capacity_bytes": EXPECTED_COMBINED_DECLARED_BYTES,
        "matcher_report_sha256": report.get("report_sha256"),
        "survivor_authority_sha256": authority["survivor_authority_sha256"],
        "current_rada_survivor_source_object_count": authority[
            "current_rada_survivor_source_object_count"
        ],
        "current_rada_survivor_declared_capacity_bytes": authority[
            "current_rada_survivor_declared_capacity_bytes"
        ],
        "rights_scope": "ARTIFACT_RETENTION_AND_REPRODUCIBILITY_ONLY",
        "rights_recheck_for_training_required": True,
        "raw_text_persisted": False,
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
        "scale_promotion_authorized": False,
        "next_gate": "CURRENT_RADA_RIGHTS_PROVENANCE_RECHECK_FOR_TRAINING",
    }
    evidence = {
        **evidence_core,
        "evidence_identity_sha256": sha256(canonical(evidence_core)),
    }
    require(not args.output_dir.exists(), "output directory already exists")
    args.output_dir.mkdir(parents=True)
    write_json(args.output_dir / "dedup-report.json", report)
    write_json(args.output_dir / "rada-survivor-authority.json", authority)
    write_json(args.output_dir / "execution-evidence.json", evidence)
    print("CURRENT_RADA_GLOBAL_DEDUP=PASS")
    print("REPORT_SHA256=" + str(report.get("report_sha256")))
    print("SURVIVOR_AUTHORITY_SHA256=" + authority["survivor_authority_sha256"])
    print(
        "CURRENT_RADA_SURVIVOR_OBJECTS="
        + str(authority["current_rada_survivor_source_object_count"])
    )
    print(
        "CURRENT_RADA_SURVIVOR_DECLARED_BYTES="
        + str(authority["current_rada_survivor_declared_capacity_bytes"])
    )
    print("CAPACITY_CREDIT=0")
    print("RIGHTS_RECHECK_REQUIRED=true")


def compare(args: argparse.Namespace) -> None:
    names = (
        "dedup-report.json",
        "rada-survivor-authority.json",
        "execution-evidence.json",
    )
    hashes: dict[str, str] = {}
    for name in names:
        a = (args.output_a / name).read_bytes()
        b = (args.output_b / name).read_bytes()
        require(a == b, f"two-clean output differs: {name}")
        hashes[name] = sha256(a)

    evidence = load_json(args.output_a / "execution-evidence.json")
    survivor = load_json(args.output_a / "rada-survivor-authority.json")
    report = load_json(args.output_a / "dedup-report.json")
    require(
        report.get("report_sha256") == evidence.get("matcher_report_sha256"),
        "two-clean report/evidence identity mismatch",
    )
    require(
        survivor.get("survivor_authority_sha256")
        == evidence.get("survivor_authority_sha256"),
        "two-clean survivor/evidence identity mismatch",
    )
    require(evidence.get("canonical_capacity_credited") == 0, "two-clean capacity widened")
    require(evidence.get("training_authorized_bytes") == 0, "two-clean training widened")
    require(evidence.get("tokenizer_fit_authorized") is False, "two-clean tokenizer widened")
    require(evidence.get("rights_recheck_for_training_required") is True, "two-clean rights gate lost")

    core = {
        "schema_version": TWO_CLEAN_SCHEMA,
        "execution_head_sha": evidence["execution_head_sha"],
        "fresh_process_count": 2,
        "byte_identical_outputs": True,
        "output_file_sha256": hashes,
        "matcher_report_sha256": evidence["matcher_report_sha256"],
        "survivor_authority_sha256": evidence["survivor_authority_sha256"],
        "current_rada_survivor_source_object_count": evidence[
            "current_rada_survivor_source_object_count"
        ],
        "current_rada_survivor_declared_capacity_bytes": evidence[
            "current_rada_survivor_declared_capacity_bytes"
        ],
        "canonical_capacity_credited": 0,
        "training_authorized_bytes": 0,
        "tokenizer_fit_authorized": False,
        "rights_recheck_for_training_required": True,
        "next_gate": "CURRENT_RADA_RIGHTS_PROVENANCE_RECHECK_FOR_TRAINING",
    }
    proof = {**core, "two_clean_identity_sha256": sha256(canonical(core))}
    require(not args.proof.exists(), "two-clean proof already exists")
    write_json(args.proof, proof)
    print("CURRENT_RADA_TWO_CLEAN_GLOBAL_DEDUP=PASS")
    print("TWO_CLEAN_IDENTITY_SHA256=" + proof["two_clean_identity_sha256"])


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(allow_abbrev=False)
    sub = result.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", allow_abbrev=False)
    run.add_argument("--candidate-jsonl", type=Path, required=True)
    run.add_argument("--nbu-helper-root", type=Path, required=True)
    run.add_argument("--v7-root", type=Path, required=True)
    run.add_argument("--bulk-workspace", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--expected-execution-head", required=True)
    run.add_argument("--max-candidate-pairs", type=int, default=25_000_000)
    run.add_argument("--max-index-postings", type=int, default=100_000_000)
    run.add_argument("--max-pair-expansions", type=int, default=100_000_000)

    cmp_parser = sub.add_parser("compare", allow_abbrev=False)
    cmp_parser.add_argument("--output-a", type=Path, required=True)
    cmp_parser.add_argument("--output-b", type=Path, required=True)
    cmp_parser.add_argument("--proof", type=Path, required=True)
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        if args.command == "run":
            execute(args)
        else:
            compare(args)
    except (
        RadaCurrentGlobalDedupError,
        rada.RadaCurrentSnapshotDedupAdapterError,
        OSError,
        ValueError,
    ) as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
