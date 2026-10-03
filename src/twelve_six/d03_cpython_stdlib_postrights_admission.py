"""Fail-closed post-execution source-rights admission for CPython stdlib."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

POLICY_PATH = "configs/data/d03_cpython_stdlib_source_rights_v1.json"
EXECUTION_REPORT_PATH = "reports/d03/cpython_stdlib_terminal_execution_v1.json"
MATERIALIZER_PATH = "tools/materialize_d03_cpython_stdlib_v1.py"
EXPECTED_POLICY_IDENTITY_SHA256 = (
    "f4436690d6ea91b17fd0a5a7f110162bc051976eb6083b71c39c1ef274078346"
)
EXECUTION_REPORT_BLOB_SHA1 = "bd15575538909583006934c824909901ad6aeb6b"
MATERIALIZER_BLOB_SHA1 = "7eb8090df493f04b298e7d1d178e92d49d86cfe3"
UPSTREAM_REPOSITORY = "python/cpython"
UPSTREAM_COMMIT = "23180c50082fe98784c78511b335d7274ed87fb7"
UPSTREAM_TREE = "8853f9fbbf23d2a8c8b7afc9ecab242a482125f2"
ROOT_LICENSE_BLOB_SHA1 = "20cf39097c68baa17cc566b64e76d34ebf034044"
TREE_URL = f"https://api.github.com/repos/{UPSTREAM_REPOSITORY}/git/trees/{UPSTREAM_TREE}?recursive=1"
EXPECTED_TREE_ENTRY_COUNT = 6_710
EXPECTED_TREE_BLOB_COUNT = 6_232
EXPECTED_TREE_TREE_COUNT = 478
EXPECTED_TREE_COMMIT_COUNT = 0
EXPECTED_TREE_SEMANTICS_BYTES = 929_075
EXPECTED_TREE_BLOB_MAP_BYTES = 611_238
EXPECTED_TREE_SEMANTICS_IDENTITY_SHA256 = (
    "fcd8a90eeb7ba5816e360f7c336b9fe8d6d56adebb51fbb4cecb13e71b6b3d1c"
)
EXPECTED_TREE_BLOB_MAP_IDENTITY_SHA256 = (
    "85316aa5932aae00757d3ee9838201a17794400fdde656ac2a21085f5fcc1286"
)
FAMILY_ID = "code.python.cpython"
PRODUCT_HEAD_SHA = "c5bdf8527c6117f7e41693d84aad4295f69f78fb"
EXECUTION_HEAD_SHA = "e339007b7a295f2262f927819778fd0d22b5a474"
EXECUTION_RUN_ID = 34563978371
HISTORICAL_REPORT_IDENTITY_SHA256 = (
    "1ae69b91ad0eb2e99592563814b054cfe16b3f3800604d9dee54e6184a89b75a"
)
HISTORICAL_INVENTORY_IDENTITY_SHA256 = (
    "e786370430e73b674ed5b4d09829cddffa6a943fc550b5316f2cb1da93bca654"
)
HISTORICAL_SELECTED_OBJECTS = 52
HISTORICAL_SELECTED_BYTES = 1_199_986

ROW_KEYS = {
    "record_id", "source_path", "family", "modality", "git_blob_sha1",
    "payload_sha256", "payload_bytes", "text", "training_eligible",
    "evaluation_eligible",
}
TRUTH_BOUNDARY = {
    "canonical_capacity_credit_bytes": 0,
    "training_authorized_bytes": 0,
    "authorized_unique_loss_positions": 0,
    "authorized_optimized_target_exposure": 0,
    "tokenizer_fit_authorized": False,
    "optimizer_updates": 0,
    "model_training_executed": False,
    "learned_weights_created": False,
    "final_test_payload_accessed": False,
    "final_test_outcomes_read": False,
    "paid_compute_used": False,
    "foreign_pretrained_weights_used": False,
    "external_llm_or_api_used_for_data_or_intelligence": False,
}
DOWNSTREAM_REQUIRED = [
    "current_global_exact_near_lineage_dedup",
    "reserved_evaluation_decontamination",
    "canonical_quality_privacy",
    "family_balance_caps",
    "cluster_safe_split",
    "deterministic_packing",
    "two_clean_build_reproducibility",
    "positive_unique_loss_accounting",
    "positive_training_authority",
]


class CPythonRightsAdmissionError(ValueError):
    """Fail-closed rights admission error."""


@dataclass(frozen=True)
class PinnedTreeAuthority:
    """Immutable complete recursive Git-tree snapshot used by canonical admission."""

    entries: tuple[tuple[str, str, str, str, int | None], ...]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CPythonRightsAdmissionError(message)


def canonical_json_bytes(value: Any) -> bytes:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return (text + "\n").encode()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_blob_sha1(data: bytes) -> str:
    prefix = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(prefix + data, usedforsecurity=False).hexdigest()


def _exact_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _validated_policy_snapshot(policy: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    _require(type(policy) is dict, "rights policy root must be an object")
    _require(
        policy.get("policy_identity_sha256") == EXPECTED_POLICY_IDENTITY_SHA256,
        "rights policy identity drift",
    )
    raw_policy = dict(policy)
    raw_policy.pop("policy_identity_sha256")
    try:
        canonical = canonical_json_bytes(raw_policy)
    except (TypeError, ValueError) as exc:
        raise CPythonRightsAdmissionError("rights policy identity drift") from exc
    identity = sha256_bytes(canonical)
    _require(identity == EXPECTED_POLICY_IDENTITY_SHA256, "rights policy identity drift")
    snapshot = json.loads(canonical)
    _require(type(snapshot) is dict, "rights policy root must be an object")
    snapshot["policy_identity_sha256"] = identity
    return snapshot, identity


def load_and_validate_policy(path: str | Path = POLICY_PATH) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CPythonRightsAdmissionError("cannot read rights policy") from exc
    _require(type(payload) is dict, "rights policy root must be an object")
    required = {"schema_version", "authority_id", "scope", "policy", "truth_boundary"}
    _require(set(payload) == required, "rights policy schema drift")
    identity = sha256_bytes(canonical_json_bytes(payload))
    _require(identity == EXPECTED_POLICY_IDENTITY_SHA256, "rights policy identity drift")
    scope = payload["scope"]
    _require(type(scope) is dict, "rights policy scope invalid")
    exact_scope = {
        "repository": UPSTREAM_REPOSITORY,
        "commit": UPSTREAM_COMMIT,
        "tree": UPSTREAM_TREE,
        "root_license_blob_sha1": ROOT_LICENSE_BLOB_SHA1,
        "family_id": FAMILY_ID,
        "product_pr": 903,
        "product_head_sha": PRODUCT_HEAD_SHA,
        "execution_head_sha": EXECUTION_HEAD_SHA,
        "execution_run_id": EXECUTION_RUN_ID,
        "execution_report_identity_sha256": HISTORICAL_REPORT_IDENTITY_SHA256,
        "inventory_identity_sha256": HISTORICAL_INVENTORY_IDENTITY_SHA256,
        "selected_objects": HISTORICAL_SELECTED_OBJECTS,
        "selected_bytes": HISTORICAL_SELECTED_BYTES,
        "complete_tree_entry_count": EXPECTED_TREE_ENTRY_COUNT,
        "complete_tree_blob_count": EXPECTED_TREE_BLOB_COUNT,
        "complete_tree_tree_count": EXPECTED_TREE_TREE_COUNT,
        "complete_tree_commit_count": EXPECTED_TREE_COMMIT_COUNT,
        "complete_tree_semantics_identity_sha256": EXPECTED_TREE_SEMANTICS_IDENTITY_SHA256,
        "complete_tree_blob_map_identity_sha256": EXPECTED_TREE_BLOB_MAP_IDENTITY_SHA256,
    }
    _require(scope == exact_scope, "rights policy scope drift")
    rule = payload["policy"]
    _require(type(rule) is dict, "rights policy rule invalid")
    _require(rule.get("root_default_license") == "PSF-2.0", "root license drift")
    _require(rule.get("blanket_incorporated_software_admission") is False, "blanket admission forbidden")
    _require(rule.get("evidence_window_chars") == 12_000, "evidence window drift")
    _require(
        payload["truth_boundary"] == {"source_admission_only": True, **TRUTH_BOUNDARY},
        "rights policy truth boundary drift",
    )
    payload["policy_identity_sha256"] = identity
    return payload


def _bound_bytes(root: Path, relpath: str, expected_blob: str) -> bytes:
    path = (root / relpath).resolve()
    _require(path.is_relative_to(root), f"{relpath} escaped repository root")
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise CPythonRightsAdmissionError(f"cannot read {relpath}") from exc
    _require(git_blob_sha1(data) == expected_blob, f"{relpath} blob drift")
    return data


def validate_repository_bindings(repo_root: str | Path = ".") -> None:
    root = Path(repo_root).resolve()
    report = json.loads(
        _bound_bytes(root, EXECUTION_REPORT_PATH, EXECUTION_REPORT_BLOB_SHA1)
    )
    materializer = _bound_bytes(root, MATERIALIZER_PATH, MATERIALIZER_BLOB_SHA1).decode()
    receipt = report.get("receipt")
    _require(type(receipt) is dict, "historical receipt missing")
    _require(
        receipt.get("source") == {
            "repository": UPSTREAM_REPOSITORY,
            "commit": UPSTREAM_COMMIT,
            "tree": UPSTREAM_TREE,
            "license_blob_sha1": ROOT_LICENSE_BLOB_SHA1,
            "family_id": FAMILY_ID,
        },
        "historical source drift",
    )
    selection = receipt.get("selection", {})
    _require(selection.get("selected_objects") == 52, "historical object count drift")
    _require(selection.get("selected_bytes") == 1_199_986, "historical byte count drift")
    reproducibility = receipt.get("reproducibility", {})
    _require(
        reproducibility.get("inventory_identity_sha256")
        == HISTORICAL_INVENTORY_IDENTITY_SHA256,
        "historical inventory identity drift",
    )
    _require(
        reproducibility.get("materializer_report_identity_sha256")
        == HISTORICAL_REPORT_IDENTITY_SHA256,
        "historical report identity drift",
    )
    rights = receipt.get("rights_retest", {})
    _require(rights.get("blanket_admission") is False, "historical blanket admission drift")
    _require(rights.get("selected_objects_rights_admitted") is False, "historical rights drift")
    _require(
        rights.get("conclusion") == "ZERO_CREDIT_PENDING_BINDING_RIGHTS_AUTHORITY",
        "historical rights conclusion drift",
    )
    _require(
        rights.get("objects_with_copyright_or_license_notice_in_first_12k") == 25,
        "historical notice count drift",
    )
    for literal in (
        f'UPSTREAM_COMMIT = "{UPSTREAM_COMMIT}"',
        f'UPSTREAM_TREE = "{UPSTREAM_TREE}"',
        f'LICENSE_BLOB_SHA1 = "{ROOT_LICENSE_BLOB_SHA1}"',
        '"selected_objects_rights_admitted": False',
        '"file_level_incorporated_license_retest_required": True',
    ):
        _require(literal in materializer, f"materializer contract drift: {literal}")


def _validated_complete_tree_payload(
    payload: Mapping[str, Any],
) -> tuple[PinnedTreeAuthority, dict[str, str]]:
    _require(type(payload) is dict, "tree response must be an object")
    _require(payload.get("sha") == UPSTREAM_TREE, "tree response identity drift")
    _require(payload.get("truncated") is False, "tree response must be complete")
    entries = payload.get("tree")
    _require(type(entries) is list, "tree entries missing")
    _require(len(entries) == EXPECTED_TREE_ENTRY_COUNT, "complete tree entry count drift")

    canonical_entries: list[dict[str, Any]] = []
    snapshots: list[tuple[str, str, str, str, int | None]] = []
    seen_paths: set[str] = set()
    counts = {"blob": 0, "tree": 0, "commit": 0}

    for entry in entries:
        _require(type(entry) is dict, "tree entry invalid")
        path = entry.get("path")
        mode = entry.get("mode")
        kind = entry.get("type")
        sha = entry.get("sha")
        _require(
            type(path) is str
            and path
            and path.isascii()
            and "\x00" not in path
            and not path.startswith("/")
            and ".." not in Path(path).parts,
            "tree path invalid",
        )
        _require(path not in seen_paths, "duplicate tree path")
        _require(type(mode) is str and mode, "tree mode invalid")
        _require(kind in counts, "tree object type invalid")
        _require(
            type(sha) is str
            and len(sha) == 40
            and sha == sha.lower()
            and all(ch in "0123456789abcdef" for ch in sha),
            "tree object sha invalid",
        )
        size: int | None = None
        if "size" in entry:
            candidate_size = entry["size"]
            _require(
                _exact_int(candidate_size) and candidate_size >= 0,
                "tree object size invalid",
            )
            size = candidate_size

        canonical: dict[str, Any] = {
            "mode": mode,
            "path": path,
            "sha": sha,
            "type": kind,
        }
        if size is not None:
            canonical["size"] = size
        canonical_entries.append(canonical)
        snapshots.append((path, mode, kind, sha, size))
        seen_paths.add(path)
        counts[kind] += 1

    paired = sorted(
        zip(canonical_entries, snapshots, strict=True),
        key=lambda item: item[0]["path"],
    )
    canonical_entries = [item[0] for item in paired]
    snapshots = [item[1] for item in paired]

    _require(counts["blob"] == EXPECTED_TREE_BLOB_COUNT, "complete tree blob count drift")
    _require(counts["tree"] == EXPECTED_TREE_TREE_COUNT, "complete tree subtree count drift")
    _require(counts["commit"] == EXPECTED_TREE_COMMIT_COUNT, "complete tree commit count drift")

    semantics = {
        "entries": canonical_entries,
        "sha": UPSTREAM_TREE,
        "truncated": False,
    }
    semantics_bytes = canonical_json_bytes(semantics)
    _require(
        len(semantics_bytes) == EXPECTED_TREE_SEMANTICS_BYTES,
        "complete tree semantics byte count drift",
    )
    _require(
        sha256_bytes(semantics_bytes) == EXPECTED_TREE_SEMANTICS_IDENTITY_SHA256,
        "complete tree semantics identity drift",
    )

    blob_projection = [
        {"path": item["path"], "sha": item["sha"]}
        for item in canonical_entries
        if item["type"] == "blob"
    ]
    blob_bytes = canonical_json_bytes(blob_projection)
    _require(
        len(blob_bytes) == EXPECTED_TREE_BLOB_MAP_BYTES,
        "complete tree blob-map byte count drift",
    )
    _require(
        sha256_bytes(blob_bytes) == EXPECTED_TREE_BLOB_MAP_IDENTITY_SHA256,
        "complete tree blob-map identity drift",
    )

    root_license = next(
        (item for item in canonical_entries if item["path"] == "LICENSE"),
        None,
    )
    _require(
        root_license
        == {
            "mode": "100644",
            "path": "LICENSE",
            "sha": ROOT_LICENSE_BLOB_SHA1,
            "size": 13_804,
            "type": "blob",
        },
        "root LICENSE authority drift",
    )
    blobs = {
        item["path"]: item["sha"]
        for item in canonical_entries
        if item["type"] == "blob"
    }
    return PinnedTreeAuthority(tuple(snapshots)), blobs


def parse_pinned_tree_response(payload: Mapping[str, Any]) -> PinnedTreeAuthority:
    authority, _ = _validated_complete_tree_payload(payload)
    return authority


def _validated_tree_snapshot(
    authority: object,
) -> tuple[dict[str, str], dict[str, Any]]:
    _require(
        type(authority) is PinnedTreeAuthority,
        "authenticated complete pinned tree authority required",
    )
    reconstructed: list[dict[str, Any]] = []
    for item in authority.entries:
        _require(type(item) is tuple and len(item) == 5, "pinned tree authority snapshot invalid")
        path, mode, kind, sha, size = item
        entry: dict[str, Any] = {
            "path": path,
            "mode": mode,
            "type": kind,
            "sha": sha,
        }
        if size is not None:
            entry["size"] = size
        reconstructed.append(entry)

    reparsed, blobs = _validated_complete_tree_payload(
        {"sha": UPSTREAM_TREE, "truncated": False, "tree": reconstructed}
    )
    _require(reparsed == authority, "pinned tree authority snapshot drift")
    evidence = {
        "tree_sha": UPSTREAM_TREE,
        "entry_count": EXPECTED_TREE_ENTRY_COUNT,
        "blob_count": EXPECTED_TREE_BLOB_COUNT,
        "tree_count": EXPECTED_TREE_TREE_COUNT,
        "commit_count": EXPECTED_TREE_COMMIT_COUNT,
        "semantics_identity_sha256": EXPECTED_TREE_SEMANTICS_IDENTITY_SHA256,
        "blob_map_identity_sha256": EXPECTED_TREE_BLOB_MAP_IDENTITY_SHA256,
        "root_license_blob_sha1": ROOT_LICENSE_BLOB_SHA1,
    }
    return dict(blobs), evidence


def fetch_pinned_tree_authority(
    url: str = TREE_URL,
    *,
    timeout: float = 60.0,
) -> PinnedTreeAuthority:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "12-6-ai-cpython-rights/1.0 (+LOCAL_FREE research)",
            "Accept": "application/vnd.github+json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CPythonRightsAdmissionError("cannot read pinned complete tree authority") from exc
    return parse_pinned_tree_response(payload)

def _validate_row(row: Mapping[str, Any], tree_blobs: Mapping[str, str]) -> bytes:
    _require(type(row) is dict and set(row) == ROW_KEYS, "candidate row schema drift")
    path = row["source_path"]
    _require(type(path) is str and path.startswith("Lib/") and path.endswith(".py"), "source path invalid")
    _require(".." not in Path(path).parts, "source path traversal")
    _require(row["record_id"] == f"cpython:{path}", "record id drift")
    _require(row["family"] == FAMILY_ID and row["modality"] == "code", "candidate identity drift")
    _require(row["training_eligible"] is False, "candidate cannot self-authorize training")
    _require(row["evaluation_eligible"] is False, "candidate cannot self-authorize evaluation")
    text = row["text"]
    _require(type(text) is str, "candidate text invalid")
    raw = text.encode()
    size = row["payload_bytes"]
    _require(_exact_int(size) and size > 0, "payload bytes invalid")
    _require(len(raw) == size, "payload byte count drift")
    _require(sha256_bytes(raw) == row["payload_sha256"], "payload sha drift")
    _require(git_blob_sha1(raw) == row["git_blob_sha1"], "git blob sha drift")
    _require(tree_blobs.get(path) == row["git_blob_sha1"], "pinned tree binding drift")
    return raw


def candidate_inventory_identity(rows: Sequence[Mapping[str, Any]]) -> str:
    projection = [{k: v for k, v in row.items() if k != "text"} for row in rows]
    return sha256_bytes(canonical_json_bytes(projection))


def validate_historical_candidate(
    rows: Sequence[Mapping[str, Any]], tree_blobs: Mapping[str, str]
) -> None:
    _require(type(rows) is list, "candidate rows must be a list")
    _require(len(rows) == HISTORICAL_SELECTED_OBJECTS, "selected object count drift")
    seen_paths: set[str] = set()
    seen_payloads: set[str] = set()
    total = 0
    previous = ""
    for row in rows:
        raw = _validate_row(row, tree_blobs)
        path, digest = row["source_path"], row["payload_sha256"]
        _require(path > previous, "candidate path ordering drift")
        _require(path not in seen_paths and digest not in seen_payloads, "candidate duplicate")
        seen_paths.add(path)
        seen_payloads.add(digest)
        previous = path
        total += len(raw)
    _require(total == HISTORICAL_SELECTED_BYTES, "selected byte total drift")
    _require(
        candidate_inventory_identity(rows) == HISTORICAL_INVENTORY_IDENTITY_SHA256,
        "historical candidate inventory identity drift",
    )


def _ancestor_marker(path: str, tree_blobs: Mapping[str, str], names: Sequence[str]) -> str | None:
    tree_paths = {item.casefold(): item for item in tree_blobs}
    parent = Path(path).parent
    while str(parent) not in {".", ""}:
        prefix = parent.as_posix()
        for name in names:
            candidate = f"{prefix}/{name}".casefold()
            if candidate in tree_paths:
                return tree_paths[candidate]
        if prefix == "Lib":
            break
        parent = parent.parent
    return None


def classify_row(
    row: Mapping[str, Any], tree_blobs: Mapping[str, str], policy: Mapping[str, Any]
) -> dict[str, Any]:
    raw = _validate_row(row, tree_blobs)
    rule = policy["policy"]
    text = raw.decode().casefold()
    notice = next(
        (item for item in rule["direct_notice_markers"] if item.casefold() in text),
        None,
    )
    ancestor = _ancestor_marker(row["source_path"], tree_blobs, rule["ancestor_marker_names"])
    if notice is not None:
        decision, reason, admitted, license_id = (
            rule["direct_notice_decision"], "DIRECT_RIGHTS_NOTICE_PRESENT", False, None
        )
    elif ancestor is not None:
        decision, reason, admitted, license_id = (
            rule["ancestor_marker_decision"], "ANCESTOR_RIGHTS_MARKER_PRESENT", False, None
        )
    else:
        decision, reason, admitted, license_id = (
            rule["no_marker_decision"],
            "PINNED_ROOT_PSF2_DEFAULT_NO_LOCAL_OVERRIDE_OBSERVED",
            True,
            rule["root_default_license"],
        )
    return {
        "record_id": row["record_id"],
        "source_path": row["source_path"],
        "git_blob_sha1": row["git_blob_sha1"],
        "payload_sha256": row["payload_sha256"],
        "payload_bytes": row["payload_bytes"],
        "decision": decision,
        "reason": reason,
        "admitted": admitted,
        "license_id": license_id,
        "direct_notice_marker": notice,
        "ancestor_rights_marker_path": ancestor,
    }


def build_admission(
    rows: Sequence[Mapping[str, Any]],
    tree_blobs: Mapping[str, str] | PinnedTreeAuthority,
    policy: Mapping[str, Any],
    *,
    require_historical_identity: bool = True,
) -> dict[str, Any]:
    verified_policy, verified_policy_identity = _validated_policy_snapshot(policy)
    tree_evidence: dict[str, Any] | None = None
    if require_historical_identity:
        tree_snapshot, tree_evidence = _validated_tree_snapshot(tree_blobs)
        validate_historical_candidate(rows, tree_snapshot)
    else:
        _require(type(tree_blobs) is dict, "non-authoritative test tree must be an exact dict")
        tree_snapshot = dict(tree_blobs)
        _require(type(rows) is list and rows, "candidate rows must be non-empty")
        for row in rows:
            _validate_row(row, tree_snapshot)
    decisions = [classify_row(row, tree_snapshot, verified_policy) for row in rows]
    admitted = [item for item in decisions if item["admitted"]]
    held = [item for item in decisions if not item["admitted"]]
    reasons: dict[str, int] = {}
    for item in held:
        reasons[item["reason"]] = reasons.get(item["reason"], 0) + 1
    admitted_bytes = sum(item["payload_bytes"] for item in admitted)
    held_bytes = sum(item["payload_bytes"] for item in held)
    result: dict[str, Any] = {
        "schema_version": "12-6.d03-cpython-stdlib-postrights-source-admission.v1",
        "authority_id": "D03-CPYTHON-STDLIB-POSTRIGHTS-SOURCE-ADMISSION-V1",
        "status": (
            "SOURCE_ADMISSION_EXECUTED_PARTIAL_ZERO_CANONICAL_CREDIT"
            if require_historical_identity and admitted
            else "SOURCE_ADMISSION_EXECUTED_ZERO_ADMITTED"
            if require_historical_identity
            else "NON_AUTHORITATIVE_TEST_ONLY"
        ),
        "execution_profile": "LOCAL_FREE",
        "historical_lineage": (
            {
                "product_pr": 903,
                "product_head_sha": PRODUCT_HEAD_SHA,
                "execution_head_sha": EXECUTION_HEAD_SHA,
                "execution_run_id": EXECUTION_RUN_ID,
                "source_commit": UPSTREAM_COMMIT,
                "source_tree": UPSTREAM_TREE,
                "root_license_blob_sha1": ROOT_LICENSE_BLOB_SHA1,
                "historical_inventory_identity_sha256": HISTORICAL_INVENTORY_IDENTITY_SHA256,
            }
            if require_historical_identity
            else {"authority": "NON_AUTHORITATIVE_TEST_ONLY"}
        ),
        "rights_authority": {
            "policy_identity_sha256": verified_policy_identity,
            "root_default_license": "PSF-2.0",
            "blanket_incorporated_software_admission": False,
            "ambiguous_objects_fail_closed": True,
            "complete_pinned_tree_authenticated": require_historical_identity,
            "complete_tree": tree_evidence,
        },
        "candidate": {
            "inventory_identity_sha256": candidate_inventory_identity(rows),
            "selected_objects": len(rows),
            "selected_bytes": sum(item["payload_bytes"] for item in decisions),
        },
        "source_admission": {
            "admitted_objects": len(admitted),
            "admitted_bytes": admitted_bytes,
            "held_objects": len(held),
            "held_bytes": held_bytes,
            "held_reason_counts": dict(sorted(reasons.items())),
            "decisions_identity_sha256": sha256_bytes(canonical_json_bytes(decisions)),
        },
        "decisions": decisions,
        "truth_boundary": {
            "source_admitted_candidate_records": len(admitted),
            "source_admitted_candidate_bytes": admitted_bytes,
            **TRUTH_BOUNDARY,
        },
        "downstream_required": DOWNSTREAM_REQUIRED,
    }
    result["report_identity_sha256"] = sha256_bytes(canonical_json_bytes(result))
    _require(b'"text"' not in canonical_json_bytes(result), "source text leaked into evidence")
    return result


def _load_materializer(root: Path) -> Any:
    spec = importlib.util.spec_from_file_location("_cpython_materializer", root / MATERIALIZER_PATH)
    _require(spec is not None and spec.loader is not None, "cannot load materializer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def execute_live_twice(repo_root: str | Path = ".") -> dict[str, Any]:
    _require(sys.version_info >= (3, 16), "live execution requires CPython >=3.16 grammar")
    root = Path(repo_root).resolve()
    validate_repository_bindings(root)
    policy = load_and_validate_policy(root / POLICY_PATH)
    materializer = _load_materializer(root)
    materializer_tree = materializer.fetch_pinned_tree_blobs()
    tree_authority = fetch_pinned_tree_authority()
    tree_snapshot, _ = _validated_tree_snapshot(tree_authority)
    _require(
        materializer_tree == tree_snapshot,
        "materializer tree differs from authenticated complete tree authority",
    )
    archive_a, archive_b = materializer.download_archive(), materializer.download_archive()
    _require(archive_a == archive_b, "independent source acquisitions differ")
    rows_a, report_a = materializer.materialize_archive_bytes(
        archive_a, expected_blobs=tree_snapshot
    )
    rows_b, report_b = materializer.materialize_archive_bytes(
        archive_b, expected_blobs=tree_snapshot
    )
    _require(rows_a == rows_b and report_a == report_b, "independent materializations differ")
    _require(
        report_a.get("report_identity_sha256") == HISTORICAL_REPORT_IDENTITY_SHA256,
        "historical materializer identity no longer reproduces",
    )
    rights_a = build_admission(rows_a, tree_authority, policy)
    rights_b = build_admission(rows_b, tree_authority, policy)
    _require(rights_a == rights_b, "independent rights evaluations differ")
    result: dict[str, Any] = {
        "schema_version": "12-6.d03-cpython-stdlib-source-rights-execution.v1",
        "status": "REAL_LOCAL_FREE_RIGHTS_EXECUTION_REPRODUCIBLE",
        "execution_profile": "LOCAL_FREE",
        "source": {
            "archive_bytes": len(archive_a),
            "archive_sha256": sha256_bytes(archive_a),
            "two_acquisitions_byte_identical": True,
        },
        "materialization": {
            "inventory_identity_sha256": candidate_inventory_identity(rows_a),
            "report_identity_sha256": report_a["report_identity_sha256"],
            "two_materializations_byte_identical": True,
        },
        "rights_evaluation": rights_a,
        "proof": {
            "historical_repository_bindings_validated": True,
            "two_clean_rights_evaluations": True,
            "source_text_retained_in_execution_evidence": False,
        },
        "truth_boundary": rights_a["truth_boundary"],
    }
    result["execution_identity_sha256"] = sha256_bytes(canonical_json_bytes(result))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--live-twice", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _require(args.live_twice, "only exact two-clean live execution is supported by this CLI")
    result = execute_live_twice(args.repo_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json_bytes(result))


if __name__ == "__main__":
    main()
