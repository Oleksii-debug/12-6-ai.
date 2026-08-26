#!/usr/bin/env python3
"""Validate the DATA-300 corpus-v03 frozen executable build contract v2."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = (
    ROOT / "configs" / "data" / "data300_corpus_v03_frozen_build_contract_v2.json"
)

EXPECTED_SOURCE_IDS = [
    "external-real:en.standardebooks.manual.8-typography",
    "external-real:en.standardebooks.manual.9-metadata",
    "external-real:ua.rada.open-data.laws-texts.d23314",
    "external-real:code.encode.httpx._content",
    "external-real:code.psf.requests._internal_utils",
]
EXPECTED_GATES = [
    "G01_CONTRACT_IDENTITY",
    "G02_COMPONENT_TERMINALITY",
    "G03_SOURCE_INVENTORY",
    "G04_RIGHTS",
    "G05_QUALITY",
    "G06_PRIVACY",
    "G07_DEDUP",
    "G08_RESERVED_DECONTAMINATION",
    "G09_BALANCE_DIVERSITY",
    "G10_SELECTION_VALIDATION",
    "G11_FINAL_TEST_ISOLATION",
    "G12_UNIQUE_LOSS",
    "G13_NO_ARTIFICIAL_REPETITION",
    "G14_TWO_CLEAN_BUILDS",
    "G15_RELEASE_TRUTH",
]
TERMINAL_LOCKS = [
    "source_registry_text",
    "source_registry_code",
    "rights",
    "dedup",
    "balance",
    "final_test_reservation",
    "code_evaluation_reservation",
    "en_selection_validation",
    "code_selection_validation",
    "unique_loss",
]
EXPECTED_NONTERMINAL = {
    "DATA-287": "EXCLUDED_NO_CHECK_RUNS_AT_FREEZE",
    "DATA-288": "EXCLUDED_NONTERMINAL",
    "DATA-296": "EXCLUDED_NONTERMINAL",
    "DATA-297": "EXCLUDED_NONTERMINAL",
    "EVAL-290": "EXCLUDED_NONTERMINAL",
}


class ContractError(RuntimeError):
    """Raised when a hard DATA-300 invariant is violated."""


def fail(message: str) -> None:
    raise ContractError(message)


def canonical_bytes(obj: Any) -> bytes:
    return (
        json.dumps(
            obj,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        row = json.loads(raw)
        if not isinstance(row, dict):
            fail(f"{path}:{line_no}: JSONL row is not an object")
        rows.append(row)
    return rows


def contract_identity(contract: dict[str, Any]) -> str:
    payload = dict(contract)
    payload.pop("contract_identity_sha256", None)
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def validate_contract(contract: dict[str, Any]) -> None:
    require(
        contract.get("schema_version")
        == "12-6.data300-corpus-v03-frozen-build-contract.v2",
        "schema_version drift",
    )
    require(
        contract.get("repository") == "Oleksii-debug/12-6-ai.",
        "repository identity drift",
    )
    require(
        contract.get("execution_profile") == "LOCAL_FREE",
        "execution profile must be LOCAL_FREE",
    )
    require(
        contract.get("contract_state") == "FROZEN_EXECUTABLE_CONTRACT",
        "contract state drift",
    )
    require(
        contract.get("corpus_state") == "NOT_BUILT_NOT_FROZEN_NOT_TERMINAL",
        "contract must not declare the corpus frozen or terminal",
    )
    expected_identity = contract_identity(contract)
    require(
        contract.get("contract_identity_sha256") == expected_identity,
        "contract self-hash mismatch",
    )

    evidence = contract["evidence_cutoff"]
    require(evidence["kind"] == "EXACT_HEAD_VECTOR", "cutoff must be exact-head vector")
    observed = evidence["source_successor_observed"]
    require(set(observed) == set(EXPECTED_NONTERMINAL), "nonterminal vector drift")
    for worker, status in EXPECTED_NONTERMINAL.items():
        require(observed[worker]["status"] == status, f"{worker} status drift")
        require(
            observed[worker]["status"].startswith("EXCLUDED_"),
            f"{worker} must remain non-authoritative",
        )

    locks = contract["terminal_component_lock"]
    for key in TERMINAL_LOCKS:
        item = locks[key]
        require(
            item.get("dedicated_workflow_conclusion") == "success",
            f"{key} lacks terminal dedicated success",
        )

    require(locks["rights"]["worker"] == "DATA-293", "rights successor drift")
    require(locks["quality_policy"]["worker"] == "DATA-32", "quality policy drift")
    require(locks["privacy_policy"]["worker"] == "DATA-33", "privacy policy drift")
    require(
        locks["quality_policy"]["successor_data296_terminal"] is False,
        "red DATA-296 must not be promoted",
    )
    require(
        locks["privacy_policy"]["successor_data297_terminal"] is False,
        "red DATA-297 must not be promoted",
    )
    require(locks["dedup"]["worker"] == "DATA-298", "dedup successor drift")
    require(locks["balance"]["worker"] == "DATA-295", "balance successor drift")
    require(locks["unique_loss"]["worker"] == "DATA-294", "unique-loss successor drift")

    inventory = contract["exact_training_candidate_inventory"]
    sources = inventory["sources"]
    source_ids = [row["source_id"] for row in sources]
    require(source_ids == EXPECTED_SOURCE_IDS, "exact source inventory/order drift")
    require(inventory["source_count"] == 5, "source count drift")
    require(inventory["independent_family_count"] == 4, "family count drift")
    require(inventory["admitted_source_bytes"] == 183061, "byte total drift")
    require(
        sum(row["normalized_bytes"] for row in sources) == 183061,
        "source byte accounting mismatch",
    )
    require(
        inventory["by_stratum_families"] == {"uk": 1, "en": 1, "code": 2},
        "stratum family-count drift",
    )
    require(
        all(row["training_rights"] == "ALLOWED" for row in sources),
        "training inventory contains non-ALLOWED object",
    )

    split = contract["split_contract"]
    require(
        split["global"]["exact_content_overlap_allowed"] is False,
        "cross-split exact content overlap must be forbidden",
    )
    require(
        split["global"]["dedup_cluster_straddling_allowed"] is False,
        "dedup clusters may not straddle splits",
    )
    require(split["train"]["may_update_model"] is True, "train must update model")
    require(
        split["selection_validation"]["may_update_model"] is False,
        "selection-validation may not update model",
    )
    require(
        split["selection_validation"]["may_fit_tokenizer"] is False,
        "selection-validation may not fit tokenizer",
    )
    require(
        split["final_test"]["may_be_read_before_selection_lock"] is False,
        "final-test must remain unread before selection lock",
    )
    for forbidden in (
        "may_fit_tokenizer",
        "may_update_model",
        "may_select_checkpoint",
        "may_select_hyperparameters",
    ):
        require(split["final_test"][forbidden] is False, f"final-test {forbidden}")

    repetition = contract["artificial_repetition"]
    require(not any(repetition.values()), "all artificial-repetition switches must be false")

    determinism = contract["build_determinism"]
    require(determinism["clean_build_count"] == 2, "exactly two clean builds required")
    require(
        determinism["complete_tree_byte_identity_required"] is True,
        "complete-tree byte identity must be required",
    )
    require(
        determinism["shared_mutable_cache_allowed"] is False,
        "clean builds may not share mutable cache",
    )

    gate_ids = [row["id"] for row in contract["release_gates"]]
    require(gate_ids == EXPECTED_GATES, "release-gate vector drift")
    require(
        all(row["severity"] == "HARD" for row in contract["release_gates"]),
        "all DATA-300 release gates must be HARD",
    )

    expected_files = contract["wave3_expected_artifact_structure"][
        "required_relative_files"
    ]
    require(len(expected_files) == len(set(expected_files)), "duplicate artifact path")
    require(
        "manifests/final-test-reservation.jsonl" in expected_files,
        "final-test reservation manifest missing",
    )
    require(
        "ledgers/train-unique-loss.jsonl" in expected_files,
        "unique-loss ledger missing",
    )
    require(
        contract["current_candidate_status"]["release_ready"] is False,
        "current candidate must remain blocked",
    )
    require(
        contract["current_candidate_status"]["corpus_freeze_authorized"] is False,
        "corpus freeze must remain unauthorized",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_map(root: Path) -> dict[str, str]:
    require(root.is_dir(), f"missing build root: {root}")
    result: dict[str, str] = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        result[rel] = sha256_file(path)
    return result


def compare_clean_builds(a: Path, b: Path) -> None:
    left = tree_map(a)
    right = tree_map(b)
    require(left.keys() == right.keys(), "clean-build relative path sets differ")
    differing = [path for path in left if left[path] != right[path]]
    require(not differing, f"clean builds differ: {differing[:8]}")


def require_source_ids(value: Iterable[str], expected: set[str], label: str) -> None:
    actual = set(value)
    require(actual == expected, f"{label} source coverage drift")


def split_keys(
    rows: list[dict[str, Any]], label: str
) -> tuple[set[str], set[str], set[str]]:
    record_ids: set[str] = set()
    content_hashes: set[str] = set()
    cluster_ids: set[str] = set()
    for row in rows:
        for key in ("record_id", "content_sha256", "dedup_cluster_id"):
            require(bool(row.get(key)), f"{label} row missing {key}")
        require(row["record_id"] not in record_ids, f"{label} duplicate record_id")
        record_ids.add(row["record_id"])
        content_hashes.add(row["content_sha256"])
        cluster_ids.add(row["dedup_cluster_id"])
    return record_ids, content_hashes, cluster_ids


def validate_wave3(root: Path, contract: dict[str, Any]) -> None:
    validate_contract(contract)
    required = contract["wave3_expected_artifact_structure"]["required_relative_files"]
    missing = [rel for rel in required if not (root / rel).is_file()]
    require(not missing, f"missing Wave-3 artifacts: {missing}")

    expected_ids = set(EXPECTED_SOURCE_IDS)
    lock = load_json(root / "lock/component-lock.json")
    require(
        lock.get("contract_identity_sha256") == contract["contract_identity_sha256"],
        "Wave-3 component lock does not bind DATA-300 v2 identity",
    )

    source_rows = load_jsonl(root / "manifests/source-inventory.jsonl")
    require(
        [row.get("source_id") for row in source_rows] == EXPECTED_SOURCE_IDS,
        "Wave-3 source manifest is not the exact frozen inventory",
    )

    train_rows = load_jsonl(root / "manifests/train.jsonl")
    selection_rows = load_jsonl(root / "manifests/selection-validation.jsonl")
    final_rows = load_jsonl(root / "manifests/final-test-reservation.jsonl")
    require(selection_rows, "selection-validation manifest must be nonempty")
    require(final_rows, "final-test reservation manifest must be nonempty")

    train_ids, train_hashes, train_clusters = split_keys(train_rows, "train")
    sel_ids, sel_hashes, sel_clusters = split_keys(selection_rows, "selection")
    final_ids, final_hashes, final_clusters = split_keys(final_rows, "final-test")
    require(train_ids.isdisjoint(sel_ids | final_ids), "record identity crosses splits")
    require(sel_ids.isdisjoint(final_ids), "record identity crosses eval splits")
    require(train_hashes.isdisjoint(sel_hashes | final_hashes), "content crosses splits")
    require(sel_hashes.isdisjoint(final_hashes), "content crosses eval splits")
    require(
        train_clusters.isdisjoint(sel_clusters | final_clusters),
        "cluster straddles split",
    )
    require(sel_clusters.isdisjoint(final_clusters), "eval cluster straddles split")

    rights = load_json(root / "evidence/rights.json")
    require(rights.get("status") == "PASS", "rights gate is not PASS")
    require_source_ids(rights.get("source_ids", []), expected_ids, "rights")

    quality = load_json(root / "evidence/quality.json")
    require(quality.get("status") == "PASS", "quality gate is not PASS")
    require_source_ids(quality.get("source_ids", []), expected_ids, "quality")

    privacy = load_json(root / "evidence/privacy.json")
    require(privacy.get("status") == "PASS", "privacy gate is not PASS")
    require_source_ids(privacy.get("source_ids", []), expected_ids, "privacy")
    require(
        privacy.get("retained_secret_like_findings") == 0,
        "privacy report retains secret-like findings",
    )

    dedup = load_json(root / "evidence/dedup.json")
    require(dedup.get("status") == "PASS", "dedup gate is not PASS")
    require(dedup.get("cluster_safe_split") is True, "dedup split is not cluster-safe")
    require(
        dedup.get("duplicate_capacity_inflation_bytes") == 0,
        "duplicate copies inflate apparent capacity",
    )

    reserved = load_json(root / "evidence/reserved-decontamination.json")
    require(reserved.get("status") == "PASS", "reserved decontamination is not PASS")
    require(
        reserved.get("reserved_overlap_count") == 0,
        "reserved evaluation material overlaps training",
    )
    require_source_ids(
        reserved.get("scanned_train_source_ids", []),
        expected_ids,
        "reserved-decontamination",
    )

    balance = load_json(root / "evidence/balance.json")
    require(balance.get("status") == "PASS", "balance gate is not PASS")
    require(
        balance.get("policy_head_sha")
        == contract["terminal_component_lock"]["balance"]["head_sha"],
        "balance policy identity drift",
    )
    family_counts = balance.get("independent_family_counts", {})
    minimum = contract["terminal_component_lock"]["balance"][
        "minimum_independent_families_per_stratum"
    ]
    for stratum in ("uk", "en", "code"):
        require(
            int(family_counts.get(stratum, 0)) >= minimum,
            f"balance family minimum fails for {stratum}",
        )
    require(balance.get("document_replication") is False, "balance replicates documents")
    require(
        balance.get("sampling_with_replacement") is False,
        "balance samples with replacement",
    )

    selection = load_json(root / "evidence/selection-authority.json")
    require(selection.get("status") == "PASS", "selection authority is not PASS")
    require(
        int(selection.get("terminal_immutable_records", 0)) > 0,
        "selection authority is empty",
    )
    require(
        selection.get("uses_final_test_bytes") is False,
        "selection authority uses final-test bytes",
    )
    require(
        selection.get("uses_final_test_outcomes") is False,
        "selection authority uses final-test outcomes",
    )

    unique_rows = load_jsonl(root / "ledgers/train-unique-loss.jsonl")
    seen_positions: set[tuple[str, int]] = set()
    for row in unique_rows:
        require(row.get("source_id") in expected_ids, "unknown source in loss ledger")
        require(row.get("optimized") is True, "non-optimized position in loss ledger")
        require(row.get("padding") is False, "padding position in loss ledger")
        segment = row.get("segment_identity_sha256")
        target = row.get("target_token_index")
        require(bool(segment), "loss row missing segment identity")
        require(isinstance(target, int) and target >= 1, "invalid target token index")
        key = (segment, target)
        require(key not in seen_positions, "optimized loss position replay detected")
        seen_positions.add(key)

    unique_summary = load_json(root / "evidence/unique-loss-summary.json")
    require(unique_summary.get("status") == "PASS", "unique-loss summary is not PASS")
    require(
        unique_summary.get("optimized_target_count") == len(unique_rows),
        "unique-loss summary count mismatch",
    )
    require(
        unique_summary.get("replayed_target_count") == 0,
        "unique-loss summary reports replay",
    )
    require(
        unique_summary.get("padding_target_count") == 0,
        "unique-loss summary counts padding",
    )

    gates = load_json(root / "release/gates.json")
    require(
        [row.get("id") for row in gates.get("gates", [])] == EXPECTED_GATES,
        "release gate file vector drift",
    )
    require(
        all(row.get("status") == "PASS" for row in gates["gates"]),
        "not every hard release gate is PASS",
    )
    require(
        gates.get("corpus_state") not in {"CORPUS_FROZEN", "TERMINAL_CORPUS"},
        "pre-release validator may not itself assert terminal corpus state",
    )

    tree_manifest_path = root / "release/tree-sha256.json"
    tree_manifest = load_json(tree_manifest_path)
    expected_tree = {
        rel: sha
        for rel, sha in tree_map(root).items()
        if rel != "release/tree-sha256.json"
    }
    require(tree_manifest.get("files") == expected_tree, "tree hash manifest mismatch")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate-contract")
    wave3 = sub.add_parser("validate-wave3")
    wave3.add_argument("root", type=Path)
    compare = sub.add_parser("compare-clean-builds")
    compare.add_argument("a", type=Path)
    compare.add_argument("b", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        contract = load_json(args.contract)
        if args.command == "validate-contract":
            validate_contract(contract)
        elif args.command == "validate-wave3":
            validate_wave3(args.root, contract)
        elif args.command == "compare-clean-builds":
            validate_contract(contract)
            compare_clean_builds(args.a, args.b)
        else:
            fail(f"unsupported command: {args.command}")
    except (ContractError, KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"DATA-300 v2 FAIL: {exc}", file=sys.stderr)
        return 1
    print(f"DATA-300 v2 PASS: {args.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
