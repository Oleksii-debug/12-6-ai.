#!/usr/bin/env python3
"""DATA-300 frozen executable build contract validator.

This validates the contract itself, one Wave-3 build tree, or byte identity
between two clean Wave-3 build trees. It never declares the corpus frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

CONTRACT_PATH = Path("configs/data/data300_corpus_v03_frozen_build_contract_v1.json")
SCHEMA = "12-6.data300-corpus-v03-frozen-build-contract.v1"
EXPECTED_SOURCE_IDS = (
    "external-real:en.standardebooks.manual.8-typography",
    "external-real:en.standardebooks.manual.9-metadata",
    "external-real:ua.rada.open-data.laws-texts.d23314",
    "external-real:code.encode.httpx._content",
    "external-real:code.psf.requests._internal_utils",
)
INTERNAL_GATE_IDS = (
    "G01_CONTRACT_IDENTITY",
    "G02_SOURCE_INVENTORY",
    "G03_RIGHTS",
    "G04_QUALITY",
    "G05_PRIVACY",
    "G06_DEDUP",
    "G07_RESERVED_DECONTAM",
    "G08_BALANCE_DIVERSITY",
    "G09_SELECTION_VALIDATION",
    "G10_FINAL_TEST_ISOLATION",
    "G11_UNIQUE_LOSS",
    "G13_RELEASE_TRUTH",
)
FORBIDDEN_CORPUS_STATES = {"CORPUS_FROZEN", "TERMINAL_CORPUS", "PRODUCTION_READY"}
VOLATILE_JSON_KEYS = {
    "build_started_at",
    "build_finished_at",
    "wall_clock_time",
    "hostname",
    "host_name",
    "workspace",
    "absolute_workspace_path",
    "random_uuid",
    "run_uuid",
}


class ContractError(RuntimeError):
    pass


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"{path} must contain a JSON object")
    return value


def _read_contract(repo: Path) -> dict[str, Any]:
    path = repo / CONTRACT_PATH
    contract = _read_json(path)
    if contract.get("schema_version") != SCHEMA:
        raise ContractError("unsupported DATA-300 contract schema")
    claimed = contract.get("contract_identity_sha256")
    if not isinstance(claimed, str):
        raise ContractError("contract identity is missing")
    unhashed = dict(contract)
    unhashed.pop("contract_identity_sha256", None)
    actual = _sha256(_canonical(unhashed))
    if claimed != actual:
        raise ContractError(f"contract identity mismatch: {claimed} != {actual}")
    return contract


def _assert_contract_invariants(contract: dict[str, Any]) -> None:
    if contract.get("contract_state") != "FROZEN_EXECUTABLE_CONTRACT":
        raise ContractError("contract itself is not frozen")
    if contract.get("corpus_state") != "NOT_BUILT_NOT_FROZEN_NOT_TERMINAL":
        raise ContractError("contract must not pre-declare corpus freeze/terminal state")
    if contract.get("local_free_only") is not True:
        raise ContractError("LOCAL_FREE must be hard-bound")

    inventory = contract["terminal_source_inventory"]
    sources = inventory["sources"]
    source_ids = tuple(item["registry_source_id"] for item in sources)
    if source_ids != EXPECTED_SOURCE_IDS:
        raise ContractError("exact source inventory/order drift")
    if inventory["source_count"] != len(EXPECTED_SOURCE_IDS):
        raise ContractError("source_count drift")
    if len(set(source_ids)) != len(source_ids):
        raise ContractError("duplicate source identity in frozen inventory")

    for item in sources:
        if item.get("origin_class") != "EXTERNAL_REAL":
            raise ContractError(f"non-external source in inventory: {item['registry_source_id']}")
        if item.get("training_rights") != "ALLOWED":
            raise ContractError(f"training rights not allowed: {item['registry_source_id']}")
        if item.get("modality") == "text":
            if len(item.get("raw_sha256", "")) != 64:
                raise ContractError(f"text raw SHA-256 not bound: {item['registry_source_id']}")
            if len(item.get("normalized_sha256", "")) != 64:
                raise ContractError(
                    f"text normalized SHA-256 not bound: {item['registry_source_id']}"
                )
        elif item.get("modality") == "code":
            if len(item.get("upstream_commit", "")) != 40:
                raise ContractError(f"code commit not bound: {item['registry_source_id']}")
            if len(item.get("git_blob_sha1", "")) != 40:
                raise ContractError(f"code blob not bound: {item['registry_source_id']}")
            if len(item.get("license_git_blob_sha1", "")) != 40:
                raise ContractError(f"code license blob not bound: {item['registry_source_id']}")
        else:
            raise ContractError(f"unsupported modality: {item['registry_source_id']}")

    components = contract["component_lock"]
    for name in (
        "source_registry",
        "rights",
        "dedup_and_reserved_decontamination",
        "reservation",
        "balance_and_diversity",
        "unique_loss",
    ):
        value = components[name]
        if value.get("dedicated_workflow_conclusion") != "success":
            raise ContractError(f"{name} is not bound to dedicated terminal success")

    excluded = inventory.get("excluded_nonterminal_candidates", [])
    if not excluded or excluded[0].get("rule") != "NO_ADMISSION":
        raise ContractError("nonterminal DATA-228 exclusion is not fail-closed")

    repetition = contract["build_contract"]["artificial_repetition"]
    if any(repetition.values()):
        raise ContractError("artificial repetition invariant weakened")
    if contract["build_contract"].get("clean_build_count") != 2:
        raise ContractError("exactly two clean builds are required")
    if contract["build_contract"].get("byte_identical_complete_tree_required") is not True:
        raise ContractError("complete-tree byte identity must be required")

    split = contract["split_contract"]
    if not split["selection_validation"]["must_be_nonempty_before_tokenizer_or_model_selection"]:
        raise ContractError("selection-validation nonempty gate weakened")
    if split["final_test"]["may_be_read_before_selection_is_locked"]:
        raise ContractError("final-test may not be exposed before selection lock")
    if split["content_hash_overlap_allowed_between_splits"]:
        raise ContractError("cross-split content overlap cannot be allowed")
    if split["dedup_cluster_straddling_allowed"]:
        raise ContractError("dedup clusters cannot straddle splits")

    gate_ids = tuple(item["id"] for item in contract["release_gates"])
    expected = INTERNAL_GATE_IDS[:-1] + ("G12_TWO_CLEAN_BUILDS",) + INTERNAL_GATE_IDS[-1:]
    if gate_ids != expected:
        raise ContractError("release gate inventory/order drift")
    if any(item.get("severity") != "HARD" for item in contract["release_gates"]):
        raise ContractError("all DATA-300 release gates must be hard")

    release = contract["wave3_release_state"]
    if set(release["forbidden_from_this_contract_alone"]) != FORBIDDEN_CORPUS_STATES:
        raise ContractError("release truth boundary drift")


def _required_paths(contract: dict[str, Any], root: Path) -> list[Path]:
    return [root / rel for rel in contract["wave3_required_artifacts"]]


def _walk_json_no_volatile(value: Any, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in VOLATILE_JSON_KEYS:
                raise ContractError(f"volatile identity field forbidden: {path}.{key}")
            _walk_json_no_volatile(child, f"{path}.{key}")
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            _walk_json_no_volatile(child, f"{path}[{idx}]")


def _manifest_hashes(path: Path) -> set[str]:
    value = _read_json(path)
    records = value.get("records")
    if not isinstance(records, list):
        raise ContractError(f"{path}: records list required")
    hashes: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            raise ContractError(f"{path}: every record must be an object")
        digest = record.get("content_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ContractError(f"{path}: each record needs content_sha256")
        if digest in hashes:
            raise ContractError(f"{path}: duplicate content_sha256 {digest}")
        hashes.add(digest)
    return hashes


def _validate_unique_loss(root: Path) -> None:
    ledger = root / "unique-loss/train-ledger.jsonl"
    seen: set[tuple[str, str, int]] = set()
    rows = 0
    with ledger.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ContractError(f"{ledger}:{line_number}: blank ledger line")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ContractError(f"{ledger}:{line_number}: invalid JSON") from exc
            for key in ("source_id", "record_id", "target_offset"):
                if key not in row:
                    raise ContractError(f"{ledger}:{line_number}: missing {key}")
            if row.get("is_padding") is True:
                raise ContractError(f"{ledger}:{line_number}: padding cannot be optimized data")
            if row.get("optimized", True) is not True:
                raise ContractError(
                    f"{ledger}:{line_number}: non-optimized row in optimized ledger"
                )
            key = (str(row["source_id"]), str(row["record_id"]), int(row["target_offset"]))
            if key in seen:
                raise ContractError(f"{ledger}:{line_number}: repeated source loss position {key}")
            seen.add(key)
            rows += 1

    if rows <= 0:
        raise ContractError("unique-loss ledger must be nonempty")
    summary = _read_json(root / "unique-loss/summary.json")
    if summary.get("unique_optimized_loss_positions") != rows:
        raise ContractError("unique-loss summary does not equal ledger row count")
    if summary.get("repeated_optimized_loss_positions") != 0:
        raise ContractError("repeated optimized loss positions must be zero")
    if summary.get("padding_counted_as_data") is not False:
        raise ContractError("padding_counted_as_data must be false")
    if summary.get("corpus_replay") is not False:
        raise ContractError("corpus_replay must be false")


def _validate_build(repo: Path, root: Path) -> dict[str, Any]:
    contract = _read_contract(repo)
    _assert_contract_invariants(contract)
    if not root.is_dir():
        raise ContractError(f"build root not found: {root}")

    for path in _required_paths(contract, root):
        if not path.is_file():
            raise ContractError(f"required Wave-3 artifact missing: {path.relative_to(root)}")

    for path in root.rglob("*.json"):
        value = _read_json(path)
        _walk_json_no_volatile(value)

    inventory = _read_json(root / "source/source-inventory.json")
    ids = inventory.get("source_ids")
    if ids != list(EXPECTED_SOURCE_IDS):
        raise ContractError("Wave-3 source inventory is not exact DATA-300 inventory")
    if inventory.get("source_count") != len(EXPECTED_SOURCE_IDS):
        raise ContractError("Wave-3 source_count mismatch")
    if inventory.get("contract_identity_sha256") != contract["contract_identity_sha256"]:
        raise ContractError("Wave-3 inventory not bound to this contract")

    rights = _read_json(root / "source/rights-evidence.json")
    decisions = rights.get("decisions")
    if not isinstance(decisions, list):
        raise ContractError("rights-evidence decisions list required")
    rights_map = {item.get("source_id"): item for item in decisions if isinstance(item, dict)}
    if set(rights_map) != set(EXPECTED_SOURCE_IDS):
        raise ContractError("rights evidence must cover exact source inventory")
    for source_id, decision in rights_map.items():
        if decision.get("model_training") != "ALLOWED":
            raise ContractError(f"training rights fail for {source_id}")
        if decision.get("basis") in (None, "", "PUBLIC_ACCESS_ONLY", "LICENSE_LABEL_ONLY"):
            raise ContractError(f"purpose-specific rights basis missing for {source_id}")

    quality = _read_json(root / "quality/quality-report.json")
    if quality.get("status") != "PASS":
        raise ContractError("quality gate did not PASS")
    if set(quality.get("covered_source_ids", [])) != set(EXPECTED_SOURCE_IDS):
        raise ContractError("quality report does not cover exact source inventory")

    privacy = _read_json(root / "privacy/privacy-report.json")
    if privacy.get("status") != "PASS":
        raise ContractError("privacy gate did not PASS")
    if set(privacy.get("covered_source_ids", [])) != set(EXPECTED_SOURCE_IDS):
        raise ContractError("privacy report does not cover exact source inventory")
    if privacy.get("secret_like_findings", 0) != 0:
        raise ContractError("secret-like findings must be zero after exclusions")

    exact_dedup = _read_json(root / "dedup/exact-dedup.json")
    if exact_dedup.get("status") != "PASS":
        raise ContractError("exact dedup gate did not PASS")

    reserved = _read_json(root / "decontamination/reserved-scan.json")
    if reserved.get("status") not in {"PASS_CLEAN", "PASS_WITH_EXCLUSIONS"}:
        raise ContractError("reserved decontamination did not PASS")
    if reserved.get("scan_executed") is not True:
        raise ContractError("reserved decontamination scan must actually execute")
    if reserved.get("universal_semantic_cleanliness_claimed") is True:
        raise ContractError("universal semantic cleanliness claim is forbidden")

    balance = _read_json(root / "balance/balance-report.json")
    if balance.get("status") != "PASS":
        raise ContractError("balance/diversity gate did not PASS")
    if balance.get("materialized_duplicate_documents") is not False:
        raise ContractError("balance may not materialize duplicate documents")
    if balance.get("sampling_with_replacement") is not False:
        raise ContractError("balance may not sample with replacement")
    if balance.get("intake_family_count", 0) < 4:
        raise ContractError("balance audit lost the four exact intake families")
    if set(balance.get("intake_source_ids", [])) != set(EXPECTED_SOURCE_IDS):
        raise ContractError("balance audit must cover the exact intake inventory")
    if not isinstance(balance.get("mass_by_source_family"), dict):
        raise ContractError("balance audit must publish mass_by_source_family")
    if not isinstance(balance.get("mass_by_language_modality"), dict):
        raise ContractError("balance audit must publish mass_by_language_modality")

    train_path = root / "splits/train/manifest.json"
    selection_path = root / "splits/selection-validation/manifest.json"
    final_path = root / "splits/final-test/manifest.json"
    train = _read_json(train_path)
    selection = _read_json(selection_path)
    final = _read_json(final_path)
    if train.get("purpose") != "train":
        raise ContractError("train manifest purpose mismatch")
    if selection.get("purpose") != "selection_validation":
        raise ContractError("selection-validation purpose mismatch")
    if final.get("purpose") != "final_test":
        raise ContractError("final-test purpose mismatch")
    if len(selection.get("records", [])) <= 0:
        raise ContractError("selection-validation must be nonempty")
    if len(final.get("records", [])) != 16:
        raise ContractError("final-test must preserve the 16-record EVAL-233 authority")
    if selection.get("tokenizer_fit_eligible") is not False:
        raise ContractError("selection-validation cannot fit tokenizer")
    if final.get("selection_eligible") is not False:
        raise ContractError("final-test cannot be used for selection")
    if final.get("tokenizer_fit_eligible") is not False:
        raise ContractError("final-test cannot fit tokenizer")
    if final.get("hyperparameter_selection_eligible") is not False:
        raise ContractError("final-test cannot tune hyperparameters")

    train_hashes = _manifest_hashes(train_path)
    selection_hashes = _manifest_hashes(selection_path)
    final_hashes = _manifest_hashes(final_path)
    if train_hashes & selection_hashes:
        raise ContractError("train/selection content overlap")
    if train_hashes & final_hashes:
        raise ContractError("train/final-test content overlap")
    if selection_hashes & final_hashes:
        raise ContractError("selection/final-test content overlap")

    _validate_unique_loss(root)

    gate = _read_json(root / "release/gate-report.json")
    if gate.get("corpus_state") != "WAVE3_BUILD_PASS_NOT_FROZEN":
        raise ContractError("single build must remain explicitly not frozen")
    gate_values = gate.get("gates")
    if not isinstance(gate_values, dict):
        raise ContractError("gate-report gates object required")
    for gate_id in INTERNAL_GATE_IDS:
        if gate_values.get(gate_id) != "PASS":
            raise ContractError(f"{gate_id} did not PASS")
    if gate_values.get("G12_TWO_CLEAN_BUILDS") != "PENDING_SECOND_BUILD":
        raise ContractError("G12 must remain pending inside each immutable build tree")

    release = _read_json(root / "release/release-manifest.json")
    if release.get("contract_identity_sha256") != contract["contract_identity_sha256"]:
        raise ContractError("release manifest contract identity mismatch")
    if release.get("corpus_state") in FORBIDDEN_CORPUS_STATES:
        raise ContractError("release manifest overclaims frozen/terminal/production state")

    return {
        "status": "WAVE3_BUILD_VALID_NOT_FROZEN",
        "contract_identity_sha256": contract["contract_identity_sha256"],
        "source_count": len(EXPECTED_SOURCE_IDS),
        "unique_loss_positions": _read_json(root / "unique-loss/summary.json")[
            "unique_optimized_loss_positions"
        ],
    }


def _tree_inventory(root: Path) -> dict[str, tuple[int, str]]:
    if not root.is_dir():
        raise ContractError(f"build root not found: {root}")
    result: dict[str, tuple[int, str]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ContractError(f"symlink forbidden in deterministic build tree: {path}")
        if path.is_file():
            rel = path.relative_to(root).as_posix()
            data = path.read_bytes()
            result[rel] = (len(data), _sha256(data))
    return result


def _compare_builds(repo: Path, left: Path, right: Path) -> dict[str, Any]:
    left_report = _validate_build(repo, left)
    right_report = _validate_build(repo, right)
    left_tree = _tree_inventory(left)
    right_tree = _tree_inventory(right)
    if left_tree != right_tree:
        all_paths = sorted(set(left_tree) | set(right_tree))
        differences = [path for path in all_paths if left_tree.get(path) != right_tree.get(path)]
        preview = ", ".join(differences[:10])
        raise ContractError(f"clean builds are not byte-identical: {preview}")
    return {
        "status": "TWO_CLEAN_BUILDS_BYTE_IDENTICAL",
        "corpus_state": "CANDIDATE_READY_FOR_SEPARATE_FREEZE_REVIEW",
        "corpus_frozen": False,
        "terminal_corpus_claimed": False,
        "contract_identity_sha256": left_report["contract_identity_sha256"],
        "file_count": len(left_tree),
        "tree_identity_sha256": _sha256(
            _canonical(
                [
                    {"path": path, "size": size, "sha256": digest}
                    for path, (size, digest) in sorted(left_tree.items())
                ]
            )
        ),
        "unique_loss_positions": left_report["unique_loss_positions"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("."))
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate-contract")
    one = sub.add_parser("validate-build")
    one.add_argument("root", type=Path)
    two = sub.add_parser("compare-builds")
    two.add_argument("left", type=Path)
    two.add_argument("right", type=Path)
    args = parser.parse_args(argv)

    try:
        contract = _read_contract(args.repo)
        _assert_contract_invariants(contract)
        if args.command == "validate-contract":
            result = {
                "status": "CONTRACT_VALID_CORPUS_NOT_BUILT_NOT_FROZEN",
                "contract_identity_sha256": contract["contract_identity_sha256"],
                "source_count": contract["terminal_source_inventory"]["source_count"],
                "release_gate_count": len(contract["release_gates"]),
            }
        elif args.command == "validate-build":
            result = _validate_build(args.repo, args.root)
        else:
            result = _compare_builds(args.repo, args.left, args.right)
    except ContractError as exc:
        print(json.dumps({"status": "FAIL", "reason": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
