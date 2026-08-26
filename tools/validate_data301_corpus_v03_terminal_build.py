#!/usr/bin/env python3
"""Validate DATA-301 terminal fail-closed evidence against the frozen DATA-300 v2 contract."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_PATH = ROOT / "configs/data/data301_corpus_v03_terminal_build_v1.json"
DATA300_PATH = ROOT / "configs/data/data300_corpus_v03_frozen_build_contract_v2.json"

EXPECTED_DATA300_BLOB = "39d4fa07ea17e66e042a3ccb1a55b8e5e1c5d7bf"
EXPECTED_DATA300_IDENTITY = "07d7beaaff4616e839450de6af3d407855c832bf75a24a959d1a12de5d9364e5"
EXPECTED_DATA300_HEAD = "8ea7f830e50a23754d189dd4134f4afad76a7ee9"
EXPECTED_TRAINER_BLOB = "8fb5e9ce4c5417986ad1f086ebc16cd7538a151e"
EXPECTED_BLOCKING_GATES = {
    "G05_QUALITY",
    "G06_PRIVACY",
    "G09_BALANCE_DIVERSITY",
    "G10_SELECTION_VALIDATION",
    "G12_UNIQUE_LOSS",
    "G14_TWO_CLEAN_BUILDS",
}


class ValidationError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_sha256_without_identity(payload: dict[str, Any]) -> str:
    stripped = copy.deepcopy(payload)
    stripped.pop("evidence_identity_sha256", None)
    stripped.pop("evidence_identity_scope", None)
    raw = json.dumps(
        stripped,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8") + b"\n"
    return hashlib.sha256(raw).hexdigest()


def _git_blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()


def _verify_streaming(evidence: dict[str, Any]) -> None:
    streaming = evidence["product_trainer_streaming"]
    trainer_path = ROOT / streaming["trainer_path"]
    _require(_git_blob_sha1(trainer_path) == EXPECTED_TRAINER_BLOB, "trainer blob drifted")
    _require(streaming["trainer_git_blob_sha1"] == EXPECTED_TRAINER_BLOB, "evidence trainer blob mismatch")
    _require(streaming["proof_kind"] == "AST_SOURCE_SEMANTICS", "unexpected trainer proof kind")

    source = trainer_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    trainer_class = next(
        (node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Trainer"),
        None,
    )
    _require(trainer_class is not None, "Trainer class missing")
    run = next(
        (node for node in trainer_class.body if isinstance(node, ast.FunctionDef) and node.name == "run"),
        None,
    )
    _require(run is not None, "Trainer.run missing")

    batches_arg = next((arg for arg in run.args.args if arg.arg == "batches"), None)
    _require(batches_arg is not None and batches_arg.annotation is not None, "batches annotation missing")
    annotation = ast.unparse(batches_arg.annotation)
    _require(annotation == "Iterable[Batch]", f"Trainer.run batches is not Iterable[Batch]: {annotation}")

    direct_for = any(
        isinstance(node, ast.For)
        and isinstance(node.iter, ast.Name)
        and node.iter.id == "batches"
        for node in ast.walk(run)
    )
    _require(direct_for, "Trainer.run no longer directly streams the batches iterable")

    forbidden_materializers = {"list", "tuple", "len", "sorted"}
    for node in ast.walk(run):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in forbidden_materializers:
            if any(isinstance(arg, ast.Name) and arg.id == "batches" for arg in node.args):
                raise ValidationError(f"Trainer.run materializes batches via {node.func.id}()")
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == "batches":
            raise ValidationError("Trainer.run requires indexable batches; streaming contract broken")

    _require("if self.optimizer_step >= self.config.max_steps:" in source, "max_steps boundary guard missing")
    _require("for batch in batches:" in source, "direct streaming loop source contract missing")


def validate() -> dict[str, Any]:
    evidence = _load_json(EVIDENCE_PATH)
    contract = _load_json(DATA300_PATH)

    _require(evidence["schema_version"] == "12-6.data301-corpus-v03-terminal-build.v1", "schema drift")
    _require(evidence["execution_profile"] == "LOCAL_FREE", "execution profile is not LOCAL_FREE")
    _require(evidence["repository"] == "Oleksii-debug/12-6-ai.", "repository identity mismatch")
    _require(
        _canonical_sha256_without_identity(evidence) == evidence["evidence_identity_sha256"],
        "DATA-301 evidence identity mismatch",
    )

    base = evidence["base_data300"]
    _require(base["head_sha"] == EXPECTED_DATA300_HEAD, "DATA-300 head mismatch")
    _require(base["contract_git_blob_sha1"] == EXPECTED_DATA300_BLOB, "DATA-300 blob evidence mismatch")
    _require(_git_blob_sha1(DATA300_PATH) == EXPECTED_DATA300_BLOB, "DATA-300 contract file drifted")
    _require(contract["contract_identity_sha256"] == EXPECTED_DATA300_IDENTITY, "DATA-300 identity drifted")
    _require(base["contract_identity_sha256"] == EXPECTED_DATA300_IDENTITY, "base identity mismatch")
    _require(contract["contract_state"] == "FROZEN_EXECUTABLE_CONTRACT", "DATA-300 contract is not frozen executable")
    _require(contract["corpus_state"] == "NOT_BUILT_NOT_FROZEN_NOT_TERMINAL", "unexpected DATA-300 corpus state")
    _require(base["dedicated_workflow_conclusion"] == "success", "DATA-300 dedicated workflow not terminal-success")

    inventory = contract["exact_training_candidate_inventory"]
    sources = inventory["sources"]
    _require(len(sources) == 5 == evidence["candidate_inventory"]["source_count"], "source count mismatch")
    _require(inventory["admitted_source_bytes"] == 183061, "candidate byte identity mismatch")
    _require(inventory["independent_family_count"] == 4, "independent family count mismatch")

    docs_by_family: Counter[str] = Counter()
    bytes_by_family: defaultdict[str, int] = defaultdict(int)
    docs_by_stratum: Counter[str] = Counter()
    families_by_stratum: defaultdict[str, set[str]] = defaultdict(set)
    bytes_by_stratum: defaultdict[str, int] = defaultdict(int)
    for source in sources:
        family = source["family"]
        language = source["language"]
        size = int(source["normalized_bytes"])
        docs_by_family[family] += 1
        bytes_by_family[family] += size
        docs_by_stratum[language] += 1
        families_by_stratum[language].add(family)
        bytes_by_stratum[language] += size
        _require(source["training_rights"] == "ALLOWED", f"training rights not allowed: {source['source_id']}")

    observed_family = {name: {"docs": docs_by_family[name], "bytes": bytes_by_family[name]} for name in sorted(docs_by_family)}
    observed_stratum = {
        name: {"docs": docs_by_stratum[name], "families": len(families_by_stratum[name]), "bytes": bytes_by_stratum[name]}
        for name in sorted(docs_by_stratum)
    }
    _require(observed_family == evidence["candidate_inventory"]["by_family"], "per-family inventory mismatch")
    _require(observed_stratum == evidence["candidate_inventory"]["by_stratum"], "per-stratum inventory mismatch")
    _require(sum(bytes_by_family.values()) == evidence["candidate_inventory"]["normalized_unique_bytes_prebuild"], "unique byte total mismatch")

    dedup = contract["terminal_component_lock"]["dedup"]
    _require(dedup["duplicate_discount_bytes"] == 0, "DATA-298 duplicate discount changed")
    _require(dedup["observed_duplicate_matches"] == 0, "DATA-298 duplicate matches changed")
    _require(dedup["terminal_evidence_bytes_after"] == 183061, "DATA-298 retained bytes mismatch")

    balance = contract["terminal_component_lock"]["balance"]
    _require(balance["minimum_independent_families_per_stratum"] == 2, "DATA-295 family minimum changed")
    _require(balance["current_family_counts"] == {"uk": 1, "en": 1, "code": 2}, "DATA-295 family counts changed")
    _require(balance["current_family_constrained_no_replay_budget"] == 0, "balanced no-replay budget is no longer zero")
    _require(balance["current_activation_state"] == "BLOCKED_SOURCE_FAMILY_DIVERSITY", "DATA-295 blocker state changed")

    contract_blockers = contract["current_candidate_status"]["blocking_gates"]
    _require(set(contract_blockers) == EXPECTED_BLOCKING_GATES, "DATA-300 blocker set changed")
    _require(evidence["blocking_gates"] == contract_blockers, "DATA-301 blocker text does not exactly bind DATA-300")

    unique_loss = contract["terminal_component_lock"]["unique_loss"]
    text_only = evidence["one_pass_loss_position_accounting"]["terminal_text_only_data294"]
    _require(unique_loss["ledger_scope"] == text_only["scope"] == "DATA-229_TEXT_ONLY_3_OBJECTS", "DATA-294 scope mismatch")
    _require(unique_loss["one_pass_unique_optimized_targets"] == text_only["one_pass_unique_optimized_targets"] == 173355, "text-only loss capacity mismatch")
    _require(unique_loss["by_language"] == text_only["by_language"], "text-only per-language loss capacity mismatch")
    _require(evidence["one_pass_loss_position_accounting"]["full_five_source_terminal_ledger_available"] is False, "false full-ledger claim")
    _require(evidence["one_pass_loss_position_accounting"]["authorized_balanced_no_replay_capacity"] == 0, "unauthorized nonzero balanced capacity")

    repetition = contract["artificial_repetition"]
    _require(all(value is False for value in repetition.values()), "DATA-300 repetition prohibition changed")

    pipeline = evidence["pipeline_attempt"]
    _require(pipeline["cluster_safe_split"] == "NOT_REACHED", "split should not be claimed after hard prebuild blockers")
    _require(pipeline["deterministic_sharding"] == "NOT_REACHED", "sharding should not be claimed after hard prebuild blockers")
    _require(pipeline["two_clean_builds"] == "NOT_PERMITTED_BECAUSE_PREBUILD_HARD_GATES_FAIL", "two-build truth mismatch")

    verdict = evidence["terminal_verdict"]
    _require(verdict["status"] == "TERMINAL_BLOCKED", "terminal verdict must fail closed")
    _require(verdict["corpus_identity"] is None and verdict["shard_identity"] is None, "blocked build must not invent corpus/shard identities")
    _require(verdict["corpus_frozen"] is False and verdict["corpus_terminal"] is False and verdict["release_ready"] is False, "blocked build must not claim release state")
    _require("successor DATA-300" in contract["current_candidate_status"]["unblock_rule"], "DATA-300 unblock rule changed")

    _verify_streaming(evidence)

    return {
        "status": verdict["status"],
        "evidence_identity_sha256": evidence["evidence_identity_sha256"],
        "corpus_identity": None,
        "shard_identity": None,
        "candidate_docs": 5,
        "candidate_unique_bytes_prebuild": 183061,
        "authorized_balanced_no_replay_capacity": 0,
        "terminal_text_only_unique_loss_positions": 173355,
        "product_trainer_streaming": "PASS_AST_SOURCE_SEMANTICS",
        "blocking_gates": sorted(EXPECTED_BLOCKING_GATES),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("validate",))
    args = parser.parse_args()
    if args.command == "validate":
        print(json.dumps(validate(), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
