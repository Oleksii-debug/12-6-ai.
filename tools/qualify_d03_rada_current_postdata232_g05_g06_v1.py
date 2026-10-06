#!/usr/bin/env python3
"""Qualify two independent current-Rada post-DATA232 G05/G06 executions.

This verifier consumes only durable text-free child outputs. It does not reconstruct
or persist training/evaluation text and does not grant corpus, tokenizer, training,
or scale authority.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

EVIDENCE_SCHEMA = "12-6.d03-rada-current-postdata232-g05-g06-execution.v1"
PROOF_SCHEMA = "12-6.d03-rada-current-postdata232-g05-g06-two-clean.v1"
PARENT_EXECUTION_HEAD = "a7982bdfd1650b062808024856e13b1d8916a634"

FILES = {
    "evidence": "post-g05-g06-evidence.json",
    "quality": "g05-authority.json",
    "privacy": "g06-authority.json",
    "survivor_inventory": "survivor-inventory.json",
}
SHA40 = re.compile(r"^[0-9a-f]{40}$")
SHA64 = re.compile(r"^[0-9a-f]{64}$")

EXPECTED_CONTENT_BOUNDARY = {
    "raw_training_text_persisted": False,
    "raw_evaluation_text_persisted": False,
    "raw_survivor_text_persisted": False,
    "durable_output_text_free": True,
}
EXPECTED_TRUTH_BOUNDARY = {
    "current_rada_data232_parent_two_clean_complete": True,
    "canonical_quality_privacy_executed": True,
    "balance_diversity_retest_complete": False,
    "family_caps_complete": False,
    "cluster_safe_split_complete": False,
    "deterministic_pack_two_clean_complete": False,
    "positive_exact_unique_loss_ledger": False,
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
    "foreign_pretrained_weights_used": False,
    "scale_promotion_authorized": False,
}
EXPECTED_NEXT_GATE = "CURRENT_RADA_BALANCE_DIVERSITY_FAMILY_CAP_RETEST"


class G05G06TwoCleanError(RuntimeError):
    """Raised when durable G05/G06 evidence cannot be qualified."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise G05G06TwoCleanError(message)


def canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_line(value: Any) -> bytes:
    return canonical(value) + b"\n"


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def require_sha256(value: Any, label: str) -> str:
    require(
        isinstance(value, str) and SHA64.fullmatch(value) is not None,
        f"{label} must be 64 lowercase hex",
    )
    return value


def require_git_sha(value: Any, label: str) -> str:
    require(
        isinstance(value, str) and SHA40.fullmatch(value) is not None,
        f"{label} must be 40 lowercase hex",
    )
    return value


def self_hash(document: Mapping[str, Any], field: str) -> str:
    core = dict(document)
    core.pop(field, None)
    return sha256(canonical(core))


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key rejected: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise G05G06TwoCleanError(f"non-finite JSON constant rejected: {value}")


def load_json_bytes(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise G05G06TwoCleanError(f"{label}: strict JSON decode failed") from exc
    require(isinstance(value, dict), f"{label}: root must be an object")
    return value


def _resolve_dir(path: Path, label: str) -> Path:
    require(path.is_dir() and not path.is_symlink(), f"{label}: directory invalid")
    try:
        return path.resolve(strict=True)
    except OSError as exc:
        raise G05G06TwoCleanError(f"{label}: directory resolution failed") from exc


def _closed_world_bundle(path: Path, label: str) -> None:
    expected = set(FILES.values())
    observed: set[str] = set()
    for entry in path.iterdir():
        require(
            entry.is_file() and not entry.is_symlink(),
            f"{label}: non-regular bundle member",
        )
        observed.add(entry.name)
    require(observed == expected, f"{label}: bundle member set drift")


def _read_bundle(path: Path, label: str) -> tuple[dict[str, bytes], dict[str, dict[str, Any]]]:
    _closed_world_bundle(path, label)
    raw: dict[str, bytes] = {}
    parsed: dict[str, dict[str, Any]] = {}
    for key, filename in FILES.items():
        payload = (path / filename).read_bytes()
        value = load_json_bytes(payload, f"{label} {filename}")
        require(
            canonical_line(value) == payload,
            f"{label} {filename}: noncanonical durable JSON",
        )
        raw[key] = payload
        parsed[key] = value
    return raw, parsed


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            text=True,
            encoding="utf-8",
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        operation = args[0] if args else "command"
        raise G05G06TwoCleanError(f"git {operation} failed") from exc


def verify_checkout(expected_execution_head: str) -> str:
    expected = require_git_sha(expected_execution_head, "expected execution head")
    require(_git("rev-parse", "HEAD") == expected, "execution HEAD drift")
    try:
        ancestry = subprocess.run(
            ["git", "merge-base", "--is-ancestor", PARENT_EXECUTION_HEAD, expected],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError as exc:
        raise G05G06TwoCleanError("git merge-base failed") from exc
    require(
        ancestry.returncode == 0,
        "execution HEAD is outside the exact matrix-DATA232 successor stack",
    )
    return expected


def write_immutable_bytes(path: Path, payload: bytes) -> None:
    require(not path.is_symlink(), "proof path must not be a symlink")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.is_file(), "proof path is not a regular file")
        require(path.read_bytes() == payload, "refusing divergent proof overwrite")
        return
    temp = path.with_name(path.name + ".tmp")
    require(not temp.is_symlink(), "proof temp path must not be a symlink")
    if temp.exists():
        require(temp.is_file(), "proof temp path is not a regular file")
        require(temp.read_bytes() == payload, "divergent interrupted proof temp")
        temp.replace(path)
        return
    created = False
    try:
        with temp.open("xb") as handle:
            created = True
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temp.replace(path)
    except OSError:
        if created:
            try:
                temp.unlink(missing_ok=True)
            except OSError:
                pass
        raise


def _verify_evidence(
    values: Mapping[str, dict[str, Any]],
    *,
    expected_execution_head: str,
    expected_parent_artifact_id: int,
    expected_parent_artifact_zip_sha256: str,
) -> dict[str, Any]:
    evidence = values["evidence"]
    quality = values["quality"]
    privacy = values["privacy"]
    inventory = values["survivor_inventory"]

    require(evidence.get("schema_version") == EVIDENCE_SCHEMA, "evidence schema drift")
    evidence_identity = require_sha256(
        evidence.get("evidence_identity_sha256"), "evidence identity"
    )
    require(
        evidence_identity == self_hash(evidence, "evidence_identity_sha256"),
        "evidence self-hash mismatch",
    )
    require(
        evidence.get("execution_head_sha") == expected_execution_head,
        "evidence execution HEAD drift",
    )

    parent = evidence.get("parent")
    require(isinstance(parent, Mapping), "parent binding missing")
    require(
        parent.get("execution_head_sha") == PARENT_EXECUTION_HEAD,
        "parent execution HEAD drift",
    )
    require(
        parent.get("artifact_id") == expected_parent_artifact_id,
        "parent artifact ID drift",
    )
    require(
        parent.get("artifact_zip_sha256")
        == require_sha256(
            expected_parent_artifact_zip_sha256,
            "expected parent artifact ZIP SHA-256",
        ),
        "parent artifact ZIP SHA-256 drift",
    )
    require(
        parent.get("two_fresh_data232_processes_byte_identical") is True,
        "parent DATA232 two-clean binding is nonterminal",
    )

    g05 = evidence.get("g05")
    g06 = evidence.get("g06")
    require(isinstance(g05, Mapping), "G05 evidence missing")
    require(isinstance(g06, Mapping), "G06 evidence missing")
    quality_id = require_sha256(
        quality.get("execution_identity_sha256"), "G05 authority identity"
    )
    privacy_id = require_sha256(
        privacy.get("execution_identity_sha256"), "G06 authority identity"
    )
    require(
        g05.get("execution_identity_sha256") == quality_id,
        "G05 evidence/authority identity drift",
    )
    require(
        g06.get("execution_identity_sha256") == privacy_id,
        "G06 evidence/authority identity drift",
    )
    require(
        g06.get("exact_payload_collision_free") is True,
        "G06 exact-payload uniqueness is nonterminal",
    )
    unique_count = g06.get("unique_payload_count")
    require(
        type(unique_count) is int and unique_count > 0,
        "G06 unique payload count invalid",
    )
    require_sha256(g06.get("payload_set_identity_sha256"), "G06 payload-set identity")

    durable = evidence.get("durable_artifacts")
    require(isinstance(durable, Mapping), "durable artifact binding missing")
    require(
        durable.get("completion_marker") == "POST_G05_G06_EVIDENCE_WRITTEN_LAST",
        "durable completion marker drift",
    )
    require(
        durable.get("g05_authority_file_sha256") == sha256(canonical_line(quality)),
        "G05 durable file hash drift",
    )
    require(
        durable.get("g06_authority_file_sha256") == sha256(canonical_line(privacy)),
        "G06 durable file hash drift",
    )
    require(
        durable.get("survivor_inventory_file_sha256")
        == sha256(canonical_line(inventory)),
        "survivor inventory durable file hash drift",
    )

    survivor = evidence.get("survivor_inventory")
    require(isinstance(survivor, Mapping), "survivor summary missing")
    for field in (
        "record_count",
        "total_payload_bytes",
        "record_inventory_digest_sha256",
        "payload_inventory_digest_sha256",
    ):
        require(
            survivor.get(field) == inventory.get(field),
            f"survivor summary/inventory drift: {field}",
        )
    require(
        survivor.get("record_payload_jsonl_sha256") is not None,
        "survivor record-payload root missing",
    )
    require_sha256(
        survivor.get("record_payload_jsonl_sha256"),
        "survivor record-payload root",
    )
    require(
        unique_count == survivor.get("record_count"),
        "G06 unique payload count differs from survivor count",
    )

    require(
        evidence.get("content_boundary") == EXPECTED_CONTENT_BOUNDARY,
        "content boundary drift",
    )
    require(
        evidence.get("truth_boundary") == EXPECTED_TRUTH_BOUNDARY,
        "zero-credit truth boundary drift",
    )
    require(evidence.get("next_gate") == EXPECTED_NEXT_GATE, "next gate drift")

    return {
        "evidence_identity_sha256": evidence_identity,
        "g05_execution_identity_sha256": quality_id,
        "g06_execution_identity_sha256": privacy_id,
        "record_payload_jsonl_sha256": survivor["record_payload_jsonl_sha256"],
        "record_inventory_digest_sha256": require_sha256(
            survivor.get("record_inventory_digest_sha256"),
            "survivor record inventory root",
        ),
        "payload_inventory_digest_sha256": require_sha256(
            survivor.get("payload_inventory_digest_sha256"),
            "survivor payload inventory root",
        ),
        "payload_set_identity_sha256": require_sha256(
            g06.get("payload_set_identity_sha256"),
            "G06 payload-set identity",
        ),
        "record_count": survivor["record_count"],
        "total_payload_bytes": survivor["total_payload_bytes"],
    }


def qualify(
    output_a: Path,
    output_b: Path,
    proof_path: Path,
    *,
    expected_execution_head: str,
    expected_parent_artifact_id: int,
    expected_parent_artifact_zip_sha256: str,
    enforce_checkout_provenance: bool = True,
) -> dict[str, Any]:
    require(
        type(expected_parent_artifact_id) is int and expected_parent_artifact_id > 0,
        "expected parent artifact ID must be a positive integer",
    )
    expected_parent_artifact_zip_sha256 = require_sha256(
        expected_parent_artifact_zip_sha256,
        "expected parent artifact ZIP SHA-256",
    )
    resolved_a = _resolve_dir(output_a, "output A")
    resolved_b = _resolve_dir(output_b, "output B")
    require(resolved_a != resolved_b, "two-clean output directories must be distinct")

    raw_a, values_a = _read_bundle(output_a, "output A")
    raw_b, values_b = _read_bundle(output_b, "output B")
    file_hashes: dict[str, str] = {}
    for key, filename in FILES.items():
        require(raw_a[key] == raw_b[key], f"two-clean output differs: {filename}")
        file_hashes[filename] = sha256(raw_a[key])

    expected_execution_head = require_git_sha(
        expected_execution_head, "expected execution head"
    )
    summary_a = _verify_evidence(
        values_a,
        expected_execution_head=expected_execution_head,
        expected_parent_artifact_id=expected_parent_artifact_id,
        expected_parent_artifact_zip_sha256=expected_parent_artifact_zip_sha256,
    )
    summary_b = _verify_evidence(
        values_b,
        expected_execution_head=expected_execution_head,
        expected_parent_artifact_id=expected_parent_artifact_id,
        expected_parent_artifact_zip_sha256=expected_parent_artifact_zip_sha256,
    )
    require(summary_a == summary_b, "two-clean semantic summary differs")

    if enforce_checkout_provenance:
        verify_checkout(expected_execution_head)

    core = {
        "schema_version": PROOF_SCHEMA,
        "execution_head_sha": expected_execution_head,
        "parent_execution_head_sha": PARENT_EXECUTION_HEAD,
        "parent_artifact_id": expected_parent_artifact_id,
        "parent_artifact_zip_sha256": expected_parent_artifact_zip_sha256,
        "fresh_execution_count": 2,
        "independent_runner_jobs": True,
        "byte_identical_outputs": True,
        "output_file_sha256": dict(sorted(file_hashes.items())),
        **summary_a,
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
        "next_gate": EXPECTED_NEXT_GATE,
    }
    proof = {**core, "proof_identity_sha256": sha256(canonical(core))}
    write_immutable_bytes(proof_path, canonical_line(proof))
    return proof


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    result.add_argument("--output-a", type=Path, required=True)
    result.add_argument("--output-b", type=Path, required=True)
    result.add_argument("--proof", type=Path, required=True)
    result.add_argument("--expected-execution-head", required=True)
    result.add_argument("--expected-parent-artifact-id", type=int, required=True)
    result.add_argument("--expected-parent-artifact-zip-sha256", required=True)
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        proof = qualify(
            args.output_a,
            args.output_b,
            args.proof,
            expected_execution_head=args.expected_execution_head,
            expected_parent_artifact_id=args.expected_parent_artifact_id,
            expected_parent_artifact_zip_sha256=args.expected_parent_artifact_zip_sha256,
            enforce_checkout_provenance=True,
        )
    except (OSError, RuntimeError, UnicodeError, ValueError) as exc:
        detail = " ".join(str(exc).split())[:500]
        print(f"D03_RADA_POST_DATA232_G05_G06_TWO_CLEAN=BLOCKED: {detail}")
        return 2
    print(
        "D03_RADA_POST_DATA232_G05_G06_TWO_CLEAN=PASS_ZERO_CREDIT "
        + proof["proof_identity_sha256"]
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
