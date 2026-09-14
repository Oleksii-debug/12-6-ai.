#!/usr/bin/env python3
"""Fail-closed Hydra configuration qualification contract and evidence builder.

This module intentionally has no Hydra dependency. It qualifies the provenance and
identity mechanics required before Hydra can be considered as an optional config layer.
It never makes Hydra canonical lineage or promotion authority.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")

EXPECTED_CONTRACT_ID = "12-6.hydra-config-qualification.v1"
EXPECTED_REGISTRY_PATH = "configs/research/open_source_reuse_registry_v2.json"
EXPECTED_REGISTRY_COMPONENT = "HYDRA"
EXPECTED_REGISTRY_DECISION = "P1_CONFIG_SYSTEM_CANDIDATE"
EXPECTED_UPSTREAM_REPOSITORY = "https://github.com/hydra-ecosystem/hydra"
EXPECTED_UPSTREAM_TAG = "v1.3.5"
EXPECTED_UPSTREAM_COMMIT = "51647a2183512bc4e5556842c494e7efdbd75375"
EXPECTED_LICENSE = "MIT"
EXPECTED_CANONICAL_AUTHORITIES = {
    "GITHUB_EXACT_SHA",
    "D05_RUN_MANIFEST",
    "D05_CHECKPOINT_MANIFEST",
}
EXPECTED_GATES = {
    "UPSTREAM_IDENTITY_BOUND",
    "REGISTRY_BINDING_BOUND",
    "CANONICAL_AUTHORITY_PRESERVED",
    "EXPLICIT_OVERRIDE_LEDGER",
    "RESOLVED_CONFIG_BOUND",
    "DETERMINISTIC_REBUILD",
    "PORTABLE_ROLLBACK_EXPORT",
    "NO_SECRET_BEARING_METADATA",
}
ALLOWED_OVERRIDE_SOURCES = {
    "CLI_EXPLICIT",
    "CONFIG_GROUP_SELECTION",
    "SWEEP_PARAMETER_EXPLICIT",
}


class QualificationError(ValueError):
    """Raised when a qualification input violates a fail-closed gate."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise QualificationError(message)


def _require_exact_keys(mapping: Any, required: set[str], where: str) -> None:
    _require(isinstance(mapping, dict), f"{where} must be an object")
    missing = required - set(mapping)
    _require(not missing, f"{where} missing required keys: {sorted(missing)}")


def validate_contract(contract: dict[str, Any]) -> dict[str, Any]:
    """Validate the static qualification contract and return a summary."""
    _require_exact_keys(
        contract,
        {
            "schema_version",
            "contract_id",
            "status",
            "authority",
            "upstream",
            "execution_policy",
            "composition_contract",
            "promotion",
            "project_owned_fixture",
            "truth_boundary",
        },
        "contract",
    )
    _require(contract["schema_version"] == 1, "unsupported schema_version")
    _require(contract["contract_id"] == EXPECTED_CONTRACT_ID, "unexpected contract_id")
    _require(contract["status"] == "CANDIDATE_NOT_ADOPTED", "status must remain candidate")

    authority = contract["authority"]
    _require_exact_keys(
        authority,
        {
            "project_base_git_sha",
            "parent_issue",
            "claim_issue",
            "registry",
            "canonical_lineage_authorities",
            "hydra_is_canonical_lineage_authority",
            "hydra_can_gate_stage_promotion",
        },
        "authority",
    )
    _require(
        isinstance(authority["project_base_git_sha"], str)
        and HEX40.fullmatch(authority["project_base_git_sha"]) is not None,
        "project_base_git_sha must be an exact 40-hex Git SHA",
    )
    _require(authority["parent_issue"] == 720, "parent issue must remain #720")
    _require(authority["claim_issue"] == 749, "claim issue must remain #749")
    _require(
        set(authority["canonical_lineage_authorities"]) == EXPECTED_CANONICAL_AUTHORITIES,
        "canonical lineage authority set changed",
    )
    _require(
        authority["hydra_is_canonical_lineage_authority"] is False,
        "Hydra cannot become canonical lineage authority",
    )
    _require(
        authority["hydra_can_gate_stage_promotion"] is False,
        "Hydra cannot become stage-promotion authority",
    )

    registry = authority["registry"]
    _require_exact_keys(
        registry,
        {"path", "blob_sha", "component_id", "decision"},
        "authority.registry",
    )
    _require(registry["path"] == EXPECTED_REGISTRY_PATH, "registry path drift")
    _require(
        isinstance(registry["blob_sha"], str)
        and HEX40.fullmatch(registry["blob_sha"]) is not None,
        "registry blob must be an exact Git blob SHA",
    )
    _require(registry["component_id"] == EXPECTED_REGISTRY_COMPONENT, "wrong registry component")
    _require(registry["decision"] == EXPECTED_REGISTRY_DECISION, "registry decision drift")

    upstream = contract["upstream"]
    _require_exact_keys(
        upstream,
        {"repository", "release_tag", "commit_sha", "license", "release_published_at"},
        "upstream",
    )
    _require(
        upstream["repository"] == EXPECTED_UPSTREAM_REPOSITORY,
        "Hydra repository identity drift",
    )
    _require(upstream["release_tag"] == EXPECTED_UPSTREAM_TAG, "Hydra release tag drift")
    _require(
        upstream["commit_sha"] == EXPECTED_UPSTREAM_COMMIT,
        "Hydra release must resolve to the pinned immutable commit",
    )
    _require(
        HEX40.fullmatch(upstream["commit_sha"]) is not None,
        "Hydra upstream commit is not an exact Git SHA",
    )
    _require(upstream["license"] == EXPECTED_LICENSE, "Hydra license identity drift")
    _require(
        upstream["release_tag"] not in {"main", "master", "latest", "HEAD"},
        "mutable upstream reference forbidden",
    )

    execution = contract["execution_policy"]
    _require(execution.get("execution_class") == "LOCAL_FREE", "only LOCAL_FREE is allowed")
    for field in (
        "paid_compute_authorized",
        "model_training_authorized",
        "canonical_base_dependency",
        "dependency_install_authorized_by_contract",
        "final_test_access_authorized",
    ):
        _require(execution.get(field) is False, f"{field} must remain false")

    composition = contract["composition_contract"]
    _require(
        composition.get("identity_algorithm") == "SHA256_CANONICAL_JSON_V1",
        "identity algorithm drift",
    )
    _require(composition.get("defaults_entries_require_sha256") is True, "defaults must hash")
    _require(
        composition.get("all_overrides_must_be_ordered_and_explicit") is True,
        "override provenance must remain explicit",
    )
    _require(
        set(composition.get("allowed_override_sources", [])) == ALLOWED_OVERRIDE_SOURCES,
        "allowed override source set drift",
    )
    _require(
        composition.get("hidden_or_implicit_overrides_forbidden") is True,
        "hidden overrides must remain forbidden",
    )
    _require(
        composition.get("runtime_environment_interpolation_for_identity") == "FORBIDDEN",
        "environment interpolation cannot contribute hidden identity state",
    )
    for field in (
        "resolved_config_must_be_exported",
        "resolved_config_must_be_hashed",
        "clean_rebuild_hash_must_match",
        "portable_export_required",
    ):
        _require(composition.get(field) is True, f"{field} must remain true")
    _require(
        composition.get("portable_export_format") == "CANONICAL_JSON",
        "portable export must remain canonical JSON",
    )
    _require(
        composition.get("hydra_runtime_output_directory_is_authority") is False,
        "Hydra output directory cannot be lineage authority",
    )

    promotion = contract["promotion"]
    _require(
        promotion.get("states") == ["DISCOVERED", "CANDIDATE", "PARITY_PROVEN", "ADOPTED"],
        "promotion state machine drift",
    )
    _require(promotion.get("current_state") == "CANDIDATE", "qualification must stay CANDIDATE")
    _require(
        set(promotion.get("required_gates", [])) == EXPECTED_GATES,
        "required promotion gate set drift",
    )
    _require(
        promotion.get("this_contract_grants_adoption") is False,
        "qualification contract cannot grant adoption",
    )
    _require(
        promotion.get("this_contract_grants_stage_promotion") is False,
        "qualification contract cannot grant stage promotion",
    )

    truth = contract["truth_boundary"]
    for field in (
        "hydra_executed",
        "hydra_dependency_installed",
        "foreign_pretrained_weights_used",
        "corpus_or_tokenizer_mutated",
        "checkpoint_mutated",
        "paid_compute_used",
    ):
        _require(truth.get(field) is False, f"truth boundary overclaims {field}")
    _require(
        truth.get("evidence_scope") == "CONTRACT_AND_IDENTITY_MECHANICS_ONLY",
        "evidence scope must not overclaim Hydra execution",
    )
    _require(
        truth.get("canonical_base_random_init_only") is True,
        "canonical Base random-init boundary weakened",
    )

    return {
        "contract_id": contract["contract_id"],
        "contract_sha256": canonical_sha256(contract),
        "project_base_git_sha": authority["project_base_git_sha"],
        "registry_blob_sha": registry["blob_sha"],
        "upstream_commit_sha": upstream["commit_sha"],
        "current_state": promotion["current_state"],
    }


def _validate_defaults_trace(defaults_trace: Any) -> None:
    _require(isinstance(defaults_trace, list) and defaults_trace, "defaults_trace is required")
    seen_paths: set[str] = set()
    for index, entry in enumerate(defaults_trace):
        _require_exact_keys(entry, {"path", "sha256"}, f"defaults_trace[{index}]")
        path = entry["path"]
        digest = entry["sha256"]
        _require(isinstance(path, str) and path, "default path must be non-empty")
        _require(path not in seen_paths, f"duplicate default path: {path}")
        seen_paths.add(path)
        _require(
            isinstance(digest, str) and HEX64.fullmatch(digest) is not None,
            f"default {path} must carry lowercase sha256",
        )


def _validate_override_ledger(ledger: Any) -> None:
    _require(isinstance(ledger, list), "override_ledger must be an ordered list")
    seen_keys: set[str] = set()
    for index, entry in enumerate(ledger):
        _require_exact_keys(entry, {"source", "key", "value"}, f"override_ledger[{index}]")
        source = entry["source"]
        key = entry["key"]
        _require(source in ALLOWED_OVERRIDE_SOURCES, f"unapproved override source: {source}")
        _require(isinstance(key, str) and key, "override key must be non-empty")
        _require(key not in seen_keys, f"duplicate override key is ambiguous: {key}")
        seen_keys.add(key)


def portable_export_payload(observation: dict[str, Any]) -> dict[str, Any]:
    return {
        "base_git_sha": observation["base_git_sha"],
        "defaults_trace": observation["defaults_trace"],
        "override_ledger": observation["override_ledger"],
        "resolved_config": observation["resolved_config"],
    }


def validate_observation(
    contract: dict[str, Any], observation: dict[str, Any]
) -> dict[str, Any]:
    """Validate one resolved-config observation against the qualification contract."""
    contract_summary = validate_contract(contract)
    _require_exact_keys(
        observation,
        {
            "fixture_id",
            "base_git_sha",
            "defaults_trace",
            "override_ledger",
            "resolved_config",
            "resolved_config_sha256",
            "clean_rebuild_resolved_config_sha256",
            "portable_export_sha256",
            "hidden_overrides_detected",
            "runtime_environment_interpolation_detected",
            "secret_bearing_fields_present",
            "hydra_runtime_output_is_lineage_authority",
        },
        "observation",
    )
    _require(
        observation["base_git_sha"] == contract_summary["project_base_git_sha"],
        "observation is not bound to the contract base Git SHA",
    )
    _validate_defaults_trace(observation["defaults_trace"])
    _validate_override_ledger(observation["override_ledger"])
    _require(isinstance(observation["resolved_config"], dict), "resolved_config must be an object")

    resolved_sha = canonical_sha256(observation["resolved_config"])
    _require(
        observation["resolved_config_sha256"] == resolved_sha,
        "resolved_config_sha256 mismatch",
    )
    _require(
        observation["clean_rebuild_resolved_config_sha256"] == resolved_sha,
        "clean rebuild does not reproduce the exact resolved config",
    )

    portable_sha = canonical_sha256(portable_export_payload(observation))
    _require(
        observation["portable_export_sha256"] == portable_sha,
        "portable export hash mismatch",
    )
    _require(
        observation["hidden_overrides_detected"] is False,
        "hidden or implicit override detected",
    )
    _require(
        observation["runtime_environment_interpolation_detected"] is False,
        "runtime environment interpolation makes config identity non-portable",
    )
    _require(
        observation["secret_bearing_fields_present"] is False,
        "secret-bearing metadata must not enter reproducibility evidence",
    )
    _require(
        observation["hydra_runtime_output_is_lineage_authority"] is False,
        "Hydra runtime output cannot be canonical lineage authority",
    )

    return {
        "resolved_config_sha256": resolved_sha,
        "portable_export_sha256": portable_sha,
        "override_count": len(observation["override_ledger"]),
        "defaults_count": len(observation["defaults_trace"]),
    }


def build_evidence(contract: dict[str, Any], observation: dict[str, Any]) -> dict[str, Any]:
    """Build deterministic evidence. No timestamps are permitted by design."""
    contract_summary = validate_contract(contract)
    observation_summary = validate_observation(contract, observation)

    gates = {
        "UPSTREAM_IDENTITY_BOUND": True,
        "REGISTRY_BINDING_BOUND": True,
        "CANONICAL_AUTHORITY_PRESERVED": True,
        "EXPLICIT_OVERRIDE_LEDGER": True,
        "RESOLVED_CONFIG_BOUND": True,
        "DETERMINISTIC_REBUILD": True,
        "PORTABLE_ROLLBACK_EXPORT": True,
        "NO_SECRET_BEARING_METADATA": True,
    }
    _require(set(gates) == EXPECTED_GATES, "internal gate implementation drift")
    _require(all(gates.values()), "qualification evidence did not pass every required gate")

    identity_payload = {
        "contract_sha256": contract_summary["contract_sha256"],
        "project_base_git_sha": contract_summary["project_base_git_sha"],
        "registry_blob_sha": contract_summary["registry_blob_sha"],
        "upstream_commit_sha": contract_summary["upstream_commit_sha"],
        "portable_export_sha256": observation_summary["portable_export_sha256"],
    }
    evidence_core = {
        "schema_version": 1,
        "evidence_id": "12-6.hydra-config-qualification-evidence.v1",
        "contract_id": contract["contract_id"],
        "contract_sha256": contract_summary["contract_sha256"],
        "fixture_id": observation["fixture_id"],
        "project_base_git_sha": contract_summary["project_base_git_sha"],
        "registry_blob_sha": contract_summary["registry_blob_sha"],
        "upstream": copy.deepcopy(contract["upstream"]),
        "resolved_config_sha256": observation_summary["resolved_config_sha256"],
        "portable_export_sha256": observation_summary["portable_export_sha256"],
        "experiment_identity_sha256": canonical_sha256(identity_payload),
        "defaults_count": observation_summary["defaults_count"],
        "override_count": observation_summary["override_count"],
        "gates": gates,
        "verdict": "PASS_CONTRACT_MECHANICS_CANDIDATE_NOT_ADOPTED",
        "hydra_executed": False,
        "hydra_adopted": False,
        "stage_promotion_granted": False,
        "paid_compute_used": False,
    }
    evidence = copy.deepcopy(evidence_core)
    evidence["evidence_sha256"] = canonical_sha256(evidence_core)
    return evidence


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(data, dict), f"{path} must contain a JSON object")
    return data


def _write_canonical_pretty(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    path.write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("configs/research/hydra_config_qualification_v1.json"),
    )
    parser.add_argument("--observation", type=Path)
    parser.add_argument("--use-contract-fixture", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    try:
        contract = _load_json(args.contract)
        summary = validate_contract(contract)
        if args.observation and args.use_contract_fixture:
            raise QualificationError("choose --observation or --use-contract-fixture, not both")

        observation: dict[str, Any] | None = None
        if args.observation:
            observation = _load_json(args.observation)
        elif args.use_contract_fixture:
            observation = contract["project_owned_fixture"]

        if observation is None:
            print(json.dumps(summary, sort_keys=True))
            return 0

        evidence = build_evidence(contract, observation)
        if args.output:
            _write_canonical_pretty(args.output, evidence)
        print(json.dumps(evidence, sort_keys=True))
        return 0
    except (OSError, json.JSONDecodeError, QualificationError, TypeError, ValueError) as exc:
        print(f"HYDRA_QUALIFICATION_REJECTED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
