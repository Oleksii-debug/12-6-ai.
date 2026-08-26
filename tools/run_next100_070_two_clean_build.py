#!/usr/bin/env python3
"""NEXT100-070 deterministic two-clean-build preflight/materialization boundary.

This worker intentionally fails closed. It validates the exact late-bound registry and
rights vector, derives the current hard-gate state, and emits canonical byte-stable
manifests for every required build surface. When a prebuild hard gate fails, split
and shard payloads are not fabricated: their manifests record NOT_REACHED.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/data/next100_070_two_clean_build_v1.json"


class ValidationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_blob_sha1_bytes(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


def normalized_source_id(value: str) -> str:
    prefix = "external-real:"
    return value[len(prefix) :] if value.startswith(prefix) else value


def stratum_for(source: dict[str, Any]) -> str:
    if source.get("modality") == "code":
        return "code"
    language = source.get("language")
    if language == "python":
        return "code"
    require(language in {"uk", "en"}, f"unexpected language/stratum: {language!r}")
    return str(language)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(payload))


def validate_live_evidence(config: dict[str, Any]) -> None:
    live = config["live_evidence_authorities"]
    required = {"quality", "privacy", "dedup", "decontamination", "selection_validation", "unique_loss"}
    require(set(live) == required, "live evidence authority set drift")

    require(live["quality"]["current_candidate_scan"] is True, "DATA-296 current-candidate scan missing")
    require(live["quality"]["current_candidate_bytes"] == 183061, "DATA-296 byte scope drift")
    require(live["quality"]["release_gate_pass"] is False, "quality gate unexpectedly promoted")

    require(live["privacy"]["current_candidate_scan"] is True, "VERIFY-307 current-candidate scan missing")
    require(live["privacy"]["candidate_sources_pass"] == 5, "VERIFY-307 source coverage drift")
    require(live["privacy"]["candidate_findings"] == 0, "VERIFY-307 candidate finding count drift")
    require(live["privacy"]["release_gate_pass"] is True, "current candidate privacy scan is no longer PASS")

    require(live["dedup"]["source_level_current_candidate_scan"] is True, "DATA-298 source-level scan missing")
    require(live["dedup"]["source_level_bytes_before"] == 183061, "DATA-298 pre-dedup byte scope drift")
    require(live["dedup"]["source_level_bytes_after"] == 183061, "DATA-298 conservative capacity drift")
    require(live["dedup"]["source_level_duplicate_discount"] == 0, "DATA-298 duplicate discount drift")
    require(live["dedup"]["release_gate_pass"] is False, "dedup release gate unexpectedly promoted")
    require(
        live["dedup"]["observed_successor"]["workflow_state"] == "QUEUED_NOT_AUTHORITY",
        "nonterminal NEXT100-065 successor was accidentally promoted",
    )

    require(live["decontamination"]["scan_executed"] is False, "decontamination scan state drift")
    require(live["decontamination"]["release_gate_pass"] is False, "decontamination unexpectedly promoted")

    require(live["selection_validation"]["record_count"] > 0, "selection-validation unexpectedly empty")
    require(live["selection_validation"]["release_gate_pass"] is True, "terminal selection-validation not PASS")

    require(live["unique_loss"]["full_five_source_ledger"] is False, "five-source loss ledger unexpectedly present")
    require(live["unique_loss"]["release_gate_pass"] is False, "unique-loss gate unexpectedly promoted")


def validate_authorities(
    config: dict[str, Any], registry_path: Path, rights_path: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    require(config["schema_version"] == "12-6.next100-070-two-clean-build.v1", "config schema drift")
    require(config["execution_profile"] == "LOCAL_FREE", "execution is not LOCAL_FREE")
    require(config["repository"] == "Oleksii-debug/12-6-ai.", "repository identity drift")
    require(config["determinism_contract"]["shared_mutable_cache_allowed"] is False, "mutable cache was enabled")
    require(config["determinism_contract"]["clean_build_count"] == 2, "clean build count drift")
    validate_live_evidence(config)

    base = config["base_contract"]
    contract_path = ROOT / base["path"]
    contract_raw = contract_path.read_bytes()
    require(git_blob_sha1_bytes(contract_raw) == base["git_blob_sha1"], "DATA-300 contract blob drift")
    contract = json.loads(contract_raw.decode("utf-8"))
    require(contract["contract_identity_sha256"] == base["contract_identity_sha256"], "DATA-300 identity drift")
    require(contract["execution_profile"] == "LOCAL_FREE", "DATA-300 execution profile drift")

    registry_raw = registry_path.read_bytes()
    registry_ref = config["late_bound_authorities"]["registry"]
    require(git_blob_sha1_bytes(registry_raw) == registry_ref["git_blob_sha1"], "late-bound registry blob drift")
    registry = json.loads(registry_raw.decode("utf-8"))
    require(registry["registry_identity_sha256"] == registry_ref["registry_identity_sha256"], "registry identity drift")
    require(registry["local_free_only"] is True, "registry is not LOCAL_FREE-only")

    rights_raw = rights_path.read_bytes()
    rights_ref = config["late_bound_authorities"]["rights"]
    require(git_blob_sha1_bytes(rights_raw) == rights_ref["git_blob_sha1"], "late-bound rights blob drift")
    rights = json.loads(rights_raw.decode("utf-8"))
    require(rights["local_free_only"] is True, "rights authority is not LOCAL_FREE-only")

    expected = config["expected_candidate"]
    require(registry["source_count"] == expected["source_count"], "source count drift")
    require(registry["independent_source_family_count"] == expected["independent_family_count"], "family count drift")
    require(registry["byte_report"]["unique_normalized_bytes"] == expected["unique_normalized_bytes"], "normalized byte total drift")

    registry_sources = {normalized_source_id(row["source_id"]): row for row in registry["sources"]}
    contract_sources = {
        normalized_source_id(row["source_id"]): row
        for row in contract["exact_training_candidate_inventory"]["sources"]
    }
    rights_sources = {normalized_source_id(row["source_id"]): row for row in rights["admitted"]}
    require(set(registry_sources) == set(contract_sources), "DATA-287 registry no longer matches DATA-300 source set")
    require(set(registry_sources) == set(rights_sources), "DATA-293 rights no longer matches current registry source set")

    normalized_records: list[dict[str, Any]] = []
    family_sets: dict[str, set[str]] = {"uk": set(), "en": set(), "code": set()}
    for source_id in sorted(registry_sources):
        reg = registry_sources[source_id]
        frozen = contract_sources[source_id]
        right = rights_sources[source_id]
        family = reg["independent_source_family"]["family_id"]
        require(family == frozen["family"] == right["source_family"], f"family drift: {source_id}")
        require(
            int(reg["snapshot"]["normalized_bytes"]) == int(frozen["normalized_bytes"]),
            f"normalized bytes drift: {source_id}",
        )
        if "normalized_sha256" in frozen:
            require(
                reg["snapshot"]["normalized_sha256"] == frozen["normalized_sha256"],
                f"normalized hash drift: {source_id}",
            )
        else:
            require(reg["snapshot"]["raw_sha256"] == frozen["raw_sha256"], f"raw hash drift: {source_id}")
        training_right = str(right["rights"]["model_training"])
        require(training_right.startswith("ALLOWED"), f"training rights not allowed: {source_id}")
        reg_training = str(reg["rights"]["model_training"]["status"])
        require(reg_training == "ALLOWED", f"registry training right not ALLOWED: {source_id}")
        stratum = stratum_for(reg)
        family_sets[stratum].add(family)
        normalized_records.append(
            {
                "source_id": source_id,
                "family": family,
                "stratum": stratum,
                "normalized_bytes": int(reg["snapshot"]["normalized_bytes"]),
                "normalized_sha256": reg["snapshot"]["normalized_sha256"],
                "normalization_policy": reg["snapshot"]["normalization_policy"],
            }
        )

    observed_family_counts = {key: len(value) for key, value in family_sets.items()}
    require(observed_family_counts == expected["family_counts"], f"family count vector drift: {observed_family_counts}")
    require(
        sum(row["normalized_bytes"] for row in normalized_records) == expected["unique_normalized_bytes"],
        "record bytes do not sum to expected candidate capacity",
    )
    require(len(rights["admitted"]) == expected["source_count"], "rights admitted count drift")
    return contract, registry, rights, normalized_records


def build_outputs(
    config: dict[str, Any],
    contract: dict[str, Any],
    registry: dict[str, Any],
    rights: dict[str, Any],
    normalized_records: list[dict[str, Any]],
    output_root: Path,
) -> dict[str, Any]:
    if output_root.exists():
        require(not any(output_root.iterdir()), "output root must be clean and empty")
    output_root.mkdir(parents=True, exist_ok=True)

    expected = config["expected_candidate"]
    live = config["live_evidence_authorities"]
    min_families = int(expected["minimum_independent_families_per_stratum"])
    family_counts = expected["family_counts"]
    diversity_pass = all(int(family_counts[key]) >= min_families for key in ("uk", "en", "code"))
    require(
        diversity_pass is False,
        "current candidate unexpectedly satisfies family-diversity gate; successor build authority required",
    )
    require(int(expected["family_constrained_no_replay_budget"]) == 0, "no-replay budget unexpectedly nonzero")

    authority_binding = {
        "registry": {
            "worker": config["late_bound_authorities"]["registry"]["worker"],
            "head_sha": config["late_bound_authorities"]["registry"]["head_sha"],
            "git_blob_sha1": config["late_bound_authorities"]["registry"]["git_blob_sha1"],
            "registry_identity_sha256": registry["registry_identity_sha256"],
        },
        "rights": {
            "worker": config["late_bound_authorities"]["rights"]["worker"],
            "head_sha": config["late_bound_authorities"]["rights"]["head_sha"],
            "git_blob_sha1": config["late_bound_authorities"]["rights"]["git_blob_sha1"],
        },
        "data300_contract_identity_sha256": contract["contract_identity_sha256"],
    }

    write_json(
        output_root / "normalized_records/manifest.json",
        {
            "schema_version": "12-6.next100-070-normalized-record-manifest.v1",
            "state": "IDENTITIES_BOUND_PAYLOAD_MATERIALIZATION_NOT_REACHED",
            "reason": "prebuild hard-gate vector fails before corpus product materialization",
            "source_count": len(normalized_records),
            "unique_normalized_bytes": sum(row["normalized_bytes"] for row in normalized_records),
            "records": normalized_records,
            "authority_binding": authority_binding,
        },
    )

    rights_map = {normalized_source_id(row["source_id"]): row for row in rights["admitted"]}
    write_json(
        output_root / "rights_manifest.json",
        {
            "schema_version": "12-6.next100-070-rights-manifest.v1",
            "state": "PASS_EXACT_SOURCE_OBJECT_RIGHTS_BOUND",
            "source_count": len(rights_map),
            "sources": [
                {
                    "source_id": source_id,
                    "family": rights_map[source_id]["source_family"],
                    "license_id": rights_map[source_id]["license_id"],
                    "model_training": rights_map[source_id]["rights"]["model_training"],
                    "redistribution": rights_map[source_id]["rights"]["redistribution"],
                    "evaluation": rights_map[source_id]["rights"]["evaluation"],
                }
                for source_id in sorted(rights_map)
            ],
            "authority_binding": authority_binding["rights"],
        },
    )

    vector = config["prebuild_evidence_vector"]
    component_lock = contract.get("terminal_component_lock", {})
    evidence_specs = {
        "quality_evidence.json": ("G05_QUALITY", "quality", "current_candidate_scan"),
        "privacy_evidence.json": ("G06_PRIVACY", "privacy", "current_candidate_scan"),
        "dedup_evidence.json": ("G07_DEDUP", "dedup", "source_level_current_candidate_scan"),
    }
    for filename, (gate, component, scan_key) in evidence_specs.items():
        authority = live[component]
        write_json(
            output_root / filename,
            {
                "schema_version": f"12-6.next100-070-{component}-evidence.v1",
                "gate": gate,
                "state": vector[gate],
                "current_candidate_source_count": expected["source_count"],
                "current_candidate_unique_normalized_bytes": expected["unique_normalized_bytes"],
                "bound_exact_current_candidate_scan": bool(authority[scan_key]),
                "release_gate_pass": bool(authority["release_gate_pass"]),
                "live_authority": authority,
                "frozen_contract_component_observation": component_lock.get(component),
                "claim_boundary": "BOUND_TERMINAL_STATUS_EVIDENCE_NOT_A_NEW_CONTENT_SCAN_BY_NEXT100_070",
            },
        )

    write_json(
        output_root / "decontamination_evidence.json",
        {
            "schema_version": "12-6.next100-070-decontamination-evidence.v1",
            "gate": "G08_RESERVED_DECONTAMINATION",
            "state": vector["G08_RESERVED_DECONTAMINATION"],
            "exact_corpus_identity": None,
            "scan_executed_by_this_worker": False,
            "live_authority": live["decontamination"],
            "claim_boundary": "NO_CONTAMINATION_PASS_CLAIM_WITHOUT_EXACT_CORPUS_IDENTITY",
        },
    )

    write_json(
        output_root / "loss_ledger.json",
        {
            "schema_version": "12-6.next100-070-loss-ledger-status.v1",
            "gate": "G12_UNIQUE_LOSS",
            "state": vector["G12_UNIQUE_LOSS"],
            "full_current_candidate_ledger_available": False,
            "authorized_balanced_no_replay_capacity": expected["family_constrained_no_replay_budget"],
            "live_authority": live["unique_loss"],
            "frozen_contract_component_observation": component_lock.get("unique_loss"),
            "claim_boundary": "NO_FULL_FIVE_SOURCE_CAUSAL_LOSS_LEDGER_INVENTED",
        },
    )

    gate_states = dict(vector)
    gate_states["G01_CONTRACT_IDENTITY"] = "PASS"
    gate_states["G03_SOURCE_INVENTORY"] = "PASS_LATE_BOUND_REGISTRY_MATCHES_FROZEN_CANDIDATE"
    gate_states["G04_RIGHTS"] = "PASS_LATE_BOUND_PURPOSE_SPECIFIC_RIGHTS_MATCH"
    gate_states["G09_BALANCE_DIVERSITY"] = "FAIL_FAMILY_DIVERSITY_NO_REPLAY_BUDGET_ZERO"
    blocking = sorted(gate for gate, state in gate_states.items() if not state.startswith("PASS"))

    nonproduct = {
        "schema_version": "12-6.next100-070-unreached-product-manifest.v1",
        "state": "NOT_REACHED_PREBUILD_HARD_GATES",
        "blocking_gates": blocking,
        "corpus_identity": None,
        "payload_file_count": 0,
        "claim_boundary": "NO_EMPTY_PAYLOAD_IS_COUNTED_AS_A_CORPUS_PRODUCT",
    }
    write_json(output_root / "split_manifests/manifest.json", {**nonproduct, "surface": "split_manifests"})
    write_json(output_root / "shards/manifest.json", {**nonproduct, "surface": "shards"})

    write_json(
        output_root / "gate_report.json",
        {
            "schema_version": "12-6.next100-070-gate-report.v1",
            "status": "BLOCKED_DETERMINISTIC_PREFLIGHT_COMPLETE",
            "gate_states": gate_states,
            "blocking_gates": blocking,
            "family_counts": family_counts,
            "minimum_independent_families_per_stratum": min_families,
            "family_constrained_no_replay_budget": expected["family_constrained_no_replay_budget"],
            "source_payload_hydration": "NOT_REACHED_AFTER_PREBUILD_HARD_GATE_FAILURE",
            "split_materialization": "NOT_REACHED",
            "shard_materialization": "NOT_REACHED",
            "model_training_executed": False,
            "authority_binding": authority_binding,
            "live_evidence_authorities": live,
        },
    )

    tree_rows: list[dict[str, Any]] = []
    for path in sorted(p for p in output_root.rglob("*") if p.is_file()):
        rel = path.relative_to(output_root).as_posix()
        data = path.read_bytes()
        tree_rows.append({"path": rel, "bytes": len(data), "sha256": sha256_bytes(data)})
    write_json(
        output_root / "tree_manifest.json",
        {
            "schema_version": "12-6.next100-070-tree-manifest.v1",
            "state": "DETERMINISTIC_BLOCKER_TREE",
            "manifest_scope": "all emitted files except tree_manifest.json itself",
            "entries": tree_rows,
            "entry_count": len(tree_rows),
            "authority_binding": authority_binding,
        },
    )

    required = set(config["determinism_contract"]["required_surfaces"])
    actual = {
        path.relative_to(output_root).as_posix()
        for path in output_root.rglob("*")
        if path.is_file()
    }
    require(
        actual == required,
        f"output path set drift: missing={sorted(required - actual)} extra={sorted(actual - required)}",
    )

    final_rows = []
    for path in sorted(p for p in output_root.rglob("*") if p.is_file()):
        rel = path.relative_to(output_root).as_posix()
        data = path.read_bytes()
        final_rows.append(f"{sha256_bytes(data)}  {len(data)}  {rel}")
    tree_listing = "\n".join(final_rows) + "\n"
    return {
        "status": "BLOCKED_DETERMINISTIC_PREFLIGHT_COMPLETE",
        "blocking_gates": blocking,
        "output_file_count": len(final_rows),
        "tree_listing_sha256": sha256_bytes(tree_listing.encode("utf-8")),
        "tree_listing": tree_listing,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("attempt",))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--rights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = load_json(args.config)
    contract, registry, rights, records = validate_authorities(config, args.registry, args.rights)
    result = build_outputs(config, contract, registry, rights, records, args.output)
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "tree_listing"},
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    print(result["tree_listing"], end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
