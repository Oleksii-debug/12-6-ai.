"""Fail-closed authorization gate for the first learned ~20M 12-6 Base campaign.

The gate does not train a model and cannot grant authorization. It derives one of
BLOCKED, READY_FOR_AUTHORIZATION_REQUEST, or TRAINING_AUTHORIZED from explicit
machine-readable evidence. The final state requires separate compute and training
authorization records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

PACKET_SCHEMA = "12-6.learned-20m-launch-authorization.v1"
RESULT_SCHEMA = "12-6.learned-20m-launch-gate-result.v1"

BLOCKED = "BLOCKED"
READY = "READY_FOR_AUTHORIZATION_REQUEST"
AUTHORIZED = "TRAINING_AUTHORIZED"

EXPECTED_MODEL341 = {
    "branch": "model341/20m-candidate-a-20260826",
    "sha": "e4ff486fd90802fc123bebf60eed4e59196a98df",
    "modelspec_sha256": "fbff24d561a2818453554d58ca23fc6ace3303b078f1935a8576c4565bd92441",
    "parameter_count": 20_613_440,
    "canonical_base": "random_init",
}

EXPECTED_R01 = {
    "path": "configs/research/r01_20m_to_100m_scaling_campaign_v1.json",
    "git_blob_sha": "c50154db609d41eceb2ffc97912360df567bcc04",
    "campaign_id": "R01-20M-TO-100M-SCALING-V1",
    "experiment_id": "R01-E20",
}

REQUIRED_EVIDENCE_GATES = (
    "corpus_admission_and_provenance",
    "evaluation_decontamination",
    "packing_and_unique_loss_ledger",
    "checkpoint_integrity_d05",
    "learned_3m_independent",
    "learned_10m_independent",
    "selection_validation",
    "independent_audit",
)

SHA256_IDENTITY_FIELDS = (
    "corpus_identity_sha256",
    "split_identity_sha256",
    "packing_identity_sha256",
    "tokenizer_identity_sha256",
    "run_config_sha256",
)


def _is_hex(value: object, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(char in "0123456789abcdef" for char in value)
    )


def _positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _positive_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0


def _nonnegative_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0


def _nonempty(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _block(blockers: list[dict[str, str]], code: str, detail: str) -> None:
    blockers.append({"code": code, "detail": detail})


def git_blob_sha(path: Path) -> str:
    """Return the Git SHA-1 object identity for a normal file blob."""
    data = path.read_bytes()
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()  # noqa: S324 - Git object identity


def git_head(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _validate_authority(
    packet: dict[str, Any],
    blockers: list[dict[str, str]],
    *,
    repo_root: Path | None,
) -> None:
    authority = packet.get("authority")
    if not isinstance(authority, dict):
        _block(blockers, "AUTHORITY_MISSING", "authority must be an object")
        return

    model = authority.get("model341")
    if not isinstance(model, dict):
        _block(blockers, "MODEL341_AUTHORITY_MISSING", "authority.model341 must be an object")
    else:
        for key, expected in EXPECTED_MODEL341.items():
            if model.get(key) != expected:
                _block(
                    blockers,
                    "MODEL341_AUTHORITY_MISMATCH",
                    f"authority.model341.{key} does not match qualified MODEL-341",
                )

    r01 = authority.get("r01")
    if not isinstance(r01, dict):
        _block(blockers, "R01_BINDING_MISSING", "authority.r01 must be an object")
    else:
        for key, expected in EXPECTED_R01.items():
            if r01.get(key) != expected:
                _block(
                    blockers,
                    "R01_BINDING_MISMATCH",
                    f"authority.r01.{key} does not match merged R01 campaign",
                )
        if repo_root is not None:
            campaign_path = repo_root / EXPECTED_R01["path"]
            if not campaign_path.is_file():
                _block(blockers, "R01_FILE_MISSING", "merged R01 campaign file is missing")
            elif git_blob_sha(campaign_path) != EXPECTED_R01["git_blob_sha"]:
                _block(
                    blockers,
                    "R01_FILE_DRIFT",
                    "merged R01 campaign no longer matches the bound Git blob",
                )

    source_sha = authority.get("source_git_sha")
    if not _is_hex(source_sha, 40):
        _block(
            blockers,
            "SOURCE_SHA_MISSING",
            "authority.source_git_sha must be an exact lowercase 40-character Git SHA",
        )
    elif repo_root is not None:
        try:
            observed_head = git_head(repo_root)
        except (OSError, subprocess.CalledProcessError):
            _block(blockers, "SOURCE_SHA_UNVERIFIABLE", "cannot resolve repository HEAD")
        else:
            if observed_head != source_sha:
                _block(
                    blockers,
                    "SOURCE_SHA_MISMATCH",
                    "launch packet source_git_sha does not equal repository HEAD",
                )


def _validate_identities(packet: dict[str, Any], blockers: list[dict[str, str]]) -> None:
    identities = packet.get("identities")
    if not isinstance(identities, dict):
        _block(blockers, "IDENTITIES_MISSING", "identities must be an object")
        return
    for field in SHA256_IDENTITY_FIELDS:
        if not _is_hex(identities.get(field), 64):
            _block(
                blockers,
                "IDENTITY_MISSING",
                f"identities.{field} must be an exact lowercase SHA-256",
            )


def _validate_exposure(packet: dict[str, Any], blockers: list[dict[str, str]]) -> None:
    exposure = packet.get("post_pack_exposure")
    if not isinstance(exposure, dict):
        _block(blockers, "EXPOSURE_MISSING", "post_pack_exposure must be an object")
        return

    if not _positive_int(exposure.get("unique_causal_loss_positions")):
        _block(
            blockers,
            "NO_UNIQUE_POST_PACK_EXPOSURE",
            "unique_causal_loss_positions must be a positive integer",
        )
    for field in ("replayed_loss_positions", "duplicate_loss_positions", "padding_loss_positions"):
        value = exposure.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value != 0:
            _block(
                blockers,
                "NON_UNIQUE_EXPOSURE",
                f"post_pack_exposure.{field} must be exactly 0",
            )
    if not _is_hex(exposure.get("ledger_sha256"), 64):
        _block(
            blockers,
            "LOSS_LEDGER_IDENTITY_MISSING",
            "post_pack_exposure.ledger_sha256 must be an exact SHA-256",
        )


def _validate_evidence(packet: dict[str, Any], blockers: list[dict[str, str]]) -> None:
    evidence = packet.get("terminal_evidence")
    if not isinstance(evidence, dict):
        _block(blockers, "TERMINAL_EVIDENCE_MISSING", "terminal_evidence must be an object")
        return
    for gate in REQUIRED_EVIDENCE_GATES:
        record = evidence.get(gate)
        if not isinstance(record, dict):
            _block(blockers, "GATE_MISSING", f"terminal_evidence.{gate} is missing")
            continue
        if record.get("terminal") is not True:
            _block(blockers, "GATE_NOT_TERMINAL", f"terminal_evidence.{gate} is not terminal")
        if not _nonempty(record.get("evidence_ref")):
            _block(
                blockers,
                "GATE_EVIDENCE_REF_MISSING",
                f"terminal_evidence.{gate}.evidence_ref is required",
            )


def _validate_recipe(packet: dict[str, Any], blockers: list[dict[str, str]]) -> None:
    recipe = packet.get("training_recipe")
    if not isinstance(recipe, dict):
        _block(blockers, "TRAINING_RECIPE_MISSING", "training_recipe must be an object")
        return

    for section_name in ("optimizer", "scheduler", "precision"):
        section = recipe.get(section_name)
        if not isinstance(section, dict):
            _block(
                blockers,
                "RECIPE_SECTION_MISSING",
                f"training_recipe.{section_name} must be an object",
            )
            continue
        if not _nonempty(section.get("name")):
            _block(
                blockers,
                "RECIPE_VALUE_MISSING",
                f"training_recipe.{section_name}.name is required",
            )
        if not _is_hex(section.get("config_sha256"), 64):
            _block(
                blockers,
                "RECIPE_IDENTITY_MISSING",
                f"training_recipe.{section_name}.config_sha256 must be SHA-256",
            )

    seeds = recipe.get("seeds")
    seeds_valid = (
        isinstance(seeds, list)
        and bool(seeds)
        and all(isinstance(seed, int) and not isinstance(seed, bool) and seed >= 0 for seed in seeds)
        and len(set(seeds)) == len(seeds)
    )
    if not seeds_valid:
        _block(
            blockers,
            "SEEDS_INVALID",
            "training_recipe.seeds must be a non-empty unique list of non-negative integers",
        )

    budget = recipe.get("budget")
    if not isinstance(budget, dict):
        _block(blockers, "RUN_BUDGET_MISSING", "training_recipe.budget must be an object")
    else:
        for field in ("target_optimized_tokens", "max_steps", "max_wall_minutes"):
            if not _positive_int(budget.get(field)):
                _block(
                    blockers,
                    "RUN_BUDGET_INVALID",
                    f"training_recipe.budget.{field} must be a positive integer",
                )

    stop_rules = recipe.get("stop_rules")
    if (
        not isinstance(stop_rules, list)
        or not stop_rules
        or any(not _nonempty(rule) for rule in stop_rules)
    ):
        _block(
            blockers,
            "STOP_RULES_MISSING",
            "training_recipe.stop_rules must be a non-empty list of explicit rules",
        )


def _validate_resources(packet: dict[str, Any], blockers: list[dict[str, str]]) -> None:
    resources = packet.get("resource_envelope")
    if not isinstance(resources, dict):
        _block(blockers, "RESOURCE_ENVELOPE_MISSING", "resource_envelope must be an object")
        return
    if not _positive_number(resources.get("estimated_flops")):
        _block(blockers, "FLOP_ESTIMATE_MISSING", "estimated_flops must be positive")
    if not _positive_number(resources.get("estimated_wall_clock_hours")):
        _block(blockers, "WALL_CLOCK_ESTIMATE_MISSING", "estimated_wall_clock_hours must be positive")
    if not _positive_int(resources.get("device_count")):
        _block(blockers, "DEVICE_COUNT_INVALID", "device_count must be a positive integer")
    if not _nonempty(resources.get("device_type")):
        _block(blockers, "DEVICE_TYPE_MISSING", "device_type is required")
    if not _nonnegative_number(resources.get("max_cost_usd")):
        _block(blockers, "COST_ENVELOPE_INVALID", "max_cost_usd must be non-negative")
    if not _nonempty(resources.get("estimate_evidence_ref")):
        _block(blockers, "RESOURCE_EVIDENCE_MISSING", "estimate_evidence_ref is required")


def _authorization_state(
    packet: dict[str, Any], blockers: list[dict[str, str]]
) -> tuple[bool, bool, list[str]]:
    auth = packet.get("authorizations")
    if not isinstance(auth, dict):
        return False, False, ["compute", "training"]

    results: dict[str, bool] = {}
    missing: list[str] = []
    resource = packet.get("resource_envelope")
    planned_cost = resource.get("max_cost_usd") if isinstance(resource, dict) else None

    for kind, authorized_value in (
        ("compute", "COMPUTE_AUTHORIZED"),
        ("training", "TRAINING_AUTHORIZED"),
    ):
        record = auth.get(kind)
        if not isinstance(record, dict):
            missing.append(kind)
            results[kind] = False
            continue
        decision = record.get("decision")
        if decision not in {authorized_value, "NOT_AUTHORIZED"}:
            _block(
                blockers,
                "AUTHORIZATION_DECISION_INVALID",
                f"authorizations.{kind}.decision must be {authorized_value} or NOT_AUTHORIZED",
            )
            missing.append(kind)
            results[kind] = False
            continue
        if decision == "NOT_AUTHORIZED":
            missing.append(kind)
            results[kind] = False
            continue
        if not _nonempty(record.get("reference")):
            _block(
                blockers,
                "AUTHORIZATION_REFERENCE_MISSING",
                f"authorizations.{kind}.reference is required for {authorized_value}",
            )
            missing.append(kind)
            results[kind] = False
            continue
        if kind == "compute":
            approved = record.get("max_cost_usd")
            if not _nonnegative_number(approved):
                _block(blockers, "COMPUTE_BUDGET_INVALID", "authorized max_cost_usd is invalid")
                missing.append(kind)
                results[kind] = False
                continue
            if _nonnegative_number(planned_cost) and approved < planned_cost:
                _block(
                    blockers,
                    "COMPUTE_BUDGET_TOO_SMALL",
                    "compute authorization is below the declared cost envelope",
                )
                missing.append(kind)
                results[kind] = False
                continue
        results[kind] = True

    return results.get("compute", False), results.get("training", False), missing


def evaluate_packet(
    packet: dict[str, Any], *, repo_root: Path | None = None
) -> dict[str, Any]:
    """Derive launch state from evidence; an input status field is never trusted."""
    blockers: list[dict[str, str]] = []
    if packet.get("schema") != PACKET_SCHEMA:
        _block(blockers, "SCHEMA_MISMATCH", f"schema must be {PACKET_SCHEMA}")

    _validate_authority(packet, blockers, repo_root=repo_root)
    _validate_identities(packet, blockers)
    _validate_exposure(packet, blockers)
    _validate_evidence(packet, blockers)
    _validate_recipe(packet, blockers)
    _validate_resources(packet, blockers)
    compute_authorized, training_authorized, missing_auth = _authorization_state(packet, blockers)

    if blockers:
        state = BLOCKED
    elif compute_authorized and training_authorized:
        state = AUTHORIZED
    else:
        state = READY

    return {
        "schema": RESULT_SCHEMA,
        "state": state,
        "training_may_start": state == AUTHORIZED,
        "blockers": blockers,
        "authorization_missing": sorted(set(missing_auth)),
        "authority": {
            "model341_sha": EXPECTED_MODEL341["sha"],
            "modelspec_sha256": EXPECTED_MODEL341["modelspec_sha256"],
            "parameter_count": EXPECTED_MODEL341["parameter_count"],
            "r01_git_blob_sha": EXPECTED_R01["git_blob_sha"],
            "experiment_id": EXPECTED_R01["experiment_id"],
        },
    }


def evaluate_path(path: Path, *, repo_root: Path | None = None) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("launch packet root must be an object")
    return evaluate_packet(data, repo_root=repo_root)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "packet",
        nargs="?",
        type=Path,
        default=Path("configs/launch/learned_20m_authorization_v1.json"),
    )
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="return zero after evaluation even when training is not authorized",
    )
    args = parser.parse_args(argv)
    repo_root = Path.cwd().resolve()
    result = evaluate_path(args.packet, repo_root=repo_root)
    print(json.dumps(result, sort_keys=True, indent=2))
    if args.inspect_only:
        return 0
    return 0 if result["state"] == AUTHORIZED else 2


if __name__ == "__main__":
    raise SystemExit(main())
