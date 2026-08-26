#!/usr/bin/env python3
"""Validate and assess the fail-closed learned-20M launch packet."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

EXPECTED_AUTHORITY = {
    "template_main_sha": "a73ab38026cb7849f478cc13ad58b93534a76e2f",
    "model341_branch": "model341/20m-candidate-a-20260826",
    "model341_sha": "e4ff486fd90802fc123bebf60eed4e59196a98df",
    "modelspec_sha256": "fbff24d561a2818453554d58ca23fc6ace3303b078f1935a8576c4565bd92441",
    "parameter_count": 20613440,
    "r01_contract_path": "configs/research/r01_20m_to_100m_scaling_campaign_v1.json",
    "r01_contract_git_blob_sha1": "c50154db609d41eceb2ffc97912360df567bcc04",
}

HEX40 = frozenset("0123456789abcdef")


def _is_hex(value: object, length: int) -> bool:
    return isinstance(value, str) and len(value) == length and all(ch in HEX40 for ch in value)


def _positive_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0


def _nonempty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _expect(errors: list[str], condition: bool, message: str) -> None:
    if not condition:
        errors.append(message)


def git_blob_sha1(payload: bytes) -> str:
    header = f"blob {len(payload)}\0".encode("ascii")
    return hashlib.sha1(header + payload).hexdigest()  # noqa: S324 - Git object identity requires SHA-1.


def _scientific_blockers(data: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    evidence = data.get("launch_evidence")
    if not isinstance(evidence, dict):
        return ["launch_evidence_missing"]

    if not _is_hex(evidence.get("code_commit_sha"), 40):
        blockers.append("code_commit_sha_missing")

    tokenizer = evidence.get("tokenizer")
    if not isinstance(tokenizer, dict):
        blockers.append("tokenizer_evidence_missing")
    else:
        byte_selected = tokenizer.get("byte_baseline_explicitly_selected") is True
        identity_ok = _is_hex(tokenizer.get("identity_sha256"), 64)
        fit_ok = _is_hex(tokenizer.get("fit_corpus_identity_sha256"), 64)
        if not (byte_selected or (identity_ok and fit_ok)):
            blockers.append("tokenizer_identity_not_qualified")

    corpus = evidence.get("corpus")
    if not isinstance(corpus, dict):
        blockers.append("corpus_evidence_missing")
    else:
        for key in ("corpus_identity_sha256", "split_identity_sha256", "packing_identity_sha256"):
            if not _is_hex(corpus.get(key), 64):
                blockers.append(f"{key}_missing")
        if not isinstance(corpus.get("unique_causal_loss_positions"), int) or isinstance(
            corpus.get("unique_causal_loss_positions"), bool
        ) or corpus.get("unique_causal_loss_positions", 0) <= 0:
            blockers.append("unique_causal_loss_positions_zero")
        if corpus.get("no_replay_proven") is not True:
            blockers.append("no_replay_not_proven")

    evaluation = evidence.get("evaluation")
    if not isinstance(evaluation, dict):
        blockers.append("evaluation_evidence_missing")
    else:
        if not _is_hex(evaluation.get("reservation_identity_sha256"), 64):
            blockers.append("evaluation_reservation_missing")
        if not _is_hex(evaluation.get("decontamination_identity_sha256"), 64):
            blockers.append("evaluation_decontamination_missing")
        if not _nonempty_string(evaluation.get("selection_validation_authority_ref")):
            blockers.append("selection_validation_not_terminal")
        if evaluation.get("final_test_firewall_preregistered") is not True:
            blockers.append("final_test_firewall_missing")

    checkpoint = evidence.get("checkpoint_integrity")
    if not isinstance(checkpoint, dict):
        blockers.append("checkpoint_integrity_missing")
    else:
        if not _nonempty_string(checkpoint.get("authority_ref")):
            blockers.append("checkpoint_authority_missing")
        if checkpoint.get("terminal_retest_passed") is not True:
            blockers.append("checkpoint_terminal_retest_missing")
        if checkpoint.get("fresh_process_resume_passed") is not True:
            blockers.append("fresh_process_resume_missing")

    ladder = evidence.get("learned_ladder")
    if not isinstance(ladder, dict):
        blockers.append("learned_ladder_missing")
    else:
        if not _nonempty_string(ladder.get("learned_3m_authority_ref")):
            blockers.append("learned_3m_authority_missing")
        if not _nonempty_string(ladder.get("learned_10m_authority_ref")):
            blockers.append("learned_10m_authority_missing")
        if ladder.get("independently_verified") is not True:
            blockers.append("learned_ladder_not_independently_verified")

    recipe = evidence.get("training_recipe")
    if not isinstance(recipe, dict):
        blockers.append("training_recipe_missing")
    else:
        for key in ("optimizer", "scheduler", "precision", "warmup", "gradient_policy"):
            if not _nonempty_string(recipe.get(key)):
                blockers.append(f"training_recipe_{key}_missing")
        if not _positive_number(recipe.get("learning_rate")):
            blockers.append("training_recipe_learning_rate_invalid")
        seeds = recipe.get("seeds")
        if not isinstance(seeds, list) or not seeds or any(
            not isinstance(seed, int) or isinstance(seed, bool) or seed < 0 for seed in seeds
        ):
            blockers.append("training_recipe_seeds_invalid")
        if not isinstance(recipe.get("total_unique_loss_positions"), int) or isinstance(
            recipe.get("total_unique_loss_positions"), bool
        ) or recipe.get("total_unique_loss_positions", 0) <= 0:
            blockers.append("training_recipe_budget_missing")
        if recipe.get("budget_matches_corpus_ledger") is not True:
            blockers.append("training_recipe_budget_not_bound_to_corpus")
        stopping = recipe.get("stopping_rules")
        if not isinstance(stopping, list) or not stopping or any(not _nonempty_string(x) for x in stopping):
            blockers.append("training_recipe_stopping_rules_missing")

    resources = evidence.get("resource_envelope")
    if not isinstance(resources, dict):
        blockers.append("resource_envelope_missing")
    else:
        if not _nonempty_string(resources.get("accelerator_profile")):
            blockers.append("accelerator_profile_missing")
        if not _positive_number(resources.get("estimated_flops")):
            blockers.append("estimated_flops_missing")
        if not _positive_number(resources.get("estimated_wall_clock_hours")):
            blockers.append("estimated_wall_clock_missing")
        max_cost = resources.get("max_cost_usd")
        if not isinstance(max_cost, (int, float)) or isinstance(max_cost, bool) or max_cost < 0:
            blockers.append("max_cost_usd_missing")
        if not _nonempty_string(resources.get("cost_estimate_authority_ref")):
            blockers.append("cost_estimate_authority_missing")

    if isinstance(corpus, dict) and isinstance(recipe, dict):
        if (
            isinstance(corpus.get("unique_causal_loss_positions"), int)
            and not isinstance(corpus.get("unique_causal_loss_positions"), bool)
            and isinstance(recipe.get("total_unique_loss_positions"), int)
            and not isinstance(recipe.get("total_unique_loss_positions"), bool)
            and corpus.get("unique_causal_loss_positions") > 0
            and recipe.get("total_unique_loss_positions") > corpus.get("unique_causal_loss_positions")
        ):
            blockers.append("training_budget_exceeds_unique_corpus_ledger")

    return sorted(set(blockers))


def derive_state(data: dict[str, Any]) -> tuple[str, list[str]]:
    blockers = _scientific_blockers(data)
    if blockers:
        return "BLOCKED", blockers

    authorizations = data.get("authorizations")
    if not isinstance(authorizations, dict):
        return "BLOCKED", ["authorizations_missing"]

    compute = authorizations.get("compute")
    training = authorizations.get("training")
    if not isinstance(compute, dict) or not isinstance(training, dict):
        return "BLOCKED", ["authorizations_malformed"]

    compute_status = compute.get("status")
    training_status = training.get("status")
    compute_ref = compute.get("authority_ref")
    training_ref = training.get("authority_ref")

    if (
        compute_status == "NOT_AUTHORIZED"
        and training_status == "NOT_AUTHORIZED"
        and compute_ref is None
        and training_ref is None
    ):
        return "READY_FOR_AUTHORIZATION_REQUEST", []

    if (
        compute_status == "AUTHORIZED"
        and training_status == "AUTHORIZED"
        and _nonempty_string(compute_ref)
        and _nonempty_string(training_ref)
    ):
        return "TRAINING_AUTHORIZED", []

    return "BLOCKED", ["authorization_state_inconsistent"]


def validate_packet(data: dict[str, Any], repo_root: Path | None = None) -> list[str]:
    errors: list[str] = []
    _expect(errors, data.get("schema_version") == 1, "schema_version must be 1")
    _expect(errors, data.get("packet_id") == "LEARNED-20M-LAUNCH-PACKET-V1", "packet_id mismatch")
    _expect(errors, data.get("issue") == 653, "issue must bind #653")
    _expect(errors, data.get("status") == "PRELAUNCH_CONTROL", "status must remain PRELAUNCH_CONTROL")

    authority = data.get("authority")
    _expect(errors, isinstance(authority, dict), "authority must be an object")
    if isinstance(authority, dict):
        for key, value in EXPECTED_AUTHORITY.items():
            _expect(errors, authority.get(key) == value, f"authority.{key} mismatch")
        if repo_root is not None:
            path = repo_root / EXPECTED_AUTHORITY["r01_contract_path"]
            _expect(errors, path.is_file(), "bound R01 contract file missing")
            if path.is_file():
                actual = git_blob_sha1(path.read_bytes())
                _expect(
                    errors,
                    actual == EXPECTED_AUTHORITY["r01_contract_git_blob_sha1"],
                    "bound R01 contract Git blob identity mismatch",
                )

    boundary = data.get("truth_boundary")
    _expect(errors, isinstance(boundary, dict), "truth_boundary must be an object")
    if isinstance(boundary, dict):
        for key in (
            "package_performs_training",
            "package_authorizes_paid_compute_by_itself",
            "parameter_count_is_authorization",
            "planning_readiness_is_training_authorization",
            "final_test_payload_access_allowed",
        ):
            _expect(errors, boundary.get(key) is False, f"truth_boundary.{key} must be false")

    evidence = data.get("launch_evidence")
    _expect(errors, isinstance(evidence, dict), "launch_evidence must be an object")
    if isinstance(evidence, dict):
        _expect(
            errors,
            evidence.get("model_spec_sha256") == EXPECTED_AUTHORITY["modelspec_sha256"],
            "launch_evidence.model_spec_sha256 mismatch",
        )

    state, blockers = derive_state(data)
    _expect(errors, data.get("declared_state") == state, f"declared_state must equal derived state {state}")

    authorizations = data.get("authorizations")
    if isinstance(authorizations, dict):
        compute = authorizations.get("compute")
        training = authorizations.get("training")
        if isinstance(compute, dict):
            _expect(
                errors,
                compute.get("status") in {"NOT_AUTHORIZED", "AUTHORIZED"},
                "compute authorization status invalid",
            )
        if isinstance(training, dict):
            _expect(
                errors,
                training.get("status") in {"NOT_AUTHORIZED", "AUTHORIZED"},
                "training authorization status invalid",
            )

    if state == "TRAINING_AUTHORIZED":
        _expect(errors, not blockers, "training authorization cannot coexist with blockers")

    return errors


def validate_path(path: Path, repo_root: Path | None = None) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return ["launch packet root must be an object"]
    return validate_packet(data, repo_root=repo_root)


def main(argv: list[str]) -> int:
    path = Path(argv[1]) if len(argv) > 1 else Path(
        "configs/training/learned_20m_launch_packet_v1.json"
    )
    repo_root = Path(argv[2]) if len(argv) > 2 else Path(".")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        print("FAIL: launch packet root must be an object")
        return 1
    errors = validate_packet(data, repo_root=repo_root)
    state, blockers = derive_state(data)
    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        return 1
    print(f"PASS: learned-20M launch packet valid; state={state}")
    if blockers:
        print("BLOCKERS: " + ",".join(blockers))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
