"""Fail-closed scientific and authorization gate for learned MODEL-341 training.

This module is control-plane only. It does not import torch, allocate model weights,
access evaluation payloads, provision hardware, or launch training.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

SCHEMA = "12-6.model341-learned-20m-launch-readiness.v1"
MODEL_STAGE = "MODEL-341-20M-CANDIDATE-A"
MODEL_SOURCE_SHA = "e4ff486fd90802fc123bebf60eed4e59196a98df"
MODEL_IDENTITY_SHA256 = "fbff24d561a2818453554d58ca23fc6ace3303b078f1935a8576c4565bd92441"
INIT_IDENTITY_SHA256 = "86483c6df623e80cab2f73aba718863fce18af6fe3b12430c1348414d92b48a5"
PARAMETER_COUNT = 20_613_440
EXPECTED_VOCAB_SIZE = 256
UNIQUE_LOSS_UNIT = "post_pack_unique_causal_loss_positions"
UNIQUE_LOSS_METHOD = "post_pack_loss_mask_ledger"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")


class LaunchReadinessError(ValueError):
    """Raised when the packet is malformed rather than merely incomplete."""


@dataclass(frozen=True)
class LaunchReadiness:
    blockers: tuple[str, ...]
    scientific_packet_complete: bool
    ready_for_authorization_request: bool
    bounded_smoke_authorized: bool
    long_training_authorized: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "blockers": list(self.blockers),
            "scientific_packet_complete": self.scientific_packet_complete,
            "ready_for_authorization_request": self.ready_for_authorization_request,
            "bounded_smoke_authorized": self.bounded_smoke_authorized,
            "long_training_authorized": self.long_training_authorized,
        }


def load_packet(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LaunchReadinessError(f"cannot read launch packet: {path}") from exc
    if not isinstance(value, dict):
        raise LaunchReadinessError("launch packet root must be an object")
    return value


def _obj(packet: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = packet.get(key)
    return value if isinstance(value, Mapping) else {}


def _valid_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _valid_git_sha(value: object) -> bool:
    return isinstance(value, str) and _GIT_SHA.fullmatch(value) is not None


def _positive_int(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value > 0


def _nonnegative_number(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and value >= 0


def _terminal_ref(authority: Mapping[str, Any], *, status: str = "TERMINAL_PASS") -> bool:
    return (
        authority.get("status") == status
        and isinstance(authority.get("authority_ref"), str)
        and bool(authority.get("authority_ref"))
        and _valid_git_sha(authority.get("authority_sha"))
    )


def _assess_binding(packet: Mapping[str, Any], blockers: list[str]) -> None:
    binding = _obj(packet, "binding")
    if binding.get("stage") != MODEL_STAGE:
        blockers.append("model_stage_not_bound")
    if binding.get("source_sha") != MODEL_SOURCE_SHA:
        blockers.append("model_source_sha_not_bound")
    if binding.get("model_identity_sha256") != MODEL_IDENTITY_SHA256:
        blockers.append("model_identity_not_bound")
    if binding.get("init_identity_sha256") != INIT_IDENTITY_SHA256:
        blockers.append("init_identity_not_bound")
    if binding.get("parameter_count") != PARAMETER_COUNT:
        blockers.append("parameter_count_not_bound")


def _assess_tokenizer(packet: Mapping[str, Any], blockers: list[str]) -> None:
    tokenizer = _obj(packet, "tokenizer")
    if tokenizer.get("status") != "TERMINAL_SELECTED":
        blockers.append("tokenizer_not_terminal")
        return
    if tokenizer.get("vocab_size") != EXPECTED_VOCAB_SIZE:
        blockers.append("tokenizer_model_vocab_mismatch")
    if not _valid_sha256(tokenizer.get("identity_sha256")):
        blockers.append("tokenizer_identity_missing")
    if not isinstance(tokenizer.get("authority_ref"), str) or not tokenizer.get("authority_ref"):
        blockers.append("tokenizer_authority_missing")


def _assess_corpus(packet: Mapping[str, Any], blockers: list[str]) -> int:
    corpus = _obj(packet, "corpus")
    if corpus.get("status") != "TERMINAL_PASS":
        blockers.append("corpus_not_terminal")
        return 0
    for key in (
        "corpus_manifest_sha256",
        "split_manifest_sha256",
        "packing_manifest_sha256",
        "unique_loss_ledger_sha256",
        "decontamination_manifest_sha256",
    ):
        if not _valid_sha256(corpus.get(key)):
            blockers.append(f"corpus_{key}_missing")
    if corpus.get("budget_unit") != UNIQUE_LOSS_UNIT:
        blockers.append("corpus_budget_unit_invalid")
    if corpus.get("measurement_method") != UNIQUE_LOSS_METHOD:
        blockers.append("unique_loss_measurement_not_authoritative")
    positions = corpus.get("unique_causal_loss_positions")
    if not _positive_int(positions):
        blockers.append("unique_causal_loss_positions_missing")
        return 0
    return int(positions)


def _assess_dependencies(packet: Mapping[str, Any], blockers: list[str]) -> None:
    if not _terminal_ref(_obj(packet, "checkpoint_recovery")):
        blockers.append("d05_checkpoint_recovery_not_terminal")

    ladder = _obj(packet, "learned_ladder")
    for scale in ("3m", "10m"):
        evidence = ladder.get(scale)
        if not isinstance(evidence, Mapping) or not _terminal_ref(
            evidence, status="INDEPENDENT_PASS"
        ):
            blockers.append(f"learned_{scale}_independent_verification_missing")

    evaluation = _obj(packet, "evaluation_firewall")
    if evaluation.get("status") != "TERMINAL_PASS":
        blockers.append("evaluation_firewall_not_terminal")
    else:
        for key in (
            "selection_validation_identity_sha256",
            "final_test_registry_sha256",
            "exclusion_manifest_sha256",
        ):
            if not _valid_sha256(evaluation.get(key)):
                blockers.append(f"evaluation_{key}_missing")
        if evaluation.get("final_test_access_before_terminal") is not False:
            blockers.append("final_test_firewall_not_closed")


def _assess_recipe_and_compute(
    packet: Mapping[str, Any], blockers: list[str], available_positions: int
) -> None:
    recipe = _obj(packet, "training_recipe")
    if recipe.get("status") != "TERMINAL_PASS":
        blockers.append("training_recipe_not_terminal")
    else:
        if not isinstance(recipe.get("optimizer"), str) or not recipe.get("optimizer"):
            blockers.append("optimizer_missing")
        if not isinstance(recipe.get("scheduler"), str) or not recipe.get("scheduler"):
            blockers.append("scheduler_missing")
        if not isinstance(recipe.get("precision"), str) or not recipe.get("precision"):
            blockers.append("precision_missing")
        lr = recipe.get("learning_rate")
        if isinstance(lr, bool) or not isinstance(lr, (int, float)) or lr <= 0:
            blockers.append("learning_rate_invalid")
        warmup = recipe.get("warmup_steps")
        if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
            blockers.append("warmup_steps_invalid")
        seeds = recipe.get("seeds")
        if not isinstance(seeds, list) or not seeds or any(
            isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds
        ):
            blockers.append("seeds_invalid")
        target = recipe.get("target_unique_loss_positions")
        if not _positive_int(target):
            blockers.append("training_unique_loss_budget_missing")
        elif available_positions and int(target) > available_positions:
            blockers.append("training_budget_exceeds_unique_corpus_authority")
        if not _positive_int(recipe.get("checkpoint_interval_steps")):
            blockers.append("checkpoint_interval_invalid")
        stop_rules = recipe.get("stop_rules")
        if not isinstance(stop_rules, list) or not stop_rules or any(
            not isinstance(rule, str) or not rule for rule in stop_rules
        ):
            blockers.append("stop_rules_missing")

    compute = _obj(packet, "compute_plan")
    if compute.get("status") != "TERMINAL_PASS":
        blockers.append("compute_plan_not_terminal")
    else:
        if not _positive_int(compute.get("estimated_flops")):
            blockers.append("estimated_flops_invalid")
        if not _positive_int(compute.get("max_wall_minutes")):
            blockers.append("max_wall_minutes_invalid")
        if not isinstance(compute.get("hardware"), str) or not compute.get("hardware"):
            blockers.append("hardware_plan_missing")
        if not _nonnegative_number(compute.get("max_cost_usd")):
            blockers.append("max_cost_invalid")


def assess_model341_launch(packet: Mapping[str, Any]) -> LaunchReadiness:
    if packet.get("schema") != SCHEMA:
        raise LaunchReadinessError(f"unsupported launch packet schema: {packet.get('schema')!r}")

    blockers: list[str] = []
    _assess_binding(packet, blockers)
    _assess_tokenizer(packet, blockers)
    positions = _assess_corpus(packet, blockers)
    _assess_dependencies(packet, blockers)
    _assess_recipe_and_compute(packet, blockers, positions)

    scientific_complete = not blockers
    ready_for_request = scientific_complete

    authorization = _obj(packet, "authorization")
    explicit_compute = authorization.get("compute_status") == "COMPUTE_AUTHORIZED"
    explicit_training = authorization.get("training_status") == "TRAINING_AUTHORIZED"
    authority_ref = authorization.get("authority_ref")
    explicit_authority = isinstance(authority_ref, str) and bool(authority_ref)
    scope = authorization.get("scope")

    bounded_smoke_authorized = (
        scientific_complete
        and explicit_compute
        and explicit_training
        and explicit_authority
        and scope == "BOUNDED_SMOKE"
    )

    smoke = _obj(packet, "smoke_result")
    smoke_passed = _terminal_ref(smoke)
    long_training_authorized = (
        scientific_complete
        and explicit_compute
        and explicit_training
        and explicit_authority
        and scope == "LONG_TRAINING"
        and smoke_passed
    )

    if scientific_complete and not (explicit_compute and explicit_training and explicit_authority):
        blockers.append("explicit_compute_and_training_authorization_missing")
    if scientific_complete and scope == "LONG_TRAINING" and not smoke_passed:
        blockers.append("bounded_smoke_not_terminal_pass")

    return LaunchReadiness(
        blockers=tuple(blockers),
        scientific_packet_complete=scientific_complete,
        ready_for_authorization_request=ready_for_request,
        bounded_smoke_authorized=bounded_smoke_authorized,
        long_training_authorized=long_training_authorized,
    )
