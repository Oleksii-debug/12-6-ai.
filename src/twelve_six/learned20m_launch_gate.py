from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any

SCHEMA = "12-6.learned20m-launch-gate.v1"
STATE_BLOCKED = "BLOCKED"
STATE_READY = "READY_FOR_AUTHORIZATION_REQUEST"
STATE_AUTHORIZED = "TRAINING_AUTHORIZED"

REPOSITORY = "Oleksii-debug/12-6-ai."
MODEL341_SHA = "e4ff486fd90802fc123bebf60eed4e59196a98df"
MODEL341_MODELSPEC_SHA256 = (
    "fbff24d561a2818453554d58ca23fc6ace3303b078f1935a8576c4565bd92441"
)
MODEL341_PARAMETERS = 20_613_440
R01_MERGE_SHA = "a73ab38026cb7849f478cc13ad58b93534a76e2f"
R01_CONFIG_PATH = "configs/research/r01_20m_to_100m_scaling_campaign_v1.json"
R01_CONFIG_GIT_BLOB_SHA1 = "c50154db609d41eceb2ffc97912360df567bcc04"

_SHA40 = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class LaunchGateError(ValueError):
    """Raised when a launch packet violates a fixed fail-closed invariant."""


@dataclass(frozen=True, slots=True)
class LaunchAssessment:
    state: str
    request_identity_sha256: str
    blockers: tuple[str, ...]

    @property
    def training_authorized(self) -> bool:
        return self.state == STATE_AUTHORIZED

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA,
            "state": self.state,
            "request_identity_sha256": self.request_identity_sha256,
            "training_authorized": self.training_authorized,
            "blockers": list(self.blockers),
        }


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def request_identity(packet: dict[str, Any]) -> str:
    """Hash every launch-defining field while excluding the two authorization signatures."""
    excluded = {"compute_authorization", "training_authorization"}
    unsigned = {k: v for k, v in packet.items() if k not in excluded}
    return hashlib.sha256(_canonical_json(unsigned)).hexdigest()


def _fixed_authority(packet: dict[str, Any]) -> None:
    if packet.get("schema_version") != SCHEMA:
        raise LaunchGateError("unexpected schema_version")
    forbidden_fields = (
        "training_authorized",
        "compute_authorized",
        "authorized_now",
        "declared_state",
    )
    for forbidden in forbidden_fields:
        if forbidden in packet:
            raise LaunchGateError(
                f"derived launch state may not be supplied by packet: {forbidden}"
            )

    authority = packet.get("authority")
    if not isinstance(authority, dict):
        raise LaunchGateError("authority must be an object")
    expected = {
        "repository": REPOSITORY,
        "model341_sha": MODEL341_SHA,
        "modelspec_sha256": MODEL341_MODELSPEC_SHA256,
        "parameter_count": MODEL341_PARAMETERS,
        "r01_merge_sha": R01_MERGE_SHA,
        "r01_config_path": R01_CONFIG_PATH,
        "r01_config_git_blob_sha1": R01_CONFIG_GIT_BLOB_SHA1,
    }
    for key, value in expected.items():
        if authority.get(key) != value:
            raise LaunchGateError(f"fixed authority drift: {key}")
    if authority.get("base_lineage") != "RANDOM_INIT_PRETRAINING_ONLY":
        raise LaunchGateError("Base lineage must remain random-init pretraining-only")


def _is_sha40(value: Any) -> bool:
    return isinstance(value, str) and _SHA40.fullmatch(value) is not None


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _is_nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _is_nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_positive_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value)) and float(value) > 0.0


def _is_nonnegative_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value)) and float(value) >= 0.0


def _terminal_evidence(value: Any, expected_kind: str) -> bool:
    if not isinstance(value, dict):
        return False
    return (
        value.get("kind") == expected_kind
        and value.get("decision") == "PASS"
        and _is_sha40(value.get("source_sha"))
        and _is_sha256(value.get("artifact_sha256"))
        and _is_nonempty(value.get("reference"))
    )


def _collect_scientific_blockers(packet: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    evidence = packet.get("evidence")
    if not isinstance(evidence, dict):
        return ["evidence_missing"]

    sha256_fields = (
        "init_spec_sha256",
        "tokenizer_identity_sha256",
        "corpus_identity_sha256",
        "split_identity_sha256",
        "packing_identity_sha256",
        "post_pack_loss_ledger_sha256",
    )
    if not _is_sha40(evidence.get("code_sha")):
        blockers.append("code_sha_missing_or_invalid")
    for field in sha256_fields:
        if not _is_sha256(evidence.get(field)):
            blockers.append(f"{field}_missing_or_invalid")

    positions = evidence.get("unique_post_pack_causal_loss_positions")
    if not _is_positive_int(positions):
        blockers.append("unique_post_pack_causal_loss_positions_must_be_nonzero")
    if evidence.get("no_replay_proven") is not True:
        blockers.append("no_replay_not_proven")

    terminal = (
        ("evaluation_decontamination", "EVALUATION_DECONTAMINATION"),
        ("checkpoint_integrity", "D05_CHECKPOINT_INTEGRITY"),
        ("learned_ladder", "INDEPENDENT_LEARNED_LADDER"),
        ("integration_ci", "QUALIFIED_INTEGRATION_CI"),
    )
    for field, kind in terminal:
        if not _terminal_evidence(evidence.get(field), kind):
            blockers.append(f"{field}_not_terminal_pass")

    recipe = packet.get("training_recipe")
    if not isinstance(recipe, dict):
        blockers.append("training_recipe_missing")
    else:
        if not _is_nonempty(recipe.get("optimizer")):
            blockers.append("optimizer_missing")
        if not _is_positive_number(recipe.get("learning_rate")):
            blockers.append("learning_rate_missing_or_invalid")
        if not _is_nonempty(recipe.get("scheduler")):
            blockers.append("scheduler_missing")
        if not _is_nonnegative_int(recipe.get("warmup_steps")):
            blockers.append("warmup_steps_missing_or_invalid")
        if not _is_nonempty(recipe.get("precision")):
            blockers.append("precision_missing")
        seeds = recipe.get("seeds")
        if (
            not isinstance(seeds, list)
            or not seeds
            or any(not _is_nonnegative_int(seed) for seed in seeds)
            or len(set(seeds)) != len(seeds)
        ):
            blockers.append("seeds_missing_invalid_or_duplicated")
        if not _is_positive_number(recipe.get("gradient_clip_norm")):
            blockers.append("gradient_clip_norm_missing_or_invalid")
        if recipe.get("target_unique_causal_loss_positions") != positions:
            blockers.append("recipe_target_positions_do_not_match_loss_ledger")
        for field in ("budget_policy_sha256", "stopping_rules_sha256", "checkpoint_policy_sha256"):
            if not _is_sha256(recipe.get(field)):
                blockers.append(f"{field}_missing_or_invalid")

    resource = packet.get("resource_envelope")
    if not isinstance(resource, dict):
        blockers.append("resource_envelope_missing")
    else:
        if not _is_sha256(resource.get("compute_envelope_sha256")):
            blockers.append("compute_envelope_sha256_missing_or_invalid")
        if not _is_nonempty(resource.get("hardware_profile_id")):
            blockers.append("hardware_profile_id_missing")
        if not _is_positive_int(resource.get("accelerator_count")):
            blockers.append("accelerator_count_missing_or_invalid")
        if not _is_positive_number(resource.get("estimated_training_flops")):
            blockers.append("estimated_training_flops_missing_or_invalid")
        if not _is_positive_number(resource.get("projected_wall_hours")):
            blockers.append("projected_wall_hours_missing_or_invalid")
        if not _is_nonnegative_number(resource.get("projected_cost_eur")):
            blockers.append("projected_cost_eur_missing_or_invalid")
        if not _is_nonnegative_number(resource.get("max_cost_eur")):
            blockers.append("max_cost_eur_missing_or_invalid")
        if _is_nonnegative_number(resource.get("projected_cost_eur")) and _is_nonnegative_number(
            resource.get("max_cost_eur")
        ):
            if float(resource["projected_cost_eur"]) > float(resource["max_cost_eur"]):
                blockers.append("projected_cost_exceeds_max_cost")
        if not _terminal_evidence(resource.get("throughput_measurement"), "THROUGHPUT_MEASUREMENT"):
            blockers.append("throughput_measurement_not_terminal_pass")

    return blockers


def _authorization_valid(value: Any, *, expected_decision: str, identity: str) -> bool:
    if not isinstance(value, dict):
        return False
    return (
        value.get("decision") == expected_decision
        and value.get("request_identity_sha256") == identity
        and _is_nonempty(value.get("authorization_id"))
        and _is_nonempty(value.get("approver_reference"))
    )


def assess_launch(packet: dict[str, Any]) -> LaunchAssessment:
    if not isinstance(packet, dict):
        raise LaunchGateError("launch packet must be a JSON object")
    _fixed_authority(packet)
    identity = request_identity(packet)
    blockers = _collect_scientific_blockers(packet)
    if blockers:
        return LaunchAssessment(STATE_BLOCKED, identity, tuple(sorted(set(blockers))))

    compute = packet.get("compute_authorization")
    training = packet.get("training_authorization")
    if compute is None and training is None:
        return LaunchAssessment(STATE_READY, identity, ())
    if compute is None or training is None:
        return LaunchAssessment(STATE_BLOCKED, identity, ("partial_authorization",))
    if not _authorization_valid(
        compute, expected_decision="COMPUTE_AUTHORIZED", identity=identity
    ):
        return LaunchAssessment(STATE_BLOCKED, identity, ("compute_authorization_invalid",))
    if not _authorization_valid(
        training, expected_decision="TRAINING_AUTHORIZED", identity=identity
    ):
        return LaunchAssessment(STATE_BLOCKED, identity, ("training_authorization_invalid",))

    resource = packet["resource_envelope"]
    authorized_max = compute.get("max_cost_eur")
    if not _is_nonnegative_number(authorized_max):
        return LaunchAssessment(
            STATE_BLOCKED, identity, ("compute_authorization_cost_cap_invalid",)
        )
    if float(resource["projected_cost_eur"]) > float(authorized_max):
        return LaunchAssessment(STATE_BLOCKED, identity, ("authorized_cost_cap_below_projection",))
    if float(resource["max_cost_eur"]) > float(authorized_max):
        return LaunchAssessment(STATE_BLOCKED, identity, ("packet_max_cost_exceeds_authorization",))

    return LaunchAssessment(STATE_AUTHORIZED, identity, ())
