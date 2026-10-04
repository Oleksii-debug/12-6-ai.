"""Fail-closed provenance checks for learned-20M checkpoint launch authority."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from twelve_six.learned20_launch_gate import MODELSPEC_SHA256, PARAMETER_COUNT, assess_launch


def _nonempty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _hex_text(value: Any, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(char in "0123456789abcdef" for char in value)
    )


def _sha256_digest(value: Any) -> bool:
    return isinstance(value, str) and value.startswith("sha256:") and _hex_text(value[7:], 64)


def validate_checkpoint_authority(evidence: Mapping[str, Any]) -> list[str]:
    """Validate a terminal checkpoint authority before the launch gate consumes it.

    The base launch assessor intentionally treats authority identities generically. This
    consumer-side layer prevents a fixture/stale checkpoint report from becoming learned-20M
    launch authority merely by setting ``terminal=true`` and two booleans.
    """

    checkpoint = evidence.get("checkpoint")
    if not isinstance(checkpoint, Mapping) or checkpoint.get("terminal") is not True:
        return []

    blockers: list[str] = []
    if not _hex_text(checkpoint.get("source_sha"), 40):
        blockers.append("checkpoint.source_sha_invalid")
    if not _positive_int(checkpoint.get("workflow_run_id")):
        blockers.append("checkpoint.workflow_run_id_invalid")
    if checkpoint.get("workflow_conclusion") != "success":
        blockers.append("checkpoint.workflow_not_success")
    if not _positive_int(checkpoint.get("artifact_id")):
        blockers.append("checkpoint.artifact_id_invalid")
    if not _sha256_digest(checkpoint.get("artifact_digest")):
        blockers.append("checkpoint.artifact_digest_invalid")

    if checkpoint.get("modelspec_sha256") != MODELSPEC_SHA256:
        blockers.append("checkpoint.modelspec_sha256_mismatch")
    if checkpoint.get("parameter_count") != PARAMETER_COUNT:
        blockers.append("checkpoint.parameter_count_mismatch")

    if checkpoint.get("corruption_matrix_passed") is not True:
        blockers.append("checkpoint.corruption_matrix_not_terminal")
    cases = checkpoint.get("corruption_matrix_cases_passed")
    if not _positive_int(cases) or cases < 13:
        blockers.append("checkpoint.corruption_matrix_case_count_insufficient")
    if checkpoint.get("corruption_matrix_failures") != 0:
        blockers.append("checkpoint.corruption_matrix_has_failures")

    for key in (
        "fresh_resume_equivalence",
        "fresh_process_reload",
        "model_state_exact",
        "trainer_state_exact",
        "optimizer_state_exact",
        "scheduler_state_exact",
        "scaler_state_exact",
        "counter_state_exact",
        "rng_state_exact",
        "dataloader_state_exact",
    ):
        if checkpoint.get(key) is not True:
            blockers.append(f"checkpoint.{key}_not_proven")

    if not _nonempty_text(checkpoint.get("next_batch_identity")):
        blockers.append("checkpoint.next_batch_identity_missing")

    corpus = evidence.get("corpus")
    if isinstance(corpus, Mapping):
        expected_dataset = corpus.get("corpus_identity")
        expected_packing = corpus.get("packing_identity")
        if checkpoint.get("dataset_identity") != expected_dataset:
            blockers.append("checkpoint.dataset_identity_mismatch")
        if checkpoint.get("packing_identity") != expected_packing:
            blockers.append("checkpoint.packing_identity_mismatch")
    else:
        blockers.append("checkpoint.corpus_binding_missing")

    tokenizer = evidence.get("tokenizer")
    if isinstance(tokenizer, Mapping):
        if checkpoint.get("tokenizer_identity") != tokenizer.get("identity"):
            blockers.append("checkpoint.tokenizer_identity_mismatch")
    else:
        blockers.append("checkpoint.tokenizer_binding_missing")

    return sorted(set(blockers))


def assess_launch_with_checkpoint_provenance(
    contract: Mapping[str, Any],
    evidence: Mapping[str, Any],
    *,
    material_cost: bool,
) -> dict[str, Any]:
    """Run the base launch gate and add exact checkpoint-provenance requirements."""

    result = assess_launch(contract, evidence, material_cost=material_cost)
    blockers = validate_checkpoint_authority(evidence)
    if not blockers:
        return result

    result["pilot_blockers"] = sorted(set(result["pilot_blockers"] + blockers))
    result["long_training_blockers"] = sorted(
        set(result["long_training_blockers"] + blockers)
    )
    result["pilot_ready"] = False
    result["long_training_ready"] = False
    return result
