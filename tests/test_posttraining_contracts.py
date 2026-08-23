import pytest

from twelve_six.posttraining import (
    ArtifactRef,
    BaseLineageViolation,
    BehavioralTrainingNotAuthorized,
    ExecutionMode,
    ExactMatchVerifier,
    FrameworkKind,
    PostTrainingExperimentConfig,
    PostTrainingMethod,
    SamplingSpec,
    SyntheticProvenance,
    VerificationContext,
    VerifierRegistry,
    content_fingerprint,
)


ZERO = "0" * 64
ONE = "1" * 64
TWO = "2" * 64


def artifact(lineage: str = "base/s0") -> ArtifactRef:
    return ArtifactRef(artifact_id="checkpoint-s0", lineage=lineage, sha256=ZERO)


def test_real_training_requires_owner_authorization_and_non_base_output() -> None:
    config = PostTrainingExperimentConfig(
        experiment_id="future-sft",
        method=PostTrainingMethod.SFT,
        source_checkpoint=artifact(),
        dataset_manifest_sha256=ONE,
        output_lineage="posttraining/future-sft",
        framework=FrameworkKind.TRL,
        execution_mode=ExecutionMode.TRAIN,
    )
    with pytest.raises(BehavioralTrainingNotAuthorized):
        config.assert_execution_allowed()


def test_posttraining_cannot_target_base_lineage_even_if_authorized() -> None:
    config = PostTrainingExperimentConfig(
        experiment_id="forbidden",
        method=PostTrainingMethod.DPO,
        source_checkpoint=artifact(),
        dataset_manifest_sha256=ONE,
        output_lineage="base/s0-aligned",
        framework=FrameworkKind.TRL,
        execution_mode=ExecutionMode.TRAIN,
        owner_behavioral_training_authorization="owner-decision-placeholder",
    )
    with pytest.raises(BaseLineageViolation):
        config.assert_execution_allowed()


def test_contract_only_configuration_is_allowed_without_weight_mutation() -> None:
    config = PostTrainingExperimentConfig(
        experiment_id="contract-smoke",
        method=PostTrainingMethod.GRPO,
        source_checkpoint=artifact(),
        dataset_manifest_sha256=ONE,
        output_lineage="posttraining/contracts/contract-smoke",
    )
    config.assert_execution_allowed()


def test_external_synthetic_generator_requires_owner_policy_reference() -> None:
    with pytest.raises(ValueError, match="owner_policy_ref"):
        SyntheticProvenance(
            generator=artifact("external/teacher"),
            generation_config_sha256=ONE,
            prompt_template_sha256=TWO,
            seed=7,
            external_generator=True,
        )


def test_schema_fingerprints_are_stable() -> None:
    first = {
        "generator": artifact(),
        "seed": 7,
        "labels": ("logic", "smoke"),
    }
    second = {
        "labels": ("logic", "smoke"),
        "seed": 7,
        "generator": artifact(),
    }
    assert content_fingerprint(first) == content_fingerprint(second)


def test_verifier_registry_and_exact_match() -> None:
    registry = VerifierRegistry()
    verifier = ExactMatchVerifier(strip=True, casefold=True)
    registry.register(verifier)
    with pytest.raises(ValueError, match="duplicate"):
        registry.register(verifier)

    result = registry.get("exact_match", "1").verify(
        VerificationContext(
            task_id="math-1",
            candidate_id="c-1",
            prompt="Return the answer.",
            candidate="  FOUR ",
            reference_answer="four",
        )
    )
    assert result.passed is True
    assert result.score == 1.0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_new_tokens": 0}, "max_new_tokens"),
        ({"max_new_tokens": 1, "temperature": -0.1}, "temperature"),
        ({"max_new_tokens": 1, "top_p": 0.0}, "top_p"),
        ({"max_new_tokens": 1, "top_k": -2}, "top_k"),
    ],
)
def test_sampling_contract_rejects_invalid_values(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        SamplingSpec(**kwargs)
