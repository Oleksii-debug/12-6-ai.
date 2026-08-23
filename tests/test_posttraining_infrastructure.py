from decimal import Decimal

import pytest

from twelve_six.posttraining import (
    Candidate,
    CheckpointRef,
    ComputeClass,
    DatasetManifest,
    ExactTextVerifier,
    LineageKind,
    ManifestEntry,
    NumericToleranceVerifier,
    PostTrainingExperiment,
    Split,
    SyntheticProvenance,
    VerifierRegistry,
    VerifierTask,
)

HEX_A = "a" * 64
HEX_B = "b" * 64
HEX_C = "c" * 64


def _base_checkpoint() -> CheckpointRef:
    return CheckpointRef(
        checkpoint_id="s0-base-example",
        sha256=HEX_A,
        git_sha="f2e94c7",
        stage="S0",
        lineage=LineageKind.BASE,
    )


def _candidate(text: str) -> Candidate:
    return Candidate(
        candidate_id="cand-1",
        prompt_id="prompt-1",
        text=text,
        checkpoint=_base_checkpoint(),
        generation_config_sha256=HEX_B,
    )


def test_posttraining_experiment_may_read_base_but_cannot_output_base() -> None:
    with pytest.raises(ValueError, match="cannot be BASE"):
        PostTrainingExperiment(
            experiment_id="exp-1",
            algorithm="sft",
            backend_id="trl",
            input_checkpoint=_base_checkpoint(),
            output_lineage=LineageKind.BASE,
            dataset_manifest_sha256=HEX_C,
            seed=7,
        )

    experiment = PostTrainingExperiment(
        experiment_id="exp-2",
        algorithm="sft",
        backend_id="trl",
        input_checkpoint=_base_checkpoint(),
        output_lineage=LineageKind.POSTTRAIN,
        dataset_manifest_sha256=HEX_C,
        seed=7,
    )
    assert experiment.input_checkpoint.lineage is LineageKind.BASE
    assert experiment.output_lineage is LineageKind.POSTTRAIN


def test_material_paid_compute_requires_external_authorization_reference() -> None:
    with pytest.raises(ValueError, match="compute_authorization_id"):
        PostTrainingExperiment(
            experiment_id="exp-paid",
            algorithm="grpo",
            backend_id="verl",
            input_checkpoint=_base_checkpoint(),
            output_lineage=LineageKind.POSTTRAIN,
            dataset_manifest_sha256=HEX_C,
            seed=0,
            compute_class=ComputeClass.MATERIAL_PAID,
        )


def test_synthetic_provenance_requires_generator_identity() -> None:
    with pytest.raises(ValueError, match="generator_id"):
        SyntheticProvenance(
            source_id="synthetic-1",
            content_sha256=HEX_A,
            synthetic=True,
        )


def test_dataset_manifest_hash_is_order_independent() -> None:
    first = ManifestEntry("a", HEX_A, Split.TRAIN, HEX_B)
    second = ManifestEntry("b", HEX_B, Split.VALIDATION, HEX_C)
    manifest_a = DatasetManifest.from_entries("dataset", [first, second])
    manifest_b = DatasetManifest.from_entries("dataset", [second, first])
    assert manifest_a.sha256 == manifest_b.sha256


def test_dataset_manifest_rejects_duplicate_record_ids() -> None:
    first = ManifestEntry("same", HEX_A, Split.TRAIN, HEX_B)
    second = ManifestEntry("same", HEX_B, Split.TRAIN, HEX_C)
    with pytest.raises(ValueError, match="duplicate"):
        DatasetManifest.from_entries("dataset", [first, second])


def test_exact_text_verifier() -> None:
    task = VerifierTask(task_id="t1", prompt="answer", reference="42")
    result = ExactTextVerifier().verify(task, _candidate(" 42 "))
    assert result.passed
    assert result.score == 1.0


def test_numeric_tolerance_verifier() -> None:
    task = VerifierTask(task_id="t2", prompt="answer", reference="3.1415")
    verifier = NumericToleranceVerifier(tolerance=Decimal("0.001"))
    assert verifier.verify(task, _candidate("3.142")).passed
    assert not verifier.verify(task, _candidate("3.2")).passed


def test_verifier_registry_fails_closed_on_duplicate_names() -> None:
    registry = VerifierRegistry()
    registry.register(ExactTextVerifier())
    with pytest.raises(ValueError, match="already registered"):
        registry.register(ExactTextVerifier())
