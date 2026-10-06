from __future__ import annotations

from dataclasses import replace

import pytest

from twelve_six.identity_manifest import (
    ArtifactIdentity,
    ArtifactKind,
    IdentityManifest,
    IdentityRef,
)


def _sha(number: int) -> str:
    return f"{number:064x}"


def _node(
    artifact_id: str,
    kind: ArtifactKind,
    number: int,
    *parents: ArtifactIdentity,
) -> ArtifactIdentity:
    return ArtifactIdentity(
        artifact_id=artifact_id,
        kind=kind,
        schema_version=1,
        artifact_sha256=_sha(number),
        parents=tuple(parent.as_ref() for parent in parents),
    )


def _complete_release_chain() -> tuple[ArtifactIdentity, ...]:
    model = _node("model-20m", ArtifactKind.MODEL_SPEC, 1)
    init = _node("init-random-v1", ArtifactKind.INIT_SPEC, 2)
    corpus = _node("corpus-v1", ArtifactKind.CORPUS, 3)
    tokenizer = _node("tokenizer-v1", ArtifactKind.TOKENIZER, 4, corpus)
    split = _node("split-v1", ArtifactKind.SPLIT, 5, corpus)
    packing = _node("packing-v1", ArtifactKind.PACKING, 6, tokenizer, split)
    exposure = _node("exposure-v1", ArtifactKind.EXPOSURE_LEDGER, 7, packing)
    run = _node("run-v1", ArtifactKind.TRAINING_RUN, 8, model, init, exposure)
    checkpoint = _node("checkpoint-v1", ArtifactKind.CHECKPOINT, 9, run)
    evaluation = _node("evaluation-v1", ArtifactKind.EVALUATION, 10, checkpoint)
    export = _node("export-v1", ArtifactKind.EXPORT, 11, checkpoint)
    release = _node(
        "release-v1",
        ArtifactKind.RELEASE,
        12,
        checkpoint,
        evaluation,
        export,
    )
    return (
        model,
        init,
        corpus,
        tokenizer,
        split,
        packing,
        exposure,
        run,
        checkpoint,
        evaluation,
        export,
        release,
    )


def test_complete_release_chain_covers_all_canonical_identity_domains() -> None:
    manifest = IdentityManifest(schema_version=1, artifacts=_complete_release_chain())

    assert {artifact.kind for artifact in manifest.artifacts} == set(ArtifactKind)
    assert len(manifest.manifest_sha256) == 64


def test_downstream_binding_changes_when_exact_upstream_artifact_changes() -> None:
    corpus = _node("corpus-v1", ArtifactKind.CORPUS, 1)
    tokenizer = _node("tokenizer-v1", ArtifactKind.TOKENIZER, 2, corpus)
    changed_corpus = replace(corpus, artifact_sha256=_sha(99))
    changed_tokenizer = replace(tokenizer, parents=(changed_corpus.as_ref(),))

    assert changed_corpus.binding_sha256 != corpus.binding_sha256
    assert changed_tokenizer.binding_sha256 != tokenizer.binding_sha256


def test_manifest_rejects_tampered_parent_artifact_identity() -> None:
    corpus = _node("corpus-v1", ArtifactKind.CORPUS, 1)
    tokenizer = _node("tokenizer-v1", ArtifactKind.TOKENIZER, 2, corpus)
    tampered_ref = replace(corpus.as_ref(), artifact_sha256=_sha(999))
    tampered_tokenizer = replace(tokenizer, parents=(tampered_ref,))

    with pytest.raises(ValueError, match="does not match exact identity"):
        IdentityManifest(schema_version=1, artifacts=(corpus, tampered_tokenizer))


def test_manifest_rejects_tampered_transitive_parent_binding() -> None:
    corpus = _node("corpus-v1", ArtifactKind.CORPUS, 1)
    tokenizer = _node("tokenizer-v1", ArtifactKind.TOKENIZER, 2, corpus)
    tampered_ref = replace(corpus.as_ref(), binding_sha256=_sha(999))
    tampered_tokenizer = replace(tokenizer, parents=(tampered_ref,))

    with pytest.raises(ValueError, match="does not match exact identity"):
        IdentityManifest(schema_version=1, artifacts=(corpus, tampered_tokenizer))


def test_manifest_rejects_missing_required_upstream_kind() -> None:
    model = _node("model-v1", ArtifactKind.MODEL_SPEC, 1)
    init = _node("init-v1", ArtifactKind.INIT_SPEC, 2)
    run = _node("run-v1", ArtifactKind.TRAINING_RUN, 3, model, init)

    with pytest.raises(ValueError, match="exposure_ledger"):
        IdentityManifest(schema_version=1, artifacts=(model, init, run))


def test_release_cannot_skip_checkpoint_evaluation_or_export_authority() -> None:
    checkpoint = ArtifactIdentity(
        artifact_id="checkpoint-v1",
        kind=ArtifactKind.CHECKPOINT,
        schema_version=1,
        artifact_sha256=_sha(1),
    )
    release = _node("release-v1", ArtifactKind.RELEASE, 2, checkpoint)

    with pytest.raises(ValueError, match="evaluation"):
        IdentityManifest(schema_version=1, artifacts=(checkpoint, release))


def test_manifest_identity_is_order_independent() -> None:
    artifacts = _complete_release_chain()
    forward = IdentityManifest(schema_version=1, artifacts=artifacts)
    reverse = IdentityManifest(schema_version=1, artifacts=tuple(reversed(artifacts)))

    assert forward.manifest_sha256 == reverse.manifest_sha256


def test_manifest_rejects_duplicate_artifact_id() -> None:
    corpus = _node("corpus-v1", ArtifactKind.CORPUS, 1)
    duplicate = _node("corpus-v1", ArtifactKind.CORPUS, 2)

    with pytest.raises(ValueError, match="artifact_id values must be unique"):
        IdentityManifest(schema_version=1, artifacts=(corpus, duplicate))


def test_manifest_rejects_cycle_before_trusting_parent_bindings() -> None:
    fake_ref_b = IdentityRef(
        artifact_id="b",
        kind=ArtifactKind.CORPUS,
        schema_version=1,
        artifact_sha256=_sha(2),
        binding_sha256=_sha(20),
    )
    fake_ref_a = IdentityRef(
        artifact_id="a",
        kind=ArtifactKind.CORPUS,
        schema_version=1,
        artifact_sha256=_sha(1),
        binding_sha256=_sha(10),
    )
    a = ArtifactIdentity(
        artifact_id="a",
        kind=ArtifactKind.CORPUS,
        schema_version=1,
        artifact_sha256=_sha(1),
        parents=(fake_ref_b,),
    )
    b = ArtifactIdentity(
        artifact_id="b",
        kind=ArtifactKind.CORPUS,
        schema_version=1,
        artifact_sha256=_sha(2),
        parents=(fake_ref_a,),
    )

    with pytest.raises(ValueError, match="contains a cycle"):
        IdentityManifest(schema_version=1, artifacts=(a, b))


@pytest.mark.parametrize("bad_digest", ["ABC", "g" * 64, "0" * 63, None])
def test_artifact_identity_rejects_noncanonical_digest(bad_digest: object) -> None:
    with pytest.raises(ValueError, match="exact lowercase SHA-256"):
        ArtifactIdentity(
            artifact_id="corpus-v1",
            kind=ArtifactKind.CORPUS,
            schema_version=1,
            artifact_sha256=bad_digest,  # type: ignore[arg-type]
        )
