from __future__ import annotations

import hashlib
import json

import pytest

from twelve_six.artifact_identity import (
    CANONICAL_ARTIFACT_KINDS,
    ArtifactKind,
    ArtifactManifest,
    ArtifactRef,
    GenerationIdentityManifest,
    ParentBinding,
    bind_artifact,
    build_generation_identity_manifest,
    parse_generation_identity_manifest,
    verify_parent_bindings,
)
from twelve_six.model import InitSpec, ModelSpec


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _ref(kind: ArtifactKind, generation: str, schema_version: int = 1) -> ArtifactRef:
    return ArtifactRef(
        kind=kind,
        schema_version=schema_version,
        identity_sha256=_sha(f"{generation}:{kind.value}"),
    )


def _refs(generation: str) -> dict[ArtifactKind, ArtifactRef]:
    return {kind: _ref(kind, generation) for kind in CANONICAL_ARTIFACT_KINDS}


def _generation(generation: str) -> GenerationIdentityManifest:
    return build_generation_identity_manifest(_refs(generation))


def test_section1_covers_exact_canonical_identity_kinds() -> None:
    assert tuple(kind.value for kind in CANONICAL_ARTIFACT_KINDS) == (
        "model_spec",
        "init_spec",
        "corpus",
        "tokenizer",
        "split",
        "packing",
        "exposure_ledger",
        "training_run",
        "checkpoint",
        "evaluation",
        "export",
        "release",
    )


def test_artifact_ref_requires_exact_version_and_sha256() -> None:
    ArtifactRef(ArtifactKind.CORPUS, 1, _sha("corpus"))

    with pytest.raises(ValueError, match="schema_version"):
        ArtifactRef(ArtifactKind.CORPUS, True, _sha("corpus"))

    with pytest.raises(ValueError, match="identity_sha256"):
        ArtifactRef(ArtifactKind.CORPUS, 1, "A" * 64)

    with pytest.raises(ValueError, match="kind"):
        ArtifactRef("corpus", 1, _sha("corpus"))  # type: ignore[arg-type]


def test_modelspec_and_initspec_existing_identities_plug_into_unified_refs() -> None:
    spec = ModelSpec(
        schema_version=1,
        vocab_size=256,
        max_seq_len=64,
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        d_ff=128,
        rope_rotary_dim=16,
    )
    init = InitSpec()

    model_ref = ArtifactRef(
        ArtifactKind.MODEL_SPEC,
        spec.schema_version,
        spec.identity_sha256(),
    )
    init_ref = ArtifactRef(
        ArtifactKind.INIT_SPEC,
        init.schema_version,
        init.identity_sha256(),
    )

    assert model_ref.identity_sha256 == spec.identity_sha256()
    assert init_ref.identity_sha256 == init.identity_sha256()



def test_bind_artifact_canonicalizes_parent_role_order() -> None:
    generation = _generation("a")
    packing = generation.artifact_ref(ArtifactKind.PACKING)

    manifest = bind_artifact(
        packing,
        parents={
            "tokenizer": generation.artifact_manifest(ArtifactKind.TOKENIZER),
            "split": generation.artifact_manifest(ArtifactKind.SPLIT),
        },
    )

    assert tuple(manifest.parents_by_role()) == ("split", "tokenizer")


def test_artifact_manifest_rejects_ambiguous_parent_order_and_duplicates() -> None:
    generation = _generation("a")
    split = generation.artifact_manifest(ArtifactKind.SPLIT)
    tokenizer = generation.artifact_manifest(ArtifactKind.TOKENIZER)
    packing = generation.artifact_ref(ArtifactKind.PACKING)

    with pytest.raises(ValueError, match="canonical role order"):
        ArtifactManifest(
            schema_version=1,
            artifact=packing,
            parents=(
                ParentBinding(
                    "tokenizer",
                    tokenizer.artifact,
                    tokenizer.manifest_identity_sha256(),
                ),
                ParentBinding(
                    "split",
                    split.artifact,
                    split.manifest_identity_sha256(),
                ),
            ),
        )

    with pytest.raises(ValueError, match="roles must be unique"):
        ArtifactManifest(
            schema_version=1,
            artifact=packing,
            parents=(
                ParentBinding(
                    "split",
                    split.artifact,
                    split.manifest_identity_sha256(),
                ),
                ParentBinding(
                    "split",
                    tokenizer.artifact,
                    tokenizer.manifest_identity_sha256(),
                ),
            ),
        )


def test_verify_parent_bindings_rejects_role_and_identity_resealing() -> None:
    a = _generation("a")
    b = _generation("b")
    manifest = a.artifact_manifest(ArtifactKind.PACKING)

    verify_parent_bindings(
        manifest,
        expected_parents={
            "split": a.artifact_manifest(ArtifactKind.SPLIT),
            "tokenizer": a.artifact_manifest(ArtifactKind.TOKENIZER),
        },
    )

    with pytest.raises(ValueError, match="role set mismatch"):
        verify_parent_bindings(
            manifest,
            expected_parents={"split": a.artifact_manifest(ArtifactKind.SPLIT)},
        )

    with pytest.raises(ValueError, match="identity mismatch"):
        verify_parent_bindings(
            manifest,
            expected_parents={
                "split": b.artifact_manifest(ArtifactKind.SPLIT),
                "tokenizer": a.artifact_manifest(ArtifactKind.TOKENIZER),
            },
        )

def test_generation_manifest_cross_binds_every_derived_artifact() -> None:
    generation = _generation("a")

    assert tuple(item.artifact.kind for item in generation.artifacts) == (
        CANONICAL_ARTIFACT_KINDS
    )
    assert generation.artifact_manifest(ArtifactKind.MODEL_SPEC).parents == ()
    assert generation.artifact_manifest(ArtifactKind.INIT_SPEC).parents == ()
    assert generation.artifact_manifest(ArtifactKind.CORPUS).parents == ()

    packing = generation.artifact_manifest(ArtifactKind.PACKING).parents_by_role()
    assert packing == {
        "split": generation.artifact_ref(ArtifactKind.SPLIT),
        "tokenizer": generation.artifact_ref(ArtifactKind.TOKENIZER),
    }

    training = generation.artifact_manifest(ArtifactKind.TRAINING_RUN).parents_by_role()
    assert training == {
        "corpus": generation.artifact_ref(ArtifactKind.CORPUS),
        "exposure_ledger": generation.artifact_ref(ArtifactKind.EXPOSURE_LEDGER),
        "init_spec": generation.artifact_ref(ArtifactKind.INIT_SPEC),
        "model_spec": generation.artifact_ref(ArtifactKind.MODEL_SPEC),
        "packing": generation.artifact_ref(ArtifactKind.PACKING),
        "split": generation.artifact_ref(ArtifactKind.SPLIT),
        "tokenizer": generation.artifact_ref(ArtifactKind.TOKENIZER),
    }

    release = generation.artifact_manifest(ArtifactKind.RELEASE).parents_by_role()
    assert release == {
        "checkpoint": generation.artifact_ref(ArtifactKind.CHECKPOINT),
        "evaluation": generation.artifact_ref(ArtifactKind.EVALUATION),
        "export": generation.artifact_ref(ArtifactKind.EXPORT),
    }



def test_generation_manifest_rejects_cross_generation_packing_parent() -> None:
    a = _generation("a")
    b = _generation("b")
    manifests = list(a.artifacts)
    packing_index = CANONICAL_ARTIFACT_KINDS.index(ArtifactKind.PACKING)
    manifests[packing_index] = bind_artifact(
        a.artifact_ref(ArtifactKind.PACKING),
        parents={
            "split": b.artifact_manifest(ArtifactKind.SPLIT),
            "tokenizer": a.artifact_manifest(ArtifactKind.TOKENIZER),
        },
    )

    with pytest.raises(ValueError, match="parent identity does not match generation"):
        GenerationIdentityManifest(schema_version=1, artifacts=tuple(manifests))


def test_generation_manifest_rejects_cross_generation_training_modelspec() -> None:
    a = _generation("a")
    b = _generation("b")
    manifests = list(a.artifacts)
    training_index = CANONICAL_ARTIFACT_KINDS.index(ArtifactKind.TRAINING_RUN)
    training = a.artifact_manifest(ArtifactKind.TRAINING_RUN)
    parents = {
        binding.role: a.artifact_manifest(binding.artifact.kind)
        for binding in training.parents
    }
    parents["model_spec"] = b.artifact_manifest(ArtifactKind.MODEL_SPEC)
    manifests[training_index] = bind_artifact(
        a.artifact_ref(ArtifactKind.TRAINING_RUN),
        parents=parents,
    )

    with pytest.raises(ValueError, match="parent identity does not match generation"):
        GenerationIdentityManifest(schema_version=1, artifacts=tuple(manifests))


def test_generation_manifest_rejects_wrong_parent_kind_even_with_valid_digest() -> None:
    a = _generation("a")
    manifests = list(a.artifacts)
    export_index = CANONICAL_ARTIFACT_KINDS.index(ArtifactKind.EXPORT)
    manifests[export_index] = bind_artifact(
        a.artifact_ref(ArtifactKind.EXPORT),
        parents={"checkpoint": a.artifact_manifest(ArtifactKind.TRAINING_RUN)},
    )

    with pytest.raises(ValueError, match="must reference checkpoint"):
        GenerationIdentityManifest(schema_version=1, artifacts=tuple(manifests))


def test_generation_rejects_resealed_parent_lineage_with_same_raw_artifact_identity() -> None:
    generation = _generation("a")
    corpus_b = bind_artifact(_ref(ArtifactKind.CORPUS, "b"))

    canonical_tokenizer = generation.artifact_manifest(ArtifactKind.TOKENIZER)
    resealed_tokenizer = bind_artifact(
        canonical_tokenizer.artifact,
        parents={"corpus": corpus_b},
    )
    assert resealed_tokenizer.artifact == canonical_tokenizer.artifact
    assert (
        resealed_tokenizer.manifest_identity_sha256()
        != canonical_tokenizer.manifest_identity_sha256()
    )

    canonical_packing = generation.artifact_manifest(ArtifactKind.PACKING)
    resealed_packing = bind_artifact(
        canonical_packing.artifact,
        parents={
            "split": generation.artifact_manifest(ArtifactKind.SPLIT),
            "tokenizer": resealed_tokenizer,
        },
    )
    assert resealed_packing.artifact == canonical_packing.artifact
    assert (
        resealed_packing.manifest_identity_sha256()
        != canonical_packing.manifest_identity_sha256()
    )

    manifests = list(generation.artifacts)
    packing_index = CANONICAL_ARTIFACT_KINDS.index(ArtifactKind.PACKING)
    manifests[packing_index] = resealed_packing
    with pytest.raises(ValueError, match="parent lineage identity does not match generation"):
        GenerationIdentityManifest(schema_version=1, artifacts=tuple(manifests))


def test_parent_binding_durable_json_rejects_lineage_hash_reseal() -> None:
    generation = _generation("a")
    payload = json.loads(generation.canonical_json_bytes())
    packing_index = CANONICAL_ARTIFACT_KINDS.index(ArtifactKind.PACKING)
    packing = payload["artifacts"][packing_index]
    packing["parents"][0]["parent_manifest_identity_sha256"] = _sha("foreign-lineage")

    with pytest.raises(ValueError, match="parent lineage identity does not match generation"):
        parse_generation_identity_manifest(
            json.dumps(payload, separators=(",", ":")).encode("utf-8")
        )


def test_child_binding_commits_to_parent_transitive_lineage() -> None:
    a = _generation("a")
    b = _generation("b")
    original_tokenizer = a.artifact_manifest(ArtifactKind.TOKENIZER)
    resealed_tokenizer = bind_artifact(
        original_tokenizer.artifact,
        parents={"corpus": b.artifact_manifest(ArtifactKind.CORPUS)},
    )

    assert resealed_tokenizer.artifact == original_tokenizer.artifact
    assert (
        resealed_tokenizer.manifest_identity_sha256()
        != original_tokenizer.manifest_identity_sha256()
    )

    original_packing = a.artifact_manifest(ArtifactKind.PACKING)
    with pytest.raises(ValueError, match="parent lineage mismatch"):
        verify_parent_bindings(
            original_packing,
            expected_parents={
                "split": a.artifact_manifest(ArtifactKind.SPLIT),
                "tokenizer": resealed_tokenizer,
            },
        )

    resealed_packing = bind_artifact(
        original_packing.artifact,
        parents={
            "split": a.artifact_manifest(ArtifactKind.SPLIT),
            "tokenizer": resealed_tokenizer,
        },
    )
    assert resealed_packing.parents_by_role() == original_packing.parents_by_role()
    assert (
        resealed_packing.manifest_identity_sha256()
        != original_packing.manifest_identity_sha256()
    )


def test_generation_manifest_strict_json_round_trip_preserves_identity() -> None:
    generation = _generation("a")
    encoded = generation.canonical_json_bytes()
    decoded = parse_generation_identity_manifest(encoded)

    assert decoded == generation
    assert decoded.identity_sha256() == generation.identity_sha256()
    assert hashlib.sha256(encoded).hexdigest() == generation.identity_sha256()


def test_generation_manifest_strict_json_rejects_duplicate_members() -> None:
    payload = b'{"schema_version":1,"schema_version":1,"artifacts":[]}'

    with pytest.raises(ValueError, match="strict unambiguous"):
        parse_generation_identity_manifest(payload)


def test_generation_manifest_strict_json_rejects_unknown_fields_and_type_aliases() -> None:
    generation = _generation("a")
    payload = json.loads(generation.canonical_json_bytes())
    payload["unexpected"] = True

    with pytest.raises(ValueError, match="fields mismatch"):
        parse_generation_identity_manifest(
            json.dumps(payload, separators=(",", ":")).encode("utf-8")
        )

    payload = json.loads(generation.canonical_json_bytes())
    payload["schema_version"] = True
    with pytest.raises(ValueError, match="schema_version"):
        parse_generation_identity_manifest(
            json.dumps(payload, separators=(",", ":")).encode("utf-8")
        )


def test_generation_manifest_strict_json_rejects_nested_ref_extension() -> None:
    generation = _generation("a")
    payload = json.loads(generation.canonical_json_bytes())
    payload["artifacts"][0]["artifact"]["unexpected"] = "resealed"

    with pytest.raises(ValueError, match="ArtifactRef fields mismatch"):
        parse_generation_identity_manifest(
            json.dumps(payload, separators=(",", ":")).encode("utf-8")
        )


def test_generation_manifest_is_deterministic_and_generation_sensitive() -> None:
    a1 = _generation("a")
    a2 = _generation("a")
    b = _generation("b")

    assert a1.identity_sha256() == a2.identity_sha256()
    assert a1.identity_sha256() != b.identity_sha256()


def test_generation_builder_rejects_missing_kind_and_key_ref_mismatch() -> None:
    refs = _refs("a")
    del refs[ArtifactKind.RELEASE]
    with pytest.raises(ValueError, match="exactly every canonical"):
        build_generation_identity_manifest(refs)

    refs = _refs("a")
    refs[ArtifactKind.EXPORT] = _ref(ArtifactKind.CHECKPOINT, "wrong")
    with pytest.raises(ValueError, match="ref kind mismatch"):
        build_generation_identity_manifest(refs)
