from __future__ import annotations

import hashlib
import json

import pytest

import twelve_six.artifact_identity as artifact_identity_module
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


def test_generation_manifest_rejects_noncanonical_equivalent_bytes() -> None:
    generation = _generation("a")
    noncanonical = json.dumps(
        generation.to_dict(),
        ensure_ascii=False,
        indent=2,
    ).encode("utf-8")

    assert noncanonical != generation.canonical_json_bytes()
    with pytest.raises(ValueError, match="canonical JSON encoding"):
        parse_generation_identity_manifest(noncanonical)


def test_artifact_kind_enum_wire_value_mutation_fails_closed() -> None:
    generation = _generation("enum-wire")
    refs = _refs("enum-wire-new")
    corpus = generation.artifact_ref(ArtifactKind.CORPUS)
    original_value = ArtifactKind.CORPUS.value
    object.__setattr__(ArtifactKind.CORPUS, "_value_", "forged_corpus")
    try:
        with pytest.raises(ValueError, match="wire value is non-canonical"):
            corpus.to_dict()
        with pytest.raises(ValueError, match="wire value is non-canonical"):
            generation.identity_sha256()
        with pytest.raises(ValueError, match="wire value is non-canonical"):
            build_generation_identity_manifest(refs)
    finally:
        object.__setattr__(ArtifactKind.CORPUS, "_value_", original_value)


def test_generation_parent_policy_is_runtime_immutable() -> None:
    release_policy = artifact_identity_module._GENERATION_PARENT_POLICY[ArtifactKind.RELEASE]

    with pytest.raises(TypeError):
        release_policy["evaluation"] = ArtifactKind.EXPORT  # type: ignore[index]

    with pytest.raises(TypeError):
        artifact_identity_module._GENERATION_PARENT_POLICY[
            ArtifactKind.RELEASE
        ] = {}  # type: ignore[index]


def test_generation_validation_fails_closed_after_policy_global_rebind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical = artifact_identity_module._GENERATION_PARENT_POLICY
    forged = dict(canonical)
    forged_release = dict(canonical[ArtifactKind.RELEASE])
    forged_release["evaluation"] = ArtifactKind.EXPORT
    forged[ArtifactKind.RELEASE] = forged_release
    monkeypatch.setattr(artifact_identity_module, "_GENERATION_PARENT_POLICY", forged)

    with pytest.raises(ValueError, match=r"release\.evaluation must reference evaluation"):
        _generation("a")


def test_generation_vocabulary_fails_closed_after_global_rebind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generation = _generation("a")
    monkeypatch.setattr(
        artifact_identity_module,
        "CANONICAL_ARTIFACT_KINDS",
        tuple(ArtifactKind)[:3],
    )

    with pytest.raises(ValueError, match="exactly one artifact of every canonical kind"):
        GenerationIdentityManifest(
            schema_version=1,
            artifacts=generation.artifacts[:3],
        )

    assert generation.artifact_ref(ArtifactKind.RELEASE).kind is ArtifactKind.RELEASE
    assert (
        generation.artifact_manifest(ArtifactKind.RELEASE).artifact.kind
        is ArtifactKind.RELEASE
    )


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


def test_generation_builder_rejects_raw_string_kind_key_aliases() -> None:
    refs = _refs("a")
    raw_key_refs = {kind.value: ref for kind, ref in refs.items()}

    with pytest.raises(ValueError, match="ArtifactKind"):
        build_generation_identity_manifest(raw_key_refs)  # type: ignore[arg-type]


def test_generation_builder_rejects_missing_kind_and_key_ref_mismatch() -> None:
    refs = _refs("a")
    del refs[ArtifactKind.RELEASE]
    with pytest.raises(ValueError, match="exactly every canonical"):
        build_generation_identity_manifest(refs)

    refs = _refs("a")
    refs[ArtifactKind.EXPORT] = _ref(ArtifactKind.CHECKPOINT, "wrong")
    with pytest.raises(ValueError, match="ref kind mismatch"):
        build_generation_identity_manifest(refs)


def test_closed_schema_from_dict_rejects_behavioral_dict_before_lookup() -> None:
    generation = _generation("a")
    corpus = generation.artifact_ref(ArtifactKind.CORPUS)

    class ForgedDict(dict):
        def __getitem__(self, key):
            raise AssertionError("behavioral dict must not be indexed")

    forged = ForgedDict(corpus.to_dict())
    with pytest.raises(ValueError, match="ArtifactRef fields mismatch"):
        ArtifactRef.from_dict(forged)


def test_closed_schema_from_dict_rejects_behavioral_lists_before_iteration() -> None:
    generation = _generation("a")
    generation_payload = generation.to_dict()

    class ForgedList(list):
        def __iter__(self):
            raise AssertionError("behavioral list must not be iterated")

    generation_payload["artifacts"] = ForgedList(generation_payload["artifacts"])
    with pytest.raises(ValueError, match="artifacts must be a JSON array"):
        GenerationIdentityManifest.from_dict(generation_payload)

    release = generation.artifact_manifest(ArtifactKind.RELEASE)
    release_payload = release.to_dict()
    release_payload["parents"] = ForgedList(release_payload["parents"])
    with pytest.raises(ValueError, match="parents must be a JSON array"):
        ArtifactManifest.from_dict(release_payload)


def test_closed_schema_rejects_artifact_ref_subclass_serialization_resealing() -> None:
    class ForgedArtifactRef(ArtifactRef):
        def to_dict(self) -> dict[str, object]:
            payload = super().to_dict()
            payload["identity_sha256"] = _sha("forged-serialized-artifact")
            return payload

    forged = ForgedArtifactRef(
        ArtifactKind.CORPUS,
        1,
        _sha("validated-artifact"),
    )

    with pytest.raises(ValueError, match="artifact must be an ArtifactRef"):
        ArtifactManifest(schema_version=1, artifact=forged, parents=())


def test_closed_schema_rejects_parent_binding_subclass_serialization_resealing() -> None:
    generation = _generation("a")
    release = generation.artifact_manifest(ArtifactKind.RELEASE)
    checkpoint = release.parent_bindings_by_role()["checkpoint"]

    class ForgedParentBinding(ParentBinding):
        def to_dict(self) -> dict[str, object]:
            payload = super().to_dict()
            artifact = dict(payload["artifact"])  # type: ignore[arg-type]
            artifact["identity_sha256"] = _sha("forged-serialized-checkpoint")
            payload["artifact"] = artifact
            return payload

    forged = ForgedParentBinding(
        role=checkpoint.role,
        artifact=checkpoint.artifact,
        parent_manifest_identity_sha256=checkpoint.parent_manifest_identity_sha256,
    )
    parents = tuple(
        forged if parent.role == "checkpoint" else parent
        for parent in release.parents
    )

    with pytest.raises(ValueError, match="only ParentBinding"):
        ArtifactManifest(
            schema_version=1,
            artifact=release.artifact,
            parents=parents,
        )


def test_closed_schema_rejects_artifact_manifest_subclass_validation_view_resealing() -> None:
    generation = _generation("a")
    release = generation.artifact_manifest(ArtifactKind.RELEASE)

    class ForgedArtifactManifest(ArtifactManifest):
        def parent_bindings_by_role(self) -> dict[str, ParentBinding]:
            return release.parent_bindings_by_role()

        def to_dict(self) -> dict[str, object]:
            payload = super().to_dict()
            payload["parents"] = []
            return payload

        def manifest_identity_sha256(self) -> str:
            return release.manifest_identity_sha256()

    forged = ForgedArtifactManifest(
        schema_version=release.schema_version,
        artifact=release.artifact,
        parents=release.parents,
    )
    artifacts = tuple(
        forged if item.artifact.kind is ArtifactKind.RELEASE else item
        for item in generation.artifacts
    )

    with pytest.raises(ValueError, match="only ArtifactManifest"):
        GenerationIdentityManifest(schema_version=1, artifacts=artifacts)


def test_closed_schema_rejects_tuple_container_subclasses() -> None:
    generation = _generation("tuple-container")
    release = generation.artifact_manifest(ArtifactKind.RELEASE)

    class ForgedTuple(tuple):
        def __iter__(self):
            return super().__iter__()

    with pytest.raises(ValueError, match="parents must be an immutable tuple"):
        ArtifactManifest(
            schema_version=release.schema_version,
            artifact=release.artifact,
            parents=ForgedTuple(release.parents),
        )

    with pytest.raises(ValueError, match="artifacts must be an immutable tuple"):
        GenerationIdentityManifest(
            schema_version=generation.schema_version,
            artifacts=ForgedTuple(generation.artifacts),
        )


def test_closed_scalar_and_encoded_inputs_reject_behavioral_subclasses() -> None:
    class ForgedStr(str):
        def strip(self) -> str:
            return "forged-valid"

    class ForgedInt(int):
        pass

    class ForgedBytes(bytes):
        def decode(self, *args: object, **kwargs: object) -> str:
            raise AssertionError("behavioral bytes subclass decode must never run")

    with pytest.raises(ValueError, match="schema_version must be a positive integer"):
        ArtifactRef(ArtifactKind.CORPUS, ForgedInt(1), _sha("scalar-version"))

    with pytest.raises(ValueError, match="identity_sha256 must be an exact lowercase SHA-256"):
        ArtifactRef(ArtifactKind.CORPUS, 1, ForgedStr(_sha("scalar-hash")))

    parent = _generation("scalar-parent").artifact_manifest(ArtifactKind.CORPUS)
    with pytest.raises(ValueError, match="parent role must be canonical lower_snake_case"):
        ParentBinding(
            role=ForgedStr("corpus"),
            artifact=parent.artifact,
            parent_manifest_identity_sha256=parent.manifest_identity_sha256(),
        )

    with pytest.raises(ValueError, match="manifest input must be bytes"):
        parse_generation_identity_manifest(ForgedBytes(b"{}"))


def test_artifact_ref_from_dict_rejects_behavioral_kind_string_before_enum_lookup() -> None:
    class ForgedKind(str):
        def __hash__(self) -> int:
            raise AssertionError("behavioral kind hash must not run")

        def __eq__(self, other: object) -> bool:
            raise AssertionError("behavioral kind equality must not run")

    payload = {
        "kind": ForgedKind("model_spec"),
        "schema_version": 1,
        "identity_sha256": _sha("kind-discriminator"),
    }
    with pytest.raises(ValueError, match="kind must be an exact string"):
        ArtifactRef.from_dict(payload)


def test_identity_snapshots_revalidate_after_object_setattr_mutation() -> None:
    ref = _ref(ArtifactKind.CORPUS, "stale-ref")
    object.__setattr__(ref, "schema_version", 0)

    with pytest.raises(ValueError, match="schema_version must be a positive integer"):
        ref.to_dict()

    manifest = bind_artifact(_ref(ArtifactKind.CORPUS, "stale-manifest"))
    object.__setattr__(manifest.artifact, "identity_sha256", "0" * 63)

    with pytest.raises(ValueError, match="identity_sha256 must be an exact lowercase SHA-256"):
        manifest.manifest_identity_sha256()


def test_generation_accessors_revalidate_mutated_nested_manifests() -> None:
    generation = _generation("stale-generation")
    corpus = generation.artifacts[2]
    object.__setattr__(corpus, "schema_version", 0)

    with pytest.raises(ValueError, match="schema_version must be a positive integer"):
        generation.identity_sha256()

    with pytest.raises(ValueError, match="schema_version must be a positive integer"):
        generation.artifact_ref(ArtifactKind.CORPUS)

    with pytest.raises(ValueError, match="schema_version must be a positive integer"):
        generation.artifact_manifest(ArtifactKind.CORPUS)


def test_identity_builders_reject_behavioral_mapping_subclasses() -> None:
    class ForgedDict(dict):
        def items(self):
            raise AssertionError("behavioral mapping methods must not run")

    refs = _refs("mapping-closed")
    with pytest.raises(ValueError, match="refs must be an exact dict mapping"):
        build_generation_identity_manifest(ForgedDict(refs))

    generation = _generation("mapping-parent")
    packing = generation.artifact_ref(ArtifactKind.PACKING)
    parents = {
        "split": generation.artifact_manifest(ArtifactKind.SPLIT),
        "tokenizer": generation.artifact_manifest(ArtifactKind.TOKENIZER),
    }
    with pytest.raises(ValueError, match="parents must be an exact dict mapping"):
        bind_artifact(packing, parents=ForgedDict(parents))

    manifest = bind_artifact(packing, parents=parents)
    with pytest.raises(ValueError, match="expected_parents must be an exact dict mapping"):
        verify_parent_bindings(
            manifest,
            expected_parents=ForgedDict(parents),
        )


