# Section 1 — Unified identities and manifest cross-binding

## Authority

This document implements the current canonical Section Plan requirement:

- **Section 1 — Єдина система ідентичностей і маніфестів**
- plan document: `16KotBqgSyf3A0FEWpN8Pnobgf2ZJ1ecT8MLgHoQXibY`
- observed plan revision:
  `AHj4eMRLaHccAytgXlAMJ8VWh7OI8IBzEvpQtlHf2LCzdZ-MS4Asxbyf4K6DsKkQqi43tx_lYVEkT5UEAut8l8X7EhXLj3Zuv65xmgfsdw`

This work is stacked after Section 0 and cannot become canonical DONE before Section 0.

## Design rule: reuse domain identities, do not replace them

The project already has domain-specific identities and self-hashes in ModelSpec/InitSpec,
corpus/data authorities, tokenizer decisions, packing manifests, exposure ledgers, run
packets, checkpoints, evaluation authorities and exports.

`src/twelve_six/artifact_identity.py` is a thin composition layer over those identities.
It does not introduce a second hashing algorithm for the underlying artifact. Each
`ArtifactRef` records:

- canonical artifact kind;
- the artifact's own schema version;
- the exact existing lowercase SHA-256 identity.

The unified identity vocabulary is pinned to the original twelve-kind tuple by the
generation validator and its typed accessors. Rebinding the public module-level tuple therefore
cannot shrink or reorder what an already loaded validator accepts as a complete generation.
The twelve canonical `ArtifactKind` wire strings are sealed independently at module load.
Canonical wire reads use the immutable underlying `str` payload and cross-check the Enum
member's stored `_value_` without dispatching the mutable `.value` descriptor. Class-level
descriptor rebinding therefore cannot split validation from canonical serialization, while a
low-level mutation of an enum singleton value is still rejected before that kind can be
serialized, hashed or accepted by generation validation.

The unified identity vocabulary covers exactly:

1. ModelSpec
2. InitSpec
3. corpus
4. tokenizer
5. split
6. packing
7. exposure ledger
8. training run
9. checkpoint
10. evaluation
11. export
12. release

## 1.1 — versioned identities

`ArtifactKind` provides one stable canonical name per required identity family.
`ArtifactRef` rejects non-enum kinds, bool/non-positive schema versions and malformed
SHA-256 values.

Permanent tests explicitly project the existing `ModelSpec.identity_sha256()` and
`InitSpec.identity_sha256()` into the unified layer without changing their hashes.

## 1.2 — cryptographic parent cross-binding

`ArtifactManifest` wraps one exact artifact identity with canonically ordered named parent
bindings. Each parent binding includes the parent's kind, schema version and exact SHA-256
plus the SHA-256 identity of that parent's own `ArtifactManifest`. The latter commits to
the parent's parents recursively, so changing an ancestor changes every descendant binding
even if an intermediate raw artifact identity is deliberately held constant.

`GenerationIdentityManifest` is a closed-world selected-generation graph. It contains
exactly one selected artifact of every canonical identity family and requires every derived
artifact to point to the exact selected parent artifact and transitive parent-manifest
identity for that generation.

The canonical parent policy is runtime-immutable: both the outer kind map and every nested
role map are sealed after module initialization. The validator also captures that sealed
mapping at class definition, so later rebinding of the module-global policy cannot silently
change acceptance semantics; a builder influenced by such a rebind fails closed at validation.
Stored parent/artifact tuples are exact built-in tuples rather than behavioral subclasses, so
validation and later canonical serialization cannot observe different container views.

Canonical identity objects are also revalidated whenever they are serialized, traversed,
bound, verified or assembled into a generation. Constructor-time validation is not trusted
after object creation, so low-level post-validation mutation cannot turn a once-valid exact
dataclass into a new accepted identity. Identity-bearing builder inputs use exact built-in
`dict` mappings rather than arbitrary behavioral `Mapping` implementations.

Artifact-reference equality is not an authority boundary. Duplicate/self-parent checks and
cross-generation parent matching compare revalidated canonical scalar signatures instead of
dispatching `ArtifactRef.__eq__` or `__hash__`, so class-level dunder rebinding cannot reseal
an incompatible parent as the selected generation identity.

Generation acceptance, parent binding and parent verification read exact stored parent tuples
directly rather than dispatching mutable parent-view methods. Parent-lineage construction and
comparison use a sealed stored-state manifest hasher, so rebinding
`ArtifactManifest.parent_bindings_by_role` or `ArtifactManifest.manifest_identity_sha256`
cannot forge a bound lineage or hide a post-validation parent/lineage mutation while different
stored state is serialized.

Identity-bearing serializers and generation accessors route through private stored-state
validators/serializers whose authority is not exposed as caller-supplied override parameters.
Rebinding class-level `__post_init__` or `to_dict`, or attempting to pass an alternate
validator/serializer/hasher into the public API, therefore cannot turn malformed stored state
into a newly accepted or differently hashed identity after construction.

The canonical graph is:

- ModelSpec, InitSpec and corpus: roots inside this Section-1 graph;
- tokenizer -> corpus;
- split -> corpus;
- packing -> split + tokenizer;
- exposure ledger -> packing;
- training run -> ModelSpec + InitSpec + corpus + tokenizer + split + packing + exposure ledger;
- checkpoint -> training run;
- evaluation -> checkpoint + split;
- export -> checkpoint;
- release -> checkpoint + evaluation + export.

This graph is a cross-binding contract, not a claim that every listed artifact already has
terminal product evidence. Later Sections remain responsible for creating and qualifying
their physical artifacts.

The regression suite attempts caller-coherent generation mixing, including:

- packing from generation A bound to split B;
- generation-A packing bound to a tokenizer with the same raw tokenizer SHA-256 but a
  generation-B corpus lineage;
- training-run A bound to ModelSpec B;
- export bound to the wrong parent kind;
- parent role deletion;
- schema/hash/type ambiguity;
- non-canonical parent ordering.

All fail closed before a generation identity can be accepted.

### Durable manifest boundary

`GenerationIdentityManifest.canonical_json_bytes()` provides one canonical UTF-8 JSON
serialization. `parse_generation_identity_manifest()` decodes that durable form with a
bounded input size, duplicate-member rejection, standards-strict finite JSON handling and
exact closed field sets at every nested manifest/reference level. It then requires the input
bytes to equal the manifest's canonical serialization exactly, rejecting semantically equal
but differently ordered/whitespace-encoded JSON. The parsed object must re-satisfy the
complete generation graph before it is accepted. The generation identity is the SHA-256 of
those exact canonical bytes.

Direct `from_dict()` schema entry points also require exact built-in `dict`/`list`
containers. Behavioral container subclasses are rejected before lookup or iteration, so a
validation-facing shape cannot later expose different deserialization content.

## Truth boundary

This Section grants no corpus admission, tokenizer fitting, training, optimizer execution,
checkpoint physical qualification, final-test access, learned weights, paid compute, scale
promotion, release, Windows/NVDA or server authority.

Section 1 remains non-DONE until Section 0 is DONE, this exact delta is integrated onto the
accepted live lineage, and terminal exact-head evidence is recorded in the canonical
Section closure registry.
