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
references. A parent binding includes the parent's kind, schema version and exact SHA-256,
not only a human-readable label.

`GenerationIdentityManifest` is a closed-world selected-generation graph. It contains
exactly one selected artifact of every canonical identity family and requires every derived
artifact to point to the exact selected parent identity for that generation.

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
- training-run A bound to ModelSpec B;
- export bound to the wrong parent kind;
- parent role deletion;
- schema/hash/type ambiguity;
- non-canonical parent ordering.

All fail closed before a generation identity can be accepted.

## Truth boundary

This Section grants no corpus admission, tokenizer fitting, training, optimizer execution,
checkpoint physical qualification, final-test access, learned weights, paid compute, scale
promotion, release, Windows/NVDA or server authority.

Section 1 remains non-DONE until Section 0 is DONE, this exact delta is integrated onto the
accepted live lineage, and terminal exact-head evidence is recorded in the canonical
Section closure registry.
