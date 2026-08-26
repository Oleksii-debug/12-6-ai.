# DATA-229 Real Snapshot Registry V1

`data/registry/real_snapshots.v1.json` is the single immutable machine-readable registry for real snapshots admitted at the DATA-229 cutoff `2026-08-26T09:48:19Z`.

## Consumed authorities

DATA-213 is terminal-success at source `5ae74eb162641712ff97a326572bec5bfe2b0607`, workflow run `32950910634`, artifact `9600107886`. Its promotion report identity is `6fc791cf0c6d3501535bfd935c847d8e2c1cff8508a4de78255ec1daf9384fd7`, and its D03 dataset identity is `8d5b2269755c8cb9c7bb619a7cdbd09e762826a698f30cf6db08ecc98339f9e9`.

No terminal DATA-227 or DATA-228 evidence was published at this cutoff, so neither worker contributes a source. This is intentional fail-closed behavior. V1 therefore has three external-real text snapshots: two English Standard Ebooks manual objects and one Ukrainian Verkhovna Rada open-data object. It has zero admitted external-real code snapshots and does not invent one.

## Registry semantics

Each source binds the raw source/version/hash/size identity, immutable source-family identity, origin class, modality/language, retrieval identity, normalization policy and retained normalized hashes, purpose-specific rights identities, D03 admitted-chunk identity, and decontamination status.

DATA-213 did not retain a distinct pre-normalization extracted-content hash. V1 records that exact absence as `NOT_RETAINED_BY_DATA213`; it does not synthesize a missing hash. The retained normalized content hash and artifact-file hash are both preserved.

Training, evaluation and redistribution are separate purposes. The DATA-24 authority explicitly admits model training and redistribution for the current three sources. V1 does not infer evaluation authority from those decisions: current evaluation status is `NOT_SEPARATELY_ADMITTED`, so `sources_for_holdout()` returns no current source until a separate evaluation authority is bound.

## Determinism and immutability

Registry bytes are canonical UTF-8 JSON: sorted keys, compact separators, one terminal newline. `registry_identity_sha256` hashes every semantic field except itself. A changed source hash, rights decision, family, origin, retrieval identity, normalization identity, D03 identity, or other included semantic field changes the registry identity.

`tools/build_real_snapshot_registry.py --verify` independently rebuilds V1 from DATA-213/DATA-24 authorities and requires byte equality with the committed registry. Focused tests perform two independent builds and require byte identity.

`verify_source_payload()` verifies materialized raw bytes against the source's immutable raw size and SHA-256 before downstream use.

## Origin separation

Registry IDs are namespaced by origin: `external-real:` and `project-authored:`. Validation requires the namespace to agree with `origin_class` and forbids one `(source_id, source_version)` raw identity from appearing under multiple origins. Relabeling an external source as project-authored without a corresponding, independently admitted source therefore fails closed.

## Consumer API

Corpus and holdout code should load this registry and call `sources_for_corpus()`, `sources_for_holdout()`, or `sources_for_redistribution()`. These APIs filter the canonical registry by purpose, language, modality and origin. They return copies of registry entries; consumers do not maintain duplicate source tables.

## Claim boundary

This registry is provenance and rights infrastructure. DATA-213's decontamination/dedup gates are retained, but V1 does not claim universal benchmark cleanliness, corpus representativeness, production readiness, intelligence, alignment or instruction following. Execution is LOCAL_FREE only.
