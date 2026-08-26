# DATA-227 — External-real code source admission V2

DATA-227 closes the specific D03 modality gap that remained after DATA-23/24/28: authoritative external-real code training bytes.

## Reconstructed boundary

- DATA-23 mechanically inspected real code but its `itsdangerous` and `pluggy` objects were rights-blocked. DATA-227 does not reinterpret or admit them.
- DATA-24 v2 remains the sole model-training rights resolver. Public access or an SPDX label alone does not authorize training.
- DATA-28 remains the source-code fidelity authority: strict UTF-8, byte-identical normalization, fail-closed generated/minified/binary rejection.
- No standalone live DATA-182 authority was found; DATA-227 therefore does not invent or inherit one.

## Newly reviewed independent families

1. `encode/httpx` at `b5addb64f0161ff6bfe94c124ef76f6a1fba5254`, object `httpx/_content.py`, BSD-3-Clause.
2. `psf/requests` at `5460f467b02e49471c0fd6cfc9ca0adab6351f98`, object `src/requests/_internal_utils.py`, Apache-2.0.

Both canonical repositories are runtime-checked as non-forks and non-mirrors. Each source object is pinned by commit, path, Git blob SHA-1 and byte size. License evidence is independently pinned by its Git blob identity. The committed policy decision separately records acquisition, storage, analysis, model-training, and redistribution permissions and redistribution obligations.

## Admission path

The exact-head workflow uses the universal execution bootstrap in `LOCAL_FREE`, downloads only pinned bounded bytes, verifies Git blob identity, excludes vendored/generated/minified/binary/secret-like material, applies DATA-28 identity normalization, writes content-addressed D03 snapshots outside Git, builds a DATA-24 v2 registry, and requires `EligibilityResolver.assert_model_training_eligible` for every object.

Exact SHA-256 duplicate rejection and 5-token-shingle Jaccard near-duplicate rejection at `>= 0.85` run before training. A bounded Product `Trainer` proof streams two 64-byte-token windows from each independent family through a scratch decoder for exactly four optimizer steps.

Raw source snapshots are uploaded only with the corresponding license evidence in the retained workflow artifact. No paid compute is used.
