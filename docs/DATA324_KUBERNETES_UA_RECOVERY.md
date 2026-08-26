# DATA-324 Kubernetes Ukrainian Documentation Recovery

Worker: `NEXT100-024-DATA-UA-KUBERNETES-RECOVERY`

Verdict target: `ADMIT`, conditional on exact-head recovery workflow success.

## Recovery boundary

DATA-228 is retained as a terminal environment/materialization failure, not a rights rejection. Its exact Kubernetes candidate was pinned at `kubernetes/website` commit `25f3dcbed7429ebe20174ccc7000428d0f0aedda`, path `content/uk/docs/concepts/_index.md`, source Git blob `ab9d757e99679b3db48a3230bf6eb07a997eec9c`, under the repository CC BY 4.0 license blob `da6ab6cc8f333d7e89a99812866df8f24374d47c`.

A live upstream refresh on 2026-08-26 observed Kubernetes website `main` at `65be3d5113725bf3cb73135e9b327cc71a183ccb`; the selected Ukrainian source path still resolves to the same Git blob. Recovery therefore uses the original immutable DATA-228 revision rather than silently changing corpus identity.

## Rights

The exact pinned `LICENSE` is Creative Commons Attribution 4.0 International. The project admits this bounded source for model-training purpose because the license grants reproduction, sharing, and adapted-material rights, and the project retains the required attribution and modification notice. Redistribution is allowed only with CC BY 4.0 attribution, license reference, supplied notices where applicable, and indication of normalization changes. No endorsement may be implied.

Evaluation use remains `NOT_SEPARATELY_ADMITTED` and requires a separate purpose authority.

## Bounded deterministic acquisition

The V1 file set is exactly one UTF-8 Markdown object:

- `content/uk/docs/concepts/_index.md`

The acquisition URL is constructed only from canonical repository, exact 40-hex revision, and exact path. The source is bounded to 100,000 bytes, the license to 50,000 bytes, combined network acquisition to 150,000 bytes, and normalized output to 100,000 bytes. Git blob SHA-1 is verified before any admission evidence is emitted.

The materializer is Python-stdlib-only and does not import the project training stack, PyTorch, tokenizer code, or model code. This directly removes the prior `ENVIRONMENT_BOOTSTRAP_MISSING_TORCH` failure mode.

## Normalization and Ukrainian-language evidence

Normalization is deterministic: strict UTF-8, LF line endings, Unicode NFKC, removal of YAML frontmatter, removal of HTML comments including embedded English originals, removal of Hugo shortcodes, retention of Markdown link labels and image alt text, basic Markdown punctuation removal, whitespace collapse, empty-line removal, and one terminal LF.

The Ukrainian gate is evaluated after English-comment removal. It requires at least 70% Cyrillic among alphabetic characters and at least 20 occurrences from `ІіЇїЄєҐґ`. The machine report records counts and ratios.

## Family lineage and dedup

Canonical family: `kubernetes.website.docs`, canonical upstream `github:kubernetes/website`.

Sibling Kubernetes website documents and language translations do not create new family credit merely because path, URL, or language differs. Any future English or other-language Kubernetes website snapshot must collapse to the same upstream family unless an explicit later lineage authority proves otherwise.

The recovery compares raw and normalized SHA-256 identities against the current DATA-293 admitted text inventory and rejects exact collisions. The normalized SHA-256 is the corpus dedup key. Broader near-duplicate scanning remains a corpus-assembly gate and is not replaced by family counting.

## Materialized evidence

Successful branch materialization writes:

- exact raw source bytes;
- deterministic normalized UTF-8 text;
- exact pinned CC BY 4.0 license bytes;
- attribution/modification notice;
- immutable manifest with raw, normalized, license and manifest hashes;
- recovery report with rights, language, privacy, dedup, evaluation and training-executed decisions.

The workflow materializes twice independently and byte-compares every generated object before committing or accepting PR state. `training_executed=false` is part of the report and manifest.

Materialized V1 identities:

- raw SHA-256: `50a790e0ece091f13fe039b5e36a23431680dec0357379f29b0029502f9b3a31`;
- normalized SHA-256: `32fccd0a0d2aa8a755a60e768443bdca1a743566b9482954031754a24a1f7bd5`;
- license SHA-256: `9ba9550ad48438d0836ddab3da480b3b69ffa0aac7b7878b5a0039e7ab429411`;
- manifest identity: `e41dc8b677760b1d6ac25c982baf9ed47388afac4cb29872e919aa1f02faa32a`;
- recovery report identity: `a3f67f96042cdcec4b2e043efcdb159a09196f58cef435f15577da52c0b5ac73`.

The materialization producer run is not by itself the terminal exact-head authority because it creates the materialization commit. A subsequent exact-head push run must pass with zero generated diff before the final `ADMIT` is sealed.
