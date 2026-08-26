# NEXT100-030 Ukrainian OER source authority

Worker: `NEXT100-030-DATA-UA-OER`

## Candidate

The bounded candidate is the Ukrainian translation of *The Rust Programming Language* in `rust-lang-ua/rustbook_ukrainian`, pinned to commit `ca2d2e4f4434c661836926017af23bdd40ad4e3d` and restricted to the `second-edition` subtree.

The training payload allowlist is exactly three Chapter 1 Markdown files. Top-level translation trees, generated output, images, build tooling, print-production material, contributor metadata, glossary/dictionary material, and evaluation-reserved content are excluded.

## Rights boundary

`second-edition/LICENSE-MIT` and `second-edition/LICENSE-APACHE` are both hash-bound. This authority does not depend on interpreting the two-file license surface as OR versus AND: redistribution must preserve the MIT copyright/permission notice and satisfy Apache-2.0 redistribution/noticing requirements. These licenses permit use, modification, derivative works and redistribution, so the bounded source is eligible for model-training use under the project's explicit-purpose policy.

Evaluation remains `NOT_SEPARATELY_ADMITTED` and no model training is executed.

## Provenance and family

Family ID: `rust-book.documentation.uk-translation`.

The Ukrainian repository is a translation/derivative of the Rust Book and therefore counts as one work-lineage family. Mirrors, forks, alternate hosts and later editions do not earn extra family credit. The family is distinct from the current DATA-287 Rada, Standard Ebooks, HTTPX and Requests families and from the Ukrainian selection-validation Kubernetes/Lang-UK lineages.

## Quality / privacy / dedup

The materializer performs strict UTF-8/NFC normalization, removes code/markup surfaces to produce a prose-only normalized derivative, verifies Ukrainian-script evidence, screens secrets/email/phone-like strings, checks intra-source exact and 5-token shingle near-duplicates, compares exact and available near-duplicate evidence against DATA-287 incumbents, and checks the hash-only reserved-fingerprint registry without reading evaluation payloads.

The source is technical educational Ukrainian. Manual review found coherent instructional prose with some legacy translation/editing imperfections, so this authority treats it as technical-language data rather than gold literary prose.

## State

Initial publication is `PROBE`; terminal `ADMIT` requires exact lock values from two byte-identical LOCAL_FREE materializations and a second locked exact-head run.
