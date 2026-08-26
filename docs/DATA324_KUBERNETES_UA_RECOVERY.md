# DATA-324 Kubernetes Ukrainian Documentation Recovery

Worker: `NEXT100-024-DATA-UA-KUBERNETES-RECOVERY`

Verdict target: `ADMIT`, conditional on exact-head recovery workflow success.

## Recovery boundary

DATA-228 is retained as a terminal environment/materialization failure, not a rights rejection. Its Kubernetes candidate was pinned at `kubernetes/website` commit `25f3dcbed7429ebe20174ccc7000428d0f0aedda` under the repository CC BY 4.0 license blob `da6ab6cc8f333d7e89a99812866df8f24374d47c`.

A live upstream refresh on 2026-08-26 confirmed that authoritative Ukrainian Kubernetes documentation remains available. DATA-324 preserves the same immutable upstream revision rather than silently changing provenance.

## Concurrency correction: EVAL-290 reservation

During final concurrency review, terminal EVAL-290 authority was discovered at head `029514654829cebc149cff6fc1fea2a8ba4fa566`, exact-head workflow run `32968339064`, artifact `9606656857`, digest `sha256:0c5f9f8d938284a1358bfb77284814b1f0569b2d6d13938f415ba31db64a6c3b`.

EVAL-290 reserved the original DATA-228 Kubernetes object `content/uk/docs/concepts/_index.md`, raw SHA-256 `50a790e0ece091f13fe039b5e36a23431680dec0357379f29b0029502f9b3a31`, for selection-validation before future training. Its immutable artifact contains four Kubernetes selection-validation records and marks future training of those records prohibited.

DATA-324 therefore does not admit, retain, or count that reserved object as training material. The previously materialized `_index.md` recovery bytes and normalized derivative were removed from this branch after the reservation was discovered.

The replacement bounded candidate is a different exact source object from the same canonical Kubernetes family:

- `content/uk/docs/concepts/overview/what-is-kubernetes.md`;
- exact upstream Git blob `b3c52cab3be6a8efbc33e91893c653df5972a794`;
- exact upstream revision `25f3dcbed7429ebe20174ccc7000428d0f0aedda`.

The recovery workflow computes the replacement raw SHA-256 from downloaded bytes, then fail-closes if it collides with any raw source object bound by EVAL-290. This preserves evaluation isolation without manufacturing an additional family.

## Rights

The exact pinned `LICENSE` is Creative Commons Attribution 4.0 International. The project admits only the non-reserved bounded technical source object for model-training purpose because the license grants reproduction, sharing, and adapted-material rights, and the project retains the required attribution and modification notice.

Redistribution is allowed only with CC BY 4.0 attribution, license reference, supplied notices where applicable, and indication of normalization changes. No endorsement may be implied.

DATA-324 itself grants no evaluation role. EVAL-290 is a separate purpose authority for its separately reserved selection-validation records.

## Bounded deterministic acquisition

The V1 training file set is exactly one UTF-8 Markdown object: `content/uk/docs/concepts/overview/what-is-kubernetes.md`.

The acquisition URL is constructed only from canonical repository, exact 40-hex revision, and exact path. The source is bounded to 100,000 bytes, the license to 50,000 bytes, combined network acquisition to 150,000 bytes, and normalized output to 100,000 bytes. Git blob SHA-1 is verified before admission evidence is emitted.

The materializer is Python-stdlib-only and does not import the project training stack, PyTorch, tokenizer code, or model code. This removes the prior DATA-228 `ENVIRONMENT_BOOTSTRAP_MISSING_TORCH` failure mode.

## Normalization and Ukrainian-language evidence

Normalization is deterministic: strict UTF-8, LF line endings, Unicode NFKC, removal of YAML frontmatter, removal of HTML comments including embedded English originals, removal of Hugo shortcodes, retention of Markdown link labels and image alt text, basic Markdown punctuation removal, whitespace collapse, empty-line removal, and one terminal LF.

The Ukrainian gate runs after English-comment removal. It requires at least 70% Cyrillic among alphabetic characters and at least 20 occurrences from `ІіЇїЄєҐґ`. Exact counts and ratios are emitted in the immutable manifest.

## Family lineage and dedup

Canonical family: `kubernetes.website.docs`, canonical upstream `github:kubernetes/website`.

Sibling Kubernetes website documents and translations do not create new family credit merely because path, URL, or language differs. The EVAL-290 reserved `_index.md` object and the DATA-324 replacement candidate remain one lineage family.

The recovery compares actual raw and normalized SHA-256 identities against the DATA-293 admitted text inventory and rejects exact collisions. It separately compares actual raw SHA-256 against terminal EVAL-290 reserved source-object identities. The replacement normalized SHA-256 is the corpus dedup key only after all gates pass. Broader near-duplicate scanning remains a downstream corpus-assembly gate.

## Materialized evidence

A successful recovery writes exact raw source bytes, deterministic normalized UTF-8 text, exact pinned CC BY 4.0 license bytes, attribution/modification notice, immutable manifest, and recovery report. The workflow materializes twice independently and byte-compares all generated evidence.

`training_executed=false` and `local_free_only=true` remain hard assertions. Final raw/normalized/report/manifest identities are bound only after the replacement candidate materializes successfully and passes exact-head replay.
