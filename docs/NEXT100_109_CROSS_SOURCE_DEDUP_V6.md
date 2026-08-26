# NEXT100-109 — global cross-source dedup V6

Status: `STACKED_CANDIDATE_LOCAL_FREE_ONLY`

Base authority: NEXT100-065C / PR #632 exact head `7fc6e3ec43ee7fb4361cd5d9b4e795bc3fd7c4b5`.

## Purpose

NEXT100-063 V3 identified two terminal authorities that were not yet simultaneously present in the active V5 global-dedup graph:

1. bounded first-party NumPy code: 36,898 exact bytes / one code family;
2. terminal-sealed Gutenberg bodies: 1,672,110 normalized bytes / three records / one English family.

V6 reconciles those authorities into the existing V5 lane instead of creating an independent dedup implementation. It reuses the V3 exact/normalized/near-copy/fragment/code-skeleton/lineage engine.

## Exact pre-dedup contract

The inherited V5 graph is 23 objects and must materialize accepted-only CPython to exactly 15,540 eligible bytes. V6 then adds five exact NumPy files and three exact Gutenberg bodies.

Expected complete vector before successor global dedup:

- 31 source objects;
- Ukrainian: 100,856 bytes / 4 families;
- English: 1,838,293 bytes / 5 families;
- code: 106,031 bytes / 5 families;
- total: 2,045,180 bytes.

These are source-capacity bytes before the V3 global dedup result. They are not tokens, packed positions, or training authorization.

## NumPy materialization

V6 reacquires five files from `numpy/numpy` at commit `4f94a9ac128175d05992ce9946e5b066603c0d9d`, verifies exact size and Git blob identity, requires strict UTF-8, no NUL bytes, and Python AST parse, and preserves the bytes exactly. All five files remain one independent family: `github:numpy/numpy`.

Authority: PR #468 head `bca7a4c8afc5cb2546c35e3a0ebad9619cd3a4a8`; dedicated workflow `32998548535 = success`; authority identity `e9d2ce633915d6b6844b35e4abb0188974ef4791b208362c4f106ec0ad79ca70`.

## Gutenberg materialization

V6 reacquires the three sealed GITenberg transport objects, verifies raw SHA-256 and Git blob identity, then reproduces `NEXT100_033_PG_BODY_NFC_LF_V1`: exact Gutenberg start/end markers, body-only extraction, LF normalization, one leading BOM removal when applicable, NFC, outer blank-line trimming, and one final LF. Each normalized body must match the terminal seal byte count and SHA-256.

The three records remain one family: `en.project-gutenberg.public-domain-books`.

Authority: PR #627 head `c50b3f9cf871792c03886bdc1ccdc144812be88f`, seal identity `1b1bad11b688826ee4f73701c08e3b5af76ba16e8d8a806e008d5b84bee0b97b`, binding parent PR #470 exact head `3f4ad26e1e8f3406a1274418cf5f485814ce3032` and dedicated workflow `32998859164 = success`.

## Fail-closed boundaries

V6 refuses parent V5 config drift, accepted-only CPython capacity drift, source identity drift, normalization drift, family/capacity arithmetic drift, or any widening of corpus/training/paid-compute claims.

A green V6 report still authorizes exactly zero optimizer updates. It does not freeze Research Corpus V1 and does not claim decontamination, balance, packing, tokenizer selection, learned 20M weights, or paid compute authorization.

## Next gates after terminal V6 evidence

1. freeze exact record-level pre-decontamination candidate identity;
2. run reserved-evaluation decontamination;
3. rerun post-composition quality/privacy and mixture/family-cap checks;
4. build cluster-safe split and deterministic packed shards twice byte-identically;
5. produce exact post-pack unique nonignored causal-loss ledger;
6. requalify tokenizer candidates and D05 checkpoint integrity;
7. run only bounded local/free pilots until explicit material compute authorization exists.
