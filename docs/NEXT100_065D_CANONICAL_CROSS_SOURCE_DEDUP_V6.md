# NEXT100-065D — canonical cross-source dedup V6

## Purpose

V6 closes the live mismatch between the active NEXT100-065C/V5 dedup graph and the stronger NEXT100-063 V3 source-authority vector. V5 already re-materializes the V4 graph, terminal MDN prose, and only the 14 DATA-228-accepted CPython chunks. V6 adds the exact-green NumPy authority and the exact terminal Project Gutenberg seal before executing the existing lineage-aware global exact/normalized/near/fragment/code-skeleton dedup engine.

This is a stacked successor to PR #632. It does not replace the canonical registry and does not create a parallel D05, corpus-freeze, tokenizer, or training implementation.

## Exact authority reconciliation

The V5 parent is pinned to PR #632 head `7fc6e3ec43ee7fb4361cd5d9b4e795bc3fd7c4b5`.

The canonical source-registry reference is NEXT100-063 V3 identity `66866a35d58b2f34431068a161986fc3eeb656e5ded1ca2ff8b40489049bac8c`.

NumPy is bound to NEXT100-049 head `bca7a4c8afc5cb2546c35e3a0ebad9619cd3a4a8`, exact-green workflow `32998548535`, authority identity `e9d2ce633915d6b6844b35e4abb0188974ef4791b208362c4f106ec0ad79ca70`, upstream NumPy commit `4f94a9ac128175d05992ce9946e5b066603c0d9d`, and five exact Git blobs totaling 36,898 bytes. All five files remain one family, `github:numpy/numpy`.

Gutenberg is bound to NEXT100-107 terminal seal head `c50b3f9cf871792c03886bdc1ccdc144812be88f`, authority identity `1b1bad11b688826ee4f73701c08e3b5af76ba16e8d8a806e008d5b84bee0b97b`, and parent execution workflow `32998859164 = success`. The three exact normalized records total 1,672,110 bytes and remain one family, `en.project-gutenberg.public-domain-books`. The V6 runner reacquires the immutable transport blobs and independently reproduces `NEXT100_033_PG_BODY_NFC_LF_V1` before admitting them to the comparison graph.

CPython remains accepted-only. The V5 materializer must reproduce exactly 14 accepted chunks, two `pii_phone` rejections, and exactly 15,540 eligible bytes. The complete 17,901-byte normalized source is never credited.

## Pre-global-dedup vector

The exact V6 input graph is 31 objects and 14 declared source families:

- Ukrainian: 100,856 bytes / 4 families.
- English: 1,838,293 bytes / 5 families.
- Code: 106,031 bytes / 5 families.
- Total: 2,045,180 bytes / 14 families.

These are source-capacity bytes before the V6 global dedup result. They are not tokenizer tokens, unique causal-loss positions, training-token exposures, or a corpus-release claim.

## Execution

Run with:

```bash
PYTHONPATH=src python tools/run_next100_065d_canonical_cross_source_dedup_v6.py run \
  --base-inventory configs/data/next100_065_cross_source_dedup_v3.json \
  --v4-extension configs/data/next100_065b_cross_source_dedup_v4.json \
  --v5-config configs/data/next100_065c_cross_source_dedup_v5.json \
  --v6-config configs/data/next100_065d_canonical_cross_source_dedup_v6.json \
  --report /tmp/next100-065d-v6-report.json
```

The report contains hashes, byte counts, match metadata, capacity discounts, and authority identities only. Raw source text is not emitted.

## Required successor gates

A green V6 dedup is not Research Corpus V1. The remaining ordered gates are exact pre-decontamination record freeze, reserved-evaluation decontamination, post-composition quality/privacy and family-cap enforcement, cluster-safe split, deterministic packing, two byte-identical clean builds, exact post-pack unique non-ignored causal-loss accounting, terminal D05 checkpoint integrity, bounded MODEL-341 training/checkpoint requalification, LEARN-345 refresh, and explicit material-compute authorization.

`LOCAL_FREE` only. No tokenizer fit, optimizer update, long training, final-test payload access, learned-20M/100M claim, or paid compute is authorized by this package.
