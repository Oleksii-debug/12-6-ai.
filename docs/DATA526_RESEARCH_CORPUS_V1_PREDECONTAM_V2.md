# DATA-526 Research Corpus V1 pre-decontamination gate V2

Status: `BLOCKED_WAIT_SOURCE_AND_GLOBAL_DEDUP_TERMINAL`.

V2 corrects the V1 unblock order. Terminal source-registry convergence by itself is not sufficient to freeze a Research Corpus V1 pre-decontamination candidate. The complete composed source graph must first pass the successor global exact/near/fragment/lineage dedup authority.

## Current exact nonterminal dependency snapshot

Source convergence candidate:
- PR #538, head `226cbc26710a75af4a864576220b270089e7c52b`;
- V4 registry identity `9fc400a3144b46c481e45d043b0a3365eb2129c83bbacde6f9e7af8a41fadc58`;
- 31 source objects / 14 independent families;
- UK 100,856 numeric bytes / 4 families;
- EN 1,838,293 numeric bytes / 5 families;
- code 106,031 numeric bytes / 5 families;
- total numeric source capacity 2,045,180 bytes;
- normalized source envelope 2,047,541 bytes, of which 2,361 bytes remain uncredited;
- exact-head CI is queued/nonterminal, so this candidate is not consumed as terminal authority.

Global dedup candidate:
- PR #632, head `8181b247fc305f96f4be02d8630ce18cdcf63eae`;
- worker `NEXT100-065D-CROSSSOURCE-DEDUP-V6`;
- dedicated run `33008762043` is pending/nonterminal;
- expected input is the same 31-object / 2,045,180-byte / UK4-EN5-code5 family vector;
- no terminal post-dedup report hash, post-dedup unique-capacity value, or record-inventory identity is claimed.

Source bytes are planning/source-capacity units. They are not tokenizer tokens, post-pack causal-loss positions, training-token exposures, epochs, FLOPs, or evidence of a learned model.

## Corrected P0 order

1. Terminalize the canonical source registry at one exact consumed head.
2. Terminalize the composed global exact/near/fragment/lineage dedup at one exact consumed head over the same source vector.
3. DATA-526 consumes the immutable terminal post-dedup report and materialized record graph, sorts the exact surviving records, and freezes a cryptographic pre-decontamination candidate identity.
4. Reserved-evaluation decontamination runs only against that frozen post-dedup identity.
5. Post-composition quality/privacy, balance/family caps, cluster-safe split/packing, two clean byte-identical builds and the post-pack unique causal-loss ledger remain mandatory.
6. Only after those gates may tokenizer/training authorization be considered.

## Truth boundary

While either current exact head is queued, pending, running, cancelled, stale, or otherwise nonterminal:
- candidate freeze is false;
- record count is zero in this blocker;
- post-dedup capacity is unknown;
- authorized unique optimized targets are zero;
- optimizer updates are zero;
- decontamination is not executed;
- tokenizer fitting, long training and paid compute remain prohibited.

The machine authority is `configs/data/research_corpus_v1_predecontam_blocker_v2.json`. Its validator fail-closes on source/dedup vector mismatch, queue-as-pass promotion, fabricated post-dedup evidence, premature record freeze, byte/token conflation and training/compute promotion.
