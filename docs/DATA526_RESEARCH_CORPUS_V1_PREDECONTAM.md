# DATA-526 Research Corpus V1 pre-decontamination gate

Status: `BLOCKED_WAIT_SOURCE_CONVERGENCE`.

DATA-526 is the composition point that must eventually freeze one exact Research Corpus V1 pre-decontamination candidate identity. It deliberately does not compose a moving or partially terminal source registry, and it must not freeze raw pre-global-dedup registry rows.

## Exact dependency snapshot

The incumbent DATA-300 contract remains bound at `8ea7f830e50a23754d189dd4134f4afad76a7ee9`, contract identity `07d7beaaff4616e839450de6af3d407855c832bf75a24a959d1a12de5d9364e5`. DATA-301 remains terminal-blocked at `8820ba1b255f6bb95c7db0531fd846078a1aae01`, evidence identity `939065abeefff8aed924415589608ff3fc721fe4b0a57fc200146a4b6a137e81`, with no corpus identity, no shard identity, and zero authorized balanced no-replay capacity.

Issue #526 requires the late source vector to come through the surviving NEXT100-063 issue #530 rather than by hard-coding whichever source PRs happen to be visible. The earlier issue #521 and PR #527 are superseded lineage. The active canonical candidate is draft PR #538 at observed exact head `226cbc26710a75af4a864576220b270089e7c52b`, based on DATA-287 head `b0523ccbc4b957615aac849d476cfa851be87578`.

The current V4 machine registry reports identity `9fc400a3144b46c481e45d043b0a3365eb2129c83bbacde6f9e7af8a41fadc58` and a provisional pre-successor-global-dedup source vector of 2,045,180 source-capacity bytes across 14 independent families: Ukrainian 100,856 bytes / four families, English 1,838,293 / five, and code 106,031 / five. The vector includes the exact 15,540-byte accepted-only CPython capacity and exact terminal Gutenberg/NumPy additions; it does not convert source bytes into tokenizer tokens or optimized causal-loss positions.

Exact-head Actions for PR #538 are currently queued. Queue state is not PASS authority, so DATA-526 records this V4 vector only as non-authoritative dependency context and consumes zero terminal source-convergence authority from it.

## Global dedup is now an explicit prerequisite

The earlier DATA-526 blocker sequence was too weak: it allowed the candidate-freeze step immediately after terminal source-registry convergence. That would permit a future implementation to freeze the pre-dedup registry graph and bypass the successor global exact/near/fragment/lineage dedup stage.

The live successor dedup lane is NEXT100-065D / PR #632, observed here at head `704a558545b158fecff5cb41ad5bd16f93884cdd`. Its V6 report schema is `12-6.next100-065d-cross-source-dedup-report.v6`. The current composed input vector is 31 source objects / 2,045,180 source-capacity bytes / 14 independent families, including exact 15,540-byte accepted-only CPython capacity. The exact-head V6 workflow is queued and therefore remains nonterminal.

DATA-526 now requires immutable terminal V6 evidence before any candidate freeze. The terminal handoff must include exact-head success, an immutable V6 report identity, conservative post-global-dedup capacity, retained/excluded object decisions, and an explicit queue-is-not-pass boundary.

## What is intentionally absent

While either source convergence or successor global dedup is nonterminal, DATA-526 publishes no successor record inventory, no record-inventory digest, no candidate-set digest, no corpus or shard identity, and no training capacity. Reserved-evaluation decontamination is not allowed to run against this blocker because there is no frozen post-global-dedup candidate identity to scan.

No final-test payload or outcome is accessed. No tokenizer fit, model training, optimizer update, corpus materialization, paid compute, replay, duplication, or quota repair is authorized by this package.

## Unblock sequence

1. NEXT100-063 / PR #538 must become terminal-success at the exact consumed source-convergence authority and expose the exact admitted training-purpose source/object/family/content/normalization vector.
2. NEXT100-065D / PR #632 must become terminal-success at the exact consumed head and publish immutable V6 evidence over the complete composed source graph.
3. DATA-526 must freeze only the post-global-dedup retained record/object graph and preserve exact lineage to both terminal authorities. Raw pre-dedup registry rows may not be frozen directly.
4. Only that exact post-global-dedup candidate identity may be handed to the reserved-evaluation exact/near-match decontamination successor.

The machine authority is `configs/data/research_corpus_v1_predecontam_blocker_v1.json`. The stdlib validator rejects nonterminal-as-terminal promotion for both upstream authorities, source/dedup vector disagreement, wrong dedup PR/head/schema, a weakened global-dedup ordering gate, fabricated candidate identities, pending-byte credit as optimized targets, and any training/compute authorization while blocked.
