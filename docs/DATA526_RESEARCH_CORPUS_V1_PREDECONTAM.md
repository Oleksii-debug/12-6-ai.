# DATA-526 Research Corpus V1 pre-decontamination gate

Status: `BLOCKED_WAIT_SOURCE_CONVERGENCE`.

DATA-526 is the composition point that must eventually freeze one exact Research Corpus V1 pre-decontamination candidate identity. It deliberately does not compose a moving or partially terminal source registry.

## Exact dependency snapshot

The incumbent DATA-300 contract remains bound at `8ea7f830e50a23754d189dd4134f4afad76a7ee9`, contract identity `07d7beaaff4616e839450de6af3d407855c832bf75a24a959d1a12de5d9364e5`. DATA-301 remains terminal-blocked at `8820ba1b255f6bb95c7db0531fd846078a1aae01`, evidence identity `939065abeefff8aed924415589608ff3fc721fe4b0a57fc200146a4b6a137e81`, with no corpus identity, no shard identity, and zero authorized balanced no-replay capacity.

Issue #526 requires the late source vector to come through NEXT100-063 / issue #521 rather than by hard-coding whichever source PRs happen to be visible. The narrower NEXT100-063 PR #527 was closed as superseded. The active canonical candidate is draft PR #538 at exact head `94dce83cbe611144f961b9f93b3be273345a7f62`, based on DATA-287 head `b0523ccbc4b957615aac849d476cfa851be87578`.

PR #538 explicitly states that it does not publish exact-head CI because the repository runner queue is saturated. It is therefore nonterminal for DATA-526 consumption. DATA-526 records its reported pre-global-dedup vector only as dependency context: 565,743 normalized source bytes across 13 independent families, with Ukrainian 100,856 bytes / four families, English 168,544 / four, and code 296,343 / five. Those numbers are not a corpus identity and are not optimized causal loss capacity.

## What is intentionally absent

While NEXT100-063 is nonterminal, DATA-526 publishes no successor record inventory, no record-inventory digest, no candidate-set digest, no corpus or shard identity, and no training capacity. Reserved-evaluation decontamination is not allowed to run against this blocker because there is no frozen candidate identity to scan.

No final-test payload or outcome is accessed. No tokenizer fit, model training, optimizer update, corpus materialization, paid compute, replay, duplication, or quota repair is authorized by this package.

## Unblock sequence

1. NEXT100-063 must become terminal-success at the exact consumed head under its dedicated validation.
2. Its terminal authority must expose the exact training-purpose admitted object/family/content/normalization vector and exclude queued, RETEST, PROBE, draft-only, or otherwise nonterminal candidates.
3. DATA-526 must rebuild from that terminal vector plus the exact incumbent DATA-300/DATA-301 authorities, sort and freeze the exact pre-decontamination record inventory, and emit cryptographic inventory and candidate-set identities.
4. Only that exact frozen candidate identity may be handed to the reserved-evaluation exact/near-match decontamination successor.

The machine authority is `configs/data/research_corpus_v1_predecontam_blocker_v1.json`. The stdlib validator rejects nonterminal-as-terminal promotion, supersession-lineage drift, fabricated candidate identities, pending-byte credit as optimized targets, and any training/compute authorization while blocked.
