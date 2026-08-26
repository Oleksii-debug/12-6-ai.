# DATA-526 — Research Corpus V1 pre-decontamination intake

## Result

DATA-526 executes the first P0 action from the live ~20M readiness controller: compose a successor Research Corpus V1 intake from terminal source authorities and freeze one exact pre-decontamination candidate-set identity.

This package is intentionally **not** a corpus release and does not authorize training.

Exact identities:

- candidate-set identity: `70b519d40ae921c7f8bee3e65e2047b26b266666c15622298c495d9924c647e8`
- manifest identity: `8d56bf3884be4e9de3b0d024c48f436ee137da38ed2bd08c9a88e4228abe85e7`
- DATA-300 frozen contract identity: `07d7beaaff4616e839450de6af3d407855c832bf75a24a959d1a12de5d9364e5`
- DATA-301 terminal-blocker head: `8820ba1b255f6bb95c7db0531fd846078a1aae01`

## Composition

The exact incumbent DATA-300/DATA-301 five-object inventory is retained without mutation:

- two Standard Ebooks manual objects in one English family;
- one Ukrainian Verkhovna Rada laws object;
- one HTTPX code object;
- one Requests code object.

Two terminal bounded source authorities are composed as successor candidates:

- NEXT100-037 CPython tutorial documentation, authority `46a00dc70db690ae2b3c4495a75283e7e752bdccb1047d4318c2ebadfa392f0d`, head `5a6a495a24bce449334cbc5126d0114f61a9f57c`;
- NEXT100-022 Ukrainian Wikisource Lesya Ukrainka 1892 page snapshot, authority `6b443faa7fef777214022028d5fdb356dae0ab1a9b71822b4e16bea8f92cd0d6`, head `84c51e42b6daa51796fd20d793b5ef1ff01cc9d2`.

The frozen intake therefore contains exactly seven source objects and six independent source families. At the source-authority level the stratum family vector is now:

- Ukrainian: 2 independent families;
- English: 2 independent families;
- code: 2 independent families.

This resolves the *source-authority composition* form of the old one-family Ukrainian/English blocker. It does **not** establish final DATA-295 balance/diversity release authority, because the new candidates still require materialization and global gates.

## Byte accounting boundary

Authority-bound source-level normalized bytes total `202,441`:

- Ukrainian: `90,044`;
- English: `102,694`;
- code: `9,703`.

These numbers are provenance accounting, not trainable-capacity accounting.

The CPython authority binds a 17,901-byte normalized source object, but only 14 of 16 quality/privacy chunks are accepted and two chunks are rejected for `pii_phone`. DATA-526 therefore freezes all 14 accepted chunk hashes, keeps the rejected chunks excluded, and leaves exact eligible CPython bytes as `null` until an exact accepted-chunk materialization exists. It never converts the full 17,901 bytes into training capacity.

The Ukrainian Wikisource snapshot has training rights for the exact bounded object but remains explicitly blocked from corpus-training selection until the standard reserved-evaluation exact/near-match decontamination runs.

## Training boundary

DATA-526 authorizes exactly:

- `0` unique optimized causal targets;
- no tokenizer fitting;
- no model update;
- no corpus identity;
- no shard identity;
- no replay or replacement sampling;
- no long training;
- no paid compute.

No learned ~20M checkpoint is created or implied by this package.

## Immediate successor

The next action is exact and machine-bound:

`RUN_STANDARD_EXACT_NEAR_MATCH_RESERVED_EVALUATION_DECONTAMINATION`

The successor must consume candidate-set identity:

`70b519d40ae921c7f8bee3e65e2047b26b266666c15622298c495d9924c647e8`

It may not silently add sources. Any source addition/removal, revision/hash/family change, or rights-purpose change requires a new pre-decontamination intake identity.

After decontamination, the remaining ordered work is:

1. materialize only authorized accepted source bytes;
2. rerun global quality/privacy and lineage-aware deduplication;
3. freeze cluster-safe splits and the 45/35/20 balance contract on actual eligible bytes;
4. produce two byte-identical clean corpus builds;
5. build the unique no-replay loss-position ledger;
6. requalify CHECKPOINT-346 and bounded TRAIN-344 mechanics against MODEL-341;
7. refresh LEARN-345 against terminal corpus/model/optimizer/recovery authorities;
8. request material compute authorization only after the campaign is data-ready.

## Truth boundary

This change advances source composition and creates the exact identity that NEXT100-066 lacked. Research Corpus V1 remains nonterminal, real ~20M training remains blocked, and the primary 20,613,440-parameter MODEL-341 mechanics identity is unchanged.
