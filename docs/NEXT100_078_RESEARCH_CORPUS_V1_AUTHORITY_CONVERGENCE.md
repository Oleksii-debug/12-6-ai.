# NEXT100-078 Research Corpus V1 authority convergence

## Purpose

This successor is stacked on NEXT100-065 cross-source dedup V3 at exact head
`efc278cec0e4773eb4ff405bf4b4d24ee63b5d13`.

It solves one narrow blocker: combine the already-terminal source-authority metadata needed to
remove the independent-family diversity failure without pretending that remote authority bytes
are already a canonical corpus.

No training, tokenizer fitting, paid compute, corpus freeze, or decontamination PASS is claimed.

## Inherited terminal vector

NEXT100-065 currently binds 243,970 bytes of source-authority capacity:

- Ukrainian: 90,044 bytes, 2 independent families.
- English: 84,793 bytes, 1 independent family.
- Code: 69,133 bytes, 4 independent families.

The fixed planning mixture remains 45% Ukrainian / 35% English / 20% code with replay forbidden.
The inherited English family count is therefore the remaining structural diversity failure.

## Exact additive authorities

NEXT100-078 binds only source authorities whose dedicated exact-head workflow completed
successfully:

1. NEXT100-026 / PR #449 / head `40950a950b60921fd856af2719e1ae2486d9e892`.
   KMu Secretariat Ukrainian source family, ADMIT, 9,153 normalized UTF-8 bytes, dedicated run
   `32997970539` SUCCESS.
2. NEXT100-038 / PR #445 / head `902eccc0b3efff09a38dc89cda789180b6c6e754`.
   MDN prose-only English source family, ADMIT_PROSE_ONLY, 6,492 normalized bytes, dedicated run
   `32998544359` SUCCESS.
3. NEXT100-034 / PR #472 / head `b7491745b34ac8679baaf69cb96cd609dcbe0a16`.
   NIST Technical Series English source family, ADMIT, 59,358 normalized bytes across three exact
   bounded objects, dedicated run `32998703545` SUCCESS.

Generic repository-wide CI failures on those source branches are not reclassified as success.
This convergence consumes only their dedicated terminal source-authority result and exact
authority identities.

## Derived authority vector

The fail-closed validator derives:

- Ukrainian: 99,197 bytes, 3 independent families.
- English: 150,643 bytes, 3 independent families.
- Code: 69,133 bytes, 4 independent families.
- Total source-authority capacity: 318,973 bytes.
- Independent-family gate: PASS against the required minimum 2 / 2 / 2.
- Authority-set identity:
  `24831f5388303ee4dfaa1186269f3cd0f52989dc67e58ac546d9dd18a5faf3db`.

This is an authority-set identity, not a corpus identity.

## Capacity truth

At the frozen 45/35/20 no-replay mixture, Ukrainian capacity is now the limiting stratum. The
maximum source-byte mixture supported by the authority vector is 220,437 bytes before downstream
quality, privacy, dedup, evaluation reservation, document-boundary, split, and causal-loss
reductions.

Against the current 20,000,000-source-byte planning target, the remaining raw authority-capacity
gaps are:

- Ukrainian: 8,900,803 bytes.
- English: 6,849,357 bytes.
- Code: 3,930,867 bytes.
- Total: 19,681,027 bytes.

The 20 MB planning target is source capacity. It must not be relabeled as 20 million optimized
causal targets. The final loss ledger can only be known after exact record materialization,
packing, and loss-mask accounting.

## Fail-closed gates that remain

The three additive source authorities still live on their own exact branches. NEXT100-078 does
not copy or synthesize their bytes. Therefore:

- exact pre-decontamination record inventory: BLOCKED;
- canonical corpus identity: BLOCKED;
- exact / near-match evaluation decontamination: BLOCKED;
- whole-corpus quality, privacy, cross-source dedup and cluster-safe split: BLOCKED;
- deterministic tokenized shards: BLOCKED;
- post-pack one-pass unique causal-loss ledger: BLOCKED;
- BPE/Unigram fit on a canonical train identity: BLOCKED;
- meaningful ~20M long training: PROHIBITED.

The correct next data step is to compose the exact admitted bytes from these terminal authorities
into one immutable record inventory, bind every record to its source/family/rights identity, then
run the existing dedup and decontamination gates on that exact inventory. Only after those gates
produce a deterministic corpus and unique-loss ledger should training mechanics be requalified on
the primary 20,613,440-parameter MODEL-341 candidate.

## Validation

Checkout-local validation is intentionally stdlib-only and adds no new dedicated workflow, so it
does not create another heavy GitHub Actions fanout during the current runner backlog.

Run:

```bash
python tools/validate_next100_078_research_corpus_v1_convergence.py
python -m unittest tests.test_next100_078_research_corpus_v1_convergence -v
```

LOCAL_FREE only. `training_authorized=false`. `training_executed=false`.
