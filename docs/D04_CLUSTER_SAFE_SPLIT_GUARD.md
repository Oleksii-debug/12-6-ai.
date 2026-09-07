# D04 independence-cluster split guard

## Why this gate exists

Global dedup and split isolation are different questions. A global dedup pass can keep
multiple capacity-bearing sibling files from one upstream origin because they are not
byte/copy duplicates. Those files may legitimately contribute source capacity, but
putting siblings from the same origin into different train/held-out splits would create
lineage leakage.

NEXT100-065F makes this distinction concrete: its terminal survivor graph contains
many more retained source objects than effective independence/origin clusters. D04 must
therefore consume the upstream `independence_cluster_identity_sha256` projection rather
than treating per-file survivor identity as split independence.

## Contract

`cluster_safe_split_guard.py` consumes two externally identified, text-free objects:

1. a **terminal post-decontamination handoff** containing only retained record
   identities, payload hashes/bytes, family/modality, exact evaluation-reservation
   purpose (`reserved_split`) and `independence_cluster_identity_sha256`; and
2. a **complete split manifest** assigning every retained record to exactly one of
   `train`, `selection`, or `final_test`.

The caller must supply the expected terminal decontamination authority identity, the
expected handoff identity and the expected split-manifest identity independently. A
self-consistent substituted object is not accepted as authority.

The guard fails closed unless:

- reserved-evaluation decontamination is terminal and complete;
- durable inputs contain no raw text and claim no final-test payload read;
- every retained record is assigned exactly once;
- every source maps to exactly one independence cluster;
- every independence cluster is wholly inside one split;
- every reserved record carries an exact `selection` or `final_test` purpose and is
  assigned to exactly that split;
- non-reserved records carry no held-out purpose and are assigned only to train; and
- neither held-out quota repair nor selection/final-test role swapping is possible.

The proof is hash-only and records split counts/bytes plus a deterministic cluster-to-
split projection identity. It explicitly records that independence-cluster leakage,
training use of reserved records, and reservation-purpose drift are all false.

## Scientific boundary

This guard does **not** choose the split, declare balance adequate, fit a tokenizer,
pack sequences, create causal-loss positions, or authorize model training. It validates
a split only after D03 provides terminal decontamination truth. `authorized_training_exposure`
remains exactly `0` here.

The next D04 stage after a terminal clean split is deterministic tokenization/sharding/
packing in two independent clean builds, followed by the exact unique non-ignored
causal-loss ledger and resume-safe next-exposure binding.
