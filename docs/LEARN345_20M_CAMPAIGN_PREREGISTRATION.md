# LEARN-345 — First Meaningful ~20M Scratch-Pretraining Campaign Preregistration

`SWARM_WORKER_ID: LEARN-345-20M-CAMPAIGN-PREREGISTRATION`

Mode: `LOCAL_FREE`.

Evidence identity: `ad95af791d9f9eea741e7ad812bd61696259df0dcaac911837088265180b10d3`.

## Verdict

`BLOCKED_MISSING_TERMINAL_RESEARCH_CORPUS_V1_PRIMARY_20M_MODELSPEC_AND_OPTIMIZER_TRANSFER`

This worker does **not** execute the long ~20M campaign. At the authority cutoff `2026-08-26T14:49:27Z`, the requested frozen Research Corpus V1 and the requested primary ~20M candidate are not discoverable as terminal authorities, and TRAIN-344 is also not published. The preregistration therefore fails closed rather than inventing any corpus identity, ModelSpec, optimizer recipe, or learned result.

## Exact observed authority state

The latest terminal final-corpus execution remains DATA-301 at exact head `8820ba1b255f6bb95c7db0531fd846078a1aae01`, evidence identity `939065abeefff8aed924415589608ff3fc721fe4b0a57fc200146a4b6a137e81`. It reports `TERMINAL_BLOCKED`, null corpus/shard identities, and `authorized_balanced_no_replay_capacity = 0`.

RESEARCH-313 at exact head `728d95018b362b0c5625751471647083a830bf44`, evidence identity `d8aa1bbd9446c4ca881a97e9218891ee4223d6de60a5ee5534462978b5f2e970`, therefore authorizes **0** unique nonignored causal training positions now. It separately records a project-local planning band of roughly 10M–40M unique positions for a meaningful nominal ~20M campaign. That band is a planning guard, not corpus capacity.

EVAL-303 selection-validation is terminal at head `5e5a1de3b594cee5612e63d3d4c2a70499740ac7`, composite identity `7b97a9ab04469236dc5bc17fc80155cb43430b01c443bb6209fac090557258fd`, with 10 immutable selection-only records: UA 8, EN 2, code 0. Code-aware selection remains unavailable at this authority vector.

EVAL-233 final-test authority is bound only by metadata/provenance. Its payload and outcomes remain unread by this worker.

No discoverable terminal `RESEARCH-339`, `MODEL-341`, or `TRAIN-344` authority exists at this cutoff. Their required identities are therefore null in the machine record.

## Activation gate

Optimizer step 1 is forbidden until all of the following are exact and terminal:

1. Research Corpus V1 is frozen and binds its corpus identity, shard-manifest identity, exact train unique-loss ledger identity, and one-pass count of unique nonignored causal loss positions.
2. That one-pass count is at least 10,000,000 positions. Below this, mechanics/short-science may exist, but this worker's **meaningful ~20M campaign** remains blocked.
3. RESEARCH-339 has selected the primary ~20M geometry and MODEL-341 has mechanically qualified that exact ModelSpec from random initialization.
4. TRAIN-344 has published the exact terminal optimizer-transfer contract.
5. The immutable selection-validation authority to be used for checkpoint selection is bound before training and remains prohibited from training/tokenizer fitting.
6. The final-test authority is sealed and unread before selection lock.

Any parent-authority drift requires a successor LEARN-345 contract identity before execution.

## Exposure contract

The requested first campaign budget is **20,000,000 actual optimized causal targets**.

The runtime-authorized budget is:

`min(20,000,000, terminal_research_corpus_v1_one_pass_unique_nonignored_causal_positions)`.

The campaign is considered activated as a meaningful ~20M campaign only if the realized authorized budget is at least **10,000,000** targets.

The hard no-replay maximum is always the terminal Research Corpus V1 ledger's exact one-pass unique-position count. The 20M requested campaign budget does not increase that maximum and does not convert source bytes into loss positions.

Forbidden capacity inflation includes epochs, replacement sampling, duplicated documents, repeated optimized positions, source aliases, mirror copies, and padding. Padding contributes zero optimized targets.

If the final frozen corpus exposes less than 20M but at least 10M unique targets, the campaign consumes only that exact smaller authorized amount and reports the reduced exposure. If it exposes less than 10M, this preregistered meaningful campaign does not start.

## Exact pre-run schedule materialization

The schedule fractions are frozen now, before any result:

`0%, 10%, 25%, 50%, 75%, 90%, 100%`.

Before optimizer step 1, the immutable planned train trace must be materialized. Each fraction is then mapped to an exact cumulative optimized-target **optimizer-update boundary** using that trace. The resulting integer boundary vector and its SHA-256 are frozen before training.

Boundary 0 is exactly zero. Boundary 100% is exactly the realized authorized campaign budget. No boundary may be moved after observing BPB, gradients, generations, or any final-test information.

At every boundary the runner must:

- publish a D05 checkpoint/recovery generation;
- evaluate the immutable selection-validation set without mutating model/optimizer/RNG state;
- record aggregate BPB plus UA/EN/code/source-family BPB where those strata exist;
- record exact cumulative optimized targets;
- record pre/post clip gradient norms, clip activation, and update/weight L2 ratio for the completed interval;
- reload the retained checkpoint for first-party logits fingerprint and fixed first-party raw Base generation.

A mandatory **fresh-process resume** occurs from the verified 50% generation.

## Checkpoint selection

The selectable checkpoints are the 10%, 25%, 50%, 75%, 90%, and 100% boundaries. Random-init 0% is diagnostic only.

A checkpoint is eligible only if:

- D05 integrity verification passes;
- all required losses/logits are finite;
- evaluation non-mutation proof passes;
- ModelSpec, tokenizer, corpus, train-ledger, train-trace, optimizer-contract, and exposure-counter identities all match this campaign.

Primary selection metric: **immutable selection-validation aggregate BPB, minimized**.

Tie rule: if aggregate BPB differs by at most `1e-9`, select the **lower-exposure** checkpoint.

The chronological final checkpoint is always retained separately from the selected-best checkpoint. Final-test evidence can never choose or re-choose a checkpoint.

## Recovery generations

The immutable D05 generations are logically named:

`g000`, `g010`, `g025`, `g050`, `g075`, `g090`, `g100`.

Every generation binds the campaign contract, exact ModelSpec, tokenizer, corpus, train-ledger, train-trace, optimizer contract, cumulative optimized-target count, Trainer state, optimizer state, RNG state, and exposure-guard state. Existing generations are never overwritten. `current` is updated atomically only after the new generation verifies.

Planned resume is legal only from an exact verified generation. `g050` must be loaded in a separate fresh Python process before the second half begins.

An unexpected failure after consuming positions but before the next verified generation **must not** be repaired by replaying those positions. That run becomes ineligible for scientific checkpoint selection. A retry, if separately authorized, is a new scratch run identity from random initialization.

## Final-test firewall

EVAL-233 final-test payload and outcomes remain sealed through all training, scheduled evaluation, checkpoint selection, and selection-lock publication.

Selection lock must bind:

- this campaign-contract identity;
- exact selected-checkpoint SHA-256;
- exact chronological-final-checkpoint SHA-256;
- complete selection-trajectory SHA-256;
- an attestation that final-test payload/outcomes were unread.

Only after this lock may a separate evaluation worker score the locked selected checkpoint under the final-test authority. That result is report-only: it cannot trigger checkpoint reselection, hyperparameter changes, extra training, or a second look at another checkpoint.

## Truth boundary

- Frozen Research Corpus V1 consumed by this worker: **no**.
- Primary ~20M ModelSpec consumed by this worker: **no**.
- TRAIN-344 optimizer contract consumed by this worker: **no**.
- Long campaign executed: **no**.
- Optimizer updates: **0**.
- Final-test outcomes read: **no**.
- Learned result claimed: **none**.

This is the exact fail-closed campaign contract that a successor may activate only after all required terminal parent identities exist.
