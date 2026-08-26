# POSTBASE-359 Teacher Factory Convergence

SWARM_WORKER_ID: `POSTBASE-359-TEACHER-FACTORY-CONVERGENCE`

Execution profile: `LOCAL_FREE`.

## Scope

This worker finalizes the existing provider-neutral POSTBASE-259 teacher/student data-factory candidate rather than creating a second factory. It is stacked directly on POSTBASE-259 exact head `5ec7bc917b506273751f0efa8d1048431bcafc8d`.

No real external teacher is called. No provider SDK, endpoint, credential, foreign model, paid compute, optimizer step, Base-weight mutation, or canonical Base training path is introduced.

## Required flow

The executable accepted-data path is now provenance-bound in this order:

1. teacher proposal;
2. independent critic review;
3. independent deterministic verification;
4. deterministic accept/reject decision;
5. curator revalidation;
6. SHA-linked versioned `POSTBASE/EXPERIMENTAL` dataset snapshot.

The critic provenance parents include the student answer and every teacher proposal. Each deterministic verification provenance record includes both the exact proposal and the completed critic review. The judge provenance includes the critic review and every deterministic verification result. The curator rejects a forged chain that skips any of those boundaries.

## Independence

Student, every teacher, critic, deterministic verifier, judge, and curator adapter IDs must be pairwise distinct. This closes the POSTBASE-259 gap where the critic could share a teacher identity or the verifier could share the critic identity.

Teacher confidence is metadata only. It has no acceptance authority.

## Deterministic verification

A proposal is eligible only if at least one deterministic verifier returns `SUPPORTED` and no deterministic verifier result for that proposal is `CONTRADICTED`.

`CONTRADICTED` is a hard veto. The curator independently repeats the gate and rejects forged judge acceptance.

The retained confidently-wrong fixture uses teacher confidence `1.0`, proposes the wrong exact answer, is deterministically contradicted, receives `REJECT`, and produces zero curated records.

## Versioned post-Base dataset

The previous candidate exposed a dataset version string but the manifest identity changed as records were appended under that same version. POSTBASE-359 replaces that weak versioning boundary with immutable SHA-linked revisions.

Each curator starts from revision 0. Every accepted record binds the exact previous manifest SHA-256. The next manifest binds that parent identity and the complete ordered record-ID set. The curator retains the full manifest revision history. A successor dataset version may bind the terminal manifest of its predecessor as its parent.

All records and manifests are hard-coded to:

- `classification = POSTBASE/EXPERIMENTAL`;
- `base_corpus_evidence = false`;
- `canonical_base_training_eligible = false`;
- `training_use = POSTBASE_SYNTHETIC_EXPERIMENTAL_ONLY`.

`as_base_corpus_evidence()` still fails closed.

## Adversarial gates

Focused convergence tests require:

- confident wrong teacher rejection despite confidence 1.0;
- teacher/critic identity separation;
- critic/verifier identity separation;
- exact proposal -> critique -> verification -> decision provenance ordering;
- rejection of a forged verification that skips critique;
- SHA-linked dataset revision history;
- successor-version parent binding;
- canonical Base eligibility remaining false in every record and manifest;
- mutation of a returned manifest copy not rewriting retained history.

The inherited POSTBASE-259 tests remain part of the convergence gate.

## Truth boundary

This is post-Base synthetic-data infrastructure only. It does not establish teacher quality on open-ended tasks, does not authorize real teacher acquisition, does not create communication training data from foreign model output, and does not authorize SFT or any other model-weight update.
