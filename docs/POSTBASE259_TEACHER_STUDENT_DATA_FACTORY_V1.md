# POSTBASE-259 Teacher–Student Data Factory V1

`SWARM_WORKER_ID: POSTBASE-259-TEACHER-STUDENT-DATA-FACTORY-V1`

This worker adds a control-plane architecture for future knowledge distillation. It does not call an external teacher, does not contain provider credentials or endpoints, and does not train canonical Base on synthetic output.

## Roles and flow

Every candidate follows the same auditable chain:

1. task intake;
2. student answer;
3. one or more teacher critiques/proposals;
4. critic review;
5. independent deterministic verification of every teacher proposal;
6. deterministic conflict resolution by a judge;
7. curator accept/reject;
8. versioned synthetic dataset record.

The six pluggable roles are `student`, `teacher`, `critic`, `deterministic_verifier`, `judge`, and `dataset_curator`. `TeacherAdapter` is the future API boundary. Its request/response types are provider-neutral and contain no key, provider, endpoint, network, or SDK dependency.

## Fail-closed acceptance

Teacher confidence is never evidence. A proposal can enter an accepted dataset only when the judge selects it and the curator independently rechecks that at least one deterministic verifier marked that exact proposal `SUPPORTED`, no deterministic verifier marked it `CONTRADICTED`, and the verifier identity is distinct from every teacher identity.

If no proposal is supported, reject. If multiple supported proposals still disagree, reject. If teachers disagree but deterministic verification supports one and contradicts the other, only the supported proposal may be selected. The curator repeats the safety checks after the judge, so a faulty or forged judge decision cannot bypass verifier contradiction.

## Provenance

Every task/contribution carries deterministic SHA-256 provenance binding its role, actor ID, source kind/reference, parent contribution IDs, and payload hash. Accepted records retain the complete task, student, teacher, critic, verifier, judge, and curator chain. Identical local inputs produce identical record identities.

## Dataset boundary

All accepted records are hard-coded as:

- classification: `POSTBASE/EXPERIMENTAL`;
- `base_corpus_evidence=false`;
- `canonical_base_training_eligible=false`;
- training use: `POSTBASE_SYNTHETIC_EXPERIMENTAL_ONLY`.

`VersionedDatasetCurator.as_base_corpus_evidence()` fails closed with `BaseCorpusBoundaryError`. This worker provides no path that can relabel synthetic teacher data as Base corpus evidence.

## Local fixtures and proof

The repository fixtures include a correct teacher, a confidently wrong teacher, and two disagreeing teachers. The deterministic exact-match verifier proves the wrong teacher is rejected despite confidence 1.0, while a disagreement is resolved only when deterministic verification supports one proposal and contradicts the other.

The dedicated workflow uses the existing universal LOCAL_FREE bootstrap, runs focused tests and Ruff, then emits a compact proof report. No network teacher call, paid compute, model training, or synthetic-to-Base promotion occurs.
