# NEXT100-090 Synthetic Teacher Factory Red-Team

SWARM_WORKER_ID: `NEXT100-090-TEACHER-FACTORY-REDTEAM`

Target: POSTBASE-259/359 synthetic teacher/student data factory.

Execution profile: `LOCAL_FREE`. No external teacher API, provider SDK, credential, network-backed teacher, paid model, optimizer step, Base-weight mutation, or canonical Base training path is used.

## Red-team result contract

Every unsafe proposal or evidence mutation must fail closed before an unsafe synthetic record can be returned as accepted training data. One objectively correct, deterministically verified local arithmetic proposal is retained as the positive control.

Attacks covered:

1. confidence `1.0` with an objectively wrong answer -> `CONTRADICTED`, rejected;
2. plausible claim with no registered deterministic evidence -> `INCONCLUSIVE`, rejected;
3. critic/verifier identity reuse -> factory construction rejected;
4. forged verifier subject binding -> factory error before decision;
5. post-verification evidence mutation -> curator integrity rejection;
6. stale evidence revision -> factory error before decision;
7. contradictory evidence -> hard veto at curation even if an acceptance decision exists;
8. dataset-parent forgery -> bare non-genesis SHA and content/hash mismatch rejected;
9. curator attempt to mark output canonical Base eligible -> factory postcondition rejection.

## Hardening introduced

Deterministic verifiers now expose an immutable SHA-256 `evidence_revision`. Each verification response also carries a SHA-256 binding to the exact task, prompt, student answer, teacher proposal, proposal answer, and completed critic review. The factory rejects a response whose revision is stale relative to the active verifier or whose subject binding is forged.

The curator revalidates payload hashes, provenance identities, contribution identities, role/actor correspondence, parent ordering, exact verification subject bindings, judge support references, and cross-task consistency before admission. This prevents a previously valid verification object from being modified and reused without detection.

Non-genesis dataset lineage can no longer be established from a caller-supplied digest alone. A successor curator requires the complete parent manifest and recomputes its SHA-256 while enforcing the POSTBASE classification and Base-training firewall.

The factory now independently validates every curator-returned record. A custom curator cannot make synthetic output `base_corpus_evidence=true`, `canonical_base_training_eligible=true`, or escape `POSTBASE_SYNTHETIC_EXPERIMENTAL_ONLY` and still return an accepted result.

## Positive control

The local task `2 + 2` with proposal `4`, exact-match verifier evidence, distinct role identities, current evidence revision, exact subject binding, deterministic acceptance, and versioned POSTBASE curation remains accepted.

## Scope boundary

This red-team proves mechanics and fail-closed boundaries only. It makes no claim about open-ended teacher quality, model intelligence, communication quality, or suitability of foreign-model outputs. It does not authorize real teacher acquisition or any Base/SFT training operation.
