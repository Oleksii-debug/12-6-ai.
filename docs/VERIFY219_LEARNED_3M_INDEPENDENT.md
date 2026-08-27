# VERIFY-219 — Independent learned 3M admission

## Purpose

VERIFY-219 closes the missing independent scientific-admission authority for the exact terminal LEARN-191 learned 3M Base artifact. It is an artifact consumer only: no retraining, optimizer update, checkpoint write, foreign/pretrained weights, instruction/alignment work, external LLM, or paid compute.

## Immutable producer binding

- Producer worker: `LEARN-191-REAL-3M`
- Producer git SHA: `a75920cef8bde37a8c590e34095be83c97b75f1d`
- Artifact ID: `9597788382`
- Artifact ZIP SHA-256: `f57bf36113a68fffd4bfcf877bf08762393479b9c09e6fd0fd613fbb91f044ee`
- ModelSpec SHA-256: `462c85da80a3c0d7d6a4f1a570b87d208b1847d8a57b12a4d9be7e36846b65dc`
- Parameter count: `3,213,120`
- DATA-25 identity: `422f545da93526ceba2c0ff738e0b9deac65f35bfb31d87d2aab068cda091fc8`
- M150 common evaluation identity: `7189e6df053574beb686727c94e684cdbaf08a34ef33aa953eff7cdae0320113`

## Independent checks

The verifier downloads only the immutable LEARN-191 artifact and fails closed if its archive digest differs. It then:

1. validates the producer report, run manifest, truth record, phase1, resume, and final fresh-load self-hashes;
2. rebuilds DATA-25 from repository sources and re-derives the canonical byte tokenizer and M150 common-evaluation identity;
3. reconstructs the exact random-init 3M model from producer ModelSpec, InitSpec, and seed;
4. independently reruns the preregistered LEARN-191 selection-validation evaluator on random init and every retained trained checkpoint;
5. verifies the retained 16,632 / 65,772 / 131,292 target checkpoint identities, exact producer SHA, ModelSpec, tokenizer, DATA-25, run identity, step, and optimized-token counters;
6. independently recomputes which retained target is best by the preregistered selection rule and requires agreement with the producer record;
7. runs the full M150 common DATA-25 held-out evaluation on random init and final learned 3M, requiring final learned 3M to improve over random init;
8. generates a new independent first-party final logits fingerprint twice and requires reproducibility;
9. reruns the final greedy Base generation and requires byte/ID-level agreement with both producer report and fresh-load proof;
10. hashes every retained checkpoint tree before and after verification and requires byte identity;
11. verifies the mandatory fresh-process midpoint resume boundary.

## Truth boundary

A terminal VERIFY-219 result admits only the exact LEARN-191 3M artifact under the exact DATA-25 research contract. DATA-25 is project-authored and is not claimed to be an external-real or representative production pretraining corpus. The authority does not establish instruction following, factuality, safety/alignment, production readiness, or a learned 20M model.

The 3M and 10M producer runs use different optimized-token budgets. Independent admission of both does not make them a matched-exposure direct scaling comparison.
