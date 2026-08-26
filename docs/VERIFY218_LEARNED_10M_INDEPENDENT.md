# VERIFY-218 — Independent learned 10M admission

## Purpose

VERIFY-218 closes the independent scientific-admission gap for the exact terminal LEARN-217 learned 10M Base artifact. It is verification-only: no optimizer update, no retraining, no foreign/pretrained weights, no instruction/alignment work, no external LLM, and no paid compute.

## Immutable producer binding

- Producer worker: `LEARN-217-TERMINAL-10M-BASE`
- Producer git SHA: `c02c8aa38e691521ae2ab6a4ff3ea1d643efd6ef`
- Artifact ID: `9602650341`
- Artifact ZIP SHA-256: `8631e90417e40365b3fc0d6bc98ee6adda5a4ed24530e675d9a91c93219537ee`
- ModelSpec SHA-256: `61caa5469123e23b9b72fc2024140bfca84c4c480dcb0a7e712ba800a4f22998`
- Parameter count: `10,000,640`
- DATA-25 identity: `422f545da93526ceba2c0ff738e0b9deac65f35bfb31d87d2aab068cda091fc8`
- Common ladder evaluation identity: `7189e6df053574beb686727c94e684cdbaf08a34ef33aa953eff7cdae0320113`

## Independent checks

The verifier downloads only the exact immutable LEARN-217 artifact and fails closed on digest mismatch. It then:

1. validates producer report, fresh-verification, phase1, and run-manifest self-hashes;
2. rebuilds DATA-25 from repository sources and re-derives the canonical byte-tokenizer/common-evaluation identity;
3. reconstructs the exact random-init 10M baseline from the producer ModelSpec, InitSpec, and seed, then reruns the common held-out evaluation;
4. independently verifies and loads retained `best` and `final` checkpoints;
5. reruns the canonical common UA/EN/code held-out evaluation and requires exact metric agreement within `1e-7`;
6. reruns first-party logits fingerprints twice for reproducibility and compares them with retained producer evidence;
7. reruns greedy Base generation and compares it with retained producer evidence;
8. hashes checkpoint trees before and after verification and requires byte identity;
9. verifies retained phase-boundary/current recovery checkpoints and monotonic step/token counters;
10. emits `VERIFIED_LEARNED_10M` only if both learned best and final beat the independently reconstructed random-init baseline.

## Truth boundary

A terminal VERIFY-218 result is admission evidence only for the exact LEARN-217 artifact under the exact DATA-25/common ladder evaluation contract. DATA-25 is project-authored and is not claimed to be an external-real or representative production pretraining corpus. VERIFY-218 does not establish instruction following, broad reasoning quality, factuality, safety/alignment, production readiness, or a learned 20M model.

The next model-scale step remains a separate learned-20M campaign using the qualified 20,613,440-parameter MODEL-341 geometry after its data and checkpoint gates are terminal.
