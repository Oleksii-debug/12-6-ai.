# MILESTONE-361 — Learned Base Ladder V5

Worker: `NEXT100-018-LEARNED-LADDER-V5`

## Final verdict

`BLOCKED_MISSING_INDEPENDENT_VERIFY_219_AND_VERIFY_218`

This successor to MILESTONE-281 does not admit the 3M or 10M learned runs. The mandatory second live GitHub check was completed across PR, branch, and commit search; both required independent scientific verifier authorities remain absent:

- 3M / LEARN-191 requires terminal `VERIFY-219`.
- 10M / LEARN-217 requires terminal `VERIFY-218-LEARNED-10M-INDEPENDENT` with state `VERIFIED_LEARNED_10M`.

Producer evidence, producer-side fresh verification, checkpoint integrity, and independent runtime/inference corroboration do not substitute for those independent scientific verifier authorities.

## Comparison boundary retained

The directly comparable ladder remains only 100K / 500K / 1M at exactly 948,504 optimized targets per rung. Its retained direct ranking is:

`1M > 500K > 100K`

The 3M run used 131,938 optimized targets. The 10M run used 2,000,060 optimized targets. These are different-budget evidence. They must not be ranked directly against the equal-budget ladder or against each other as if exposure were matched.

## DATA-25 truth boundary retained

DATA-25 remains project-authored evidence with identity `422f545da93526ceba2c0ff738e0b9deac65f35bfb31d87d2aab068cda091fc8`. It is not external-real evidence and is not claimed to be a representative real corpus or final corpus freeze.

## Execution boundary

No retraining, optimizer work, paid compute, foreign/pretrained weights, SFT, RLHF, or DPO was performed. `LOCAL_FREE` only.

Machine authority: `evidence/milestone361/learned-base-ladder-v5.json`.
