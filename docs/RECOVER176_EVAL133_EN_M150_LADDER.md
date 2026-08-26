# EVAL-216 / RECOVER-176 — EVAL-133 English on M150 learned Base ladder

RECOVER-176 is a convergence recovery. It does not change the accepted EVAL-133 examples, reservation, scorer, or original tests.

## Exact failed regression

Terminal workflow run `32938876543` passed bootstrap, frozen-suite tests, M150 reconstruction/training/verification, and M150 ladder finalization. It failed only when the recovery bridge began checkpoint evaluation. The bridge read `scale_report["model"]["model_spec_sha256"]`, but the M150 scale-report contract stores that identity as `scale_report["model"]["spec_sha256"]`. The result was `KeyError: 'model_spec_sha256'` before any EVAL-133 scoring completed.

This is a RECOVER-176 consumer-schema defect, not an EVAL-133 semantic/scoring defect and not an environment/bootstrap defect. A regression test now locks the actual M150 scale-report field.

## Producer authority

EVAL-216 consumes the terminal frozen M150 retained-evidence incumbent rather than retraining another ladder:

- source `5838cd16869dcfcf762368d8673eddf52d51b7e3`;
- workflow run `32937411703`, terminal `SUCCESS`;
- artifact `9595677772`, `milestone150-learned-base-ladder-v1`;
- artifact digest `sha256:c00b7e9006320f8916c739a3311e8cc47ad0d0b16957f8ebd7d19233fd9f1c71`;
- ladder report `1f8350bed574a7b78778f0ebb7854ca5311173006820ec27110122f8965c9a5a`;
- DATA-25 identity `422f545da93526ceba2c0ff738e0b9deac65f35bfb31d87d2aab068cda091fc8`;
- common M150 evaluation identity `7189e6df053574beb686727c94e684cdbaf08a34ef33aa953eff7cdae0320113`;
- canonical `s0-byte-v1` tokenizer.

The 100K/500K/1M random-init states are deterministically reconstructed with the producer seed and must match the exact `phase1.json` state hashes. Learned evaluations use the exact retained best checkpoints. No 10M numerical result is emitted until a terminal learned 10M checkpoint exists.

## Immutable EVAL-133 authority

Suite ID remains `eval133-en-raw-v1`, version `1.0.0`, SHA-256 `f9e713ff336e6189f7aa0ddbb21303431ab2041b6700ed38243eaf65865805cb`. Reservation SHA-256 remains `850e0c34fd6ab35d0829b3f78ff5e81fbcb8c1ee900f3e7f1b967ea23a8f2e40`. The suite, reservation index, cloze scorer, EVAL-133 evaluator, reservation implementation, and original test file remain guarded by their original Git blob identities.

The deterministic DATA-25 corpus is rebuilt twice and must reproduce the exact producer corpus identity. The immutable EVAL-133 reservation/decontamination logic then scans DATA-25 train and validation plus retained legacy S0 packaged train/validation. Any collision fails closed.

## Evaluation proof

Each scale is evaluated on the identical 32-pair, eight-phenomenon suite. Learned checkpoint identity is bound to producer SHA, ModelSpec, tokenizer config/vocab, DATA-25, run manifest, checkpoint ID, and seed. The first-party checkpoint backend is checked for finite logits, and scoring uses the unchanged first-party EVAL-133 conditional-likelihood path.

Every scoring pass records exact model-state and Trainer-state hashes before/after and requires optimizer-step delta `0` and optimized-token delta `0`. Output includes pair accuracy, raw likelihood margin, token/UTF-8-byte normalized margins, preferred/dispreferred conditional BPB, context BPB, per-phenomenon results, learned-vs-random deltas, and learned scale trend.

Recovery report schema is `12-6.recover176-eval133-learned-base-ladder.v2` because producer binding and machine evidence changed. EVAL-133 itself remains v1 unchanged.

Truth boundary: LOCAL_FREE only; no foreign pretrained weights, SFT, RLHF, DPO, paid compute, instruction-following claim, broad English-proficiency claim, intelligence claim, alignment claim, or production-readiness claim.
