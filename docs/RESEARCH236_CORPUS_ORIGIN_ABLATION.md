# RESEARCH-236 Corpus-Origin Ablation

Worker: `RESEARCH-236-CORPUS-ORIGIN-ABLATION`

## Decision at this source cut

`BLOCKED_MISSING_OR_INVALID_AUTHORITY` for numerical external-real results.

The live repository audit on 2026-08-26 found a terminal DATA-25 learned-control authority but no published branch or pull request for `DATA-230-CORPUS-V03-EXTERNAL-REAL` or `LEARN-234-EXTERNAL-REAL-500K` at the audit cut. `DATA-183` is an older mixed-origin candidate and is explicitly not substituted for DATA-230. Therefore this worker preregisters and tests the ablation contract but does not fabricate an external-real BPB result.

Authority base: `INTEGRATE-222` head `6afaf5889f9898037b53e8b0bc2b731d77782111`.

Known DATA-25 control identity:

- corpus: `422f545da93526ceba2c0ff738e0b9deac65f35bfb31d87d2aab068cda091fc8`;
- tokenizer: canonical `s0-byte-v1`, vocabulary 256;
- evaluation identity: `7189e6df053574beb686727c94e684cdbaf08a34ef33aa953eff7cdae0320113`;
- ~500K: 467,808 parameters, ModelSpec `208ac8ca113388e76f280d0154cae815785bee7705546f4d854d9447b9dd1f4a`;
- ~1M: 1,037,696 parameters, ModelSpec `ff3cee542a1f75bb4e1eff8d7d24d72533af8f4f3d82bd064fb1cbfeba8c8d07`.

## Frozen experiment

The mandatory experiment is ~500K. ~1M is the same preregistered comparison if the terminal external-real corpus and LOCAL_FREE execution envelope make it available without changing the data composition or exposure budget.

Every corpus arm uses the same ModelSpec at a given scale, tokenizer, AdamW recipe, sequence length 128, batch 8, deterministic FP32 execution, and paired seeds 1337 / 1338 / 7331. The common boundary is 131,938 actual optimized loss targets.

The exposure ledger counts actual unique non-ignored causal source/loss-token positions. Padding is never exposure. Repeated source token positions are forbidden. A larger padded tensor count, optimizer-step count, or nominal sequence-length product cannot be used to claim matched data exposure.

The head-to-head state is always the fixed final state at the matched token boundary. Neither corpus is allowed to gain a checkpoint-selection advantage by choosing a different validation minimum.

## Evaluation design

Direct comparisons are made only when both models are scored on identical immutable bytes.

Both training arms are scored on DATA-25 selection validation, both on DATA-230 selection validation, and both on the common immutable real holdout from EVAL-233. This yields two explicit cross-corpus transfer directions instead of comparing one corpus's own validation BPB against another corpus's different validation bytes.

The common real holdout reports aggregate BPB plus UA, EN, code and source-family heldouts. Source-family keys must be identical for the paired arms. The report retains mean BPB by family, worst and best family, the worst-minus-best sensitivity spread, and paired external-trained minus DATA25-trained deltas for every family.

EVAL-233 selection-validation and final-test purposes remain physically and semantically separate. Final-test material cannot select the tokenizer, model, hyperparameters, checkpoint, source documents, filter policy or corpus winner.

## Metrics and interpretation

Training BPB is retained for each corpus but is labeled corpus-conditional; it is not treated as a same-bytes head-to-head metric.

Held-out comparisons report paired seed deltas with the convention `external-real-trained minus DATA25-trained`; negative BPB deltas favor external-real training on that particular fixed evaluation set. With three paired seeds, the analyzer enumerates the exact 3^3 empirical paired bootstrap resamples and reports p05 / p50 / p95 plus direction consistency.

Cross-corpus transfer is the two-by-two matrix of training corpus versus DATA-25 and external-real selection validation. The common real holdout is the primary origin-neutral quality surface.

Memorization uses fixed hash-sampled training probes from each corpus, with no canary injection and no text emitted in public reports. The proxy is own-training-probe BPB minus own-selection BPB. It is a bounded exposure-specific-fit diagnostic, not a privacy-leakage claim.

Generalization gap is common-real-holdout BPB minus corpus-conditional training BPB. Because the training distributions differ, it is interpreted as a within-arm generalization diagnostic rather than a universal corpus score.

## Fail-closed prerequisites

Numerical execution is forbidden until:

1. DATA-230 is terminal, deterministic across two clean builds, contains explicit `EXTERNAL_REAL` origin, provides at least 131,938 non-repeated training loss tokens, and did not inflate supply with artificial corpus repetition.
2. EVAL-233 is terminal and retains distinct selection-validation and final-test purposes with no final-test exposure to selection.
3. The exact DATA-230 and EVAL-233 identities are recorded before optimizer step 1.
4. Cross-corpus validation and training-probe records are proven uncontaminated for the comparison in which they are used.
5. Every arm begins from a fresh random initialization generated from its paired seed; no foreign or pretrained weights are accepted.

If any prerequisite is missing, the only valid worker result is a blocker report. An older external corpus candidate is not an acceptable substitute.

## Claim boundary

External origin is not a quality label. A result may favor DATA-25 on some or all metrics, favor external-real on some or all metrics, or be mixed by language/source family. RESEARCH-236 reports those outcomes without converting corpus provenance into a superiority claim.

LOCAL_FREE only. No SFT, RLHF, DPO, foreign pretrained weights, paid compute, broad intelligence claim, or stage-promotion claim.
