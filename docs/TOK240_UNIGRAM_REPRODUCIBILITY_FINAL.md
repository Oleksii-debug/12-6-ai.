# TOK-240 Unigram reproducibility final

## Decision

`INELIGIBLE_FOR_RESEARCH_SELECTION`

The incumbent Hugging Face Tokenizers Unigram path is not eligible for tokenizer-family research selection under the current project reproducibility contract. Model comparisons stop here; TOK-241 must not rank this Unigram path against BPE unless a later independently reviewed runtime changes the determinism facts.

## Exact incumbent environment

TOK-240 retains the existing algorithm and runtime rather than introducing a replacement tokenizer implementation:

- CPython 3.11.16;
- `tokenizers==0.23.1`;
- tokenizers wheel SHA-256 `5075b405006415ea148a992d093699c66eb01952bf59f4d5727089a98bda45a4`;
- ENV-151 `linux-x86_64-tokenizer-experiment` capability bootstrap;
- immutable S0 train SHA-256 `61d24b7138df56527d201cea405d11c9f607684b4a9593dfa20c599cc2ee6998`;
- Unigram training-manifest SHA-256 `bbe6fc282af46aa0d62c4405d3c2dc92da76b03bb46c1dfb9c9f2f4d738dcca4`.

The final workflow trains three fresh child-process artifacts from the identical immutable input. It fixes manifest record order, Python random seed, `PYTHONHASHSEED=0`, disables tokenizer parallelism, and forces Rayon/OMP/MKL to one thread. Those are bounded execution controls only; the Unigram algorithm is unchanged.

## Reconstructed failure

The earlier exact-runtime real experiment on source `e925109473822bcd11ceef71f98f1441a6816f62`, workflow `32861353159`, job `97845863963`, already trained two independent Unigram artifacts from the same 10 records and the same training-manifest identity.

Artifact A:

- tokenizer JSON SHA-256 `ce7160e219e3fe27b544c9b1d3375e2166a5d1b8c65a2261eac188886195682f`;
- ordered token-ID vocabulary SHA-256 `a31147ec8ca30f932205f62b0a0a521821028e8a94b130d103233cdbb2093fd4`;
- config SHA-256 `7676193855768d67c630ce59ed98fed7729d2719d94339b46ce1a0ec8a41046e`.

Artifact B:

- tokenizer JSON SHA-256 `c17e93cd3d42a0b234a27ab20780e1e1a07eff472a764b63fd20a60b68423ca0`;
- ordered token-ID vocabulary SHA-256 `8cd6b5f273f2bb0f7b1b14bd017b49c933b7679d2571659e77e8abfbc71904a1`;
- config SHA-256 `afb700ff33e444466e151731f706169fd34a73259b125475e603433952a3e8f7`.

Both had vocabulary size 497 and special-token metadata `<unk>: 0`, but held-out token ID sequences differed. Strict decode round-trip still passed and unknown-token count was zero. Therefore this is not harmless JSON formatting drift: token-ID semantics changed.

The old dedicated root-cause audit then stopped with `serialization itself drifted; root-cause classification is ambiguous`. That assertion was too strong. Serialize→reload→serialize byte normalization is a separate artifact-format behavior. It cannot explain the already observed ordered vocabulary and probe-token-ID drift across independently trained models. TOK-240 records serialization behavior separately and uses the project's semantic identity contract for the eligibility decision.

## Cause classification

Library randomness is the primary supported mechanism. The pinned upstream Unigram trainer uses `AHashMap` and `AHashSet`. Seed-piece construction consumes hash-map iteration and sorts equal-frequency entries only by frequency, so tied entries retain hash-derived order before EM training. Required-character handling also consumes hash-set iteration. The exact Python binding exposes no public randomness seed for `UnigramTrainer`.

Input ordering is not a sufficient fix because drift occurs with identical manifest order. Threading is not a sufficient fix because drift persists with tokenizer parallelism disabled and one Rayon thread. Floating-point reductions may amplify a prior ordering difference, but they are not required to establish the failure. Artifact serialization is secondary, because independent models already differ in ordered token-ID vocabulary and encodings. Special-token metadata is not the cause because `<unk>: 0` remains stable.

## Semantic canonicalization boundary

TOK-240 permits only canonicalization that removes representation-only JSON ordering differences. It does not canonicalize away ordered vocabulary IDs, Unigram scores, model semantics, special-token behavior, pre-tokenizer/decoder semantics, or probe encodings. Reordering token IDs would change embedding and LM-head meaning and therefore cannot qualify as semantic identity.

The current `ordered_vocab_sha256()` contract explicitly hashes complete token→ID semantics. Consequently the observed vocabulary drift is sufficient to reject canonicalized semantic identity even if serialized JSON were normalized.

## Post-decision scope

Because Unigram remains ineligible, TOK-240 does not run tokenizer-family model comparisons, fertility ranking, or tokenizer speed ranking. Those checks are only relevant if reproducibility is repaired. The only retained mechanics observations are strict round-trip and unknown-token behavior needed to distinguish semantic drift from gross tokenizer corruption.

No foreign weights, model training, SFT, RLHF, DPO, paid compute, new tokenizer algorithm, or tokenizer-family quality winner is introduced by TOK-240.
