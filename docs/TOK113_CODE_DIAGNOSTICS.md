# TOK-113 Code Tokenization Diagnostics — 2026-08-26

Status: **DIAGNOSTICS_COMPLETE / PARAMETER-MATCHED TRAINING PROBES BLOCKED**.

## Truth boundary

The live repository is `Oleksii-debug/12-6-ai.` (trailing period). The milestone control remains `s0-byte-v1` with vocabulary 256. TOK-37 supplies reproducible serialized ByteLevel BPE candidates. Its 512/768/1024 requested-vocabulary artifacts all saturate at actual vocabulary 437 with identical vocabulary/merge semantics, so no evidence exists for a real capacity increase above 437 on that tokenizer-training fixture.

The retained real-code samples total **4,998 bytes** across two project families, but DATA-23 leaves their rights state at `REVIEW_REQUIRED`, its intake workflow failed and produced no artifact, and later external-source registries checked remain empty. These bytes are therefore used **only for diagnostics**, never tokenizer or model training. Only Python is represented in the retained real-code pilot, so the requested several-programming-language diagnostic breadth cannot be truthfully claimed.

Frozen diagnostic identity: `d23861ed8a90dce2cfc7823a755f04812db2612d2f60a8268b77c5cce7ec58d9`.

## Artifact fidelity

The local read-only evaluator was parity-gated against TOK-37's published held-out token counts. It reproduced every frozen TOK-37 probe count exactly for requested vocabularies 257, 320, 384 and 512 before any TOK-113 metric was accepted. Hugging Face `tokenizers==0.23.1` remains the authoritative tokenizer runtime; the local reader is evidence extraction only, not a new tokenizer implementation.

D04 Unigram actual-vocab 497 is rejected from this comparison because identical repeated builds drifted in `config_sha256`, `tokenizer_json_sha256`, and `vocab_sha256`.

## Real-code compression and fragmentation

| tokenizer | tokens/source-byte | tokens/non-WS char | token reduction vs byte | mean tokens/identifier | mean tokens/frozen operator | mean tokens/Unicode identifier | learned entries activated by 4,998B real code |
|---|---:|---:|---:|---:|---:|---:|---:|
| byte256 | 1.0000 | 1.3254 | 0.0% | 6.93 | 1.95 | 11.88 | — |
| bpe-r257-a257 | 1.0000 | 1.3254 | 0.0% | 6.93 | 1.95 | 11.88 | — |
| bpe-r320-a320 | 0.8093 | 1.0727 | 19.1% | 5.73 | 1.95 | 8.88 | 32/63 (50.8%) |
| bpe-r384-a384 | 0.7141 | 0.9464 | 28.6% | 5.28 | 1.95 | 8.25 | 62/127 (48.8%) |
| bpe-r512-a437 | 0.6833 | 0.9056 | 31.7% | 5.15 | 1.95 | 8.00 | 73/180 (40.6%) |

Interpretation:

- BPE-320 removes 19.1% of token traffic versus byte; BPE-384 removes 28.6%; BPE-437 removes 31.7%. The incremental gain from 384 to 437 is only about 4.3% of BPE-384 token traffic.
- Identifier fragmentation improves, but modestly: mean 6.93 byte tokens per real identifier occurrence becomes 5.73 / 5.28 / 5.15 at 320 / 384 / 437. The split rate falls only at 384 (94.98% -> 86.87%) and does not improve further at 437.
- Operator handling does **not** improve on the frozen explicit operator set: mean remains 1.95 tokens/operator for byte, 320, 384 and 437. The extra learned capacity is not buying operator atomicity here.
- Layout compression arrives early. Four spaces are 4 byte tokens but 2 tokens at BPE-320 and remain 2 at 384/437; LF+four-spaces is 5 byte tokens versus 2 BPE tokens at all three learned sizes. The 80/256/1024/4096-byte long-line probes likewise show essentially no gain after 320 (~0.74 tokens/source-byte).
- Unicode identifiers improve from 11.88 mean byte tokens to 8.88 / 8.25 / 8.00. This is useful, but again diminishing.
- Source-code activation of learned vocabulary declines as vocabulary grows: 32/63 learned entries at 320, 62/127 at 384, and 73/180 at 437. On this small real-code diagnostic, 59.4% of BPE-437 learned entries are never activated. That is **not** proof they are wasted globally: Ukrainian/English may use them.

The most frequently activated learned code substrings include indentation/newline chunks, `in`, `ce`, `ion`, `as`, `__`, and at 437 ` =`. This demonstrates that code benefits mostly from ordinary text/identifier and whitespace merges rather than a specialized operator vocabulary.

## Long-line and saturation result

Requested vocabulary 512, 768 and 1024 all yield actual vocabulary 437 with identical vocabulary and merge semantics. A claim that “1024 is better/worse for code” from this fixture would therefore be false: it is the same tokenizer semantics.

## Parameter-matched 250K/500K probe plans

The probe geometry preserves TOK-40's tied embedding, MHA, SwiGLU/RMSNorm assumptions and rebalances the vocabulary tax through `d_ff`, aligned to 8. Parameter count formula used by the incumbent model contract is:

`V*d_model + L*(4*d_model^2 + 3*d_model*d_ff + 2*d_model) + d_model`.

| target | vocab | d_model | layers | d_ff | exact params | delta | embedding share | execution |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 250K | 256 | 64 | 4 | 216 | 248,384 | -1,616 | 6.60% | NOT_RUN_RIGHTS_GATE |
| 250K | 320 | 64 | 4 | 216 | 252,480 | +2,480 | 8.11% | NOT_RUN_RIGHTS_GATE |
| 250K | 384 | 64 | 4 | 208 | 250,432 | +432 | 9.81% | NOT_RUN_RIGHTS_GATE |
| 250K | 437 | 64 | 4 | 200 | 247,680 | -2,320 | 11.29% | NOT_RUN_RIGHTS_GATE |
| 250K | 472 | 64 | 4 | 200 | 249,920 | -80 | 12.09% | NOT_RUN_RIGHTS_GATE |
| 500K | 256 | 96 | 4 | 280 | 495,456 | -4,544 | 4.96% | NOT_RUN_RIGHTS_GATE |
| 500K | 320 | 96 | 4 | 280 | 501,600 | +1,600 | 6.12% | NOT_RUN_RIGHTS_GATE |
| 500K | 384 | 96 | 4 | 272 | 498,528 | -1,472 | 7.39% | NOT_RUN_RIGHTS_GATE |
| 500K | 437 | 96 | 4 | 272 | 503,616 | +3,616 | 8.33% | NOT_RUN_RIGHTS_GATE |
| 500K | 472 | 96 | 4 | 264 | 497,760 | -2,240 | 9.10% | NOT_RUN_RIGHTS_GATE |

All proposed specs are close enough for an honest matched-capacity study, but **none was trained**. Running code BPB, training throughput/source-byte or held-out project-family generalization on DATA-23 would violate the repository's fail-closed rights contract. Status for each is `NOT_RUN_RIGHTS_GATE_NO_TRAINING_ELIGIBLE_REAL_CODE_SOURCE`.

## Ukrainian/English guardrail

No representative Ukrainian/English model-BPB comparison exists for these tokenizer candidates. TOK-37 provides only a synthetic DATA-10 mechanics probe:

| tokenizer | code BPB | English BPB | Ukrainian BPB | Unicode BPB | aggregate BPB |
|---|---:|---:|---:|---:|---:|
| bpe-r320-a320 | 6.240 | 6.557 | 5.018 | 7.798 | 6.016 |
| bpe-r384-a384 | 5.391 | 6.245 | 4.580 | 8.039 | 5.713 |
| bpe-r512-a437 | 5.359 | 5.714 | 4.431 | 8.024 | 5.568 |

On that **non-representative** fixture, 437 does not harm Ukrainian/English BPB versus 384; it improves both. That is encouraging but insufficient for selection or freeze. D04's controlled held-out compression also improves UK/EN fertility as vocabulary grows, but compression is not model BPB.

## Multi-objective recommendation

**Do not freeze or promote a tokenizer from TOK-113 yet.** The strongest next authorized candidate is **ByteLevel BPE requested 512 / actual 437**, with **BPE-384 retained as the lower-vocabulary challenger** and byte-256 retained as the canonical control. Do not spend experiments on requested 768/1024 until a representative tokenizer-training corpus can actually grow the learned vocabulary beyond 437.

Rationale: 437 is best on current real-code token traffic and the synthetic UK/EN BPB guard does not reveal a regression, but its incremental code gain over 384 is small, its learned-entry activation is lower, operators and long-line behavior have already saturated, and the required real-code learning/generalization evidence is legally/data-governance blocked. A promotion decision must wait for: (1) at least one rights-approved real code registry entry, (2) representative UK/EN/code tokenizer training data, (3) frozen project-family held-out split, and (4) the 250K/500K matched-capacity runs above.

No pretrained weights were used. No paid compute was used. No instruction tuning was performed. No broad-intelligence claim is made.
