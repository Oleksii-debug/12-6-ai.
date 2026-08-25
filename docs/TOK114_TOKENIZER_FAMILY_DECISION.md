# TOK-114 tokenizer family decision

Decision: `KEEP_BYTE_CONTROL`

Authority: evidence review only. This does not freeze the special-token policy, promote BPE, or alter a canonical stage configuration.

## Exact audit snapshot

Repository: `Oleksii-debug/12-6-ai.` (the live repository name includes the trailing period).

Review branch base: `015593b22a600184fb4c8001fe3d70893bfc51d5` (`milestone100/first-learned-base-20260826`).

Evidence anchors:

- byte incumbent: `s0-byte-v1`; config SHA-256 `b04055c1061dd641dcab7cb9d62a931f09b8d1a070140a926ceb4e91d73ca8e1`; vocabulary SHA-256 `905ed40bb42cc4d550e228ff5f24158d504b38e8ed5974dfa3077bd5867ad571`.
- BPE sweep: `tok37/bpe-vocab-sweep-20260825@25cf4798202c41dda4b5413052f5efc6ebbbbf2a`; workflow run `32881557815`; retained artifact digest `sha256:f3fedb2799fc4d16fae637e0cfd96861f7f528deb45145af51ef4726f270e75e`.
- Unigram reproducibility audit: `d10/tokenizer-mixture-future-20260824@e925109473822bcd11ceef71f98f1441a6816f62`; pinned `tokenizers==0.23.1`.
- model rebalance evidence: `tok40/model-rebalance-20260825@cf4e1ad1326c5f86e696faa57e3d4fd724a089fc`; construction evidence only, not tokenizer quality evidence; current x86 CI is red.
- current corpus: DATA-25 V0.1 identity `422f545da93526ceba2c0ff738e0b9deac65f35bfb31d87d2aab068cda091fc8`; 20,000,775 train byte tokens, 1,410,473 validation byte tokens; `external_training_eligible_sources=0`.
- real-corpus gate: DATA-30 reports `BLOCKED_NO_REAL_TRAINING_ELIGIBLE_UK_EN_CODE_CORPUS`.

## Family comparison

| Criterion | byte | BPE | Unigram |
| --- | --- | --- | --- |
| Exact artifact reproducibility | PASS by construction | PASS on TOK-37 repeated builds | FAIL on pinned maintained backend |
| Exact text round trip | PASS for valid UTF-8 text | PASS on retained probes | semantic tests can pass, but artifact/token-id identity drifts |
| Unknown tokens | zero | zero on retained probes | not selection-eligible |
| Actual vocabulary | 256 | 437 for TOK-37 requested 512; training corpus saturates at 437 | 497 in the controlled audit, but ineligible |
| Learned tokenizer artifact | none | 7,436-byte tokenizer JSON for the actual-437 candidate | unstable identity; do not compare artifact bytes as canonical |
| Embedding tax vs byte | baseline | +8,688 params at d_model=48; +23,168 at d_model=128 for vocab 437 | not selection-eligible |
| Tokenizer throughput | deterministic simple byte path; no same-harness retained number | NOT_RETAINED in TOK-37; no defensible same-harness comparison | not selection-eligible |
| ~100K parameter-matched model BPB | byte remains incumbent control | only a 32-step, single-seed mechanics probe on the non-representative TOK-37 fixture; not a byte-vs-BPE promotion experiment | not selection-eligible |
| ~1M parameter-matched model BPB | byte is the current MILESTONE-100 model binding | MISSING | not selection-eligible |
| Representative-corpus model evidence | MISSING because current DATA-25 is project-authored-only | MISSING | MISSING and ineligible |

## Fertility on the retained TOK-37 held-out mechanics probes

The requested-512 BPE trained to actual vocabulary 437. Across 638 UTF-8 bytes it emitted 417 tokens, a 34.64% token-count reduction relative to the byte tokenizer's 638 tokens on the same strings.

- Ukrainian: 337 UTF-8 bytes / 186 codepoints -> 175 BPE tokens = 0.9409 tokens/codepoint and 0.5193 tokens/byte. Byte control is 337 tokens = 1.8118 tokens/codepoint.
- English: 71 bytes/codepoints -> 48 BPE tokens = 0.6761 tokens/codepoint; byte control is 71.
- code: 72 bytes/codepoints -> 46 BPE tokens = 0.6389 tokens/codepoint; byte control is 72.
- Unicode edge probes remain materially less compressed and therefore must stay in future regression coverage.

These are useful mechanics/capacity results, not representative-corpus quality evidence.

## Unigram ruling

`INELIGIBLE_REPRODUCIBILITY`.

With `tokenizers==0.23.1`, the maintained Unigram trainer exposes no supported deterministic seed and fresh processes can produce different ordered vocabulary/token IDs and held-out segmentations. Checkpoint embedding and output rows bind exact token IDs, so semantic-equivalence hashing is not an acceptable workaround. No project-authored trainer is introduced. Unigram may re-enter only if a separately pinned maintained backend/version provides a supported deterministic contract and repeated fresh-process training yields byte-identical tokenizer artifacts, identical ordered vocabulary/token IDs, identical held-out encodings, exact round trips and zero unknowns.

## BPE ruling

`ELIGIBLE_EXPERIMENTAL_CHALLENGER`, not selected.

TOK-37 proves exact repeated artifact identity, exact round trip and substantial UA/EN/code token-count reduction. It does not prove promotion because:

1. its training corpus explicitly declares `representative_corpus: false`;
2. its model probe is ~100K only, 32 steps and one seed;
3. there is no parameter-matched ~1M BPE-vs-byte BPB result;
4. there is no same-harness tokenizer throughput result;
5. current DATA-25 itself is project-authored-only and cannot satisfy the requested real-world representative-corpus gate;
6. TOK-40 proves geometry construction, not BPE quality, and its current x86 CI is red.

## Decision rationale

The selection rule requires a deterministic artifact plus material model-level evidence against byte or a quantified capacity/throughput tradeoff. BPE satisfies the deterministic-artifact half but not the model-level/throughput half on one eligible current corpus. Unigram fails the prerequisite itself. Therefore the only defensible architecture decision is `KEEP_BYTE_CONTROL`.

This is stronger than `INSUFFICIENT_EVIDENCE`: evidence is sufficient to reject Unigram and to withhold BPE promotion while retaining the already deterministic byte control. It is not evidence that byte is intrinsically optimal.

## Exact BPE re-entry gates

Before `SELECT_BPE` can be reconsidered, run one exact corpus manifest through both byte and one deterministic BPE artifact and retain:

- two fresh-process BPE builds with byte-identical artifact and token-id surface;
- exact round trip and zero unknowns on train/held-out edge suites;
- UA/EN/code fertility and serialized artifact bytes;
- same-machine, same-process/harness tokenizer encode throughput;
- exact actual vocabulary and tied-embedding tax;
- parameter-matched ~100K and ~1M models, same optimizer/data order/token budget policy, at least three seeds where affordable;
- held-out BPB normalized by original UTF-8 bytes, with per-stratum UK/EN/code results;
- evidence that any win survives capacity rebalance rather than merely buying more/fewer non-embedding parameters.

The special-token policy remains outside this decision.