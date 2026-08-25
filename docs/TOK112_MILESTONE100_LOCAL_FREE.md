# TOK-112 / MILESTONE-100 LOCAL_FREE convergence record

This branch is evidence-first. It does not replace the model, Trainer, checkpoint, streaming/packing, observability, or inference systems. It binds exact proven incumbents and records a LOCAL_FREE source-reconstructed execution performed because the connected runtime did not mount the repository checkout.

## Incumbents

- Product runtime: `integration/s1-transition-20260825@fb9c6d9b73ce436d637077892d73edf136fcaeac` for `TwelveSixDecoder`, `Trainer`, checkpoint contracts, observability, and first-party inference.
- DATA-107/D04: `f9b4783b936055a4165f8bedd00b27f34332d67e` for streaming/packing.
- DATA-25: `8af17afa7baf3d75c2328caf8b08af2400a95e09`, corpus identity `422f545da93526ceba2c0ff738e0b9deac65f35bfb31d87d2aab068cda091fc8`, for the LOCAL_FREE learning run.
- TOK-37: `25cf4798202c41dda4b5413052f5efc6ebbbbf2a`, Actions artifact `9576009309`, digest `sha256:f3fedb2799fc4d16fae637e0cfd96861f7f528deb45145af51ef4726f270e75e`, for reproducible BPE candidates.
- TOK-111: `c252a5c0f4bc365b89a07e0882718752ed59336e` is accepted as the current corpus/tokenizer adjudication. It trained zero tokenizer candidates and intentionally fail-closed until a canonical real-source-backed corpus exists.

The old `milestone100/first-learned-base-20260826@401048afaeb772b8e162fee0554c3464c387d07d` is rejected because its implementation remains a placeholder.

## Ukrainian tokenizer findings

The reserved diagnostic set contains 24 project-authored records and is excluded from tokenizer and model training. SHA-256: `b86d645bf0c1d58988de5fce5978da5d73bd037709f1e5e719bfaab36e04b4c1`.

Byte control uses 14.512 tokens/word and 1.771 tokens/codepoint. TOK-37 BPE437 uses 7.527 tokens/word and 0.918 tokens/codepoint. BPE437 allocates 66 vocabulary entries containing Cyrillic and 11 containing Ukrainian-specific letters. All BPE candidates have zero unknown-token incidence on the diagnostic set. Retained TOK-37 fixed-probe counts reproduce exactly for BPE257/320/384/437.

Matched ~100K and ~500K probes both select BPE437 by final Ukrainian bits-per-byte. At ~500K, final diagnostic BPB is 6.645 for byte control versus 4.288 for BPE437. Tokens/word and final Ukrainian BPB correlate at +0.9993 (~100K) and +0.9981 (~500K) across the five tokenizer conditions.

This is not a tokenizer freeze. TOK-37 was fitted on a 1,454-byte project-authored DATA-10 corpus, and inflection-boundary consistency is not monotonically better as BPE grows. The evidence does not justify a Ukrainian-specific tokenizer. Retest general ByteLevel BPE384/BPE437 after the canonical real-source corpus gate passes.

## Learned Base result

The complete vertical remains byte-tokenized because the proven D07 path currently fail-closes on non-canonical tokenizer metadata. The learned Base is 467,808 parameters, random initialized with seed 1337, `d_model=96`, four layers, four heads, `d_ff=256`, context 256, and AdamW at `3e-4`.

The run completed 800 optimizer steps and 201,600 optimized causal tokens. Checkpoints were retained at 200, 400, 600, and 800. A fresh Python process restored step 400 and continued to 800.

Mean train loss fell from 4.7828 over the first 32 updates to 0.8261 over the final 32 post-resume updates. Initial -> final BPB: Ukrainian DATA-25 validation 7.9652 -> 1.8727, English 8.0226 -> 0.5555, code 7.5486 -> 0.7855, reserved Ukrainian diagnostics 7.9843 -> 4.4361. Evaluation state hashes prove non-mutation in both phases.

The reserved Ukrainian diagnostic reached 3.3355 BPB at step 400 and regressed to 4.4361 at step 800 while in-domain validation continued improving. This late OOD-overfit signal is explicitly retained.

## Verdict

`PASS_GENUINELY_LEARNED_LOCAL_BASE` for the learned artifact. `FAIL_UNMET` for the strict real representative corpus gate. DATA-25 has zero external training-eligible sources and cannot support a real-world representativeness claim. Therefore the overall status is `PARTIAL_SUCCESS_STRONGEST_DEFENSIBLE_CURRENT_LOCAL_ARTIFACT`.

No foreign pretrained weights, instruction tuning, paid compute, or broad intelligence claim was used.

Verify committed evidence with:

```bash
python -m pytest -q tests/test_tok112_milestone100_evidence.py
```
