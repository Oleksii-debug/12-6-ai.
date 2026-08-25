# Vocabulary parameter allocation: stage-aware engineering policy

Status: experimental engineering guidance, not a tokenizer freeze or stage promotion.

## Measured controlled evidence

The exact-head D04 real-tokenizer workflow at source
`3527bf50d5d6cd7d4cbf9e477863430a31873baa` used the manifested D03 controlled train
split and held out validation plus project-authored code/Unicode probes. The byte baseline
requires 520 held-out tokens.

At requested vocabulary 512, BPE produced an actual vocabulary of 472 entries and 286
held-out tokens: a 45.0% reduction from bytes. It was exact-round-trip, zero-unknown and
repeatable at full artifact identity. Unigram produced 497 entries and 284 tokens, but its
repeated artifact identity failed. Therefore neither tokenizer is frozen, and Unigram is not
eligible for a reproducible freeze decision in the current runtime.

For S1 width 48, the repeatable BPE vocabulary costs 22,656 tied parameters versus 12,288
for the 256-byte control. The extra 10,368 parameters buy a 45% token-count reduction on
this controlled suite. The output-projection work proxy is nearly flat: BPE is
`286*472/(520*256) = 1.014`, so the larger vocabulary is roughly offset by fewer token
positions in the vocabulary projection while attention/MLP sequence work falls strongly.
This is mechanics evidence only; representative-corpus quality remains untested.

## Parameter allocation finding

The current tied-vocabulary shares are approximately 22.8% at S1, 24.6% at S2, 26.1% at
S3, 25.1% at S4, then only 8.4% at S5. The repeated ~25% allocation from S1 through S4 is
not itself evidence that those vocabulary sizes are optimal. Untying the output head doubles
those dominant matrices before any optional output bias, reaching roughly 45.6%, 49.2%,
52.1% and 50.1% of the current S1-S4 candidates respectively. Untied heads therefore stay
out of the default search until model scale and quality evidence justify the cost.

## Stage decisions changed

- S1: replace a single assumed 512 vocabulary with a 256-byte control and real learned
  tokenizer search over 320/384/448/512 requested sizes. Bind the ModelSpec to the actual
  trained vocabulary, not the requested number. The measured 472-entry BPE is a mechanics
  anchor, not a freeze.
- S2: stop treating 2,048 as the default. Search 512-1,536 first, with 1,024 as the
  parameter-rebalanced anchor. A 1,024 vocabulary at D=128 uses 131,072 tied parameters;
  rebalancing d_ff from 352 to 392 yields 996,480 total parameters, moving budget from the
  embedding table back into transformer capacity.
- S3: search 2,048-6,144 first. Keep 8,192 only as an upper stress control until
  representative fertility/quality evidence justifies its ~26% tied share.
- S4: search 8,192-24,576 first. Keep 32K as a stress/control point rather than a default;
  at D=768 it consumes 25,165,824 tied parameters (~25.1% of the current 100M candidate).
- S5: 32K becomes parameter-reasonable (~8.4% tied). Search 24,576-65,536 only with
  representative multilingual fertility and model-quality evidence.

Machine-readable bands are in `configs/vocabulary/stage_search_bands.v1.json`. Non-frozen
S1/S2 rebalanced candidates are in
`configs/vocabulary/s1_s2_rebalanced_candidates.v1.json`.

## Solver interface

`twelve_six.vocabulary.rebalance_d_ff_for_vocabulary` changes vocabulary size and then
retargets `d_ff` to the closest aligned parameter budget using the existing ModelSpec exact
parameter algebra. `python -m twelve_six.vocabulary matrix` consumes real tokenizer reports,
uses each trained artifact's actual vocabulary size, computes parameter/token Pareto points,
excludes non-repeatable artifacts from the freeze-eligible frontier, and emits rebalanced
ModelSpecs.

The MODEL37 workflow executes real BPE/Unigram experiments at requested vocabularies 320,
384, 448 and 512 on the same manifested controlled corpus. This matrix is qualification
evidence only. Representative S1 corpus, training loss/quality, multilingual balance and
rights approval remain mandatory before tokenizer selection.
