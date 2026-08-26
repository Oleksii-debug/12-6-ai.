# TOK-239 BPE Research Corpus V1

Worker: `TOK-239-BPE-RESEARCH-CORPUS-V1`

## Result at this source cutoff

`BLOCKED_NO_TERMINAL_EXTERNAL_REAL_RESEARCH_CORPUS`

TOK-239 did not execute tokenizer fitting, model optimization, or BPB ranking because the required post-MILESTONE-238 external-real research-corpus authority is not terminal and there is no eligible immutable selection-validation set to score promotion candidates.

This is a scientific blocker, not a tokenizer failure.

## Live authority reconstruction

The strongest relevant repository state observed before opening TOK-239 is:

- no terminal branch, PR, or commit for `MILESTONE-238-CORPUS-RESEARCH-V1` was published;
- branch `data230/corpus-v03-external-real-20260826` exists, but its observed head is `6d994e2aece6c44e28c1a2c344ac98b5a8fd5e08`, whose commit subject is `DATA-214 restore retained quality and privacy evidence`, not a terminal DATA-230 corpus publication;
- EVAL-233 PR #365 at observed head `b5512b4648cb09dd052b08884dc53f291e1ce935` explicitly publishes selection-validation as `BLOCKED_NO_IMMUTABLE_SELECTION_VALIDATION_AUTHORITY` with 0 records while DATA-230 is absent;
- DATA-229 PR #369 at observed head `90bc0b7f8b696ec35202532b13edf6ab29a662fe` converges three external-real text snapshots at its cutoff and records zero admitted code sources there. That is useful provenance inventory, but it is not a terminal UA/EN/code research-corpus freeze.

Therefore neither DATA-25 nor DATA-183 is relabelled as Research Corpus V1, and final-test bytes are not reused for tokenizer or model selection.

## Existing BPE implementation retained

TOK-239 is stacked on exact TOK-187 head `289b3b8d3e0b348bc699b88359542b9cb695c024` and reuses the existing deterministic HF Tokenizers path:

- runner: `src/twelve_six/tok187_bpe_real.py`;
- implementation: `src/twelve_six/tokenization/experiments.py::train_hf_tokenizer`;
- library/runtime: `tokenizers==0.23.1`;
- no second BPE library is introduced.

TOK-187 already contains the required mechanics for deterministic tokenizer training, strict roundtrip, unknown-token checks, fertility, throughput, embedding tax and parameter-matched learned probes. TOK-239 changes the data-authority boundary, not the BPE algorithm.

## Preregistered V1 grid and reproducibility gate

The requested vocabulary grid remains the prior promising family:

`320, 384, 437, 512`

For each requested size, two independent tokenizer trainings are mandatory. Promotion evidence is invalid unless their serialized artifact identities are byte-identical.

Every admitted candidate must report:

- strict text roundtrip;
- unintended unknown-token count/behavior;
- Ukrainian fertility;
- English fertility;
- code fertility;
- worst-modality fertility;
- tokenizer throughput;
- embedding parameter tax.

## Parameter-matched ~500K probes

The primary model comparison remains the incumbent matched-capacity target:

- total trainable parameters: `467,808`;
- tokenizer-dependent embedding growth is offset by rebalancing non-embedding capacity through the incumbent vocabulary/model helper rather than granting larger vocabularies a larger total model;
- bounded optimized-token budget: `16,384` actual optimized loss tokens per arm;
- paired model seeds: `1337`, `7331`, `18701`;
- identical model/training controls apart from tokenizer-dependent geometry required to preserve total capacity.

The promotion ranking is primary by paired held-out aggregate BPB on immutable selection-validation. Compression/fertility is secondary. A candidate cannot be promoted from a single model seed.

## Evaluation isolation

`final-test` is prohibited from tokenizer fitting, vocabulary selection, model hyperparameter selection, checkpoint selection and candidate ranking.

TOK-239 may execute numerical evidence only after it can bind, before tokenizer fit or optimizer step 1:

1. a terminal MILESTONE-238 decision or exact later superseding research-corpus authority;
2. the exact terminal external-real corpus identity and immutable train inventory;
3. a non-empty immutable selection-validation identity eligible for tokenizer/model selection;
4. UA/EN/code selection-validation strata required by this experiment;
5. exact source/corpus/split hashes.

If those conditions are not true, the correct result remains blocked. Earlier DATA-25 or DATA-183 evidence may be cited only as historical controls, never as `BPE Research Corpus V1` evidence.

## Committed blocker evidence

`evidence/tok239/authority-gate.json` is self-hashed and records the current fail-closed state plus the frozen experiment contract. `tools/validate_tok239_authority_gate.py` and `tests/test_tok239_authority_gate.py` reject weakened numerical, fallback, final-test, vocabulary-grid, seed, capacity or implementation boundaries.

Training started: **false**. Optimizer updates: **0**. Numerical tokenizer/model result claimed: **none**. Tokenizer promoted: **none**.

LOCAL_FREE only.
