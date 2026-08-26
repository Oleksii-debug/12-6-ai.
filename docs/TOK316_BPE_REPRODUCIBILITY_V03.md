# TOK-316 BPE Reproducibility V03

`SWARM_WORKER_ID: TOK-316-BPE-REPRODUCIBILITY-V03`

## Verdict

`BLOCKED_NO_TERMINAL_TOKENIZER_FIT_CORPUS`

No BPE candidate was trained at this evidence cutoff. This is a purpose/data-authority blocker, not a BPE failure and not a tokenizer-family decision.

TOK-316 is bound to DATA-300 v2 exact head `8ea7f830e50a23754d189dd4134f4afad76a7ee9` and contract identity `07d7beaaff4616e839450de6af3d407855c832bf75a24a959d1a12de5d9364e5`. DATA-300 explicitly reports `NOT_BUILT_NOT_FROZEN_NOT_TERMINAL` and requires a later Wave-3 materialization before train bytes are authoritative.

At the TOK-316 cutoff, branch `data301/corpus-v03-terminal-build-20260826` exists but is identical to the DATA-300 head: zero commits ahead, no terminal corpus identity, no train-shard hashes and no frozen tokenizer-fit inventory are published there. No `tok315` branch or exact superseding tokenizer-fit eligibility authority is published at the cutoff.

Running BPE against DATA-300's source candidates, DATA-229 snapshots, DATA-183, DATA-25, selection-validation, final-test, or any locally reconstructed approximation would therefore weaken the frozen purpose boundary. TOK-316 fails closed instead.

## Maintained implementation binding

The only allowed BPE implementation for this experiment remains:

- `src/twelve_six/tokenization/experiments.py::train_hf_tokenizer`
- Hugging Face `tokenizers==0.23.1`
- `models.BPE` + `trainers.BpeTrainer`
- ByteLevel pre-tokenizer and decoder
- no tokenizer normalization
- `<unk>` as the first special token
- `min_frequency=2`
- ByteLevel initial alphabet

No second BPE implementation, homemade fallback, runtime version drift or alternate library is permitted as a substitute for the requested maintained HF trainer.

## Frozen candidate protocol

The candidate grid is exactly:

`320, 384, 437, 512`

Each candidate requires two independent trainings from the exact same ordered eligible tokenizer-fit train texts. The complete serialized tokenizer JSON bytes and ordered token-to-ID vocabulary semantics must be identical between the two runs. Any drift is a reproducibility failure for that candidate.

Every completed candidate must additionally prove, on the eligible tokenizer-fit train corpus only:

- strict UTF-8 text roundtrip for every record;
- zero unintended `<unk>` tokens;
- Ukrainian fertility in tokens/codepoint and tokens/UTF-8-byte;
- English fertility in the same units;
- code fertility in the same units;
- tokenizer throughput over five repeated measurements, reporting UTF-8 bytes/s and tokens/s;
- embedding parameter tax relative to the canonical 256-byte vocabulary.

Selection-validation and final-test bytes are not inputs to TOK-316 fitting or metrics. TOK-316 does not select a tokenizer winner and does not make a model-family choice.

## Embedding-tax accounting

The byte baseline is the canonical `s0-byte-v1` vocabulary size 256. For model width `d_model`, incremental vocabulary parameters are:

- tied input/output embedding: `(V - 256) * d_model`;
- untied input embedding plus LM head: `2 * (V - 256) * d_model`.

| Requested V | Tied tax | Untied tax |
| ---: | ---: | ---: |
| 320 | `64 * d_model` | `128 * d_model` |
| 384 | `128 * d_model` | `256 * d_model` |
| 437 | `181 * d_model` | `362 * d_model` |
| 512 | `256 * d_model` | `512 * d_model` |

These are accounting coefficients, not a model-family recommendation.

## Numerical result matrix at this cutoff

| Vocab | Run 1 | Run 2 | Byte-identical | Roundtrip | Unintended UNK | UA fertility | EN fertility | Code fertility | Throughput |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 320 | NOT_RUN | NOT_RUN | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED |
| 384 | NOT_RUN | NOT_RUN | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED |
| 437 | NOT_RUN | NOT_RUN | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED |
| 512 | NOT_RUN | NOT_RUN | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED |

Tokenizer trainings completed: `0 / 8`.

Final-test bytes read: `false`.

Selection-validation bytes read: `false`.

Tokenizer/model-family winner claimed: `false`.

## Unblock gate

Numerical TOK-316 execution becomes eligible only when all of the following are simultaneously available and exactly bound before training run 1:

1. terminal frozen external-real corpus identity and deterministic train-shard hashes;
2. TOK-315 or an exact superseding authority that binds the train-only tokenizer-fit inventory and its identity/hash;
3. explicit proof that selection-validation and final-test bytes are excluded from tokenizer training;
4. exact local/free `tokenizers==0.23.1` runtime.

Once those conditions exist, all four candidates must be rerun twice from scratch. No result in this blocker checkpoint may be reused as numerical evidence because none exists.

## Durable evidence

- `evidence/tok316/authority-gate.json` — self-hashed exact cutoff, protocol and blocker truth.
- `tools/validate_tok316_authority_gate.py` — rejects weakened corpus, runtime, grid, exposure or numerical claims.
- `tests/test_tok316_authority_gate.py` — adversarially verifies that invented training, final-test exposure, runtime substitution and grid drift fail even after recomputing the evidence self-hash.

LOCAL_FREE only.
