# Real learned-tokenizer mechanics preflight

Status: EXPERIMENTAL / controlled mechanics evidence only.

This package continues PR #73. It closes only the previous `NOT_RUN` maintained-library
execution gap for future BPE/Unigram experiments. It does not change canonical S0
`s0-byte-v1`, approve an external corpus, freeze an S1 tokenizer, or make a model-quality
claim.

## Runtime identity

The dedicated workflow runs on Ubuntu 24.04 with CPython 3.11.16 and installs only the
experiment dependency `tokenizers==0.23.1` from a one-line hash-locked requirement using
`pip --require-hashes --only-binary=:all: --no-deps`.

The admitted x86-64 wheel SHA-256 is:

`5075b405006415ea148a992d093699c66eb01952bf59f4d5727089a98bda45a4`

This is an experiment-only runtime identity. It deliberately does not mutate or weaken the
canonical D08 environment locks.

## Corpus boundary

Tokenizer training consumes only the committed D03 controlled **train** split:

- dataset: `s0-tiny-controlled-v1`
- dataset identity:
  `bab60119d49e93303c972b77900fcb5553817f754cbc5d9a58019228cfa0ca89`
- train JSONL SHA-256:
  `61d24b7138df56527d201cea405d11c9f607684b4a9593dfa20c599cc2ee6998`
- validation JSONL SHA-256:
  `57f18a846dcca75955a82612382d4635ba9583965aa6628e77626cd2a3eb19c5`

The ten train records are the only tokenizer-training inputs. The two validation records
are held out from tokenizer training and used only as probes. Project-authored code and
mixed-Unicode probes are also evaluation-only. The runner verifies split IDs, committed
hashes and D03 manifest assignments before training.

The D03 fixture is purpose-written project data. It is useful for deterministic mechanics
but is not representative S1 training data and is not a universal license/benchmark-clean
claim.

## Comparison contract

BPE and Unigram receive the exact same ordered train texts, exact dataset identity and
requested vocabulary size (512 by default). Each algorithm is rebuilt twice. Evidence
fails closed unless both rebuilds have identical tokenizer JSON identity, ordered
token-ID vocabulary identity and runtime config identity.

Held-out probes must have exact round trip and zero unknown tokens. The report records
token counts, fertility and vocabulary parameter cost, but no algorithm is declared the
winner.

A successful report has authority:

`CONTROLLED_S0_TRAIN_SPLIT_MECHANICS_ONLY_NOT_S1_CORPUS_OR_FREEZE`

It must still report `NOT_TESTED` for representative S1 corpus suitability, external
source rights approval, S1 tokenizer freeze and model quality.

## What a green run does not authorize

A green run does not:

- approve any public or external source for Base training;
- admit validation or benchmark/test material into tokenizer training;
- select BPE or Unigram for S1;
- freeze vocabulary size or ModelSpec;
- grant AUDIT PASS, CANDIDATE or STABLE;
- authorize paid compute or foreign pretrained weights.

## Next stage gate

The next meaningful tokenizer decision requires a reviewed representative corpus with
immutable retrieval/source identities, explicit rights state, exact/near dedup and
benchmark decontamination. BPE and Unigram must then be rerun on that same accepted
train-only corpus under an exact runtime lock, followed by parameter-budget and controlled
model-loss comparisons. Until then the winner remains `null` and canonical S0 remains
the byte tokenizer.
