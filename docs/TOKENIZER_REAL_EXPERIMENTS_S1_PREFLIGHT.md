# Real learned-tokenizer mechanics preflight

Status: EXPERIMENTAL / controlled mechanics evidence only.

This package continues PR #73. It closes only the previous `NOT_RUN` maintained-library
execution gap for future BPE/Unigram experiments. It does not change canonical S0
`s0-byte-v1`, approve an external corpus, freeze an S1 tokenizer, or make a model-quality
claim.

## Runtime identity

The dedicated workflow runs on Ubuntu 24.04 with CPython 3.11.16. Its isolated venv first
installs the committed canonical x86-64 toolchain/runtime lock groups with `--require-hashes`
and `--no-deps`, then adds only the experiment backend `tokenizers==0.23.1` from a one-line
hash-locked requirement using `--require-hashes --only-binary=:all: --no-deps`.

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

The D03 fixture is purpose-written project data. It is useful for controlled mechanics
but is not representative S1 training data and is not a universal license/benchmark-clean
claim.

## Comparison and repeatability contract

BPE and Unigram receive the exact same ordered train texts, exact dataset identity and
requested vocabulary size (512 by default). Each algorithm is rebuilt twice. Both builds
must independently preserve strict held-out round trip and emit zero unknown tokens.

Exact artifact repeatability is an **observed gate**, not an assumption. The report compares
the complete tokenizer artifact identity from both builds and records changed identity
fields when they drift. A deterministic repeat reports `PASS`; a differing exact artifact
reports `FAIL`. The validator rejects any report that relabels observed drift as PASS.

The first real maintained-library run on exact head
`c3a3fe4672faea1a9b94dc328f2761bce407e5d8` showed that BPE reached the repeatability check
but Unigram produced different exact artifact identity across identical repeated builds.
That head therefore failed before an evidence artifact was retained. This is treated as a
scientific blocker, not hidden as an infrastructure error. The v2 runner retains such a
truthful result with decision `NO_FREEZE_REPEATABILITY_BLOCKED` so later evidence can state
exactly which fields drifted while still proving that real execution occurred.

The public `UnigramTrainer` interface used by the pinned backend exposes training controls
such as vocabulary size, alphabet, shrinking factor and EM sub-iterations, but no explicit
random-seed control. The project therefore does not invent a seed guarantee that the
maintained backend does not expose. A future version/backend change must be separately
locked and compared rather than silently accepted.

Held-out probes record token counts, fertility and vocabulary parameter cost for both
repeated builds. No algorithm is declared the winner.

A retained report has authority:

`CONTROLLED_S0_TRAIN_SPLIT_MECHANICS_ONLY_NOT_S1_CORPUS_OR_FREEZE`

It must still report `NOT_TESTED` for representative S1 corpus suitability, external
source rights approval, S1 tokenizer freeze and model quality.

## What a green evidence-capture run does not authorize

A green workflow means the evidence runner completed and the report validated its own
claims. It does **not** mean every scientific gate passed. In particular a report may
truthfully contain `repeatable_artifact_identity=FAIL` and still be retained for audit.

A green evidence-capture run does not:

- approve any public or external source for Base training;
- admit validation or benchmark/test material into tokenizer training;
- select BPE or Unigram for S1;
- freeze vocabulary size or ModelSpec;
- turn a validator-mechanics PASS into model-quality PASS;
- grant AUDIT PASS, CANDIDATE or STABLE;
- authorize paid compute or foreign pretrained weights.

## Next stage gate

The next meaningful tokenizer decision requires two independent closures:

1. Resolve or explicitly exclude any algorithm/runtime whose exact repeated artifact
   identity is unstable under the locked experiment contract.
2. Build a reviewed representative corpus with immutable retrieval/source identities,
   explicit rights state, exact/near dedup and benchmark decontamination, then rerun all
   eligible tokenizer algorithms on the same accepted train-only corpus.

Only after those gates should parameter-budget and controlled model-loss comparisons be
used for an S1 tokenizer decision. Until then the winner remains `null` and canonical S0
remains the byte tokenizer.
