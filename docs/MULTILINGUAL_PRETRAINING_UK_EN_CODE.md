# DATA-10 multilingual Base-pretraining pathway

Status: EXPERIMENTAL, NOT FROZEN, NOT A PROMOTION CLAIM.

This work composes the existing production-corpus foundation from PR #64 with the
existing tokenizer/mixture work from PR #73. The composition commit has both exact
heads as parents. No parallel tokenizer or data framework is introduced, and the
canonical S0 byte tokenizer remains unchanged.

## Scope and truth boundary

The initial data strata are Ukrainian (`uk`), English (`en`) and source code (`code`).
This is Base next-token pretraining only. It does not add instruction data, alignment,
refusal behavior, assistant formatting or domain-specialized model behavior. The
architecture remains language-universal.

At the time this recipe was created, the external-source registry contained zero
sources approved for model training. Therefore the executable DATA-10 benchmark uses
only project-authored synthetic/local text. That evidence proves mechanics; it is not
representative-corpus evidence and cannot freeze a tokenizer or authorize a larger run.
Any external source must pass the incumbent D03 rights/provenance contract with
`APPROVED_FOR_TRAINING` and an explicit model-training permission before ingestion.

## Admission pipeline

The production path is deliberately fail-closed:

1. Bind source identity, version, immutable manifest SHA-256, purpose and rights.
2. Reject validation, evaluation, test, held-out and benchmark purposes from training.
3. Validate strict UTF-8 scalars. U+FFFD and surrogate code points are rejected.
4. Natural text keeps the incumbent D03 NFKC normalization semantics.
5. Code uses layout-preserving NFKC: newline form is canonicalized but indentation and
   internal whitespace are not collapsed.
6. Compute the normalized-content SHA-256 and reject reserved benchmark/evaluation
   fingerprints before the record can enter training.
7. Apply conservative UK/EN language evidence. Code is an explicit source modality,
   not inferred from alphabetic ratios.
8. Use the incumbent corpus-quality hooks, exact SQLite deduplication and the D03
   scalable near-dedup seam. Do not create a second dedup store.
9. Freeze split membership before tokenizer training. Tokenizer training consumes only
   the training split; held-out probes never enter tokenizer fitting.
10. Run cross-split exact/near contamination checks. When a collision exists, training
    loses the record; held-out material is not silently moved into training.

The current UK/EN detector is intentionally conservative and suitable as an admission
preflight, not as the permanent universal LID system. A larger approved corpus should
plug a maintained audited LID implementation into the existing quality-policy seam.

## Mixture and restart semantics

Initial experimental token-budget weights are:

- Ukrainian: 45 units
- English: 35 units
- code: 20 units

These units target post-tokenization loss tokens, not document counts, characters or
raw bytes. That distinction matters because raw bytes systematically charge Ukrainian
more token positions than English. The weights are an initial Ukrainian-focused
experiment, not a permanent language policy.

Selection and restart reuse PR #73 `MixturePlan`, `MixtureSource` and `RestartCursor`.
A restart is valid only when the plan SHA, tokenizer config SHA, tokenizer vocab SHA,
packing SHA and all source-manifest SHAs match. The DATA-10 evidence replays a 10,000
sample schedule in one pass and as 4,321 + 5,679 samples and requires identical counts
and the identical final cursor.

Source-level sub-mixtures should be capacity-aware and quality-aware inside each
language/modality stratum. A high weight must never bypass a source's rights decision,
quality gate or contamination status.

## Tokenization evidence and training cost

The exact-green incumbent PR #73 controlled experiment established the following on
its small held-out fixture:

- raw byte: 256 vocabulary; Ukrainian fertility 1.8681 tokens/codepoint on the
  controlled Ukrainian fixture, versus 1.0 for the controlled English fixture;
- ByteLevel BPE: requested 512, actual 472 vocabulary; 286 held-out tokens versus 520
  byte tokens, a 45.0% token-count reduction; Ukrainian fertility 0.7518; strict
  round-trip; zero unknowns; repeated artifact identity passed;
- Unigram with ByteLevel I/O: requested 512, actual 497; 284 held-out tokens, a
  45.38% reduction; Ukrainian fertility 0.7447; strict round-trip; zero unknowns;
  repeated artifact identity failed.

Those numbers are controlled mechanics evidence, not a representative Ukrainian corpus
result. Unigram therefore remains blocked from freeze by repeatability, and neither
learned tokenizer is frozen until representative rights-approved UK/EN/code data is
measured through the same manifested comparison.

For a tied output head, vocabulary parameters are `vocab_size * d_model`. Relative to
raw bytes, the controlled BPE vocabulary adds 216 embedding parameters per hidden
width and Unigram adds 241. At widths 128/320/768 that is +27,648/+69,120/+165,888 for
BPE and +30,848/+77,120/+185,088 for Unigram.

Tokenizer choice must be co-designed with ModelSpec. The current S2 engineering config
has vocab 2,048 and 1,066,112 trainable parameters. Replacing only that vocabulary with
472 entries would remove `(2048 - 472) * 128 = 201,728` tied vocabulary parameters and
would change model identity and the stage's parameter count. The current S3 config has
vocab 8,192 and width 320; the same substitution would remove 2,470,400 parameters.
A tokenizer must never be silently swapped into an existing stage config.

A 45% token-count reduction means approximately 45% fewer token positions to expose
the same controlled text, but it is not automatically a 45% end-to-end compute saving.
Softmax vocabulary cost, packing utilization, sequence length, attention cost and model
geometry also change. Larger-run accounting must report token positions, model
parameters, step throughput, wall time and accelerator-hours separately.

## Packing and executable next-scale probe

Canonical S0 has no semantic EOS, so the incumbent packer isolates documents by
default. The current experimental learned tokenizers also have no frozen BOS/EOS
semantics; cross-document packing remains forbidden unless an explicit versioned EOS
contract is introduced.

`tools/benchmark_multilingual_pretraining.py` admits local UK/EN/code records, trains
BPE and Unigram on the same manifested training corpus, evaluates held-out Ukrainian
morphology/orthography, English, code and Unicode probes, measures token cost, proves
mixture replay, packs records with the incumbent packer, instantiates the real current
S2 `1,066,112`-parameter random-init decoder and performs forward/backward/AdamW update.
The evidence explicitly records that no foreign pretrained weights and no paid compute
were used.

## Reproducible recipe

The committed experimental recipe is
`configs/data/multilingual_uk_en_code_v1.experimental.json`.

Core regression:

```text
pytest tests/test_multilingual_pretraining.py
```

The retained evidence job is `.github/workflows/data10-multilingual-pretraining.yml`.
It binds the exact PR head, installs the project's hash-locked runtime plus the existing
hash-locked Tokenizers backend and emits `data10-multilingual-evidence.json`.

## Corpus requirements for serious stages

The following are planning floors at 20 post-tokenization training tokens per target
parameter. They are not quality guarantees and do not authorize training by themselves.

| Stage | Total train tokens | Ukrainian 45% | English 35% | code 20% | held-out minimum |
| --- | ---: | ---: | ---: | ---: | ---: |
| ~1M | 20M | 9M | 7M | 4M | 100k |
| ~10M | 200M | 90M | 70M | 40M | 1M |
| ~100M | 2B | 900M | 700M | 400M | 10M |

Before a ~1M serious run: require at least two independently reviewed source families
per stratum, no single source family above 60% of a stratum, 100% provenance/rights
coverage, a frozen held-out registry, real exact and near-dedup evidence, and manual LID
review of at least 100 sampled natural-language records per language.

Before ~10M: require at least three source families per stratum, no single family above
40%, distributed near-dedup evidence, at least 500 manually reviewed LID samples per
natural language, a held-out Ukrainian morphology/fertility diagnostic with at least
10,000 wordforms, source-level quality reports and an exact restart/resume proof on the
final manifests.

Before ~100M: require at least five source families per stratum, no single family above
25%, cross-source near-deduplication at full corpus scale, at least 2,000 manually
reviewed LID samples per natural language, benchmark decontamination with exact and
near-match checks, code-license allowlisting with repository/file provenance, source
quality dashboards and retained end-to-end restart evidence.

These diversity caps are engineering safeguards for this recipe, not claims that a
particular source is already approved. Until external review populates the registry,
only permitted project-authored synthetic/local data may execute this pathway.
