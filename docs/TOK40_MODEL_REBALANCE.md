# TOK-40 tokenizer-bound model geometry rebalance

Status: experimental integration layer. This does not freeze a tokenizer, promote a model, or modify a canonical stage config.

## Purpose

A vocabulary change changes a substantial trainable parameter surface in small models. TOK-40 therefore treats the actual tokenizer artifact as an architecture input. The solver consumes an exact target parameter budget, the actual tokenizer vocabulary cardinality plus exact serialized tokenizer SHA-256, and an explicit depth/head search envelope. It derives `d_model` from valid head geometry and delegates `d_ff` retargeting to the existing MODEL37 vocabulary allocation solver.

No second parameter solver is introduced. `twelve_six.model_rebalance.search_model_geometry` calls `twelve_six.vocabulary.rebalance_d_ff_for_vocabulary` for the exact `d_ff` budget solution.

## Tokenizer identity contract

For a raw tokenizer JSON, `TokenizerArtifactIdentity.from_artifact` hashes the exact bytes and sizes the embedding by the token-id surface. When token ids are present, cardinality is `max(token_id) + 1`, including added tokens; it is not inferred from the requested trainer vocabulary size.

The controlled MODEL37 BPE mechanics artifact retained the exact tokenizer serialization hash but not the raw tokenizer JSON in its uploaded evidence bundle. The TOK-40 checked-in descriptor therefore binds the stage table to that retained exact hash:

- actual vocabulary: 472
- tokenizer JSON SHA-256: `006c84fc0d05d3bedb5b0bceb587aab1631dd0295cc2063e97823c2121e08be0`
- tokenizer artifact identity SHA-256: `cf44fb576e37c9bcd0bb80b3a26793c4976035875248900b9c3b4d14fe37db74`
- source evidence SHA-256: `3307daec835e96a63fd5a7d14543de3d9f0781ec3f7d4d0cc98d992bf0f8bae6`

This is a mechanics anchor, not a tokenizer freeze or a representative-corpus quality result.

## Geometry validity and budget policy

A candidate is valid only when:

- `d_model = n_heads * head_dim`;
- `n_heads` is divisible by `n_kv_heads`;
- `head_dim` is even for RoPE and the searched rotary dimension equals `head_dim`;
- `d_ff / d_model` is inside the configured interval, currently 2.0 through 4.0;
- the exact parameter delta is inside the configured target tolerance, currently 3%;
- the tied embedding allocation is at most 30% of the resulting model budget.

The 30% ceiling is an engineering rejection threshold, not a claim that 30% is optimal. It is deliberately just above the approximately 23-26% tied vocabulary allocation already observed in the small MODEL37 stage analysis, so vocabulary growth cannot silently consume most of a small model. Every candidate separately reports both `embedding_fraction` and total `vocabulary_fraction`; under the current canonical tied-weight Base geometry these are identical. An untied-head campaign should lower or reinterpret the search envelope explicitly rather than inherit this tied-only operating assumption silently.

When KV-head constraints are omitted, TOK-40 preserves MHA by setting `n_kv_heads = n_heads`. GQA is a separate research line and is not silently introduced by vocabulary rebalancing.

## Bound ModelSpec identity

The existing `ModelSpec.identity_sha256()` remains unchanged for compatibility. TOK-40 adds `bound_modelspec_identity_sha256`, whose canonical payload contains the complete ModelSpec, its existing identity, the actual tokenizer vocabulary cardinality, the exact tokenizer JSON SHA-256, and the tokenizer artifact identity SHA-256. Two byte-different tokenizer artifacts therefore produce different bound identities even if their vocabulary cardinality is equal.

This is additive and does not rewrite canonical ModelSpec v1 semantics.

## Stage-specific candidate table

The retained machine table is `configs/vocabulary/model_rebalance_stage_candidates.v1.json`. It is checked by tests against live solver output so a stale table fails CI.

| Target | Top geometry `(L,H,HD,D,FF)` | Exact params | Delta | Embedding share | Block share |
| --- | --- | ---: | ---: | ---: | ---: |
| 100K | `(3,4,12,48,112)` | 99,024 | -976 | 22.8793% | 77.0722% |
| 250K | `(4,4,16,64,200)` | 249,920 | -80 | 12.0871% | 87.8873% |
| 500K | `(4,4,24,96,264)` | 497,760 | -2,240 | 9.1032% | 90.8775% |
| 1M | `(4,4,32,128,440)` | 999,552 | -448 | 6.0443% | 93.9429% |
| 10M | `(10,8,32,256,944)` | 9,997,568 | -2,432 | 1.2086% | 98.7888% |

The JSON retains all candidates surviving each deliberately small search envelope, not just these first-ranked rows.

## API and CLI

API entry points:

- `TokenizerArtifactIdentity.from_artifact(path)`
- `GeometryConstraints(...)`
- `search_model_geometry(base_spec, target_parameters=..., tokenizer=..., constraints=...)`
- `build_stage_candidate_table(...)`
- `one_training_step_smoke(candidate, ...)`

Installed CLI:

```text
twelve-six-rebalance search \
  --stage-config configs/stages/s1_100k.json \
  --tokenizer-artifact path/to/tokenizer.json \
  --target-parameters 100000 \
  --layers 2,3,4 \
  --heads 2,4 \
  --head-dims 8,12,16 \
  --output model-search.json
```

The five-target table can be regenerated from the retained exact tokenizer identity:

```text
twelve-six-rebalance stage-table \
  --profiles configs/vocabulary/model_rebalance_profiles.v1.json \
  --tokenizer-identity configs/vocabulary/measured_bpe_472_tokenizer_identity.v1.json \
  --output model-rebalance-stage-table.json
```

## Executable evidence

`tests/test_model_rebalance.py` performs the following integration checks:

- exact tokenizer byte hash and token-id cardinality handling;
- tokenizer hash changes bound ModelSpec identity;
- the known MODEL37 472-vocabulary S1 rebalance is recovered rather than replaced by a second algorithm;
- unreasonable small-model vocabulary allocation fails closed;
- all required 100K, 250K, 500K, 1M and 10M targets produce valid candidate sets;
- the checked-in stage table matches live solver output;
- representative 100K and 1M winners instantiate the real `TwelveSixDecoder`, run a causal cross-entropy backward pass, execute exactly one optimizer step, and prove a real parameter changed.

The construction smoke uses project code and random initialization only. It is not a quality-training claim.

## Migration rule

A tokenizer change must not be implemented by editing only `vocab_size` in a canonical model config. Experimental migration is:

1. hash and inspect the actual tokenizer artifact;
2. derive actual token-id cardinality;
3. run TOK-40 against an explicit target parameter budget and explicit geometry envelope;
4. reject over-budget vocabulary candidates;
5. select a searched candidate only through experiment evidence;
6. persist both the normal ModelSpec identity and tokenizer-bound ModelSpec identity;
7. promote a new canonical config only through the project stage-promotion process.

Existing canonical configs remain byte-for-byte outside the TOK-40 change set.
