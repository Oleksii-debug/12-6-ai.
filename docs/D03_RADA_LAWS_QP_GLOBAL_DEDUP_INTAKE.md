# D03 Rada laws Q/P → incumbent global-dedup intake

Status: `AUTHENTICATED_ONE_TO_ONE_PROJECTION_READY / CANONICAL_MATCH_NOT_EXECUTED`.

This package closes the missing source-specific seam between the real Rada laws
quality/privacy execution in PR #655 and the incumbent NEXT100-065 global-dedup
contract. It does not create a new matcher, change any threshold, aggregate source
chunks, or grant corpus/training credit.

## Exact upstream authority

The adapter is bound to the final real-execution evidence commit
`43a7d62854255da4426d6a3dc4d5ad9b1e4898e9` and LOCAL_FREE run
`34669418910` / job `103487804898`.

Observed, still-zero-credit payload:

- 101,559 accepted chunks;
- 192,078,166 accepted UTF-8 text bytes;
- 224,528,099 accepted JSONL transport bytes;
- accepted JSONL SHA-256
  `48324e41385e12a5b9f590bd478516643edb21992f1a5ca5ca2c9eca75f7ab12`;
- accepted metadata inventory SHA-256
  `afb940b48b5de52e598d8adf026c78cf5d51b228f1fbf56c9f0fbe472d946f65`;
- exact duplicate accepted hashes observed but intentionally not removed: 764;
- execution evidence identity
  `6f4952add1a2dabaf37170fac5ec2ffae9bdd3366b0a032c6a89886a8b898e74`.

The adapter verifies the candidate transport hash/size before parsing, the exact
quality-report file hash plus self identity, and the exact execution-evidence
self identity. Every accepted row is then cross-bound to the report metadata and
its UTF-8 bytes/SHA are recomputed.

## Projection semantics

Every accepted chunk becomes exactly one incumbent V3 source object:

- `source_family = ua.rada.open-data.laws-texts`;
- `modality = uk`;
- `stable_origin_id` preserves the parent law/document;
- `stable_object_id = sha256:<normalized_sha256>`;
- `origin_key` is chunk-specific, so sibling chunks are never incorrectly
  collapsed as origin aliases;
- declared/expected bytes and expected raw SHA are recomputed from the exact text;
- `evidence_status = DEDICATED_TERMINAL`.

The 764 observed exact-content aliases remain separate source objects with the
same stable-object identity. They are not pre-deduplicated. Exact/near/fragment,
lineage, connected-component, capacity and survivor decisions remain owned by the
incumbent matcher.

## Why canonical matching is still blocked

The direct incumbent V3 path is quadratic. Rada alone implies
5,157,064,461 unordered pair dispatches before existing corpus objects are added.
Silently dropping duplicates or grouping by parent would change the science and
is forbidden.

PR #1459 / issue #1764 owns the separate performance-equivalent executor repair.
This package therefore emits a deterministic text-free projection receipt and
exposes `delegate_to_incumbent_matcher(...)`, but does not automatically call the
quadratic matcher. Once the executor lineage is terminally qualified, it can be
supplied through that callback seam without changing this adapter or the matcher
semantics.

## Projection-only execution

With the exact PR #655 artifacts in local ephemeral storage:

```bash
python tools/prepare_d03_rada_laws_qp_dedup_intake.py \
  --candidate-jsonl /path/to/quality-accepted.jsonl \
  --quality-report /path/to/quality-report.json \
  --execution-evidence /path/to/d03-rada-bulk-quality-privacy-real-execution-v1.json \
  --upstream-head 43a7d62854255da4426d6a3dc4d5ad9b1e4898e9 \
  --output-receipt /tmp/rada-laws-qp-dedup-intake-receipt.json
```

The receipt contains no source text. A PASS proves authentication and projection
mechanics only.

## Scientific truth boundary

This package does **not** assert canonical dedup completion, corpus admission,
balance, split/packing, tokenizer-fit authority, optimized-target exposure,
optimizer updates, training, learned weights, or final-test access.

The following remain fail-closed:

- canonical capacity credited: 0;
- training-authorized bytes: 0;
- authorized optimized-target exposure: 0;
- tokenizer fit authorized: false;
- optimizer updates on real targets: 0;
- training executed: false;
- learned weights created: false;
- final-test outcomes read: false;
- paid compute used: false;
- foreign pretrained weights: false;
- external LLM/API used for data or intelligence: false.

Fresh exact-head shared CI and a different-worker audit are required before
integration. Real global matching must wait for terminal performance-equivalent
executor authority and must still consume all 101,559 projected source objects.
