# D03 incumbent dedup indexed execution v1

## Purpose

The terminal NEXT100-065 V3 matcher performs an all-pairs dispatch to the incumbent V1 `_pair_matches` function. That is scientifically simple but not a bounded LOCAL_FREE execution path for the 101,559 accepted Rada-law chunks: Rada alone implies 5,157,064,461 unordered pair dispatches, and Rada plus the current 262 V8 survivors implies 5,183,707,110.

This package changes **execution mechanics only**. It does not define a new duplicate relation, threshold, normalization, cluster rule, capacity policy, survivor policy, or lineage rule.

## Safety contract

`incumbent_dedup_indexed_execution.py` binds execution to the exact terminal V3/V1 Git blobs used by the incumbent authority. It validates exact built-in positive threshold types before pruning. Candidate generation is a conservative union of necessary conditions for every V1 pair predicate:

- equal `origin_key`;
- equal raw SHA-256;
- equal normalized SHA-256;
- a shared content shingle within the natural/code modality class;
- a shared code-skeleton shingle for code-copy candidates;
- a shared normalized qualifying edge line for publisher-boilerplate candidates.

Every retained pair is still passed to the exact incumbent `v1._pair_matches`. V3 stable-object and explicit-lineage matches remain delegated to exact `v3._lineage_matches`; capacity/independence summaries remain delegated to exact `v3._summary_for_ids`.

If runtime source identity drifts, a relevant threshold becomes non-positive or type-coerced, the candidate budget is exceeded, or payload coverage differs, execution fails closed. The package never silently switches thresholds or drops a known candidate.

## Differential qualification

`tools/verify_incumbent_dedup_indexed_equivalence.py` runs both paths on the same inventory/payload bytes and requires byte-identical canonical reports, including identical report SHA-256. It emits only text-free execution telemetry. This is the required qualification path on an authority-bound V3 runtime.

Focused local unit tests cover every candidate family, modality separation, bounded failure, and the exact Rada theoretical pair counts. These mechanics tests are not a substitute for exact-runtime differential qualification.

## Scientific boundary

- LOCAL_FREE only.
- No source admission or canonical capacity credit.
- No dedup science change.
- No tokenizer fit or optimizer update.
- No model training or learned weights.
- No final-test payload access.
- No paid compute, foreign pretrained weights, or external LLM/API corpus intelligence.

PR #655 qualification/integration, #1448 source-specific Rada adapter ownership, and #1363 integration authority remain separate. Real Rada global-dedup execution becomes lawful only when those upstream authorities and an independently qualified exact-runtime differential receipt are terminal.
