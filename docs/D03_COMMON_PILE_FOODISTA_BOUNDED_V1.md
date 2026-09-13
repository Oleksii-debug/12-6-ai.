# D03 Common Pile Foodista bounded intake v1

This package adds one bounded, LOCAL_FREE, zero-credit candidate intake for Common Pile source key `foodista`. It reuses the merged `COMMON-PILE-SOURCE-RIGHTS-V1` authority and does not create a bulk Common Pile ingestion path.

Exact immutable payload authority is `common-pile/foodista@04d1b7a6562c6d6459426d2a3b88184b8a98f7b6`, shard `v0/documents/00000_foodista.jsonl.gz`, SHA-256 `286b801bc826f161efad892af5529b201f91f42c5a932e3962d74d24037fe087`, compressed bytes `6,760,466`. The audited upstream collector is `r-three/common-pile@9457f04a14cb2355ab00023420369d46ffd4a395`; `sources/food/README.md` is pinned by Git blob `155d861c68194b68a18df1fcfa3c32e9105c645d`.

The materializer accepts only exact `foodista` rows whose per-record metadata carries the frozen Creative Commons Attribution 3.0 string, a Foodista HTTPS origin, and the expected Common Pile provenance form. Metadata is never emitted as candidate training text. Structural drift, authority drift, obvious contact/secret/control markers, duplicate IDs, exact-normalized duplicate text, and quality failures are rejected or accounted fail-closed.

The candidate is capped below 4.8 MB normalized UTF-8 and remains `training_eligible=false` / `evaluation_eligible=false`. This package grants zero canonical bytes, family credit, tokenizer-fit authority, optimized loss positions, optimizer updates, or final-test access. Real source execution, if separately performed, is candidate-only evidence until global cross-source dedup, reserved-evaluation decontamination, final source-rights review, post-composition quality/privacy, family/balance controls, cluster-safe split, deterministic two-clean-build packing, and positive unique-loss accounting all terminalize.
