# NEXT100-065F exact-hash prefilter

This package is a narrow successor checkpoint for the incumbent #632 global-dedup semantics. It consumes the exact terminal text-free V7 artifact from #632 and the exact terminal text-free source report from merged #818.

It proves only one useful fact: across the 35 historical V7 records and all 229 new permissive-Python records, there are zero raw SHA-256 byte-exact intersections, and the 229 new records contain no internal raw SHA-256 duplicates.

This is **not** terminal global dedup. The V7 matcher still has to execute on the composed payload graph to cover common normalization, fragment containment, near-copy similarity, code-copy/skeleton semantics, and lineage-connected components. Therefore post-dedup capacity remains unclaimed, tokenizer fitting and learned-20M remain blocked, and training exposure remains zero.

All durable data in this package are text-free hashes/identities and byte counts. No final-test payload or paid compute is used.
