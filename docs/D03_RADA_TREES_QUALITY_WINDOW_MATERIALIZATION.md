# D03 Rada_Trees quality-window materialization

## Purpose

This package is a narrow successor to the incumbent Rada_Trees quality/privacy gate on PR #916. That gate intentionally keeps every `RETAIN_PARTIAL` document on hold because accepted #840 quality windows are not materialized. This package closes only that capacity-loss seam. It does not define a second quality policy, privacy detector, source materializer, dedup engine, or corpus authority.

Implementation was first reconstructed against `#916@e13aa59de5d1e18e851601b89c13977314a491ea`. Before Product mutation, the empty successor branch was rebound to the then-current exact parent `#916@36bef7281fc4a05aedb005ac15b454bbd3a8ef63`. PR #930 is a separate current-main convergence of the frozen #840/#849 mechanics and is not modified by this package.

## Contract

`tools/materialize_d03_rada_trees_quality_windows.py` consumes the same zero-credit candidate JSONL emitted by the #916 rights-scoped handoff. It performs a two-pass LOCAL_FREE execution:

1. hash the complete candidate as a stream and require the exact expected upstream SHA-256 before loading or executing policy mechanics;
2. stream JSONL records one at a time rather than loading the approximately 878 MB candidate into RAM;
3. verify source family/dataset/revision/archive, rights scope, attribution, zero-credit state, and each decoded-text SHA-256/UTF-8-byte binding;
4. bind the live frozen #840 granularity-policy SHA-256 and #849 privacy-policy SHA-256;
5. call `twelve_six.data.quality_granularity.apply_frozen_granularity()` without redefining thresholds;
6. for `RETAIN_PARTIAL`, require ordered, contiguous, authoritative windows that exactly reconstruct the parent and reproduce retained/rejected byte accounting;
7. reconstruct only quality-accepted windows;
8. call `twelve_six.data.privacy_filter_v3.hash_safe_scan()` on every actual retained unit and independently verify the scan-input SHA/byte binding;
9. emit only `ALLOW` units; `REDACT`, `QUARANTINE`, and `EXCLUDE` remain held because this package does not invent privacy mutation semantics;
10. enforce exact conservation: input decoded bytes = quality retained + quality rejected, and quality retained bytes/units = privacy ALLOW + HOLD bytes/units;
11. publish a deterministic text-free self-hashed report plus the local zero-credit survivor JSONL.

For `RETAIN_ALL`, the source-native document remains one byte-identical payload unit. `REJECT_DOCUMENT` reaches no privacy scan and emits no payload. A partial survivor gets a deterministic record ID binding parent ID, quality-window index, and full unit SHA-256. The generic decoded-text identity is rebound to the emitted unit while the parent decoded-text identity remains in dedicated parent fields. Each survivor also records the exact quality-policy, privacy-policy, and privacy-scan input identities.

## Mutation safety

Candidate, output, and report paths must be distinct. Candidate/output/report symlink aliasing is rejected. Existing output or report files are never silently overwritten. The survivor JSONL and report are first written to partial paths; failed validation removes partial artifacts, and a report-publication failure removes the newly published output rather than leaving a half-published pair.

## Authority boundary

This package does not bind the independent #917 language decision to the new survivor inventory, so `language_quality_privacy_complete=false` remains mandatory. It does not run expanded global dedup, fresh reserved-evaluation decontamination, family caps, cluster-safe split, tokenizer/packing, or unique-loss accounting.

Every survivor remains `training_eligible=false` and `evaluation_eligible=false`. The report keeps `training_authorized_bytes=0`, `unique_causal_loss_positions_authorized=0`, `optimizer_updates=0`, no model training, no final-test payload access, and no paid compute.

Privacy evidence is fail-closed: nested payload-bearing fields such as raw text, previews, matched values, generic `value`, or match hashes are rejected. Durable report content is aggregate/hash-safe only; survivor text exists only in the operator-selected local JSONL and is not committed.

## LOCAL_FREE validation

Focused/adversarial tests cover accepted-window reconstruction, per-unit privacy scanning, non-ALLOW holds, `RETAIN_ALL`, `REJECT_DOCUMENT`, partition and authority drift, privacy-input binding, unsafe nested evidence, boolean-count traps, decoded-text identity drift, duplicate IDs, exact candidate-SHA mismatch before mechanics load, path alias/symlink/overwrite prevention, deterministic output, report self-hashing, and exact-branch composition of the incumbent APIs.

The isolated Work harness completed `15 passed` with the repository-composition test deselected because the Work directory does not contain the `twelve_six` package. The composition test is intentionally retained for exact stacked-branch CI; that CI must not be reported PASS until it actually finishes on the branch.

Repository-root command:

```bash
pytest -q tests/test_d03_rada_trees_quality_window_materialization.py
```

Real-candidate command, only after the exact #916 handoff is available locally:

```bash
python tools/materialize_d03_rada_trees_quality_windows.py \
  --candidate-jsonl /path/to/rada-trees-rights-scoped.jsonl \
  --expected-candidate-sha256 <sha256-from-exact-upstream-handoff> \
  --output-jsonl /path/to/rada-trees-quality-window-survivors.jsonl \
  --report /path/to/rada-trees-quality-window-report.json
```

The current Work container does not possess the 697,768,591-byte source archive or the resulting approximately 877.9 MB rights-scoped candidate. Therefore no real Rada_Trees quality/privacy result is claimed here, and this engineering package grants zero corpus/training credit.

## Next scientific successor

After real materialization, bind the exact output/report identities and converge #917 language authority onto that exact survivor inventory. Only then should the expanded current global exact/near/lineage dedup and fresh reserved-evaluation decontamination consume the new Rada_Trees survivors. No learned-20M launch authority follows from this package alone.
