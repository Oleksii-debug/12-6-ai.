# D03 Rada_Trees #958 authority remediation

Status: zero-credit Product repair for #975 `AUDIT958-001/002/003`.

## Why this successor exists

Merged #958 correctly reuses the incumbent quality-granularity and privacy APIs, but its low-level CLI accepts the expected candidate hash from the caller and copies the complete parent mapping into survivors. Independent audit #975 therefore blocked real Rada_Trees terminal promotion: a caller could present a self-consistent substituted candidate/hash pair, and an unexpected payload-bearing parent field could survive even though only `text` was privacy-scanned.

`tools/run_d03_rada_trees_quality_windows_authorized.py` is the canonical scientific-execution entrypoint for this lineage. It is deliberately a thin authority wrapper, not a second materializer or a new quality/privacy policy.

## Frozen upstream authority

The wrapper requires the exact canonical #916 handoff report whose self-hash was independently recomputed from specialist artifact `10128388325` by #974:

- handoff report SHA-256: `83b2cc636aa4cf9a754cd55534035c44782722d2b457755c00f6780112ebfffa`;
- source dataset/revision: `uacorpus/Rada_Trees@1b994a5804dcda122721e8d33a03fd172cf8d867`;
- archive SHA-256: `737fa5df061c55555a8c761023559a206cbcabf0299ae52d1ccae387f685803e`;
- rights config SHA-256: `34da44a047c5e0d562ee6a86987cb66e3ed266e1c1f09af95e63c38c219fe1a3`;
- rights report SHA-256: `7eea6d0b79353ef565910738dce9de93008a961059cab503b6f05686c0f27a7d`;
- accepted path inventory: `7b93056f38fbc87c11e14df9069066e21380db1f1b8d8016314330f841bfa6fc`;
- held path inventory: `566760e10157cd835ed0879abb37f052b57d31cff6af358a81ff717f4f7f59d9`;
- expected candidate records/source bytes: `4,384 / 877,899,128`.

The candidate SHA is read from that pinned report and then checked against the candidate file. There is no caller-supplied expected-candidate hash in the authorized runner.

## Closed candidate schema

Before the #958 delegate is called, every JSONL row must contain exactly the 20 fields emitted by #916. Unknown keys are rejected, including any accidental text/secret backup field. The runner also rechecks source family/dataset/revision/archive, rights/attribution and zero-credit flags, record-id to source-path/source-payload identity, session date, decoded-text hash/bytes, record uniqueness, aggregate record/source/decoded-byte counts, and whole-file candidate SHA/bytes.

Because the delegate only receives a candidate that has passed this exact closed schema, #958's internal `dict(parent)` cannot carry an unscanned extra parent field into a survivor.

## Privacy dependency

This repair does not copy, fork, or redefine privacy detectors. It delegates to `twelve_six.data.privacy_filter_v3` exactly as #958 does. `AUDIT958-002` is terminal only when this branch is late-bound to the canonical independently-audited privacy repair successor of #930/#955/#957. Until that repair is integrated into the branch ancestry, real ~878 MB Rada_Trees quality-window execution remains blocked.

## Operator invocation

```bash
PYTHONPATH=src python tools/run_d03_rada_trees_quality_windows_authorized.py \
  --candidate-jsonl /path/rada-candidate.jsonl \
  --upstream-handoff-report /path/rada-handoff-report.json \
  --output-jsonl /path/rada-quality-windows.jsonl \
  --materialization-report /path/rada-quality-windows-report.json \
  --authority-report /path/rada-quality-windows-authority.json
```

The authority report is deterministic, text-free and self-hashed. It binds the terminal #916 report/candidate to the delegated #958 report/output while preserving zero training/evaluation credit.

## Scientific boundary

This package itself authorizes `0` training bytes, `0` unique causal-loss positions and `0` optimizer updates. It does not fit a tokenizer, train a model, access final-test payloads/outcomes, or use paid compute. Real Rada execution and downstream dedup/decontamination/balance/split/packing authority must remain fail-closed until all current gates are terminal.
