# D03 CulturaX UA/EN bounded acquisition successor

## Purpose

This package turns the terminal AUDIT-B CulturaX handoff (#747 / PR #762) into an
executable, current-main bounded acquisition contract without pretending that gated
payloads, source-level rights, privacy review, or training admission have already happened.

The current authority is deliberately **blocked**. CulturaX exposes Ukrainian and English
data and preserves `text`, `timestamp`, `url`, and `source`, but the dataset is gated and
inherits mC4/OSCAR/Common-Crawl rights/privacy constraints. Public file listings or an
upstream dataset/checksum label are not canonical 12-6 training authority.

## What is executable now

`src/twelve_six/culturax_bounded_acquisition.py` provides:

- strict validation of the exact current claim/audit/dataset-card identities;
- parsing of per-language `checksum.sha256` manifests;
- deterministic selection of exactly one lexicographically first shard from each of `en`
  and `uk`, never an accidental bulk acquisition;
- a stable bounded-plan SHA-256 identity;
- post-download receipt verification against the selected upstream SHA-256 values;
- exact per-record provenance validation for `text`, `timestamp`, `url`, and the original
  mC4/OSCAR lineage.

The validator can be run before access is granted:

```text
PYTHONPATH=src python tools/validate_d03_culturax_bounded_acquisition.py
```

That must report `VALID BLOCKED_GATED_ACCESS_ACCEPTANCE_REQUIRED`.

After the repository owner has independently accepted the upstream gated-dataset terms and
provided an authorized Hugging Face access context, a successor execution may save the exact
`uk/checksum.sha256` and `en/checksum.sha256` snapshots and build a frozen plan:

```text
PYTHONPATH=src python tools/validate_d03_culturax_bounded_acquisition.py \
  --uk-checksum /path/to/uk-checksum.sha256 \
  --en-checksum /path/to/en-checksum.sha256 \
  --output /path/to/culturax-bounded-plan.json
```

This step still authorizes **zero** training bytes.

## Non-negotiable downstream gates

Any downloaded records remain quarantined until source-level rights classification, project
privacy/PII review, reserved-evaluation decontamination, project-global cross-source dedup,
quality, balance/family caps, deterministic split, packing, and a positive unique-loss ledger
all become terminal on exact immutable artifacts.

A downloaded shard is not accepted merely because its transfer completed. Its independently
computed project SHA-256 must exactly match the selected checksum entry, and record provenance
must not be collapsed from mC4/individual OSCAR release identities into a generic `CulturaX`
source label.

## Truth boundary

At this package head:

- payload download: not executed;
- gated access acceptance/credentials: not observed by this Work;
- training-authorized bytes: 0;
- authorized unique causal-loss positions: 0;
- tokenizer fit: unauthorized;
- optimizer/model updates: 0;
- final-test payload access: false;
- paid compute: false.

The blocker requiring owner action is only the external gated-access acceptance/credential.
All repository-side machinery needed to freeze a bounded checksum-selected acquisition is
implemented here.
