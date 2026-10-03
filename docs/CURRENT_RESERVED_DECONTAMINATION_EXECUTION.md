# Current reserved-evaluation decontamination execution carrier

## Purpose

`tools/run_current_reserved_decontamination_v1.py` is the operator carrier for the
already-canonical `current_reserved_decontamination_v1` adapter. It does not implement
a second matcher or a new decontamination policy. It exists to make the transition from
a terminal current post-dedup training payload plus independently bound reserved
evaluation payloads to reproducible, code-bound, hash-only execution evidence.

This carrier is **execution mechanics only**. Its presence in the repository does not
mean that the current expanded corpus, tokenizer fit, optimized-target exposure, or
training is authorized or has happened.

## Required authority before launch

Launch only when all of the following are independently fixed outside the runner:

- the exact prequalified Git commit containing the decontamination implementation;
- a current training JSONL whose rows exactly match the canonical post-dedup matcher
  projection;
- the matching `12-6.postdedup-decontam-handoff.v1` evidence and independently expected
  inventory, survivor-authority, and handoff identities;
- resolved selection-validation and final-test payload rows used only for overlap
  detection, plus a `12-6.reserved-evaluation-payload-binding.v1` that independently
  binds those rows;
- independently expected reserved-binding, selection-validation, and final-test
  identities.

Do not derive an "expected" implementation SHA from the checkout at launch time. The
value passed to `--expected-decontamination-implementation-git-sha` must come from the
prequalified authority chain. The runner checks that exact SHA before reading any
payload-bearing input and also rejects tracked working-tree drift.

## Invocation

```text
python tools/run_current_reserved_decontamination_v1.py \
  --repo-root . \
  --training-records-jsonl <ephemeral-training.jsonl> \
  --training-handoff-json <postdedup-handoff.json> \
  --evaluation-records-jsonl <ephemeral-reserved-evaluation.jsonl> \
  --reserved-binding-json <reserved-binding.json> \
  --output-dir <new-output-directory> \
  --expected-inventory-identity-sha256 <64-hex> \
  --expected-survivor-authority-sha256 <64-hex> \
  --expected-training-handoff-identity-sha256 <64-hex> \
  --expected-reserved-binding-identity-sha256 <64-hex> \
  --expected-selection-validation-identity-sha256 <64-hex> \
  --expected-final-test-identity-sha256 <64-hex> \
  --expected-decontamination-implementation-git-sha <40-hex>
```

The production carrier always keeps cross-source-family quarantine enabled. There is no
CLI bypass.

## Publication contract

The output directory must not already exist. The carrier fully executes and verifies the
canonical adapter in memory first, builds all output bytes, then publishes one directory
rename containing exactly:

- `decontamination_report.json` — canonical DATA-232 hash-only report;
- `execution_evidence.json` — canonical current-decontamination execution evidence;
- `execution_receipt.json` — self-hashed runner receipt binding the exact implementation
  Git SHA, SHA-256 of all four physical input files, SHA-256 of the two canonical output
  files, and all canonical logical identities.

The receipt is the bundle publication marker. A report/evidence pair copied without the
matching receipt is not this carrier's complete execution artifact. Raw text and record
IDs are not intentionally persisted in the bundle; the receipt stores only hashes and
logical authority identities.

## Scientific and privacy boundary

A successful run truthfully records that final-test **payload text** was accessed for
contamination matching while final-test **outcomes** were not read. It does not authorize
model selection from final-test data. It does not authorize tokenizer fitting, positive
training exposure, optimizer updates, training, paid compute, or learned weights.

The payload-bearing training/evaluation files are operational inputs and should remain
ephemeral under the existing evaluation firewall. Only the three hash-only bundle files
are intended for durable repository/evidence handling. A real execution still requires
fresh independent verification and the canonical integration authority before any
upstream readiness state may consume it.
