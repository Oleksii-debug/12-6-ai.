# Plan 2 / Section 8 — Reserved evaluation firewall

## Source and trust boundary

Section 8 consumes **the accepted Section-7 near-dedup survivor set**. It never
creates a second exact, near, privacy or evaluation-overlap matcher. Each
inspection re-executes Section 7 (including upstream S6/S5/S4 verification),
then calls the incumbent `twelve_six.data.decontamination_authority_v2`
DATA-232 matching and verifies its hash-only output.

A separate, pinned, project-authored synthetic holdout is committed at
`configs/data/plan2_reserved_eval_fixture_v1.json`. It provides one
`selection_validation` and one `final_test` source and is reserved **before
packing**. The exact Git blob is pinned in the adapter. This is a local/free
engineering holdout and must **not** be represented as a real benchmark,
confidential final test, model training dataset or production-ready release.

## Policies

- Reject changed or unreviewed reserved bytes, duplicate JSON keys, alternate
  roles, new fields, evaluation outcomes or missing members.
- The two reserved roles never become training material. Any S7 source record
  with raw/normalized/near/fragment/cross-source contamination detected by
  DATA-232 is excluded before downstream training/tokenizer consideration.
- All explicitly teacher or self-generated rows are quarantined by default,
  even if the current fixture does not show an overlap: absent independent
  provenance, generation carries potential evaluation-answer contamination.
- Report only record IDs, SHA-256 manifests, exclusion counts and authority
  bindings; never publish source content, holdout text, evaluation outcomes,
  model hyperparameter decisions or trained weights.
- Immutable publication rejects drift and symlink destinations; rerun and
  independent fresh-build results must match exactly.
- `training_corpus_authorized`, `tokenizer_fit_authorized`, and
  `evaluation_authorized` remain false. No paid compute is permitted here.

## LOCAL_FREE qualification

```sh
PYTHONPATH=src:. pytest -q tests/test_plan2_eval_firewall_v1.py
ruff check tools/plan2_eval_firewall_v1.py tests/test_plan2_eval_firewall_v1.py
python tools/plan2_eval_firewall_v1.py --out-dir /tmp/plan2-eval-firewall
python tools/plan2_eval_firewall_v1.py --out-dir /tmp/plan2-eval-firewall
```

Section-8 component closure does not authorize a real corpus, learned model,
secret eval disclosure, external teacher content, paid GPU or deployment.
