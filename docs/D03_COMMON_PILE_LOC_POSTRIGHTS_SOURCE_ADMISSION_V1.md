# D03 Common Pile Library of Congress post-rights source admission v1

This package consumes two already-qualified facts that previously stopped short of each other:

1. the exact real LOCAL_FREE Library of Congress execution retained by PR #1276; and
2. the exact conditional LoC source-rights authority from PR #1162, independently audited by #1167.

It performs the missing post-hoc conjunction for that exact historical candidate. It does not
download or rematerialize the 358,594,502-byte shard, and it does not grant canonical corpus or
training credit.

## Exact real execution

The retained execution authority is:

- Product PR #1276 final evidence head
  `cdde55b75a10639e8233de7ccccaaea335b4055a`;
- real execution head `8ce516160b91ba48b4dd4096e0ffeea409d221de`;
- shared-CI run `34606219695`, job `103285166822`;
- dataset `common-pile/library_of_congress`;
- revision `d31bdba02cdad5104ccec2c02ae799c0bcb5a9a7`;
- shard `data/00000_loc_books.jsonl.gz`;
- compressed bytes `358594502`;
- shard SHA-256
  `a6a6023d9cae067b5531ba84537877c180ab3ff1579f99aeac446d65d465ae0f`;
- two independent full-shard acquisitions and byte-identical materializations;
- 256 examined records, 28 retained records, and 4,499,908 retained normalized UTF-8 bytes;
- candidate SHA-256
  `f87201d71eb8c3f2510cdf2c9b3ce1a32b83d1a11198e8a1ff0ad67d0a3fec20`.

The post-rights verifier binds the exact execution receipt plus the exact materializer, intake
configuration, and intake module blobs. It also resolves those three mechanics files at the
historical execution head with `git show`, so a later current-main substitution cannot silently
rewrite what actually ran.

## Exact rights authority

The source-rights authority is the merged PR #1162 Product head
`7e096ffcfd17f6b0eb215037a60370f255202c02`, merge commit
`3ee45353424991e471b749f807f231cad486a797`, with independent audit #1167
`PASS_FOR_INTEGRATION_MECHANICS`.

The authority permits conditional source admission only for the exact audited lineage:

- `source == "loc_books"`;
- `license == "Public Domain"`;
- `language == "english"`;
- record IDs acting as LCCNs;
- exact `https://www.loc.gov/item/{lccn}` item URLs;
- OCR text URLs hosted only by `tile.loc.gov` or `tiles.loc.gov`;
- the exact Selected Digitized Books / audited Common Pile dataset, revision, and shard.

The project does not claim a legal opinion. Package-level dataset or repository licensing is not
substituted for record-level provenance.

## What this package proves

The validator requires all of the following simultaneously:

- exact retained real-execution evidence;
- exact historical execution-code blobs;
- exact source shard identity and byte count;
- exact candidate identity, record count, retained byte count, inventory identity, and intake
  contract identity;
- exact current source-rights policy/module identities and repository bindings;
- exact alignment between the executed intake contract and the later-qualified rights contract.

A successful validation therefore permits only the statement that the exact historical LoC
candidate is source-admitted for downstream canonical processing.

## What this package does not prove

Source admission is deliberately zero-credit. It does **not** mean:

- canonical corpus capacity has increased;
- any family/balance credit has been granted;
- tokenizer fitting is authorized;
- any unique causal-loss position is authorized;
- optimizer step 1 is authorized;
- training has executed;
- learned weights exist;
- evaluation or final-test data may be read.

The exact candidate still must pass current global exact/near/lineage dedup, reserved-evaluation
decontamination, canonical quality/privacy, family balance caps, cluster-safe split, deterministic
packing, two-clean-build reproducibility, positive unique-loss accounting, and positive training
authority.

## Verification

From the repository root:

```text
python tools/validate_d03_common_pile_loc_postrights_admission.py
pytest -q tests/test_d03_loc_postrights_admission.py
```

Shared exact-head CI remains authoritative. A different worker must independently audit the exact
Product head before the current integration owner consumes it.

## Scientific truth boundary

At this package boundary:

- real LoC payload execution: true, inherited from retained run `34606219695`;
- source-admitted candidate records: `28`;
- source-admitted candidate normalized bytes: `4,499,908`;
- canonical capacity credited: `0`;
- training-authorized bytes: `0`;
- authorized unique loss positions: `0`;
- authorized optimized target exposure: `0`;
- tokenizer fit authorized: false;
- optimizer updates on real targets: `0`;
- training executed: false;
- learned weights created: false;
- final-test payload/outcomes accessed: false;
- paid compute used: false;
- foreign pretrained weights used: false;
- external LLM/API corpus intelligence used: false.
