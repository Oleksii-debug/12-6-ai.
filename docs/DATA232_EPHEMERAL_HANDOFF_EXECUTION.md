# DATA-232 ephemeral post-dedup handoff execution

## Purpose

`tools/prepare_data232_ephemeral_handoff_v1.py` is the operator bridge between the
already-selected retained inventory from PR #940 and the existing current reserved-
evaluation decontamination carrier. It does not select survivors, rerun deduplication,
change DATA-232 matching policy, or grant corpus/training authority.

The carrier consumes the exact text-free
`12-6.expanded-postdedup-inventory.v1` plus the corresponding physical payload JSONL.
It delegates all logical payload/inventory validation to the exact authenticated PR #940
`prepare_ephemeral_data232_rows()` implementation and emits the exact
`12-6.postdedup-decontam-handoff.v1` contract consumed by the current DATA-232 adapter.

## Authority required before launch

The operator must independently retain all three identities below. Do not derive an
"expected" value from the candidate files or checkout being verified.

1. the audited exact carrier Git SHA;
2. the audited PR #940 retained-inventory implementation Git SHA;
3. the exact retained-inventory identity SHA-256 published by the upstream authority.

The CLI re-executes itself under Python `-I -S` before any authoritative work. This
removes user-site, `PYTHONPATH`, `.pth`, and site-startup import influence from the
actual carrier process. The authenticated process then proves that its own resolved
`__file__` is exactly `tools/prepare_data232_ephemeral_handoff_v1.py` inside the supplied
repository and that the physical carrier bytes are exactly the bytes tracked at the
independently expected carrier commit. A copied or tampered runner cannot point at a
clean checkout and borrow that checkout's Git identity.

The retained-inventory SHA must be an ancestor of the carrier checkout. The carrier
also proves that `src/twelve_six/data/expanded_postdedup_inventory_v1.py` contains no
symlink/path substitution, is unchanged from the independently expected PR #940
authority, and is physically byte-identical to the exact tracked Git blob. Only after
those gates pass does it compile those verified source bytes into a fresh private
namespace and retrieve `prepare_ephemeral_data232_rows()` from that namespace. It does
not trust a callable imported before the gate, so preloaded or monkeypatched canonical
module state cannot mint trusted handoff evidence.

All code/authentication gates run before inventory or payload bytes are read.

## Input contract

`--inventory-json` is the text-free retained inventory. `--payload-jsonl` contains one
JSON object per retained record with at least `record_id` and `text`. The canonical
PR #940 function requires exact complete record coverage and checks every text's UTF-8
byte count and SHA-256 against the independently bound inventory. Missing, extra,
duplicate, or tampered rows fail closed.

The physical payload file is hashed while it is streamed into canonical validation; it
is not read into a second whole-file buffer merely to compute evidence.

## Invocation

```text
python tools/prepare_data232_ephemeral_handoff_v1.py \
  --repo-root . \
  --inventory-json <retained-inventory.json> \
  --payload-jsonl <retained-payload.jsonl> \
  --output-dir <new-ephemeral-workspace> \
  --expected-inventory-identity-sha256 <64-hex> \
  --expected-carrier-git-sha <40-hex> \
  --expected-retained-inventory-git-sha <40-hex>
```

The user does not need to add `-I -S`; the tracked CLI performs one isolated re-exec
itself and refuses authoritative `main()` execution outside that isolation boundary.

The output directory must not exist. A successful run publishes the directory with a
no-replace atomic rename and contains exactly:

- `training_records.jsonl` — payload-bearing canonical matcher rows;
- `training_handoff.json` — canonical `12-6.postdedup-decontam-handoff.v1`;
- `execution_receipt.json` — self-hashed, text-free receipt.

The first two files are the direct `--training-records-jsonl` and
`--training-handoff-json` inputs of the existing current reserved-decontamination
runner. The payload-bearing JSONL is an ephemeral execution artifact and must not be
committed as durable repository evidence.

## Receipt and truth boundary

The receipt binds the exact carrier/upstream Git identities, physical input-file hashes,
physical emitted JSONL/handoff hashes, retained inventory and survivor identities,
handoff identity, record count, retained payload bytes, and serialized training-file
bytes. The receipt contains neither raw text nor record IDs. Its nested physical hash
maps are closed world: inputs must contain exactly `retained_inventory_json` and
`payload_rows_jsonl`; outputs must contain exactly `training_records.jsonl` and
`training_handoff.json`. Missing, renamed, or extra entries are rejected even if a
caller recomputes the receipt self-hash.

A successful carrier run proves only exact materialization of an already-selected
post-dedup payload into the canonical decontamination input contract. It does not prove
reserved-evaluation decontamination is complete, does not grant corpus credit or
tokenizer-fit authority, and does not authorize or execute an optimizer step.

The carrier also does **not** establish whether the current corpus is external-LLM
clean. The receipt deliberately has no global
`external_llm_or_api_used_for_data_or_intelligence=false` assertion. Instead it carries
only the scoped fail-closed statement
`current_corpus_external_llm_free_claimed_by_this_carrier=false`. That value is fixed by
the verifier and cannot be changed to `true` by resealing the receipt. Any future
positive clean-corpus statement must come from a separate independently authenticated
physical successor authority; this carrier does not invent or infer that authority.

The remaining fail-closed receipt boundary is:

- authorized optimized-target exposure: `0`;
- tokenizer fit authorized: `false`;
- optimizer updates on real targets: `0`;
- training executed: `false`;
- learned weights created: `false`;
- final-test outcomes read: `false`;
- paid compute used: `false`;
- foreign pretrained weights: `false`;
- current-corpus external-LLM cleanliness claimed by this carrier: `false`.

The merged Nomis1864/Sonnet quarantine guard prevents known contaminated authority from
being treated as launch-authoritative, but it does not prove all remaining corpus bytes
clean. A physical clean successor corpus and its independently authenticated authority
therefore remain prerequisites before any positive clean-corpus claim, tokenizer-fit
authority, optimized-target exposure, or training launch.

After this bridge runs, the current reserved-evaluation decontamination carrier remains
the next scientific gate; its final-test payload access is for overlap detection only,
while final-test outcomes stay unread.
