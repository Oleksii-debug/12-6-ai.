# NEXT100-063 — terminal source registry convergence V3

## Current canonical authority

The current working authority is `configs/data/next100_063_terminal_source_registry_v3.json`, registry identity `1f7eebfd735ad7219a33ad82edd15192dda88281994e1488d40a13b57016df29`.

V3 converges late terminal source-admission decisions on top of the exact-green NEXT100-065 cross-source dedup V3 authority, rather than recomputing capacity from the older DATA-287 registry alone. V1 and V2 are superseded historical steps retained in Git history / versioned evidence; they must not be treated as the current capacity authority.

Scientific parent:

- NEXT100-065 exact head `efc278cec0e4773eb4ff405bf4b4d24ee63b5d13`;
- config blob `c1e05f09490e25f6fed765dfb70d900717528f4d`;
- dedicated workflow run `32999969398` = `success`;
- certified parent capacity: 243,970 bytes across 11 objects / 7 independent families.

The corrected late authority layer produces **320,632 bytes of numeric training-source capacity across 12 source families before successor global dedup**. Its source-normalized envelope is **338,533 bytes**. The 17,901-byte difference is CPython documentation whose source authority is terminal but whose training permission is chunk-scoped: 14/16 chunks are accepted and two are rejected for `pii_phone`. V3 gives those CPython source bytes zero numeric capacity until an exact accepted-chunk byte ledger/materialization becomes terminal.

Current pre-successor-dedup strata:

- Ukrainian: 100,856 numeric bytes / 4 families.
- English: 150,643 numeric bytes / 4 source-authority families; 168,544-byte normalized envelope including zero-credit CPython source bytes.
- Code: 69,133 numeric bytes / 4 families.

The minimum two-family rule is therefore structurally satisfiable at the source-authority layer. It is not yet a post-dedup, post-decontamination corpus claim.

## Audit correction

An independent exact-head workflow audit found that two source PR descriptions labelled their candidates as admitted while their dedicated source-admission workflows were actually red:

- PR #465 Pydantic, exact head `ca1755886f052d272029d6d68b2f1b7f02187936`: dedicated run `32999061340` = `failure`.
- PR #475 Rich, exact head `78cada1d69b3f0c438012c4e6cf79143aae2f603`: dedicated run `32999511493` = `failure`.

They therefore receive **zero capacity and zero family credit** in V3. Generic or unrelated green workflows do not override a failed dedicated admission gate.

## Late authorities counted after NEXT100-065

- PR #449 — KMu Secretariat, 9,153 UA bytes, dedicated run `32997970539` success.
- PR #462 — Verba/Nomis1864, 1,659 UA bytes, dedicated run `32998503672` success.
- PR #445 — MDN prose-only snapshot, 6,492 EN bytes, dedicated run `32998544359` success.
- PR #472 — NIST technical-series bounded subset, 59,358 EN bytes, dedicated run `32998703545` success.
- PR #467 — CPython documentation source authority, 17,901 normalized EN source bytes, dedicated run `32998356906` success; numeric capacity remains zero here until the accepted 14-chunk byte ledger is terminal.

UA Wikisource, Django and Starlette are not re-added because they are already inside the exact-green NEXT100-065 parent vector. This avoids duplicate family/capacity accounting.

## Research Corpus V1 boundary

The 20,000,000-byte acquisition-planning proxy still has a **19,679,368-byte numeric-capacity gap**. More importantly, source bytes are not causal loss tokens. This authority does not imply that a 20 MB corpus is sufficient to train a 20.6M-parameter model well.

Research Corpus V1 remains blocked until the late-source vector passes a successor global exact/near/lineage dedup and then an immutable accepted-record inventory is bound. Evaluation decontamination, post-composition quality/privacy validation, cluster-safe split, deterministic shard/pack materialization, two clean builds and the unique post-pack causal-loss ledger remain mandatory.

Authorized balanced no-replay loss positions remain exactly zero. Long training and paid compute remain prohibited.

## Next executable package

1. Consume V3 in the successor global cross-source dedup lane; do not duplicate the NEXT100-065 parent objects.
2. Materialize exact accepted records/chunks for every late source, including the CPython 14/16 training-eligible subset.
3. Freeze the post-dedup candidate identity.
4. Execute evaluation decontamination without making final-test payloads available to training.
5. Re-run quality/privacy and balance/diversity on that exact identity.
6. Build cluster-safe train/validation splits and deterministic shards/packs twice from clean roots.
7. Compute the exact unique no-replay causal-loss ledger.
8. Only after those gates, bind tokenizer/model/run identities and authorize bounded learned ~20M experiments.

No model training, tokenizer fitting, optimizer update, paid compute, corpus freeze, or representativeness claim is introduced by NEXT100-063 V3.
