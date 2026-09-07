# D03 eCFR point-in-time materialization V1

This package is the execution-ready successor to merged PR #707. It does not create a
second corpus pipeline and does not grant eCFR data training authority.

## Exact source boundary

The parent authority is merged PR #707, whose eCFR discovery contract is bound by exact
Git blob and SHA-256 identity. V1 selects one bounded pilot object only:

- eCFR historical date: `2026-08-06`;
- Title: `5`;
- endpoint: `https://www.ecfr.gov/api/versioner/v1/full/2026-08-06/title-5.xml`;
- family: `us.federal-regulations.ecfr`;
- execution class: `LOCAL_FREE`.

Title 35 remains reserved and is rejected. Requests later than the frozen metadata date
are rejected. Titles are not counted as independent families.

## Acquisition contract

The materializer performs two independent GETs of the same exact historical object. It
requires HTTPS, an approved eCFR host, unchanged exact path/query after redirects,
identity content encoding, XML content type, bounded object size, and byte-identical
SHA-256/size/final-URL/content-type results across both acquisitions.

Only the first verified copy is retained in the local workspace. The durable report
contains hashes, byte counts, identities and gate state only; it never contains raw eCFR
text.

## XML safety and deterministic text identity

Before parsing, the complete payload is scanned for `DOCTYPE` and entity declarations;
either causes a fail-closed result. XML must be well formed. Text identity covers all
character data and excludes attributes. Unicode whitespace is deterministically collapsed
to one ASCII space and leading/trailing whitespace is removed. The report records only
normalized-text SHA-256, UTF-8 byte count and character count.

This is an extraction identity, not a final training record format. Rights/provenance,
quality/privacy and downstream corpus policy can still exclude or transform material.

## Rights and claim boundary

The parent #707 rights boundary is preserved. Federal hosting or public availability is
not blanket model-training rights authority. Before any training eligibility a successor
must exclude or separately clear incorporated-by-reference content, contractor/private
authorship, transferred copyright, third-party media/tables, and provenance-ambiguous
payloads.

Even a successful execution therefore keeps all of the following at zero/false:
training-authorized bytes, family credit, unique loss positions, tokenizer-fit authority,
optimizer updates, model training, final-test access, paid compute and Research Corpus V1
release.

The next gate after successful acquisition is `RIGHTS_AND_PROVENANCE_CLASSIFICATION`,
followed by quality/language/privacy, global dedup, evaluation decontamination,
balance/family caps, cluster-safe split, deterministic packing/two-clean-build proof and
the post-pack unique causal-loss ledger.

No workflow or `ci.yml` change is part of this package. Network execution is intentionally
separate from code qualification while the active Rada secondary probe owns the shared
`ci.yml` execution surface.
